#****************************************************************************
# Copyright 2019-2026 Matthew Ballance and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#****************************************************************************
"""
FSM IR: Intermediate representation for synthesized FSMs.

This module defines the data structures for representing FSMs after
SPRTL transformation. The FSM IR is a lower-level representation that
can be directly converted to SystemVerilog or other HDL.
"""

import dataclasses as dc
from typing import List, Optional, Dict, Any, Union
from enum import Enum, auto


class FSMStateKind(Enum):
    """Kind of FSM state."""
    NORMAL = auto()      # Regular state with operations
    WAIT_COND = auto()   # State waiting for a condition
    WAIT_CYCLES = auto() # State waiting for N cycles


@dc.dataclass
class FSMOperation:
    """Base class for operations within an FSM state."""
    pass


@dc.dataclass
class ExprStructField:
    """Struct field reference: instance_name.field_name (e.g. dec_out.funct3)."""
    instance_name: str
    field_name: str


@dc.dataclass
class ExprStructRef:
    """Whole-struct reference: instance_name (e.g. dec_out as an lvalue or rvalue)."""
    instance_name: str


@dc.dataclass
class FSMStructDef:
    """Packed struct type definition: typedef struct packed { ... } name;"""
    name: str                          # e.g. "DecodeResult_t"
    fields: List[tuple]                # list of (field_name: str, width: int)


@dc.dataclass
class FSMStructInstance:
    """Struct instance declaration: struct_type instance_name;"""
    instance_name: str                 # e.g. "dec_out"
    struct_type: str                   # e.g. "DecodeResult_t"


@dc.dataclass
class FSMAssign(FSMOperation):
    """Assignment operation: target = value."""
    target: Any           # Target: str for plain signals, ExprStructField/ExprStructRef for structs
    value: Any            # Expression for the value (can be IR Expr)
    is_nonblocking: bool = True  # True for <= (register), False for = (wire)


@dc.dataclass
class FSMCond(FSMOperation):
    """Conditional operation block: if (cond) { then_ops } else { else_ops }."""
    condition: Any        # Condition expression (IR Expr)
    then_ops: List[FSMOperation] = dc.field(default_factory=list)
    else_ops: List[FSMOperation] = dc.field(default_factory=list)


@dc.dataclass
class FSMTransition:
    """Represents a transition from one FSM state to another.
    
    Attributes:
        target_state: ID of the target state
        condition: Optional condition for the transition (None = unconditional)
        priority: For multiple transitions, lower number = higher priority
    """
    target_state: int
    condition: Optional[Any] = None  # None means unconditional (always taken)
    priority: int = 0
    
    @property
    def is_unconditional(self) -> bool:
        return self.condition is None


@dc.dataclass
class FSMState:
    """Represents a single state in the FSM.
    
    Attributes:
        id: Unique state identifier
        name: Human-readable state name (e.g., "IDLE", "LOAD", "COMPUTE")
        kind: Type of state (normal, wait_cond, wait_cycles)
        operations: List of operations executed in this state
        transitions: List of possible transitions from this state
        source_info: Optional back-reference to source code location
    """
    id: int
    name: str
    kind: FSMStateKind = FSMStateKind.NORMAL
    operations: List[FSMOperation] = dc.field(default_factory=list)
    transitions: List[FSMTransition] = dc.field(default_factory=list)
    source_info: Optional[Dict[str, Any]] = None
    
    # For WAIT_COND states
    wait_condition: Optional[Any] = None
    
    # For WAIT_CYCLES states
    wait_cycles: int = 1
    
    def add_operation(self, op: FSMOperation):
        """Add an operation to this state."""
        self.operations.append(op)
    
    def add_transition(self, target: int, condition: Any = None, priority: int = 0):
        """Add a transition from this state."""
        self.transitions.append(FSMTransition(
            target_state=target,
            condition=condition,
            priority=priority
        ))
    
    def get_default_transition(self) -> Optional[FSMTransition]:
        """Get the unconditional (default) transition if any."""
        for t in self.transitions:
            if t.is_unconditional:
                return t
        return None


@dc.dataclass
class FSMRegRead(FSMOperation):
    """Register read operation: result_var = reg.read()."""
    reg_name: str         # Human-readable register path (e.g. "ctrl", "ch_src")
    result_var: str = ""  # Local variable name that receives the read value


@dc.dataclass
class FSMRegWrite(FSMOperation):
    """Register write operation: reg.write(value)."""
    reg_name: str   # Human-readable register path
    value: Any = None  # Value expression to write


@dc.dataclass
class FSMMemRequest(FSMOperation):
    """Memory-interface request drive: port.request(req).
    Kept for backward compatibility; new code uses FSMPortCall."""
    port_name: str   # Port name (e.g. "mem")
    req_var: str = ""  # Local variable holding the MemReq struct


@dc.dataclass
class FSMMemResponse(FSMOperation):
    """Memory-interface response latch: result_var = port.response().
    Kept for backward compatibility; new code uses FSMPortCall."""
    port_name: str    # Port name (e.g. "mem")
    result_var: str = ""  # Local variable that receives the MemRsp struct


@dc.dataclass
class FSMPortCall(FSMOperation):
    """Awaited protocol-port method call: result = await self.PORT.METHOD(args).

    Lowers to a WAIT_COND state that asserts PORT_METHOD_valid, drives
    PORT_METHOD_argN outputs, waits for PORT_METHOD_ack, then latches
    PORT_METHOD_rdata into result_var.
    """
    port_name: str        # e.g. "mem"
    method_name: str      # e.g. "read_word"
    arg_exprs: list       # IR expression per argument
    result_var: str = ""  # local variable that receives the return value (empty = void)


@dc.dataclass
class FSMPortOutput(FSMOperation):
    """Non-awaited protocol-port method call: self.PORT.METHOD(args).

    Combinatorial output assertion for one FSM state — no ack, no result.
    Lowers to PORT_METHOD_valid=1 and PORT_METHOD_argN outputs driven in that state.
    """
    port_name: str    # e.g. "monitor"
    method_name: str  # e.g. "on_retire"
    arg_exprs: list   # IR expression per argument


@dc.dataclass
class FSMAddrDecode:
    """Combinational address-decode block for a memory-mapped register file.

    Emitted as ``always @(*)`` in Verilog-2005.  One instance is attached to
    an ``FSMModule`` that exposes a ``cfg`` MemIF export bound to a RegFile.
    """
    port_name: str       # e.g. "cfg"
    n_chan: int          # Number of channels
    regs_per_chan: int   # Bytes per channel (e.g. 0x10)
    reg_names: List[str]  # e.g. ["src", "dst", "length", "ctrl"]
    reg_widths: List[int]  # e.g. [32, 32, 32, 3]


@dc.dataclass
class FSMPort:
    """Port declaration for an FSM module."""
    name: str
    direction: str  # 'input' or 'output'
    width: int = 1
    reset_value: Optional[Any] = None  # Reset value for outputs


@dc.dataclass
class FSMRegister:
    """Internal register declaration for an FSM module."""
    name: str
    width: int = 1
    reset_value: Optional[Any] = None


@dc.dataclass
class DomainBinding:
    """Resolved clock/reset binding used by synthesis emitters.

    Created by :class:`ClockDomainAnalysisPass` (Phase 4) from a component's
    ``clock_domain`` / ``reset_domain`` IR attributes.  All names here are the
    concrete signal names that will appear in the generated SV.

    Attributes:
        clock_name:     Clock signal name (e.g. ``"clk"``).
        reset_name:     Reset signal name (e.g. ``"rst_n"``).
        reset_active_low: ``True`` → reset asserts low (``rst_n`` style).
        reset_async:    ``True`` → asynchronous reset sensitivity list.
        period:         Clock period if known (None → not generated in SDC).
    """
    clock_name:      str  = "clk"
    reset_name:      str  = "rst_n"
    reset_active_low: bool = True
    reset_async:     bool  = False
    period:          Optional[Any] = None   # Time | None

    @classmethod
    def from_component_ir(cls, component_ir) -> "DomainBinding":
        """Build a DomainBinding from a DataTypeComponent's domain slots.

        Falls back to sensible defaults when no domain is declared.

        :param component_ir: A ``DataTypeComponent`` IR node.
        """
        # Lazy import to avoid circular dependency with zuspec-dataclasses
        try:
            from zuspec.dataclasses.domain import (
                ClockDomain, ResetDomain,
            )
        except ImportError:
            return cls()

        clock_domain = getattr(component_ir, "clock_domain", None)
        reset_domain = getattr(component_ir, "reset_domain", None)

        # Determine clock name
        if isinstance(clock_domain, ClockDomain):
            clock_name = clock_domain.name or "clk"
            period = clock_domain.period
        else:
            clock_name = "clk"
            period = None

        # Determine reset name, polarity, style
        if isinstance(reset_domain, ResetDomain):
            active_low = (reset_domain.polarity == "active_low")
            reset_name = "rst_n" if active_low else "rst"
            reset_async = (reset_domain.style == "async")
        else:
            active_low = True
            reset_name = "rst_n"
            reset_async = False

        return cls(
            clock_name=clock_name,
            reset_name=reset_name,
            reset_active_low=active_low,
            reset_async=reset_async,
            period=period,
        )


@dc.dataclass
class FSMModule:
    """Complete FSM module representation.
    
    This is the top-level container for a synthesized FSM, containing
    all the information needed to generate HDL code.
    
    Attributes:
        name: Module name
        ports: List of port declarations
        registers: List of internal register declarations
        states: List of FSM states
        initial_state: ID of the initial state (after reset)
        clock_signal: Name of the clock signal
        reset_signal: Name of the reset signal
        reset_active_low: True if reset is active-low (rst_n)
        reset_async: True if reset is asynchronous (appears in posedge/negedge sensitivity list)
    """
    name: str
    ports: List[FSMPort] = dc.field(default_factory=list)
    registers: List[FSMRegister] = dc.field(default_factory=list)
    states: List[FSMState] = dc.field(default_factory=list)
    initial_state: int = 0
    clock_signal: str = "clk"
    reset_signal: str = "rst_n"
    reset_active_low: bool = True
    reset_async: bool = False
    # New-style domain binding (takes precedence over clock_signal/reset_signal when set)
    domain_binding: Optional[DomainBinding] = dc.field(default=None)
    # User-defined @zdc.enum types referenced by this module.
    # Each entry is a dict with keys: name, width, items (name→value OrderedDict).
    user_enums: List[Dict[str, Any]] = dc.field(default_factory=list)

    # Packed struct type definitions derived from Buffer[T] payload classes.
    user_structs: List['FSMStructDef'] = dc.field(default_factory=list)

    # Struct register instances (one per Buffer field occurrence in activity loop).
    struct_instances: List['FSMStructInstance'] = dc.field(default_factory=list)
    
    # Optional address-decode block (emitted as always @(*) combinational logic)
    addr_decode: Optional['FSMAddrDecode'] = None

    # Array fields: name → depth  (e.g. {'gpr': 32})
    # Emitted as  logic [31:0] name [0:depth-1];
    array_fields: Dict[str, int] = dc.field(default_factory=dict)

    # Field name map: ExprRefField.index → field name
    field_names: Dict[int, str] = dc.field(default_factory=dict)

    # State encoding
    state_width: int = 0  # Computed from number of states
    state_encoding: Dict[int, int] = dc.field(default_factory=dict)  # state_id -> encoded value
    
    def __post_init__(self):
        """Compute state width after initialization."""
        if self.states and self.state_width == 0:
            self._compute_state_encoding()
    
    def _compute_state_encoding(self):
        """Compute state encoding (one-hot or binary)."""
        n_states = len(self.states)
        if n_states == 0:
            return
        
        # Use binary encoding
        self.state_width = (n_states - 1).bit_length() or 1
        
        for i, state in enumerate(self.states):
            self.state_encoding[state.id] = i
    
    def add_port(self, name: str, direction: str, width: int = 1, 
                 reset_value: Any = None) -> FSMPort:
        """Add a port to the module."""
        port = FSMPort(name=name, direction=direction, width=width,
                       reset_value=reset_value)
        self.ports.append(port)
        return port
    
    def add_register(self, name: str, width: int = 1, 
                     reset_value: Any = None) -> FSMRegister:
        """Add an internal register to the module."""
        reg = FSMRegister(name=name, width=width, reset_value=reset_value)
        self.registers.append(reg)
        return reg
    
    def add_state(self, name: str, kind: FSMStateKind = FSMStateKind.NORMAL,
                  **kwargs) -> FSMState:
        """Add a state to the FSM."""
        state_id = len(self.states)
        state = FSMState(id=state_id, name=name, kind=kind, **kwargs)
        self.states.append(state)
        self._compute_state_encoding()
        return state
    
    def get_state(self, state_id: int) -> Optional[FSMState]:
        """Get a state by ID."""
        for state in self.states:
            if state.id == state_id:
                return state
        return None
    
    def get_state_by_name(self, name: str) -> Optional[FSMState]:
        """Get a state by name."""
        for state in self.states:
            if state.name == name:
                return state
        return None
    
    def get_input_ports(self) -> List[FSMPort]:
        """Get all input ports."""
        return [p for p in self.ports if p.direction == 'input']
    
    def get_output_ports(self) -> List[FSMPort]:
        """Get all output ports."""
        return [p for p in self.ports if p.direction == 'output']

    def add_user_enum(self, name: str, width: int, items: Dict[str, int]) -> None:
        """Register a user-defined enum type for ``typedef enum`` emission.

        Args:
            name:  Enum type name (e.g. ``"State"``).
            width: Bit width of the enum.
            items: Ordered dict mapping member name → integer value.
        """
        # Avoid duplicates by name.
        if any(e['name'] == name for e in self.user_enums):
            return
        self.user_enums.append({'name': name, 'width': width, 'items': dict(items)})
