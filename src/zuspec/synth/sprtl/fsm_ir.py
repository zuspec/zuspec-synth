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
class FSMAssign(FSMOperation):
    """Assignment operation: target = value."""
    target: str           # Target signal/register name
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
    """Memory-interface request drive: port.request(req)."""
    port_name: str   # Port name (e.g. "mem")
    req_var: str = ""  # Local variable holding the MemReq struct


@dc.dataclass
class FSMMemResponse(FSMOperation):
    """Memory-interface response latch: result_var = port.response()."""
    port_name: str    # Port name (e.g. "mem")
    result_var: str = ""  # Local variable that receives the MemRsp struct


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
    """
    name: str
    ports: List[FSMPort] = dc.field(default_factory=list)
    registers: List[FSMRegister] = dc.field(default_factory=list)
    states: List[FSMState] = dc.field(default_factory=list)
    initial_state: int = 0
    clock_signal: str = "clk"
    reset_signal: str = "rst_n"
    reset_active_low: bool = True
    
    # Optional address-decode block (emitted as always @(*) combinational logic)
    addr_decode: Optional['FSMAddrDecode'] = None

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
