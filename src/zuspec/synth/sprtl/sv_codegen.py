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
SystemVerilog Code Generator: Generates synthesizable SV from FSM IR.

This module converts FSM IR to synthesizable SystemVerilog code following
best practices for FPGA and ASIC synthesis:
- Explicit state machine coding style
- Synchronous reset (or async reset with proper coding)
- Separate combinational and sequential logic
- Clean port declarations
"""

from typing import List, Dict, Set, Optional, Any, TextIO
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO

from .fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMTransition,
    FSMOperation, FSMAssign, FSMCond, FSMPort, FSMRegister, FSMRegWrite,
    FSMPortCall, FSMPortOutput,
    ExprStructField, ExprStructRef, FSMStructDef, FSMStructInstance,
)
from .fsm_generator import StateEncoding


class _SyntheticAndCond:
    """Synthetic AND condition with leading terms stripped.

    Used when decomposing ``ExprBool(And, [C1, C2, C3])`` — after extracting C1
    as the primary case discriminator, the residual ``[C2, C3, ...]`` is stored
    here so that :meth:`SVGenerator._extract_case_key` and
    :meth:`SVGenerator._generate_operations` can recurse without constructing
    new IR nodes.
    """
    __slots__ = ('values',)

    def __init__(self, values: list):
        self.values = list(values)


class ResetStyle(Enum):
    """Reset coding styles."""
    ASYNC_LOW = auto()    # Asynchronous active-low reset (rst_n)
    ASYNC_HIGH = auto()   # Asynchronous active-high reset (rst)
    SYNC_LOW = auto()     # Synchronous active-low reset
    SYNC_HIGH = auto()    # Synchronous active-high reset


class FSMStyle(Enum):
    """FSM coding styles."""
    ONE_PROCESS = auto()      # Single always_ff block
    TWO_PROCESS = auto()      # Separate comb and seq blocks
    THREE_PROCESS = auto()    # State reg, next state logic, output logic


@dataclass
class SVGenConfig:
    """Configuration for SystemVerilog generation.
    
    Attributes:
        reset_style: Reset coding style
        fsm_style: FSM coding style
        use_logic: Use 'logic' instead of 'reg/wire'
        use_always_ff: Use always_ff/always_comb (SV) vs always (Verilog)
        indent: Indentation string
        generate_comments: Include comments in output
        state_type_name: Name for state enum type
    """
    reset_style: ResetStyle = ResetStyle.ASYNC_LOW
    fsm_style: FSMStyle = FSMStyle.TWO_PROCESS
    use_logic: bool = True
    use_always_ff: bool = True
    indent: str = "  "
    generate_comments: bool = True
    state_type_name: str = "state_t"


def _walk_ops(ops):
    """Yield every FSMOperation in *ops*, recursing into FSMCond branches."""
    for op in ops:
        yield op
        if isinstance(op, FSMCond):
            yield from _walk_ops(op.then_ops)
            yield from _walk_ops(op.else_ops)


def _walk_exprs(expr: Any):
    """Yield every expression node reachable from *expr* (depth-first)."""
    if expr is None:
        return
    yield expr
    # Recurse into known compound expression types.
    for attr in ('value', 'lhs', 'rhs', 'left', 'right', 'slice', 'base',
                 'func', 'operand'):
        child = getattr(expr, attr, None)
        if child is not None and not isinstance(child, (int, float, str, bool)):
            yield from _walk_exprs(child)
    for child_list in (getattr(expr, 'args', None),
                       getattr(expr, 'values', None),
                       getattr(expr, 'comparators', None)):
        if child_list:
            for child in child_list:
                if not isinstance(child, (int, float, str, bool)):
                    yield from _walk_exprs(child)


def _collect_struct_usage(fsm) -> tuple:
    """Analyse struct instance usage across all FSM states.

    Returns:
        scratch_structs  : set of instance names written ONLY via ExprStructField
                           (blocking field-by-field).  No reset needed since
                           these are combinationally recomputed each cycle and
                           a non-blocking ``<= '0`` reset would cause a mixed
                           blocking/non-blocking assignment on the same variable.
    """
    field_targets: set = set()   # instance names written via ExprStructField
    ref_targets:   set = set()   # instance names written via ExprStructRef

    for state in fsm.states:
        for op in _walk_ops(state.operations):
            if isinstance(op, FSMAssign):
                if isinstance(op.target, ExprStructField):
                    field_targets.add(op.target.instance_name)
                elif isinstance(op.target, ExprStructRef):
                    ref_targets.add(op.target.instance_name)

    # Scratch: only field-written, never whole-struct copied TO it.
    scratch_structs = field_targets - ref_targets

    return scratch_structs, set()   # dead_ref_structs always empty for now


class SVCodeGenerator:
    """Generates SystemVerilog code from FSM IR.
    
    Usage:
        generator = SVCodeGenerator()
        sv_code = generator.generate(fsm_module)
    """
    
    def __init__(self, config: Optional[SVGenConfig] = None):
        """Initialize the code generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config or SVGenConfig()
        self._indent_level = 0
        self._output: StringIO = StringIO()
        # Within-state expression forwarding: maps register name → formatted RHS
        # string for assignments made earlier in the current state.  Cleared
        # before each state's output logic is generated so that substitution
        # never crosses state boundaries.
        self._state_pending: Dict[str, str] = {}
        # Cross-state struct alias map: maps a register that holds a whole-struct
        # copy (e.g. _action_result_mem_dec) to the source register
        # (e.g. _action_result_dec_out).  Sub-field accesses on the alias target
        # are transparently redirected to the source so undeclared aggregate
        # signals are never emitted.
        self._struct_aliases: Dict[str, str] = {}
        # Depth counter for nested conditionals.  When > 0 we are inside a
        # branch of an FSMCond; assignments use blocking "=" so that subsequent
        # reads within the same always_ff block see the updated value.
        self._cond_depth: int = 0

    def generate(self, fsm: FSMModule) -> str:
        """Generate SystemVerilog code for an FSM module.
        
        Args:
            fsm: FSM module to generate code for
            
        Returns:
            SystemVerilog source code as string
        """
        self._output = StringIO()
        self._indent_level = 0

        # Stash field names and array fields for ExprRefField resolution
        self._field_names: Dict[int, str] = getattr(fsm, 'field_names', {})
        self._array_fields: Dict[str, int] = getattr(fsm, 'array_fields', {})

        # Build port-width lookup for narrow-port truncation in _generate_assign.
        self._port_widths: Dict[str, int] = {p.name: p.width for p in fsm.ports}

        # Build a deduplicated state-name map (state.id → SV enum literal).
        # If two states share a name, append the encoded value to disambiguate.
        self._state_sv_names: Dict[int, str] = {}
        _seen: Dict[str, int] = {}
        for i, st in enumerate(fsm.states):
            enc = fsm.state_encoding.get(st.id, i)
            sname = st.name
            if sname in _seen and _seen[sname] != enc:
                sname = f"{sname}_{enc}"
            _seen.setdefault(st.name, enc)
            self._state_sv_names[st.id] = sname
        # Auto-infer registers: collect all FSMAssign targets not yet in fsm.registers.
        # Skip array-indexed targets (e.g. "gpr[rd]") — those are covered by array_fields.
        # Skip ports (declared in module header), array fields, and clock/reset signals.
        known_reg_names = {r.name for r in fsm.registers}
        known_port_names = {p.name for p in fsm.ports}
        _clk = fsm.clock_signal or 'clk'
        _rst = fsm.reset_signal or 'rst_n'
        _skip = known_reg_names | known_port_names | {_clk, _rst} | set(fsm.array_fields.keys())
        for state in fsm.states:
            for op in _walk_ops(state.operations):
                if (isinstance(op, FSMAssign)
                        and isinstance(op.target, str)
                        and '[' not in op.target
                        and op.target not in _skip):
                    fsm.registers.append(FSMRegister(name=op.target, width=32))
                    known_reg_names.add(op.target)
                    _skip.add(op.target)

        # Auto-infer port signals from FSMPortCall / FSMPortOutput operations
        known_port_names = {p.name for p in fsm.ports}
        for state in fsm.states:
            for op in _walk_ops(state.operations):
                if isinstance(op, (FSMPortCall, FSMPortOutput)):
                    prefix = f"{op.port_name}_{op.method_name}"
                    # valid output (always present)
                    _add_port(fsm, known_port_names, f"{prefix}_valid", 'output', 1)
                    # arg outputs
                    for i in range(len(op.arg_exprs)):
                        _add_port(fsm, known_port_names, f"{prefix}_arg{i}", 'output', 32)
                    if isinstance(op, FSMPortCall):
                        # ack input + rdata input
                        _add_port(fsm, known_port_names, f"{prefix}_ack", 'input', 1)
                        if op.result_var:
                            _add_port(fsm, known_port_names, f"{prefix}_rdata", 'input', 32)
                            # result register (latched on ack)
                            if op.result_var not in known_reg_names:
                                fsm.registers.append(
                                    FSMRegister(name=op.result_var, width=32))
                                known_reg_names.add(op.result_var)

        self._generate_header(fsm)
        self._generate_user_enum_typedefs(fsm)
        self._generate_user_struct_typedefs(fsm)
        self._generate_module_declaration(fsm)
        self._generate_state_encoding(fsm)
        # Classify struct instances: scratch (no reset) vs dead refs (omit entirely).
        self._scratch_structs, self._dead_struct_refs = _collect_struct_usage(fsm)
        self._generate_register_declarations(fsm)
        self._generate_state_register(fsm)
        # Precompute per-state forwarding maps so both next-state comb logic
        # and output ff logic can use them consistently.
        self._precompute_state_forwarding(fsm)
        self._struct_aliases: Dict[str, str] = {}  # retired — struct IR handles aliasing
        self._generate_next_state_logic(fsm)
        self._generate_output_logic(fsm)
        self._generate_module_end()
        
        return self._output.getvalue()
    
    def _emit(self, text: str = ""):
        """Emit text to output."""
        self._output.write(text)
    
    def _emitln(self, text: str = ""):
        """Emit text with newline."""
        self._emit(self._get_indent() + text + "\n")
    
    def _get_indent(self) -> str:
        """Get current indentation string."""
        return self.config.indent * self._indent_level
    
    def _indent(self):
        """Increase indentation."""
        self._indent_level += 1
    
    def _dedent(self):
        """Decrease indentation."""
        self._indent_level = max(0, self._indent_level - 1)

    def _eff_clk(self, fsm: FSMModule) -> str:
        """Effective clock signal name — prefers DomainBinding when set."""
        if fsm.domain_binding is not None:
            return fsm.domain_binding.clock_name
        return fsm.clock_signal

    def _eff_rst(self, fsm: FSMModule) -> str:
        """Effective reset signal name — prefers DomainBinding when set."""
        if fsm.domain_binding is not None:
            return fsm.domain_binding.reset_name
        return fsm.reset_signal

    def _eff_reset_active_low(self, fsm: FSMModule) -> bool:
        """Effective reset polarity — prefers DomainBinding when set."""
        if fsm.domain_binding is not None:
            return fsm.domain_binding.reset_active_low
        return fsm.reset_active_low

    def _eff_reset_async(self, fsm: FSMModule) -> bool:
        """Effective reset style — prefers DomainBinding when set."""
        if fsm.domain_binding is not None:
            return fsm.domain_binding.reset_async
        return fsm.reset_async

    def _generate_header(self, fsm: FSMModule):
        """Generate file header comment."""
        if self.config.generate_comments:
            self._emitln("//")
            self._emitln(f"// Module: {fsm.name}")
            self._emitln("// Generated by zuspec-synth SPRTL compiler")
            self._emitln("//")
            self._emitln()

    def _generate_user_enum_typedefs(self, fsm: FSMModule):
        """Emit ``typedef enum logic`` declarations for user-defined enums.

        One typedef is emitted per entry in ``fsm.user_enums``.  These are
        emitted *before* the module declaration so they can be used as port
        or register types within the module.
        """
        if not fsm.user_enums:
            return
        if self.config.generate_comments:
            self._emitln("// User-defined enum types")
        for enum_info in fsm.user_enums:
            name = enum_info['name']
            width = enum_info['width']
            items = enum_info['items']
            self._emitln(f"typedef enum logic [{width - 1}:0] {{")
            self._indent()
            item_list = list(items.items())
            for i, (member, value) in enumerate(item_list):
                comma = "," if i < len(item_list) - 1 else ""
                self._emitln(f"{member} = {width}'d{value}{comma}")
            self._dedent()
            self._emitln(f"}} {name}_t;")
            self._emitln()

    def _generate_user_struct_typedefs(self, fsm: FSMModule):
        """Emit ``typedef struct packed`` declarations for Buffer[T] payload types.

        One typedef is emitted per entry in ``fsm.user_structs``.  These are
        emitted *before* the module declaration so they can be used as register
        types within the module.
        """
        if not fsm.user_structs:
            return
        if self.config.generate_comments:
            self._emitln("// Buffer payload struct types")
        for sd in fsm.user_structs:
            self._emitln(f"typedef struct packed {{")
            self._indent()
            for field_name, width in sd.fields:
                width_str = self._format_width(width)
                self._emitln(f"logic {width_str}{field_name};")
            self._dedent()
            self._emitln(f"}} {sd.name};")
            self._emitln()

    def _generate_module_declaration(self, fsm: FSMModule):
        """Generate module declaration with ports."""
        self._emitln("/* verilator lint_off DECLFILENAME */")
        self._emitln(f"module {fsm.name} (")
        self._indent()

        clk = self._eff_clk(fsm)
        rst = self._eff_rst(fsm)

        # Clock and reset first
        self._emitln(f"input  logic {clk},")
        self._emitln(f"input  logic {rst},")

        # Other ports
        ports = [p for p in fsm.ports
                if p.name not in (clk, rst)]

        for i, port in enumerate(ports):
            direction = "input " if port.direction == "input" else "output"
            width_str = self._format_width(port.width)
            comma = "," if i < len(ports) - 1 else ""
            self._emitln(f"{direction} logic {width_str}{port.name}{comma}")

        self._dedent()
        self._emitln(");")
        self._emitln()
    
    def _format_width(self, width: int) -> str:
        """Format port/signal width declaration."""
        if width == 1:
            return ""
        return f"[{width-1}:0] "
    
    def _generate_state_encoding(self, fsm: FSMModule):
        """Generate state type and encoding."""
        if not fsm.states:
            return
        
        if self.config.generate_comments:
            self._emitln("// State encoding")
        
        # Generate enum type
        width = fsm.state_width or 1
        self._emitln(f"typedef enum logic [{width-1}:0] {{")
        self._indent()
        
        for i, state in enumerate(fsm.states):
            encoded = fsm.state_encoding.get(state.id, i)
            comma = "," if i < len(fsm.states) - 1 else ""
            sv_name = self._state_sv_names[state.id]
            self._emitln(f"{sv_name} = {width}'d{encoded}{comma}")
        
        self._dedent()
        self._emitln(f"}} {self.config.state_type_name};")
        self._emitln()
        
        # State registers
        self._emitln(f"{self.config.state_type_name} state, next_state;")
        self._emitln()
    
    def _generate_register_declarations(self, fsm: FSMModule):
        """Generate internal register declarations."""
        # Filter out state register and port signals (already declared in module header)
        port_names = {p.name for p in fsm.ports}
        regs = [r for r in fsm.registers if r.name != "state" and r.name not in port_names]

        has_content = bool(regs or self._array_fields or fsm.struct_instances)
        if not has_content:
            return

        if self.config.generate_comments:
            self._emitln("// Internal registers")

        self._emitln("/* verilator lint_off UNUSEDSIGNAL */")

        # Struct instances (typed registers from Buffer[T] fields).
        # Dead ref structs (written but never read) are omitted entirely.
        for si in fsm.struct_instances:
            if si.instance_name not in self._dead_struct_refs:
                self._emitln(f"{si.struct_type} {si.instance_name};")

        for reg in regs:
            width_str = self._format_width(reg.width)
            self._emitln(f"logic {width_str}{reg.name};")

        # Declare array fields (e.g. logic [31:0] gpr [0:31];)
        for arr_name, depth in self._array_fields.items():
            self._emitln(f"logic [31:0] {arr_name} [0:{depth-1}];")
        self._emitln("/* verilator lint_on UNUSEDSIGNAL */")

        self._emitln()
    
    def _generate_state_register(self, fsm: FSMModule):
        """Generate state register with reset."""
        if not fsm.states:
            return
        
        if self.config.generate_comments:
            self._emitln("// State register")
        
        initial_state = fsm.get_state(fsm.initial_state)
        initial_name = self._state_sv_names.get(initial_state.id, "IDLE") if initial_state else "IDLE"
        
        if self.config.reset_style == ResetStyle.ASYNC_LOW:
            self._emitln(f"always_ff @(posedge {self._eff_clk(fsm)} or negedge {self._eff_rst(fsm)}) begin")
            self._indent()
            self._emitln(f"if (!{self._eff_rst(fsm)})")
            self._indent()
            self._emitln(f"state <= {initial_name};")
            self._dedent()
            self._emitln("else")
            self._indent()
            self._emitln("state <= next_state;")
            self._dedent()
            self._dedent()
            self._emitln("end")
        elif self.config.reset_style == ResetStyle.SYNC_LOW:
            self._emitln(f"always_ff @(posedge {self._eff_clk(fsm)}) begin")
            self._indent()
            self._emitln(f"if (!{self._eff_rst(fsm)})")
            self._indent()
            self._emitln(f"state <= {initial_name};")
            self._dedent()
            self._emitln("else")
            self._indent()
            self._emitln("state <= next_state;")
            self._dedent()
            self._dedent()
            self._emitln("end")
        
        self._emitln()
    
    def _precompute_state_forwarding(self, fsm: FSMModule):
        """Precompute the within-state forwarding map for every state.

        For each state we perform a dry run of its *unconditional* top-level
        FSMAssign operations in order, building up ``_state_pending`` exactly
        as the live code generation would.  The resulting dict is stored in
        ``self._state_forward_maps[state.id]`` so that both next-state comb
        logic and output ff logic can use the same forwarded expressions.
        """
        self._state_forward_maps: Dict[int, Dict[str, str]] = {}
        for state in fsm.states:
            self._state_pending = {}
            for op in state.operations:
                if isinstance(op, FSMAssign) and isinstance(op.target, str) and '[' not in op.target:
                    value_str = self._format_assign_value(op.value)
                    self._state_pending[op.target] = value_str
                elif isinstance(op, FSMAssign) and not isinstance(op.target, str):
                    target_str = self._format_target(op.target)
                    value_str = self._format_assign_value(op.value)
                    self._state_pending[target_str] = value_str
            self._state_forward_maps[state.id] = dict(self._state_pending)
        self._state_pending = {}

    def _build_struct_aliases(self):
        """Build cross-state struct alias map from state forwarding maps.

        When a state assigns one ``_action_result_*`` register to another
        (a whole-struct copy), record the alias so that sub-field accesses
        through the alias target are transparently redirected to the source.
        For example, after ``_action_result_mem_dec <= _action_result_dec_out``,
        any access ``_action_result_mem_dec_funct3`` resolves to
        ``_action_result_dec_out_funct3`` via this map.
        """
        import re
        _simple_ar = re.compile(r'^_action_result_[A-Za-z0-9_]+$')
        self._struct_aliases = {}
        for fwd_map in self._state_forward_maps.values():
            for target, value_str in fwd_map.items():
                if _simple_ar.match(target) and _simple_ar.match(value_str):
                    self._struct_aliases[target] = value_str

    def _generate_next_state_logic(self, fsm: FSMModule):
        """Generate combinational next state logic."""
        if not fsm.states:
            return
        
        if self.config.generate_comments:
            self._emitln("// Next state logic")
        
        self._emitln("always_comb begin")
        self._indent()
        self._emitln("next_state = state;")
        self._emitln("case (state)")
        self._indent()
        
        for state in fsm.states:
            comment = self._source_comment(state)
            if comment and self.config.generate_comments:
                self._emitln(comment)
            self._emitln(f"{self._state_sv_names[state.id]}: begin")
            self._indent()

            # Load this state's forwarding map so conditions use forwarded
            # (combinatorial) values rather than stale registered values.
            self._state_pending = self._state_forward_maps.get(state.id, {})
            
            # Generate transitions
            if state.transitions:
                # Handle conditional and unconditional transitions
                conditional = [t for t in state.transitions if t.condition is not None]
                unconditional = [t for t in state.transitions if t.condition is None]
                
                for i, trans in enumerate(conditional):
                    target = fsm.get_state(trans.target_state)
                    target_name = self._state_sv_names.get(trans.target_state, f"S{trans.target_state}") if target else f"S{trans.target_state}"
                    cond_str = self._format_condition(trans.condition)
                    keyword = "else if" if i > 0 else "if"
                    self._emitln(f"{keyword} ({cond_str})")
                    self._indent()
                    self._emitln(f"next_state = {target_name};")
                    self._dedent()
                
                if unconditional:
                    target = fsm.get_state(unconditional[0].target_state)
                    target_name = self._state_sv_names.get(unconditional[0].target_state, f"S{unconditional[0].target_state}") if target else f"S{unconditional[0].target_state}"
                    if conditional:
                        self._emitln("else")
                        self._indent()
                        self._emitln(f"next_state = {target_name};")
                        self._dedent()
                    else:
                        self._emitln(f"next_state = {target_name};")
            
            self._dedent()
            self._emitln("end")
        
        self._emitln("default: next_state = state;")
        self._dedent()
        self._emitln("endcase")
        self._dedent()
        self._emitln("end")
        self._emitln()
        self._state_pending = {}
    
    @staticmethod
    def _source_comment(state: Any) -> str:
        """Return a comment string for a state's source_info, or empty string."""
        info = getattr(state, 'source_info', None)
        if not info:
            return ''
        stack = info.get('stack', [])
        reason = info.get('reason', '')
        cond = info.get('cond', '')
        parts = []
        if stack:
            parts.append(' \u2192 '.join(f'{s}()' for s in stack))
        if reason:
            parts.append(reason)
        if cond:
            parts.append(f'[{cond}]')
        return f"// {': '.join(parts)}" if parts else ''

    def _format_target(self, target: Any) -> str:
        """Format an FSMAssign target (str, ExprStructField, or ExprStructRef) as SV."""
        if isinstance(target, str):
            return target
        if isinstance(target, ExprStructField):
            return f"{target.instance_name}.{target.field_name}"
        if isinstance(target, ExprStructRef):
            return target.instance_name
        return str(target)

    def _ensure_bool(self, expr_node: Any) -> str:
        """Format expr_node as a 1-bit boolean SV expression.

        Comparisons (ExprCompare) and boolean operators (ExprBool) already
        produce a 1-bit result.  Logical-NOT (ExprUnary with 'Not' op) is
        also 1-bit.  Everything else — bare variable references, arithmetic —
        is wrapped with ``!= 0`` so the condition is always 1-bit.
        """
        t = type(expr_node).__name__
        if t in ('ExprCompare', 'ExprBool'):
            return self._format_expr(expr_node)
        if t == 'ExprUnary':
            op = getattr(expr_node, 'op', None)
            op_name = getattr(op, 'name', str(op)) if op else ''
            if op_name in ('Not', '!'):
                return self._format_expr(expr_node)
        sv = self._format_expr(expr_node)
        return f"({sv} != 0)"

    def _format_condition(self, condition: Any) -> str:
        """Format a condition expression as SystemVerilog."""
        if condition is None:
            return "1'b1"

        # Synthetic AND condition (residual from case decomposition)
        if isinstance(condition, _SyntheticAndCond):
            parts = [self._format_condition(v) for v in condition.values]
            return '(' + ' && '.join(parts) + ')'

        # Tuple form from transformer: ('lt', lhs, rhs)
        if isinstance(condition, tuple) and len(condition) == 3:
            op_str, lhs, rhs = condition
            op_map = {'lt': '<', 'lte': '<=', 'gt': '>', 'gte': '>=',
                      'eq': '==', 'ne': '!='}
            sv_op = op_map.get(str(op_str), str(op_str))
            return f"{self._format_expr(lhs)} {sv_op} {self._format_expr(rhs)}"

        # Handle different condition types
        if hasattr(condition, 'left') and hasattr(condition, 'op') and hasattr(condition, 'right'):
            # Binary comparison
            left = self._format_expr(condition.left)
            right = self._format_expr(condition.right)
            op = self._format_operator(condition.op)
            return f"{left} {op} {right}"

        # Simple expression — ensure 1-bit boolean context
        return self._ensure_bool(condition)

    def _format_expr(self, expr: Any) -> str:
        """Format an expression as SystemVerilog."""
        if expr is None:
            return "0"

        if isinstance(expr, (int, float)):
            return str(expr)

        if isinstance(expr, str):
            # Apply within-state forwarding: if this name was written earlier
            # in the same state, substitute its expression so that sequential
            # Python semantics are preserved despite non-blocking assignments.
            if expr in self._state_pending:
                forwarded = self._state_pending[expr]
                # Follow one transitive hop: a scalar temp may have been
                # assigned from a struct field (e.g. tmp <= dec_out.rs2)
                # before that field was itself set.  Re-check pending so the
                # final expression is the concrete source value.
                if isinstance(forwarded, str) and forwarded in self._state_pending:
                    return self._state_pending[forwarded]
                return forwarded
            return expr

        t = type(expr).__name__

        # Struct field reference: dec_out.funct3
        if isinstance(expr, ExprStructField):
            field_key = f"{expr.instance_name}.{expr.field_name}"
            if field_key in self._state_pending:
                return self._state_pending[field_key]
            # Check whole-struct forwarding: if the instance itself was copied from
            # another instance earlier in this state, redirect the field access.
            inst_key = expr.instance_name
            if inst_key in self._state_pending:
                aliased = self._state_pending[inst_key]
                aliased_field = f"{aliased}.{expr.field_name}"
                if aliased_field in self._state_pending:
                    return self._state_pending[aliased_field]
                return aliased_field
            return field_key

        # Whole-struct reference: dec_out
        if isinstance(expr, ExprStructRef):
            inst = expr.instance_name
            if inst in self._state_pending:
                return self._state_pending[inst]
            return inst

        # IR ExprConstant
        if t == 'ExprConstant':
            v = expr.value
            if isinstance(v, bool):
                return "1'b1" if v else "1'b0"
            if isinstance(v, int) and (v > 0xFFFF or v < -0x8000):
                return f"32'h{v & 0xFFFFFFFF:08X}"
            return str(v)

        # IR ExprRefLocal — a local variable
        if t == 'ExprRefLocal':
            name = expr.name
            # Apply within-state forwarding (with one transitive hop).
            if name in self._state_pending:
                forwarded = self._state_pending[name]
                if isinstance(forwarded, str) and forwarded in self._state_pending:
                    return self._state_pending[forwarded]
                return forwarded
            return name

        # IR ExprRefField(base=TypeExprRefSelf, index=N) → field name
        if t == 'ExprRefField':
            base = getattr(expr, 'base', None)
            if base is not None and type(base).__name__ == 'TypeExprRefSelf':
                idx = expr.index
                return self._field_names.get(idx, f'field_{idx}')
            # Fallback: try .name
            if hasattr(expr, 'name'):
                return expr.name

        # IR ExprRefParam — a method parameter
        if t == 'ExprRefParam':
            return expr.name

        # IR ExprBin — binary expression
        if t == 'ExprBin':
            lhs_str = self._format_expr(getattr(expr, 'lhs', None))
            rhs_str = self._format_expr(getattr(expr, 'rhs', None))
            op_str = self._format_operator(getattr(expr, 'op', None))
            # Arithmetic right shift: zdc.signed(x) >> n → $signed(x) >>> n
            if op_str == '>>' and lhs_str.startswith('$signed('):
                op_str = '>>>'
            return f"({lhs_str} {op_str} {rhs_str})"

        # IR ExprUnary — unary expression
        if t == 'ExprUnary':
            op = getattr(expr, 'op', None)
            op_str = self._format_operator(op) if op else '~'
            # Constant-fold ~N → hex literal
            inner = getattr(expr, 'operand', None)
            op_name = getattr(op, 'name', str(op)) if op else ''
            if op_name == 'Invert' and inner is not None and type(inner).__name__ == 'ExprConstant':
                v = getattr(inner, 'value', None)
                if isinstance(v, int) and not isinstance(v, bool):
                    return f"32'h{(~v) & 0xFFFFFFFF:08X}"
            operand_str = self._format_expr(inner)
            return f"({op_str}{operand_str})"

        # IR ExprBool — boolean expression (and/or)
        if t == 'ExprBool':
            op = getattr(expr, 'op', None)
            op_name = getattr(op, 'name', str(op)) if op else 'And'
            sv_op = '&&' if 'And' in str(op_name) else '||'
            values = getattr(expr, 'values', [])
            parts = [self._ensure_bool(v) for v in values]
            return f"({f' {sv_op} '.join(parts)})"

        # IR ExprCompare — comparison expression
        if t == 'ExprCompare':
            left_str = self._format_expr(getattr(expr, 'left', None))
            ops = getattr(expr, 'ops', [])
            comparators = getattr(expr, 'comparators', [])
            cmp_map = {1: '==', 2: '!=', 3: '<', 4: '<=', 5: '>', 6: '>='}
            parts = [left_str]
            for op, comp in zip(ops, comparators):
                op_val = getattr(op, 'value', op) if hasattr(op, 'value') else op
                # CmpOp.In (9) / CmpOp.NotIn (10): expand to OR/AND chain
                if op_val in (9, 10):
                    elts = getattr(comp, 'elts', None)
                    if elts is None and type(comp).__name__ == 'ExprTuple':
                        elts = getattr(comp, 'elts', [])
                    if elts is not None:
                        eq_parts = [f"({left_str} == {self._format_expr(e)})" for e in elts]
                        joiner = ' || ' if op_val == 9 else ' && '
                        return f"({joiner.join(eq_parts)})"
                op_str = cmp_map.get(op_val, '==')
                parts.append(f"{op_str} {self._format_expr(comp)}")
            return ' '.join(parts)

        # IR ExprSubscript — array or dict-style access
        # ExprSubscript(value=ExprRefField(index=N), slice=idx_expr)
        # → field_name[idx_sv]  (array indexing)
        if t == 'ExprSubscript':
            val_expr = getattr(expr, 'value', None)
            slc = getattr(expr, 'slice', None)
            # Array field: ExprRefField on self → name[idx]
            if (val_expr is not None
                    and type(val_expr).__name__ == 'ExprRefField'
                    and type(getattr(val_expr, 'base', None)).__name__ == 'TypeExprRefSelf'):
                arr_name = self._format_expr(val_expr)
                idx_sv = self._format_expr(slc)
                return f"{arr_name}[{idx_sv}]"
            # Constant-key subscript: integer → bit-select [N], string → struct field _name
            val = self._format_expr(val_expr)
            if slc is not None and type(slc).__name__ == 'ExprConstant':
                key = slc.value
                if isinstance(key, int):
                    return f"{val}[{key}]"
                return f"{val}_{key}"
            # Bit-slice: expr[msb:lsb] → val[msb:lsb]
            # SV does not allow (complex_expr)[hi:lo] — only simple identifiers can be sliced.
            # If val is a complex expression, use shift-and-mask instead.
            if slc is not None and type(slc).__name__ == 'ExprSlice':
                import re as _re
                upper_node = getattr(slc, 'upper', None)  # upper = LSB
                lower_node = getattr(slc, 'lower', None)  # lower = MSB
                # Detect runtime-variable part-select: MSB = LSB_expr + constant k
                # Emit SV indexed part-select: val[lsb_sv +: (k+1)]
                if (lower_node is not None and type(lower_node).__name__ == 'ExprBin'
                        and str(lower_node.op) in ('BinOp.Add', 'Add')
                        and type(lower_node.rhs).__name__ == 'ExprConstant'):
                    k = lower_node.rhs.value
                    width = k + 1
                    lsb_sv = self._format_expr(upper_node)
                    return f"{val}[{lsb_sv} +: {width}]"
                slice_sv = self._format_expr(slc)
                is_simple = bool(_re.match(r'^[A-Za-z_\$][A-Za-z0-9_]*(\[[^\]]+\])*$', val))
                if is_simple:
                    return f"{val}[{slice_sv}]"
                # Complex base: use SV width cast to extract the right number of bits.
                # This avoids shift-and-mask whose mask constant triggers WIDTHEXPAND
                # warnings, and produces the correct output width.
                lsb = getattr(upper_node, 'value', None)
                msb = getattr(lower_node, 'value', None)
                if isinstance(msb, int) and isinstance(lsb, int):
                    width = msb - lsb + 1
                    if lsb == 0:
                        return f"{width}'({val})"
                    return f"{width}'(({val}) >> {lsb})"
                return f"({val})[{slice_sv}]"
            return f"{val}[{self._format_expr(slc)}]"

        # IR ExprSlice — bit-select range; used as subscript in signal[upper:lower]
        if t == 'ExprSlice':
            upper = getattr(expr, 'upper', None)
            lower = getattr(expr, 'lower', None)
            upper_sv = self._format_expr(upper)
            lower_sv = self._format_expr(lower)
            # In the IR, 'lower' holds the MSB and 'upper' holds the LSB
            # (matching SV's [MSB:LSB] convention)
            return f"{lower_sv}:{upper_sv}"

        # IR ExprTuple — tuple literal; in concat context represents (value, width) padding
        if t == 'ExprTuple':
            elts = getattr(expr, 'elts', [])
            if len(elts) == 2:
                val_sv = self._format_expr(elts[0])
                width_expr = elts[1]
                width_val = getattr(width_expr, 'value', None)
                if isinstance(width_val, int) and width_val > 0:
                    # Literal zero-padding: (0, N) → N'h0
                    return f"{width_val}'h{int(getattr(elts[0], 'value', 0)):X}"
                width_sv = self._format_expr(width_expr)
                return f"{width_sv}'({val_sv})"
            return '{' + ', '.join(self._format_expr(e) for e in elts) + '}'

        # IR ExprSext / ExprZext / ExprCbit / ExprSigned — typed zdc built-in nodes
        if t == 'ExprSext':
            val_sv = self._format_expr(expr.value)
            n = expr.bits
            shift = 32 - n
            return f"$signed($signed(32'({val_sv}) << {shift}) >>> {shift})"
        if t == 'ExprZext':
            val_sv = self._format_expr(expr.value)
            return f"{val_sv}[{expr.bits - 1}:0]"
        if t == 'ExprCbit':
            inner_sv = self._format_expr(expr.value)
            if type(expr.value).__name__ == 'ExprCompare':
                return inner_sv
            return f"({inner_sv}[0])"
        if t == 'ExprSigned':
            return f"$signed({self._format_expr(expr.value)})"

        # IR ExprCall — called function (e.g., self.gpr.get(rd))
        if t == 'ExprCall':
            func = getattr(expr, 'func', None)
            args = getattr(expr, 'args', [])
            # zdc built-in lowering: recognise ExprRefUnresolved with 'zdc.*' name
            if func is not None and type(func).__name__ == 'ExprRefUnresolved':
                fname = func.name
                if fname == 'zdc.sext' and len(args) == 2:
                    val_sv = self._format_expr(args[0])
                    bits_val = getattr(args[1], 'value', None)
                    if isinstance(bits_val, int) and bits_val > 0:
                        n = bits_val
                        shift = 32 - n
                        return f"$signed($signed(32'({val_sv}) << {shift}) >>> {shift})"
                    bits_sv = self._format_expr(args[1])
                    return f"$signed($signed(32'({val_sv}) << (32-{bits_sv})) >>> (32-{bits_sv}))"
                if fname == 'zdc.zext' and len(args) == 2:
                    val_sv = self._format_expr(args[0])
                    bits_val = getattr(args[1], 'value', None)
                    if isinstance(bits_val, int) and bits_val > 0:
                        return f"{val_sv}[{bits_val-1}:0]"
                    bits_sv = self._format_expr(args[1])
                    return f"{val_sv}[{bits_sv}-1:0]"
                if fname == 'zdc.cbit' and len(args) == 1:
                    inner = args[0]
                    inner_sv = self._format_expr(inner)
                    if type(inner).__name__ == 'ExprCompare':
                        return inner_sv
                    return f"({inner_sv}[0])"
                if fname == 'zdc.signed' and len(args) == 1:
                    return f"$signed({self._format_expr(args[0])})"
                if fname == '_illegal':
                    return "/* illegal */"
                # Python type-cast builtins are no-ops in synthesizable SV
                if fname == 'int' and len(args) == 1:
                    return self._format_expr(args[0])
                if fname == 'bool' and len(args) == 1:
                    return self._format_expr(args[0])
            func_str = self._format_expr(func)
            # zdc.concat(*parts) → Verilog {a, b, c} concatenation
            if func_str == 'zdc_concat':
                return self._format_concat_args(args)
            # zdc.bv32(x) is a bit-width cast identity in SV; strip at expression level
            # (wrapping done in assignment context where width is needed)
            if func_str == 'zdc_bv32' and len(args) == 1:
                return self._format_expr(args[0])
            arg_strs = ', '.join(self._format_expr(a) for a in args)
            return f"{func_str}({arg_strs})"

        # IR ExprAttribute — attribute access
        if t == 'ExprAttribute':
            value_expr = getattr(expr, 'value', None)
            attr = getattr(expr, 'attr', 'attr')
            # Try enum constant resolution: ExprRefUnresolved('EnumName').MEMBER → integer
            if value_expr is not None and type(value_expr).__name__ == 'ExprRefUnresolved':
                resolved = self._try_resolve_enum_attr(
                    getattr(value_expr, 'name', ''), attr)
                if resolved is not None:
                    return str(resolved)
            receiver = self._format_expr(value_expr)
            flat = f"{receiver}_{attr}"
            # Apply within-state forwarding on the composed name (with transitive hop).
            if flat in self._state_pending:
                forwarded = self._state_pending[flat]
                if isinstance(forwarded, str) and forwarded in self._state_pending:
                    return self._state_pending[forwarded]
                return forwarded
            # Cross-state struct alias: if receiver is a struct alias for another
            # signal, compose the flat name with the alias source so sub-field
            # accesses (e.g. _action_result_mem_dec_funct3) are redirected to the
            # declared field (e.g. _action_result_dec_out_funct3).
            if receiver in self._struct_aliases:
                aliased = f"{self._struct_aliases[receiver]}_{attr}"
                if aliased in self._state_pending:
                    forwarded = self._state_pending[aliased]
                    if isinstance(forwarded, str) and forwarded in self._state_pending:
                        return self._state_pending[forwarded]
                    return forwarded
                return aliased
            return flat

        # ExprRefUnresolved — an unresolved name reference; try to resolve as integer constant
        if t == 'ExprRefUnresolved':
            name = expr.name
            resolved = self._try_resolve_int_const(name)
            if resolved is not None:
                if resolved > 0xFFFF or resolved < -0x8000:
                    return f"32'h{resolved & 0xFFFFFFFF:08X}"
                return str(resolved)
            if name in self._state_pending:
                return self._state_pending[name]
            return name

        # AugOp enum (e.g. AugOp.Add)
        if t == 'AugOp':
            aug_map = {1: '+', 2: '-', 3: '*', 4: '/'}
            return aug_map.get(expr.value, str(expr.value))

        # Generic fallbacks
        if hasattr(expr, 'name'):
            return expr.name
        if hasattr(expr, 'attr'):
            return expr.attr
        if hasattr(expr, 'value'):
            return str(expr.value)

        return str(expr)
    
    def _format_operator(self, op: Any) -> str:
        """Format an operator."""
        op_map = {
            'Eq': '==',
            'NotEq': '!=',
            'Lt': '<',
            'LtE': '<=',
            'Gt': '>',
            'GtE': '>=',
            'Add': '+',
            'Sub': '-',
            'Mult': '*',
            'Div': '/',
            'Mod': '%',
            'BitAnd': '&',
            'BitOr': '|',
            'BitXor': '^',
            'And': '&&',
            'Or': '||',
            'LShift': '<<',
            'RShift': '>>',
            'Not': '!',
            'USub': '-',
            'Invert': '~',
        }
        
        if hasattr(op, 'name'):
            result = op_map.get(op.name, str(op.name))
            if result != str(op.name):
                return result
            # Also try value-based lookup for enum types like BinOp
            val = getattr(op, 'value', None)
            if val is not None:
                val_map = {
                    1: '+', 2: '-', 3: '*', 4: '/', 5: '%', 6: '/',
                    8: '&', 9: '|', 10: '^', 11: '<<', 12: '>>',
                    13: '==', 14: '!=', 15: '<', 16: '<=', 17: '>', 18: '>=',
                    19: '&&', 20: '||',
                }
                return val_map.get(val, str(op.name))
            return result
        
        return op_map.get(str(op), str(op))

    def _format_concat_args(self, args: list) -> str:
        """Format arguments to zdc.concat() as SV concatenation {a, b, c}.

        Each arg is either:
          - ExprTuple([value, width]) — literal padding, e.g. (0, 24) → 24'h0
          - Any other expression — formatted normally
        """
        parts = []
        for a in args:
            if type(a).__name__ == 'ExprTuple':
                elts = getattr(a, 'elts', [])
                if len(elts) == 2:
                    val_arg = elts[0]
                    width_arg = elts[1]
                    width_val = getattr(width_arg, 'value', None)
                    val_val = getattr(val_arg, 'value', 0)
                    if isinstance(width_val, int) and width_val > 0:
                        parts.append(f"{width_val}'h{int(val_val):X}")
                        continue
                # Fallback: format each element
                parts.extend(self._format_expr(e) for e in elts)
            else:
                parts.append(self._format_expr(a))
        return '{' + ', '.join(parts) + '}'

    def _try_resolve_int_const(self, name: str):
        """Try to resolve a bare name to a module-level integer constant via sys.modules scan.

        Returns the integer value if found unambiguously, otherwise None.
        Used to convert names like MASK32 to SV integer literals.
        """
        import sys
        if not name:
            return None
        candidates = []
        for mod in sys.modules.values():
            if mod is None:
                continue
            val = getattr(mod, name, None)
            if val is None:
                continue
            if isinstance(val, int) and not isinstance(val, bool) and not isinstance(val, type):
                candidates.append(val)
        if not candidates:
            return None
        if len(set(candidates)) == 1:
            return candidates[0]
        return None

    def _try_resolve_enum_attr(self, class_name: str, member: str):
        """Try to resolve class_name.member to an integer constant via sys.modules scan.

        Returns the integer value if found unambiguously, otherwise None.
        Used to convert enum references (e.g. InstrKind.LUI) to SV integer literals.
        """
        import sys
        if not class_name:
            return None
        candidates = []
        for mod in sys.modules.values():
            if mod is None:
                continue
            cls = getattr(mod, class_name, None)
            if cls is not None and isinstance(cls, type):
                val = getattr(cls, member, None)
                if isinstance(val, int):
                    candidates.append(val)
        if len(candidates) == 1:
            return candidates[0]
        if candidates and len(set(candidates)) == 1:
            return candidates[0]  # multiple modules, same value
        return None
    
    def _generate_output_logic(self, fsm: FSMModule):
        """Generate output/datapath logic."""
        # Collect all operations from states
        has_operations = any(state.operations for state in fsm.states)
        if not has_operations:
            return

        if self.config.generate_comments:
            self._emitln("// Output and datapath logic")

        # Get output ports for reset
        output_ports = fsm.get_output_ports()
        rst = self._eff_rst(fsm)
        clk = self._eff_clk(fsm)
        act_low = self._eff_reset_active_low(fsm)
        async_ = self._eff_reset_async(fsm)
        rst_cond = f"!{rst}" if act_low else rst

        # Build always_ff sensitivity list
        self._emitln("/* verilator lint_off BLKSEQ */")
        if async_:
            edge = f"negedge {rst}" if act_low else f"posedge {rst}"
            self._emitln(f"always_ff @(posedge {clk} or {edge}) begin")
        else:
            self._emitln(f"always_ff @(posedge {clk}) begin")

        self._indent()
        self._emitln(f"if ({rst_cond}) begin")
        self._indent()

        # Reset assignments for outputs
        for port in output_ports:
            reset_val = self._format_reset_value(port.reset_value, port.width)
            self._emitln(f"{port.name} <= {reset_val};")

        # Reset struct instances (SV zero-fill is valid for packed structs).
        # Skip scratch structs (field-only writes use blocking = so no reset needed)
        # and dead ref structs (not declared at all).
        for si in fsm.struct_instances:
            if (si.instance_name not in self._scratch_structs
                    and si.instance_name not in self._dead_struct_refs):
                self._emitln(f"{si.instance_name} <= '0;")

        # Reset internal registers (skip 'state' — handled by state register block)
        for reg in fsm.registers:
            if reg.name != "state":
                reset_val = self._format_reset_value(reg.reset_value, reg.width)
                self._emitln(f"{reg.name} <= {reset_val};")

        # Reset array fields (zero all elements)
        for arr_name, depth in self._array_fields.items():
            self._emitln(f"for (int i = 0; i < {depth}; i++) {arr_name}[i] <= 32'd0;")

        self._dedent()
        self._emitln("end else begin")
        self._indent()

        # Default-clear all output ports each cycle so that valid/strobe
        # signals are asserted for exactly one clock cycle per transaction.
        # Individual states override these defaults by assigning later in the
        # same always_ff block (last-assignment-wins for non-blocking).
        for port in output_ports:
            reset_val = self._format_reset_value(port.reset_value, port.width)
            self._emitln(f"{port.name} <= {reset_val};")

        self._emitln("case (state)")
        self._indent()

        for state in fsm.states:
            if state.operations:
                comment = self._source_comment(state)
                if comment and self.config.generate_comments:
                    self._emitln(comment)
                self._emitln(f"{self._state_sv_names[state.id]}: begin")
                self._indent()
                self._state_pending.clear()
                self._generate_operations(state.operations)
                self._dedent()
                self._emitln("end")

        self._emitln("default: ; // No operation")
        self._dedent()
        self._emitln("endcase")
        self._dedent()
        self._emitln("end")
        self._dedent()
        self._emitln("end")
        self._emitln("/* verilator lint_on BLKSEQ */")
        self._emitln()
    
    def _format_reset_value(self, value: Any, width: int) -> str:
        """Format a reset value."""
        if value is None:
            return f"{width}'d0"
        if isinstance(value, int):
            return f"{width}'d{value}"
        return str(value)
    
    def _generate_operations(self, operations: List[FSMOperation]):
        """Generate code for a list of operations.

        Consecutive FSMCond operations with no else-branch that all compare the
        same discriminator expression against distinct constant values are
        automatically grouped into a ``case`` statement (recursively, enabling
        nested ``case`` for decoded AND conditions).
        """
        i = 0
        while i < len(operations):
            op = operations[i]
            # Try to build a case group starting at this position
            if isinstance(op, FSMCond) and not op.else_ops:
                result = self._group_case_arms(operations, i)
                if result is not None:
                    discrim_str, arms, next_i = result
                    self._emit_case_group(discrim_str, arms)
                    i = next_i
                    continue
            # Normal single-op emission
            if isinstance(op, FSMAssign):
                self._generate_assign(op)
            elif isinstance(op, FSMRegWrite):
                self._generate_reg_write(op)
            elif isinstance(op, FSMPortCall):
                self._generate_port_call(op)
            elif isinstance(op, FSMPortOutput):
                self._generate_port_output(op)
            elif isinstance(op, FSMCond):
                self._generate_conditional(op)
            i += 1

    # ------------------------------------------------------------------
    # Case-grouping helpers
    # ------------------------------------------------------------------

    def _extract_case_key(self, cond: Any):
        """Return ``(discrim_str, [val_strs])`` if *cond* is a case-groupable equality, else ``None``.

        Supported condition shapes:

        - ``ExprCompare(left, Eq, const)``
          → ``(fmt(left), [fmt(const)])``
        - ``ExprBool(Or, [ExprCompare(d, Eq, c1), ExprCompare(d, Eq, c2), ...])``
          → ``(fmt(d), [fmt(c1), fmt(c2), ...])`` (all terms must share the same *d*)
        - ``ExprBool(And, [C_primary, ...])``
          → ``_extract_case_key(C_primary)`` (primary discriminator is the first term)
        - ``_SyntheticAndCond([C_primary, ...])``
          → ``_extract_case_key(C_primary)``
        """
        if cond is None:
            return None
        t = type(cond).__name__

        if t == 'ExprCompare':
            ops_list = getattr(cond, 'ops', None) or []
            comparators = getattr(cond, 'comparators', None) or []
            if not ops_list or not comparators:
                return None
            op0 = ops_list[0]
            op_val = getattr(op0, 'value', None)
            if op_val is None:
                try:
                    op_val = int(op0)
                except (TypeError, ValueError):
                    return None
            if op_val == 1:  # CmpOp.Eq
                return (self._format_expr(cond.left), [self._format_expr(comparators[0])])

        elif t == 'ExprBin':
            # ExprBin(BinOp.Or, lhs, rhs) — binary OR of two comparisons
            op = getattr(cond, 'op', None)
            op_name = str(getattr(op, 'name', op) if op is not None else '')
            if 'Or' in op_name:
                lhs = getattr(cond, 'lhs', None)
                rhs = getattr(cond, 'rhs', None)
                lk = self._extract_case_key(lhs)
                rk = self._extract_case_key(rhs)
                if lk is None or rk is None or lk[0] != rk[0]:
                    return None
                return (lk[0], lk[1] + rk[1])

        elif t == 'ExprBool':
            op = getattr(cond, 'op', None)
            op_name = str(getattr(op, 'name', op) if op is not None else '')
            values = getattr(cond, 'values', None) or []
            if not values:
                return None
            if 'Or' in op_name:
                # All terms must be same-discriminator Eq comparisons
                first_key = self._extract_case_key(values[0])
                if first_key is None:
                    return None
                discrim_str, all_vals = first_key[0], list(first_key[1])
                for v in values[1:]:
                    vk = self._extract_case_key(v)
                    if vk is None or vk[0] != discrim_str:
                        return None
                    all_vals.extend(vk[1])
                return (discrim_str, all_vals)
            elif 'And' in op_name:
                # Primary discriminator is the first And term
                return self._extract_case_key(values[0])

        elif isinstance(cond, _SyntheticAndCond):
            if cond.values:
                return self._extract_case_key(cond.values[0])

        return None

    def _make_residual_cond(self, cond: Any) -> Any:
        """Return the condition remaining after stripping the primary AND term, or ``None``.

        - ``ExprBool(And, [C1, C2])``          → ``C2``
        - ``ExprBool(And, [C1, C2, C3, ...])`` → ``_SyntheticAndCond([C2, C3, ...])``
        - ``ExprBool(Or, ...)``                → ``None``
        - ``ExprCompare(Eq, ...)``             → ``None``
        - ``_SyntheticAndCond([C1])``          → ``C1`` (unwrap)
        - ``_SyntheticAndCond([C1, C2])``      → ``C2``
        - ``_SyntheticAndCond([C1, C2, ...])`` → ``_SyntheticAndCond([C2, ...])``
        """
        t = type(cond).__name__
        if t == 'ExprBool':
            op = getattr(cond, 'op', None)
            op_name = str(getattr(op, 'name', op) if op is not None else '')
            if 'And' in op_name:
                values = getattr(cond, 'values', None) or []
                if len(values) == 2:
                    return values[1]
                elif len(values) > 2:
                    return _SyntheticAndCond(values[1:])
        elif isinstance(cond, _SyntheticAndCond):
            if len(cond.values) == 1:
                return cond.values[0]
            elif len(cond.values) == 2:
                return cond.values[1]
            elif len(cond.values) > 2:
                return _SyntheticAndCond(cond.values[1:])
        return None

    def _flatten_if_else_chain(self, op: FSMCond):
        """Flatten a nested if/elif/.../else chain into a list of (cond, then_ops) pairs.

        Follows each node's ``else_ops`` as long as it is exactly one FSMCond
        (i.e. an elif branch).  Returns ``None`` if a non-singleton or
        non-FSMCond else block is encountered (real else clause), which cannot
        safely be represented as a case arm.
        """
        chain = []
        current = op
        while current is not None:
            chain.append((current.condition, current.then_ops))
            if not current.else_ops:
                current = None
            elif len(current.else_ops) == 1 and isinstance(current.else_ops[0], FSMCond):
                current = current.else_ops[0]
            else:
                # Real else clause — cannot flatten to case
                return None
        return chain if len(chain) >= 2 else None

    def _detect_case_from_chain(self, chain: list):
        """Check if all conditions in *chain* share a common discriminant.

        Returns ``(discrim_str, arms)`` in the same format expected by
        :meth:`_emit_case_group`, or ``None`` if the conditions are not
        case-groupable.
        """
        discrim_str = None
        arms: list = []
        all_vals_seen: set = set()

        for cond, then_ops in chain:
            key = self._extract_case_key(cond)
            if key is None:
                return None
            d, vals = key
            if discrim_str is None:
                discrim_str = d
            elif d != discrim_str:
                return None
            if all_vals_seen.intersection(vals):
                return None  # value conflict — not a safe case group
            all_vals_seen.update(vals)
            residual = self._make_residual_cond(cond)
            arms.append((list(vals), [(residual, then_ops)]))

        return (discrim_str, arms) if len(arms) >= 2 else None

    def _group_case_arms(self, operations: list, start: int):
        """Try to group consecutive FSMCond ops from *start* into case arms.

        Returns ``(discrim_str, arms, next_index)`` where *arms* is a list of
        ``(vals_list, inner_items)`` and *inner_items* is a list of
        ``(residual_or_None, then_ops)``.  Returns ``None`` if fewer than 2
        distinct arms can be formed (a single-arm case is not useful).

        Arms that share the same ``vals_tuple`` (i.e. AND conditions with the
        same primary opcode value) are merged: their inner items are collected
        into one arm for recursive case-grouping at the secondary level.

        Value-conflict detection: if a new op's primary value(s) already appear
        in a *different* existing arm, grouping stops — this prevents combining
        independent if-statements that assign to different variables but share
        an opcode value (e.g. ``opcode in (3, 19, 103)`` for imm-decode and
        ``opcode == 19`` for kind-decode must stay in separate groups).
        """
        op0 = operations[start]
        if not (isinstance(op0, FSMCond) and not op0.else_ops):
            return None
        key0 = self._extract_case_key(op0.condition)
        if key0 is None:
            return None
        discrim_str = key0[0]

        arms: list = []           # [(vals_list, inner_items_list)]
        arm_map: dict = {}        # vals_tuple -> index in arms
        all_vals_seen: set = set()

        j = start
        while j < len(operations):
            op = operations[j]
            if not (isinstance(op, FSMCond) and not op.else_ops):
                break
            key = self._extract_case_key(op.condition)
            if key is None or key[0] != discrim_str:
                break
            vals = key[1]
            vals_tuple = tuple(vals)
            residual = self._make_residual_cond(op.condition)

            if vals_tuple in arm_map:
                # Same primary value(s): add inner item to existing arm
                arms[arm_map[vals_tuple]][1].append((residual, op.then_ops))
            else:
                # New arm: check for conflicts with existing arms
                if all_vals_seen.intersection(vals):
                    break
                arm_map[vals_tuple] = len(arms)
                arms.append((list(vals), [(residual, op.then_ops)]))
                all_vals_seen.update(vals)
            j += 1

        if len(arms) < 2:
            return None
        return (discrim_str, arms, j)

    def _emit_case_group(self, discrim_str: str, arms: list) -> None:
        """Emit a ``case`` statement for the given arms.

        Each arm is ``(vals_list, inner_items)`` where *inner_items* is a list
        of ``(residual_or_None, then_ops)``.

        Single-item arms with no residual emit *then_ops* directly.
        Multi-item arms or arms with residuals build FSMCond nodes and call
        :meth:`_generate_operations` recursively, enabling nested case grouping
        at the secondary discriminator level (e.g. funct3 inside opcode).
        """
        self._emitln(f"unique case ({discrim_str})")
        self._indent()
        for (vals, inner_items) in arms:
            saved_pending = dict(self._state_pending)
            self._emitln(f"{', '.join(vals)}: begin")
            self._indent()
            self._cond_depth += 1
            if len(inner_items) == 1 and inner_items[0][0] is None:
                # Single item, no residual: emit then_ops directly
                self._generate_operations(inner_items[0][1])
            else:
                # Build inner op list and recurse (enables nested case grouping)
                inner_ops: list = []
                for (residual, then_ops) in inner_items:
                    if residual is not None:
                        inner_ops.append(FSMCond(condition=residual, then_ops=then_ops, else_ops=[]))
                    else:
                        inner_ops.extend(then_ops)
                self._generate_operations(inner_ops)
            self._cond_depth -= 1
            self._state_pending = dict(saved_pending)
            self._dedent()
            self._emitln("end")
        self._emitln("default: ;")
        self._dedent()
        self._emitln("endcase")

    def _generate_port_call(self, op: FSMPortCall):
        """Generate outputs for an awaited port-method call (FSMPortCall).

        In the WAIT_COND state the DUT asserts valid and drives arg signals.
        On ack the result is latched from the rdata input.
        """
        prefix = f"{op.port_name}_{op.method_name}"
        self._emitln(f"{prefix}_valid <= 1'b1;")
        for i, arg in enumerate(op.arg_exprs):
            self._emitln(f"{prefix}_arg{i} <= {self._format_expr(arg)};")
        if op.result_var:
            # Latch result when ack is asserted
            self._emitln(f"if ({prefix}_ack) {op.result_var} <= {prefix}_rdata;")

    def _generate_port_output(self, op: FSMPortOutput):
        """Generate outputs for a non-awaited port-method call (FSMPortOutput).

        Assert valid and drive arg signals for one FSM state (no ack/wait).
        """
        prefix = f"{op.port_name}_{op.method_name}"
        self._emitln(f"{prefix}_valid <= 1'b1;")
        for i, arg in enumerate(op.arg_exprs):
            self._emitln(f"{prefix}_arg{i} <= {self._format_expr(arg)};")

    def _generate_reg_write(self, op: FSMRegWrite):
        """Generate a register write: reg_name_q <= value."""
        value_str = self._format_assign_value(op.value)
        self._emitln(f"{op.reg_name}_q <= {value_str};")
    
    def _forward_subscript_index(self, target: str) -> str:
        """Replace an array subscript index variable with its forwarded expression.

        When a target like ``gpr[_execute_rd]`` is generated, ``_execute_rd``
        may have been assigned with ``<=`` (non-blocking) earlier in the same
        state, meaning its old value is still in scope at evaluation time.
        Forwarding replaces it with the expression that *would* be written
        (e.g. ``((instr >> 7) & 31)``), producing the correct target.
        """
        bracket = target.find('[')
        if bracket == -1:
            return target
        arr = target[:bracket]
        idx = target[bracket + 1:-1]
        if idx in self._state_pending:
            return f"{arr}[{self._state_pending[idx]}]"
        return target

    def _generate_assign(self, op: FSMAssign):
        """Generate assignment statement."""
        # Resolve target to a string for emission and forwarding map key.
        target = self._format_target(op.target)

        # For expression forwarding via _state_pending, use the RAW formatted
        # expression (without any width-cast wrappers added for emit context).
        # This ensures forwarded values can be safely subscripted / composed.
        raw_value = self._format_expr(op.value) if op.value is not None else "0"

        # The emit value may add a width cast (e.g. 32'(...) for ExprCbit).
        value = self._format_assign_value(op.value)

        # Record the RAW value for within-state expression forwarding.
        # Track scalar targets (no array index) and struct references (has dot).
        if '[' not in target:
            self._state_pending[target] = raw_value

        # Array-indexed targets (e.g. gpr[rd]) are state elements that must
        # always use non-blocking assignment (<=). Their subscript index is
        # forwarded through _state_pending so an index variable assigned with
        # <= earlier in the same state resolves to the correct expression
        # (e.g. _execute_rd → ((instr >> 7) & 31)) rather than the old value.
        is_array = '[' in target
        if is_array:
            target = self._forward_subscript_index(target)

        # Inside a conditional branch (_cond_depth > 0) use blocking assignment
        # for scalar temporaries so downstream reads within the same always_ff
        # evaluation see the freshly computed value. Array-indexed state
        # elements always use non-blocking regardless of nesting depth.
        # ExprStructField targets are intermediate decode/execute scratch fields
        # (e.g. dec_out.rs2) computed combinationally within the state; they
        # must use blocking = so subsequent reads see the new value without
        # relying solely on codegen forwarding.
        is_struct_field = type(op.target).__name__ == 'ExprStructField'
        use_nonblocking = op.is_nonblocking and (self._cond_depth == 0 or is_array) and not is_struct_field

        # Struct field targets have a defined width from the typedef — use the raw
        # expression without any 32'(...) wrapper to avoid WIDTHTRUNC warnings.
        if is_struct_field:
            value = raw_value

        # For narrow output ports (e.g. dmem_wmask is 4-bit, dmem_we is 1-bit),
        # the internal 32-bit register may be wider. Use a SV width cast to
        # truncate cleanly without triggering WIDTHTRUNC.
        raw_target = target.split('[')[0]  # strip any array index
        pw = self._port_widths.get(raw_target, 0)
        if pw > 0 and pw < 32:
            value = f"{pw}'({value})"

        if use_nonblocking:
            self._emitln(f"{target} <= {value};")
        else:
            self._emitln(f"{target} = {value};")
    
    def _format_assign_value(self, value: Any) -> str:
        """Format the RHS of an assignment."""
        if value is None:
            return "0"

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, str):
            return value

        if isinstance(value, tuple) and len(value) == 3:
            # Augmented assignment: (lhs, op, rhs) — elements may be strings, IR nodes, or AugOp
            lhs, op, rhs = value
            lhs_str = self._format_expr(lhs)
            rhs_str = self._format_expr(rhs)
            # op may be AugOp enum, '+'/'-' string, or operator string
            t = type(op).__name__
            if t == 'AugOp':
                aug_map = {1: '+', 2: '-', 3: '*', 4: '/'}
                op_str = aug_map.get(op.value, str(op.value))
            else:
                op_str = self._format_operator(op)
            return f"{lhs_str} {op_str} {rhs_str}"

        # Comparison results are 1-bit; zero-extend to 32 bits for register assignment.
        if type(value).__name__ in ('ExprCompare', 'ExprCbit'):
            return f"32'({self._format_expr(value)})"

        # Bit-slice results are narrower than 32 bits; zero-extend for register assignment.
        # Handles ExprSubscript (static slice), ExprAttribute (resolved bit-field),
        # and ExprStructField (packed struct field may be narrower than 32 bits).
        if type(value).__name__ in ('ExprSubscript', 'ExprAttribute', 'ExprStructField'):
            slc = getattr(value, 'slice', None)
            if type(value).__name__ == 'ExprSubscript':
                if slc is not None and type(slc).__name__ == 'ExprSlice':
                    return f"32'({self._format_expr(value)})"
            elif type(value).__name__ == 'ExprStructField':
                # Struct fields may be narrower than 32 bits; always zero-extend.
                return f"32'({self._format_expr(value)})"
            else:
                # ExprAttribute — wrap only if the formatted result contains '[',
                # indicating the field resolves to a bit-slice expression.
                fmtd = self._format_expr(value)
                if '[' in fmtd:
                    return f"32'({fmtd})"

        # zdc.bv32(x) / int(x) in assignment context: zero-extend to 32 bits.
        # 'int' is how Python's built-in int() cast appears in the IR when a
        # narrow subscript is zero-extended (e.g. LBU/LHU in memory.py).
        # Must be handled here (not in _format_expr) because 32'(expr)[slice]
        # is illegal SV syntax when the base is used with a subscript later.
        if type(value).__name__ == 'ExprCall':
            func = getattr(value, 'func', None)
            if func is not None:
                func_str_inner = self._format_expr(func)
                if func_str_inner in ('zdc_bv32', 'int'):
                    args = getattr(value, 'args', [])
                    if len(args) == 1:
                        return f"32'({self._format_expr(args[0])})"

        return self._format_expr(value)
    
    def _generate_conditional(self, op: FSMCond):
        """Generate conditional (if/else) statement.

        Assignments inside a branch use blocking ``=`` so that subsequent reads
        within the same ``always_ff`` block see the updated values.  The outer
        ``_state_pending`` map is saved before each branch and restored
        afterwards so that conditional assignments never pollute the
        forwarding scope of the enclosing state.

        When the if/else forms a pure elif chain on a single discriminant
        (e.g. constraint-driven match/case dispatch), the chain is detected
        and emitted as a ``unique case`` statement instead.
        """
        chain = self._flatten_if_else_chain(op)
        if chain is not None:
            result = self._detect_case_from_chain(chain)
            if result is not None:
                self._emit_case_group(*result)
                return

        cond_str = self._format_condition(op.condition)
        saved_pending = dict(self._state_pending)

        self._emitln(f"if ({cond_str}) begin")
        self._indent()
        self._cond_depth += 1
        self._generate_operations(op.then_ops)
        self._cond_depth -= 1
        self._state_pending = dict(saved_pending)
        self._dedent()

        if op.else_ops:
            self._emitln("end else begin")
            self._indent()
            self._cond_depth += 1
            self._generate_operations(op.else_ops)
            self._cond_depth -= 1
            self._state_pending = dict(saved_pending)
            self._dedent()

        self._emitln("end")
    
    def _generate_module_end(self):
        """Generate module end."""
        self._emitln("endmodule")


def _add_port(fsm: FSMModule, known: set, name: str, direction: str, width: int):
    """Add *name* as a port to *fsm* if not already present."""
    if name not in known:
        fsm.ports.append(FSMPort(name=name, direction=direction, width=width))
        known.add(name)


def generate_sv(fsm: FSMModule, config: Optional[SVGenConfig] = None) -> str:
    """Convenience function to generate SystemVerilog code.
    
    When ``config`` is ``None``, the reset style is auto-derived from
    ``fsm.reset_async`` and ``fsm.reset_active_low``:

    ==============================  ==================
    fsm fields                      Derived reset_style
    ==============================  ==================
    reset_async=True,  active_low   ASYNC_LOW
    reset_async=True,  active_high  ASYNC_HIGH
    reset_async=False, active_low   SYNC_LOW
    reset_async=False, active_high  SYNC_HIGH
    ==============================  ==================

    Args:
        fsm: FSM module to generate
        config: Optional generation configuration; overrides auto-derivation.
        
    Returns:
        SystemVerilog source code
    """
    if config is None:
        # Prefer domain_binding for reset style derivation
        if fsm.domain_binding is not None:
            reset_async    = fsm.domain_binding.reset_async
            reset_act_low  = fsm.domain_binding.reset_active_low
        else:
            reset_async   = fsm.reset_async
            reset_act_low = fsm.reset_active_low
        if reset_async and reset_act_low:
            reset_style = ResetStyle.ASYNC_LOW
        elif reset_async and not reset_act_low:
            reset_style = ResetStyle.ASYNC_HIGH
        elif not reset_async and reset_act_low:
            reset_style = ResetStyle.SYNC_LOW
        else:
            reset_style = ResetStyle.SYNC_HIGH
        config = SVGenConfig(reset_style=reset_style)
    generator = SVCodeGenerator(config)
    return generator.generate(fsm)


def generate_regfile_sv(meta, module_prefix: str = "") -> str:
    """Generate SystemVerilog for all IndexedRegFile fields in *meta*.

    Args:
        meta: ``ComponentSynthMeta`` instance (from the elaborator).
        module_prefix: Optional prefix string applied to all generated module names.

    Returns:
        Concatenated SV source for every register file declared in the
        component, separated by blank lines.  Returns an empty string if
        the component has no ``IndexedRegFile`` fields.
    """
    from .regfile_synth import RegFileHazardAnalyzer, RegFileSVGenerator

    analyzer  = RegFileHazardAnalyzer()
    generator = RegFileSVGenerator()
    parts = []

    for decl in meta.regfiles:
        hazards = analyzer.analyze(decl)
        sv      = generator.generate(decl, hazards, module_prefix=module_prefix)
        parts.append(sv)

    return '\n\n'.join(parts)
