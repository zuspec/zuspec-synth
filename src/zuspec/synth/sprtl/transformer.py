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
SPRTL Transformer: Converts sync processes to FSM representations.

This transformer takes a zuspec.dataclasses IR representation of a
synchronous process and converts it to an FSM IR suitable for HDL generation.

Key transformation rules:
1. Statements before 'while True:' become reset initialization
2. Each 'await' creates a new FSM state boundary
3. Operations between 'await's are grouped into a single state
4. 'await <condition>' creates a WAIT_COND state
5. 'await zdc.cycles(N)' creates WAIT_CYCLES state(s)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMTransition,
    FSMOperation, FSMAssign, FSMCond, FSMPort, FSMRegister,
    FSMRegRead, FSMRegWrite, FSMMemRequest, FSMMemResponse,
    FSMPortCall, FSMPortOutput,
)

# IR expression/statement types used for inline-method rewriting
try:
    from zuspec.ir.core.expr import (
        ExprRefParam, ExprRefLocal, ExprRefField, ExprRefUnresolved,
        ExprConstant, ExprCall, ExprAttribute, ExprAwait, ExprBin,
        TypeExprRefSelf,
    )
    from zuspec.ir.core.stmt import (
        StmtAssign, StmtAugAssign, StmtFor, StmtExpr, StmtReturn,
        StmtWhile, StmtIf,
    )
    _HAVE_IR = True
except ImportError:
    _HAVE_IR = False


@dataclass
class TransformContext:
    """Context for the transformation pass."""
    module: FSMModule
    current_state: Optional[FSMState] = None
    state_counter: int = 0
    reset_operations: List[FSMOperation] = field(default_factory=list)
    in_reset_section: bool = True  # True until we hit 'while True'
    # Set by _transform_await_method_call; read by _transform_return.
    _inline_result_var: Optional[str] = None
    
    def new_state(self, name: str, kind: FSMStateKind = FSMStateKind.NORMAL,
                  **kwargs) -> FSMState:
        """Create a new state."""
        state = self.module.add_state(name, kind, **kwargs)
        self.state_counter += 1
        return state
    
    def make_state_name(self, base: str = "S") -> str:
        """Generate a unique state name."""
        return f"{base}_{self.state_counter}"


class SPRTLTransformer:
    """Transforms SPRTL (sync process) IR to FSM IR.
    
    Usage:
        transformer = SPRTLTransformer()
        fsm_module = transformer.transform(component_ir, process_ir)
    """
    
    def __init__(self):
        self._ctx: Optional[TransformContext] = None
        self._field_names: Dict[int, str] = {}  # ExprRefField.index → field name
        self._field_kinds: Dict[int, Any] = {}  # ExprRefField.index → FieldKind
        self._component_ir: Any = None           # saved for function lookup in inlining

    def transform(self, component_ir: Any, process_ir: Any) -> FSMModule:
        """Transform a sync process to an FSM module.

        Args:
            component_ir: The DataTypeComponent IR node
            process_ir: The Function or Process IR node for the process

        Returns:
            FSMModule representing the transformed FSM
        """
        # Extract module name from component
        module_name = getattr(component_ir, 'name', 'fsm') or 'fsm'

        # Create the FSM module
        module = FSMModule(name=module_name)

        # Set up transformation context
        self._ctx = TransformContext(module=module)
        self._component_ir = component_ir

        # Build field-index → name map and field-index → kind map
        fields = getattr(component_ir, 'fields', [])
        self._field_names = {
            i: getattr(f, 'name', f'field_{i}')
            for i, f in enumerate(fields)
        }
        self._field_kinds = {
            i: getattr(f, 'kind', None)
            for i, f in enumerate(fields)
        }

        # Store field names and array fields in the module for sv_codegen
        module.field_names = dict(self._field_names)
        for i, f in enumerate(fields):
            dt = getattr(f, 'datatype', None)
            if dt is not None and type(dt).__name__ == 'DataTypeArray':
                depth = getattr(dt, 'size', 0)
                name = self._field_names.get(i, f'field_{i}')
                if depth and depth > 0:
                    module.array_fields[name] = depth
        
        # Extract clock and reset from process metadata
        metadata = getattr(process_ir, 'metadata', {})
        if 'clock' in metadata:
            # The clock/reset are stored as ExprRefField - extract signal name
            clock_expr = metadata['clock']
            module.clock_signal = self._extract_signal_name(clock_expr, 'clk')
        if 'reset' in metadata:
            reset_expr = metadata['reset']
            module.reset_signal = self._extract_signal_name(reset_expr, 'rst_n')
        module.reset_async = bool(metadata.get('reset_async', False))
        module.reset_active_low = bool(metadata.get('reset_active_low', True))
        
        # Extract ports from component fields
        self._extract_ports(component_ir)
        
        # Transform the process body
        body = getattr(process_ir, 'body', [])
        self._transform_body(body)
        
        # Finalize the FSM
        self._finalize()
        
        return module
    
    def _extract_signal_name(self, expr: Any, default: str) -> str:
        """Extract signal name from an IR expression."""
        # Handle ExprRefField
        if hasattr(expr, 'index'):
            # We'd need the component to map index -> name
            # For now, return the default
            return default
        if hasattr(expr, 'name'):
            return expr.name
        return default
    
    def _is_protocol_port_field(self, field_idx: int) -> bool:
        """Return True if the field at *field_idx* is a ProtocolPort or ProtocolExport."""
        try:
            from zuspec.ir.core.fields import FieldKind
            kind = self._field_kinds.get(field_idx)
            return kind in (
                FieldKind.ProtocolPort, FieldKind.ProtocolExport,
                FieldKind.CallablePort, FieldKind.CallableExport,
            )
        except ImportError:
            return False

    def _extract_ports(self, component_ir: Any):
        """Extract port declarations from component fields."""
        fields = getattr(component_ir, 'fields', [])
        
        for f in fields:
            name = getattr(f, 'name', None)
            if not name:
                continue
            
            kind = getattr(f, 'kind', None)
            
            # Determine direction
            direction = None
            if kind and hasattr(kind, 'name'):
                kind_name = kind.name
                if kind_name == 'INPUT':
                    direction = 'input'
                elif kind_name == 'OUTPUT':
                    direction = 'output'
            
            if direction is None:
                continue
            
            # Get width from type
            width = 1
            dtype = getattr(f, 'dtype', None)
            if dtype and hasattr(dtype, 'bits'):
                width = dtype.bits
            
            # Get reset value from metadata
            reset_value = None
            metadata = getattr(f, 'metadata', {}) or {}
            if 'reset' in metadata:
                reset_value = metadata['reset']
            
            self._ctx.module.add_port(name, direction, width, reset_value)
    
    def _transform_body(self, stmts: List[Any]):
        """Transform a list of statements."""
        for stmt in stmts:
            self._transform_stmt(stmt)
    
    def _transform_stmt(self, stmt: Any):
        """Transform a single statement."""
        stmt_type = type(stmt).__name__
        
        # Dispatch based on statement type
        if stmt_type == 'StmtWhile':
            self._transform_while(stmt)
        elif stmt_type == 'StmtIf':
            self._transform_if(stmt)
        elif stmt_type == 'StmtAssign':
            self._transform_assign(stmt)
        elif stmt_type == 'StmtAugAssign':
            self._transform_aug_assign(stmt)
        elif stmt_type == 'StmtExpr':
            self._transform_expr_stmt(stmt)
        elif stmt_type == 'StmtFor':
            self._transform_for(stmt)
        elif stmt_type == 'StmtReturn':
            self._transform_return(stmt)
        elif stmt_type == 'StmtAnnAssign':
            self._transform_ann_assign(stmt)
        # Add more statement types as needed
    
    def _transform_ann_assign(self, stmt: Any):
        """Transform an annotated assignment (``name: type = value``).

        In the reset section these declare registers; skip them here (already
        handled at finalize time).  Outside the reset section they arise from
        inlined helper bodies where the DMF emits tuple temporaries:

            _tu_0: _zsp_tuple = (expr0, expr1, ...)

        We expand each such tuple into flat scalar assignments:

            _tu_0_v0 <= expr0
            _tu_0_v1 <= expr1
            ...

        Plain scalar annotated assignments (e.g. ``pc: int = 0``) are treated
        as ordinary non-blocking assignments.
        """
        if self._ctx.in_reset_section:
            return
        if not self._ctx.current_state:
            return

        target = getattr(stmt, 'target', None)
        value  = getattr(stmt, 'value', None)
        if target is None or value is None:
            return

        target_name = self._expr_to_name(target)

        # Tuple binding: ExprTuple value -> expand to _v0, _v1, ...
        if type(value).__name__ == 'ExprTuple':
            for i, elt in enumerate(getattr(value, 'elts', [])):
                flat_name = f"{target_name}_v{i}"
                self._ctx.current_state.add_operation(
                    FSMAssign(target=flat_name, value=elt, is_nonblocking=True)
                )
        else:
            # Scalar annotated assignment
            self._ctx.current_state.add_operation(
                FSMAssign(target=target_name, value=value, is_nonblocking=True)
            )
    
    def _transform_while(self, stmt: Any):
        """Transform a while statement.
        
        'while True:' marks the end of reset section and start of FSM body.
        """
        test = getattr(stmt, 'test', None)
        
        # Check if this is 'while True:'
        is_while_true = False
        if test and hasattr(test, 'value') and test.value is True:
            is_while_true = True
        
        if is_while_true:
            # End reset section, start FSM body
            self._ctx.in_reset_section = False
            
            # Create initial state
            initial_state = self._ctx.new_state("IDLE", FSMStateKind.NORMAL)
            self._ctx.current_state = initial_state
            self._ctx.module.initial_state = initial_state.id
            
            # Transform the while body
            body = getattr(stmt, 'body', [])
            self._transform_body(body)
            
            # Add loop-back transition from last state to initial state
            if self._ctx.current_state and self._ctx.current_state.id != initial_state.id:
                # Only add if we haven't already added transitions
                if not self._ctx.current_state.transitions:
                    self._ctx.current_state.add_transition(initial_state.id)
        else:
            # Regular while loop - transform condition and body
            self._transform_loop(stmt)
    
    def _transform_loop(self, stmt: Any):
        """Transform a regular (non-while-True) loop."""
        # Create a loop state that checks condition
        test = getattr(stmt, 'test', None)
        body = getattr(stmt, 'body', [])
        
        loop_state = self._ctx.new_state(
            self._ctx.make_state_name("LOOP"),
            FSMStateKind.WAIT_COND,
            wait_condition=test
        )
        
        # Transition from current state to loop state
        if self._ctx.current_state:
            self._ctx.current_state.add_transition(loop_state.id)
        
        self._ctx.current_state = loop_state
        
        # Transform loop body
        self._transform_body(body)
        
        # Add back-edge to loop state with condition
        if self._ctx.current_state:
            self._ctx.current_state.add_transition(loop_state.id, condition=test)
    
    def _branch_has_await(self, stmts: List[Any]) -> bool:
        """Return True if any statement in *stmts* (recursively) contains ExprAwait."""
        for s in stmts:
            t = type(s).__name__
            if t == 'StmtAssign':
                val = getattr(s, 'value', None)
                if type(val).__name__ == 'ExprAwait':
                    return True
            elif t == 'StmtExpr':
                # StmtExpr uses .expr, not .value
                val = getattr(s, 'expr', None)
                if type(val).__name__ == 'ExprAwait':
                    return True
            if t in ('StmtIf', 'StmtWhile', 'StmtFor'):
                if self._branch_has_await(getattr(s, 'body', [])):
                    return True
                if self._branch_has_await(getattr(s, 'orelse', [])):
                    return True
        return False

    def _transform_if(self, stmt: Any):
        """Transform an if statement.

        All branches of an if/else must be *symmetric* with respect to ``await``:
        either every branch contains an await (multi-cycle) or none do (single-cycle
        combinational).  An asymmetric if/else is a synthesis error.

        * No-await branches → single :class:`FSMCond` op in the current state.
        * All-await branches → FSM fork/join: separate state chains per branch that
          reconverge at a common join state.
        """
        test = getattr(stmt, 'test', None)
        then_body = getattr(stmt, 'body', [])
        else_body = getattr(stmt, 'orelse', [])
        
        if self._ctx.in_reset_section:
            return
        
        if not self._ctx.current_state:
            return

        then_has_await = self._branch_has_await(then_body)
        else_has_await = self._branch_has_await(else_body)

        # Asymmetric branches (one arm awaits, the other doesn't) are allowed.
        # We treat the non-awaiting arm as a pass-through state that transitions
        # immediately to the join, consuming one clock cycle.  This keeps the
        # FSM flat and correct without requiring the user to add dummy ticks.
        either_has_await = then_has_await or else_has_await

        if not either_has_await:
            # ── Pure combinational ─────────────────────────────────────────
            then_ops: List[FSMOperation] = []
            else_ops: List[FSMOperation] = []
            for s in then_body:
                then_ops.extend(self._stmt_to_operations(s))
            for s in else_body:
                else_ops.extend(self._stmt_to_operations(s))
            if then_ops or else_ops:
                self._ctx.current_state.add_operation(
                    FSMCond(condition=test, then_ops=then_ops, else_ops=else_ops)
                )
        else:
            # ── Multi-cycle: fork/join ─────────────────────────────────────
            # fork_state holds the conditional transition; its then/else arms
            # are separate state chains that both converge to join_state.
            fork_state = self._ctx.current_state

            # Transform then-branch into its own state chain
            then_entry = self._ctx.new_state(
                self._ctx.make_state_name("S_IF"), FSMStateKind.NORMAL
            )
            fork_state.add_transition(then_entry.id, condition=test)
            self._ctx.current_state = then_entry
            self._transform_body(then_body)
            then_end = self._ctx.current_state  # last state after then-chain

            # Transform else-branch (if present) into its own state chain
            if else_body:
                else_entry = self._ctx.new_state(
                    self._ctx.make_state_name("S_ELSE"), FSMStateKind.NORMAL
                )
                fork_state.add_transition(else_entry.id)  # unconditional → else
                self._ctx.current_state = else_entry
                self._transform_body(else_body)
                else_end = self._ctx.current_state
            else:
                else_end = None

            # Create common join state and wire both arms to it
            join_state = self._ctx.new_state(
                self._ctx.make_state_name("S_JOIN"), FSMStateKind.NORMAL
            )
            if then_end:
                then_end.add_transition(join_state.id)
            if else_end:
                else_end.add_transition(join_state.id)
            else:
                # No else branch: fork goes directly to join on the else arm
                fork_state.add_transition(join_state.id)

            self._ctx.current_state = join_state
    
    def _stmt_to_operations(self, stmt: Any) -> List[FSMOperation]:
        """Convert a statement to FSM operations."""
        ops = []
        stmt_type = type(stmt).__name__
        
        if stmt_type == 'StmtAssign':
            targets = getattr(stmt, 'targets', [])
            value = getattr(stmt, 'value', None)
            for target in targets:
                target_name = self._expr_to_name(target)
                ops.append(FSMAssign(target=target_name, value=value))
        
        elif stmt_type == 'StmtAnnAssign':
            target = getattr(stmt, 'target', None)
            value  = getattr(stmt, 'value', None)
            if target is not None and value is not None:
                target_name = self._expr_to_name(target)
                if type(value).__name__ == 'ExprTuple':
                    for i, elt in enumerate(getattr(value, 'elts', [])):
                        ops.append(FSMAssign(target=f"{target_name}_v{i}", value=elt))
                else:
                    ops.append(FSMAssign(target=target_name, value=value))

        elif stmt_type == 'StmtAugAssign':
            target = getattr(stmt, 'target', None)
            op = getattr(stmt, 'op', None)
            value = getattr(stmt, 'value', None)
            target_name = self._expr_to_name(target)
            # Aug assign: target = target op value
            ops.append(FSMAssign(target=target_name, value=(target, op, value)))
        
        elif stmt_type == 'StmtIf':
            # Nested if
            test = getattr(stmt, 'test', None)
            then_body = getattr(stmt, 'body', [])
            else_body = getattr(stmt, 'orelse', [])
            then_ops = []
            else_ops = []
            for s in then_body:
                then_ops.extend(self._stmt_to_operations(s))
            for s in else_body:
                else_ops.extend(self._stmt_to_operations(s))
            ops.append(FSMCond(condition=test, then_ops=then_ops, else_ops=else_ops))
        
        return ops
    
    def _expr_to_name(self, expr: Any) -> str:
        """Convert an expression to a signal name.

        For simple locals/params this returns the identifier name.
        For array-field subscripts (``self.arr[idx]``) it returns
        ``arr_name[idx_repr]`` so the FSMAssign target can be
        rendered as an indexed write in SV.
        """
        if expr is None:
            return "unknown"
        t = type(expr).__name__
        if hasattr(expr, 'name'):
            return expr.name
        if t == 'ExprSubscript':
            val_expr = getattr(expr, 'value', None)
            slc = getattr(expr, 'slice', None)
            if (val_expr is not None
                    and type(val_expr).__name__ == 'ExprRefField'
                    and type(getattr(val_expr, 'base', None)).__name__ == 'TypeExprRefSelf'):
                arr_name = self._field_names.get(val_expr.index, f'field_{val_expr.index}')
                idx_name = self._expr_to_name(slc) if slc is not None else '?'
                return f"{arr_name}[{idx_name}]"
        if hasattr(expr, 'attr'):
            # ExprAttribute: for local-variable attribute access (e.g. dec.rd)
            # flatten to base_attr so the register name matches what the
            # synthesiser assigned.  For self-field chains (e.g. self.mem) the
            # value is ExprRefField which has an 'index' attribute rather than
            # a 'name', so this branch is not taken.
            value = getattr(expr, 'value', None)
            if value is not None and type(value).__name__ == 'ExprRefLocal':
                base_name = getattr(value, 'name', '')
                return f"{base_name}_{expr.attr}" if base_name else expr.attr
            return expr.attr
        if hasattr(expr, 'index'):
            # ExprRefField — look up by index
            return self._field_names.get(expr.index, f'field_{expr.index}')
        return "unknown"
    
    def _transform_assign(self, stmt: Any):
        """Transform an assignment statement."""
        if self._ctx.in_reset_section:
            return

        if not self._ctx.current_state:
            return

        targets = getattr(stmt, 'targets', [])
        value = getattr(stmt, 'value', None)

        # Await on the RHS — the target variable receives the result
        if value is not None and type(value).__name__ == 'ExprAwait':
            # For tuple unpacking (e.g. src, dst, length, ctrl = await read_all(...))
            # pass the full target list so read_all can name each register result.
            first_target = targets[0] if targets else None
            if first_target is not None and type(first_target).__name__ == 'ExprTuple':
                result_vars = [
                    self._expr_to_name(e)
                    for e in getattr(first_target, 'elts', [])
                ]
                self._transform_await(value, result_var=result_vars)
            else:
                result_var = self._expr_to_name(first_target) if first_target else None
                self._transform_await(value, result_var=result_var)
            return

        # Detect next(generator_exp, default) — synthesise as priority encoder.
        if value is not None and self._is_next_generator_call(value):
            for target in targets:
                target_name = self._expr_to_name(target)
                self._transform_priority_encode(target_name, value)
            return

        for target in targets:
            target_name = self._expr_to_name(target)
            self._ctx.current_state.add_operation(
                FSMAssign(target=target_name, value=value)
            )
    
    def _transform_aug_assign(self, stmt: Any):
        """Transform an augmented assignment (+=, -=, etc.)."""
        if self._ctx.in_reset_section:
            return
        
        if not self._ctx.current_state:
            return
        
        target = getattr(stmt, 'target', None)
        op = getattr(stmt, 'op', None)
        value = getattr(stmt, 'value', None)
        
        target_name = self._expr_to_name(target)
        self._ctx.current_state.add_operation(
            FSMAssign(target=target_name, value=(target, op, value))
        )
    
    def _transform_expr_stmt(self, stmt: Any):
        """Transform an expression statement (e.g., await or bare port call)."""
        expr = getattr(stmt, 'expr', None)
        if not expr:
            return

        expr_type = type(expr).__name__

        if expr_type == 'ExprAwait':
            self._transform_await(expr)
        elif expr_type == 'ExprCall':
            # Non-awaited call — check if it's a port-method output
            func = getattr(expr, 'func', None)
            receiver = getattr(func, 'value', None) if func else None
            if (receiver is not None
                    and type(receiver).__name__ == 'ExprRefField'
                    and self._is_protocol_port_field(receiver.index)):
                self._transform_port_output(expr)

    def _transform_await(self, expr: Any, result_var=None):
        """Transform an await expression.

        Dispatch is receiver-kind-based:
        - ExprRefField whose kind is ProtocolPort/ProtocolExport → generic port-method handler
        - ExprRefField whose kind is Field/Reg → zdc framework method dispatch (read/write/…)
        - TypeExprRefSelf → self.wait / zdc.cycles / private helper inlining

        ``result_var`` carries the name (str) or names (list[str]) of the
        local variable(s) that receive the awaited value.
        """
        if self._ctx.in_reset_section:
            return

        await_value = getattr(expr, 'value', None)
        if not await_value:
            return

        await_type = type(await_value).__name__

        if await_type == 'ExprCall':
            func = getattr(await_value, 'func', None)
            if func:
                attr = getattr(func, 'attr', None)
                receiver = getattr(func, 'value', None)
                receiver_type = type(receiver).__name__ if receiver is not None else ''

                if receiver_type == 'ExprRefField':
                    # Dispatch by field kind
                    if self._is_protocol_port_field(receiver.index):
                        # Generic protocol port-method call
                        self._transform_await_port_method(await_value, result_var)
                        return
                    else:
                        # Register or other framework field — dispatch by method name
                        if attr == 'read':
                            self._transform_await_reg_read(await_value, result_var)
                            return
                        elif attr == 'write':
                            self._transform_await_reg_write(await_value)
                            return
                        elif attr == 'read_all':
                            self._transform_await_read_all(await_value, result_var)
                            return
                        elif attr == 'wait':
                            self._transform_await_reg_wait(await_value, result_var)
                            return

                elif receiver_type == 'TypeExprRefSelf':
                    if attr == 'wait':
                        self._transform_await_idle(await_value)
                        return
                    elif self._is_cycles_call(func):
                        self._transform_await_cycles(await_value)
                        return
                    elif attr is not None and (
                        (attr.startswith('_') and not attr.startswith('__'))
                        or any(
                            getattr(fn, 'name', None) == attr
                            for fn in getattr(self._component_ir, 'functions', [])
                        )
                    ):
                        # Private or public async helper on self — inline it.
                        self._transform_await_method_call(await_value, result_var)
                        return

                elif self._is_cycles_call(func):
                    self._transform_await_cycles(await_value)
                    return

        if await_type == 'ExprCompare':
            self._transform_await_condition(await_value)
            return

        # Generic await — treat as wait-for-condition
        self._transform_await_condition(await_value)

    # ------------------------------------------------------------------
    # Await helper: extract names from IR expressions
    # ------------------------------------------------------------------

    def _extract_reg_name(self, func_expr: Any) -> str:
        """Return the register attribute name from a .read()/.write() call func.

        For ``ExprAttribute(attr='read', value=ExprAttribute(attr='ctrl', …))``
        this returns ``'ctrl'``.
        """
        reg_expr = getattr(func_expr, 'value', None)
        attr = getattr(reg_expr, 'attr', None) if reg_expr is not None else None
        return attr if attr else 'reg'

    def _extract_port_name(self, func_expr: Any) -> str:
        """Return the port name from a .request()/.response() call func.

        ``func_expr.value`` is an ``ExprRefField`` whose ``index`` maps to a
        field name via ``self._field_names``.
        """
        port_expr = getattr(func_expr, 'value', None) if func_expr else None
        if port_expr is not None and type(port_expr).__name__ == 'ExprRefField':
            idx = getattr(port_expr, 'index', 0)
            return self._field_names.get(idx, f'port_{idx}')
        return 'mem'

    # ------------------------------------------------------------------
    # Await handlers: reg read / write
    # ------------------------------------------------------------------

    def _transform_await_reg_read(self, call_expr: Any,
                                   result_var: Optional[str] = None):
        """Transform ``result = await reg.read()`` → single NORMAL state."""
        reg_name = self._extract_reg_name(getattr(call_expr, 'func', None))
        state_name = f"{reg_name.upper()}_READ"

        reg_state = self._ctx.new_state(state_name, FSMStateKind.NORMAL)
        reg_state.add_operation(
            FSMRegRead(reg_name=reg_name, result_var=result_var or '')
        )

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(reg_state.id)

        self._ctx.current_state = reg_state

    def _transform_await_reg_write(self, call_expr: Any):
        """Transform ``await reg.write(value)`` → single NORMAL state."""
        reg_name = self._extract_reg_name(getattr(call_expr, 'func', None))
        args = getattr(call_expr, 'args', [])
        write_value = args[0] if args else None
        state_name = f"{reg_name.upper()}_WRITE"

        reg_state = self._ctx.new_state(state_name, FSMStateKind.NORMAL)
        reg_state.add_operation(
            FSMRegWrite(reg_name=reg_name, value=write_value)
        )

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(reg_state.id)

        self._ctx.current_state = reg_state

    # ------------------------------------------------------------------
    # Await handler: generic protocol-port method call
    # ------------------------------------------------------------------

    def _transform_await_port_method(self, call_expr: Any,
                                      result_var: Optional[str] = None):
        """Transform ``result = await self.PORT.METHOD(args)`` → WAIT_COND state.

        Creates a handshake state that:
        - Asserts PORT_METHOD_valid and drives PORT_METHOD_argN outputs
        - Stays in the state (self-loop) while PORT_METHOD_ack is not asserted
        - On ack: latches PORT_METHOD_rdata into result_var (if non-void)
        - Advances to a DONE state after the handshake
        """
        func = getattr(call_expr, 'func', None)
        receiver = getattr(func, 'value', None)
        method_name = getattr(func, 'attr', 'call')
        field_idx = getattr(receiver, 'index', 0)
        port_name = self._field_names.get(field_idx, f'port_{field_idx}')
        args = getattr(call_expr, 'args', [])

        state_name = f"{port_name.upper()}_{method_name.upper()}_REQ"
        ack_signal = f"{port_name}_{method_name}_ack"

        req_state = self._ctx.new_state(
            state_name, FSMStateKind.WAIT_COND, wait_condition=ack_signal
        )
        req_state.add_operation(FSMPortCall(
            port_name=port_name,
            method_name=method_name,
            arg_exprs=list(args),
            result_var=result_var or '',
        ))
        req_state.add_transition(req_state.id, priority=1)  # self-loop while !ack

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(req_state.id)

        # Advance to a post-handshake NORMAL state on ack
        done_state = self._ctx.new_state(f"{state_name}_DONE", FSMStateKind.NORMAL)
        req_state.add_transition(done_state.id, condition=ack_signal)

        self._ctx.current_state = done_state

    def _transform_port_output(self, call_expr: Any):
        """Transform a non-awaited ``self.PORT.METHOD(args)`` → FSMPortOutput op.

        Emits the port output signals in the current FSM state for one clock
        cycle with no ack/wait (fire-and-forget / void output).
        """
        if self._ctx.current_state is None:
            return
        func = getattr(call_expr, 'func', None)
        receiver = getattr(func, 'value', None)
        method_name = getattr(func, 'attr', 'call')
        field_idx = getattr(receiver, 'index', 0)
        port_name = self._field_names.get(field_idx, f'port_{field_idx}')
        args = getattr(call_expr, 'args', [])

        self._ctx.current_state.add_operation(FSMPortOutput(
            port_name=port_name,
            method_name=method_name,
            arg_exprs=list(args),
        ))

    # ------------------------------------------------------------------
    # Await handlers: memory-interface request / response (legacy)

    def _transform_await_mem_request(self, call_expr: Any):
        """Transform ``await port.request(req)`` → WAIT_COND state.

        The FSM stays in this state until ``{port}_req_ready`` is asserted.
        """
        port_name = self._extract_port_name(getattr(call_expr, 'func', None))
        args = getattr(call_expr, 'args', [])
        req_var = self._expr_to_name(args[0]) if args else ''
        cond_signal = f"{port_name}_req_ready"
        state_name = f"{port_name.upper()}_REQ"

        req_state = self._ctx.new_state(
            state_name, FSMStateKind.WAIT_COND, wait_condition=cond_signal
        )
        req_state.add_operation(FSMMemRequest(port_name=port_name, req_var=req_var))
        # Self-loop (stay) when not ready; advance when ready
        req_state.add_transition(req_state.id, priority=1)

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(req_state.id)

        # Post-request state (advance when ready)
        post_state = self._ctx.new_state(f"{state_name}_DONE", FSMStateKind.NORMAL)
        req_state.add_transition(post_state.id, condition=cond_signal)

        self._ctx.current_state = post_state

    def _transform_await_mem_response(self, call_expr: Any,
                                       result_var: Optional[str] = None):
        """Transform ``result = await port.response()`` → WAIT_COND state.

        The FSM stays until ``{port}_rsp_valid`` is asserted.
        """
        port_name = self._extract_port_name(getattr(call_expr, 'func', None))
        cond_signal = f"{port_name}_rsp_valid"
        state_name = f"{port_name.upper()}_RSP"

        rsp_state = self._ctx.new_state(
            state_name, FSMStateKind.WAIT_COND, wait_condition=cond_signal
        )
        rsp_state.add_operation(
            FSMMemResponse(port_name=port_name, result_var=result_var or '')
        )
        rsp_state.add_transition(rsp_state.id, priority=1)

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(rsp_state.id)

        post_state = self._ctx.new_state(f"{state_name}_DONE", FSMStateKind.NORMAL)
        rsp_state.add_transition(post_state.id, condition=cond_signal)

        self._ctx.current_state = post_state

    # ------------------------------------------------------------------
    # Await handler: self.wait(Time.ns(N)) — idle for one cycle
    # ------------------------------------------------------------------

    def _transform_await_idle(self, call_expr: Any):
        """Transform ``await self.wait(Time.ns(N))`` → single NORMAL idle state."""
        idle_state = self._ctx.new_state("IDLE_WAIT", FSMStateKind.NORMAL)

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(idle_state.id)

        self._ctx.current_state = idle_state

    # ------------------------------------------------------------------
    # Await handler: self.regs.wait(regs, cond) — poll registers
    # ------------------------------------------------------------------

    def _transform_await_reg_wait(self, call_expr: Any, result_var=None):
        """Transform ``ctrls = await self.regs.wait(regs, cond)`` → WAIT_COND state.

        The RegFile.wait() call polls a list of registers (0-time, combinational)
        and stalls the FSM until ``cond(values)`` is true.  In RTL this maps to a
        single WAIT_COND state that reads all listed registers combinationally and
        checks the condition every cycle.

        The result variable(s) (if any) receive the register values so the
        caller can use them immediately — exactly as in the Python model.
        """
        args = getattr(call_expr, 'args', [])
        # args[0] = register list expression (ExprListComp or ExprList)
        # args[1] = condition expression   (ExprLambda or ExprGeneratorExp)
        regs_expr = args[0] if len(args) > 0 else None
        cond_expr = args[1] if len(args) > 1 else None

        # Build a string description of the wait condition for the FSM IR.
        # Downstream passes (code generators, optimisers) can inspect it.
        cond_str = self._summarise_reg_cond(cond_expr)

        wait_state = self._ctx.new_state(
            "IDLE_WAIT",
            FSMStateKind.WAIT_COND,
            wait_condition=('reg_cond', cond_str, regs_expr),
        )

        # Add a FSMRegRead for each register in the list so the code generator
        # knows which registers drive the combinational condition check.
        for reg_name, var_name in self._extract_reg_list(regs_expr, result_var):
            wait_state.add_operation(FSMRegRead(reg_name=reg_name, result_var=var_name))

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(wait_state.id)

        # Advance to a new NORMAL state when the condition is met.
        post_state = self._ctx.new_state(
            self._ctx.make_state_name("S"),
            FSMStateKind.NORMAL,
        )
        wait_state.add_transition(post_state.id, condition=('reg_cond', cond_str, regs_expr))
        wait_state.add_transition(wait_state.id, priority=1)  # self-loop while not met

        self._ctx.current_state = post_state

    def _summarise_reg_cond(self, cond_expr: Any) -> str:
        """Return a compact string description of a register wait condition.

        For the common ``lambda vals: any(v.en for v in vals)`` pattern this
        produces ``'any.en'``; unknown forms fall back to ``'cond'``.
        """
        if cond_expr is None:
            return 'cond'
        cond_type = type(cond_expr).__name__
        if cond_type == 'ExprLambda':
            body = getattr(cond_expr, 'body', None)
            if body is not None:
                body_type = type(body).__name__
                if body_type == 'ExprCall':
                    func = getattr(body, 'func', None)
                    func_name = getattr(func, 'name', '') or getattr(func, 'attr', '')
                    gen_args = getattr(body, 'args', [])
                    if func_name in ('any', 'all') and gen_args:
                        gen = gen_args[0]
                        elt = getattr(gen, 'elt', None)
                        attr = getattr(elt, 'attr', None) if elt is not None else None
                        if attr:
                            return f'{func_name}.{attr}'
        return 'cond'

    def _extract_reg_list(self, regs_expr: Any, result_var=None):
        """Yield (reg_name, result_var_name) pairs from a register list expression.

        Handles ``ExprListComp`` (``[self.regs.ch[i].ctrl for i in range(N)]``) and
        ``ExprList`` (``[ch.src, ch.dst, ...]``).
        """
        if regs_expr is None:
            return
        regs_type = type(regs_expr).__name__

        if regs_type == 'ExprList':
            elts = getattr(regs_expr, 'elts', [])
            result_names = result_var if isinstance(result_var, list) else [None] * len(elts)
            for i, elt in enumerate(elts):
                reg_name = getattr(elt, 'attr', None) or f'reg_{i}'
                var_name = result_names[i] if i < len(result_names) else ''
                yield reg_name, var_name or ''
        elif regs_type == 'ExprListComp':
            # e.g. [self.regs.ch[i].ctrl for i in range(N)]
            elt = getattr(regs_expr, 'elt', None)
            reg_name = getattr(elt, 'attr', None) or 'reg' if elt is not None else 'reg'
            var_name = result_var if isinstance(result_var, str) else ''
            yield reg_name, var_name

    # ------------------------------------------------------------------
    # Await handler: self.regs.read_all(regs) — bulk combinational read
    # ------------------------------------------------------------------

    def _transform_await_read_all(self, call_expr: Any, result_var=None):
        """Transform ``src, dst, length, ctrl = await self.regs.read_all([...])`` →
        single NORMAL state with one FSMRegRead per register.

        This is a 0-time (combinational) operation in RTL — all registers are
        read in the same clock cycle, so only a single FSM state is needed.
        """
        args = getattr(call_expr, 'args', [])
        regs_expr = args[0] if args else None

        read_state = self._ctx.new_state("REG_FETCH", FSMStateKind.NORMAL)

        for reg_name, var_name in self._extract_reg_list(regs_expr, result_var):
            read_state.add_operation(FSMRegRead(reg_name=reg_name, result_var=var_name))

        if self._ctx.current_state:
            self._ctx.current_state.add_transition(read_state.id)

        self._ctx.current_state = read_state

    # ------------------------------------------------------------------
    # Combinational: next(generator, default) → priority encoder
    # ------------------------------------------------------------------

    def _is_next_generator_call(self, expr: Any) -> bool:
        """Return True if *expr* is ``next(<generator-exp>, <default>)``."""
        if type(expr).__name__ != 'ExprCall':
            return False
        func = getattr(expr, 'func', None)
        if func is None or getattr(func, 'name', None) != 'next':
            return False
        args = getattr(expr, 'args', [])
        if not args:
            return False
        return type(args[0]).__name__ == 'ExprGeneratorExp'

    def _transform_priority_encode(self, target_name: str, call_expr: Any):
        """Emit a combinational priority-encoder assignment for ``next(gen, default)``.

        ``next((i for i, v in enumerate(seq) if pred(v)), default)`` describes a
        priority encoder: scan ``seq`` and return the first index where ``pred``
        holds, else ``default``.  In RTL this is a purely combinational always@(*)
        block — no extra FSM state is needed; the operation is added to the
        *current* state.
        """
        if self._ctx.current_state is None:
            return
        args = getattr(call_expr, 'args', [])
        gen_expr = args[0]   # ExprGeneratorExp
        default_expr = args[1] if len(args) > 1 else None

        # Record as a structured FSMAssign so code generators can lower it
        # to a priority-mux always-block.
        self._ctx.current_state.add_operation(
            FSMAssign(
                target=target_name,
                value=('priority_encode', gen_expr, default_expr),
                is_nonblocking=False,
            )
        )
    
    def _is_cycles_call(self, func: Any) -> bool:
        """Check if a function call is zdc.cycles() or zdc.tick()."""
        if hasattr(func, 'attr') and func.attr in ('cycles', 'tick'):
            return True
        if hasattr(func, 'name') and func.name in ('cycles', 'tick'):
            return True
        return False
    
    def _transform_await_cycles(self, call_expr: Any):
        """Transform 'await zdc.cycles(N)'."""
        # Extract N from arguments
        args = getattr(call_expr, 'args', [])
        n_cycles = 1
        if args and hasattr(args[0], 'value'):
            n_cycles = args[0].value
        
        # Create a new state for after the wait
        next_state = self._ctx.new_state(
            self._ctx.make_state_name("S"),
            FSMStateKind.WAIT_CYCLES,
            wait_cycles=n_cycles
        )
        
        # Add transition from current state to next state
        if self._ctx.current_state:
            self._ctx.current_state.add_transition(next_state.id)
        
        self._ctx.current_state = next_state
    
    def _transform_await_condition(self, cond_expr: Any):
        """Transform 'await <condition>'."""
        # Create a wait state
        wait_state = self._ctx.new_state(
            self._ctx.make_state_name("WAIT"),
            FSMStateKind.WAIT_COND,
            wait_condition=cond_expr
        )
        
        # Transition from current to wait state
        if self._ctx.current_state:
            self._ctx.current_state.add_transition(wait_state.id)
        
        # Create next state for after condition is met
        next_state = self._ctx.new_state(
            self._ctx.make_state_name("S"),
            FSMStateKind.NORMAL
        )
        
        # Wait state transitions to next state when condition is true
        wait_state.add_transition(next_state.id, condition=cond_expr)
        # Wait state loops back to itself when condition is false (implicit)
        wait_state.add_transition(wait_state.id, priority=1)  # Lower priority = fallback
        
        self._ctx.current_state = next_state

    # ------------------------------------------------------------------
    # Private async method inlining
    # ------------------------------------------------------------------

    def _rewrite_expr(self, expr: Any, bindings: Dict[str, Any], prefix: str) -> Any:
        """Recursively rewrite an IR expression for method inlining.

        * ``ExprRefParam(name)``  → the bound argument expression from *bindings*
        * ``ExprRefLocal(name)``  → ``ExprRefLocal(name=prefix + name)``
        * All other nodes        → same type with rewritten children
        """
        if not _HAVE_IR or expr is None:
            return expr

        etype = type(expr).__name__

        if etype == 'ExprRefParam':
            return bindings.get(expr.name, expr)

        if etype == 'ExprRefLocal':
            import copy
            node = copy.copy(expr)
            node.name = prefix + expr.name
            return node

        if etype == 'ExprCall':
            import copy
            node = copy.copy(expr)
            node.func = self._rewrite_expr(expr.func, bindings, prefix)
            node.args = [self._rewrite_expr(a, bindings, prefix) for a in (expr.args or [])]
            return node

        if etype == 'ExprAttribute':
            import copy
            node = copy.copy(expr)
            node.value = self._rewrite_expr(expr.value, bindings, prefix)
            return node

        if etype == 'ExprAwait':
            import copy
            node = copy.copy(expr)
            node.value = self._rewrite_expr(expr.value, bindings, prefix)
            return node

        if etype == 'ExprBin':
            import copy
            node = copy.copy(expr)
            node.lhs = self._rewrite_expr(expr.lhs, bindings, prefix)
            node.rhs = self._rewrite_expr(expr.rhs, bindings, prefix)
            return node

        if etype == 'ExprCompare':
            import copy
            node = copy.copy(expr)
            node.left = self._rewrite_expr(expr.left, bindings, prefix)
            node.comparators = [self._rewrite_expr(c, bindings, prefix)
                                 for c in (expr.comparators or [])]
            return node

        if etype == 'ExprBool':
            import copy
            node = copy.copy(expr)
            node.values = [self._rewrite_expr(v, bindings, prefix)
                           for v in (expr.values or [])]
            return node

        if etype == 'ExprUnary':
            import copy
            node = copy.copy(expr)
            node.operand = self._rewrite_expr(expr.operand, bindings, prefix)
            return node

        if etype == 'ExprTuple':
            import copy
            node = copy.copy(expr)
            node.elts = [self._rewrite_expr(e, bindings, prefix)
                         for e in (expr.elts or [])]
            return node

        if etype == 'ExprSubscript':
            import copy
            node = copy.copy(expr)
            node.value = self._rewrite_expr(expr.value, bindings, prefix)
            node.slice = self._rewrite_expr(expr.slice, bindings, prefix)
            return node

        if etype in ('ExprSext', 'ExprZext'):
            import copy
            node = copy.copy(expr)
            node.value = self._rewrite_expr(expr.value, bindings, prefix)
            # bits is int (compile-time constant), not an Expr — no rewrite needed
            return node

        if etype in ('ExprCbit', 'ExprSigned'):
            import copy
            node = copy.copy(expr)
            node.value = self._rewrite_expr(expr.value, bindings, prefix)
            return node

        # ExprConstant, ExprRefField, TypeExprRefSelf, ExprRefUnresolved — unchanged
        return expr

    def _rewrite_stmt(self, stmt: Any, bindings: Dict[str, Any], prefix: str) -> Any:
        """Rewrite a single statement for method inlining."""
        if not _HAVE_IR or stmt is None:
            return stmt

        import copy
        stype = type(stmt).__name__

        if stype == 'StmtAssign':
            node = copy.copy(stmt)
            node.targets = [self._rewrite_expr(t, bindings, prefix) for t in (stmt.targets or [])]
            node.value = self._rewrite_expr(stmt.value, bindings, prefix)
            return node

        if stype == 'StmtAugAssign':
            node = copy.copy(stmt)
            node.target = self._rewrite_expr(stmt.target, bindings, prefix)
            node.value = self._rewrite_expr(stmt.value, bindings, prefix)
            return node

        if stype == 'StmtFor':
            node = copy.copy(stmt)
            node.target = self._rewrite_expr(stmt.target, bindings, prefix)
            node.iter = self._rewrite_expr(stmt.iter, bindings, prefix)
            node.body = [self._rewrite_stmt(s, bindings, prefix) for s in (stmt.body or [])]
            node.orelse = [self._rewrite_stmt(s, bindings, prefix) for s in (stmt.orelse or [])]
            return node

        if stype == 'StmtExpr':
            node = copy.copy(stmt)
            node.expr = self._rewrite_expr(stmt.expr, bindings, prefix)
            return node

        if stype == 'StmtReturn':
            node = copy.copy(stmt)
            node.value = self._rewrite_expr(stmt.value, bindings, prefix)
            return node

        if stype == 'StmtWhile':
            node = copy.copy(stmt)
            node.test = self._rewrite_expr(stmt.test, bindings, prefix)
            node.body = [self._rewrite_stmt(s, bindings, prefix) for s in (stmt.body or [])]
            return node

        if stype == 'StmtIf':
            node = copy.copy(stmt)
            node.test = self._rewrite_expr(stmt.test, bindings, prefix)
            node.body = [self._rewrite_stmt(s, bindings, prefix) for s in (stmt.body or [])]
            node.orelse = [self._rewrite_stmt(s, bindings, prefix) for s in (stmt.orelse or [])]
            return node

        if stype == 'StmtAnnAssign':
            node = copy.copy(stmt)
            node.target = self._rewrite_expr(stmt.target, bindings, prefix)
            node.value = self._rewrite_expr(stmt.value, bindings, prefix)
            return node

        return stmt

    def _transform_await_method_call(self, call_expr: Any, result_var: Optional[str]) -> None:
        """Inline an ``await self._helper(args)`` call into the FSM.

        Looks up the function by name in ``self._component_ir.functions``,
        builds a parameter→argument binding, prefixes all local variables with
        ``_{func_name}_`` to avoid name clashes with the outer scope, rewrites
        the body, then transforms it as if it were inlined at the call site.
        ``StmtReturn`` in the inlined body assigns to *result_var*.
        """
        func_name = getattr(call_expr.func, 'attr', None)
        if func_name is None:
            return

        # Locate the function IR in the component
        helper_fn = None
        for fn in getattr(self._component_ir, 'functions', []):
            if getattr(fn, 'name', None) == func_name:
                helper_fn = fn
                break
        if helper_fn is None:
            return

        # Build param→arg bindings: map each parameter name to the call argument expr
        param_names = [a.arg for a in getattr(helper_fn.args, 'args', [])]
        call_args = getattr(call_expr, 'args', [])
        bindings: Dict[str, Any] = {}
        for pname, arg_expr in zip(param_names, call_args):
            bindings[pname] = arg_expr

        # Local variable prefix keeps names unique across inlined helpers.
        prefix = f"{func_name}_"

        # Rewrite the function body with the bindings applied
        rewritten_body = [
            self._rewrite_stmt(s, bindings, prefix)
            for s in getattr(helper_fn, 'body', [])
        ]

        # Save and set the inline return target so StmtReturn knows where to put it
        saved_return_var = getattr(self._ctx, '_inline_result_var', None)
        self._ctx._inline_result_var = result_var

        # Transform the rewritten body — this creates FSM states for the helper
        self._transform_body(rewritten_body)

        # Restore context
        self._ctx._inline_result_var = saved_return_var

    def _transform_return(self, stmt: Any) -> None:
        """Transform a ``return`` statement inside an inlined helper.

        Assigns the return value to the variable recorded in
        ``self._ctx._inline_result_var`` (set by :meth:`_transform_await_method_call`).
        """
        result_var = getattr(self._ctx, '_inline_result_var', None)
        if result_var is None or self._ctx.current_state is None:
            return

        value = getattr(stmt, 'value', None)
        if value is not None:
            self._ctx.current_state.add_operation(
                FSMAssign(target=result_var, value=value, is_nonblocking=True)
            )

    def _transform_for(self, stmt: Any):
        """Transform a for loop into LOOP_CHK / LOOP_BODY FSM states.

        Supports ``for <var> in range(<bound>):`` where ``<bound>`` is either
        a constant or a local variable.  The loop variable is synthesized as a
        counter register.

        Generated states
        ----------------
        ``LOOP_<VAR>_CHK``   — WAIT_COND: check ``var < bound``; exit when false
        ``LOOP_<VAR>_BODY``  — NORMAL: first state of the loop body
        …                    — states created by body's await expressions
        ``LOOP_<VAR>_DONE``  — NORMAL: first state after the loop
        """
        target = getattr(stmt, 'target', None)
        iter_expr = getattr(stmt, 'iter', None)
        body = getattr(stmt, 'body', [])

        loop_var = getattr(target, 'name', 'loop_i') if target else 'loop_i'

        # Extract bound from range(bound)
        bound_expr = None
        if iter_expr is not None and type(iter_expr).__name__ == 'ExprCall':
            args = getattr(iter_expr, 'args', [])
            if args:
                bound_expr = args[0]

        # Initialise loop counter in current state
        if self._ctx.current_state is not None:
            self._ctx.current_state.add_operation(
                FSMAssign(target=loop_var, value=0, is_nonblocking=True)
            )

        # LOOP_CHK — stay while var < bound (WAIT_COND exits when ≥ bound)
        loop_chk = self._ctx.new_state(
            f"LOOP_{loop_var.upper()}_CHK",
            FSMStateKind.WAIT_COND,
            wait_condition=('lt', loop_var, bound_expr),
        )
        if self._ctx.current_state is not None:
            self._ctx.current_state.add_transition(loop_chk.id)
        self._ctx.current_state = loop_chk

        # LOOP_BODY — first state executed when condition is met
        loop_body = self._ctx.new_state(
            f"LOOP_{loop_var.upper()}_BODY",
            FSMStateKind.NORMAL,
        )
        # Conditional advance: var < bound → body
        loop_chk.add_transition(loop_body.id, condition=('lt', loop_var, bound_expr))

        self._ctx.current_state = loop_body

        # Transform body statements (may create more states via awaits)
        self._transform_body(body)

        # Increment counter and loop back from whatever state body ended in
        if self._ctx.current_state is not None:
            self._ctx.current_state.add_operation(
                FSMAssign(target=loop_var, value=(loop_var, '+', 1),
                          is_nonblocking=True)
            )
            self._ctx.current_state.add_transition(loop_chk.id)

        # LOOP_DONE — reached when var >= bound (unconditional else from chk)
        loop_done = self._ctx.new_state(
            f"LOOP_{loop_var.upper()}_DONE",
            FSMStateKind.NORMAL,
        )
        loop_chk.add_transition(loop_done.id)  # unconditional = else branch

        self._ctx.current_state = loop_done
    def _finalize(self):
        """Finalize the FSM after transformation."""
        module = self._ctx.module
        
        # Ensure state encoding is computed
        module._compute_state_encoding()
        
        # Add state register
        if module.states:
            module.add_register(
                name="state",
                width=module.state_width,
                reset_value=module.initial_state
            )
