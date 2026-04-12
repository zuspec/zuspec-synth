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
    FSMOperation, FSMAssign, FSMCond, FSMPort, FSMRegister
)


@dataclass
class TransformContext:
    """Context for the transformation pass."""
    module: FSMModule
    current_state: Optional[FSMState] = None
    state_counter: int = 0
    reset_operations: List[FSMOperation] = field(default_factory=list)
    in_reset_section: bool = True  # True until we hit 'while True'
    
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
    
    def transform(self, component_ir: Any, process_ir: Any) -> FSMModule:
        """Transform a sync process to an FSM module.
        
        Args:
            component_ir: The DataTypeComponent IR node
            process_ir: The Function IR node for the sync process
            
        Returns:
            FSMModule representing the transformed FSM
        """
        # Extract module name from component
        module_name = getattr(component_ir, 'name', 'fsm') or 'fsm'
        
        # Create the FSM module
        module = FSMModule(name=module_name)
        
        # Set up transformation context
        self._ctx = TransformContext(module=module)
        
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
        # Add more statement types as needed
    
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
    
    def _transform_if(self, stmt: Any):
        """Transform an if statement."""
        test = getattr(stmt, 'test', None)
        then_body = getattr(stmt, 'body', [])
        else_body = getattr(stmt, 'orelse', [])
        
        if self._ctx.in_reset_section:
            # In reset section, if statements become conditional reset logic
            # For now, skip - reset values come from port declarations
            return
        
        if not self._ctx.current_state:
            return
        
        # Create conditional operation in current state
        then_ops = []
        else_ops = []
        
        # Transform then branch
        for s in then_body:
            ops = self._stmt_to_operations(s)
            then_ops.extend(ops)
        
        # Transform else branch
        for s in else_body:
            ops = self._stmt_to_operations(s)
            else_ops.extend(ops)
        
        if then_ops or else_ops:
            cond_op = FSMCond(
                condition=test,
                then_ops=then_ops,
                else_ops=else_ops
            )
            self._ctx.current_state.add_operation(cond_op)
    
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
        """Convert an expression to a signal name."""
        if hasattr(expr, 'name'):
            return expr.name
        if hasattr(expr, 'attr'):
            return expr.attr
        if hasattr(expr, 'index'):
            # ExprRefField - would need to look up by index
            return f"field_{expr.index}"
        return "unknown"
    
    def _transform_assign(self, stmt: Any):
        """Transform an assignment statement."""
        if self._ctx.in_reset_section:
            # Assignment in reset section - extract reset value
            # Reset values should come from port declarations instead
            return
        
        if not self._ctx.current_state:
            return
        
        targets = getattr(stmt, 'targets', [])
        value = getattr(stmt, 'value', None)
        
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
        """Transform an expression statement (e.g., await)."""
        expr = getattr(stmt, 'expr', None)
        if not expr:
            return
        
        expr_type = type(expr).__name__
        
        if expr_type == 'ExprAwait':
            self._transform_await(expr)
    
    def _transform_await(self, expr: Any):
        """Transform an await expression.
        
        This creates a new FSM state boundary.
        """
        if self._ctx.in_reset_section:
            return
        
        await_value = getattr(expr, 'value', None)
        if not await_value:
            return
        
        # Determine what kind of await this is
        await_type = type(await_value).__name__
        
        if await_type == 'ExprCall':
            # Check if it's zdc.cycles()
            func = getattr(await_value, 'func', None)
            if func and self._is_cycles_call(func):
                self._transform_await_cycles(await_value)
                return
        
        if await_type == 'ExprCompare':
            # await <expr> == <value> - wait for condition
            self._transform_await_condition(await_value)
            return
        
        # Generic await - treat as wait for condition
        self._transform_await_condition(await_value)
    
    def _is_cycles_call(self, func: Any) -> bool:
        """Check if a function call is zdc.cycles()."""
        if hasattr(func, 'attr') and func.attr == 'cycles':
            return True
        if hasattr(func, 'name') and func.name == 'cycles':
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
    
    def _transform_for(self, stmt: Any):
        """Transform a for loop.
        
        Note: For synthesis, for loops should typically be unrolled.
        This handles the case where unrolling hasn't been applied.
        """
        # For now, treat like a while loop
        # Full implementation would check for @Unroll directive
        target = getattr(stmt, 'target', None)
        iter_expr = getattr(stmt, 'iter', None)
        body = getattr(stmt, 'body', [])
        
        # TODO: Implement proper for loop handling with bounds analysis
        # For now, just transform the body
        self._transform_body(body)
    
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
