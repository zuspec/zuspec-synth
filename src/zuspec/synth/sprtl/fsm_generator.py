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
FSM Generator: Generates FSM from scheduled operations.

This module converts a schedule (mapping of operations to time steps)
into an FSM representation suitable for HDL code generation.

Key concepts:
- Each time step becomes an FSM state
- Operations scheduled at the same time are executed in parallel
- Transitions connect sequential states
- Conditional branches create multiple transitions with conditions
"""

from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from .fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMTransition,
    FSMOperation, FSMAssign, FSMCond, FSMPort, FSMRegister
)
from .scheduler import (
    Schedule, DependencyGraph, ScheduledOperation, OperationType
)


class StateEncoding(Enum):
    """State encoding strategies."""
    BINARY = auto()      # log2(n) bits, sequential encoding
    ONEHOT = auto()      # n bits, one-hot encoding
    GRAY = auto()        # Gray code encoding


@dataclass
class FSMGeneratorConfig:
    """Configuration for FSM generation.
    
    Attributes:
        state_prefix: Prefix for state names (e.g., "S" -> S0, S1, ...)
        encoding: State encoding strategy
        optimize_single_cycle: Merge single-cycle paths
        generate_done_signal: Add 'done' output signal
        generate_start_signal: Expect 'start' input signal
    """
    state_prefix: str = "S"
    encoding: StateEncoding = StateEncoding.BINARY
    optimize_single_cycle: bool = True
    generate_done_signal: bool = False
    generate_start_signal: bool = False


@dataclass
class LiveRange:
    """Live range of a variable.
    
    Attributes:
        variable: Variable name
        start: First time step where variable is defined
        end: Last time step where variable is used
    """
    variable: str
    start: int
    end: int
    
    def overlaps(self, other: 'LiveRange') -> bool:
        """Check if this live range overlaps with another."""
        return not (self.end < other.start or other.end < self.start)


class FSMGenerator:
    """Generates FSM from scheduled operations.
    
    The generator creates an FSM where:
    - Each time step corresponds to one or more states
    - Operations scheduled at the same time execute in parallel
    - Sequential transitions link states
    - Control flow (branches, loops) creates conditional transitions
    """
    
    def __init__(self, config: Optional[FSMGeneratorConfig] = None):
        """Initialize the FSM generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config or FSMGeneratorConfig()
    
    def generate(self, schedule: Schedule, graph: DependencyGraph,
                 source_fsm: Optional[FSMModule] = None) -> FSMModule:
        """Generate FSM from schedule.
        
        Args:
            schedule: Schedule mapping operations to time steps
            graph: Dependency graph with operation details
            source_fsm: Optional source FSM for metadata (ports, name)
            
        Returns:
            Generated FSM module
        """
        # Create module with metadata from source if available
        if source_fsm:
            module = FSMModule(
                name=source_fsm.name,
                clock_signal=source_fsm.clock_signal,
                reset_signal=source_fsm.reset_signal,
                reset_active_low=source_fsm.reset_active_low
            )
            # Copy ports
            for port in source_fsm.ports:
                module.add_port(port.name, port.direction, port.width, port.reset_value)
        else:
            module = FSMModule(name="scheduled_fsm")
        
        # Group operations by time step
        time_to_ops = self._group_by_time(schedule, graph)
        
        # Create states for each time step
        states = self._create_states(time_to_ops, module)
        
        # Create transitions between states
        self._create_transitions(states, time_to_ops, graph, module)
        
        # Apply state encoding
        self._encode_states(module)
        
        return module
    
    def _group_by_time(self, schedule: Schedule, 
                       graph: DependencyGraph) -> Dict[int, List[ScheduledOperation]]:
        """Group operations by their scheduled time step."""
        time_to_ops: Dict[int, List[ScheduledOperation]] = {}
        
        for op_id, time in schedule.operation_times.items():
            if time not in time_to_ops:
                time_to_ops[time] = []
            time_to_ops[time].append(graph.operations[op_id])
        
        return time_to_ops
    
    def _create_states(self, time_to_ops: Dict[int, List[ScheduledOperation]],
                       module: FSMModule) -> Dict[int, FSMState]:
        """Create FSM states for each time step."""
        states: Dict[int, FSMState] = {}
        
        # Sort time steps to ensure deterministic ordering
        sorted_times = sorted(time_to_ops.keys())
        
        for time in sorted_times:
            ops = time_to_ops[time]
            state_name = f"{self.config.state_prefix}{time}"
            
            # Determine state kind
            kind = self._determine_state_kind(ops)
            
            # Create state
            state = module.add_state(state_name, kind)
            
            # Add operations to state
            for op in ops:
                if op.source_op:
                    state.add_operation(op.source_op)
            
            states[time] = state
        
        # Set initial state
        if sorted_times:
            module.initial_state = states[sorted_times[0]].id
        
        return states
    
    def _determine_state_kind(self, ops: List[ScheduledOperation]) -> FSMStateKind:
        """Determine the kind of state based on operations."""
        # Check if any operation requires waiting
        for op in ops:
            if op.source_op:
                # Check for wait conditions in source FSM
                if hasattr(op.source_op, 'wait_condition'):
                    return FSMStateKind.WAIT_COND
                if hasattr(op.source_op, 'wait_cycles'):
                    return FSMStateKind.WAIT_CYCLES
        return FSMStateKind.NORMAL
    
    def _create_transitions(self, states: Dict[int, FSMState],
                           time_to_ops: Dict[int, List[ScheduledOperation]],
                           graph: DependencyGraph,
                           module: FSMModule):
        """Create transitions between states."""
        sorted_times = sorted(states.keys())
        
        for i, time in enumerate(sorted_times):
            state = states[time]
            
            # Default: transition to next state
            if i + 1 < len(sorted_times):
                next_time = sorted_times[i + 1]
                next_state = states[next_time]
                state.add_transition(next_state.id)
            else:
                # Last state - loop back to initial or stay
                if module.initial_state is not None:
                    state.add_transition(module.initial_state)
    
    def _encode_states(self, module: FSMModule):
        """Apply state encoding to the FSM."""
        if not module.states:
            return
        
        n_states = len(module.states)
        
        if self.config.encoding == StateEncoding.BINARY:
            # Binary encoding: ceil(log2(n)) bits
            module.state_width = max(1, (n_states - 1).bit_length())
            for i, state in enumerate(module.states):
                module.state_encoding[state.id] = i
        
        elif self.config.encoding == StateEncoding.ONEHOT:
            # One-hot encoding: n bits
            module.state_width = n_states
            for i, state in enumerate(module.states):
                module.state_encoding[state.id] = 1 << i
        
        elif self.config.encoding == StateEncoding.GRAY:
            # Gray code encoding
            module.state_width = max(1, (n_states - 1).bit_length())
            for i, state in enumerate(module.states):
                module.state_encoding[state.id] = i ^ (i >> 1)


class RegisterAllocator:
    """Allocates physical registers for variables.
    
    Uses live range analysis to determine when variables are active
    and allocates physical registers to minimize register count
    while avoiding conflicts.
    """
    
    def __init__(self):
        self._live_ranges: Dict[str, LiveRange] = {}
    
    def analyze_live_ranges(self, schedule: Schedule, 
                           graph: DependencyGraph) -> Dict[str, LiveRange]:
        """Compute live ranges for all variables.
        
        A variable is live from when it's defined until its last use.
        
        Args:
            schedule: Operation schedule
            graph: Dependency graph
            
        Returns:
            Mapping from variable name to live range
        """
        # Track definition and last use times
        def_time: Dict[str, int] = {}
        last_use: Dict[str, int] = {}
        
        for op_id, time in schedule.operation_times.items():
            op = graph.operations[op_id]
            
            if op.source_op and isinstance(op.source_op, FSMAssign):
                # Track definition
                target = op.source_op.target
                if target not in def_time:
                    def_time[target] = time
                
                # Track uses (from value expression)
                uses = self._extract_uses(op.source_op)
                for use in uses:
                    last_use[use] = max(last_use.get(use, time), time)
        
        # Create live ranges
        live_ranges = {}
        for var in set(def_time.keys()) | set(last_use.keys()):
            start = def_time.get(var, 0)
            end = last_use.get(var, start)
            live_ranges[var] = LiveRange(var, start, end)
        
        self._live_ranges = live_ranges
        return live_ranges
    
    def _extract_uses(self, op: FSMAssign) -> Set[str]:
        """Extract variable uses from an assignment."""
        uses = set()
        
        value = op.value
        if isinstance(value, tuple) and len(value) == 3:
            # Augmented assignment: (target, op, value)
            target, _, _ = value
            if hasattr(target, 'name'):
                uses.add(target.name)
            elif hasattr(target, 'attr'):
                uses.add(target.attr)
        
        return uses
    
    def allocate(self, live_ranges: Optional[Dict[str, LiveRange]] = None) -> Dict[str, str]:
        """Allocate physical registers using left-edge algorithm.
        
        Args:
            live_ranges: Live ranges (uses cached if not provided)
            
        Returns:
            Mapping from variable name to allocated register name
        """
        if live_ranges is None:
            live_ranges = self._live_ranges
        
        allocation: Dict[str, str] = {}
        registers: List[List[LiveRange]] = []  # Each register has list of assigned ranges
        
        # Sort by start time (left-edge algorithm)
        sorted_vars = sorted(live_ranges.values(), key=lambda lr: lr.start)
        
        for lr in sorted_vars:
            allocated = False
            
            # Try to find existing register that doesn't conflict
            for reg_idx, assigned in enumerate(registers):
                conflict = False
                for existing in assigned:
                    if lr.overlaps(existing):
                        conflict = True
                        break
                
                if not conflict:
                    assigned.append(lr)
                    allocation[lr.variable] = f"r{reg_idx}"
                    allocated = True
                    break
            
            # Need new register
            if not allocated:
                reg_idx = len(registers)
                registers.append([lr])
                allocation[lr.variable] = f"r{reg_idx}"
        
        return allocation
    
    def get_register_count(self) -> int:
        """Get the number of registers allocated."""
        allocation = self.allocate()
        if not allocation:
            return 0
        return len(set(allocation.values()))


class ScheduleToFSMBuilder:
    """High-level builder that combines scheduling and FSM generation.
    
    This class provides a convenient interface for:
    1. Taking a dependency graph
    2. Scheduling with optional resource constraints
    3. Generating an FSM
    4. Allocating registers
    """
    
    def __init__(self, fsm_config: Optional[FSMGeneratorConfig] = None):
        self.fsm_config = fsm_config or FSMGeneratorConfig()
        self.generator = FSMGenerator(self.fsm_config)
        self.allocator = RegisterAllocator()
    
    def build(self, graph: DependencyGraph, schedule: Schedule,
              source_fsm: Optional[FSMModule] = None) -> FSMModule:
        """Build FSM from dependency graph and schedule.
        
        Args:
            graph: Dependency graph with operations
            schedule: Pre-computed schedule
            source_fsm: Optional source FSM for metadata
            
        Returns:
            Generated FSM module
        """
        # Generate FSM from schedule
        fsm = self.generator.generate(schedule, graph, source_fsm)
        
        # Analyze live ranges and allocate registers
        live_ranges = self.allocator.analyze_live_ranges(schedule, graph)
        allocation = self.allocator.allocate(live_ranges)
        
        # Add allocated registers to FSM
        for var, reg_name in allocation.items():
            if not any(r.name == reg_name for r in fsm.registers):
                fsm.add_register(reg_name, width=32)  # Default width
        
        return fsm
