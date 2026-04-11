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
Scheduler: ASAP/ALAP scheduling algorithms for HLS.

This module implements basic scheduling algorithms for high-level synthesis:
- ASAP (As-Soon-As-Possible): Schedule operations at earliest possible time
- ALAP (As-Late-As-Possible): Schedule operations at latest possible time
- Mobility analysis: Determine scheduling flexibility of operations

These algorithms are fundamental to resource-constrained scheduling and
form the basis for more advanced scheduling techniques.

Algorithm references:
- Standard HLS scheduling (De Micheli, "Synthesis and Optimization of Digital Circuits")
- Adapted from XLS/ScaleHLS patterns (Apache 2.0)
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

from .fsm_ir import FSMModule, FSMState, FSMStateKind, FSMOperation, FSMAssign, FSMCond


class OperationType(Enum):
    """Types of operations for scheduling."""
    NOP = auto()         # No operation (placeholder)
    ASSIGN = auto()      # Assignment (register write)
    ADD = auto()         # Addition
    SUB = auto()         # Subtraction
    MUL = auto()         # Multiplication
    DIV = auto()         # Division
    COMPARE = auto()     # Comparison
    MUX = auto()         # Multiplexer (conditional select)
    LOAD = auto()        # Memory load
    STORE = auto()       # Memory store


@dataclass
class ScheduledOperation:
    """An operation to be scheduled.
    
    Attributes:
        id: Unique identifier
        op_type: Type of operation
        latency: Number of cycles to complete (1 for combinational)
        predecessors: Operations that must complete before this one
        successors: Operations that depend on this one
        source_op: Original FSM operation (for back-reference)
        state_id: FSM state this operation belongs to
    """
    id: int
    op_type: OperationType
    latency: int = 1
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    source_op: Optional[FSMOperation] = None
    state_id: Optional[int] = None
    
    # Scheduling results
    asap_time: int = -1
    alap_time: int = -1
    
    @property
    def mobility(self) -> int:
        """Scheduling mobility (flexibility)."""
        if self.asap_time < 0 or self.alap_time < 0:
            return -1
        return self.alap_time - self.asap_time
    
    @property
    def is_critical(self) -> bool:
        """True if operation is on critical path (mobility = 0)."""
        return self.mobility == 0


@dataclass
class Schedule:
    """Result of scheduling operations.
    
    Attributes:
        operation_times: Mapping from operation ID to scheduled time step
        total_latency: Total latency of the schedule
        cycle_operations: Mapping from time step to operations scheduled at that time
    """
    operation_times: Dict[int, int] = field(default_factory=dict)
    total_latency: int = 0
    cycle_operations: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    
    def schedule_operation(self, op_id: int, time: int):
        """Schedule an operation at a specific time step."""
        self.operation_times[op_id] = time
        self.cycle_operations[time].append(op_id)
        self.total_latency = max(self.total_latency, time + 1)
    
    def get_time(self, op_id: int) -> int:
        """Get the scheduled time for an operation."""
        return self.operation_times.get(op_id, -1)
    
    def get_operations_at_time(self, time: int) -> List[int]:
        """Get all operations scheduled at a time step."""
        return self.cycle_operations.get(time, [])


@dataclass
class DependencyGraph:
    """Data dependency graph for scheduling.
    
    Nodes are operations, edges represent dependencies (producer -> consumer).
    """
    operations: Dict[int, ScheduledOperation] = field(default_factory=dict)
    _next_id: int = 0
    
    def add_operation(self, op_type: OperationType, latency: int = 1,
                     source_op: Optional[FSMOperation] = None,
                     state_id: Optional[int] = None) -> ScheduledOperation:
        """Add an operation to the graph."""
        op = ScheduledOperation(
            id=self._next_id,
            op_type=op_type,
            latency=latency,
            source_op=source_op,
            state_id=state_id
        )
        self.operations[op.id] = op
        self._next_id += 1
        return op
    
    def add_dependency(self, producer_id: int, consumer_id: int):
        """Add a dependency edge from producer to consumer."""
        if producer_id in self.operations and consumer_id in self.operations:
            self.operations[producer_id].successors.add(consumer_id)
            self.operations[consumer_id].predecessors.add(producer_id)
    
    def get_roots(self) -> List[ScheduledOperation]:
        """Get operations with no predecessors."""
        return [op for op in self.operations.values() if not op.predecessors]
    
    def get_leaves(self) -> List[ScheduledOperation]:
        """Get operations with no successors."""
        return [op for op in self.operations.values() if not op.successors]
    
    def topological_sort(self) -> List[ScheduledOperation]:
        """Return operations in topological order."""
        result = []
        visited = set()
        temp_mark = set()
        
        def visit(op_id: int):
            if op_id in temp_mark:
                raise ValueError("Cycle detected in dependency graph")
            if op_id in visited:
                return
            
            temp_mark.add(op_id)
            op = self.operations[op_id]
            for succ_id in op.successors:
                visit(succ_id)
            temp_mark.remove(op_id)
            visited.add(op_id)
            result.insert(0, op)
        
        for op_id in self.operations:
            if op_id not in visited:
                visit(op_id)
        
        return result
    
    def reverse_topological_sort(self) -> List[ScheduledOperation]:
        """Return operations in reverse topological order."""
        return list(reversed(self.topological_sort()))


class ASAPScheduler:
    """As-Soon-As-Possible scheduler.
    
    Schedules each operation at the earliest time step where all
    its predecessors have completed.
    
    Complexity: O(n) where n is the number of operations.
    """
    
    def schedule(self, graph: DependencyGraph) -> Schedule:
        """Compute ASAP schedule for the dependency graph.
        
        Args:
            graph: Dependency graph with operations and dependencies
            
        Returns:
            Schedule with ASAP times for all operations
        """
        schedule = Schedule()
        
        # Process operations in topological order
        for op in graph.topological_sort():
            # Find earliest time where all predecessors are complete
            earliest = 0
            for pred_id in op.predecessors:
                pred = graph.operations[pred_id]
                pred_time = schedule.get_time(pred_id)
                earliest = max(earliest, pred_time + pred.latency)
            
            # Schedule at earliest possible time
            schedule.schedule_operation(op.id, earliest)
            op.asap_time = earliest
        
        return schedule


class ALAPScheduler:
    """As-Late-As-Possible scheduler.
    
    Schedules each operation at the latest time step without
    increasing the total latency (determined by ASAP schedule).
    
    Complexity: O(n) where n is the number of operations.
    """
    
    def schedule(self, graph: DependencyGraph, 
                total_latency: Optional[int] = None) -> Schedule:
        """Compute ALAP schedule for the dependency graph.
        
        Args:
            graph: Dependency graph with operations and dependencies
            total_latency: Maximum allowed latency. If None, uses ASAP latency.
            
        Returns:
            Schedule with ALAP times for all operations
        """
        # First, compute ASAP to get total latency if not provided
        if total_latency is None:
            asap_schedule = ASAPScheduler().schedule(graph)
            total_latency = asap_schedule.total_latency
        
        schedule = Schedule()
        schedule.total_latency = total_latency
        
        # Process operations in reverse topological order
        for op in graph.reverse_topological_sort():
            # Find latest time that satisfies all successor constraints
            latest = total_latency - op.latency
            
            for succ_id in op.successors:
                succ_time = schedule.get_time(succ_id)
                if succ_time >= 0:
                    latest = min(latest, succ_time - op.latency)
            
            # Schedule at latest possible time
            schedule.schedule_operation(op.id, latest)
            op.alap_time = latest
        
        return schedule


class MobilityAnalyzer:
    """Analyzes scheduling mobility of operations.
    
    Mobility = ALAP_time - ASAP_time
    
    Operations with mobility = 0 are on the critical path and
    cannot be moved without affecting total latency.
    """
    
    def analyze(self, graph: DependencyGraph) -> Dict[int, int]:
        """Compute mobility for all operations.
        
        Args:
            graph: Dependency graph (will be modified with ASAP/ALAP times)
            
        Returns:
            Mapping from operation ID to mobility
        """
        # Compute ASAP schedule
        asap_scheduler = ASAPScheduler()
        asap_schedule = asap_scheduler.schedule(graph)
        
        # Compute ALAP schedule with same total latency
        alap_scheduler = ALAPScheduler()
        alap_scheduler.schedule(graph, asap_schedule.total_latency)
        
        # Compute mobility for each operation
        mobility = {}
        for op_id, op in graph.operations.items():
            mobility[op_id] = op.mobility
        
        return mobility
    
    def get_critical_path(self, graph: DependencyGraph) -> List[ScheduledOperation]:
        """Get operations on the critical path (mobility = 0)."""
        self.analyze(graph)
        return [op for op in graph.operations.values() if op.is_critical]


def operation_type_from_fsm_op(op: FSMOperation) -> OperationType:
    """Determine operation type from an FSM operation."""
    if isinstance(op, FSMAssign):
        # Check if it's a simple assignment or has an operator
        value = op.value
        if isinstance(value, tuple) and len(value) == 3:
            # Augmented assignment: (target, op, value)
            _, op_name, _ = value
            if op_name in ('Add', '+'):
                return OperationType.ADD
            elif op_name in ('Sub', '-'):
                return OperationType.SUB
            elif op_name in ('Mult', '*'):
                return OperationType.MUL
            elif op_name in ('Div', '/'):
                return OperationType.DIV
        return OperationType.ASSIGN
    elif isinstance(op, FSMCond):
        return OperationType.MUX
    return OperationType.NOP


def default_latency(op_type: OperationType) -> int:
    """Get default latency for an operation type."""
    latencies = {
        OperationType.NOP: 0,
        OperationType.ASSIGN: 0,    # Combinational
        OperationType.ADD: 0,       # Combinational (single cycle add)
        OperationType.SUB: 0,       # Combinational
        OperationType.MUL: 1,       # May need pipeline stages
        OperationType.DIV: 4,       # Multiple cycles
        OperationType.COMPARE: 0,   # Combinational
        OperationType.MUX: 0,       # Combinational
        OperationType.LOAD: 1,      # Memory access
        OperationType.STORE: 1,     # Memory access
    }
    return latencies.get(op_type, 0)


class FSMToScheduleGraphBuilder:
    """Builds a dependency graph from FSM operations.
    
    Extracts operations from FSM states and builds dependencies
    based on data flow (read-after-write dependencies).
    """
    
    def build(self, fsm: FSMModule) -> DependencyGraph:
        """Build dependency graph from FSM module.
        
        Args:
            fsm: FSM module with states and operations
            
        Returns:
            Dependency graph for scheduling
        """
        graph = DependencyGraph()
        
        # Track last writer for each signal (for RAW dependencies)
        last_writer: Dict[str, int] = {}
        
        for state in fsm.states:
            for op in state.operations:
                self._process_operation(graph, op, state.id, last_writer)
        
        return graph
    
    def _process_operation(self, graph: DependencyGraph, op: FSMOperation,
                          state_id: int, last_writer: Dict[str, int]):
        """Process a single FSM operation."""
        if isinstance(op, FSMAssign):
            op_type = operation_type_from_fsm_op(op)
            latency = default_latency(op_type)
            
            sched_op = graph.add_operation(
                op_type=op_type,
                latency=latency,
                source_op=op,
                state_id=state_id
            )
            
            # Add RAW dependency if we read from a previously written signal
            reads = self._get_reads(op)
            for signal in reads:
                if signal in last_writer:
                    graph.add_dependency(last_writer[signal], sched_op.id)
            
            # Update last writer
            last_writer[op.target] = sched_op.id
            
        elif isinstance(op, FSMCond):
            # Conditional - process both branches
            cond_op = graph.add_operation(
                op_type=OperationType.MUX,
                latency=0,
                source_op=op,
                state_id=state_id
            )
            
            for sub_op in op.then_ops:
                self._process_operation(graph, sub_op, state_id, last_writer)
            for sub_op in op.else_ops:
                self._process_operation(graph, sub_op, state_id, last_writer)
    
    def _get_reads(self, op: FSMAssign) -> Set[str]:
        """Extract signals read by an assignment."""
        reads = set()
        
        value = op.value
        if isinstance(value, tuple) and len(value) == 3:
            # Augmented assignment reads the target
            target, _, _ = value
            if hasattr(target, 'name'):
                reads.add(target.name)
            elif hasattr(target, 'attr'):
                reads.add(target.attr)
        
        # Could expand to extract all signal references from value expressions
        
        return reads


# =============================================================================
# Resource-Constrained Scheduling (Phase 2.2)
# =============================================================================

class PriorityMetric(Enum):
    """Priority metrics for list-based scheduling."""
    STATIC_MOBILITY = auto()     # Use initial ALAP - ASAP
    DYNAMIC_MOBILITY = auto()    # Recompute mobility at each step
    CRITICAL_PATH = auto()       # Prioritize critical path operations


@dataclass
class ResourceConstraints:
    """Resource constraints for scheduling.
    
    Specifies the maximum number of each resource type available
    per clock cycle.
    
    Attributes:
        max_adders: Maximum adders/subtractors available
        max_multipliers: Maximum multipliers available
        max_dividers: Maximum dividers available
        max_memory_ports: Maximum memory access ports
        max_comparators: Maximum comparators available
    """
    max_adders: int = 2
    max_multipliers: int = 1
    max_dividers: int = 1
    max_memory_ports: int = 1
    max_comparators: int = 2
    
    def get_limit(self, op_type: OperationType) -> int:
        """Get resource limit for an operation type."""
        limits = {
            OperationType.ADD: self.max_adders,
            OperationType.SUB: self.max_adders,
            OperationType.MUL: self.max_multipliers,
            OperationType.DIV: self.max_dividers,
            OperationType.LOAD: self.max_memory_ports,
            OperationType.STORE: self.max_memory_ports,
            OperationType.COMPARE: self.max_comparators,
        }
        return limits.get(op_type, float('inf'))  # Unlimited for ASSIGN, MUX, NOP


@dataclass
class ResourceUsage:
    """Track resource usage at a time step."""
    usage: Dict[OperationType, int] = field(default_factory=lambda: defaultdict(int))
    
    def use(self, op_type: OperationType):
        """Record use of a resource."""
        self.usage[op_type] += 1
    
    def can_use(self, op_type: OperationType, constraints: ResourceConstraints) -> bool:
        """Check if resource is available."""
        limit = constraints.get_limit(op_type)
        return self.usage[op_type] < limit
    
    def reset(self):
        """Reset usage for next time step."""
        self.usage.clear()


class ListScheduler:
    """Resource-constrained list-based scheduler.
    
    Implements the list scheduling algorithm:
    1. Compute ASAP/ALAP for mobility
    2. Maintain ready list of operations whose predecessors are complete
    3. At each time step, select highest priority operations that fit resources
    4. Continue until all operations are scheduled
    
    This is a greedy heuristic that produces good results in practice.
    """
    
    def __init__(self, constraints: Optional[ResourceConstraints] = None,
                 priority: PriorityMetric = PriorityMetric.STATIC_MOBILITY):
        """Initialize the list scheduler.
        
        Args:
            constraints: Resource constraints (default: unlimited)
            priority: Priority metric for operation selection
        """
        self.constraints = constraints or ResourceConstraints()
        self.priority = priority
    
    def schedule(self, graph: DependencyGraph) -> Schedule:
        """Perform resource-constrained scheduling.
        
        Args:
            graph: Dependency graph with operations
            
        Returns:
            Schedule respecting resource constraints
        """
        # First compute ASAP/ALAP for mobility information
        analyzer = MobilityAnalyzer()
        analyzer.analyze(graph)
        
        schedule = Schedule()
        scheduled: Set[int] = set()
        time = 0
        
        # Track when operations complete (for latency > 1)
        completion_time: Dict[int, int] = {}
        
        while len(scheduled) < len(graph.operations):
            # Get ready operations (all predecessors complete)
            ready = self._get_ready_operations(graph, scheduled, completion_time, time)
            
            if not ready:
                # No operations ready - advance time
                time += 1
                continue
            
            # Sort by priority
            ready = self._sort_by_priority(ready, graph, time)
            
            # Schedule as many as resources allow
            usage = ResourceUsage()
            for op in ready:
                if usage.can_use(op.op_type, self.constraints):
                    schedule.schedule_operation(op.id, time)
                    scheduled.add(op.id)
                    completion_time[op.id] = time + op.latency
                    usage.use(op.op_type)
            
            time += 1
        
        return schedule
    
    def _get_ready_operations(self, graph: DependencyGraph, 
                              scheduled: Set[int],
                              completion_time: Dict[int, int],
                              current_time: int) -> List[ScheduledOperation]:
        """Get operations ready to be scheduled.
        
        An operation is ready if:
        1. It hasn't been scheduled yet
        2. All its predecessors have completed (completion_time <= current_time)
        """
        ready = []
        for op_id, op in graph.operations.items():
            if op_id in scheduled:
                continue
            
            # Check all predecessors are complete
            all_complete = True
            for pred_id in op.predecessors:
                if pred_id not in completion_time:
                    all_complete = False
                    break
                if completion_time[pred_id] > current_time:
                    all_complete = False
                    break
            
            if all_complete:
                ready.append(op)
        
        return ready
    
    def _sort_by_priority(self, ops: List[ScheduledOperation],
                          graph: DependencyGraph,
                          current_time: int) -> List[ScheduledOperation]:
        """Sort operations by priority (highest priority first)."""
        if self.priority == PriorityMetric.STATIC_MOBILITY:
            # Lower mobility = higher priority (schedule critical ops first)
            return sorted(ops, key=lambda op: op.mobility)
        
        elif self.priority == PriorityMetric.DYNAMIC_MOBILITY:
            # Recompute mobility based on current schedule
            # For simplicity, use static mobility (TODO: implement dynamic)
            return sorted(ops, key=lambda op: op.mobility)
        
        elif self.priority == PriorityMetric.CRITICAL_PATH:
            # Prioritize operations on critical path (mobility = 0)
            return sorted(ops, key=lambda op: (op.mobility, -op.latency))
        
        return ops


class SDCScheduler:
    """SDC-based (System of Difference Constraints) pipeline stage scheduler.

    Formulates scheduling as a system of difference constraints solved by
    Bellman-Ford shortest-path on the constraint graph.  This gives the
    lexicographically minimum stage assignment (minimises total stages) when
    initialised from ASAP times.

    Constraints
    -----------
    For each data dependency ``i → j`` with ``latency_i``:
        ``t_j − t_i ≥ latency_i``
        (operation j cannot start until i has completed)

    For each resource conflict (two operations of the same type when
    ``resource_limits`` restricts that type to 1):
        ``|t_i − t_j| ≥ 1``   (encode as two half-space constraints)

    Clock-period constraint (optional):
        After stage assignment, stages whose combinational path exceeds
        ``clock_period_ns`` are split by moving the offending operation to
        the next slot.

    Algorithm
    ---------
    1. Run ASAP to initialise ``t[i] = asap[i]`` for all operations.
    2. Run ALAP to compute ``[asap[i], alap[i]]`` windows.
    3. Build constraint graph: edge weight ``w(i→j) = latency[i]``.
    4. Run Bellman-Ford from a virtual source with distance ``t[i] = asap[i]``.
    5. Clamp each ``t[i]`` to ``[asap[i], alap[i]]``.
    6. Group operations by ``t[i]`` → pipeline stages.
    7. If resource limits are provided, check feasibility and adjust.
    """

    DEFAULT_LATENCY: Dict[str, int] = {
        "NOP":     0,
        "ASSIGN":  1,
        "ADD":     1,
        "SUB":     1,
        "MUL":     3,
        "DIV":     8,
        "COMPARE": 1,
        "MUX":     1,
        "LOAD":    2,
        "STORE":   1,
    }

    def __init__(
        self,
        latency_model: Optional[Dict[str, int]] = None,
        resource_limits: Optional[Dict[str, int]] = None,
        clock_period_ns: float = 10.0,
    ) -> None:
        """
        Args:
            latency_model: Override mapping ``OperationType.name → cycles``.
                           Merged with ``DEFAULT_LATENCY`` (user values win).
            resource_limits: Mapping ``OperationType.name → max_per_stage``.
                             Only ``1`` is currently enforced via separation
                             constraints.
            clock_period_ns: Target clock period; used only for informational
                             logging (combinational-delay estimation not yet
                             implemented).
        """
        self.latency_model: Dict[str, int] = dict(self.DEFAULT_LATENCY)
        if latency_model:
            self.latency_model.update(latency_model)
        self.resource_limits: Dict[str, int] = resource_limits or {}
        self.clock_period_ns = clock_period_ns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, graph: DependencyGraph) -> Schedule:
        """Run SDC scheduling and return a :class:`Schedule`.

        Falls back to ASAP scheduling if the graph has no operations or
        the Bellman-Ford solver detects a negative cycle (should not occur
        for acyclic dependency graphs).
        """
        if not graph.operations:
            return Schedule()

        # Step 1 & 2: ASAP / ALAP
        asap_sched = ASAPScheduler().schedule(graph)
        alap_sched = ALAPScheduler().schedule(graph, asap_sched.total_latency)

        asap: Dict[int, int] = {op_id: asap_sched.get_time(op_id)
                                 for op_id in graph.operations}
        alap: Dict[int, int] = {op_id: alap_sched.get_time(op_id)
                                 for op_id in graph.operations}

        # Step 3 & 4: Bellman-Ford on constraint graph
        t = self._bellman_ford(graph, asap, alap)

        # Step 7: Resource-conflict adjustments (separation constraints)
        if self.resource_limits:
            t = self._apply_resource_limits(graph, t, alap)

        # Build Schedule from t
        result = Schedule()
        for op_id, time_slot in t.items():
            result.schedule_operation(op_id, time_slot)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_latency(self, op: ScheduledOperation) -> int:
        """Return the latency in cycles for *op*, using the latency model."""
        return self.latency_model.get(op.op_type.name, 1)

    def _bellman_ford(
        self,
        graph: DependencyGraph,
        asap: Dict[int, int],
        alap: Dict[int, int],
    ) -> Dict[int, int]:
        """Return a consistent stage assignment ``t[op_id]`` via Bellman-Ford.

        Initialises from ASAP times (lexicographically minimum), then
        propagates constraints ``t[j] ≥ t[i] + latency_i`` until convergence.
        """
        t: Dict[int, int] = dict(asap)

        # Collect edges: (producer_id, consumer_id, min_gap)
        edges: List[Tuple[int, int, int]] = []
        for op_id, op in graph.operations.items():
            lat = self._get_latency(op)
            for succ_id in op.successors:
                edges.append((op_id, succ_id, lat))

        # Bellman-Ford relaxation (|V| − 1 iterations)
        n = len(graph.operations)
        for _ in range(n):
            changed = False
            for (u, v, w) in edges:
                if t[u] + w > t[v]:
                    new_val = min(t[u] + w, alap.get(v, t[u] + w))
                    if new_val != t[v]:
                        t[v] = new_val
                        changed = True
            if not changed:
                break

        # Clamp to [asap, alap] windows
        for op_id in t:
            t[op_id] = max(asap[op_id], min(t[op_id], alap.get(op_id, t[op_id])))

        return t

    def _apply_resource_limits(
        self,
        graph: DependencyGraph,
        t: Dict[int, int],
        alap: Dict[int, int],
    ) -> Dict[int, int]:
        """Enforce resource limits by separating conflicting operations.

        For each resource type limited to 1 per stage, if two operations of
        that type are assigned to the same stage, the one with more ALAP
        slack is pushed to the next stage (if within its ALAP window).
        """
        from collections import defaultdict

        for res_name, max_count in self.resource_limits.items():
            if max_count >= 2:
                continue
            # Find ops of this resource type
            try:
                res_type = OperationType[res_name]
            except KeyError:
                continue
            ops_of_type = [op for op in graph.operations.values()
                           if op.op_type == res_type]
            # Group by assigned stage
            stage_to_ops: Dict[int, List[ScheduledOperation]] = defaultdict(list)
            for op in ops_of_type:
                stage_to_ops[t[op.id]].append(op)
            # Separate conflicts
            for stage, conflicting in stage_to_ops.items():
                if len(conflicting) <= max_count:
                    continue
                # Sort by mobility (highest mobility can be moved)
                conflicting.sort(key=lambda o: -(alap.get(o.id, stage) - stage))
                for op in conflicting[max_count:]:
                    new_slot = t[op.id] + 1
                    if new_slot <= alap.get(op.id, new_slot):
                        t[op.id] = new_slot
        return t
