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
"""Tests for ASAP/ALAP scheduling algorithms."""

import pytest
import sys
import os

# Ensure paths are set correctly for development
_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

# Insert at front but after '' if present
if '' in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

from zuspec.synth.sprtl.scheduler import (
    OperationType, ScheduledOperation, Schedule, DependencyGraph,
    ASAPScheduler, ALAPScheduler, MobilityAnalyzer,
    FSMToScheduleGraphBuilder, default_latency,
    # Resource-constrained scheduling
    PriorityMetric, ResourceConstraints, ResourceUsage, ListScheduler, SDCScheduler
)
from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMState, FSMStateKind, FSMAssign


class TestDependencyGraph:
    """Tests for dependency graph construction."""
    
    def test_empty_graph(self):
        """Test empty dependency graph."""
        graph = DependencyGraph()
        assert len(graph.operations) == 0
        assert graph.get_roots() == []
        assert graph.get_leaves() == []
    
    def test_add_operation(self):
        """Test adding operations to graph."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=0)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        
        assert len(graph.operations) == 2
        assert op1.id == 0
        assert op2.id == 1
        assert op1.op_type == OperationType.ADD
        assert op2.latency == 1
    
    def test_add_dependency(self):
        """Test adding dependencies between operations."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD)
        op2 = graph.add_operation(OperationType.ADD)
        
        graph.add_dependency(op1.id, op2.id)
        
        assert op2.id in op1.successors
        assert op1.id in op2.predecessors
    
    def test_roots_and_leaves(self):
        """Test finding roots and leaves."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD)  # Root
        op2 = graph.add_operation(OperationType.ADD)
        op3 = graph.add_operation(OperationType.ADD)  # Leaf
        
        graph.add_dependency(op1.id, op2.id)
        graph.add_dependency(op2.id, op3.id)
        
        roots = graph.get_roots()
        leaves = graph.get_leaves()
        
        assert len(roots) == 1
        assert roots[0].id == op1.id
        assert len(leaves) == 1
        assert leaves[0].id == op3.id
    
    def test_topological_sort(self):
        """Test topological sorting of operations."""
        graph = DependencyGraph()
        op_a = graph.add_operation(OperationType.ADD)
        op_b = graph.add_operation(OperationType.ADD)
        op_c = graph.add_operation(OperationType.ADD)
        
        # A -> B -> C
        graph.add_dependency(op_a.id, op_b.id)
        graph.add_dependency(op_b.id, op_c.id)
        
        sorted_ops = graph.topological_sort()
        ids = [op.id for op in sorted_ops]
        
        assert ids.index(op_a.id) < ids.index(op_b.id)
        assert ids.index(op_b.id) < ids.index(op_c.id)
    
    def test_topological_sort_diamond(self):
        """Test topological sort with diamond dependency pattern."""
        graph = DependencyGraph()
        op_a = graph.add_operation(OperationType.ADD)  # A
        op_b = graph.add_operation(OperationType.ADD)  # B
        op_c = graph.add_operation(OperationType.ADD)  # C
        op_d = graph.add_operation(OperationType.ADD)  # D
        
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        graph.add_dependency(op_a.id, op_b.id)
        graph.add_dependency(op_a.id, op_c.id)
        graph.add_dependency(op_b.id, op_d.id)
        graph.add_dependency(op_c.id, op_d.id)
        
        sorted_ops = graph.topological_sort()
        ids = [op.id for op in sorted_ops]
        
        # A must come before B and C
        assert ids.index(op_a.id) < ids.index(op_b.id)
        assert ids.index(op_a.id) < ids.index(op_c.id)
        # B and C must come before D
        assert ids.index(op_b.id) < ids.index(op_d.id)
        assert ids.index(op_c.id) < ids.index(op_d.id)


class TestASAPScheduler:
    """Tests for ASAP scheduling algorithm."""
    
    def test_single_operation(self):
        """Test scheduling a single operation."""
        graph = DependencyGraph()
        op = graph.add_operation(OperationType.ADD, latency=0)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        assert schedule.get_time(op.id) == 0
        assert op.asap_time == 0
    
    def test_chain_zero_latency(self):
        """Test chain of operations with zero latency."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=0)
        op2 = graph.add_operation(OperationType.ADD, latency=0)
        op3 = graph.add_operation(OperationType.ADD, latency=0)
        
        graph.add_dependency(op1.id, op2.id)
        graph.add_dependency(op2.id, op3.id)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        # All can be scheduled at time 0 (combinational)
        assert schedule.get_time(op1.id) == 0
        assert schedule.get_time(op2.id) == 0
        assert schedule.get_time(op3.id) == 0
    
    def test_chain_with_latency(self):
        """Test chain of operations with latency."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.MUL, latency=1)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        op3 = graph.add_operation(OperationType.ADD, latency=0)
        
        graph.add_dependency(op1.id, op2.id)
        graph.add_dependency(op2.id, op3.id)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        assert schedule.get_time(op1.id) == 0
        assert schedule.get_time(op2.id) == 1  # After op1 completes
        assert schedule.get_time(op3.id) == 2  # After op2 completes
        assert schedule.total_latency == 3
    
    def test_parallel_operations(self):
        """Test independent operations are scheduled at same time."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD)
        op2 = graph.add_operation(OperationType.ADD)
        op3 = graph.add_operation(OperationType.ADD)
        
        # No dependencies - all independent
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        # All scheduled at time 0
        assert schedule.get_time(op1.id) == 0
        assert schedule.get_time(op2.id) == 0
        assert schedule.get_time(op3.id) == 0
    
    def test_diamond_pattern(self):
        """Test diamond dependency pattern."""
        graph = DependencyGraph()
        op_a = graph.add_operation(OperationType.ADD, latency=1)
        op_b = graph.add_operation(OperationType.ADD, latency=1)
        op_c = graph.add_operation(OperationType.MUL, latency=2)
        op_d = graph.add_operation(OperationType.ADD, latency=0)
        
        graph.add_dependency(op_a.id, op_b.id)
        graph.add_dependency(op_a.id, op_c.id)
        graph.add_dependency(op_b.id, op_d.id)
        graph.add_dependency(op_c.id, op_d.id)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        assert schedule.get_time(op_a.id) == 0
        assert schedule.get_time(op_b.id) == 1
        assert schedule.get_time(op_c.id) == 1
        # D waits for both B and C; C takes longer
        assert schedule.get_time(op_d.id) == 3


class TestALAPScheduler:
    """Tests for ALAP scheduling algorithm."""
    
    def test_single_operation(self):
        """Test ALAP scheduling of single operation."""
        graph = DependencyGraph()
        op = graph.add_operation(OperationType.ADD, latency=1)
        
        scheduler = ALAPScheduler()
        schedule = scheduler.schedule(graph)
        
        # Single operation scheduled at time 0
        assert schedule.get_time(op.id) == 0
        assert op.alap_time == 0
    
    def test_chain_with_latency(self):
        """Test ALAP scheduling of chain with latency."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.MUL, latency=1)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        op3 = graph.add_operation(OperationType.ADD, latency=1)
        
        graph.add_dependency(op1.id, op2.id)
        graph.add_dependency(op2.id, op3.id)
        
        scheduler = ALAPScheduler()
        schedule = scheduler.schedule(graph)
        
        # Critical path - same as ASAP
        assert schedule.get_time(op1.id) == 0
        assert schedule.get_time(op2.id) == 1
        assert schedule.get_time(op3.id) == 2
    
    def test_alap_with_slack(self):
        """Test ALAP scheduling pushes operations to latest time."""
        graph = DependencyGraph()
        # A -> D (short path)
        # B -> C -> D (longer path)
        op_a = graph.add_operation(OperationType.ADD, latency=1)
        op_b = graph.add_operation(OperationType.ADD, latency=1)
        op_c = graph.add_operation(OperationType.ADD, latency=1)
        op_d = graph.add_operation(OperationType.ADD, latency=1)
        
        graph.add_dependency(op_a.id, op_d.id)
        graph.add_dependency(op_b.id, op_c.id)
        graph.add_dependency(op_c.id, op_d.id)
        
        scheduler = ALAPScheduler()
        schedule = scheduler.schedule(graph)
        
        # B-C-D is critical path
        assert schedule.get_time(op_b.id) == 0
        assert schedule.get_time(op_c.id) == 1
        assert schedule.get_time(op_d.id) == 2
        # A can be delayed since A->D is shorter
        assert schedule.get_time(op_a.id) == 1  # Pushed to time 1


class TestMobilityAnalyzer:
    """Tests for mobility analysis."""
    
    def test_single_operation_zero_mobility(self):
        """Single operation has zero mobility."""
        graph = DependencyGraph()
        graph.add_operation(OperationType.ADD, latency=1)
        
        analyzer = MobilityAnalyzer()
        mobility = analyzer.analyze(graph)
        
        assert mobility[0] == 0
    
    def test_critical_path_zero_mobility(self):
        """Operations on critical path have zero mobility."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.MUL, latency=1)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        
        graph.add_dependency(op1.id, op2.id)
        
        analyzer = MobilityAnalyzer()
        mobility = analyzer.analyze(graph)
        
        # Both are on critical path
        assert mobility[op1.id] == 0
        assert mobility[op2.id] == 0
    
    def test_non_critical_has_mobility(self):
        """Non-critical operations have positive mobility."""
        graph = DependencyGraph()
        # A -> D (short)
        # B -> C -> D (long - critical)
        op_a = graph.add_operation(OperationType.ADD, latency=1)
        op_b = graph.add_operation(OperationType.ADD, latency=1)
        op_c = graph.add_operation(OperationType.ADD, latency=1)
        op_d = graph.add_operation(OperationType.ADD, latency=1)
        
        graph.add_dependency(op_a.id, op_d.id)
        graph.add_dependency(op_b.id, op_c.id)
        graph.add_dependency(op_c.id, op_d.id)
        
        analyzer = MobilityAnalyzer()
        mobility = analyzer.analyze(graph)
        
        # B, C, D are on critical path
        assert mobility[op_b.id] == 0
        assert mobility[op_c.id] == 0
        assert mobility[op_d.id] == 0
        # A has slack
        assert mobility[op_a.id] == 1
    
    def test_get_critical_path(self):
        """Test extracting critical path operations."""
        graph = DependencyGraph()
        op_a = graph.add_operation(OperationType.ADD, latency=1)
        op_b = graph.add_operation(OperationType.ADD, latency=1)
        op_c = graph.add_operation(OperationType.ADD, latency=1)
        op_d = graph.add_operation(OperationType.ADD, latency=1)
        
        graph.add_dependency(op_a.id, op_d.id)
        graph.add_dependency(op_b.id, op_c.id)
        graph.add_dependency(op_c.id, op_d.id)
        
        analyzer = MobilityAnalyzer()
        critical = analyzer.get_critical_path(graph)
        critical_ids = {op.id for op in critical}
        
        # B, C, D are critical
        assert op_b.id in critical_ids
        assert op_c.id in critical_ids
        assert op_d.id in critical_ids
        # A is not critical
        assert op_a.id not in critical_ids


class TestFSMToScheduleGraphBuilder:
    """Tests for building schedule graphs from FSM."""
    
    def test_build_from_empty_fsm(self):
        """Test building from FSM with no operations."""
        fsm = FSMModule(name="test")
        fsm.add_state("IDLE")
        
        builder = FSMToScheduleGraphBuilder()
        graph = builder.build(fsm)
        
        assert len(graph.operations) == 0
    
    def test_build_from_single_assign(self):
        """Test building from FSM with single assignment."""
        fsm = FSMModule(name="test")
        state = fsm.add_state("S0")
        state.add_operation(FSMAssign(target="out", value=1))
        
        builder = FSMToScheduleGraphBuilder()
        graph = builder.build(fsm)
        
        assert len(graph.operations) == 1
        assert graph.operations[0].op_type == OperationType.ASSIGN
    
    def test_build_raw_dependency(self):
        """Test RAW dependency detection."""
        fsm = FSMModule(name="test")
        state = fsm.add_state("S0")
        # a = 1
        state.add_operation(FSMAssign(target="a", value=1))
        # b = a (RAW dependency on a)
        state.add_operation(FSMAssign(target="b", value="a"))
        
        builder = FSMToScheduleGraphBuilder()
        graph = builder.build(fsm)
        
        assert len(graph.operations) == 2


class TestDefaultLatency:
    """Tests for default operation latencies."""
    
    def test_combinational_ops(self):
        """Combinational operations have zero latency."""
        assert default_latency(OperationType.ADD) == 0
        assert default_latency(OperationType.SUB) == 0
        assert default_latency(OperationType.ASSIGN) == 0
        assert default_latency(OperationType.MUX) == 0
    
    def test_multi_cycle_ops(self):
        """Multi-cycle operations have positive latency."""
        assert default_latency(OperationType.MUL) >= 1
        assert default_latency(OperationType.DIV) >= 1
        assert default_latency(OperationType.LOAD) >= 1


# =============================================================================
# Tests for Resource-Constrained Scheduling (Phase 2.2)
# =============================================================================

class TestResourceConstraints:
    """Tests for resource constraint specification."""
    
    def test_default_constraints(self):
        """Test default resource constraints."""
        constraints = ResourceConstraints()
        assert constraints.max_adders == 2
        assert constraints.max_multipliers == 1
        assert constraints.max_dividers == 1
    
    def test_custom_constraints(self):
        """Test custom resource constraints."""
        constraints = ResourceConstraints(max_adders=4, max_multipliers=2)
        assert constraints.max_adders == 4
        assert constraints.max_multipliers == 2
    
    def test_get_limit(self):
        """Test getting resource limits for operation types."""
        constraints = ResourceConstraints(max_adders=3, max_multipliers=2)
        
        assert constraints.get_limit(OperationType.ADD) == 3
        assert constraints.get_limit(OperationType.SUB) == 3  # Same as add
        assert constraints.get_limit(OperationType.MUL) == 2
        # ASSIGN has unlimited resources
        assert constraints.get_limit(OperationType.ASSIGN) == float('inf')


class TestResourceUsage:
    """Tests for resource usage tracking."""
    
    def test_initial_usage(self):
        """Test initial state of resource usage."""
        usage = ResourceUsage()
        constraints = ResourceConstraints(max_adders=2)
        
        # Can use resources initially
        assert usage.can_use(OperationType.ADD, constraints)
    
    def test_use_resource(self):
        """Test using resources."""
        usage = ResourceUsage()
        constraints = ResourceConstraints(max_adders=2)
        
        usage.use(OperationType.ADD)
        assert usage.can_use(OperationType.ADD, constraints)  # Still have 1 left
        
        usage.use(OperationType.ADD)
        assert not usage.can_use(OperationType.ADD, constraints)  # At limit
    
    def test_reset_usage(self):
        """Test resetting usage for new time step."""
        usage = ResourceUsage()
        constraints = ResourceConstraints(max_adders=1)
        
        usage.use(OperationType.ADD)
        assert not usage.can_use(OperationType.ADD, constraints)
        
        usage.reset()
        assert usage.can_use(OperationType.ADD, constraints)


class TestListScheduler:
    """Tests for resource-constrained list scheduling."""
    
    def test_unconstrained_matches_asap(self):
        """With no constraints, list scheduling matches ASAP."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=1)
        op2 = graph.add_operation(OperationType.ADD, latency=1)
        
        graph.add_dependency(op1.id, op2.id)
        
        # Use very high limits (effectively unconstrained)
        constraints = ResourceConstraints(max_adders=100)
        scheduler = ListScheduler(constraints)
        schedule = scheduler.schedule(graph)
        
        asap = ASAPScheduler().schedule(graph)
        
        assert schedule.get_time(op1.id) == asap.get_time(op1.id)
        assert schedule.get_time(op2.id) == asap.get_time(op2.id)
    
    def test_constrained_scheduling(self):
        """Test that resource constraints are respected."""
        graph = DependencyGraph()
        # Three independent multiplications
        op1 = graph.add_operation(OperationType.MUL, latency=1)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        op3 = graph.add_operation(OperationType.MUL, latency=1)
        
        # Only 1 multiplier available
        constraints = ResourceConstraints(max_multipliers=1)
        scheduler = ListScheduler(constraints)
        schedule = scheduler.schedule(graph)
        
        # All three must be scheduled at different times
        times = [schedule.get_time(op1.id), 
                 schedule.get_time(op2.id), 
                 schedule.get_time(op3.id)]
        assert len(set(times)) == 3  # All different
    
    def test_mixed_resources(self):
        """Test scheduling with different resource types."""
        graph = DependencyGraph()
        # One mul and two adds (independent)
        mul_op = graph.add_operation(OperationType.MUL, latency=1)
        add1 = graph.add_operation(OperationType.ADD, latency=0)
        add2 = graph.add_operation(OperationType.ADD, latency=0)
        
        # 1 multiplier, 2 adders
        constraints = ResourceConstraints(max_multipliers=1, max_adders=2)
        scheduler = ListScheduler(constraints)
        schedule = scheduler.schedule(graph)
        
        # Adds can be at same time, mul is separate resource
        # All can be scheduled at time 0 since they use different resources
        assert schedule.get_time(add1.id) == 0
        assert schedule.get_time(add2.id) == 0
        assert schedule.get_time(mul_op.id) == 0
    
    def test_priority_by_mobility(self):
        """Test that critical operations are scheduled first."""
        graph = DependencyGraph()
        # A -> D (short path)
        # B -> C -> D (long path - critical)
        op_a = graph.add_operation(OperationType.MUL, latency=1)
        op_b = graph.add_operation(OperationType.MUL, latency=1)
        op_c = graph.add_operation(OperationType.MUL, latency=1)
        op_d = graph.add_operation(OperationType.MUL, latency=1)
        
        graph.add_dependency(op_a.id, op_d.id)
        graph.add_dependency(op_b.id, op_c.id)
        graph.add_dependency(op_c.id, op_d.id)
        
        # Only 1 multiplier - must serialize
        constraints = ResourceConstraints(max_multipliers=1)
        scheduler = ListScheduler(constraints, PriorityMetric.CRITICAL_PATH)
        schedule = scheduler.schedule(graph)
        
        # B should be scheduled before A (B is on critical path)
        assert schedule.get_time(op_b.id) <= schedule.get_time(op_a.id)
    
    def test_dependency_order_preserved(self):
        """Test that dependencies are always respected."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.MUL, latency=2)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        
        graph.add_dependency(op1.id, op2.id)
        
        constraints = ResourceConstraints(max_multipliers=1)
        scheduler = ListScheduler(constraints)
        schedule = scheduler.schedule(graph)
        
        # op2 must start after op1 completes (time 2)
        assert schedule.get_time(op2.id) >= schedule.get_time(op1.id) + 2


class TestSDCScheduler:
    """Tests for SDC-based scheduler (currently using list scheduler fallback)."""
    
    def test_sdc_basic(self):
        """Test SDC scheduler produces valid schedule."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=1)
        op2 = graph.add_operation(OperationType.ADD, latency=1)
        
        graph.add_dependency(op1.id, op2.id)
        
        scheduler = SDCScheduler()
        schedule = scheduler.schedule(graph)
        
        # Dependencies respected
        assert schedule.get_time(op2.id) >= schedule.get_time(op1.id) + 1
