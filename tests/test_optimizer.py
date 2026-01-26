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
"""Tests for FSM optimization passes."""

import pytest
import sys
import os

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

if '' in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

from zuspec.synth.sprtl.optimizer import (
    DeadStateEliminator, StateMinimizer, TransitionOptimizer,
    OperationMerger, FSMOptimizer, OptimizationStats, optimize_fsm
)
from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMStateKind, FSMAssign


class TestOptimizationStats:
    """Tests for optimization statistics."""
    
    def test_default_stats(self):
        """Test default statistics are zero."""
        stats = OptimizationStats()
        assert stats.states_removed == 0
        assert stats.transitions_removed == 0
        assert stats.operations_merged == 0
    
    def test_str_format(self):
        """Test string representation."""
        stats = OptimizationStats(states_removed=2, operations_merged=5)
        s = str(stats)
        assert "States removed: 2" in s
        assert "Operations merged: 5" in s


class TestDeadStateEliminator:
    """Tests for dead state elimination."""
    
    def test_no_dead_states(self):
        """Test FSM with no dead states."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        s0.add_transition(s1.id)
        s1.add_transition(s0.id)
        
        eliminator = DeadStateEliminator()
        stats = eliminator.optimize(fsm)
        
        assert stats.states_removed == 0
        assert len(fsm.states) == 2
    
    def test_remove_unreachable_state(self):
        """Test removing unreachable state."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        s2 = fsm.add_state("UNREACHABLE")  # No transitions to this
        
        s0.add_transition(s1.id)
        s1.add_transition(s0.id)
        # s2 is never reached
        
        eliminator = DeadStateEliminator()
        stats = eliminator.optimize(fsm)
        
        assert stats.states_removed == 1
        assert len(fsm.states) == 2
        assert not any(s.name == "UNREACHABLE" for s in fsm.states)
    
    def test_chain_reachability(self):
        """Test state reachable through chain."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        s2 = fsm.add_state("S2")
        s3 = fsm.add_state("S3")
        
        s0.add_transition(s1.id)
        s1.add_transition(s2.id)
        s2.add_transition(s3.id)
        s3.add_transition(s0.id)
        
        eliminator = DeadStateEliminator()
        stats = eliminator.optimize(fsm)
        
        assert stats.states_removed == 0
        assert len(fsm.states) == 4


class TestStateMinimizer:
    """Tests for state minimization."""
    
    def test_no_equivalent_states(self):
        """Test FSM with no equivalent states."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        
        # Different operations make them non-equivalent
        s0.add_operation(FSMAssign(target="a", value=1))
        s1.add_operation(FSMAssign(target="b", value=2))
        
        s0.add_transition(s1.id)
        s1.add_transition(s0.id)
        
        minimizer = StateMinimizer()
        stats = minimizer.optimize(fsm)
        
        assert stats.states_merged == 0
        assert len(fsm.states) == 2
    
    def test_single_state(self):
        """Test single-state FSM (nothing to minimize)."""
        fsm = FSMModule(name="test")
        fsm.add_state("S0")
        
        minimizer = StateMinimizer()
        stats = minimizer.optimize(fsm)
        
        assert stats.states_merged == 0
        assert len(fsm.states) == 1


class TestTransitionOptimizer:
    """Tests for transition optimization."""
    
    def test_remove_duplicate_transitions(self):
        """Test removing duplicate transitions."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        
        # Add duplicate transitions
        s0.add_transition(s1.id)
        s0.add_transition(s1.id)  # Duplicate
        s0.add_transition(s1.id)  # Duplicate
        
        optimizer = TransitionOptimizer()
        stats = optimizer.optimize(fsm)
        
        assert stats.transitions_removed == 2
        assert len(s0.transitions) == 1
    
    def test_sort_by_priority(self):
        """Test transitions are sorted by priority."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        s2 = fsm.add_state("S2")
        
        s0.add_transition(s1.id, priority=2)
        s0.add_transition(s2.id, priority=1)
        
        optimizer = TransitionOptimizer()
        optimizer.optimize(fsm)
        
        # Lower priority first
        assert s0.transitions[0].target_state == s2.id
        assert s0.transitions[1].target_state == s1.id


class TestOperationMerger:
    """Tests for operation merging."""
    
    def test_no_merge_needed(self):
        """Test state with non-mergeable operations."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s0.add_operation(FSMAssign(target="a", value=1))
        s0.add_operation(FSMAssign(target="b", value=2))
        
        merger = OperationMerger()
        stats = merger.optimize(fsm)
        
        # Different targets, no merge
        assert stats.operations_merged == 0
        assert len(s0.operations) == 2
    
    def test_merge_same_target(self):
        """Test merging assignments to same target."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s0.add_operation(FSMAssign(target="a", value=1))
        s0.add_operation(FSMAssign(target="a", value=2))  # Overwrites
        
        merger = OperationMerger()
        stats = merger.optimize(fsm)
        
        assert stats.operations_merged == 1
        assert len(s0.operations) == 1
        # Should keep the last assignment
        assert s0.operations[0].value == 2


class TestFSMOptimizer:
    """Tests for high-level FSM optimizer."""
    
    def test_all_passes(self):
        """Test running all optimization passes."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        s2 = fsm.add_state("DEAD")  # Unreachable
        
        s0.add_transition(s1.id)
        s0.add_transition(s1.id)  # Duplicate
        s1.add_transition(s0.id)
        
        optimizer = FSMOptimizer()
        stats = optimizer.optimize(fsm)
        
        assert stats.states_removed >= 1  # DEAD removed
        assert stats.transitions_removed >= 1  # Duplicate removed
    
    def test_selective_passes(self):
        """Test running selective passes."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("DEAD")
        
        s0.add_transition(s0.id)
        
        # Only run dead state elimination
        optimizer = FSMOptimizer(
            eliminate_dead=True,
            minimize_states=False,
            optimize_transitions=False,
            merge_operations=False
        )
        stats = optimizer.optimize(fsm)
        
        assert stats.states_removed == 1
        assert len(fsm.states) == 1


class TestOptimizeFSMFunction:
    """Tests for convenience function."""
    
    def test_optimize_fsm(self):
        """Test optimize_fsm convenience function."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("DEAD")
        s0.add_transition(s0.id)
        
        stats = optimize_fsm(fsm)
        
        assert stats.states_removed >= 1
        assert len(fsm.states) == 1
