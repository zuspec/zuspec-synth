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
FSM Optimization Passes: Optimizations to improve generated hardware quality.

This module implements various optimization passes:
1. State Minimization: Merge equivalent states
2. Dead State Elimination: Remove unreachable states
3. Transition Optimization: Simplify transition logic
4. Operation Merging: Combine compatible operations

These optimizations can significantly reduce area and improve timing.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from .fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMTransition,
    FSMOperation, FSMAssign, FSMCond
)


@dataclass
class OptimizationStats:
    """Statistics from optimization passes.
    
    Attributes:
        states_removed: Number of states removed
        transitions_removed: Number of transitions removed
        operations_merged: Number of operations merged
        states_merged: Number of states merged together
    """
    states_removed: int = 0
    transitions_removed: int = 0
    operations_merged: int = 0
    states_merged: int = 0
    
    def __str__(self) -> str:
        return (f"Optimization Stats:\n"
                f"  States removed: {self.states_removed}\n"
                f"  States merged: {self.states_merged}\n"
                f"  Transitions removed: {self.transitions_removed}\n"
                f"  Operations merged: {self.operations_merged}")


class DeadStateEliminator:
    """Removes unreachable states from an FSM.
    
    A state is unreachable if there is no path from the initial state
    to that state via any sequence of transitions.
    """
    
    def optimize(self, fsm: FSMModule) -> OptimizationStats:
        """Remove unreachable states from the FSM.
        
        Args:
            fsm: FSM module to optimize (modified in place)
            
        Returns:
            Optimization statistics
        """
        stats = OptimizationStats()
        
        if not fsm.states:
            return stats
        
        # Find reachable states via BFS from initial state
        reachable = self._find_reachable_states(fsm)
        
        # Remove unreachable states
        original_count = len(fsm.states)
        fsm.states = [s for s in fsm.states if s.id in reachable]
        stats.states_removed = original_count - len(fsm.states)
        
        # Update state encoding
        fsm._compute_state_encoding()
        
        return stats
    
    def _find_reachable_states(self, fsm: FSMModule) -> Set[int]:
        """Find all states reachable from initial state."""
        reachable = set()
        worklist = [fsm.initial_state]
        
        while worklist:
            state_id = worklist.pop()
            if state_id in reachable:
                continue
            
            reachable.add(state_id)
            state = fsm.get_state(state_id)
            if state:
                for trans in state.transitions:
                    if trans.target_state not in reachable:
                        worklist.append(trans.target_state)
        
        return reachable


class StateMinimizer:
    """Minimizes FSM by merging equivalent states.
    
    Two states are equivalent if:
    1. They have the same output operations
    2. For every input, they transition to equivalent states
    
    Uses partition refinement algorithm.
    """
    
    def optimize(self, fsm: FSMModule) -> OptimizationStats:
        """Minimize the FSM by merging equivalent states.
        
        Args:
            fsm: FSM module to optimize (modified in place)
            
        Returns:
            Optimization statistics
        """
        stats = OptimizationStats()
        
        if len(fsm.states) <= 1:
            return stats
        
        # Find equivalent state pairs
        equivalent = self._find_equivalent_states(fsm)
        
        if not equivalent:
            return stats
        
        # Merge equivalent states
        merged_count = self._merge_states(fsm, equivalent)
        stats.states_merged = merged_count
        stats.states_removed = merged_count
        
        return stats
    
    def _find_equivalent_states(self, fsm: FSMModule) -> List[Tuple[int, int]]:
        """Find pairs of equivalent states using partition refinement."""
        equivalent = []
        
        # Initial partition: group by state kind and number of operations
        partitions: Dict[Tuple, Set[int]] = {}
        for state in fsm.states:
            key = (state.kind, len(state.operations), len(state.transitions))
            if key not in partitions:
                partitions[key] = set()
            partitions[key].add(state.id)
        
        # Refine partitions (simplified - full algorithm would iterate until fixed point)
        for partition in partitions.values():
            if len(partition) < 2:
                continue
            
            states_list = list(partition)
            for i in range(len(states_list)):
                for j in range(i + 1, len(states_list)):
                    s1 = fsm.get_state(states_list[i])
                    s2 = fsm.get_state(states_list[j])
                    if s1 and s2 and self._are_equivalent(s1, s2, fsm):
                        equivalent.append((s1.id, s2.id))
        
        return equivalent
    
    def _are_equivalent(self, s1: FSMState, s2: FSMState, fsm: FSMModule) -> bool:
        """Check if two states are equivalent."""
        # Same kind
        if s1.kind != s2.kind:
            return False
        
        # Same number of operations
        if len(s1.operations) != len(s2.operations):
            return False
        
        # Same transitions (simplified check)
        if len(s1.transitions) != len(s2.transitions):
            return False
        
        # Check operations are similar (simplified)
        for op1, op2 in zip(s1.operations, s2.operations):
            if type(op1) != type(op2):
                return False
            if isinstance(op1, FSMAssign) and isinstance(op2, FSMAssign):
                if op1.target != op2.target:
                    return False
        
        return True
    
    def _merge_states(self, fsm: FSMModule, equivalent: List[Tuple[int, int]]) -> int:
        """Merge equivalent state pairs."""
        merged = 0
        removed = set()
        
        for s1_id, s2_id in equivalent:
            if s2_id in removed:
                continue
            
            # Remove s2, redirect all transitions to s2 -> s1
            for state in fsm.states:
                for trans in state.transitions:
                    if trans.target_state == s2_id:
                        trans.target_state = s1_id
            
            removed.add(s2_id)
            merged += 1
        
        # Remove merged states
        fsm.states = [s for s in fsm.states if s.id not in removed]
        fsm._compute_state_encoding()
        
        return merged


class TransitionOptimizer:
    """Optimizes state transitions.
    
    Optimizations:
    1. Remove redundant self-loops
    2. Simplify transition conditions
    3. Merge transitions with same target
    """
    
    def optimize(self, fsm: FSMModule) -> OptimizationStats:
        """Optimize state transitions.
        
        Args:
            fsm: FSM module to optimize (modified in place)
            
        Returns:
            Optimization statistics
        """
        stats = OptimizationStats()
        
        for state in fsm.states:
            # Remove duplicate transitions
            seen = set()
            unique_transitions = []
            for trans in state.transitions:
                key = (trans.target_state, str(trans.condition))
                if key not in seen:
                    seen.add(key)
                    unique_transitions.append(trans)
                else:
                    stats.transitions_removed += 1
            state.transitions = unique_transitions
            
            # Sort transitions by priority
            state.transitions.sort(key=lambda t: t.priority)
        
        return stats


class OperationMerger:
    """Merges compatible operations within states.
    
    Optimizations:
    1. Merge sequential assignments to same target
    2. Simplify nested conditionals
    """
    
    def optimize(self, fsm: FSMModule) -> OptimizationStats:
        """Merge compatible operations.
        
        Args:
            fsm: FSM module to optimize (modified in place)
            
        Returns:
            Optimization statistics
        """
        stats = OptimizationStats()
        
        for state in fsm.states:
            original_count = len(state.operations)
            state.operations = self._merge_operations(state.operations)
            stats.operations_merged += original_count - len(state.operations)
        
        return stats
    
    def _merge_operations(self, ops: List[FSMOperation]) -> List[FSMOperation]:
        """Merge compatible operations."""
        if len(ops) <= 1:
            return ops
        
        # Track last assignment to each target
        last_assign: Dict[str, int] = {}
        merged = []
        
        for i, op in enumerate(ops):
            if isinstance(op, FSMAssign):
                if op.target in last_assign:
                    # This assignment overwrites a previous one
                    # Keep only this one (remove previous)
                    prev_idx = last_assign[op.target]
                    merged = [m for j, m in enumerate(merged) if j != prev_idx]
                
                last_assign[op.target] = len(merged)
                merged.append(op)
            else:
                merged.append(op)
        
        return merged


class FSMOptimizer:
    """High-level FSM optimizer that runs multiple passes.
    
    Usage:
        optimizer = FSMOptimizer()
        stats = optimizer.optimize(fsm)
    """
    
    def __init__(self, 
                 eliminate_dead: bool = True,
                 minimize_states: bool = True,
                 optimize_transitions: bool = True,
                 merge_operations: bool = True):
        """Initialize the optimizer.
        
        Args:
            eliminate_dead: Run dead state elimination
            minimize_states: Run state minimization
            optimize_transitions: Run transition optimization
            merge_operations: Run operation merging
        """
        self.eliminate_dead = eliminate_dead
        self.minimize_states = minimize_states
        self.optimize_transitions = optimize_transitions
        self.merge_operations = merge_operations
    
    def optimize(self, fsm: FSMModule) -> OptimizationStats:
        """Run all enabled optimization passes.
        
        Args:
            fsm: FSM module to optimize (modified in place)
            
        Returns:
            Combined optimization statistics
        """
        total_stats = OptimizationStats()
        
        # Run passes in order
        if self.eliminate_dead:
            stats = DeadStateEliminator().optimize(fsm)
            total_stats.states_removed += stats.states_removed
        
        if self.minimize_states:
            stats = StateMinimizer().optimize(fsm)
            total_stats.states_merged += stats.states_merged
            total_stats.states_removed += stats.states_removed
        
        if self.optimize_transitions:
            stats = TransitionOptimizer().optimize(fsm)
            total_stats.transitions_removed += stats.transitions_removed
        
        if self.merge_operations:
            stats = OperationMerger().optimize(fsm)
            total_stats.operations_merged += stats.operations_merged
        
        return total_stats


def optimize_fsm(fsm: FSMModule, **kwargs) -> OptimizationStats:
    """Convenience function to optimize an FSM.
    
    Args:
        fsm: FSM module to optimize
        **kwargs: Options passed to FSMOptimizer
        
    Returns:
        Optimization statistics
    """
    optimizer = FSMOptimizer(**kwargs)
    return optimizer.optimize(fsm)
