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
"""Tests for FSM generation and register allocation."""

import pytest
import sys
import os

# Ensure paths are set correctly for development
_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

if '' in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

from zuspec.synth.sprtl.fsm_generator import (
    StateEncoding, FSMGeneratorConfig, LiveRange,
    FSMGenerator, RegisterAllocator, ScheduleToFSMBuilder
)
from zuspec.synth.sprtl.scheduler import (
    Schedule, DependencyGraph, OperationType, ASAPScheduler
)
from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMStateKind, FSMAssign


class TestLiveRange:
    """Tests for live range analysis."""
    
    def test_live_range_creation(self):
        """Test creating a live range."""
        lr = LiveRange("x", start=0, end=5)
        assert lr.variable == "x"
        assert lr.start == 0
        assert lr.end == 5
    
    def test_overlaps_true(self):
        """Test overlapping live ranges."""
        lr1 = LiveRange("a", 0, 5)
        lr2 = LiveRange("b", 3, 8)
        
        assert lr1.overlaps(lr2)
        assert lr2.overlaps(lr1)
    
    def test_overlaps_false(self):
        """Test non-overlapping live ranges."""
        lr1 = LiveRange("a", 0, 3)
        lr2 = LiveRange("b", 5, 8)
        
        assert not lr1.overlaps(lr2)
        assert not lr2.overlaps(lr1)
    
    def test_overlaps_adjacent(self):
        """Test adjacent live ranges (touching but not overlapping)."""
        lr1 = LiveRange("a", 0, 3)
        lr2 = LiveRange("b", 3, 5)
        
        # Adjacent ranges do overlap (at time 3)
        assert lr1.overlaps(lr2)
    
    def test_overlaps_contained(self):
        """Test when one range is contained in another."""
        lr1 = LiveRange("a", 0, 10)
        lr2 = LiveRange("b", 3, 5)
        
        assert lr1.overlaps(lr2)
        assert lr2.overlaps(lr1)


class TestFSMGeneratorConfig:
    """Tests for FSM generator configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FSMGeneratorConfig()
        assert config.state_prefix == "S"
        assert config.encoding == StateEncoding.BINARY
        assert config.optimize_single_cycle == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FSMGeneratorConfig(
            state_prefix="STATE",
            encoding=StateEncoding.ONEHOT,
            generate_done_signal=True
        )
        assert config.state_prefix == "STATE"
        assert config.encoding == StateEncoding.ONEHOT
        assert config.generate_done_signal == True


class TestFSMGenerator:
    """Tests for FSM generation from schedule."""
    
    def test_empty_schedule(self):
        """Test generating FSM from empty schedule."""
        graph = DependencyGraph()
        schedule = Schedule()
        
        generator = FSMGenerator()
        fsm = generator.generate(schedule, graph)
        
        assert fsm.name == "scheduled_fsm"
        assert len(fsm.states) == 0
    
    def test_single_operation(self):
        """Test FSM with single operation."""
        graph = DependencyGraph()
        op = graph.add_operation(OperationType.ADD, latency=1)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        generator = FSMGenerator()
        fsm = generator.generate(schedule, graph)
        
        assert len(fsm.states) == 1
        assert fsm.states[0].name == "S0"
    
    def test_sequential_operations(self):
        """Test FSM with sequential operations."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.MUL, latency=1)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        
        graph.add_dependency(op1.id, op2.id)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        generator = FSMGenerator()
        fsm = generator.generate(schedule, graph)
        
        assert len(fsm.states) == 2
        # First state should transition to second
        assert len(fsm.states[0].transitions) == 1
        assert fsm.states[0].transitions[0].target_state == fsm.states[1].id
    
    def test_parallel_operations(self):
        """Test FSM with parallel operations (same time step)."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=0)
        op2 = graph.add_operation(OperationType.ADD, latency=0)
        # No dependencies - both at time 0
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        generator = FSMGenerator()
        fsm = generator.generate(schedule, graph)
        
        # Single state with both operations
        assert len(fsm.states) == 1
    
    def test_state_encoding_binary(self):
        """Test binary state encoding."""
        graph = DependencyGraph()
        for _ in range(4):
            graph.add_operation(OperationType.MUL, latency=1)
        
        # Chain them
        for i in range(3):
            graph.add_dependency(i, i + 1)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        config = FSMGeneratorConfig(encoding=StateEncoding.BINARY)
        generator = FSMGenerator(config)
        fsm = generator.generate(schedule, graph)
        
        assert fsm.state_width == 2  # 4 states need 2 bits
        assert fsm.state_encoding[fsm.states[0].id] == 0
        assert fsm.state_encoding[fsm.states[1].id] == 1
    
    def test_state_encoding_onehot(self):
        """Test one-hot state encoding."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.MUL, latency=1)
        op2 = graph.add_operation(OperationType.MUL, latency=1)
        op3 = graph.add_operation(OperationType.MUL, latency=1)
        
        graph.add_dependency(op1.id, op2.id)
        graph.add_dependency(op2.id, op3.id)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        config = FSMGeneratorConfig(encoding=StateEncoding.ONEHOT)
        generator = FSMGenerator(config)
        fsm = generator.generate(schedule, graph)
        
        assert fsm.state_width == 3  # 3 states
        assert fsm.state_encoding[fsm.states[0].id] == 0b001
        assert fsm.state_encoding[fsm.states[1].id] == 0b010
        assert fsm.state_encoding[fsm.states[2].id] == 0b100
    
    def test_state_encoding_gray(self):
        """Test Gray code state encoding."""
        graph = DependencyGraph()
        for _ in range(4):
            graph.add_operation(OperationType.MUL, latency=1)
        
        for i in range(3):
            graph.add_dependency(i, i + 1)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        config = FSMGeneratorConfig(encoding=StateEncoding.GRAY)
        generator = FSMGenerator(config)
        fsm = generator.generate(schedule, graph)
        
        # Gray code: 0, 1, 3, 2 (00, 01, 11, 10)
        assert fsm.state_encoding[fsm.states[0].id] == 0  # 00
        assert fsm.state_encoding[fsm.states[1].id] == 1  # 01
        assert fsm.state_encoding[fsm.states[2].id] == 3  # 11
        assert fsm.state_encoding[fsm.states[3].id] == 2  # 10
    
    def test_with_source_fsm(self):
        """Test generation with source FSM for metadata."""
        source = FSMModule(name="counter", clock_signal="clk", reset_signal="rst_n")
        source.add_port("count", "output", 32, reset_value=0)
        
        graph = DependencyGraph()
        graph.add_operation(OperationType.ADD, latency=1)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        generator = FSMGenerator()
        fsm = generator.generate(schedule, graph, source_fsm=source)
        
        assert fsm.name == "counter"
        assert fsm.clock_signal == "clk"
        assert len(fsm.ports) == 1
        assert fsm.ports[0].name == "count"


class TestRegisterAllocator:
    """Tests for register allocation."""
    
    def test_allocator_creation(self):
        """Test creating a register allocator."""
        allocator = RegisterAllocator()
        assert allocator.get_register_count() == 0
    
    def test_allocate_single_variable(self):
        """Test allocating a single variable."""
        allocator = RegisterAllocator()
        live_ranges = {
            "a": LiveRange("a", 0, 5)
        }
        
        allocation = allocator.allocate(live_ranges)
        
        assert "a" in allocation
        assert allocation["a"] == "r0"
    
    def test_allocate_non_overlapping(self):
        """Test allocating non-overlapping variables share register."""
        allocator = RegisterAllocator()
        live_ranges = {
            "a": LiveRange("a", 0, 2),
            "b": LiveRange("b", 5, 8),
        }
        
        allocation = allocator.allocate(live_ranges)
        
        # Non-overlapping can share register
        assert allocation["a"] == allocation["b"]
    
    def test_allocate_overlapping(self):
        """Test allocating overlapping variables need separate registers."""
        allocator = RegisterAllocator()
        live_ranges = {
            "a": LiveRange("a", 0, 5),
            "b": LiveRange("b", 3, 8),
        }
        
        allocation = allocator.allocate(live_ranges)
        
        # Overlapping need different registers
        assert allocation["a"] != allocation["b"]
    
    def test_register_count(self):
        """Test counting allocated registers."""
        allocator = RegisterAllocator()
        live_ranges = {
            "a": LiveRange("a", 0, 5),
            "b": LiveRange("b", 3, 8),
            "c": LiveRange("c", 10, 15),  # Non-overlapping with a
        }
        
        allocation = allocator.allocate(live_ranges)
        count = len(set(allocation.values()))
        
        # a and c can share, b needs separate = 2 registers
        assert count == 2
    
    def test_left_edge_algorithm(self):
        """Test left-edge algorithm produces optimal allocation."""
        allocator = RegisterAllocator()
        # Classic left-edge example
        live_ranges = {
            "a": LiveRange("a", 0, 3),
            "b": LiveRange("b", 1, 4),
            "c": LiveRange("c", 2, 5),
            "d": LiveRange("d", 4, 6),  # Can share with a
        }
        
        allocation = allocator.allocate(live_ranges)
        
        # a, b, c overlap - need 3 registers
        # d can share with a (doesn't overlap)
        assert allocation["a"] == allocation["d"]


class TestScheduleToFSMBuilder:
    """Tests for the high-level builder."""
    
    def test_builder_basic(self):
        """Test basic builder usage."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=1)
        op2 = graph.add_operation(OperationType.ADD, latency=1)
        graph.add_dependency(op1.id, op2.id)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        builder = ScheduleToFSMBuilder()
        fsm = builder.build(graph, schedule)
        
        assert len(fsm.states) == 2
    
    def test_builder_with_source(self):
        """Test builder with source FSM."""
        source = FSMModule(name="test_module")
        source.add_port("clk", "input", 1)
        source.add_port("out", "output", 8)
        
        graph = DependencyGraph()
        graph.add_operation(OperationType.ADD, latency=1)
        
        scheduler = ASAPScheduler()
        schedule = scheduler.schedule(graph)
        
        builder = ScheduleToFSMBuilder()
        fsm = builder.build(graph, schedule, source_fsm=source)
        
        assert fsm.name == "test_module"
        assert len(fsm.ports) == 2
