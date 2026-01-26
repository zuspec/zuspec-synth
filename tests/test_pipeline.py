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
"""Tests for pipelining support."""

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

from zuspec.synth.sprtl.pipeline import (
    HazardType, PipelineStage, PipelineRegister, Pipeline,
    PipelineConfig, HazardDetector, PipelineScheduler,
    PipelineGenerator, PipelineSVGenerator, generate_pipeline
)
from zuspec.synth.sprtl.scheduler import DependencyGraph, OperationType, Schedule
from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMAssign


class TestPipelineStage:
    """Tests for pipeline stage data structure."""
    
    def test_basic_stage(self):
        """Test creating a pipeline stage."""
        stage = PipelineStage(stage_id=0)
        assert stage.stage_id == 0
        assert stage.valid_signal == "stage0_valid"
    
    def test_stage_with_operations(self):
        """Test stage with operations."""
        stage = PipelineStage(stage_id=1)
        stage.operations.append(FSMAssign(target="out", value=1))
        stage.inputs.add("in")
        stage.outputs.add("out")
        
        assert len(stage.operations) == 1
        assert "in" in stage.inputs
        assert "out" in stage.outputs


class TestPipeline:
    """Tests for pipeline data structure."""
    
    def test_empty_pipeline(self):
        """Test creating an empty pipeline."""
        pipe = Pipeline(name="test")
        assert pipe.name == "test"
        assert pipe.num_stages == 0
    
    def test_add_stages(self):
        """Test adding stages to pipeline."""
        pipe = Pipeline(name="test")
        s0 = pipe.add_stage()
        s1 = pipe.add_stage()
        
        assert pipe.num_stages == 2
        assert s0.stage_id == 0
        assert s1.stage_id == 1
    
    def test_add_register(self):
        """Test adding pipeline registers."""
        pipe = Pipeline(name="test")
        pipe.add_stage()
        pipe.add_stage()
        
        reg = pipe.add_register("x_p0", 32, 0, 1, "x")
        
        assert len(pipe.registers) == 1
        assert reg.source_stage == 0
        assert reg.dest_stage == 1


class TestPipelineConfig:
    """Tests for pipeline configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        assert config.target_ii == 1
        assert config.insert_valid_signals == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(target_ii=2, max_stages=4)
        assert config.target_ii == 2
        assert config.max_stages == 4


class TestHazardDetector:
    """Tests for hazard detection."""
    
    def test_no_hazards(self):
        """Test graph with no hazards."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=0)
        op2 = graph.add_operation(OperationType.ADD, latency=0)
        # Independent operations
        
        schedule = Schedule()
        schedule.schedule_operation(op1.id, 0)
        schedule.schedule_operation(op2.id, 0)
        
        detector = HazardDetector()
        hazards = detector.detect_hazards(graph, schedule)
        
        # No hazards for independent ops
        assert len(hazards) == 0


class TestPipelineScheduler:
    """Tests for pipeline scheduling."""
    
    def test_simple_schedule(self):
        """Test scheduling simple graph."""
        graph = DependencyGraph()
        op1 = graph.add_operation(OperationType.ADD, latency=1)
        op2 = graph.add_operation(OperationType.ADD, latency=1)
        graph.add_dependency(op1.id, op2.id)
        
        scheduler = PipelineScheduler()
        schedule, ii = scheduler.schedule(graph)
        
        assert ii >= 1
        assert schedule.get_time(op1.id) < schedule.get_time(op2.id)
    
    def test_target_ii(self):
        """Test scheduling with target II."""
        graph = DependencyGraph()
        graph.add_operation(OperationType.MUL, latency=1)
        graph.add_operation(OperationType.MUL, latency=1)
        
        config = PipelineConfig(target_ii=2)
        scheduler = PipelineScheduler(config)
        schedule, ii = scheduler.schedule(graph)
        
        assert ii >= 1


class TestPipelineGenerator:
    """Tests for pipeline generation."""
    
    def test_generate_simple(self):
        """Test generating simple pipeline."""
        graph = DependencyGraph()
        op = graph.add_operation(OperationType.ADD, latency=1)
        
        schedule = Schedule()
        schedule.schedule_operation(op.id, 0)
        
        generator = PipelineGenerator()
        pipeline = generator.generate(schedule, graph)
        
        assert pipeline.num_stages >= 1
    
    def test_generate_with_source_fsm(self):
        """Test generation with source FSM."""
        fsm = FSMModule(name="adder")
        fsm.add_port("a", "input", 8)
        fsm.add_port("b", "input", 8)
        fsm.add_port("sum", "output", 8)
        
        graph = DependencyGraph()
        graph.add_operation(OperationType.ADD, latency=1)
        
        schedule = Schedule()
        schedule.schedule_operation(0, 0)
        
        generator = PipelineGenerator()
        pipeline = generator.generate(schedule, graph, fsm)
        
        assert pipeline.name == "adder"
        assert len(pipeline.input_ports) == 2
        assert len(pipeline.output_ports) == 1


class TestPipelineSVGenerator:
    """Tests for pipeline SystemVerilog generation."""
    
    def test_generate_empty(self):
        """Test generating empty pipeline."""
        pipeline = Pipeline(name="test")
        
        generator = PipelineSVGenerator()
        code = generator.generate(pipeline)
        
        assert "module test_pipe" in code
        assert "valid_in" in code
        assert "valid_out" in code
    
    def test_generate_with_stages(self):
        """Test generating pipeline with stages."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage()
        pipeline.add_stage()
        
        generator = PipelineSVGenerator()
        code = generator.generate(pipeline)
        
        assert "stage0_valid" in code
        assert "stage1_valid" in code
    
    def test_generate_with_registers(self):
        """Test generating pipeline with registers."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage()
        pipeline.add_stage()
        pipeline.add_register("x_p0", 32, 0, 1, "x")
        
        generator = PipelineSVGenerator()
        code = generator.generate(pipeline)
        
        assert "x_p0" in code
        assert "Pipeline registers" in code


class TestGeneratePipeline:
    """Tests for convenience function."""
    
    def test_generate_pipeline(self):
        """Test generate_pipeline function."""
        fsm = FSMModule(name="test")
        fsm.add_state("S0")
        
        pipeline, code = generate_pipeline(fsm)
        
        assert pipeline.name == "test"
        assert "module test_pipe" in code
