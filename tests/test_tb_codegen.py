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
"""Tests for testbench generation."""

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

from zuspec.synth.sprtl.tb_codegen import (
    TVVector, TVSequence, TBGenConfig, TBGenerator, generate_testbench
)
from zuspec.synth.sprtl.fsm_ir import FSMModule


class TestTestVector:
    """Tests for TestVector class."""
    
    def test_basic_vector(self):
        """Test creating a basic test vector."""
        vec = TVVector(cycle=0, inputs={"a": 1}, outputs={"b": 2})
        assert vec.cycle == 0
        assert vec.inputs["a"] == 1
        assert vec.outputs["b"] == 2
    
    def test_vector_with_comment(self):
        """Test vector with comment."""
        vec = TVVector(cycle=5, comment="Test increment")
        assert vec.cycle == 5
        assert vec.comment == "Test increment"


class TestTestSequence:
    """Tests for TestSequence class."""
    
    def test_basic_sequence(self):
        """Test creating a test sequence."""
        seq = TVSequence(name="test1", description="First test")
        assert seq.name == "test1"
        assert seq.description == "First test"
        assert len(seq.vectors) == 0
    
    def test_add_vector(self):
        """Test adding vectors to sequence."""
        seq = TVSequence(name="test")
        seq.add_vector(0, inputs={"en": 1})
        seq.add_vector(1, outputs={"count": 1})
        
        assert len(seq.vectors) == 2
        assert seq.vectors[0].inputs["en"] == 1
        assert seq.vectors[1].outputs["count"] == 1


class TestTBGenConfig:
    """Tests for testbench configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TBGenConfig()
        assert config.clock_period_ns == 10
        assert config.timescale == "1ns/1ps"
        assert config.dump_waves == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TBGenConfig(
            clock_period_ns=20,
            dump_format="fst",
            verbose_display=False
        )
        assert config.clock_period_ns == 20
        assert config.dump_format == "fst"
        assert config.verbose_display == False


class TestTestbenchGeneratorBasic:
    """Basic tests for testbench generation."""
    
    def test_empty_fsm(self):
        """Test generating testbench for minimal FSM."""
        fsm = FSMModule(name="minimal")
        
        generator = TBGenerator()
        code = generator.generate(fsm)
        
        assert "module minimal_tb" in code
        assert "endmodule" in code
        assert "logic clk" in code
        assert "logic rst_n" in code
    
    def test_clock_generation(self):
        """Test clock generation code."""
        fsm = FSMModule(name="test")
        
        config = TBGenConfig(clock_period_ns=10)
        generator = TBGenerator(config)
        code = generator.generate(fsm)
        
        assert "initial clk = 0" in code
        assert "always #5 clk = ~clk" in code  # 10ns period -> 5ns half
    
    def test_dut_instantiation(self):
        """Test DUT instantiation."""
        fsm = FSMModule(name="counter")
        fsm.add_port("enable", "input", 1)
        fsm.add_port("count", "output", 8)
        
        generator = TBGenerator()
        code = generator.generate(fsm)
        
        assert "counter dut" in code
        assert ".clk(clk)" in code
        assert ".rst_n(rst_n)" in code
        assert ".enable(enable)" in code
        assert ".count(count)" in code


class TestTestbenchGeneratorSignals:
    """Tests for signal declarations."""
    
    def test_input_signals(self):
        """Test input signal declarations."""
        fsm = FSMModule(name="test")
        fsm.add_port("data_in", "input", 8)
        fsm.add_port("enable", "input", 1)
        
        generator = TBGenerator()
        code = generator.generate(fsm)
        
        assert "logic [7:0] data_in" in code
        assert "logic enable" in code
    
    def test_output_signals(self):
        """Test output signal declarations."""
        fsm = FSMModule(name="test")
        fsm.add_port("data_out", "output", 16)
        fsm.add_port("valid", "output", 1)
        
        generator = TBGenerator()
        code = generator.generate(fsm)
        
        assert "logic [15:0] data_out" in code
        assert "logic valid" in code


class TestTestbenchGeneratorWaves:
    """Tests for waveform dump generation."""
    
    def test_vcd_dump(self):
        """Test VCD waveform dump."""
        fsm = FSMModule(name="test")
        
        config = TBGenConfig(dump_waves=True, dump_format="vcd")
        generator = TBGenerator(config)
        code = generator.generate(fsm)
        
        assert '$dumpfile("test_tb.vcd")' in code
        assert "$dumpvars(0, dut)" in code
    
    def test_fst_dump(self):
        """Test FST waveform dump."""
        fsm = FSMModule(name="test")
        
        config = TBGenConfig(dump_waves=True, dump_format="fst")
        generator = TBGenerator(config)
        code = generator.generate(fsm)
        
        assert '$dumpfile("test_tb.fst")' in code
    
    def test_no_dump(self):
        """Test disabling waveform dump."""
        fsm = FSMModule(name="test")
        
        config = TBGenConfig(dump_waves=False)
        generator = TBGenerator(config)
        code = generator.generate(fsm)
        
        assert "$dumpfile" not in code


class TestTestbenchGeneratorTests:
    """Tests for test sequence generation."""
    
    def test_reset_task(self):
        """Test reset task generation."""
        fsm = FSMModule(name="test")
        
        generator = TBGenerator()
        code = generator.generate(fsm)
        
        assert "task automatic do_reset" in code
        assert "rst_n = 0" in code
        assert "rst_n = 1" in code
    
    def test_custom_test_sequence(self):
        """Test with custom test sequence."""
        fsm = FSMModule(name="counter")
        fsm.add_port("inc_en", "input", 1)
        fsm.add_port("count", "output", 8, reset_value=0)
        
        test = TVSequence(name="increment", description="Test counter increment")
        test.add_vector(0, inputs={"inc_en": 0}, comment="Initial")
        test.add_vector(1, inputs={"inc_en": 1}, outputs={"count": 1})
        test.add_vector(2, inputs={"inc_en": 1}, outputs={"count": 2})
        
        generator = TBGenerator()
        code = generator.generate(fsm, [test])
        
        assert "task automatic test_increment" in code
        assert "// Initial" in code
        assert "inc_en = 0" in code
        assert "inc_en = 1" in code
    
    def test_assertions(self):
        """Test assertion generation."""
        fsm = FSMModule(name="test")
        fsm.add_port("out", "output", 8)
        
        test = TVSequence(name="check")
        test.add_vector(0, outputs={"out": 42})
        
        config = TBGenConfig(use_assertions=True)
        generator = TBGenerator(config)
        code = generator.generate(fsm, [test])
        
        assert "assert (out == 42)" in code
    
    def test_no_assertions(self):
        """Test without SVA assertions (if-else style)."""
        fsm = FSMModule(name="test")
        fsm.add_port("out", "output", 8)
        
        test = TVSequence(name="check")
        test.add_vector(0, outputs={"out": 42})
        
        config = TBGenConfig(use_assertions=False)
        generator = TBGenerator(config)
        code = generator.generate(fsm, [test])
        
        assert "if (out !== 42)" in code
        assert "assert" not in code


class TestTestbenchGeneratorIntegration:
    """Integration tests for testbench generation."""
    
    def test_counter_testbench(self):
        """Generate complete testbench for counter."""
        fsm = FSMModule(name="counter")
        fsm.add_port("inc_en", "input", 1)
        fsm.add_port("dec_en", "input", 1)
        fsm.add_port("count", "output", 32, reset_value=0)
        
        test = TVSequence(name="basic", description="Basic counter test")
        test.add_vector(0, inputs={"inc_en": 0, "dec_en": 0})
        test.add_vector(1, inputs={"inc_en": 1, "dec_en": 0})
        test.add_vector(2, inputs={"inc_en": 1, "dec_en": 0})
        test.add_vector(3, inputs={"inc_en": 0, "dec_en": 0})
        
        code = generate_testbench(fsm, [test])
        
        # Verify key elements
        assert "module counter_tb" in code
        assert "counter dut" in code
        assert "task automatic test_basic" in code
        assert "do_reset" in code
        assert "$finish" in code
        assert "endmodule" in code
        
        print("\n=== Generated Testbench ===")
        print(code)
    
    def test_convenience_function(self):
        """Test generate_testbench convenience function."""
        fsm = FSMModule(name="test")
        
        code = generate_testbench(fsm)
        
        assert "module test_tb" in code
        assert "test_smoke_test" in code  # Default test
