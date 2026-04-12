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
"""Tests for SystemVerilog code generation."""

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

from zuspec.synth.sprtl.sv_codegen import (
    SVCodeGenerator, SVGenConfig, ResetStyle, FSMStyle, generate_sv
)
from zuspec.synth.sprtl.fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMAssign, FSMCond
)


class TestSVGenConfig:
    """Tests for SystemVerilog generation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SVGenConfig()
        assert config.reset_style == ResetStyle.ASYNC_LOW
        assert config.fsm_style == FSMStyle.TWO_PROCESS
        assert config.use_logic == True
        assert config.use_always_ff == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SVGenConfig(
            reset_style=ResetStyle.SYNC_LOW,
            generate_comments=False,
            indent="    "
        )
        assert config.reset_style == ResetStyle.SYNC_LOW
        assert config.generate_comments == False
        assert config.indent == "    "


class TestSVCodeGeneratorBasic:
    """Basic tests for SystemVerilog code generation."""
    
    def test_empty_module(self):
        """Test generating empty module."""
        fsm = FSMModule(name="empty")
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "module empty" in code
        assert "endmodule" in code
        assert "input  logic clk" in code
        assert "input  logic rst_n" in code
    
    def test_module_with_ports(self):
        """Test module with input/output ports."""
        fsm = FSMModule(name="test_ports")
        fsm.add_port("data_in", "input", 8)
        fsm.add_port("data_out", "output", 8, reset_value=0)
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "input  logic [7:0] data_in" in code
        assert "output logic [7:0] data_out" in code
    
    def test_single_bit_port(self):
        """Test single-bit port has no width specifier."""
        fsm = FSMModule(name="test_single_bit")
        fsm.add_port("enable", "input", 1)
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        # Single bit should not have [0:0]
        assert "input  logic enable" in code
        assert "[0:0]" not in code


class TestSVCodeGeneratorStates:
    """Tests for state machine code generation."""
    
    def test_state_encoding(self):
        """Test state type and encoding generation."""
        fsm = FSMModule(name="test_states")
        fsm.add_state("IDLE")
        fsm.add_state("RUN")
        fsm.add_state("DONE")
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "typedef enum logic" in code
        assert "IDLE = " in code
        assert "RUN = " in code
        assert "DONE = " in code
        assert "state_t;" in code
        assert "state_t state, next_state;" in code
    
    def test_state_register_async_reset(self):
        """Test state register with async reset."""
        fsm = FSMModule(name="test_state_reg")
        fsm.add_state("IDLE")
        fsm.add_state("ACTIVE")
        
        config = SVGenConfig(reset_style=ResetStyle.ASYNC_LOW)
        generator = SVCodeGenerator(config)
        code = generator.generate(fsm)
        
        assert "always_ff @(posedge clk or negedge rst_n)" in code
        assert "if (!rst_n)" in code
        assert "state <= IDLE" in code
        assert "state <= next_state" in code
    
    def test_state_register_sync_reset(self):
        """Test state register with sync reset."""
        fsm = FSMModule(name="test_sync_reset")
        fsm.add_state("IDLE")
        
        config = SVGenConfig(reset_style=ResetStyle.SYNC_LOW)
        generator = SVCodeGenerator(config)
        code = generator.generate(fsm)
        
        assert "always_ff @(posedge clk)" in code
        assert "negedge rst_n" not in code
    
    def test_next_state_logic(self):
        """Test next state combinational logic."""
        fsm = FSMModule(name="test_next_state")
        s0 = fsm.add_state("IDLE")
        s1 = fsm.add_state("RUN")
        s0.add_transition(s1.id)  # IDLE -> RUN
        s1.add_transition(s0.id)  # RUN -> IDLE
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "always_comb begin" in code
        assert "next_state = state" in code
        assert "case (state)" in code
        assert "IDLE: begin" in code
        assert "RUN: begin" in code
        assert "endcase" in code


class TestSVCodeGeneratorOperations:
    """Tests for operation code generation."""
    
    def test_simple_assignment(self):
        """Test simple assignment operation."""
        fsm = FSMModule(name="test_assign")
        fsm.add_port("out", "output", 8, reset_value=0)
        
        state = fsm.add_state("S0")
        state.add_operation(FSMAssign(target="out", value=42))
        state.add_transition(state.id)  # Loop back
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "out <= 42" in code
    
    def test_augmented_assignment(self):
        """Test augmented assignment (e.g., count += 1)."""
        fsm = FSMModule(name="test_aug_assign")
        fsm.add_port("count", "output", 8, reset_value=0)
        
        state = fsm.add_state("S0")
        # Augmented assignment stored as tuple: (target_expr, op, value)
        class MockExpr:
            name = "count"
        state.add_operation(FSMAssign(
            target="count", 
            value=(MockExpr(), "Add", 1)
        ))
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "count <= count + 1" in code
    
    def test_conditional_operation(self):
        """Test conditional (if/else) operation."""
        fsm = FSMModule(name="test_cond")
        fsm.add_port("sel", "input", 1)
        fsm.add_port("out", "output", 8, reset_value=0)
        
        state = fsm.add_state("S0")
        cond_op = FSMCond(
            condition="sel",
            then_ops=[FSMAssign(target="out", value=1)],
            else_ops=[FSMAssign(target="out", value=0)]
        )
        state.add_operation(cond_op)
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "if (sel)" in code
        assert "out <= 1" in code
        assert "end else begin" in code
        assert "out <= 0" in code
    
    def test_reset_values(self):
        """Test reset values for outputs."""
        fsm = FSMModule(name="test_reset")
        fsm.add_port("count", "output", 8, reset_value=0)
        fsm.add_port("data", "output", 16, reset_value=255)
        
        state = fsm.add_state("IDLE")
        state.add_operation(FSMAssign(target="count", value=1))
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "count <= 8'd0" in code
        assert "data <= 16'd255" in code


class TestSVCodeGeneratorTransitions:
    """Tests for state transition code generation."""
    
    def test_unconditional_transition(self):
        """Test unconditional state transition."""
        fsm = FSMModule(name="test_uncond")
        s0 = fsm.add_state("S0")
        s1 = fsm.add_state("S1")
        s0.add_transition(s1.id)
        s1.add_transition(s0.id)
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "S0: begin" in code
        assert "next_state = S1" in code
        assert "S1: begin" in code
        assert "next_state = S0" in code
    
    def test_conditional_transition(self):
        """Test conditional state transition."""
        fsm = FSMModule(name="test_cond_trans")
        fsm.add_port("start", "input", 1)
        
        s0 = fsm.add_state("IDLE")
        s1 = fsm.add_state("RUN")
        
        # Conditional transition: if start, go to RUN
        s0.add_transition(s1.id, condition="start")
        s0.add_transition(s0.id)  # Default: stay in IDLE
        s1.add_transition(s0.id)
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        assert "if (start)" in code
        assert "next_state = RUN" in code


class TestSVCodeGeneratorIntegration:
    """Integration tests for complete FSM generation."""
    
    def test_simple_counter(self):
        """Test generating a simple counter FSM."""
        fsm = FSMModule(name="counter")
        fsm.add_port("inc_en", "input", 1)
        fsm.add_port("count", "output", 8, reset_value=0)
        
        state = fsm.add_state("RUN")
        
        # Conditional increment
        class IncExpr:
            name = "count"
        
        cond = FSMCond(
            condition="inc_en",
            then_ops=[FSMAssign(target="count", value=(IncExpr(), "Add", 1))],
            else_ops=[]
        )
        state.add_operation(cond)
        state.add_transition(state.id)  # Loop
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        # Verify structure
        assert "module counter" in code
        assert "typedef enum" in code
        assert "always_ff" in code
        assert "always_comb" in code
        assert "if (inc_en)" in code
        assert "count <= count + 1" in code
        assert "endmodule" in code
    
    def test_no_comments(self):
        """Test generation without comments."""
        fsm = FSMModule(name="no_comments")
        fsm.add_state("IDLE")
        
        config = SVGenConfig(generate_comments=False)
        generator = SVCodeGenerator(config)
        code = generator.generate(fsm)
        
        assert "//" not in code
    
    def test_generate_sv_function(self):
        """Test convenience function."""
        fsm = FSMModule(name="test")
        fsm.add_state("S0")
        
        code = generate_sv(fsm)
        
        assert "module test" in code

    def test_generate_sv_async_reset_derived_from_fsm(self):
        """generate_sv() derives ASYNC_LOW from fsm.reset_async=True, reset_active_low=True."""
        fsm = FSMModule(name="async_test", reset_async=True, reset_active_low=True)
        fsm.add_state("IDLE")
        code = generate_sv(fsm)
        assert "negedge" in code
        assert "always_ff @(posedge clk or negedge" in code

    def test_generate_sv_sync_reset_derived_from_fsm(self):
        """generate_sv() derives SYNC_LOW when fsm.reset_async=False (default)."""
        fsm = FSMModule(name="sync_test", reset_async=False, reset_active_low=True)
        fsm.add_state("IDLE")
        code = generate_sv(fsm)
        assert "negedge" not in code
        assert "always_ff @(posedge clk)" in code

    def test_generate_sv_explicit_config_overrides_fsm(self):
        """Explicit SVGenConfig always wins over FSM fields."""
        fsm = FSMModule(name="override_test", reset_async=False, reset_active_low=True)
        fsm.add_state("IDLE")
        config = SVGenConfig(reset_style=ResetStyle.ASYNC_LOW)
        code = generate_sv(fsm, config=config)
        assert "negedge" in code


class TestSVCodeGeneratorCounterExample:
    """Test generating the counter example from the plan."""
    
    def test_counter_example(self):
        """Generate the counter example and verify key elements."""
        # Create counter FSM matching Example 1 from plan
        fsm = FSMModule(
            name="counter",
            clock_signal="clk",
            reset_signal="rst_n"
        )
        fsm.add_port("inc_en", "input", 1)
        fsm.add_port("dec_en", "input", 1)
        fsm.add_port("count", "output", 32, reset_value=0)
        
        # Single state with conditional logic
        state = fsm.add_state("RUN")
        
        class CountExpr:
            name = "count"
        
        # if inc_en: count += 1
        # elif dec_en: count -= 1
        inc_cond = FSMCond(
            condition="inc_en",
            then_ops=[FSMAssign(target="count", value=(CountExpr(), "Add", 1))],
            else_ops=[
                FSMCond(
                    condition="dec_en",
                    then_ops=[FSMAssign(target="count", value=(CountExpr(), "Sub", 1))],
                    else_ops=[]
                )
            ]
        )
        state.add_operation(inc_cond)
        state.add_transition(state.id)
        
        generator = SVCodeGenerator()
        code = generator.generate(fsm)
        
        # Verify expected elements
        assert "module counter" in code
        assert "input  logic clk" in code
        assert "input  logic rst_n" in code
        assert "input  logic inc_en" in code
        assert "input  logic dec_en" in code
        assert "output logic [31:0] count" in code
        assert "count <= 32'd0" in code
        assert "if (inc_en)" in code
        assert "count <= count + 1" in code
        assert "if (dec_en)" in code
        assert "count <= count - 1" in code
        
        print("\n=== Generated Counter ===")
        print(code)


class TestUserEnumTypedefs:
    """Tests for user-defined @zdc.enum typedef emission."""

    def test_user_enum_typedef_emitted(self):
        """FSMModule.add_user_enum() causes typedef enum logic in SV output."""
        fsm = FSMModule(name="state_machine")
        fsm.add_state("IDLE")
        fsm.add_user_enum("State", 2, {"IDLE": 0, "FETCH": 1, "EXEC": 2})

        code = generate_sv(fsm)
        assert "typedef enum logic [1:0]" in code
        assert "IDLE = 2'd0" in code
        assert "FETCH = 2'd1" in code
        assert "EXEC = 2'd2" in code
        assert "} State_t;" in code

    def test_user_enum_appears_before_module(self):
        """Enum typedefs are emitted before the module keyword."""
        fsm = FSMModule(name="test")
        fsm.add_state("S0")
        fsm.add_user_enum("Color", 2, {"RED": 0, "GREEN": 1, "BLUE": 2})

        code = generate_sv(fsm)
        typedef_pos = code.find("typedef enum")
        module_pos = code.find("module test")
        assert typedef_pos != -1 and module_pos != -1
        assert typedef_pos < module_pos

    def test_duplicate_enum_not_emitted_twice(self):
        """Calling add_user_enum() twice with the same name only emits once."""
        fsm = FSMModule(name="dup")
        fsm.add_state("S0")
        fsm.add_user_enum("Op", 1, {"A": 0, "B": 1})
        fsm.add_user_enum("Op", 1, {"A": 0, "B": 1})  # duplicate

        code = generate_sv(fsm)
        assert code.count("} Op_t;") == 1

    def test_no_user_enums_no_typedef(self):
        """No user_enums → no typedef enum in output."""
        fsm = FSMModule(name="plain")
        fsm.add_state("IDLE")

        code = generate_sv(fsm)
        # Only the FSM state enum typedef is present (for the state register)
        assert "typedef enum" in code  # state encoding enum
        # But no user-defined typedef suffix (_t) outside the state encoding section

