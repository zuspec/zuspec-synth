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
"""Tests for SVA assertion generation."""

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

from zuspec.synth.sprtl.sva_gen import (
    SVAGenerator, SVAGenConfig, SVAProperty, AssertionType, PropertyKind,
    generate_sva, _ConstraintExprToSV,
)
from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMStateKind
from zuspec.dataclasses.decorators import constraint as _constraint


class TestSVAGenConfig:
    """Tests for SVA generation configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SVAGenConfig()
        assert config.generate_state_assertions == True
        assert config.generate_coverage == True
        assert config.clock_name == "clk"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SVAGenConfig(
            generate_coverage=False,
            clock_name="sys_clk",
            reset_active_low=False
        )
        assert config.generate_coverage == False
        assert config.clock_name == "sys_clk"
        assert config.reset_active_low == False


class TestSVAProperty:
    """Tests for SVA property data structure."""
    
    def test_property_creation(self):
        """Test creating an SVA property."""
        prop = SVAProperty(
            name="test_prop",
            expression="@(posedge clk) a |-> b",
            assertion_type=AssertionType.ASSERT,
            kind=PropertyKind.SAFETY
        )
        assert prop.name == "test_prop"
        assert prop.assertion_type == AssertionType.ASSERT


class TestSVAGeneratorBasic:
    """Basic tests for SVA generation."""
    
    def test_empty_fsm(self):
        """Test generating SVA for empty FSM."""
        fsm = FSMModule(name="empty")
        
        generator = SVAGenerator()
        code = generator.generate(fsm)
        
        assert "SVA Assertions for: empty" in code
    
    def test_single_state(self):
        """Test SVA for single-state FSM."""
        fsm = FSMModule(name="single")
        fsm.add_state("IDLE")
        
        generator = SVAGenerator()
        code = generator.generate(fsm)
        
        assert "valid_state" in code
        assert "state == IDLE" in code


class TestSVAGeneratorStateAssertions:
    """Tests for state-related assertions."""
    
    def test_valid_state_assertion(self):
        """Test valid state assertion generation."""
        fsm = FSMModule(name="test")
        fsm.add_state("S0")
        fsm.add_state("S1")
        fsm.add_state("S2")
        
        generator = SVAGenerator()
        code = generator.generate(fsm)
        
        assert "valid_state" in code
        assert "state == S0" in code
        assert "state == S1" in code
        assert "state == S2" in code
    
    def test_reset_assertion(self):
        """Test reset brings FSM to initial state."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("INIT")
        fsm.add_state("RUN")
        fsm.initial_state = s0.id
        
        generator = SVAGenerator()
        code = generator.generate(fsm)
        
        assert "reset_initial_state" in code
        assert "state == INIT" in code


class TestSVAGeneratorCoverage:
    """Tests for coverage property generation."""
    
    def test_state_coverage(self):
        """Test coverage properties for each state."""
        fsm = FSMModule(name="test")
        fsm.add_state("IDLE")
        fsm.add_state("RUN")
        
        config = SVAGenConfig(generate_coverage=True)
        generator = SVAGenerator(config)
        code = generator.generate(fsm)
        
        assert "cover_state_IDLE" in code
        assert "cover_state_RUN" in code
        assert "cover property" in code
    
    def test_transition_coverage(self):
        """Test coverage properties for transitions."""
        fsm = FSMModule(name="test")
        s0 = fsm.add_state("IDLE")
        s1 = fsm.add_state("RUN")
        s0.add_transition(s1.id)
        s1.add_transition(s0.id)
        
        config = SVAGenConfig(generate_coverage=True)
        generator = SVAGenerator(config)
        code = generator.generate(fsm)
        
        assert "cover_trans_IDLE_to_RUN" in code
        assert "cover_trans_RUN_to_IDLE" in code
    
    def test_no_coverage(self):
        """Test disabling coverage generation."""
        fsm = FSMModule(name="test")
        fsm.add_state("IDLE")
        
        config = SVAGenConfig(generate_coverage=False)
        generator = SVAGenerator(config)
        code = generator.generate(fsm)
        
        assert "cover property" not in code


class TestSVAGeneratorOutputAssertions:
    """Tests for output-related assertions."""
    
    def test_output_reset_value(self):
        """Test assertions for output reset values."""
        fsm = FSMModule(name="test")
        fsm.add_port("count", "output", 8, reset_value=0)
        fsm.add_state("IDLE")
        
        config = SVAGenConfig(generate_output_assertions=True)
        generator = SVAGenerator(config)
        code = generator.generate(fsm)
        
        assert "reset_count" in code
        assert "count == 0" in code


class TestSVAGeneratorIntegration:
    """Integration tests for SVA generation."""
    
    def test_counter_sva(self):
        """Generate SVA for counter FSM."""
        fsm = FSMModule(name="counter")
        fsm.add_port("count", "output", 8, reset_value=0)
        
        s = fsm.add_state("RUN")
        s.add_transition(s.id)  # Loop
        
        code = generate_sva(fsm)
        
        assert "SVA Assertions for: counter" in code
        assert "valid_state" in code
        assert "cover_state_RUN" in code
        
        print("\n=== Generated SVA ===")
        print(code)
    
    def test_get_properties(self):
        """Test retrieving generated properties."""
        fsm = FSMModule(name="test")
        fsm.add_state("S0")
        fsm.add_state("S1")
        
        generator = SVAGenerator()
        generator.generate(fsm)
        
        props = generator.get_properties()
        assert len(props) > 0
        
        # Should have at least valid_state assertion
        names = [p.name for p in props]
        assert "valid_state" in names


# ---------------------------------------------------------------------------
# T-P7: generate_role_properties — contract roles → SVA assume/assert
# ---------------------------------------------------------------------------

class SimpleReqAction:
    @_constraint.requires
    def c_positive(self):
        self.value > 0

    @_constraint.ensures
    def c_result_bounded(self):
        self.value < 200


class MultiRoleAction:
    @_constraint.requires
    def req_a_small(self):
        self.a < 128

    @_constraint.requires
    def req_b_nonzero(self):
        self.b > 0

    @_constraint.ensures
    def ens_sum_bounded(self):
        self.a < 100


class NoRoleAction:
    @_constraint
    def c_x(self):
        self.x < 50


class TestGenerateRoleProperties:
    """T-P7: SVAGenerator.generate_role_properties() tests."""

    def setup_method(self):
        self.gen = SVAGenerator()

    def test_requires_maps_to_assume(self):
        props = self.gen.generate_role_properties(SimpleReqAction)
        assume_props = [p for p in props if p.assertion_type == AssertionType.ASSUME]
        assert len(assume_props) == 1
        assert assume_props[0].name == "requires_c_positive"

    def test_ensures_maps_to_assert(self):
        props = self.gen.generate_role_properties(SimpleReqAction)
        assert_props = [p for p in props if p.assertion_type == AssertionType.ASSERT]
        assert len(assert_props) == 1
        assert assert_props[0].name == "ensures_c_result_bounded"

    def test_multiple_requires(self):
        props = self.gen.generate_role_properties(MultiRoleAction)
        assume_props = [p for p in props if p.assertion_type == AssertionType.ASSUME]
        names = {p.name for p in assume_props}
        assert "requires_req_a_small" in names
        assert "requires_req_b_nonzero" in names

    def test_expression_contains_clock(self):
        props = self.gen.generate_role_properties(SimpleReqAction, clock="sys_clk")
        for p in props:
            assert "posedge sys_clk" in p.expression

    def test_trigger_added_to_expression(self):
        props = self.gen.generate_role_properties(
            SimpleReqAction, trigger="action_valid"
        )
        for p in props:
            assert "action_valid |-> " in p.expression

    def test_no_trigger_no_implication(self):
        props = self.gen.generate_role_properties(SimpleReqAction)
        for p in props:
            assert "|-> " not in p.expression

    def test_no_role_constraints_returns_empty(self):
        props = self.gen.generate_role_properties(NoRoleAction)
        assert props == []

    def test_description_contains_class_and_method(self):
        props = self.gen.generate_role_properties(SimpleReqAction)
        for p in props:
            assert "SimpleReqAction" in p.description

    def test_property_kind_is_safety(self):
        props = self.gen.generate_role_properties(SimpleReqAction)
        for p in props:
            assert p.kind == PropertyKind.SAFETY

    def test_expression_contains_field_name(self):
        props = self.gen.generate_role_properties(SimpleReqAction)
        exprs = [p.expression for p in props]
        assert any("value" in e for e in exprs)


class TestConstraintExprToSV:
    """Unit tests for _ConstraintExprToSV translator."""

    def _tr(self, node):
        return _ConstraintExprToSV().translate(node)

    def test_attribute(self):
        assert self._tr({'type': 'attribute', 'value': 'self', 'attr': 'foo'}) == 'foo'

    def test_constant(self):
        assert self._tr({'type': 'constant', 'value': 42}) == '42'

    def test_compare_eq(self):
        node = {
            'type': 'compare',
            'left': {'type': 'attribute', 'attr': 'x'},
            'ops': ['=='],
            'comparators': [{'type': 'constant', 'value': 5}],
        }
        result = self._tr(node)
        assert result == '(x == 5)'

    def test_compare_lt(self):
        node = {
            'type': 'compare',
            'left': {'type': 'attribute', 'attr': 'v'},
            'ops': ['<'],
            'comparators': [{'type': 'constant', 'value': 100}],
        }
        assert '< 100' in self._tr(node)

    def test_bool_op_and(self):
        node = {
            'type': 'bool_op',
            'op': 'and',
            'values': [
                {'type': 'attribute', 'attr': 'a'},
                {'type': 'attribute', 'attr': 'b'},
            ],
        }
        result = self._tr(node)
        assert '&&' in result

    def test_unary_not(self):
        node = {
            'type': 'unary_op',
            'op': 'not',
            'operand': {'type': 'attribute', 'attr': 'en'},
        }
        result = self._tr(node)
        assert '!en' in result

    def test_bad_node_returns_none(self):
        assert self._tr({'type': 'unknown_xyz'}) is None
