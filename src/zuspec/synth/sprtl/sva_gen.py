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
SVA Assertion Generator: Generates SystemVerilog Assertions for verification.

This module generates SVA properties and assertions that can be used to:
1. Verify FSM behavior matches specification
2. Check safety properties (no deadlock, valid state encoding)
3. Check liveness properties (eventually reaches state)
4. Generate cover properties for coverage analysis

The generated assertions can be used with:
- Simulation-based verification (Verilator, VCS, etc.)
- Formal verification tools (JasperGold, SymbiYosys, etc.)
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO

from .fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMTransition,
    FSMOperation, FSMAssign, FSMCond, FSMPort
)


class AssertionType(Enum):
    """Types of SVA assertions."""
    ASSERT = auto()      # assert property - must hold
    ASSUME = auto()      # assume property - constrain inputs
    COVER = auto()       # cover property - reachability


class PropertyKind(Enum):
    """Kinds of temporal properties."""
    SAFETY = auto()      # Something bad never happens
    LIVENESS = auto()    # Something good eventually happens
    FAIRNESS = auto()    # Something happens infinitely often


@dataclass
class SVAProperty:
    """An SVA property definition.
    
    Attributes:
        name: Property name
        description: Human-readable description
        expression: SVA expression
        assertion_type: ASSERT, ASSUME, or COVER
        kind: SAFETY, LIVENESS, or FAIRNESS
    """
    name: str
    expression: str
    assertion_type: AssertionType = AssertionType.ASSERT
    kind: PropertyKind = PropertyKind.SAFETY
    description: str = ""


@dataclass
class SVAGenConfig:
    """Configuration for SVA generation.
    
    Attributes:
        generate_state_assertions: Generate state-related assertions
        generate_output_assertions: Generate output validity assertions
        generate_coverage: Generate cover properties
        generate_fairness: Generate fairness constraints
        clock_name: Name of clock signal
        reset_name: Name of reset signal
        reset_active_low: True if reset is active low
    """
    generate_state_assertions: bool = True
    generate_output_assertions: bool = True
    generate_coverage: bool = True
    generate_fairness: bool = False
    clock_name: str = "clk"
    reset_name: str = "rst_n"
    reset_active_low: bool = True


class SVAGenerator:
    """Generates SystemVerilog Assertions from FSM IR.
    
    Usage:
        generator = SVAGenerator()
        sva_code = generator.generate(fsm_module)
    """
    
    def __init__(self, config: Optional[SVAGenConfig] = None):
        self.config = config or SVAGenConfig()
        self._output = StringIO()
        self._indent_level = 0
        self._properties: List[SVAProperty] = []
    
    def generate(self, fsm: FSMModule) -> str:
        """Generate SVA assertions for an FSM module.
        
        Args:
            fsm: FSM module to generate assertions for
            
        Returns:
            SystemVerilog assertion code
        """
        self._output = StringIO()
        self._indent_level = 0
        self._properties = []
        
        # Collect properties
        if self.config.generate_state_assertions:
            self._generate_state_properties(fsm)
        
        if self.config.generate_output_assertions:
            self._generate_output_properties(fsm)
        
        if self.config.generate_coverage:
            self._generate_coverage_properties(fsm)
        
        if self.config.generate_fairness:
            self._generate_fairness_properties(fsm)
        
        # Generate output
        self._emit_header(fsm)
        self._emit_properties()
        
        return self._output.getvalue()
    
    def get_properties(self) -> List[SVAProperty]:
        """Get the list of generated properties."""
        return self._properties
    
    def _emit(self, text: str = ""):
        self._output.write(text)
    
    def _emitln(self, text: str = ""):
        self._emit("  " * self._indent_level + text + "\n")
    
    def _indent(self):
        self._indent_level += 1
    
    def _dedent(self):
        self._indent_level = max(0, self._indent_level - 1)
    
    def _reset_condition(self) -> str:
        """Get the reset active condition."""
        if self.config.reset_active_low:
            return self.config.reset_name
        return f"!{self.config.reset_name}"
    
    def _generate_state_properties(self, fsm: FSMModule):
        """Generate state-related properties."""
        if not fsm.states:
            return
        
        # Property: Valid state encoding
        state_names = [s.name for s in fsm.states]
        valid_states = " || ".join([f"(state == {s})" for s in state_names])
        self._properties.append(SVAProperty(
            name="valid_state",
            description="State is always one of the valid states",
            expression=f"@(posedge {self.config.clock_name}) disable iff (!{self._reset_condition()}) ({valid_states})",
            assertion_type=AssertionType.ASSERT,
            kind=PropertyKind.SAFETY
        ))
        
        # Property: No stuck in unknown state
        self._properties.append(SVAProperty(
            name="no_stuck_state",
            description="FSM never gets stuck (always has valid transition)",
            expression=f"@(posedge {self.config.clock_name}) disable iff (!{self._reset_condition()}) (state == state)",
            assertion_type=AssertionType.ASSERT,
            kind=PropertyKind.SAFETY
        ))
        
        # Property: Reset brings FSM to initial state
        initial_state = fsm.get_state(fsm.initial_state)
        if initial_state:
            reset_cond = f"!{self.config.reset_name}" if self.config.reset_active_low else self.config.reset_name
            self._properties.append(SVAProperty(
                name="reset_initial_state",
                description="Reset brings FSM to initial state",
                expression=f"@(posedge {self.config.clock_name}) ({reset_cond}) |=> (state == {initial_state.name})",
                assertion_type=AssertionType.ASSERT,
                kind=PropertyKind.SAFETY
            ))
    
    def _generate_output_properties(self, fsm: FSMModule):
        """Generate output-related properties."""
        output_ports = fsm.get_output_ports()
        
        for port in output_ports:
            if port.reset_value is not None:
                # Property: Output has correct reset value
                reset_cond = f"!{self.config.reset_name}" if self.config.reset_active_low else self.config.reset_name
                self._properties.append(SVAProperty(
                    name=f"reset_{port.name}",
                    description=f"{port.name} has correct reset value",
                    expression=f"@(posedge {self.config.clock_name}) ({reset_cond}) |=> ({port.name} == {port.reset_value})",
                    assertion_type=AssertionType.ASSERT,
                    kind=PropertyKind.SAFETY
                ))
    
    def _generate_coverage_properties(self, fsm: FSMModule):
        """Generate coverage properties."""
        # Cover: Each state is reachable
        for state in fsm.states:
            self._properties.append(SVAProperty(
                name=f"cover_state_{state.name}",
                description=f"State {state.name} is reachable",
                expression=f"@(posedge {self.config.clock_name}) disable iff (!{self._reset_condition()}) (state == {state.name})",
                assertion_type=AssertionType.COVER,
                kind=PropertyKind.LIVENESS
            ))
        
        # Cover: Each transition is taken
        for state in fsm.states:
            for i, trans in enumerate(state.transitions):
                target = fsm.get_state(trans.target_state)
                if target and target.id != state.id:  # Skip self-loops for brevity
                    self._properties.append(SVAProperty(
                        name=f"cover_trans_{state.name}_to_{target.name}",
                        description=f"Transition from {state.name} to {target.name}",
                        expression=f"@(posedge {self.config.clock_name}) disable iff (!{self._reset_condition()}) (state == {state.name}) ##1 (state == {target.name})",
                        assertion_type=AssertionType.COVER,
                        kind=PropertyKind.LIVENESS
                    ))
    
    def _generate_fairness_properties(self, fsm: FSMModule):
        """Generate fairness constraints for liveness proofs."""
        # Fairness: Reset doesn't stay asserted forever
        reset_cond = f"!{self.config.reset_name}" if self.config.reset_active_low else self.config.reset_name
        self._properties.append(SVAProperty(
            name="fair_reset",
            description="Reset is eventually deasserted",
            expression=f"@(posedge {self.config.clock_name}) ({reset_cond}) |-> ##[1:$] ({self._reset_condition()})",
            assertion_type=AssertionType.ASSUME,
            kind=PropertyKind.FAIRNESS
        ))
    
    def _emit_header(self, fsm: FSMModule):
        self._emitln("//")
        self._emitln(f"// SVA Assertions for: {fsm.name}")
        self._emitln("// Generated by zuspec-synth SPRTL compiler")
        self._emitln("//")
        self._emitln()
    
    def _emit_properties(self):
        # Group by assertion type
        asserts = [p for p in self._properties if p.assertion_type == AssertionType.ASSERT]
        assumes = [p for p in self._properties if p.assertion_type == AssertionType.ASSUME]
        covers = [p for p in self._properties if p.assertion_type == AssertionType.COVER]
        
        if assumes:
            self._emitln("// Assumptions (input constraints)")
            for prop in assumes:
                if prop.description:
                    self._emitln(f"// {prop.description}")
                self._emitln(f"assume property ({prop.name});")
                self._emitln(f"property {prop.name};")
                self._indent()
                self._emitln(f"{prop.expression};")
                self._dedent()
                self._emitln("endproperty")
                self._emitln()
        
        if asserts:
            self._emitln("// Assertions (must hold)")
            for prop in asserts:
                if prop.description:
                    self._emitln(f"// {prop.description}")
                self._emitln(f"assert property ({prop.name});")
                self._emitln(f"property {prop.name};")
                self._indent()
                self._emitln(f"{prop.expression};")
                self._dedent()
                self._emitln("endproperty")
                self._emitln()
        
        if covers:
            self._emitln("// Coverage properties")
            for prop in covers:
                if prop.description:
                    self._emitln(f"// {prop.description}")
                self._emitln(f"cover property ({prop.name});")
                self._emitln(f"property {prop.name};")
                self._indent()
                self._emitln(f"{prop.expression};")
                self._dedent()
                self._emitln("endproperty")
                self._emitln()


def generate_sva(fsm: FSMModule, config: Optional[SVAGenConfig] = None) -> str:
    """Convenience function to generate SVA assertions.
    
    Args:
        fsm: FSM module
        config: Optional configuration
        
    Returns:
        SVA assertion code
    """
    generator = SVAGenerator(config)
    return generator.generate(fsm)
