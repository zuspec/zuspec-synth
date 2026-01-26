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
Testbench Generator: Generates SystemVerilog testbenches for FSM verification.

This module generates testbenches that:
- Instantiate the DUT (Device Under Test)
- Generate clock and reset signals
- Apply stimulus based on test vectors
- Check outputs with assertions
- Generate waveform dumps (VCD/FST)
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from io import StringIO

from .fsm_ir import FSMModule, FSMPort


@dataclass
class TVVector:
    """A single test vector with inputs and expected outputs.
    
    Attributes:
        cycle: Clock cycle number (relative to start)
        inputs: Dictionary of input signal -> value
        outputs: Dictionary of expected output signal -> value (None = don't check)
        comment: Optional description of this test vector
    """
    cycle: int
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    comment: str = ""


@dataclass
class TVSequence:
    """A sequence of test vectors forming a test case.
    
    Attributes:
        name: Test case name
        description: Description of what this test verifies
        vectors: List of test vectors
        reset_cycles: Number of cycles to hold reset
    """
    name: str
    description: str = ""
    vectors: List[TVVector] = field(default_factory=list)
    reset_cycles: int = 2
    
    def add_vector(self, cycle: int, inputs: Dict[str, Any] = None,
                   outputs: Dict[str, Any] = None, comment: str = "") -> TVVector:
        """Add a test vector to the sequence."""
        vec = TVVector(
            cycle=cycle,
            inputs=inputs or {},
            outputs=outputs or {},
            comment=comment
        )
        self.vectors.append(vec)
        return vec


@dataclass
class TBGenConfig:
    """Configuration for testbench generation.
    
    Attributes:
        clock_period_ns: Clock period in nanoseconds
        timescale: Verilog timescale directive
        dump_waves: Generate VCD/FST dump
        dump_format: Waveform format ('vcd' or 'fst')
        use_assertions: Generate SVA assertions
        verbose_display: Include $display statements
    """
    clock_period_ns: int = 10
    timescale: str = "1ns/1ps"
    dump_waves: bool = True
    dump_format: str = "vcd"
    use_assertions: bool = True
    verbose_display: bool = True


class TBGenerator:
    """Generates SystemVerilog testbenches for FSM modules.
    
    Usage:
        generator = TBGenerator()
        tb_code = generator.generate(fsm, test_sequence)
    """
    
    def __init__(self, config: Optional[TBGenConfig] = None):
        self.config = config or TBGenConfig()
        self._output: StringIO = StringIO()
        self._indent_level = 0
    
    def generate(self, fsm: FSMModule, 
                 tests: Optional[List[TVSequence]] = None) -> str:
        """Generate testbench code.
        
        Args:
            fsm: FSM module to generate testbench for
            tests: Optional list of test sequences
            
        Returns:
            SystemVerilog testbench code
        """
        self._output = StringIO()
        self._indent_level = 0
        
        if tests is None:
            tests = [self._generate_default_test(fsm)]
        
        self._generate_header(fsm)
        self._generate_module_declaration(fsm)
        self._generate_signal_declarations(fsm)
        self._generate_dut_instantiation(fsm)
        self._generate_clock_generation(fsm)
        self._generate_reset_task()
        self._generate_stimulus_tasks(fsm, tests)
        self._generate_main_test(tests)
        self._generate_module_end()
        
        return self._output.getvalue()
    
    def _emit(self, text: str = ""):
        self._output.write(text)
    
    def _emitln(self, text: str = ""):
        self._emit("  " * self._indent_level + text + "\n")
    
    def _indent(self):
        self._indent_level += 1
    
    def _dedent(self):
        self._indent_level = max(0, self._indent_level - 1)
    
    def _generate_default_test(self, fsm: FSMModule) -> TVSequence:
        """Generate a default smoke test."""
        test = TVSequence(
            name="smoke_test",
            description="Basic smoke test - reset and run",
            reset_cycles=2
        )
        
        # Just let it run for a few cycles after reset
        for i in range(5):
            test.add_vector(i, comment=f"Cycle {i}")
        
        return test
    
    def _generate_header(self, fsm: FSMModule):
        self._emitln("//")
        self._emitln(f"// Testbench for: {fsm.name}")
        self._emitln("// Generated by zuspec-synth SPRTL compiler")
        self._emitln("//")
        self._emitln()
        self._emitln(f"`timescale {self.config.timescale}")
        self._emitln()
    
    def _generate_module_declaration(self, fsm: FSMModule):
        self._emitln(f"module {fsm.name}_tb;")
        self._emitln()
    
    def _generate_signal_declarations(self, fsm: FSMModule):
        self._indent()
        self._emitln("// Clock and reset")
        self._emitln(f"logic {fsm.clock_signal};")
        self._emitln(f"logic {fsm.reset_signal};")
        self._emitln()
        
        # Separate inputs and outputs
        inputs = [p for p in fsm.ports 
                 if p.direction == "input" and 
                 p.name not in (fsm.clock_signal, fsm.reset_signal)]
        outputs = [p for p in fsm.ports if p.direction == "output"]
        
        if inputs:
            self._emitln("// DUT inputs")
            for port in inputs:
                width = self._format_width(port.width)
                self._emitln(f"logic {width}{port.name};")
            self._emitln()
        
        if outputs:
            self._emitln("// DUT outputs")
            for port in outputs:
                width = self._format_width(port.width)
                self._emitln(f"logic {width}{port.name};")
            self._emitln()
        
        self._emitln("// Test control")
        self._emitln("int test_pass;")
        self._emitln("int test_fail;")
        self._emitln()
        self._dedent()
    
    def _format_width(self, width: int) -> str:
        if width == 1:
            return ""
        return f"[{width-1}:0] "
    
    def _generate_dut_instantiation(self, fsm: FSMModule):
        self._indent()
        self._emitln("// DUT instantiation")
        self._emitln(f"{fsm.name} dut (")
        self._indent()
        
        # Connect all ports
        all_ports = [fsm.clock_signal, fsm.reset_signal]
        all_ports.extend(p.name for p in fsm.ports 
                        if p.name not in (fsm.clock_signal, fsm.reset_signal))
        
        for i, port in enumerate(all_ports):
            comma = "," if i < len(all_ports) - 1 else ""
            self._emitln(f".{port}({port}){comma}")
        
        self._dedent()
        self._emitln(");")
        self._emitln()
        self._dedent()
    
    def _generate_clock_generation(self, fsm: FSMModule):
        self._indent()
        half_period = self.config.clock_period_ns // 2
        
        self._emitln("// Clock generation")
        self._emitln(f"initial {fsm.clock_signal} = 0;")
        self._emitln(f"always #{half_period} {fsm.clock_signal} = ~{fsm.clock_signal};")
        self._emitln()
        
        if self.config.dump_waves:
            self._emitln("// Waveform dump")
            self._emitln("initial begin")
            self._indent()
            if self.config.dump_format == "fst":
                self._emitln(f'$dumpfile("{fsm.name}_tb.fst");')
            else:
                self._emitln(f'$dumpfile("{fsm.name}_tb.vcd");')
            self._emitln("$dumpvars(0, dut);")
            self._dedent()
            self._emitln("end")
            self._emitln()
        
        self._dedent()
    
    def _generate_reset_task(self):
        self._indent()
        self._emitln("// Reset task")
        self._emitln("task automatic do_reset(input int cycles);")
        self._indent()
        self._emitln("rst_n = 0;")
        self._emitln("repeat (cycles) @(posedge clk);")
        self._emitln("rst_n = 1;")
        self._emitln('@(posedge clk);')
        self._dedent()
        self._emitln("endtask")
        self._emitln()
        self._dedent()
    
    def _generate_stimulus_tasks(self, fsm: FSMModule, tests: List[TVSequence]):
        self._indent()
        
        # Generate a task for each test sequence
        for test in tests:
            self._emitln(f"// Test: {test.name}")
            if test.description:
                self._emitln(f"// {test.description}")
            self._emitln(f"task automatic test_{test.name}();")
            self._indent()
            
            if self.config.verbose_display:
                self._emitln(f'$display("Running test: {test.name}");')
            
            # Initialize inputs
            inputs = [p for p in fsm.ports 
                     if p.direction == "input" and 
                     p.name not in (fsm.clock_signal, fsm.reset_signal)]
            for port in inputs:
                self._emitln(f"{port.name} = 0;")
            
            self._emitln()
            self._emitln(f"do_reset({test.reset_cycles});")
            self._emitln()
            
            # Apply test vectors
            for vec in test.vectors:
                if vec.comment:
                    self._emitln(f"// {vec.comment}")
                
                # Set inputs
                for signal, value in vec.inputs.items():
                    self._emitln(f"{signal} = {self._format_value(value)};")
                
                self._emitln("@(posedge clk);")
                
                # Check outputs
                for signal, expected in vec.outputs.items():
                    if expected is not None:
                        exp_str = self._format_value(expected)
                        if self.config.use_assertions:
                            self._emitln(f"assert ({signal} == {exp_str}) else begin")
                            self._indent()
                            self._emitln(f'$error("Mismatch: {signal} = %0d, expected {exp_str}", {signal});')
                            self._emitln("test_fail++;")
                            self._dedent()
                            self._emitln("end")
                        else:
                            self._emitln(f"if ({signal} !== {exp_str}) begin")
                            self._indent()
                            self._emitln(f'$display("FAIL: {signal} = %0d, expected {exp_str}", {signal});')
                            self._emitln("test_fail++;")
                            self._dedent()
                            self._emitln("end else test_pass++;")
                
                self._emitln()
            
            if self.config.verbose_display:
                self._emitln(f'$display("Test {test.name} complete");')
            
            self._dedent()
            self._emitln("endtask")
            self._emitln()
        
        self._dedent()
    
    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "1'b1" if value else "1'b0"
        if isinstance(value, int):
            return str(value)
        return str(value)
    
    def _generate_main_test(self, tests: List[TVSequence]):
        self._indent()
        self._emitln("// Main test sequence")
        self._emitln("initial begin")
        self._indent()
        
        self._emitln("test_pass = 0;")
        self._emitln("test_fail = 0;")
        self._emitln()
        
        if self.config.verbose_display:
            self._emitln('$display("=== Starting Tests ===");')
        
        for test in tests:
            self._emitln(f"test_{test.name}();")
        
        self._emitln()
        if self.config.verbose_display:
            self._emitln('$display("=== Test Summary ===");')
            self._emitln('$display("PASS: %0d", test_pass);')
            self._emitln('$display("FAIL: %0d", test_fail);')
        
        self._emitln()
        self._emitln("if (test_fail == 0)")
        self._indent()
        self._emitln('$display("ALL TESTS PASSED");')
        self._dedent()
        self._emitln("else")
        self._indent()
        self._emitln('$display("SOME TESTS FAILED");')
        self._dedent()
        
        self._emitln()
        self._emitln("$finish;")
        self._dedent()
        self._emitln("end")
        self._emitln()
        self._dedent()
    
    def _generate_module_end(self):
        self._emitln("endmodule")


def generate_testbench(fsm: FSMModule, 
                       tests: Optional[List[TVSequence]] = None,
                       config: Optional[TBGenConfig] = None) -> str:
    """Convenience function to generate a testbench.
    
    Args:
        fsm: FSM module to test
        tests: Optional list of test sequences
        config: Optional testbench configuration
        
    Returns:
        SystemVerilog testbench code
    """
    generator = TBGenerator(config)
    return generator.generate(fsm, tests)
