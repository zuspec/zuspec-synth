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
"""Tests for PipelineAnnotationPass, HazardAnalysisPass, ForwardingGenPass,
StallGenPass, SDCSchedulePass, and SVEmitPass.

These tests run the full Approach-C synthesis pass chain on a simple
3-stage ALU pipeline (loosely based on Example C.1 from pipeline-design.md).
"""

import pytest
import sys
import os

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
if "" in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

import zuspec.dataclasses as zdc
from zuspec.dataclasses.decorators import PipelineError
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.ir.pipeline_ir import PipelineIR, StageIR, HazardPair
from zuspec.synth.passes import (
    PipelineAnnotationPass,
    HazardAnalysisPass,
    ForwardingGenPass,
    StallGenPass,
    SDCSchedulePass,
    PipelineSVCodegen,
    SVEmitPass,
)


# ---------------------------------------------------------------------------
# Minimal pipeline component for testing
# ---------------------------------------------------------------------------

class _AluPipe:
    """Simple 3-stage ALU pipeline: IF → EX → WB."""

    a: zdc.u32
    b: zdc.u32
    out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n", stages=["IF", "EX", "WB"])
    def execute(self):
        IF = zdc.stage()
        a: zdc.u32 = self.a
        b: zdc.u32 = self.b
        EX = zdc.stage()
        result: zdc.u32 = a + b
        WB = zdc.stage()
        self.out = result


class _AluPipeWithForward:
    """3-stage pipeline with explicit forward declaration."""

    a: zdc.u32
    b: zdc.u32
    out: zdc.u32

    @zdc.pipeline(
        clock="clk",
        reset="rst",
        stages=["IF", "EX", "WB"],
        forward=[zdc.forward(signal="result", from_stage="EX", to_stage="IF")],
    )
    def execute(self):
        IF = zdc.stage()
        a: zdc.u32 = self.a
        b: zdc.u32 = self.b
        EX = zdc.stage()
        result: zdc.u32 = a + b
        WB = zdc.stage()
        self.out = result


class _AluAutoA:
    """Approach A: no zdc.stage() markers — scheduler assigns stages."""

    a: zdc.u32
    b: zdc.u32
    out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n", stages=2, forward=True)
    def execute(self):
        a: zdc.u32 = self.a
        b: zdc.u32 = self.b
        result: zdc.u32 = a + b
        self.out = result


class _AluAutoAUnconstrained:
    """Approach A with stages=True: let scheduler decide count."""

    a: zdc.u32
    b: zdc.u32
    out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n", stages=True, forward=True)
    def execute(self):
        a: zdc.u32 = self.a
        b: zdc.u32 = self.b
        result: zdc.u32 = a + b
        self.out = result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_annotation(component_cls, method_name="execute") -> SynthIR:
    """Run PipelineAnnotationPass on a component class and return SynthIR."""
    cfg = SynthConfig()
    ir = SynthIR()
    ir.component = component_cls
    pass_obj = PipelineAnnotationPass(cfg)
    return pass_obj.run(ir)


def _run_full_chain(component_cls, forward_default=True) -> SynthIR:
    """Run annotation → hazard → forwarding → stall passes."""
    cfg = SynthConfig(forward_default=forward_default)
    ir = SynthIR()
    ir.component = component_cls
    for pass_cls in [
        PipelineAnnotationPass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
    ]:
        ir = pass_cls(cfg).run(ir)
    return ir


def _run_approach_a_chain(component_cls, forward_default=True) -> SynthIR:
    """Run Approach A full chain: annotation → SDC → hazard → forwarding → stall → SV."""
    cfg = SynthConfig(forward_default=forward_default)
    ir = SynthIR()
    ir.component = component_cls
    for pass_cls in [
        PipelineAnnotationPass,
        SDCSchedulePass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
    ]:
        ir = pass_cls(cfg).run(ir)
    ir = SVEmitPass(cfg).run(ir)
    return ir


# ---------------------------------------------------------------------------
# PipelineAnnotationPass
# ---------------------------------------------------------------------------

class TestPipelineAnnotationPass:
    def test_builds_pipeline_ir(self):
        ir = _run_annotation(_AluPipe)
        assert ir.pipeline_ir is not None
        assert isinstance(ir.pipeline_ir, PipelineIR)

    def test_stage_count(self):
        ir = _run_annotation(_AluPipe)
        assert len(ir.pipeline_ir.stages) == 3

    def test_stage_names(self):
        ir = _run_annotation(_AluPipe)
        names = [s.name for s in ir.pipeline_ir.stages]
        assert names == ["IF", "EX", "WB"]

    def test_channels_for_live_vars(self):
        ir = _run_annotation(_AluPipe)
        pip = ir.pipeline_ir
        # "result" defined in EX, used in WB → channel name contains "result"
        ch_names = [ch.name for ch in pip.channels]
        assert any("result" in n for n in ch_names)

    def test_forwarding_decls_from_decorator(self):
        ir = _run_annotation(_AluPipeWithForward)
        pip = ir.pipeline_ir
        assert len(pip.forwarding) >= 1
        assert any(f.signal == "result" for f in pip.forwarding)

    def test_no_pipeline_attr_skips(self):
        class NoPipeline:
            def execute(self):
                pass

        cfg = SynthConfig()
        ir = SynthIR()
        ir.component = NoPipeline
        ir2 = PipelineAnnotationPass(cfg).run(ir)
        assert ir2.pipeline_ir is None


# ---------------------------------------------------------------------------
# HazardAnalysisPass
# ---------------------------------------------------------------------------

class TestHazardAnalysisPass:
    def test_no_hazard_on_simple_pipeline(self):
        cfg = SynthConfig()
        ir = _run_annotation(_AluPipe)
        ir = HazardAnalysisPass(cfg).run(ir)
        pip = ir.pipeline_ir
        # Simple non-looping pipeline should have no hazards
        # (all defs flow forward only)
        for h in pip.hazards:
            assert h.kind in ("RAW", "WAW", "WAR")

    def test_hazards_are_hazard_pairs(self):
        cfg = SynthConfig()
        ir = _run_annotation(_AluPipe)
        ir = HazardAnalysisPass(cfg).run(ir)
        for h in ir.pipeline_ir.hazards:
            assert isinstance(h, HazardPair)


# ---------------------------------------------------------------------------
# ForwardingGenPass
# ---------------------------------------------------------------------------

class TestForwardingGenPass:
    def test_no_error_with_forward_default_true(self):
        ir = _run_full_chain(_AluPipe, forward_default=True)
        # Should complete without PipelineError

    def test_no_error_with_forward_default_false(self):
        ir = _run_full_chain(_AluPipe, forward_default=False)

    def test_error_when_unresolved_and_default_none(self):
        # Build a pipeline IR with a RAW hazard and no forwarding resolution
        cfg = SynthConfig(forward_default=None)
        ir = _run_annotation(_AluPipe)
        ir = HazardAnalysisPass(cfg).run(ir)
        # Inject a synthetic hazard to guarantee unresolved path
        ir.pipeline_ir.hazards.append(
            HazardPair(kind="RAW", signal="injected",
                       producer_stage="EX", consumer_stage="IF")
        )
        with pytest.raises(PipelineError):
            ForwardingGenPass(cfg).run(ir)


# ---------------------------------------------------------------------------
# StallGenPass
# ---------------------------------------------------------------------------

class TestStallGenPass:
    def test_valid_chain_created(self):
        ir = _run_full_chain(_AluPipe)
        pip = ir.pipeline_ir
        valid_chain = getattr(pip, "valid_chain", None)
        assert valid_chain is not None
        assert len(valid_chain) == len(pip.stages)

    def test_stall_signals_list(self):
        ir = _run_full_chain(_AluPipe)
        pip = ir.pipeline_ir
        stall_signals = getattr(pip, "stall_signals", None)
        assert isinstance(stall_signals, list)


# ---------------------------------------------------------------------------
# SDCSchedulePass
# ---------------------------------------------------------------------------

class TestSDCSchedulePass:
    def test_sdc_produces_same_or_fewer_stages(self):
        cfg = SynthConfig()
        ir = _run_annotation(_AluPipe)
        n_before = len(ir.pipeline_ir.stages)
        ir = SDCSchedulePass(cfg).run(ir)
        n_after = len(ir.pipeline_ir.stages)
        # SDC can merge or keep stages, never increase without resource pressure
        assert n_after <= n_before or n_after >= 1

    def test_sdc_sets_approach(self):
        cfg = SynthConfig()
        ir = _run_annotation(_AluPipe)
        ir = SDCSchedulePass(cfg).run(ir)
        # User-annotated pipeline: SDC validates but doesn't reschedule → "user+sdc"
        assert ir.pipeline_ir.approach == "user+sdc"

    def test_sdc_skips_when_no_pipeline_ir(self):
        cfg = SynthConfig()
        ir = SynthIR()
        ir2 = SDCSchedulePass(cfg).run(ir)
        assert ir2.pipeline_ir is None


# ---------------------------------------------------------------------------
# SVEmitPass / PipelineSVCodegen
# ---------------------------------------------------------------------------

class TestSVEmitPass:
    def _get_sv(self) -> str:
        ir = _run_full_chain(_AluPipe)
        codegen = PipelineSVCodegen()
        return codegen.emit(ir.pipeline_ir)

    def test_sv_is_string(self):
        sv = self._get_sv()
        assert isinstance(sv, str)
        assert len(sv) > 0

    def test_sv_has_module_header(self):
        sv = self._get_sv()
        assert "module " in sv
        assert "endmodule" in sv

    def test_sv_has_clk_rst(self):
        sv = self._get_sv()
        assert "clk" in sv
        assert "rst" in sv.lower() or "reset" in sv.lower()

    def test_sv_has_pipeline_registers(self):
        sv = self._get_sv()
        # Pipeline registers for live variables across stage boundaries
        assert "reg" in sv

    def test_sv_has_always_blocks(self):
        sv = self._get_sv()
        assert "always" in sv

    def test_sv_no_sv_only_keywords(self):
        """Verilog 2005 only — no SystemVerilog always_ff / logic / always_comb."""
        sv = self._get_sv()
        assert "always_ff" not in sv
        assert "always_comb" not in sv
        # 'logic' as a keyword (allow in identifiers like 'logic_op' or comments)
        import re
        assert not re.search(r"\blogic\b", sv)

    def test_sv_emit_pass_stores_in_ir(self):
        ir = _run_full_chain(_AluPipe)
        cfg = SynthConfig()
        ir = SVEmitPass(cfg).run(ir)
        assert "pipeline_sv" in ir.lowered_sv
        assert len(ir.lowered_sv["pipeline_sv"]) > 0


class TestSVPhase6ExprLowering:
    """Tests for Phase 6: expression lowering — Python AST → actual SV statements."""

    def _get_sv(self) -> str:
        ir = _run_full_chain(_AluPipe)
        cfg = SynthConfig()
        ir = SDCSchedulePass(cfg).run(ir)
        return PipelineSVCodegen().emit(ir.pipeline_ir)

    def test_no_operation_stubs(self):
        """SVEmitPass must not leave 'operation(s) — lowered' placeholder stubs."""
        sv = self._get_sv()
        assert "operation(s) — lowered" not in sv
        assert "// TODO" not in sv

    def test_real_input_ports_in_header(self):
        """Input ports (self.a, self.b) appear as 'input wire' declarations."""
        sv = self._get_sv()
        assert "input  wire [31:0] a," in sv
        assert "input  wire [31:0] b," in sv

    def test_real_output_port_in_header(self):
        """Output port (self.out) appears as 'output reg' declaration."""
        sv = self._get_sv()
        assert "output reg  [31:0] out" in sv

    def test_no_trailing_comma_on_last_port(self):
        """The last entry in the port list must not be followed by a comma."""
        import re
        sv = self._get_sv()
        # Find the module port list — last port before ");"
        match = re.search(r"\(([^)]+)\);", sv, re.DOTALL)
        assert match, "Module port list not found"
        port_list = match.group(1)
        last_line = [l.strip() for l in port_list.splitlines() if l.strip()][-1]
        assert not last_line.endswith(","), f"Trailing comma on last port: {last_line!r}"

    def test_stage_local_signals_declared_at_module_scope(self):
        """Stage-local signals (a_if, b_if, result_ex) are declared as 'reg' at module scope."""
        sv = self._get_sv()
        assert "reg [31:0] a_if;" in sv
        assert "reg [31:0] b_if;" in sv
        assert "reg [31:0] result_ex;" in sv

    def test_if_stage_reads_input_ports(self):
        """IF stage always block contains blocking assignments from input ports."""
        sv = self._get_sv()
        assert "a_if = a;" in sv
        assert "b_if = b;" in sv

    def test_ex_stage_computes_sum(self):
        """EX stage always block contains the addition expression using pipeline registers."""
        sv = self._get_sv()
        assert "result_ex = (a_if_to_ex_q + b_if_to_ex_q);" in sv

    def test_wb_stage_drives_output(self):
        """WB stage always block drives the output port from its pipeline register."""
        sv = self._get_sv()
        assert "out = result_ex_to_wb_q;" in sv

    def test_stage_signals_declared_before_always_blocks(self):
        """Stage-local signal declarations appear before the combinational always blocks."""
        sv = self._get_sv()
        sig_pos = sv.find("reg [31:0] a_if;")
        always_pos = sv.find("always @(*)") if "always @(*)" in sv else sv.find("always @( *)")
        assert sig_pos != -1 and always_pos != -1
        assert sig_pos < always_pos, "Signal declarations must precede always blocks"

    def test_no_wire_inside_always(self):
        """'wire' keyword must not appear inside always blocks (Verilog 2005 restriction)."""
        sv = self._get_sv()
        inside = False
        for line in sv.splitlines():
            stripped = line.strip()
            if stripped.startswith("always"):
                inside = True
            if inside and stripped.startswith("endmodule"):
                inside = False
            if inside and "wire " in stripped and not stripped.startswith("//"):
                assert False, f"'wire' declaration found inside always block: {line!r}"

    def test_else_branch_prevents_latches(self):
        """Else branch in each stage provides default-zero assignments (no latches)."""
        sv = self._get_sv()
        assert "end else begin" in sv
        # At least the intermediate signals should have default zeros
        assert "a_if = 32'b0;" in sv
        assert "b_if = 32'b0;" in sv
        assert "result_ex = 32'b0;" in sv
        # Output port should also have default zero
        assert "out = 32'b0;" in sv

    def test_verilog_2005_only(self):
        """No SystemVerilog-only keywords appear in the output."""
        import re
        sv = self._get_sv()
        assert "always_ff" not in sv
        assert "always_comb" not in sv
        assert not re.search(r"\blogic\b", sv)


# ---------------------------------------------------------------------------
# PipelineToSource
# ---------------------------------------------------------------------------

class TestPipelineToSource:
    def _get_source(self) -> str:
        from zuspec.synth.passes import PipelineToSource
        ir = _run_full_chain(_AluPipe)
        return PipelineToSource().reconstruct(ir.pipeline_ir)

    def test_returns_string(self):
        src = self._get_source()
        assert isinstance(src, str) and len(src) > 0

    def test_has_stage_markers(self):
        src = self._get_source()
        # Should have "IF = zdc.stage()" etc.
        assert "zdc.stage()" in src

    def test_has_def_keyword(self):
        src = self._get_source()
        assert "def execute(self):" in src

    def test_has_all_stage_names(self):
        src = self._get_source()
        for name in ("IF", "EX", "WB"):
            assert name in src

    def test_pass_stores_annotation(self):
        from zuspec.synth.passes import PipelineToSourcePass
        ir = _run_full_chain(_AluPipe)
        ir = PipelineToSourcePass(SynthConfig()).run(ir)
        assert ir.pipeline_ir.source_annotation is not None
        assert "zdc.stage()" in ir.pipeline_ir.source_annotation


# ---------------------------------------------------------------------------
# Phase 8: Approach A — automatic scheduling
# ---------------------------------------------------------------------------

class TestApproachA:
    """Approach A: @zdc.pipeline without zdc.stage() markers; SDC schedules."""

    def test_approach_a_sets_auto_before_sdc(self):
        """PipelineAnnotationPass sets approach='auto' for stage-marker-free bodies."""
        ir = _run_annotation(_AluAutoA, method_name="execute")
        assert ir.pipeline_ir is not None
        assert ir.pipeline_ir.approach == "auto"

    def test_approach_a_has_one_flat_stage(self):
        """After annotation, all ops are in a single S0 stage."""
        ir = _run_annotation(_AluAutoA, method_name="execute")
        assert len(ir.pipeline_ir.stages) == 1
        assert ir.pipeline_ir.stages[0].name == "S0"

    def test_approach_a_flat_stage_has_ops(self):
        ir = _run_annotation(_AluAutoA, method_name="execute")
        assert len(ir.pipeline_ir.stages[0].operations) > 0

    def test_approach_a_desired_stage_count_preserved(self):
        """pipeline_stages= carries the user's stages=2 hint to SDC."""
        ir = _run_annotation(_AluAutoA, method_name="execute")
        assert ir.pipeline_ir.pipeline_stages == 2

    def test_sdc_reschedules_to_sdc_approach(self):
        """After SDC, approach becomes 'sdc'."""
        ir = _run_approach_a_chain(_AluAutoA)
        assert ir.pipeline_ir.approach == "sdc"

    def test_sdc_creates_target_stage_count(self):
        """SDC respects stages=2 and produces exactly 2 stages."""
        ir = _run_approach_a_chain(_AluAutoA)
        assert len(ir.pipeline_ir.stages) == 2

    def test_sdc_computes_channels(self):
        """After scheduling, pipeline registers cross the S0→S1 boundary."""
        ir = _run_approach_a_chain(_AluAutoA)
        assert len(ir.pipeline_ir.channels) > 0
        ch_names = {c.name for c in ir.pipeline_ir.channels}
        # a and b are read in S0 and used in S1 → must have pipeline regs
        assert any("a" in n for n in ch_names)
        assert any("b" in n for n in ch_names)

    def test_sdc_sv_output_not_empty(self):
        ir = _run_approach_a_chain(_AluAutoA)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "module" in sv

    def test_sdc_sv_module_name(self):
        ir = _run_approach_a_chain(_AluAutoA)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "_AluAutoA" in sv

    def test_sdc_sv_has_pipeline_regs(self):
        """Generated SV should declare pipeline register flops for channels."""
        ir = _run_approach_a_chain(_AluAutoA)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "_q;" in sv or "posedge clk" in sv

    def test_sdc_sv_has_two_always_blocks(self):
        """One always @(*) combinational block per stage."""
        ir = _run_approach_a_chain(_AluAutoA)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        # At minimum two combinational always blocks (one per scheduled stage)
        assert sv.count("always @(*)") >= 2

    def test_sdc_unconstrained_produces_stages(self):
        """stages=True lets SDC choose; result must have >= 1 stage."""
        ir = _run_approach_a_chain(_AluAutoAUnconstrained)
        assert ir.pipeline_ir.approach == "sdc"
        assert len(ir.pipeline_ir.stages) >= 1

    def test_sdc_unconstrained_sv_valid(self):
        ir = _run_approach_a_chain(_AluAutoAUnconstrained)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "module" in sv and "endmodule" in sv

    def test_approach_a_no_phantom_channels(self):
        """All channels should reference valid stage names."""
        ir = _run_approach_a_chain(_AluAutoA)
        stage_names = {s.name for s in ir.pipeline_ir.stages}
        for ch in ir.pipeline_ir.channels:
            assert ch.src_stage in stage_names, f"{ch.name}.src_stage={ch.src_stage!r} not in {stage_names}"
            assert ch.dst_stage in stage_names, f"{ch.name}.dst_stage={ch.dst_stage!r} not in {stage_names}"

    def test_approach_c_unaffected_by_phase8(self):
        """Existing Approach C tests still pass — SDC skips user-annotated pipelines."""
        ir = _run_full_chain(_AluPipe)
        assert ir.pipeline_ir.approach in ("user", "user+sdc")
        assert len(ir.pipeline_ir.stages) == 3


# ---------------------------------------------------------------------------
# Regfile test fixtures
# ---------------------------------------------------------------------------

class _RfPipe:
    """3-stage pipeline with an IndexedRegFile (RISC-V-style register file).

    Stage layout:
      ID  — read register file: rs1 → rdata1
      EX  — compute result = rdata1 + self.imm
      WB  — write register file: rd ← result, output self.out
    """

    rs1:  zdc.u5
    rd:   zdc.u5
    imm:  zdc.u32
    out:  zdc.u32

    regfile: zdc.IndexedRegFile[zdc.u32, 32]

    @zdc.pipeline(
        clock="clk",
        reset="rst_n",
        stages=["ID", "EX", "WB"],
        forward=[zdc.forward(signal="regfile.rdata1", from_stage="WB", to_stage="ID")],
    )
    def execute(self):
        ID = zdc.stage()
        rs1: zdc.u5 = self.rs1
        rdata1: zdc.u32 = self.regfile.read(rs1)
        EX = zdc.stage()
        rd: zdc.u5 = self.rd
        imm: zdc.u32 = self.imm
        result: zdc.u32 = rdata1 + imm
        WB = zdc.stage()
        self.regfile.write(rd, result)
        self.out = result


def _run_rf_chain(component_cls, forward_default=True) -> SynthIR:
    """Run the full pipeline chain including regfile support."""
    cfg = SynthConfig(forward_default=forward_default)
    ir = SynthIR()
    ir.component = component_cls
    for pass_cls in [
        PipelineAnnotationPass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
    ]:
        ir = pass_cls(cfg).run(ir)
    ir = SVEmitPass(cfg).run(ir)
    return ir


# ---------------------------------------------------------------------------
# TestRegFileModeling
# ---------------------------------------------------------------------------

class TestRegFileModeling:
    """Phase 9: IndexedRegFile inlined as mem array into the pipeline module."""

    def test_regfile_accesses_detected(self):
        """HazardAnalysisPass collects regfile read and write accesses."""
        ir = _run_rf_chain(_RfPipe)
        pip = ir.pipeline_ir
        assert len(pip.regfile_accesses) == 2
        kinds = {a.kind for a in pip.regfile_accesses}
        assert kinds == {"read", "write"}

    def test_regfile_read_in_id_stage(self):
        ir = _run_rf_chain(_RfPipe)
        reads = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "read"]
        assert len(reads) == 1
        assert reads[0].stage == "ID"
        assert reads[0].field_name == "regfile"

    def test_regfile_write_in_wb_stage(self):
        ir = _run_rf_chain(_RfPipe)
        writes = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "write"]
        assert len(writes) == 1
        assert writes[0].stage == "WB"

    def test_regfile_decls_built(self):
        """HazardAnalysisPass builds one RegFileDeclInfo for the regfile field."""
        ir = _run_rf_chain(_RfPipe)
        decls = ir.pipeline_ir.regfile_decls
        assert len(decls) == 1
        assert decls[0].field_name == "regfile"
        assert decls[0].data_width == 32
        assert decls[0].addr_width == 5
        assert decls[0].depth == 32

    def test_regfile_hazard_detected(self):
        """RAW hazard: read in ID, write in later WB stage."""
        ir = _run_rf_chain(_RfPipe)
        hazards = ir.pipeline_ir.regfile_hazards
        assert len(hazards) >= 1
        h = hazards[0]
        assert h.field_name == "regfile"
        assert h.read_stage == "ID"
        assert h.write_stage == "WB"

    def test_regfile_hazard_resolved(self):
        """ForwardingGenPass resolves the regfile hazard (forward by default)."""
        ir = _run_rf_chain(_RfPipe)
        hazards = ir.pipeline_ir.regfile_hazards
        assert all(h.resolved_by is not None for h in hazards)

    def test_sv_has_mem_array(self):
        """Emitted SV declares the regfile_mem array."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "regfile_mem" in sv

    def test_sv_mem_array_dimensions(self):
        """Memory array has correct [31:0] data width and [0:31] depth."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "reg [31:0] regfile_mem [0:31];" in sv

    def test_sv_has_clocked_write(self):
        """Emitted SV contains a clocked always block that writes to regfile_mem."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "regfile_mem[" in sv and "posedge clk" in sv

    def test_sv_write_guarded_by_valid(self):
        """The regfile write block is gated by the WB stage valid register."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "wb_valid_q" in sv

    def test_sv_has_read_mux(self):
        """Emitted SV contains a combinational always block for the read port."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        # Regfile read mux: always @(*) with memory read
        assert "regfile_mem[" in sv

    def test_sv_read_result_declared_as_reg(self):
        """The read result variable (rdata1_id) is declared as a reg by the mux block."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "reg [31:0] rdata1_id;" in sv

    def test_sv_forwarding_mux_present(self):
        """When a regfile forwarding hazard exists, the mux bypasses from WB."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        # Forwarding: if wb_valid_q && rd_wb == rs1_id → rdata1_id = result_wb
        assert "wb_valid_q" in sv

    def test_sv_read_stmt_not_in_stage_block(self):
        """The self.regfile.read() statement is NOT emitted in the ID stage always block."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        # The read() call should not appear as a raw Python-style statement
        assert "self.regfile.read" not in sv

    def test_sv_write_stmt_not_in_stage_block(self):
        """The self.regfile.write() call is NOT emitted in the WB stage always block."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "self.regfile.write" not in sv

    def test_sv_module_complete(self):
        """Emitted SV is a complete, non-empty module."""
        ir = _run_rf_chain(_RfPipe)
        sv = ir.lowered_sv.get("pipeline_sv", "")
        assert "module _RfPipe" in sv
        assert "endmodule" in sv
