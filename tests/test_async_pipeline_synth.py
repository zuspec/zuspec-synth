"""Tests for async pipeline synthesis — ``@zdc.pipeline`` async method API.

Three test tiers:
  Tier 1 — Unit: Assert PipelineIR structure (stages, channels, clock/reset).
  Tier 2 — Codegen: Assert Verilog structural properties and Verilator lint.
  Tier 3 — System: Full synth chain, Verilator lint-clean for realistic designs.

All component classes must be defined at module level (required for
``inspect.getsource``).
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import inspect

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
for _p in [_synth_src, _dc_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zuspec.dataclasses as zdc
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.passes import (
    HazardAnalysisPass,
    ForwardingGenPass,
    StallGenPass,
    SVEmitPass,
    AsyncPipelineElaboratePass,
    AsyncPipelineToIrPass,
)
from zuspec.dataclasses.ir.pipeline_async_pass import AsyncPipelineFrontendPass

# ---------------------------------------------------------------------------
# Helper: parse source string with AsyncPipelineFrontendPass
# ---------------------------------------------------------------------------

def _parse_src(src: str):
    tree = ast.parse(textwrap.dedent(src))
    fp = AsyncPipelineFrontendPass()
    fp.visit(tree)
    return fp.result


def run_async_pipeline_synth(component_cls, forward_default: bool = True, return_ir: bool = False):
    """Run the async pipeline synth pass chain and return the emitted SV text.

    Pass order:
      AsyncPipelineElaboratePass → AsyncPipelineToIrPass →
      HazardAnalysisPass → ForwardingGenPass → StallGenPass → SVEmitPass

    When *return_ir* is True, returns ``(pip, sv_text)`` instead of just *sv_text*.
    """
    cfg = SynthConfig(forward_default=forward_default)
    ir = SynthIR()
    ir.component = component_cls
    for pass_cls in [
        AsyncPipelineElaboratePass,
        AsyncPipelineToIrPass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
    ]:
        ir = pass_cls(cfg).run(ir)
    ir = SVEmitPass(cfg).run(ir)
    sv = ir.lowered_sv.get("sv/pipeline/top", "")
    if return_ir:
        return ir.pipeline_ir, sv
    return sv


def _verilator_lint(sv: str) -> None:
    """Run ``verilator --lint-only`` on *sv*; skip if Verilator not on PATH."""
    verilator = os.path.join(
        os.path.dirname(_this_dir), "..", "..", "packages", "verilator", "bin", "verilator"
    )
    if not os.path.isfile(verilator):
        verilator = shutil.which("verilator") or "verilator"
    if not shutil.which(verilator) and not os.path.isfile(verilator):
        pytest.skip("verilator not found")
    with tempfile.TemporaryDirectory() as d:
        sv_file = os.path.join(d, "dut.sv")
        with open(sv_file, "w") as f:
            f.write(sv)
        result = subprocess.run(
            [verilator, "--lint-only", "--sv", sv_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Verilator lint failed:\n{result.stdout}{result.stderr}\n"
                f"--- SV ---\n{sv}"
            )


# ---------------------------------------------------------------------------
# ── Component definitions (module-level for inspect.getsource) ──────────────
# ---------------------------------------------------------------------------

@zdc.dataclass
class _PassThrough3(zdc.Component):
    """Simple 3-stage pass-through pipeline."""
    data_in: zdc.u32
    data_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as FETCH:
            data: zdc.u32 = self.data_in
        async with zdc.pipeline.stage() as PROC:
            processed: zdc.u32 = data
        async with zdc.pipeline.stage() as WB:
            self.data_out = processed


@zdc.dataclass
class _Adder3(zdc.Component):
    """3-stage adder pipeline: FETCH → EXEC → WB."""
    a_in: zdc.u32
    b_in: zdc.u32
    sum_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as FETCH:
            a: zdc.u32 = self.a_in
            b: zdc.u32 = self.b_in
        async with zdc.pipeline.stage() as EXEC:
            result: zdc.u32 = a + b
        async with zdc.pipeline.stage() as WB:
            self.sum_out = result


@zdc.dataclass
class _AutoThread5(zdc.Component):
    """5-stage pipeline; tag skips stages (auto-threading test)."""
    tag_in: zdc.u32
    data_in: zdc.u32
    tag_out: zdc.u32
    data_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as IF:
            tag: zdc.u32 = self.tag_in
            data: zdc.u32 = self.data_in
        async with zdc.pipeline.stage() as ID:
            d2: zdc.u32 = data
        async with zdc.pipeline.stage() as EX:
            d3: zdc.u32 = d2
        async with zdc.pipeline.stage() as MEM:
            d4: zdc.u32 = d3
        async with zdc.pipeline.stage() as WB:
            self.tag_out = tag
            self.data_out = d4


@zdc.dataclass
class _MultiplyAccum4(zdc.Component):
    """4-stage multiply-accumulate pipeline."""
    a_in: zdc.u32
    b_in: zdc.u32
    acc_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as FETCH:
            a: zdc.u32 = self.a_in
            b: zdc.u32 = self.b_in
        async with zdc.pipeline.stage() as MUL:
            product: zdc.u32 = a * b
        async with zdc.pipeline.stage() as ACC:
            accum: zdc.u32 = product + product
        async with zdc.pipeline.stage() as WB:
            self.acc_out = accum


@zdc.dataclass
class _WidthVariety(zdc.Component):
    """Pipeline with different variable widths."""
    byte_in: zdc.u8
    word_in: zdc.u16
    out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as FETCH:
            b: zdc.u8 = self.byte_in
            w: zdc.u16 = self.word_in
        async with zdc.pipeline.stage() as EXTEND:
            b32: zdc.u32 = b
            w32: zdc.u32 = w
        async with zdc.pipeline.stage() as COMBINE:
            result: zdc.u32 = b32 + w32
        async with zdc.pipeline.stage() as WB:
            self.out = result


@zdc.dataclass
class _MultiCyclePipe(zdc.Component):
    """Pipeline with a multi-cycle stage (cycles=3)."""
    data_in: zdc.u32
    data_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as FETCH:
            data: zdc.u32 = self.data_in
        async with zdc.pipeline.stage(cycles=3) as COMPUTE:
            result: zdc.u32 = data
        async with zdc.pipeline.stage() as WB:
            self.data_out = result


class _BubblePipe(zdc.Component):
    """Pipeline with a bubble stage (FILTER inserts bubbles on invalid data)."""
    data_in: zdc.u32
    data_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as FETCH:
            data: zdc.u32 = self.data_in
        async with zdc.pipeline.stage() as FILTER:
            await zdc.pipeline.bubble()
        async with zdc.pipeline.stage() as WB:
            self.data_out = data


class _RegfilePipe(zdc.Component):
    """3-stage pipeline with a register-file RAW hazard.

    - ID:  captures rs1/rd ports and block-reads rs1 from the regfile.
    - EX:  passes rdata through.
    - WB:  writes result into regfile[rd] (auto-threaded from ID).

    The WB write is in a later stage than the ID read, producing a RAW
    hazard that HazardAnalysisPass should detect.
    """
    regfile: zdc.IndexedRegFile[zdc.u5, zdc.u32] = zdc.indexed_regfile()
    rs1: zdc.u5
    rd: zdc.u5
    data_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as ID:
            rs1: zdc.u5 = self.rs1
            rd: zdc.u5 = self.rd
            rdata: zdc.u32 = await zdc.pipeline.block(self.regfile[rs1])
        async with zdc.pipeline.stage() as EX:
            result: zdc.u32 = rdata
        async with zdc.pipeline.stage() as WB:
            self.data_out = result
            zdc.pipeline.write(self.regfile[rd], result)


class _BypassLockPipe(zdc.Component):
    """3-stage pipeline using PipelineResource with BypassLock.

    Declared with ``zdc.pipeline.resource(16, lock=zdc.BypassLock())``.
    RAW hazards are resolved via a bypass forwarding mux.
    """
    rf = zdc.pipeline.resource(16, lock=zdc.BypassLock())
    rs1: zdc.u4
    rd: zdc.u4
    data_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as ID:
            rs1: zdc.u4 = self.rs1
            rd: zdc.u4 = self.rd
            rdata: zdc.u32 = await zdc.pipeline.block(self.rf[rs1])
        async with zdc.pipeline.stage() as EX:
            result: zdc.u32 = rdata
        async with zdc.pipeline.stage() as WB:
            self.data_out = result
            zdc.pipeline.write(self.rf[rd], result)


class _QueueLockPipe(zdc.Component):
    """3-stage pipeline using PipelineResource with QueueLock.

    Declared with ``zdc.pipeline.resource(16, lock=zdc.QueueLock())``.
    RAW hazards are resolved via stall (no bypass mux).
    """
    rf = zdc.pipeline.resource(16, lock=zdc.QueueLock())
    rs1: zdc.u4
    rd: zdc.u4
    data_out: zdc.u32

    @zdc.pipeline(clock="clk", reset="rst_n")
    async def run(self):
        async with zdc.pipeline.stage() as ID:
            rs1: zdc.u4 = self.rs1
            rd: zdc.u4 = self.rd
            rdata: zdc.u32 = await zdc.pipeline.block(self.rf[rs1])
        async with zdc.pipeline.stage() as EX:
            result: zdc.u32 = rdata
        async with zdc.pipeline.stage() as WB:
            self.data_out = result
            zdc.pipeline.write(self.rf[rd], result)


# ---------------------------------------------------------------------------
# Tier 1 — Unit tests: IrPipeline extraction
# ---------------------------------------------------------------------------

class TestTier1FrontendPass:
    """Validate AsyncPipelineFrontendPass extraction."""

    def test_three_stage_extraction(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as FETCH:
                        pass
                    async with pipeline.stage() as EXEC:
                        pass
                    async with pipeline.stage() as WB:
                        pass
        """)
        assert ip is not None
        assert [s.name for s in ip.stages] == ["FETCH", "EXEC", "WB"]

    def test_clock_reset_extraction(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as S:
                        pass
        """)
        assert ip.clock_field == "clk"
        assert ip.reset_field == "rst_n"

    def test_cycles_kwarg(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage(cycles=4) as S:
                        pass
        """)
        assert ip.stages[0].cycles == 4

    def test_non_await_write_extracted(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as EX:
                        pipeline.write(self.rf[rd], result)
        """)
        ops = ip.stages[0].hazard_ops
        assert any(o.op == "write" for o in ops)

    def test_non_await_release_extracted(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as WB:
                        pipeline.release(self.rf[rd])
        """)
        ops = ip.stages[0].hazard_ops
        assert any(o.op == "release" for o in ops)

    def test_await_bubble_extracted(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as S:
                        await S.bubble()
        """)
        from zuspec.dataclasses.ir.pipeline_async import IrBubble
        body = ip.stages[0].body
        assert any(isinstance(n, IrBubble) for n in body)

    def test_await_block_result_var_extracted(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as ID:
                        v1: u32 = await pipeline.block(self.rf[rs1])
        """)
        from zuspec.dataclasses.ir.pipeline_async import IrHazardOp
        ops = ip.stages[0].hazard_ops
        block_ops = [o for o in ops if o.op == "block"]
        assert len(block_ops) == 1
        assert block_ops[0].result_var == "v1"
        assert block_ops[0].result_width == 32

    def test_ann_assign_width_inference(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as S:
                        x: u8 = self.in8
        """)
        # AnnAssign is preserved in body
        body = ip.stages[0].body
        assert len(body) == 1


# ---------------------------------------------------------------------------
# Tier 1 — Unit tests: IR translation
# ---------------------------------------------------------------------------

class TestTier1ToIrPass:
    """Validate AsyncPipelineToIrPass PipelineIR construction."""

    def _run_to_ir(self, ip, forward_default=True):
        cfg = SynthConfig(forward_default=forward_default)
        ir = SynthIR()
        ir.async_pipeline_ir = ip
        ir = AsyncPipelineToIrPass(cfg).run(ir)
        return ir.pipeline_ir

    def test_stage_count(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as A: pass
                    async with pipeline.stage() as B: pass
                    async with pipeline.stage() as C: pass
        """)
        pip = self._run_to_ir(ip)
        assert len(pip.stages) == 3

    def test_cross_stage_channel_generated(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as FETCH:
                        a: u32 = self.a_in
                    async with pipeline.stage() as EXEC:
                        result: u32 = a + 1
        """)
        pip = self._run_to_ir(ip)
        names = [c.name for c in pip.channels]
        assert any("a_" in n for n in names), f"Expected channel for 'a', got: {names}"

    def test_auto_threading_channel(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as A:
                        tag: u32 = self.tag_in
                    async with pipeline.stage() as B:
                        x: u32 = self.data_in
                    async with pipeline.stage() as C:
                        self.out = tag
        """)
        pip = self._run_to_ir(ip)
        # tag must be threaded from A to B and B to C
        src_names = [c.name for c in pip.channels]
        a_to_b = any("a_to_b" in n.lower() and "tag" in n.lower() for n in src_names)
        b_to_c = any("b_to_c" in n.lower() and "tag" in n.lower() for n in src_names)
        assert a_to_b and b_to_c, f"Expected auto-threaded channels for 'tag', got: {src_names}"

    def test_clock_reset_on_pipeline_ir(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"myclk\", reset=\"my_rst\")
                async def run(self):
                    async with pipeline.stage() as S: pass
        """)
        pip = self._run_to_ir(ip)
        assert pip.clock_field == "myclk"
        assert pip.reset_field == "my_rst"

    def test_channel_width_u8(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage() as A:
                        b: u8 = self.b_in
                    async with pipeline.stage() as B:
                        self.out = b
        """)
        pip = self._run_to_ir(ip)
        byte_ch = [c for c in pip.channels if c.name.startswith("b_")]
        assert byte_ch, "Expected a channel for 'b'"
        assert byte_ch[0].width == 8

    def test_multicycle_stage_cycle_hi(self):
        ip = _parse_src("""
            class P:
                @pipeline(clock=\"clk\", reset=\"rst_n\")
                async def run(self):
                    async with pipeline.stage(cycles=4) as LONG:
                        x: u32 = self.in_val
                    async with pipeline.stage() as OUT:
                        self.out_val = x
        """)
        pip = self._run_to_ir(ip)
        long_stage = next(s for s in pip.stages if s.name == "LONG")
        assert long_stage.cycle_hi == 3  # cycles=4 → cycle_hi=3 (0-indexed)


# ---------------------------------------------------------------------------
# Tier 2 — Codegen tests: Verilog structure + Verilator lint
# ---------------------------------------------------------------------------

class TestTier2CodegenPassThrough:
    """Simple 3-stage pass-through pipeline."""

    def test_ir_built(self):
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _PassThrough3
        ir = AsyncPipelineElaboratePass(cfg).run(ir)
        ir = AsyncPipelineToIrPass(cfg).run(ir)
        assert ir.pipeline_ir is not None
        assert len(ir.pipeline_ir.stages) == 3

    def test_module_declaration(self):
        sv = run_async_pipeline_synth(_PassThrough3)
        assert "PassThrough3" in sv
        assert "module" in sv

    def test_stage_valid_regs(self):
        sv = run_async_pipeline_synth(_PassThrough3)
        for stage in ("fetch", "proc", "wb"):
            assert f"{stage}_valid_q" in sv.lower() or f"{stage}_valid" in sv.lower(), \
                f"Missing valid register for {stage}"

    def test_pipeline_register_generated(self):
        sv = run_async_pipeline_synth(_PassThrough3)
        assert "data_fetch_to_proc" in sv or "data_fetch_to_proc_q" in sv, \
            "Expected pipeline register for 'data'"

    def test_lint_clean(self):
        _verilator_lint(run_async_pipeline_synth(_PassThrough3))


class TestTier2CodegenAdder3:
    """3-stage adder pipeline: FETCH → EXEC → WB."""

    def test_stage_count(self):
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _Adder3
        ir = AsyncPipelineElaboratePass(cfg).run(ir)
        ir = AsyncPipelineToIrPass(cfg).run(ir)
        assert len(ir.pipeline_ir.stages) == 3

    def test_sum_port_present(self):
        sv = run_async_pipeline_synth(_Adder3)
        assert "sum_out" in sv

    def test_ab_channels_present(self):
        sv = run_async_pipeline_synth(_Adder3)
        assert "a_fetch_to_exec" in sv or "a_fetch" in sv.lower()
        assert "b_fetch_to_exec" in sv or "b_fetch" in sv.lower()

    def test_result_channel_present(self):
        sv = run_async_pipeline_synth(_Adder3)
        assert "result_exec_to_wb" in sv

    def test_lint_clean(self):
        _verilator_lint(run_async_pipeline_synth(_Adder3))


class TestTier2AutoThread5:
    """5-stage pipeline with skip-stage auto-threading."""

    def test_five_stages_generated(self):
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _AutoThread5
        ir = AsyncPipelineElaboratePass(cfg).run(ir)
        ir = AsyncPipelineToIrPass(cfg).run(ir)
        assert len(ir.pipeline_ir.stages) == 5

    def test_tag_threaded_through_all_stages(self):
        sv = run_async_pipeline_synth(_AutoThread5)
        # tag must appear in at least 3 pipeline registers (IF→ID, ID→EX, EX→MEM, MEM→WB)
        count = sv.lower().count("tag_if_to_") + sv.lower().count("tag_id_to_") + \
                sv.lower().count("tag_ex_to_") + sv.lower().count("tag_mem_to_")
        assert count >= 3, f"Expected tag threaded across ≥3 stage boundaries, got {count}"

    def test_lint_clean(self):
        _verilator_lint(run_async_pipeline_synth(_AutoThread5))


class TestTier2MultiplyAccum:
    """4-stage multiply-accumulate pipeline."""

    def test_four_stages(self):
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _MultiplyAccum4
        ir = AsyncPipelineElaboratePass(cfg).run(ir)
        ir = AsyncPipelineToIrPass(cfg).run(ir)
        assert len(ir.pipeline_ir.stages) == 4

    def test_acc_out_port_present(self):
        sv = run_async_pipeline_synth(_MultiplyAccum4)
        assert "acc_out" in sv

    def test_lint_clean(self):
        _verilator_lint(run_async_pipeline_synth(_MultiplyAccum4))


class TestTier2WidthVariety:
    """Pipeline with u8/u16/u32 variables."""

    def test_lint_clean(self):
        _verilator_lint(run_async_pipeline_synth(_WidthVariety))

    def test_u8_channel_width(self):
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _WidthVariety
        for cls in [AsyncPipelineElaboratePass, AsyncPipelineToIrPass]:
            ir = cls(cfg).run(ir)
        byte_chs = [c for c in ir.pipeline_ir.channels if c.name.startswith("b_")]
        assert any(c.width == 8 for c in byte_chs), \
            "Expected u8 channel width=8"


class TestTier2MultiCyclePipe:
    """Pipeline with a multi-cycle stage."""

    def test_lint_clean(self):
        _verilator_lint(run_async_pipeline_synth(_MultiCyclePipe))

    def test_cycle_hi_set(self):
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _MultiCyclePipe
        for cls in [AsyncPipelineElaboratePass, AsyncPipelineToIrPass]:
            ir = cls(cfg).run(ir)
        compute = next(s for s in ir.pipeline_ir.stages if s.name == "COMPUTE")
        assert compute.cycle_hi == 2  # cycles=3 → cycle_hi=2


# ---------------------------------------------------------------------------
# Tier 3 — System tests: realistic designs, full Verilator lint
# ---------------------------------------------------------------------------

class TestTier3SystemPassThrough:
    """Smoke: pass-through with valid signal propagation."""

    def test_module_valid_chain(self):
        sv = run_async_pipeline_synth(_PassThrough3)
        # All three stages should have valid regs
        assert sv.count("_valid_q") >= 3

    def test_always_posedge(self):
        sv = run_async_pipeline_synth(_PassThrough3)
        assert "always @(posedge clk)" in sv

    def test_async_reset(self):
        sv = run_async_pipeline_synth(_PassThrough3)
        assert "rst_n" in sv
        assert "!rst_n" in sv or "~rst_n" in sv


class TestTier3SystemAdder:
    """Adder pipeline: functional correctness checks on Verilog structure."""

    def test_input_ports_declared(self):
        sv = run_async_pipeline_synth(_Adder3)
        assert "input  wire [31:0] a_in" in sv or "input wire [31:0] a_in" in sv
        assert "input  wire [31:0] b_in" in sv or "input wire [31:0] b_in" in sv

    def test_output_port_declared(self):
        sv = run_async_pipeline_synth(_Adder3)
        assert "sum_out" in sv

    def test_addition_in_exec_block(self):
        sv = run_async_pipeline_synth(_Adder3)
        # The EXEC stage body should contain an addition
        assert "+" in sv

    def test_no_latch_warning_from_verilator(self):
        """Verilator clean means no latch inferred."""
        _verilator_lint(run_async_pipeline_synth(_Adder3))


class TestTier3SystemAutoThread:
    """Auto-threading: tag variable skips middle stages."""

    def test_tag_available_in_wb(self):
        sv = run_async_pipeline_synth(_AutoThread5)
        assert "tag_out" in sv

    def test_data_passes_through_all_stages(self):
        sv = run_async_pipeline_synth(_AutoThread5)
        assert "data_out" in sv

    def test_five_valid_regs(self):
        sv = run_async_pipeline_synth(_AutoThread5)
        count = sv.count("_valid_q")
        assert count >= 5, f"Expected ≥5 valid regs for 5-stage pipeline, got {count}"


# ---------------------------------------------------------------------------
# Tier 2 — SVCodegen extension tests: multi-cycle and bubble RTL
# ---------------------------------------------------------------------------

class TestTier2SVCodegenMultiCycle:
    """Validate SVEmitPass multi-cycle counter emission for _MultiCyclePipe."""

    def setup_method(self):
        self.sv = run_async_pipeline_synth(_MultiCyclePipe)

    def test_cycle_counter_reg_emitted(self):
        """Counter register declared with correct width for 3-cycle stage."""
        # cycle_hi=2 → need 2 bits for 0..2
        assert "compute_cycle_q" in self.sv

    def test_cycle_counter_width(self):
        """Counter width is ceil(log2(2+2)) = 2 bits for 3-cycle stage."""
        assert "[1:0] compute_cycle_q" in self.sv

    def test_done_wire_emitted(self):
        """_done fires when valid_q && cycle_q == cycle_hi."""
        assert "compute_done" in self.sv
        assert "compute_cycle_q == 2'd2" in self.sv

    def test_mc_stall_wire_emitted(self):
        """_mc_stall fires while still counting."""
        assert "compute_mc_stall" in self.sv
        assert "compute_cycle_q < 2'd2" in self.sv

    def test_counter_increments_in_always_block(self):
        """Counter increments inside an always @(posedge clk) block."""
        assert "compute_cycle_q <= compute_cycle_q + 2'd1" in self.sv

    def test_counter_resets_when_done(self):
        """Counter wraps back to 0 when it reaches cycle_hi."""
        assert "compute_cycle_q <= 2'b0;  // reset when done" in self.sv

    def test_wb_valid_uses_done_not_valid_q(self):
        """WB stage takes its valid from compute_done (not compute_valid_q)."""
        assert "wb_valid_q <= compute_done" in self.sv

    def test_upstream_frozen_by_mc_stall(self):
        """FETCH valid register is frozen when compute_mc_stall is high."""
        assert "compute_mc_stall" in self.sv
        # Either fetch_valid_q <= fetch_valid_q or an if-freeze construct
        assert "fetch_valid_q" in self.sv

    def test_lint_clean(self):
        """Multi-cycle counter RTL is Verilator-lint-clean."""
        _verilator_lint(self.sv)


class TestTier2SVCodegenBubble:
    """Validate SVEmitPass bubble wire emission for _BubblePipe."""

    def setup_method(self):
        self.sv = run_async_pipeline_synth(_BubblePipe)

    def test_bubble_wire_emitted(self):
        """_bubble wire is declared for the FILTER stage."""
        assert "filter_bubble" in self.sv

    def test_bubble_stages_populated(self):
        """PipelineIR.bubble_stages includes FILTER."""
        pip, _ = run_async_pipeline_synth(_BubblePipe, return_ir=True)
        assert "FILTER" in pip.bubble_stages

    def test_wb_valid_gated_on_bubble(self):
        """WB valid is conditioned on !filter_bubble."""
        assert "filter_bubble" in self.sv
        # The valid chain should gate next stage on !bubble
        assert "filter_valid_q && !filter_bubble" in self.sv or \
               "!filter_bubble" in self.sv

    def test_stall_gen_source_valid(self):
        """StallGenPass sets source_valid for WB to include !filter_bubble."""
        pip, _ = run_async_pipeline_synth(_BubblePipe, return_ir=True)
        vc = pip.valid_chain
        wb_entry = next(e for e in vc if e.stage_name == "WB")
        assert "filter_bubble" in wb_entry.source_valid

    def test_lint_clean(self):
        """Bubble pipeline RTL is Verilator-lint-clean."""
        _verilator_lint(self.sv)


class TestTier3SystemMultiCycle:
    """System-level tests for multi-cycle pipeline."""

    def test_counter_always_block_before_valid_chain(self):
        """Counter always block appears before the valid chain block."""
        sv = run_async_pipeline_synth(_MultiCyclePipe)
        mc_pos = sv.find("compute_cycle_q")
        vc_pos = sv.find("// \u2500\u2500 Valid-signal chain")
        assert mc_pos < vc_pos, "MC counter should be emitted before valid chain"

    def test_valid_chain_uses_done_signal(self):
        """The valid chain references compute_done for the WB stage."""
        sv = run_async_pipeline_synth(_MultiCyclePipe)
        # Find the valid chain section
        vc_start = sv.find("// \u2500\u2500 Valid-signal chain")
        vc_end   = sv.find("// \u2500\u2500 Stage", vc_start + 1)
        vc_text  = sv[vc_start:vc_end]
        assert "compute_done" in vc_text

    def test_mc_stall_freezes_fetch(self):
        """FETCH frozen by compute_mc_stall is encoded in the valid chain."""
        sv = run_async_pipeline_synth(_MultiCyclePipe)
        vc_start = sv.find("// \u2500\u2500 Valid-signal chain")
        vc_end   = sv.find("// \u2500\u2500 Stage", vc_start + 1)
        vc_text  = sv[vc_start:vc_end]
        assert "compute_mc_stall" in vc_text

    def test_compute_mc_stall_freezes_compute_too(self):
        """COMPUTE stage itself is also frozen by compute_mc_stall."""
        sv = run_async_pipeline_synth(_MultiCyclePipe)
        vc_start = sv.find("// \u2500\u2500 Valid-signal chain")
        vc_end   = sv.find("// \u2500\u2500 Stage", vc_start + 1)
        vc_text  = sv[vc_start:vc_end]
        # mc_stall should appear more than once (freezes both FETCH and COMPUTE)
        assert vc_text.count("compute_mc_stall") >= 2


# ---------------------------------------------------------------------------
# Tier 2: Regfile hazard pass integration
# ---------------------------------------------------------------------------

def _run_to_ir(cls, forward_default: bool = True):
    """Run AsyncPipelineElaboratePass + AsyncPipelineToIrPass and return SynthIR."""
    from zuspec.synth.passes.async_pipeline_elaborate import AsyncPipelineElaboratePass
    from zuspec.synth.passes.async_pipeline_to_ir import AsyncPipelineToIrPass
    cfg = SynthConfig(forward_default=forward_default)
    ir = SynthIR()
    ir.component = cls
    for pass_cls in [AsyncPipelineElaboratePass, AsyncPipelineToIrPass]:
        ir = pass_cls(cfg).run(ir)
    return ir, cfg


class TestTier2RegfileHazard:
    """Tests that regfile block/write ops flow through HazardAnalysisPass correctly."""

    def test_regfile_accesses_populated_by_to_ir_pass(self):
        """AsyncPipelineToIrPass populates regfile_accesses from IrHazardOp nodes."""
        ir, _ = _run_to_ir(_RegfilePipe)
        pip = ir.pipeline_ir
        assert pip is not None, "pipeline_ir must be set"
        assert len(pip.regfile_accesses) > 0, "regfile_accesses must be populated"

    def test_regfile_accesses_preserved_after_hazard_pass(self):
        """HazardAnalysisPass preserves regfile_accesses set by ToIrPass."""
        from zuspec.synth.passes.hazard_analysis import HazardAnalysisPass
        ir, cfg = _run_to_ir(_RegfilePipe)
        accesses_before = list(ir.pipeline_ir.regfile_accesses)
        ir = HazardAnalysisPass(cfg).run(ir)
        assert len(ir.pipeline_ir.regfile_accesses) == len(accesses_before), \
            "HazardAnalysisPass must not discard pre-populated regfile_accesses"

    def test_regfile_decls_preserved_after_hazard_pass(self):
        """HazardAnalysisPass preserves regfile_decls set by ToIrPass."""
        from zuspec.synth.passes.hazard_analysis import HazardAnalysisPass
        ir, cfg = _run_to_ir(_RegfilePipe)
        decls_before = list(ir.pipeline_ir.regfile_decls)
        ir = HazardAnalysisPass(cfg).run(ir)
        assert len(ir.pipeline_ir.regfile_decls) == len(decls_before), \
            "HazardAnalysisPass must not discard pre-populated regfile_decls"

    def test_block_op_produces_read_access(self):
        """block() in ID stage creates a read RegFileAccess."""
        ir, _ = _run_to_ir(_RegfilePipe)
        reads = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "read"]
        assert len(reads) >= 1, "block() must produce a read RegFileAccess"

    def test_write_op_produces_write_access(self):
        """write() in WB stage creates a write RegFileAccess."""
        ir, _ = _run_to_ir(_RegfilePipe)
        writes = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "write"]
        assert len(writes) >= 1, "write() must produce a write RegFileAccess"

    def test_regfile_pipeline_sv_lint_clean(self):
        """Full pass chain for _RegfilePipe produces Verilator-lint-clean Verilog."""
        sv = run_async_pipeline_synth(_RegfilePipe)
        assert "endmodule" in sv
        _verilator_lint(sv)


# ---------------------------------------------------------------------------
# Tier 2 — PipelineResource with BypassLock and QueueLock
# ---------------------------------------------------------------------------

class TestTier2PipelineResource:
    """Tests that PipelineResource fields generate correct IR and Verilog."""

    def test_bypass_lock_decl_has_correct_lock_type(self):
        """RegFileDeclInfo for BypassLock resource has lock_type='bypass'."""
        ir, _ = _run_to_ir(_BypassLockPipe)
        pip = ir.pipeline_ir
        decls = pip.regfile_decls
        assert len(decls) >= 1, "regfile_decls must be populated"
        d = next(d for d in decls if d.field_name == "rf")
        assert d.lock_type == "bypass", f"expected 'bypass', got {d.lock_type!r}"

    def test_bypass_lock_decl_depth(self):
        """BypassLock resource with size=16 → depth=16, addr_width=4."""
        ir, _ = _run_to_ir(_BypassLockPipe)
        pip = ir.pipeline_ir
        d = next(d for d in pip.regfile_decls if d.field_name == "rf")
        assert d.depth == 16, f"expected depth=16, got {d.depth}"
        assert d.addr_width == 4, f"expected addr_width=4, got {d.addr_width}"

    def test_queue_lock_decl_has_correct_lock_type(self):
        """RegFileDeclInfo for QueueLock resource has lock_type='queue'."""
        ir, _ = _run_to_ir(_QueueLockPipe)
        pip = ir.pipeline_ir
        decls = pip.regfile_decls
        assert len(decls) >= 1, "regfile_decls must be populated"
        d = next(d for d in decls if d.field_name == "rf")
        assert d.lock_type == "queue", f"expected 'queue', got {d.lock_type!r}"

    def test_bypass_lock_hazard_resolved_as_forward(self):
        """BypassLock RAW hazard resolves to 'forward' (bypass mux)."""
        from zuspec.synth.passes import HazardAnalysisPass, ForwardingGenPass
        ir, cfg = _run_to_ir(_BypassLockPipe)
        ir = HazardAnalysisPass(cfg).run(ir)
        ir = ForwardingGenPass(cfg).run(ir)
        pip = ir.pipeline_ir
        hazards = pip.regfile_hazards
        fwd = [h for h in hazards if h.resolved_by == "forward"]
        assert len(fwd) >= 1, "BypassLock hazard must resolve to 'forward'"

    def test_queue_lock_hazard_resolved_as_stall(self):
        """QueueLock RAW hazard resolves to 'stall' (no bypass mux)."""
        from zuspec.synth.passes import HazardAnalysisPass, ForwardingGenPass
        ir, cfg = _run_to_ir(_QueueLockPipe)
        ir = HazardAnalysisPass(cfg).run(ir)
        ir = ForwardingGenPass(cfg).run(ir)
        pip = ir.pipeline_ir
        hazards = pip.regfile_hazards
        stall = [h for h in hazards if h.resolved_by == "stall"]
        assert len(stall) >= 1, "QueueLock hazard must resolve to 'stall'"

    def test_bypass_lock_sv_lint_clean(self):
        """_BypassLockPipe synthesizes to Verilator-lint-clean Verilog."""
        sv = run_async_pipeline_synth(_BypassLockPipe)
        assert "endmodule" in sv
        _verilator_lint(sv)

    def test_queue_lock_sv_lint_clean(self):
        """_QueueLockPipe synthesizes to Verilator-lint-clean Verilog."""
        sv = run_async_pipeline_synth(_QueueLockPipe)
        assert "endmodule" in sv
        _verilator_lint(sv)

    def test_bypass_lock_sv_has_bypass_mux(self):
        """_BypassLockPipe Verilog contains a bypass forwarding mux."""
        sv = run_async_pipeline_synth(_BypassLockPipe)
        assert "bypass from" in sv or ("if (" in sv and "_mem[" in sv), \
            "bypass mux should reference memory array and a conditional"

    def test_queue_lock_sv_no_bypass_mux(self):
        """_QueueLockPipe Verilog has a direct memory read (no forwarding branch)."""
        sv = run_async_pipeline_synth(_QueueLockPipe)
        # Direct read: rf_mem[rs1_id] without a bypass if-else
        assert "rf_mem[" in sv, "QueueLock must emit a direct memory read"

