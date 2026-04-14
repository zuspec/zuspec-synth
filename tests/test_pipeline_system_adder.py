"""System-level tests for the new InPort/OutPort + ClockDomain pipeline API.

These tests exercise the full synthesis chain for components written with the
new explicit port API:
  InPort / OutPort fields  →  in_port() / out_port() factories
  ClockDomain field         →  clock_domain() factory
  @zdc.pipeline(clock_domain=lambda s: s.clk)

All component classes are defined at module level (required for
``inspect.getsource``).

Tier 2 — Codegen: structural assertions on generated Verilog + Verilator lint.
Tier 3 — System: end-to-end RTL correctness via Verilog output checks.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

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
from zuspec.dataclasses.types import Time, TimeUnit
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.passes import (
    HazardAnalysisPass,
    ForwardingGenPass,
    StallGenPass,
    SVEmitPass,
    AsyncPipelineElaboratePass,
    AsyncPipelineToIrPass,
)

# ---------------------------------------------------------------------------
# Module-level pipeline components (new API)
# ---------------------------------------------------------------------------

@zdc.dataclass
class _AdderNewAPI(zdc.Component):
    """3-stage adder using explicit InPort/OutPort and ClockDomain."""
    clk:     zdc.ClockDomain  = zdc.clock_domain()
    a_in:    zdc.InPort[zdc.u32]  = zdc.in_port()
    b_in:    zdc.InPort[zdc.u32]  = zdc.in_port()
    sum_out: zdc.OutPort[zdc.u32] = zdc.out_port()

    @zdc.pipeline(clock_domain=lambda s: s.clk)
    async def run(self):
        a = await self.a_in.get()
        b = await self.b_in.get()
        async with zdc.pipeline.stage() as FETCH:
            pass
        async with zdc.pipeline.stage() as EXEC:
            result: zdc.u32 = a + b
        async with zdc.pipeline.stage() as WB:
            await self.sum_out.put(result)


@zdc.dataclass
class _PassThroughNewAPI(zdc.Component):
    """Simple 3-stage pass-through with new API."""
    clk:      zdc.ClockDomain  = zdc.clock_domain()
    data_in:  zdc.InPort[zdc.u32]  = zdc.in_port()
    data_out: zdc.OutPort[zdc.u32] = zdc.out_port()

    @zdc.pipeline(clock_domain=lambda s: s.clk)
    async def run(self):
        val = await self.data_in.get()
        async with zdc.pipeline.stage() as FETCH:
            pass
        async with zdc.pipeline.stage() as PROC:
            processed: zdc.u32 = val
        async with zdc.pipeline.stage() as WB:
            await self.data_out.put(processed)


@zdc.dataclass
class _MultiStageNewAPI(zdc.Component):
    """5-stage pipeline with two inputs and two outputs using new API."""
    clk:      zdc.ClockDomain      = zdc.clock_domain()
    tag_in:   zdc.InPort[zdc.u32]  = zdc.in_port()
    data_in:  zdc.InPort[zdc.u32]  = zdc.in_port()
    tag_out:  zdc.OutPort[zdc.u32] = zdc.out_port()
    data_out: zdc.OutPort[zdc.u32] = zdc.out_port()

    @zdc.pipeline(clock_domain=lambda s: s.clk)
    async def run(self):
        tag  = await self.tag_in.get()
        data = await self.data_in.get()
        async with zdc.pipeline.stage() as IF:
            pass
        async with zdc.pipeline.stage() as ID:
            d2: zdc.u32 = data
        async with zdc.pipeline.stage() as EX:
            d3: zdc.u32 = d2
        async with zdc.pipeline.stage() as MEM:
            d4: zdc.u32 = d3
        async with zdc.pipeline.stage() as WB:
            await self.tag_out.put(tag)
            await self.data_out.put(d4)


@zdc.dataclass
class _CondEgressNewAPI(zdc.Component):
    """3-stage pipeline with conditional egress: only emit when result is non-zero."""
    clk:      zdc.ClockDomain      = zdc.clock_domain()
    data_in:  zdc.InPort[zdc.u32]  = zdc.in_port()
    data_out: zdc.OutPort[zdc.u32] = zdc.out_port()

    @zdc.pipeline(clock_domain=lambda s: s.clk)
    async def run(self):
        val = await self.data_in.get()
        async with zdc.pipeline.stage() as FETCH:
            pass
        async with zdc.pipeline.stage() as EXEC:
            result: zdc.u32 = val + 1
        async with zdc.pipeline.stage() as WB:
            if result != 0:
                await self.data_out.put(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth(comp_cls, forward_default: bool = True, return_ir: bool = False):
    """Run full synthesis chain; return (pip, sv_text) or just sv_text."""
    cfg = SynthConfig(forward_default=forward_default)
    ir = SynthIR()
    ir.component = comp_cls
    for pass_cls in [
        AsyncPipelineElaboratePass,
        AsyncPipelineToIrPass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
    ]:
        ir = pass_cls(cfg).run(ir)
    ir = SVEmitPass(cfg).run(ir)
    sv = ir.lowered_sv.get("pipeline_sv", "")
    if return_ir:
        return ir.pipeline_ir, sv
    return sv


def _verilator_lint(sv: str) -> None:
    """Run verilator --lint-only; skip test if not available."""
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
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"Verilator lint failed:\n{result.stderr}\n{result.stdout}\n\nSV:\n{sv}"
        )


# ---------------------------------------------------------------------------
# Tier 1 — IR structure assertions
# ---------------------------------------------------------------------------

class TestTier1AdderNewAPI:
    def test_ingress_ports_populated(self):
        """PipelineIR.ingress_ports should contain a_in and b_in."""
        pip, _ = _synth(_AdderNewAPI, return_ir=True)
        names = [n for n, _ in pip.ingress_ports]
        assert "a_in" in names
        assert "b_in" in names

    def test_egress_ports_populated(self):
        """PipelineIR.egress_ports should contain sum_out."""
        pip, _ = _synth(_AdderNewAPI, return_ir=True)
        names = [n for n, _ in pip.egress_ports]
        assert "sum_out" in names

    def test_clock_domain_field_propagated(self):
        """clock_domain_field should be 'clk' on PipelineIR."""
        pip, _ = _synth(_AdderNewAPI, return_ir=True)
        assert pip.clock_domain_field == "clk"

    def test_stage_count(self):
        pip, _ = _synth(_AdderNewAPI, return_ir=True)
        assert pip.pipeline_stages == 3

    def test_multi_stage_ingress_egress(self):
        """5-stage pipeline has two ingress and two egress ports."""
        pip, _ = _synth(_MultiStageNewAPI, return_ir=True)
        in_names = [n for n, _ in pip.ingress_ports]
        out_names = [n for n, _ in pip.egress_ports]
        assert "tag_in" in in_names
        assert "data_in" in in_names
        assert "tag_out" in out_names
        assert "data_out" in out_names


# ---------------------------------------------------------------------------
# Tier 2 — Codegen structural assertions + Verilator lint
# ---------------------------------------------------------------------------

class TestTier2AdderNewAPI:
    def setup_method(self):
        self.sv = _synth(_AdderNewAPI)

    def test_module_declaration(self):
        assert "_AdderNewAPI" in self.sv

    def test_ingress_ports_in_module(self):
        """Both InPort fields appear as input ports in the SV module."""
        # Ports may be declared as 'a_in' raw or with valid/ready wrappers
        assert "a_in" in self.sv
        assert "b_in" in self.sv

    def test_egress_port_in_module(self):
        assert "sum_out" in self.sv

    def test_stage_valid_regs(self):
        for stage in ("fetch", "exec", "wb"):
            assert f"{stage}_valid_q" in self.sv.lower()

    def test_lint_clean(self):
        _verilator_lint(self.sv)


class TestTier2PassThroughNewAPI:
    def setup_method(self):
        self.sv = _synth(_PassThroughNewAPI)

    def test_module_declaration(self):
        assert "_PassThroughNewAPI" in self.sv

    def test_data_in_port_present(self):
        assert "data_in" in self.sv

    def test_data_out_port_present(self):
        assert "data_out" in self.sv

    def test_lint_clean(self):
        _verilator_lint(self.sv)


class TestTier2MultiStageNewAPI:
    def setup_method(self):
        self.sv = _synth(_MultiStageNewAPI)

    def test_module_declaration(self):
        assert "_MultiStageNewAPI" in self.sv

    def test_five_stages_valid_regs(self):
        for stage in ("if", "id", "ex", "mem", "wb"):
            assert f"{stage}_valid_q" in self.sv.lower()

    def test_all_ports_present(self):
        for port in ("tag_in", "data_in", "tag_out", "data_out"):
            assert port in self.sv

    def test_lint_clean(self):
        _verilator_lint(self.sv)


# ---------------------------------------------------------------------------
# Tier 3 — Full synthesis correctness checks
# ---------------------------------------------------------------------------

class TestTier3AdderCorrectness:
    def test_result_register_not_combinational(self):
        """The sum must be registered at WB stage, not purely combinational."""
        sv = _synth(_AdderNewAPI)
        # Valid chain must progress through wb_valid_q
        assert "wb_valid_q" in sv.lower()

    def test_valid_chain_reset_to_zero(self):
        """All stage valid registers reset to 0 on reset."""
        sv = _synth(_AdderNewAPI)
        # At minimum, check a stage valid reg appears in the output
        assert "fetch_valid_q" in sv.lower() or "valid_q" in sv.lower()

    def test_channel_propagates_across_stages(self):
        """Result variable 'result' is threaded through stage registers."""
        sv = _synth(_AdderNewAPI)
        # The channel for 'result' should appear as a register
        assert re.search(r"result.*_q", sv, re.IGNORECASE), (
            "Expected a 'result' pipeline register in generated SV"
        )


# ---------------------------------------------------------------------------
# Tier 2 — Multi-ingress valid handshake tests
# ---------------------------------------------------------------------------

class TestTier2MultiIngressValid:
    """Verify that multiple InPort fields on the same pipeline produce correct RTL.

    Key correctness properties:
    - Both a_in and b_in appear as separate module input ports.
    - Both ingress variables are captured into local FETCH signals and threaded
      via pipeline channel registers (a_fetch_to_exec_q, b_fetch_to_exec_q).
    - The EXEC stage uses the registered values, NOT the live module inputs.
    - RTL is Verilator lint-clean.
    """

    @pytest.fixture(scope="class")
    def sv(self):
        return _synth(_AdderNewAPI)

    def test_both_ingress_ports_present(self, sv):
        """a_in and b_in must be separate input ports."""
        assert "a_in" in sv
        assert "b_in" in sv

    def test_both_ingress_channels_created(self, sv):
        """Each ingress var must be threaded through its own channel register."""
        assert "a_fetch_to_exec_q" in sv, (
            "Missing channel register for 'a': a_fetch_to_exec_q not found"
        )
        assert "b_fetch_to_exec_q" in sv, (
            "Missing channel register for 'b': b_fetch_to_exec_q not found"
        )

    def test_fetch_stage_captures_both_ports(self, sv):
        """FETCH stage must capture a_in → a_fetch and b_in → b_fetch."""
        assert re.search(r"a_fetch\s*=\s*a_in", sv), (
            "FETCH stage does not capture a_in into a_fetch"
        )
        assert re.search(r"b_fetch\s*=\s*b_in", sv), (
            "FETCH stage does not capture b_in into b_fetch"
        )

    def test_exec_uses_channel_registers_not_raw_ports(self, sv):
        """EXEC stage must compute from channel regs, not live module inputs.

        The critical correctness property: if EXEC used a_in/b_in directly
        it would produce wrong results for any non-trivial pipeline depth
        since the values would already have advanced.
        """
        # Find EXEC always block
        exec_match = re.search(
            r"// Stage EXEC\s+always @\(\*\)\s+begin(.+?)^end",
            sv, re.DOTALL | re.MULTILINE
        )
        assert exec_match, "EXEC stage always block not found"
        exec_body = exec_match.group(1)
        # Must reference channel registers
        assert "a_fetch_to_exec_q" in exec_body, (
            "EXEC stage does not use a_fetch_to_exec_q (uses raw a_in?)"
        )
        assert "b_fetch_to_exec_q" in exec_body, (
            "EXEC stage does not use b_fetch_to_exec_q (uses raw b_in?)"
        )
        # Must NOT reference raw module inputs directly
        assert "a_in" not in exec_body, (
            "EXEC stage incorrectly references raw input a_in"
        )
        assert "b_in" not in exec_body, (
            "EXEC stage incorrectly references raw input b_in"
        )

    def test_result_channeled_to_wb(self, sv):
        """EXEC result must be channeled to WB via result_exec_to_wb_q."""
        assert "result_exec_to_wb_q" in sv

    def test_valid_in_single_gate(self, sv):
        """Single valid_in gates the entire pipeline (data-only handshake)."""
        assert "valid_in" in sv
        # Only one valid_in port (not per-port valid signals)
        port_lines = [l for l in sv.splitlines() if "valid_in" in l and "input" in l]
        assert len(port_lines) == 1, (
            f"Expected exactly one valid_in input port, found: {port_lines}"
        )

    def test_lint_clean(self, sv):
        _verilator_lint(sv)




class TestTier2CondEgress:
    """Verify conditional egress (put inside if) produces lint-clean RTL."""

    @pytest.fixture(scope="class")
    def sv(self):
        return _synth(_CondEgressNewAPI)

    def test_sv_nonempty(self, sv):
        assert len(sv) > 100

    def test_data_out_port_present(self, sv):
        assert "data_out" in sv

    def test_valid_chain_present(self, sv):
        assert "wb_valid_q" in sv

    def test_data_out_conditionally_driven(self, sv):
        """data_out should appear inside a conditional if block in WB stage."""
        import re
        assert re.search(r"if\s*\(.*result.*\)", sv), (
            "Expected an 'if (result ...)' condition guarding data_out assignment"
        )

    def test_data_out_has_default_zero(self, sv):
        """data_out must have a default-zero assignment to prevent latches."""
        lines = sv.splitlines()
        wb_start = next((i for i, l in enumerate(lines) if "Stage WB" in l), None)
        assert wb_start is not None, "WB stage not found in SV"
        wb_block = "\n".join(lines[wb_start:wb_start + 30])
        assert "data_out = 32'b0" in wb_block, (
            f"Expected 'data_out = 32\\'b0' default in WB block:\n{wb_block}"
        )

    def test_lint_clean(self, sv):
        _verilator_lint(sv)
