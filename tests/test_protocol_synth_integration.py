"""Phase 7 integration tests — end-to-end synthesis pipeline.

Tests that take real ``@zdc.dataclass`` component classes, run the full
``ProtocolSynthPipeline``, and Verilator-lint the assembled SystemVerilog.

The pipeline exercises:
  DataModelFactory → IfProtocolLowerPass → QueueLowerPass →
  SpawnLowerPass → SelectLowerPass → CompletionAnalysisPass →
  ProtocolSVEmitPass → assemble_sv_module
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

import zuspec.dataclasses as zdc
from zuspec.synth.protocol_pipeline import ProtocolSynthPipeline, assemble_sv_module

# ---------------------------------------------------------------------------
# Locate Verilator
# ---------------------------------------------------------------------------

_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_VERILATOR_BIN = os.path.join(_repo_root, "packages", "verilator", "bin", "verilator")


def _verilator() -> str:
    """Return path to verilator or skip the test if not found."""
    if os.path.isfile(_VERILATOR_BIN):
        return _VERILATOR_BIN
    found = shutil.which("verilator")
    if found:
        return found
    pytest.skip("verilator not found")


def _lint(sv: str, top_module: str) -> None:
    """Verilator-lint *sv* (a string) and fail the test if there are errors."""
    verilator = _verilator()
    with tempfile.NamedTemporaryFile(suffix=".sv", mode="w", delete=False) as f:
        f.write(sv)
        sv_path = f.name
    try:
        result = subprocess.run(
            [verilator, "--lint-only", "--sv", "--top-module", top_module, sv_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Verilator lint failed for {top_module}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    finally:
        os.unlink(sv_path)


# ---------------------------------------------------------------------------
# Protocol definitions (defined at module level so DataModelFactory can
# retrieve source code via inspect.getsource)
# ---------------------------------------------------------------------------

class MemReadIface(zdc.IfProtocol, max_outstanding=1,
                   req_always_ready=False, resp_always_valid=False):
    """Scenario B — single-outstanding read."""
    async def read(self, addr: zdc.u32) -> zdc.u32: ...


class MultiReadIface(zdc.IfProtocol, max_outstanding=4, in_order=True,
                     req_always_ready=False, resp_always_valid=False):
    """Scenario C — multi-outstanding, in-order."""
    async def read(self, addr: zdc.u32, burst: zdc.u8) -> zdc.u32: ...


class OOOReadIface(zdc.IfProtocol, max_outstanding=4, in_order=False,
                   req_always_ready=False, resp_always_valid=False):
    """Scenario D — multi-outstanding, out-of-order."""
    async def read(self, addr: zdc.u32, id_: zdc.u8) -> zdc.u32: ...


class FastIface(zdc.IfProtocol, max_outstanding=1,
                req_always_ready=True, resp_always_valid=True,
                fixed_latency=2):
    """Scenario A — fixed latency, always-ready."""
    async def compute(self, data: zdc.u32) -> zdc.u32: ...


# ---------------------------------------------------------------------------
# Component classes
# ---------------------------------------------------------------------------

@zdc.dataclass
class ScenarioBComp(zdc.Component):
    """Single IfProtocol port (Scenario B) — no FIFOs."""
    mem: MemReadIface


@zdc.dataclass
class ScenarioCComp(zdc.Component):
    """Multi-outstanding in-order port (Scenario C)."""
    mem: MultiReadIface


@zdc.dataclass
class ScenarioAComp(zdc.Component):
    """Fixed-latency always-ready port (Scenario A)."""
    fast: FastIface


@zdc.dataclass
class QueueOnlyComp(zdc.Component):
    """Component with only Queue fields (no IfProtocol ports)."""
    req_q: zdc.Queue[zdc.u32] = zdc.queue(depth=4)
    resp_q: zdc.Queue[zdc.u64] = zdc.queue(depth=8)


@zdc.dataclass
class MixedComp(zdc.Component):
    """Component with both IfProtocol port and Queue fields."""
    mem: MultiReadIface
    req_q: zdc.Queue[zdc.u32] = zdc.queue(depth=4)
    resp_q: zdc.Queue[zdc.u64] = zdc.queue(depth=8)


@zdc.dataclass
class MultiPortComp(zdc.Component):
    """Component with two IfProtocol ports of different scenarios."""
    load: MultiReadIface
    store: OOOReadIface


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_pipeline(comp_cls) -> tuple[str, "SynthIR"]:  # type: ignore[name-defined]
    """Run ProtocolSynthPipeline and return (sv_text, ir)."""
    pipeline = ProtocolSynthPipeline(comp_cls)
    sv = pipeline.run()
    return sv, pipeline.ir


# ============================================================
# IR-level tests (no Verilator required)
# ============================================================

class TestPipelineIR:
    """Verify that the pipeline populates the SynthIR correctly."""

    def test_scenario_b_port_discovered(self):
        _, ir = _run_pipeline(ScenarioBComp)
        assert len(ir.protocol_ports) == 1
        assert ir.protocol_ports[0].name == "mem"

    def test_scenario_b_scenario_label(self):
        from zuspec.synth.ir.protocol_ir import IfProtocolScenario
        _, ir = _run_pipeline(ScenarioBComp)
        assert ir.protocol_ports[0].scenario == IfProtocolScenario.B

    def test_scenario_c_scenario_label(self):
        from zuspec.synth.ir.protocol_ir import IfProtocolScenario
        _, ir = _run_pipeline(ScenarioCComp)
        assert ir.protocol_ports[0].scenario == IfProtocolScenario.C

    def test_scenario_a_scenario_label(self):
        from zuspec.synth.ir.protocol_ir import IfProtocolScenario
        _, ir = _run_pipeline(ScenarioAComp)
        assert ir.protocol_ports[0].scenario == IfProtocolScenario.A

    def test_req_fields_extracted(self):
        _, ir = _run_pipeline(ScenarioBComp)
        req_names = [f.name for f in ir.protocol_ports[0].req_fields]
        assert "addr" in req_names

    def test_resp_fields_extracted(self):
        _, ir = _run_pipeline(ScenarioBComp)
        resp_names = [f.name for f in ir.protocol_ports[0].resp_fields]
        assert len(resp_names) >= 1  # at least one response field

    def test_queue_nodes_discovered(self):
        _, ir = _run_pipeline(QueueOnlyComp)
        assert len(ir.queue_nodes) == 2

    def test_queue_depths_correct(self):
        _, ir = _run_pipeline(QueueOnlyComp)
        depth_by_name = {q.name: q.depth for q in ir.queue_nodes}
        assert depth_by_name["req_q"] == 4
        assert depth_by_name["resp_q"] == 8

    def test_queue_widths_correct(self):
        _, ir = _run_pipeline(QueueOnlyComp)
        width_by_name = {q.name: q.elem_width for q in ir.queue_nodes}
        assert width_by_name["req_q"] == 32
        assert width_by_name["resp_q"] == 64

    def test_mixed_comp_ports_and_queues(self):
        _, ir = _run_pipeline(MixedComp)
        assert len(ir.protocol_ports) == 1
        assert len(ir.queue_nodes) == 2

    def test_multi_port_comp_two_ports(self):
        _, ir = _run_pipeline(MultiPortComp)
        assert len(ir.protocol_ports) == 2
        names = {p.name for p in ir.protocol_ports}
        assert names == {"load", "store"}

    def test_lowered_sv_port_keys(self):
        _, ir = _run_pipeline(ScenarioBComp)
        assert any(k.startswith("sv/port/") for k in ir.lowered_sv)

    def test_lowered_sv_fifo_keys(self):
        _, ir = _run_pipeline(QueueOnlyComp)
        assert any(k.startswith("sv/module/") for k in ir.lowered_sv)


# ============================================================
# SV content tests
# ============================================================

class TestAssembledSV:
    """Verify that the assembled SV text has correct structure."""

    def test_has_module_keyword(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        assert "module ScenarioBComp" in sv

    def test_has_clk_port(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        assert "clk" in sv

    def test_has_rst_port(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        assert "rst" in sv

    def test_scenario_b_has_req_valid(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        assert "mem_req_valid" in sv

    def test_scenario_b_has_resp_data(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        assert "mem_resp_data" in sv

    def test_queue_comp_has_fifo_module(self):
        sv, _ = _run_pipeline(QueueOnlyComp)
        assert "module req_q_fifo" in sv
        assert "module resp_q_fifo" in sv

    def test_no_trailing_comma_before_close_paren(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        # The last port before ); must not end with a comma
        for i, line in enumerate(sv.splitlines()):
            if line.strip() == ");":
                prev = sv.splitlines()[i - 1].strip()
                assert not prev.endswith(","), f"Trailing comma before ')': {prev!r}"
                break

    def test_endmodule_present(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        assert "endmodule" in sv

    def test_mixed_comp_sv_has_port_and_fifo(self):
        sv, _ = _run_pipeline(MixedComp)
        assert "mem_req_valid" in sv
        assert "module req_q_fifo" in sv


# ============================================================
# Verilator lint tests
# ============================================================

class TestSVLintIntegration:
    """Verilator-lint the complete assembled module for each scenario."""

    def test_scenario_b_lints(self):
        sv, _ = _run_pipeline(ScenarioBComp)
        _lint(sv, "ScenarioBComp")

    def test_scenario_c_lints(self):
        sv, _ = _run_pipeline(ScenarioCComp)
        _lint(sv, "ScenarioCComp")

    def test_queue_only_lints(self):
        sv, _ = _run_pipeline(QueueOnlyComp)
        _lint(sv, "QueueOnlyComp")

    def test_mixed_comp_lints(self):
        sv, _ = _run_pipeline(MixedComp)
        _lint(sv, "MixedComp")

    def test_multi_port_lints(self):
        sv, _ = _run_pipeline(MultiPortComp)
        _lint(sv, "MultiPortComp")

    def test_submodule_fifos_lint_individually(self):
        """Each generated FIFO submodule must lint as a standalone module."""
        from zuspec.synth.protocol_pipeline import ProtocolSynthPipeline
        pipeline = ProtocolSynthPipeline(QueueOnlyComp)
        pipeline.run()
        ir = pipeline.ir
        for key, fragments in ir.lowered_sv.items():
            if key.startswith("sv/module/") and "_fifo" in key:
                sv_frag = fragments[0]
                # Extract module name from first line
                mod_name = None
                for line in sv_frag.splitlines():
                    if line.startswith("module "):
                        mod_name = line.split()[1]
                        break
                assert mod_name is not None
                _lint(sv_frag, mod_name)
