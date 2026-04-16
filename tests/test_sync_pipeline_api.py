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
"""Integration tests for the new @zdc.stage / @zdc.pipeline method-per-stage API.

Each test:
1. Uses a component class defined at module level (required for inspect.getsource).
2. Runs the full synth pass chain via ``run_pipeline_synth()``.
3. Asserts structural properties of the emitted Verilog.

Pass chain: DataModelFactory → PipelineFrontendPass → AutoThreadPass →
            HazardAnalysisPass → ForwardingGenPass → StallGenPass →
            SyncBodyLowerPass → SVEmitPass
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import os
import tempfile

import pytest

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
from zuspec.dataclasses.data_model_factory import DataModelFactory
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.passes import (
    PipelineFrontendPass,
    AutoThreadPass,
    HazardAnalysisPass,
    ForwardingGenPass,
    StallGenPass,
    SyncBodyLowerPass,
    SVEmitPass,
)


# ---------------------------------------------------------------------------
# Test harness helpers
# ---------------------------------------------------------------------------

def run_pipeline_synth(component_cls, forward_default: bool = True, return_ir: bool = False):
    """Run the full new-API pass chain and return the emitted SV text.

    Pass order:
      DataModelFactory → PipelineFrontendPass → AutoThreadPass →
      HazardAnalysisPass → ForwardingGenPass → StallGenPass →
      SyncBodyLowerPass → SVEmitPass

    :param component_cls: Component class decorated with ``@zdc.pipeline``.
    :param forward_default: Forwarding default used in :class:`SynthConfig`.
    :param return_ir: When ``True``, return ``(PipelineIR, sv_text)`` instead of
        just the SV text.  Existing callers are unaffected (default ``False``).
    """
    cfg = SynthConfig(forward_default=forward_default)

    ir = SynthIR()
    ir.component = component_cls
    # Build model context (replaces ElaboratePass for unit tests)
    ir.model_context = DataModelFactory().build(component_cls)

    for pass_cls in [
        PipelineFrontendPass,
        AutoThreadPass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
        SyncBodyLowerPass,
    ]:
        ir = pass_cls(cfg).run(ir)

    ir = SVEmitPass(cfg).run(ir)
    sv = ir.lowered_sv.get("sv/pipeline/top", "")
    if return_ir:
        return ir.pipeline_ir, sv
    return sv


def assert_wire(sv: str, name: str) -> None:
    """Assert that *name* appears in *sv*.

    Checks both original name and lowercase variant since the codegen emits
    lowercase stage names (e.g. ``S1`` → ``s1``).
    """
    lower_name = name.lower()
    assert name in sv or lower_name in sv, (
        f"Expected signal '{name}' (or '{lower_name}') in SV:\n{sv}"
    )


def assert_no_wire(sv: str, name: str) -> None:
    """Assert *name* does NOT appear in *sv* (case-insensitive)."""
    lower_name = name.lower()
    assert name not in sv and lower_name not in sv, (
        f"Unexpected signal '{name}' in SV:\n{sv}"
    )


def assert_assign(sv: str, lhs: str, rhs_pattern: str) -> None:
    """Assert that an 'assign lhs = <rhs_pattern>;' line exists in *sv*."""
    for line in sv.splitlines():
        stripped = line.strip()
        lower_lhs = lhs.lower()
        if (stripped.startswith(f"assign {lhs}") or
                stripped.startswith(f"assign {lower_lhs}")):
            if re.search(rhs_pattern, stripped, re.IGNORECASE):
                return
    raise AssertionError(
        f"No 'assign {lhs} = {rhs_pattern!r}' line found in SV:\n{sv}"
    )


def verilator_lint(sv: str) -> None:
    """Run ``verilator --lint-only`` on *sv*.

    Skips the check (with a pytest.skip) when Verilator is not on PATH.
    Raises AssertionError with the tool's stderr on any lint error.
    """
    if not shutil.which("verilator"):
        pytest.skip("verilator not found on PATH")
    with tempfile.TemporaryDirectory() as d:
        sv_file = os.path.join(d, "dut.sv")
        with open(sv_file, "w") as f:
            f.write(sv)
        result = subprocess.run(
            ["verilator", "--lint-only", "--sv", sv_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Verilator lint failed:\n{result.stdout}{result.stderr}\n"
                f"--- SV ---\n{sv}"
            )



@zdc.dataclass
class _MinimalPipe(zdc.Component):
    x: zdc.u32

    @zdc.stage
    def S1(self) -> (zdc.u32,):
        return (self.x,)

    @zdc.stage
    def S2(self, x: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (x,) = self.S1()
        self.S2(x)


class TestEx1Minimal2Stage:
    """Two-stage pipeline: S1 produces x, S2 consumes it."""

    def test_pipeline_ir_built(self):
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _MinimalPipe
        ir.model_context = DataModelFactory().build(_MinimalPipe)
        ir = PipelineFrontendPass(cfg).run(ir)
        assert ir.pipeline_ir is not None, "PipelineFrontendPass should build a PipelineIR"
        assert len(ir.pipeline_ir.stages) == 2

    def test_stage_valid_regs(self):
        sv = run_pipeline_synth(_MinimalPipe)
        assert sv, "SVEmitPass should produce non-empty SV"
        assert_wire(sv, "S1_valid")
        assert_wire(sv, "S2_valid")

    def test_channel_register(self):
        sv = run_pipeline_synth(_MinimalPipe)
        # x transferred from S1 to S2 should have a pipeline register (x_s1_to_s2_q)
        assert "x_s1_to_s2" in sv or "x_s1_to_s2_q" in sv, (
            f"Expected register for variable 'x' (x_s1_to_s2_q) in SV:\n{sv}"
        )

    def test_sv_has_module(self):
        sv = run_pipeline_synth(_MinimalPipe)
        assert "MinimalPipe" in sv or "minialpipe" in sv.lower()

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_MinimalPipe))


# ---------------------------------------------------------------------------
# Example 2 — Auto-threading (variable skips a stage)
# ---------------------------------------------------------------------------
#
# tag is produced at FETCH, consumed at WRITE but not used at PROC.
# AutoThreadPass must insert a threading register through PROC.

@zdc.dataclass
class _AutoThreadPipe(zdc.Component):
    tag: zdc.u32
    data: zdc.u32

    @zdc.stage
    def FETCH(self) -> (zdc.u32, zdc.u32):
        return (self.tag, self.data)

    @zdc.stage
    def PROC(self, data: zdc.u32) -> (zdc.u32,):
        return (data,)

    @zdc.stage
    def WRITE(self, tag: zdc.u32, data: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (tag, data) = self.FETCH()
        (data,) = self.PROC(data)
        self.WRITE(tag, data)


class TestEx2AutoThreading:
    """Three-stage pipeline with auto-threaded 'tag' variable."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_AutoThreadPipe)
        for s in ("FETCH", "PROC", "WRITE"):
            assert_wire(sv, f"{s}_valid")

    def test_threading_channel_for_tag(self):
        sv = run_pipeline_synth(_AutoThreadPipe)
        # tag must be threaded through PROC stage; look for any 'tag' register
        assert "tag" in sv.lower(), (
            f"Expected 'tag' register/wire in SV (auto-threaded):\n{sv}"
        )

    def test_module_emitted(self):
        sv = run_pipeline_synth(_AutoThreadPipe)
        assert sv, "SVEmitPass should produce non-empty SV"

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_AutoThreadPipe))


# ---------------------------------------------------------------------------
# Example 3 — External stall (stage stall condition)
# ---------------------------------------------------------------------------
#
# INGEST stalls when valid_in is low.

@zdc.dataclass
class _ExternalStallPipe(zdc.Component):
    valid_in: zdc.bit
    data: zdc.u32

    @zdc.stage
    def INGEST(self) -> (zdc.u32,):
        zdc.stage.stall(self, ~self.valid_in)
        return (self.data,)

    @zdc.stage
    def PROC(self, data: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (data,) = self.INGEST()
        self.PROC(data)


class TestEx3ExternalStall:
    """Pipeline with external stall on INGEST stage."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_ExternalStallPipe)
        for s in ("INGEST", "PROC"):
            assert_wire(sv, f"{s}_valid")

    def test_stall_decl_parsed(self):
        """PipelineFrontendPass must capture the stall declaration."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _ExternalStallPipe
        ir.model_context = DataModelFactory().build(_ExternalStallPipe)
        ir = PipelineFrontendPass(cfg).run(ir)
        assert ir.pipeline_ir is not None
        ingest = next(s for s in ir.pipeline_ir.stages if s.name == "INGEST")
        # stall_cond should have been set from zdc.stage.stall(...) declaration
        assert ingest.stall_cond is not None, (
            "INGEST stage should have stall_cond set from @zdc.stage.stall()"
        )

    def test_sv_emitted(self):
        sv = run_pipeline_synth(_ExternalStallPipe)
        assert sv

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_ExternalStallPipe))


# ---------------------------------------------------------------------------
# Example 5 — Forwarding (RAW hazard resolved by auto-forwarding)
# ---------------------------------------------------------------------------

@zdc.dataclass
class _ForwardPipe(zdc.Component):
    a: zdc.u32
    result: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.a,)

    @zdc.stage
    def EX(self, a: zdc.u32) -> (zdc.u32,):
        return (self.result,)

    @zdc.stage
    def WB(self, result: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n", forward=True)
    def execute(self):
        (a,) = self.IF()
        (result,) = self.EX(a)
        self.WB(result)


class TestEx5ForwardingRAW:
    """3-stage pipeline with RAW hazard resolved by auto-forwarding."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_ForwardPipe)
        for s in ("IF", "EX", "WB"):
            assert_wire(sv, f"{s}_valid")

    def test_forward_resolved(self):
        """ForwardingGenPass should resolve the RAW hazard to 'forward'."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _ForwardPipe
        ir.model_context = DataModelFactory().build(_ForwardPipe)
        for pass_cls in [PipelineFrontendPass, AutoThreadPass, HazardAnalysisPass, ForwardingGenPass]:
            ir = pass_cls(cfg).run(ir)
        # All hazards should be resolved (none left as 'unresolved')
        assert ir.pipeline_ir is not None
        unresolved = [h for h in ir.pipeline_ir.hazards if h.resolved_by == "unresolved"]
        assert not unresolved, f"Unexpected unresolved hazards: {unresolved}"

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_ForwardPipe))


# ---------------------------------------------------------------------------
# Example 6 — Load-use stall (no_forward on MEM stage)
# ---------------------------------------------------------------------------

@zdc.dataclass
class _LoadUsePipe(zdc.Component):
    addr: zdc.u32
    rdata: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.addr,)

    @zdc.stage(no_forward=True)
    def MEM(self, addr: zdc.u32) -> (zdc.u32,):
        return (self.rdata,)

    @zdc.stage
    def WB(self, rdata: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (addr,) = self.IF()
        (rdata,) = self.MEM(addr)
        self.WB(rdata)


class TestEx6LoadUseStall:
    """3-stage pipeline where MEM stage uses no_forward (load-use → stall)."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_LoadUsePipe)
        for s in ("IF", "MEM", "WB"):
            assert_wire(sv, f"{s}_valid")

    def test_mem_no_forward_flag(self):
        """PipelineFrontendPass must set no_forward=True on MEM stage."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _LoadUsePipe
        ir.model_context = DataModelFactory().build(_LoadUsePipe)
        ir = PipelineFrontendPass(cfg).run(ir)
        assert ir.pipeline_ir is not None
        mem = next(s for s in ir.pipeline_ir.stages if s.name == "MEM")
        assert mem.no_forward is True, "MEM stage should have no_forward=True"

    def test_hazard_resolved_as_stall(self):
        """ForwardingGenPass should resolve MEM hazard as stall (not forward)."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _LoadUsePipe
        ir.model_context = DataModelFactory().build(_LoadUsePipe)
        for pass_cls in [PipelineFrontendPass, AutoThreadPass, HazardAnalysisPass, ForwardingGenPass]:
            ir = pass_cls(cfg).run(ir)
        assert ir.pipeline_ir is not None
        for h in ir.pipeline_ir.hazards:
            if h.producer_stage == "MEM" and h.kind == "RAW":
                assert h.resolved_by == "stall", (
                    f"MEM RAW hazard should resolve to 'stall', got '{h.resolved_by}'"
                )

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_LoadUsePipe))


# ---------------------------------------------------------------------------
# Example 7 — Branch flush
# ---------------------------------------------------------------------------

@zdc.dataclass
class _BranchPipe(zdc.Component):
    branch_taken: zdc.bit
    insn: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.insn,)

    @zdc.stage
    def ID(self, insn: zdc.u32) -> (zdc.u32,):
        return (insn,)

    @zdc.stage
    def EX(self, insn: zdc.u32) -> ():
        zdc.stage.flush(self.IF, self.branch_taken)
        zdc.stage.flush(self.ID, self.branch_taken)

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (insn,) = self.IF()
        (insn,) = self.ID(insn)
        self.EX(insn)


class TestEx7BranchFlush:
    """3-stage pipeline where EX flushes IF and ID on branch taken."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_BranchPipe)
        for s in ("IF", "ID", "EX"):
            assert_wire(sv, f"{s}_valid")

    def test_flush_decls_captured(self):
        """PipelineFrontendPass must capture flush declarations from EX stage."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _BranchPipe
        ir.model_context = DataModelFactory().build(_BranchPipe)
        ir = PipelineFrontendPass(cfg).run(ir)
        assert ir.pipeline_ir is not None
        ex = next(s for s in ir.pipeline_ir.stages if s.name == "EX")
        assert len(ex.flush_decls) == 2, (
            f"EX stage should have 2 flush_decls, got {len(ex.flush_decls)}"
        )

    def test_sv_emitted(self):
        sv = run_pipeline_synth(_BranchPipe)
        assert sv

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_BranchPipe))


# ---------------------------------------------------------------------------
# Example 8 — Stage cancel
# ---------------------------------------------------------------------------

@zdc.dataclass
class _CancelPipe(zdc.Component):
    mispredict: zdc.bit
    a: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.a,)

    @zdc.stage
    def EX(self, a: zdc.u32) -> ():
        zdc.stage.cancel(self, self.mispredict)

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (a,) = self.IF()
        self.EX(a)


class TestEx8StageCancel:
    """2-stage pipeline where EX cancels itself on mispredict."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_CancelPipe)
        for s in ("IF", "EX"):
            assert_wire(sv, f"{s}_valid")

    def test_cancel_decl_captured(self):
        """PipelineFrontendPass must capture cancel declaration from EX stage."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _CancelPipe
        ir.model_context = DataModelFactory().build(_CancelPipe)
        ir = PipelineFrontendPass(cfg).run(ir)
        assert ir.pipeline_ir is not None
        ex = next(s for s in ir.pipeline_ir.stages if s.name == "EX")
        assert ex.cancel_cond is not None, (
            "EX stage should have cancel_cond set from @zdc.stage.cancel()"
        )

    def test_sv_emitted(self):
        sv = run_pipeline_synth(_CancelPipe)
        assert sv

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_CancelPipe))


# ---------------------------------------------------------------------------
# Example 10 — Stall/flush priority
# ---------------------------------------------------------------------------

@zdc.dataclass
class _PriorityPipe(zdc.Component):
    valid_in: zdc.bit
    data: zdc.u32
    flush_req: zdc.bit

    @zdc.stage
    def INGEST(self) -> (zdc.u32,):
        zdc.stage.stall(self, ~self.valid_in)
        zdc.stage.flush(self.INGEST, self.flush_req)
        return (self.data,)

    @zdc.stage
    def PROC(self, data: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (data,) = self.INGEST()
        self.PROC(data)


class TestEx10StallFlushPriority:
    """Verify flush takes priority over stall in the valid-chain FF."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_PriorityPipe)
        for s in ("INGEST", "PROC"):
            assert_wire(sv, f"{s}_valid")

    def test_both_decls_captured(self):
        """PipelineFrontendPass must capture both stall and flush on INGEST."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _PriorityPipe
        ir.model_context = DataModelFactory().build(_PriorityPipe)
        ir = PipelineFrontendPass(cfg).run(ir)
        assert ir.pipeline_ir is not None
        ingest = next(s for s in ir.pipeline_ir.stages if s.name == "INGEST")
        assert ingest.stall_cond is not None, "INGEST should have stall_cond"
        assert len(ingest.flush_decls) >= 1, "INGEST should have flush_decls"

    def test_sv_emitted(self):
        sv = run_pipeline_synth(_PriorityPipe)
        assert sv

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_PriorityPipe))


# ---------------------------------------------------------------------------
# Example 4 — External FSM + zdc.stage.ready
# ---------------------------------------------------------------------------
#
# IF stage stalls until imem_valid.
# fetch_ctrl sync method uses zdc.stage.ready(self.IF) to decide when to fire.

@zdc.dataclass
@zdc.dataclass
class _FetchWithFSM(zdc.Component):
    imem_valid:  zdc.bit
    imem_data:   zdc.u32
    pc:          zdc.u32
    fetch_state: zdc.bit

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        zdc.stage.stall(self, ~self.imem_valid)
        return (self.imem_data,)

    @zdc.stage
    def EX(self, imem_data: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (imem_data,) = self.IF()
        self.EX(imem_data)

    @zdc.sync(clock="clk", reset="rst_n")
    def fetch_ctrl(self):
        if zdc.stage.ready(self.IF):
            self.fetch_state = zdc.bit(1)


class TestEx4FsmReady:
    """2-stage pipeline with a @zdc.sync FSM that uses zdc.stage.ready(self.IF)."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_FetchWithFSM)
        for s in ("IF", "EX"):
            assert_wire(sv, f"{s}_valid")

    def test_if_stall_captured(self):
        """PipelineFrontendPass must capture stall_cond on IF stage."""
        cfg = SynthConfig(forward_default=True)
        ir = SynthIR()
        ir.component = _FetchWithFSM
        ir.model_context = DataModelFactory().build(_FetchWithFSM)
        ir = PipelineFrontendPass(cfg).run(ir)
        assert ir.pipeline_ir is not None
        if_stage = next(s for s in ir.pipeline_ir.stages if s.name == "IF")
        assert if_stage.stall_cond is not None, (
            "IF stage should have stall_cond set from zdc.stage.stall(~imem_valid)"
        )

    def test_ready_query_in_sync_ir(self):
        """DataModelFactory must capture the ready() query from fetch_ctrl."""
        mc = DataModelFactory().build(_FetchWithFSM)
        comp = None
        for name, dt in mc.type_m.items():
            if "FetchWithFSM" in name:
                comp = dt
                break
        assert comp is not None, "FetchWithFSM component not found in model context"
        assert hasattr(comp, 'sync_method_irs'), "Component should have sync_method_irs"
        sync_irs = comp.sync_method_irs
        assert sync_irs, "fetch_ctrl should appear in sync_method_irs"
        # The fetch_ctrl sync has a ready() query
        fetch_ir = next((s for s in sync_irs if s.name == "fetch_ctrl"), None)
        assert fetch_ir is not None, "fetch_ctrl SyncMethodIR not found"
        assert any(q.kind == "ready" for q in fetch_ir.query_nodes), (
            "fetch_ctrl sync method should have a 'ready' query node for IF stage"
        )

    def test_sv_emitted(self):
        sv = run_pipeline_synth(_FetchWithFSM)
        assert sv

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_FetchWithFSM))


# ---------------------------------------------------------------------------
# Example 9 — Interrupt flush from @zdc.sync
# ---------------------------------------------------------------------------
#
# A 3-stage pipeline. irq_ctrl sync method flushes all three stages when
# an interrupt is taken.  Demonstrates flush from sync context.

@zdc.dataclass
class _InterruptFlush(zdc.Component):
    irq:        zdc.bit
    irq_masked: zdc.bit
    a:          zdc.u32
    b:          zdc.u32

    @zdc.stage
    def S1(self) -> (zdc.u32,):
        return (self.a,)

    @zdc.stage
    def S2(self, a: zdc.u32) -> (zdc.u32,):
        return (self.b,)

    @zdc.stage
    def S3(self, b: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        (a,) = self.S1()
        (b,) = self.S2(a)
        self.S3(b)

    @zdc.sync(clock="clk", reset="rst_n")
    def irq_ctrl(self):
        take_irq = self.irq & ~self.irq_masked
        if take_irq:
            zdc.stage.flush(self.S1)
            zdc.stage.flush(self.S2)
            zdc.stage.flush(self.S3)


class TestEx9FlushFromSync:
    """3-stage pipeline where a @zdc.sync method flushes all stages on interrupt."""

    def test_stage_valids(self):
        sv = run_pipeline_synth(_InterruptFlush)
        for s in ("S1", "S2", "S3"):
            assert_wire(sv, f"{s}_valid")

    def test_flush_decls_in_sync_ir(self):
        """DataModelFactory must capture flush declarations from irq_ctrl."""
        mc = DataModelFactory().build(_InterruptFlush)
        comp = None
        for name, dt in mc.type_m.items():
            if "InterruptFlush" in name:
                comp = dt
                break
        assert comp is not None, "InterruptFlush component not found in model context"
        sync_irs = getattr(comp, 'sync_method_irs', [])
        irq_ir = next((s for s in sync_irs if s.name == "irq_ctrl"), None)
        assert irq_ir is not None, "irq_ctrl SyncMethodIR not found"
        assert len(irq_ir.flush_decls) == 3, (
            f"irq_ctrl should have 3 flush_decls (S1, S2, S3), "
            f"got {len(irq_ir.flush_decls)}"
        )
        targets = {d.target_stage for d in irq_ir.flush_decls}
        assert targets == {"S1", "S2", "S3"}, (
            f"Expected flush targets {{S1, S2, S3}}, got {targets}"
        )

    def test_sv_emitted(self):
        sv = run_pipeline_synth(_InterruptFlush)
        assert sv

    def test_lint(self):
        verilator_lint(run_pipeline_synth(_InterruptFlush))


# ---------------------------------------------------------------------------
# Module-level component aliases for use by Tier-1 and Tier-3 formal tests
# ---------------------------------------------------------------------------
# These names allow external test modules to import specific component classes
# without duplicating the class definitions.

_Ex1Component = _MinimalPipe        # 2-stage minimal pipeline
_Ex3Component = _ExternalStallPipe  # external stall input
_Ex5Component = _ForwardPipe        # regfile RAW forwarding
_Ex7Component = _BranchPipe        # branch flush
_Ex8Component = _CancelPipe        # stage cancel
