"""Tests for the lowered_sv key naming convention (Item 1 — ECO Foundation).

Verifies:
- SVEmitPass stores output under "sv/pipeline/top" (not the old "pipeline_sv").
- validate_lowered_sv_key() accepts valid hierarchical keys.
- validate_lowered_sv_key() rejects malformed / legacy keys.
"""
from __future__ import annotations

import os
import sys

import pytest

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
for _p in [_synth_src, _dc_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zuspec.dataclasses as zdc
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR, validate_lowered_sv_key
from zuspec.synth.passes import (
    AsyncPipelineElaboratePass,
    AsyncPipelineToIrPass,
    HazardAnalysisPass,
    ForwardingGenPass,
    StallGenPass,
    SVEmitPass,
)


# ---------------------------------------------------------------------------
# Minimal component for a 2-stage pipeline
# ---------------------------------------------------------------------------

@zdc.dataclass
class _LoweredSvTestComp(zdc.Component):
    pass


class _LoweredSvTestPipelineComp(_LoweredSvTestComp):
    @zdc.pipeline
    async def pipe(self, val: zdc.u32) -> zdc.u32:
        stage1 = val + zdc.u32(1)
        return stage1


def _run_sv_emit_pass(comp_cls):
    cfg = SynthConfig(forward_default=True)
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
    return ir


# ---------------------------------------------------------------------------
# Test: SVEmitPass uses the new canonical key
# ---------------------------------------------------------------------------

def test_sv_pipeline_top_key_present_after_sv_emit_pass():
    """SVEmitPass stores output under 'sv/pipeline/top', not 'pipeline_sv'."""
    ir = _run_sv_emit_pass(_LoweredSvTestPipelineComp)
    assert "sv/pipeline/top" in ir.lowered_sv, (
        "Expected 'sv/pipeline/top' in ir.lowered_sv; got keys: "
        + str(list(ir.lowered_sv.keys()))
    )
    assert "pipeline_sv" not in ir.lowered_sv, (
        "Legacy key 'pipeline_sv' must not be present — rename missed somewhere."
    )
    assert len(ir.lowered_sv["sv/pipeline/top"]) > 0


# ---------------------------------------------------------------------------
# Tests: validate_lowered_sv_key — valid keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "sv/pipeline/top",
    "sw/module/bus_bridge",
    "fv/assume/reset_seq",
    "trace/waveform/vcd",
    "sv/stage/FetchStage",
    "sv/regfile/regfile_rv",
    "sv/stage/_debug",          # underscore-prefixed item (experimental)
])
def test_validate_lowered_sv_key_accepts_valid_keys(key):
    """validate_lowered_sv_key raises nothing for correctly-formed keys."""
    validate_lowered_sv_key(key)  # must not raise


# ---------------------------------------------------------------------------
# Tests: validate_lowered_sv_key — invalid / legacy keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "pipeline_sv",          # legacy flat key
    "sv_pipeline_top",      # underscores instead of slashes
    "",                     # empty string
    "sv/",                  # trailing slash, missing item
    "/pipeline/top",        # leading slash
    "SV/pipeline/top",      # uppercase backend
    "sv/Pipeline/top",      # uppercase category (category must be lower)
])
def test_validate_lowered_sv_key_rejects_invalid_keys(key):
    """validate_lowered_sv_key raises ValueError for malformed keys."""
    with pytest.raises(ValueError, match="lowered_sv key"):
        validate_lowered_sv_key(key)
