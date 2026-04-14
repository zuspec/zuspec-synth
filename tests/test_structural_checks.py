"""Tests for zuspec.synth.verify.structural — no Yosys or SMT solver required."""
from __future__ import annotations

import ast
import copy

import pytest

from test_sync_pipeline_api import (
    run_pipeline_synth,
    _Ex1Component,
    _Ex5Component,
)

from zuspec.synth.verify.structural import (
    run_all_checks,
    check_forwarding_completeness,
    StructuralError,
)
from zuspec.synth.ir.pipeline_ir import RegFileHazard, ForwardingDecl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pip(component_cls, forward_default: bool = True):
    """Return the PipelineIR for *component_cls*."""
    pip, _sv = run_pipeline_synth(component_cls, forward_default=forward_default, return_ir=True)
    assert pip is not None, "run_pipeline_synth should produce a PipelineIR"
    return pip


# ---------------------------------------------------------------------------
# T2-1: All checks pass on a clean minimal pipeline
# ---------------------------------------------------------------------------

def test_structural_ex1_clean():
    """Ex-1 two-stage pipeline has no structural errors."""
    pip = _get_pip(_Ex1Component)
    errors = run_all_checks(pip)
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# T2-2: Forwarding completeness check passes on Ex-5
# ---------------------------------------------------------------------------

def test_structural_ex5_forwarding_complete():
    """Ex-5 pipeline: forwarding completeness check raises no FWD_MISSING errors."""
    pip = _get_pip(_Ex5Component, forward_default=True)
    fwd_errors = [e for e in run_all_checks(pip) if e.code == "FWD_MISSING"]
    assert fwd_errors == [], [e.message for e in fwd_errors]


# ---------------------------------------------------------------------------
# T2-3: Injecting a wrong width triggers WIDTH_MISMATCH
# ---------------------------------------------------------------------------

def test_structural_injected_width_mismatch():
    """Manually corrupting a channel width triggers WIDTH_MISMATCH."""
    pip = _get_pip(_Ex1Component)
    assert pip.channels, "Ex-1 should have at least one channel"

    # Deep copy to avoid mutating the shared IR
    pip = copy.deepcopy(pip)
    original_width = pip.channels[0].width
    pip.channels[0].width = original_width + 7  # corrupt

    # Only triggers when the variable is in annotation_map
    var_name = pip.channels[0].name.split("_")[0]
    if var_name not in pip.annotation_map:
        # Insert a fake annotation so the check can fire
        pip.annotation_map[var_name] = ast.parse(f"zdc.u{original_width}", mode="eval").body

    errors = run_all_checks(pip)
    assert any(e.code == "WIDTH_MISMATCH" for e in errors), (
        f"Expected WIDTH_MISMATCH, got: {[e.code for e in errors]}"
    )


# ---------------------------------------------------------------------------
# T2-4: Injecting a missing ForwardingDecl triggers FWD_MISSING
# ---------------------------------------------------------------------------

def test_structural_injected_fwd_missing():
    """Injecting a RegFileHazard resolved_by='forward' without a matching
    ForwardingDecl triggers FWD_MISSING."""
    pip = _get_pip(_Ex1Component)
    pip = copy.deepcopy(pip)

    # Inject a synthetic regfile hazard that claims forwarding
    synthetic_hz = RegFileHazard(
        field_name="regfile",
        write_stage="S1",
        read_stage="S2",
        write_addr_var="rd",
        read_addr_var="rs1",
        write_data_var="result",
        read_result_var="rdata",
        resolved_by="forward",
    )
    pip.regfile_hazards.append(synthetic_hz)
    # Do NOT add a matching ForwardingDecl → should trigger FWD_MISSING

    errors = run_all_checks(pip)
    assert any(e.code == "FWD_MISSING" for e in errors), (
        f"Expected FWD_MISSING, got: {[e.code for e in errors]}"
    )


# ---------------------------------------------------------------------------
# T2-5: Injecting a False stall_cond triggers DEAD_STALL
# ---------------------------------------------------------------------------

def test_structural_dead_stall_detected():
    """Injecting an AST-False stall_cond on a stage triggers DEAD_STALL."""
    pip = _get_pip(_Ex1Component)
    pip = copy.deepcopy(pip)

    false_node = ast.Constant(value=False)
    pip.stages[0].stall_cond = false_node

    errors = run_all_checks(pip)
    assert any(e.code == "DEAD_STALL" for e in errors), (
        f"Expected DEAD_STALL, got: {[e.code for e in errors]}"
    )
