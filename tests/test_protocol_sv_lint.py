"""Tier 2 — Verilator lint tests for Phase 5/6 generated SystemVerilog.

Tests are automatically skipped if ``verilator`` is not found on PATH or at
the bundled location ``packages/verilator/bin/verilator``.

Covers:
  - Synchronous FIFO module  (``generate_fifo_sv``)
  - Priority arbiter module   (``generate_priority_arbiter_sv``)
  - Round-robin arbiter module (``generate_rr_arbiter_sv``)
  - IfProtocol port-bundle declaration snippet wrapped in a stub module

The lint helper follows the same pattern as ``test_async_pipeline_synth.py``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_tests_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_tests_dir, "..", "src")
_dc_src = os.path.join(_tests_dir, "..", "..", "zuspec-dataclasses", "src")
for _p in [_synth_src, _dc_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Verilator helper (same pattern as test_async_pipeline_synth.py)
# ---------------------------------------------------------------------------

_packages_root = os.path.abspath(
    os.path.join(_tests_dir, "..", "..", "..")
)
_VERILATOR_BIN = os.path.join(_packages_root, "packages", "verilator", "bin", "verilator")


def _verilator_lint(sv: str, top_module: str = "dut") -> None:
    """Run ``verilator --lint-only`` on *sv*; skip the test if Verilator not found.

    Args:
        sv:         SV source text (must define at least the module *top_module*).
        top_module: Module name for ``--top-module`` flag.
    """
    verilator = _VERILATOR_BIN
    if not os.path.isfile(verilator):
        verilator = shutil.which("verilator") or "verilator"
    if not shutil.which(verilator) and not os.path.isfile(verilator):
        pytest.skip("verilator not found")

    with tempfile.TemporaryDirectory(prefix="zuspec-lint-") as d:
        sv_file = os.path.join(d, f"{top_module}.sv")
        with open(sv_file, "w") as f:
            f.write(sv)
        result = subprocess.run(
            [verilator, "--lint-only", "--sv", "--top-module", top_module, sv_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Verilator lint failed for '{top_module}':\n"
                f"{result.stdout}{result.stderr}\n"
                f"--- SV ---\n{sv}"
            )


# ---------------------------------------------------------------------------
# Imports from Phase 5/6
# ---------------------------------------------------------------------------

from zuspec.synth.ir.protocol_ir import (
    IfProtocolPortIR,
    IfProtocolScenario,
    ProtocolField,
    QueueIR,
    SelectBranchIR,
    SelectIR,
)
from zuspec.synth.sprtl.protocol_sv import (
    generate_fifo_sv,
    generate_ifprotocol_port_decls,
    generate_priority_arbiter_sv,
    generate_rr_arbiter_sv,
)


# ---------------------------------------------------------------------------
# Helper: wrap a port-decl snippet in a valid stub module
# ---------------------------------------------------------------------------

def _wrap_port_decls(port_decls: str, module_name: str = "dut") -> str:
    """Wrap raw port declaration lines in a minimal SV module for lint.

    The port list is taken verbatim from ``generate_ifprotocol_port_decls``;
    the trailing comma on the last port is stripped before closing the list.
    """
    lines = [l for l in port_decls.splitlines() if l.strip()]
    # Strip trailing comma from last real port line
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    port_body = "\n".join(lines)
    return (
        f"module {module_name} (\n"
        f"  input  logic clk,\n"
        f"  input  logic rst,\n"
        f"{port_body}\n"
        f");\n"
        f"endmodule\n"
    )


# ===========================================================================
# FIFO lint tests
# ===========================================================================

class TestFifoLint:
    """Verilator lint-only checks for ``generate_fifo_sv``."""

    def test_fifo_default_params(self):
        """Default: depth=16, width=32."""
        sv = generate_fifo_sv(QueueIR(name="req_q"))
        _verilator_lint(sv, top_module="req_q_fifo")

    def test_fifo_narrow_shallow(self):
        """Narrow (8-bit) shallow (4-entry) FIFO."""
        sv = generate_fifo_sv(QueueIR(name="byte_q", elem_width=8, depth=4))
        _verilator_lint(sv, top_module="byte_q_fifo")

    def test_fifo_wide_deep(self):
        """Wide (64-bit) deep (256-entry) FIFO."""
        sv = generate_fifo_sv(QueueIR(name="wide_q", elem_width=64, depth=256))
        _verilator_lint(sv, top_module="wide_q_fifo")

    def test_fifo_depth_1(self):
        """Edge case: single-entry FIFO."""
        sv = generate_fifo_sv(QueueIR(name="one_q", depth=1))
        _verilator_lint(sv, top_module="one_q_fifo")

    def test_fifo_depth_2(self):
        """Depth-2 FIFO (addr_bits boundary)."""
        sv = generate_fifo_sv(QueueIR(name="two_q", depth=2))
        _verilator_lint(sv, top_module="two_q_fifo")

    def test_fifo_power_of_two_depth(self):
        """Power-of-two depth=64."""
        sv = generate_fifo_sv(QueueIR(name="p2_q", depth=64))
        _verilator_lint(sv, top_module="p2_q_fifo")

    def test_fifo_with_prefix(self):
        """Module prefix is applied correctly."""
        sv = generate_fifo_sv(QueueIR(name="data_q"), module_prefix="top_")
        _verilator_lint(sv, top_module="top_data_q_fifo")


# ===========================================================================
# Priority arbiter lint tests
# ===========================================================================

class TestPriorityArbiterLint:
    """Verilator lint-only checks for ``generate_priority_arbiter_sv``."""

    def _sel(self, n: int) -> SelectIR:
        return SelectIR(
            name="sel",
            branches=[SelectBranchIR(f"q{i}", i) for i in range(n)],
        )

    def test_arbiter_1_branch(self):
        """Single-branch (trivial) priority arbiter."""
        sv = generate_priority_arbiter_sv(self._sel(1))
        _verilator_lint(sv, top_module="sel_arb")

    def test_arbiter_2_branches(self):
        """Two-branch priority arbiter."""
        sv = generate_priority_arbiter_sv(self._sel(2))
        _verilator_lint(sv, top_module="sel_arb")

    def test_arbiter_3_branches(self):
        """Three-branch priority arbiter."""
        sv = generate_priority_arbiter_sv(self._sel(3))
        _verilator_lint(sv, top_module="sel_arb")

    def test_arbiter_4_branches(self):
        """Four-branch priority arbiter."""
        sv = generate_priority_arbiter_sv(self._sel(4))
        _verilator_lint(sv, top_module="sel_arb")

    def test_arbiter_8_branches(self):
        """Eight-branch priority arbiter (id field needs 3 bits)."""
        sv = generate_priority_arbiter_sv(self._sel(8))
        _verilator_lint(sv, top_module="sel_arb")

    def test_arbiter_with_prefix(self):
        """Module prefix applied to arbiter."""
        sv = generate_priority_arbiter_sv(self._sel(2), module_prefix="dut_")
        _verilator_lint(sv, top_module="dut_sel_arb")


# ===========================================================================
# Round-robin arbiter lint tests
# ===========================================================================

class TestRRArbiterLint:
    """Verilator lint-only checks for ``generate_rr_arbiter_sv``."""

    def _sel(self, n: int) -> SelectIR:
        return SelectIR(
            name="rr_sel",
            branches=[SelectBranchIR(f"q{i}", i) for i in range(n)],
            round_robin=True,
        )

    def test_rr_arbiter_2_branches(self):
        """Two-branch round-robin arbiter."""
        sv = generate_rr_arbiter_sv(self._sel(2))
        _verilator_lint(sv, top_module="rr_sel_rr_arb")

    def test_rr_arbiter_3_branches(self):
        """Three-branch round-robin arbiter."""
        sv = generate_rr_arbiter_sv(self._sel(3))
        _verilator_lint(sv, top_module="rr_sel_rr_arb")

    def test_rr_arbiter_4_branches(self):
        """Four-branch round-robin arbiter."""
        sv = generate_rr_arbiter_sv(self._sel(4))
        _verilator_lint(sv, top_module="rr_sel_rr_arb")

    def test_rr_arbiter_with_prefix(self):
        """Module prefix applied to RR arbiter."""
        sv = generate_rr_arbiter_sv(self._sel(2), module_prefix="pfx_")
        _verilator_lint(sv, top_module="pfx_rr_sel_rr_arb")


# ===========================================================================
# IfProtocol port-bundle lint tests
# ===========================================================================

class TestIfProtocolPortLint:
    """Wrap port declarations in a stub module and Verilator-lint it."""

    def test_scenario_b_initiator(self):
        """Scenario B initiator: full req/resp handshake."""
        port = IfProtocolPortIR(
            name="mem",
            scenario=IfProtocolScenario.B,
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        )
        decls = generate_ifprotocol_port_decls(port)
        sv = _wrap_port_decls(decls, "dut_scenario_b")
        _verilator_lint(sv, top_module="dut_scenario_b")

    def test_scenario_b_export(self):
        """Scenario B export: directions flipped."""
        port = IfProtocolPortIR(
            name="mem",
            is_export=True,
            scenario=IfProtocolScenario.B,
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        )
        decls = generate_ifprotocol_port_decls(port)
        sv = _wrap_port_decls(decls, "dut_scenario_b_exp")
        _verilator_lint(sv, top_module="dut_scenario_b_exp")

    def test_scenario_a_minimal(self):
        """Scenario A: no ready/valid — only data wires."""
        class _Props:
            req_always_ready = True
            resp_always_valid = True
            resp_has_backpressure = False

        port = IfProtocolPortIR(
            name="fast",
            scenario=IfProtocolScenario.A,
            properties=_Props(),
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        )
        decls = generate_ifprotocol_port_decls(port)
        sv = _wrap_port_decls(decls, "dut_scenario_a")
        _verilator_lint(sv, top_module="dut_scenario_a")

    def test_scenario_d_with_id(self):
        """Scenario D: request/response ID fields."""
        class _Props:
            req_always_ready = False
            resp_always_valid = False
            resp_has_backpressure = False

        port = IfProtocolPortIR(
            name="ooo",
            scenario=IfProtocolScenario.D,
            properties=_Props(),
            req_fields=[ProtocolField("addr", 64)],
            resp_fields=[ProtocolField("data", 64, is_response=True)],
            id_bits=2,
        )
        decls = generate_ifprotocol_port_decls(port)
        sv = _wrap_port_decls(decls, "dut_scenario_d")
        _verilator_lint(sv, top_module="dut_scenario_d")

    def test_multi_field_protocol(self):
        """Multiple request and response fields."""
        port = IfProtocolPortIR(
            name="axi",
            scenario=IfProtocolScenario.B,
            req_fields=[
                ProtocolField("addr", 64),
                ProtocolField("len", 8),
                ProtocolField("size", 3),
            ],
            resp_fields=[
                ProtocolField("data", 64, is_response=True),
                ProtocolField("resp", 2, is_response=True),
            ],
        )
        decls = generate_ifprotocol_port_decls(port)
        sv = _wrap_port_decls(decls, "dut_multi_field")
        _verilator_lint(sv, top_module="dut_multi_field")
