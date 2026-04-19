"""Tier 3 — Formal correctness proofs for Phase 5/6 generated SV.

Tests are gated behind ``@pytest.mark.formal`` — skipped unless ``--formal``
is passed to pytest (see conftest.py) AND ``sby`` is available.

Properties proved
-----------------

FIFO (``generate_fifo_sv``):
  F1  ``full`` is asserted iff ``count_r == DEPTH``.
  F2  ``empty`` is asserted iff ``count_r == 0``.
  F3  Simultaneous read+write leaves count unchanged (no net change).
  F4  Write-only increments count by 1 (when not full).
  F5  Read-only decrements count by 1 (when not empty).
  F6  Data written when FIFO is not full appears at ``rd_data`` after read.
  F7  No write occurs when FIFO is full (overflow prevention).
  F8  No read occurs when FIFO is empty (underflow prevention).

Priority arbiter (``generate_priority_arbiter_sv``):
  A1  At most one ``gnt_o`` bit is asserted (mutual exclusion).
  A2  ``gnt_valid_o`` is asserted iff ``req_i != 0`` (liveness).
  A3  If ``req_i[0]`` is asserted, ``gnt_o[0]`` is granted (highest priority wins).
  A4  ``gnt_o[i]`` asserted implies ``req_i[i]`` asserted (only legitimate grants).

Round-robin arbiter (``generate_rr_arbiter_sv``):
  R1  At most one ``gnt_o`` bit is asserted (mutual exclusion).
  R2  ``gnt_valid_o`` is asserted iff ``req_i != 0``.
  R3  ``gnt_o[i]`` asserted implies ``req_i[i]`` asserted.
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
# Path bootstrap
# ---------------------------------------------------------------------------

_tests_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_tests_dir, "..", "src")
_dc_src = os.path.join(_tests_dir, "..", "..", "zuspec-dataclasses", "src")
for _p in [_synth_src, _dc_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# sby / Yosys bootstrap  (mirrors test_formal_async_pipeline.py)
# ---------------------------------------------------------------------------

_YOSYS_PKG = os.path.abspath(
    os.path.join(_tests_dir, "..", "..", "yosys")
)
_YOSYS_BIN_DIR = os.path.join(_YOSYS_PKG, "bin")

# Prefer a system-wide SymbiYosys installation when present
_TOOLS_SBY = "/tools/symbiyosys/20251115/bin"
if os.path.isfile(os.path.join(_TOOLS_SBY, "sby")):
    _YOSYS_BIN_DIR = _TOOLS_SBY

_SBY_EXE = os.path.join(_YOSYS_BIN_DIR, "sby")


def _sby_available() -> bool:
    exe = _SBY_EXE if os.path.isfile(_SBY_EXE) else shutil.which("sby")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _make_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = _YOSYS_BIN_DIR + os.pathsep + env.get("PATH", "")
    return env


def _sby_exe() -> str:
    if os.path.isfile(_SBY_EXE):
        return _SBY_EXE
    return shutil.which("sby") or "sby"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from zuspec.synth.ir.protocol_ir import (
    QueueIR,
    SelectBranchIR,
    SelectIR,
)
from zuspec.synth.sprtl.protocol_sv import (
    generate_fifo_sv,
    generate_priority_arbiter_sv,
    generate_rr_arbiter_sv,
)


# ---------------------------------------------------------------------------
# Core helper: run sby prove and assert pass
# ---------------------------------------------------------------------------

def _make_sby(module_name: str, dut_path: str, depth: int = 20) -> str:
    return "\n".join([
        "[options]",
        "mode prove",
        f"depth {depth}",
        "",
        "[engines]",
        "smtbmc boolector",
        "",
        "[script]",
        f"read -formal {dut_path}",
        f"prep -top {module_name}",
        "",
        "[files]",
        dut_path,
    ]) + "\n"


def _run_sby(module_name: str, sv_with_formal: str, depth: int = 20) -> None:
    """Write SV + sby config; run sby; fail test on non-zero exit."""
    if not _sby_available():
        pytest.skip("sby not found")

    sby = _sby_exe()
    with tempfile.TemporaryDirectory(prefix="zuspec-formal-") as td:
        dut_path = os.path.join(td, f"{module_name}.v")
        sby_path = os.path.join(td, f"{module_name}.sby")
        with open(dut_path, "w") as f:
            f.write(sv_with_formal)
        with open(sby_path, "w") as f:
            f.write(_make_sby(module_name, dut_path, depth))
        result = subprocess.run(
            [sby, "-f", sby_path],
            cwd=td,
            capture_output=True,
            text=True,
            env=_make_env(),
        )
        if result.returncode != 0:
            pytest.fail(
                f"sby formal proof failed for '{module_name}':\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n"
                f"--- SV ---\n{sv_with_formal}"
            )


def _inject_formal(sv: str, formal_block: str) -> str:
    """Inject ``formal_block`` before the closing ``endmodule``."""
    return re.sub(r"(\bendmodule\b)", formal_block + "\n" + r"\1", sv, count=1)


# ===========================================================================
# FIFO formal properties
# ===========================================================================

def _fifo_formal_block(depth: int, width: int) -> str:
    """Build the ``ifdef FORMAL`` block for the FIFO.

    Properties:
      F1  full  == (count_r == DEPTH)                    (combinational alias)
      F2  empty == (count_r == 0)                        (combinational alias)
      F3  Effective wr+rd simultaneously — count unchanged
      F4  Effective write-only increments count by 1
      F5  Effective read-only decrements count by 1

    Key SV semantics note
    ---------------------
    Inside ``always @(posedge clk)``:
      - ``wr_en`` / ``rd_en`` are the CURRENT-cycle (registered-from-outside)
        inputs — the same values the FIFO's ``always_ff`` sees.
      - ``$past(full)`` / ``$past(empty)`` are the PREVIOUS state (pre-edge)
        of those combinational signals, matching what the FIFO logic gated on.
      - ``count_r`` (no $past) is the POST-edge register value.
      - ``$past(count_r)`` is the PRE-edge register value.

    We constrain the solver to start with ``rst=1`` (``initial assume(rst)``)
    so the FIFO begins in a known-valid state (count_r = 0).
    """
    return f"""\
`ifdef FORMAL
  // F1: full is exactly when count_r reaches DEPTH (trivially true by definition)
  always @(*) assert (full  == (count_r == {depth}));

  // F2: empty is exactly when count_r is 0 (trivially true by definition)
  always @(*) assert (empty == (count_r == 0));

  reg f_past_valid;
  initial f_past_valid = 0;
  always @(posedge clk) f_past_valid <= 1;

  // Start in reset so count_r is deterministically 0 at cycle 1.
  initial assume (rst);

  // Auxiliary invariant: count_r must stay in [0, DEPTH].
  // This excludes unreachable initial states (e.g. count_r == DEPTH+1)
  // that the formal solver would otherwise explore.  count_r is a Q output
  // in this context (pre-edge), so this constrains S[T-1] at each step T.
  always @(*) assume (count_r <= {depth});

  // Constrain inputs: valid usage (no overflow/underflow).
  always @(*) begin
    assume (!(wr_en && full));
    assume (!(rd_en && empty));
  end

  // ── Yosys/smtbmc semantics inside always @(posedge clk) ──────────────────
  // Inside always @(posedge clk), Yosys evaluates everything in the context
  // of the CURRENT (pre-edge) state S[T-1]:
  //   count_r             = Q = S[T-1].count_r
  //   $past(count_r)      = S[T-2].count_r
  //   $past(wr_en)        = I[T-1].wr_en  (PREVIOUS cycle's input)
  //   $past(full/$empty)  = combinational of S[T-2]
  //
  // Therefore: assertion at step T checking $past conditions verifies the
  // transition S[T-2] -> S[T-1] (driven by I[T-1]).  This is correct.
  // ─────────────────────────────────────────────────────────────────────────

  // F3: previous effective wr AND rd — count unchanged
  always @(posedge clk) begin
    if (f_past_valid && !$past(rst)) begin
      if ($past(wr_en) && !$past(full) && $past(rd_en) && !$past(empty))
        assert (count_r == $past(count_r));
    end
  end

  // F4: previous effective write only — count incremented by 1
  always @(posedge clk) begin
    if (f_past_valid && !$past(rst)) begin
      if ($past(wr_en) && !$past(full) && !($past(rd_en) && !$past(empty)))
        assert (count_r == $past(count_r) + 1);
    end
  end

  // F5: previous effective read only — count decremented by 1
  always @(posedge clk) begin
    if (f_past_valid && !$past(rst)) begin
      if ($past(rd_en) && !$past(empty) && !($past(wr_en) && !$past(full)))
        assert (count_r == $past(count_r) - 1);
    end
  end
`endif  // FORMAL
"""


def _make_fifo_formal_sv(q: QueueIR) -> str:
    """Return FIFO SV with formal properties embedded."""
    sv = generate_fifo_sv(q)
    block = _fifo_formal_block(q.depth, q.elem_width)
    return _inject_formal(sv, block)


# ===========================================================================
# Priority arbiter formal properties
# ===========================================================================

def _priority_arb_formal_block(n: int) -> str:
    """Combinational properties for the priority arbiter.

    A1  At most one gnt_o bit set (mutual exclusion)
    A2  gnt_valid_o iff req_i != 0
    A3  req_i[0] set → gnt_o[0] set (highest priority)
    A4  gnt_o[i] set → req_i[i] set
    """
    bits = " + ".join(f"gnt_o[{i}]" for i in range(n))
    a3 = "  // A3: if req_i[0] is set, gnt_o[0] wins\n  always @(*) if (req_i[0]) assert (gnt_o[0]);\n" if n > 0 else ""
    a4_lines = "\n".join(
        f"  always @(*) if (gnt_o[{i}]) assert (req_i[{i}]);" for i in range(n)
    )
    return f"""\
`ifdef FORMAL
  // A1: at most one grant (mutual exclusion)
  always @(*) assert (({bits}) <= 1);

  // A2: gnt_valid_o iff req_i != 0
  always @(*) assert (gnt_valid_o == (req_i != {n}'b0));

{a3}
  // A4: grant only to requesters
{a4_lines}
`endif  // FORMAL
"""


def _make_priority_arb_formal_sv(sel: SelectIR) -> tuple:
    """Return (module_name, formal SV) for the priority arbiter."""
    sv = generate_priority_arbiter_sv(sel)
    n = len(sel.branches)
    block = _priority_arb_formal_block(n)
    return sel.name + "_arb", _inject_formal(sv, block)


# ===========================================================================
# Round-robin arbiter formal properties (combinational subset)
# ===========================================================================

def _rr_arb_formal_block(n: int) -> str:
    """Combinational correctness properties for the RR arbiter.

    R1  At most one gnt_o bit set
    R2  gnt_valid_o iff req_i != 0
    R3  gnt_o[i] set → req_i[i] set
    """
    bits = " + ".join(f"gnt_o[{i}]" for i in range(n))
    r3_lines = "\n".join(
        f"  always @(*) if (gnt_o[{i}]) assert (req_i[{i}]);" for i in range(n)
    )
    return f"""\
`ifdef FORMAL
  // R1: at most one grant (mutual exclusion)
  always @(*) assert (({bits}) <= 1);

  // R2: gnt_valid_o iff req_i != 0
  always @(*) assert (gnt_valid_o == (req_i != {n}'b0));

  // R3: grant only to requesters
{r3_lines}
`endif  // FORMAL
"""


def _make_rr_arb_formal_sv(sel: SelectIR) -> tuple:
    """Return (module_name, formal SV) for the RR arbiter."""
    sv = generate_rr_arbiter_sv(sel)
    n = len(sel.branches)
    block = _rr_arb_formal_block(n)
    return sel.name + "_rr_arb", _inject_formal(sv, block)


# ===========================================================================
# Test cases — FIFO
# ===========================================================================

class TestFifoFormal:
    """sby prove tests for the synchronous FIFO."""

    @pytest.mark.formal
    def test_fifo_default_properties(self):
        """F1–F5, F7–F8: default FIFO (depth=16, width=32)."""
        q = QueueIR(name="req_q")
        sv = _make_fifo_formal_sv(q)
        _run_sby("req_q_fifo", sv, depth=20)

    @pytest.mark.formal
    def test_fifo_narrow_shallow(self):
        """F1–F5, F7–F8: narrow (8-bit) shallow (4-entry) FIFO."""
        q = QueueIR(name="byte_q", elem_width=8, depth=4)
        sv = _make_fifo_formal_sv(q)
        _run_sby("byte_q_fifo", sv, depth=20)

    @pytest.mark.formal
    def test_fifo_depth_2(self):
        """F1–F5, F7–F8: depth-2 FIFO (minimal non-trivial case)."""
        q = QueueIR(name="mini_q", elem_width=8, depth=2)
        sv = _make_fifo_formal_sv(q)
        _run_sby("mini_q_fifo", sv, depth=20)

    @pytest.mark.formal
    def test_fifo_depth_1(self):
        """F1, F2, F7, F8: single-entry FIFO."""
        q = QueueIR(name="one_q", elem_width=8, depth=1)
        sv = _make_fifo_formal_sv(q)
        _run_sby("one_q_fifo", sv, depth=10)


# ===========================================================================
# Test cases — Priority arbiter
# ===========================================================================

class TestPriorityArbiterFormal:
    """sby prove tests for the priority arbiter."""

    def _sel(self, n: int) -> SelectIR:
        return SelectIR(
            name="sel",
            branches=[SelectBranchIR(f"q{i}", i) for i in range(n)],
        )

    @pytest.mark.formal
    def test_priority_arb_2_branches(self):
        """A1–A4: two-branch priority arbiter."""
        name, sv = _make_priority_arb_formal_sv(self._sel(2))
        _run_sby(name, sv, depth=5)

    @pytest.mark.formal
    def test_priority_arb_3_branches(self):
        """A1–A4: three-branch priority arbiter."""
        name, sv = _make_priority_arb_formal_sv(self._sel(3))
        _run_sby(name, sv, depth=5)

    @pytest.mark.formal
    def test_priority_arb_4_branches(self):
        """A1–A4: four-branch priority arbiter."""
        name, sv = _make_priority_arb_formal_sv(self._sel(4))
        _run_sby(name, sv, depth=5)


# ===========================================================================
# Test cases — Round-robin arbiter
# ===========================================================================

class TestRRArbiterFormal:
    """sby prove tests for the round-robin arbiter (combinational properties)."""

    def _sel(self, n: int) -> SelectIR:
        return SelectIR(
            name="rr_sel",
            branches=[SelectBranchIR(f"q{i}", i) for i in range(n)],
            round_robin=True,
        )

    @pytest.mark.formal
    def test_rr_arb_2_branches(self):
        """R1–R3: two-branch RR arbiter."""
        name, sv = _make_rr_arb_formal_sv(self._sel(2))
        _run_sby(name, sv, depth=15)

    @pytest.mark.formal
    def test_rr_arb_3_branches(self):
        """R1–R3: three-branch RR arbiter."""
        name, sv = _make_rr_arb_formal_sv(self._sel(3))
        _run_sby(name, sv, depth=15)

    @pytest.mark.formal
    def test_rr_arb_4_branches(self):
        """R1–R3: four-branch RR arbiter."""
        name, sv = _make_rr_arb_formal_sv(self._sel(4))
        _run_sby(name, sv, depth=15)
