"""Tier 3: Formal proofs for async pipeline synthesis via sby.

Tests are skipped unless ``--formal`` is passed to pytest (or ``sby`` is
not in PATH).  The Yosys bundle at ``packages/yosys`` is used.

Properties verified
-------------------
P1  After a reset cycle every stage valid register is 0.
P2  Stage valid propagates from its source when not stalled/flushed.
P7  Multi-cycle stage: ``{sl}_done`` fires exactly when valid and
    ``cycle_q == cycle_hi``.
P8  Multi-cycle stage: upstream stages are frozen while ``mc_stall`` is high.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from io import StringIO

import pytest

# ---------------------------------------------------------------------------
# Yosys/sby bootstrap
# ---------------------------------------------------------------------------

_YOSYS_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "yosys")
)
_YOSYS_BIN = os.path.join(_YOSYS_PKG, "bin")

# Prefer the system-wide SymbiYosys installation if available
_TOOLS_SBY = "/tools/symbiyosys/20251115/bin"
if os.path.isfile(os.path.join(_TOOLS_SBY, "sby")):
    _YOSYS_BIN = _TOOLS_SBY

_SBY_EXE   = os.path.join(_YOSYS_BIN, "sby")


def _sby_available() -> bool:
    if not os.path.isfile(_SBY_EXE):
        exe = shutil.which("sby")
        return exe is not None
    try:
        r = subprocess.run([_SBY_EXE, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _make_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = _YOSYS_BIN + os.pathsep + env.get("PATH", "")
    return env


# ---------------------------------------------------------------------------
# DUT component definitions (same as in test_async_pipeline_synth)
# ---------------------------------------------------------------------------
import sys as _sys
_tests_dir = os.path.dirname(__file__)
if _tests_dir not in _sys.path:
    _sys.path.insert(0, _tests_dir)

from test_async_pipeline_synth import (
    _PassThrough3,
    _Adder3,
    _AutoThread5,
    _MultiCyclePipe,
    run_async_pipeline_synth,
)

try:
    from test_pipeline_system_adder import (
        _AdderNewAPI,
        _PassThroughNewAPI,
    )
    _NEW_API_AVAILABLE = True
except ImportError:
    _NEW_API_AVAILABLE = False


# ---------------------------------------------------------------------------
# AsyncPipelinePropertyWrapper — embeds formal properties into DUT Verilog
# ---------------------------------------------------------------------------

class AsyncPipelinePropertyWrapper:
    """Embed P1, P2, P7, P8 formal properties for async pipeline RTL.

    :param pip: Completed :class:`~zuspec.synth.ir.pipeline_ir.PipelineIR`.
    """

    def __init__(self, pip) -> None:
        self._pip = pip

    def generate_formal_dut(self, dut_verilog: str) -> str:
        """Insert ``ifdef FORMAL`` block before ``endmodule``."""
        block = self._build_formal_block()
        return re.sub(r"(\bendmodule\b)", block + r"\n\1", dut_verilog, count=1)

    def _build_formal_block(self) -> str:
        pip = self._pip
        vc  = pip.valid_chain
        buf = StringIO()

        buf.write("`ifdef FORMAL\n")
        buf.write("  reg f_past_valid;\n")
        buf.write("  initial f_past_valid = 0;\n")
        buf.write("  always @(posedge clk) f_past_valid <= 1;\n\n")
        buf.write("  initial assume (!rst_n);\n\n")

        # ── P1: after a reset cycle all valid regs are 0 ─────────────────
        buf.write("  // P1: after reset all valid registers are 0\n")
        buf.write("  always @(posedge clk) begin\n")
        buf.write("    if (f_past_valid && $past(!rst_n, 1)) begin\n")
        for entry in vc:
            buf.write(f"      assert ({entry.valid_reg} == 1'b0);\n")
        buf.write("    end\n")
        buf.write("  end\n\n")

        # ── P2: valid propagates when not stalled/flushed/cancelled ───────
        buf.write("  // P2: valid propagates from source when not inhibited\n")
        for entry in vc:
            guards = ["f_past_valid", "$past(rst_n, 1)"]
            if entry.flush_signal:
                guards.append(f"!$past({entry.flush_signal}, 1)")
            if entry.cancel_signal:
                guards.append(f"!$past({entry.cancel_signal}, 1)")
            for ss in entry.stall_signals:
                guards.append(f"!$past({ss}, 1)")
            cond = " && ".join(guards)
            buf.write("  always @(posedge clk) begin\n")
            buf.write(f"    if ({cond})\n")
            buf.write(
                f"      assert ({entry.valid_reg} == $past({entry.source_valid}, 1));\n"
            )
            buf.write("  end\n\n")

        # ── P7 / P8: multi-cycle stage properties ─────────────────────────
        for stage in pip.stages:
            if stage.cycle_hi <= 0:
                continue
            sl = stage.name.lower()
            N  = stage.cycle_hi
            import math
            w = max(1, math.ceil(math.log2(N + 2)))

            # P7: done fires iff valid_q && cycle_q == cycle_hi
            buf.write(f"  // P7 ({stage.name}): done asserted iff valid and counter at max\n")
            buf.write("  always @(*) begin\n")
            buf.write(
                f"    assert ({sl}_done == ({sl}_valid_q && ({sl}_cycle_q == {w}'d{N})));\n"
            )
            buf.write("  end\n\n")

            # P8: while mc_stall is high, upstream valid regs are frozen.
            # Build list of upstream stages (index <= MC stage index).
            upstream = [s for s in pip.stages if s.index <= stage.index]
            if upstream:
                buf.write(
                    f"  // P8 ({stage.name}): mc_stall freezes upstream valid registers\n"
                )
                for up in upstream:
                    up_vr = f"{up.name.lower()}_valid_q"
                    buf.write("  always @(posedge clk) begin\n")
                    buf.write(
                        f"    if (f_past_valid && $past(rst_n, 1) && $past({sl}_mc_stall, 1))\n"
                    )
                    buf.write(f"      assert ({up_vr} == $past({up_vr}, 1));\n")
                    buf.write("  end\n\n")

        buf.write("`endif  // FORMAL\n")
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Core formal execution helper
# ---------------------------------------------------------------------------

def _run_async_formal(component_cls) -> None:
    """Synthesise *component_cls*, embed properties, run sby, assert pass."""
    if not _sby_available():
        pytest.skip("sby not found")

    with tempfile.TemporaryDirectory(prefix="zuspec-async-formal-") as td:
        pip, sv_text = run_async_pipeline_synth(component_cls, return_ir=True)
        assert pip is not None

        wrapper = AsyncPipelinePropertyWrapper(pip)
        formal_sv = wrapper.generate_formal_dut(sv_text)

        dut_path = os.path.join(td, f"{pip.module_name}_formal.v")
        sby_path  = os.path.join(td, f"{pip.module_name}.sby")

        with open(dut_path, "w") as f:
            f.write(formal_sv)

        # Write sby config
        sby_text = _make_sby(pip.module_name, dut_path)
        with open(sby_path, "w") as f:
            f.write(sby_text)

        result = subprocess.run(
            [_SBY_EXE, "-f", sby_path],
            cwd=td,
            capture_output=True,
            text=True,
            env=_make_env(),
        )
        if result.returncode != 0:
            pytest.fail(
                f"sby failed for {pip.module_name}:\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )


def _make_sby(module_name: str, dut_path: str, depth: int = 25) -> str:
    lines = [
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
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.formal
def test_formal_async_passthrough():
    """P1, P2: 3-stage pass-through — basic async pipeline properties."""
    _run_async_formal(_PassThrough3)


@pytest.mark.formal
def test_formal_async_adder3():
    """P1, P2: 3-stage adder — data flows through pipeline registers."""
    _run_async_formal(_Adder3)


@pytest.mark.formal
def test_formal_async_autothread5():
    """P1, P2: 5-stage auto-threaded tag — tag reaches WB stage correctly."""
    _run_async_formal(_AutoThread5)


@pytest.mark.formal
def test_formal_async_multicycle():
    """P1, P2, P7, P8: multi-cycle COMPUTE stage — counter and freeze."""
    _run_async_formal(_MultiCyclePipe)


# ---------------------------------------------------------------------------
# New-API formal tests (InPort/OutPort method port pipelines)
# ---------------------------------------------------------------------------

def _run_new_api_formal(component_cls, extra_props: str = "") -> None:
    """Like _run_async_formal but also injects *extra_props* into the FORMAL block."""
    if not _sby_available():
        pytest.skip("sby not found")
    if not _NEW_API_AVAILABLE:
        pytest.skip("test_pipeline_system_adder not importable")

    with tempfile.TemporaryDirectory(prefix="zuspec-new-api-formal-") as td:
        pip, sv_text = run_async_pipeline_synth(component_cls, return_ir=True)
        assert pip is not None

        wrapper = AsyncPipelinePropertyWrapper(pip)
        formal_sv = wrapper.generate_formal_dut(sv_text)

        # Inject extra properties right before the closing `endif
        if extra_props:
            formal_sv = formal_sv.replace(
                "`endif  // FORMAL",
                extra_props + "\n`endif  // FORMAL",
            )

        dut_path = os.path.join(td, f"{pip.module_name}_formal.v")
        sby_path  = os.path.join(td, f"{pip.module_name}.sby")

        with open(dut_path, "w") as f:
            f.write(formal_sv)

        sby_text = _make_sby(pip.module_name, dut_path)
        with open(sby_path, "w") as f:
            f.write(sby_text)

        result = subprocess.run(
            [_SBY_EXE, "-f", sby_path],
            cwd=td,
            capture_output=True,
            text=True,
            env=_make_env(),
        )
        if result.returncode != 0:
            pytest.fail(
                f"sby failed for {pip.module_name}:\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )


@pytest.mark.formal
def test_formal_new_api_passthrough():
    """P1, P2: PassThrough new-API pipeline — basic valid chain properties."""
    if not _NEW_API_AVAILABLE:
        pytest.skip("test_pipeline_system_adder not importable")
    _run_new_api_formal(_PassThroughNewAPI)


@pytest.mark.formal
def test_formal_new_api_adder():
    """P1, P2: Adder new-API pipeline — valid chain properties.

    Also verifies P_DATA: when wb_valid_q is asserted the output sum_out
    equals the registered EXEC result (result_exec_to_wb_q).
    This proves ingress values are correctly captured and threaded through
    the pipeline channel registers rather than leaking the live module inputs.
    """
    if not _NEW_API_AVAILABLE:
        pytest.skip("test_pipeline_system_adder not importable")

    p_data = """\
  // P_DATA: when wb_valid_q is asserted, sum_out equals result_exec_to_wb_q
  // Both are combinational so this is a direct equivalence check.
  always @(*) begin
    if (wb_valid_q)
      assert (sum_out == result_exec_to_wb_q);
  end"""

    _run_new_api_formal(_AdderNewAPI, extra_props=p_data)


@pytest.mark.formal
def test_formal_multi_ingress_adder():
    """P1, P2, P_EXEC, P_DATA: Multi-ingress adder — prove both ingress vars
    are captured into pipeline registers and the addition uses them correctly.

    P_EXEC: in EXEC stage, result_exec == a_fetch_to_exec_q + b_fetch_to_exec_q.
            This proves the synthesizer did NOT leak the live a_in/b_in inputs.
    P_DATA: in WB stage, sum_out == result_exec_to_wb_q (output uses registered result).
    """
    if not _NEW_API_AVAILABLE:
        pytest.skip("test_pipeline_system_adder not importable")

    extra = """\
  // P_EXEC: EXEC stage computes from channel registers, not live inputs
  always @(*) begin
    if (exec_valid_q)
      assert (result_exec == (a_fetch_to_exec_q + b_fetch_to_exec_q));
  end

  // P_DATA: WB output equals the registered EXEC result
  always @(*) begin
    if (wb_valid_q)
      assert (sum_out == result_exec_to_wb_q);
  end"""

    _run_new_api_formal(_AdderNewAPI, extra_props=extra)
