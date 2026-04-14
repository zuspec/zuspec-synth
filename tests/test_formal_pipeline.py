"""Tier 3: Verilog-level formal proofs via sby.

These tests are skipped unless ``--formal`` is passed to pytest.
The yosys bundle at ``packages/yosys`` is used; a Python-based ``yosys_py``
wrapper is set via the ``YOSYS`` env variable so that the cp312 pyosys
extension is used instead of the standalone binary (which requires
``libcrypt.so.2`` and ``libpython3.10``, neither of which may be present).
"""
from __future__ import annotations

import os
import subprocess
import shutil
import sys
import tempfile

import pytest

# Path to the yosys package in the repo
_YOSYS_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "yosys")
)
_YOSYS_PY = os.path.join(_YOSYS_PKG, "bin", "yosys_py")


def _sby_available() -> bool:
    """Return True if ``sby`` is on PATH and executes successfully."""
    sby = shutil.which("sby")
    if not sby:
        return False
    try:
        r = subprocess.run([sby, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _make_env() -> dict:
    """Return an env dict suitable for running sby with the Python yosys wrapper."""
    env = os.environ.copy()
    # Use the pyosys-based wrapper to avoid libcrypt.so.2 / libpython3.10 issues
    if os.path.isfile(_YOSYS_PY):
        env["YOSYS"] = f"python3 {_YOSYS_PY}"
        # Make pyosys importable inside the subprocess
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _YOSYS_PKG + ((":" + prev) if prev else "")
    return env


_SBY = shutil.which("sby") or "sby"


def _run_formal(component_cls) -> None:
    """Execute the full Tier-3 formal flow for *component_cls*.

    Steps:

    1. Run synthesis to get :class:`PipelineIR` and Verilog text.
    2. Embed ``ifdef FORMAL`` properties into the DUT Verilog.
    3. Write a ``.sby`` file.
    4. Execute ``sby`` (with Python yosys wrapper if available).
    5. Assert exit code is 0 (PASS).

    :param component_cls: A component class decorated with ``@zdc.pipeline``.
    :raises pytest.skip.Exception: When ``sby`` is unavailable.
    :raises pytest.fail.Exception: When ``sby`` exits non-zero.
    """
    from test_sync_pipeline_api import run_pipeline_synth
    from zuspec.synth.verify.verilog_props import VerilogPropertyWrapper
    from zuspec.synth.verify.sby_gen import generate_sby

    if not _sby_available():
        pytest.skip("sby not found in PATH")

    with tempfile.TemporaryDirectory(prefix="zuspec-formal-") as td:
        pip, verilog_text = run_pipeline_synth(component_cls, return_ir=True)
        assert pip is not None, "run_pipeline_synth should produce a PipelineIR"

        formal_dut_path = os.path.join(td, f"{pip.module_name}_formal.v")
        sby_path = os.path.join(td, f"{pip.module_name}.sby")

        formal_dut_text = VerilogPropertyWrapper(pip).generate_formal_dut(verilog_text)
        with open(formal_dut_path, "w") as f:
            f.write(formal_dut_text)

        sby_text = generate_sby(pip, formal_dut_path)
        with open(sby_path, "w") as f:
            f.write(sby_text)

        result = subprocess.run(
            [_SBY, "-f", sby_path],
            cwd=td,
            capture_output=True,
            text=True,
            env=_make_env(),
        )
        if result.returncode != 0:
            pytest.fail(
                f"sby failed for {pip.module_name}:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )


# ---------------------------------------------------------------------------
# Component class imports (test_sync_pipeline_api is in the same tests dir)
# ---------------------------------------------------------------------------
from test_sync_pipeline_api import (
    _Ex1Component,
    _Ex3Component,
    _Ex5Component,
    _Ex7Component,
    _Ex8Component,
)
from test_sync_pipeline_api import (
    _AutoThreadPipe,
    _LoadUsePipe,
    _PriorityPipe,
    _FetchWithFSM,
    _InterruptFlush,
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.formal
def test_formal_ex1_properties():
    """P1, P2: basic 2-stage pipeline properties."""
    _run_formal(_Ex1Component)


@pytest.mark.formal
def test_formal_ex3_stall():
    """P1, P2: external stall input pipeline."""
    _run_formal(_Ex3Component)


@pytest.mark.formal
def test_formal_ex5_forwarding():
    """P1, P2: three-stage pipeline with forwarding."""
    _run_formal(_Ex5Component)


@pytest.mark.formal
def test_formal_ex7_flush():
    """P1, P2, P5: branch flush clears valid bits."""
    _run_formal(_Ex7Component)


@pytest.mark.formal
def test_formal_ex8_cancel():
    """P1, P2, P6: cancel clears valid without stalling upstream."""
    _run_formal(_Ex8Component)


@pytest.mark.formal
def test_formal_autothread():
    """P1, P2: three-stage auto-thread pipeline."""
    _run_formal(_AutoThreadPipe)


@pytest.mark.formal
def test_formal_loaduse():
    """P1, P2: three-stage load-use pipeline."""
    _run_formal(_LoadUsePipe)


@pytest.mark.formal
def test_formal_priority():
    """P1, P2, P5: priority pipeline with flush on INGEST stage."""
    _run_formal(_PriorityPipe)


@pytest.mark.formal
def test_formal_fetch_with_fsm():
    """P1, P2: FSM-driven two-stage fetch pipeline."""
    _run_formal(_FetchWithFSM)


@pytest.mark.formal
def test_formal_interrupt_flush():
    """P1, P2, P5: three-stage pipeline with per-stage flush signals."""
    _run_formal(_InterruptFlush)
