"""Formal equivalence: action.py ≡ counter.py (Yosys sequential induction).

Gate ``@pytest.mark.formal`` — skipped unless ``--formal`` is passed to pytest.

The test synthesizes both the direct-write counter (examples/01_counter) and
the action-based counter (examples/02_action) to SystemVerilog, then uses
Yosys's ``equiv_induct`` + ``equiv_status -assert`` flow to formally prove
that the two circuits are sequentially equivalent under all input sequences.

Yosys pass sequence
-------------------
1. ``read_verilog`` — read both SV texts (module names: ``gold``, ``uut``)
2. ``proc``         — convert ``always`` blocks to internal RTL
3. ``async2sync``   — rewrite async-reset FFs so the SAT solver can handle them
4. ``equiv_make``   — create a combined equivalence-checking module
5. ``equiv_induct`` — prove equivalence by k-induction (k=1 for this circuit)
6. ``equiv_status -assert`` — fail (exit 1) if any cell is unproven
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, "packages/zuspec-dataclasses/src")
sys.path.insert(0, "packages/zuspec-synth/src")

# ---------------------------------------------------------------------------
# Yosys binary — prefer the bundled build
# ---------------------------------------------------------------------------

_YOSYS_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "yosys")
)
_YOSYS_BIN = os.path.join(_YOSYS_PKG, "bin", "yosys")
if not os.path.isfile(_YOSYS_BIN):
    import shutil
    _YOSYS_BIN = shutil.which("yosys") or "yosys"


def _yosys_available() -> bool:
    try:
        r = subprocess.run(
            [_YOSYS_BIN, "--version"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Domain models — at module level so inspect.getsource resolves them
# ---------------------------------------------------------------------------

import zuspec.dataclasses as zdc


@zdc.dataclass
class IncrCountFormal(zdc.Action["_CounterFormal"]):
    async def body(self):
        await self.comp.count.write(self.comp.count.read() + 1)


@zdc.dataclass
class _CounterDirect(zdc.Component):
    """Direct counter — equivalent to examples/01_counter."""
    count: zdc.Reg[zdc.b32] = zdc.output()

    @zdc.proc
    async def _count(self):
        while True:
            await self.count.write(self.count.read() + 1)


@zdc.dataclass
class _CounterFormal(zdc.Component):
    """Action-based counter — equivalent to examples/02_action."""
    count: zdc.Reg[zdc.b32] = zdc.output()

    @zdc.proc
    async def _count(self):
        while True:
            await IncrCountFormal()(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YOSYS_EQUIV_SCRIPT = """\
read_verilog -sv {gold}
read_verilog -sv {uut}
proc
async2sync
equiv_make gold uut equiv
equiv_induct equiv
equiv_status -assert equiv
"""


def _synthesize_as(component_cls, module_name: str) -> str:
    """Synthesize *component_cls* and return SV with module renamed to *module_name*."""
    from zuspec.synth import synthesize

    sv = synthesize(component_cls)
    # Replace the first 'module <anything>(' with 'module <module_name>('
    sv = re.sub(r"\bmodule\s+\w+\s*\(", f"module {module_name}(", sv, count=1)
    return sv


# ---------------------------------------------------------------------------
# Formal test
# ---------------------------------------------------------------------------

@pytest.mark.formal
def test_action_equiv_counter():
    """Prove that the action-based counter is sequentially equivalent to the direct counter.

    Equivalence is established by Yosys k-induction (k=1 suffices for a
    1-register design).  The proof covers all 2^32 initial states and all
    infinite input sequences.
    """
    if not _yosys_available():
        pytest.skip(f"Yosys not found at {_YOSYS_BIN}")

    gold_sv = _synthesize_as(_CounterDirect, "gold")
    uut_sv = _synthesize_as(_CounterFormal, "uut")

    with tempfile.TemporaryDirectory(prefix="zuspec-equiv-") as td:
        gold_path = os.path.join(td, "gold.sv")
        uut_path = os.path.join(td, "uut.sv")
        script_path = os.path.join(td, "equiv.ys")
        log_path = os.path.join(td, "equiv.log")

        with open(gold_path, "w") as f:
            f.write(gold_sv)
        with open(uut_path, "w") as f:
            f.write(uut_sv)
        with open(script_path, "w") as f:
            f.write(_YOSYS_EQUIV_SCRIPT.format(gold=gold_path, uut=uut_path))

        result = subprocess.run(
            [_YOSYS_BIN, "-q", script_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            pytest.fail(
                f"Yosys equivalence check FAILED\n"
                f"--- gold.sv ---\n{gold_sv}\n"
                f"--- uut.sv ---\n{uut_sv}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
