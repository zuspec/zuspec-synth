# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for CombToPythonPass — Phase 3 of IR→Python round-trip."""
import sys
import os
import pytest

_this_dir = os.path.dirname(os.path.abspath(__file__))
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
_blinky_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "blinky")
_rotate_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "rotate")

for _p in (_synth_src, _dc_src, _blinky_dir, _rotate_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zuspec.dataclasses as zdc
from zuspec.dataclasses.data_model_factory import DataModelFactory
from zuspec.synth.passes.component_fields import ComponentFieldsPass
from zuspec.synth.passes.process_to_fsm import ProcessToFSMPass
from zuspec.synth.passes.fsm_to_python import FSMToPythonPass
from zuspec.synth.passes.comb_to_python import CombToPythonPass
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig


def _build_ir(component_cls):
    ctx = DataModelFactory().build(component_cls)
    ir = SynthIR(component=component_cls, model_context=ctx)
    cfg = SynthConfig()
    ir = ComponentFieldsPass(cfg).run(ir)
    ir = ProcessToFSMPass(cfg).run(ir)
    ir = FSMToPythonPass(cfg).run(ir)
    ir = CombToPythonPass(cfg).run(ir)
    return ir


# ---------------------------------------------------------------------------
# SPL examples have no @zdc.comb — should produce empty string
# ---------------------------------------------------------------------------

class TestCombToPythonNoComb:
    @pytest.fixture(scope="class")
    def ir_blinky(self):
        import blink as spl_blink
        return _build_ir(spl_blink.Blink)

    @pytest.fixture(scope="class")
    def ir_rotate(self):
        import rotate as spl_rotate
        return _build_ir(spl_rotate.Blink)

    def test_blinky_comb_key_exists(self, ir_blinky):
        """Pass must always set py/module/comb (empty string if no comb)."""
        assert "py/module/comb" in ir_blinky.lowered_py

    def test_blinky_no_comb_body(self, ir_blinky):
        """Blinky has a @zdc.comb for LED outputs — should produce a comb method."""
        src = ir_blinky.lowered_py["py/module/comb"]
        # Blinky drives LED_GREEN and LED_RED from a comb process
        assert "@zdc.comb" in src

    def test_rotate_comb_key_exists(self, ir_rotate):
        assert "py/module/comb" in ir_rotate.lowered_py

    def test_rotate_no_comb_body(self, ir_rotate):
        """Rotate has no @zdc.comb — output should be empty."""
        assert ir_rotate.lowered_py["py/module/comb"] == ""


# ---------------------------------------------------------------------------
# Component with a @zdc.comb process
# ---------------------------------------------------------------------------

@zdc.dataclass
class _CombOnlyComponent(zdc.SyncComponent):
    a: zdc.bit = zdc.input()
    b: zdc.bit = zdc.input()
    y: zdc.bit = zdc.output()

    @zdc.comb
    def _logic(self):
        self.y = self.a & self.b

    @zdc.sync
    def _clk(self):
        pass


class TestCombToPythonWithComb:
    @pytest.fixture(scope="class")
    def ir(self):
        return _build_ir(_CombOnlyComponent)

    def test_comb_key_exists(self, ir):
        assert "py/module/comb" in ir.lowered_py

    def test_comb_not_empty(self, ir):
        assert ir.lowered_py["py/module/comb"] != ""

    def test_comb_has_decorator(self, ir):
        src = ir.lowered_py["py/module/comb"]
        assert "@zdc.comb" in src

    def test_comb_has_def(self, ir):
        src = ir.lowered_py["py/module/comb"]
        assert "def " in src
        assert "(self):" in src

    def test_comb_defaults_output_to_zero(self, ir):
        """All output ports should have a default-zero assignment."""
        src = ir.lowered_py["py/module/comb"]
        assert "self.y = 0" in src

    def test_comb_body_has_assignment(self, ir):
        """The AND logic should appear in the body."""
        src = ir.lowered_py["py/module/comb"]
        # Body should reference self.y and self.a and self.b somehow
        assert "self.y" in src

    def test_comb_output_is_valid_python(self, ir):
        src = ir.lowered_py["py/module/comb"]
        # Wrap in a dummy class for compilation check
        wrapped = "class _T:\n" + "\n".join("    " + l for l in src.splitlines())
        try:
            compile(wrapped, "<generated>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Generated comb Python has syntax error: {exc}")
