# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for lower_to_python() public API — Phase 5."""
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


class TestLowerToPythonBlinky:
    @pytest.fixture(scope="class")
    def src(self):
        from zuspec.synth import lower_to_python
        import blink as spl_blink
        return lower_to_python(spl_blink.Blink)

    def test_returns_non_empty_string(self, src):
        assert isinstance(src, str)
        assert src

    def test_has_import(self, src):
        assert "import zuspec.dataclasses as zdc" in src

    def test_has_class_declaration(self, src):
        assert "class Blink" in src

    def test_has_zdc_dataclass(self, src):
        assert "@zdc.dataclass" in src

    def test_has_sync_component(self, src):
        assert "zdc.SyncComponent" in src

    def test_has_sync_method(self, src):
        assert "@zdc.sync" in src

    def test_has_comb_method(self, src):
        assert "@zdc.comb" in src

    def test_counter_register(self, src):
        assert "_counter" in src

    def test_ports_present(self, src):
        assert "BTN_N" in src
        assert "LED_GREEN" in src
        assert "LED_RED" in src

    def test_is_valid_python(self, src):
        try:
            compile(src, "<blinky_roundtrip>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Blinky round-trip output has syntax error: {exc}")

    def test_no_tick_calls(self, src):
        assert "tick()" not in src

    def test_no_sv_keywords(self, src):
        """Output should not contain SV artifacts."""
        for kw in ("always_ff", "always_comb", "endmodule", "logic ", "wire "):
            assert kw not in src, f"SV keyword {kw!r} found in Python output"


class TestLowerToPythonRotate:
    @pytest.fixture(scope="class")
    def src(self):
        from zuspec.synth import lower_to_python
        import rotate as spl_rotate
        return lower_to_python(spl_rotate.Blink)

    def test_returns_non_empty_string(self, src):
        assert isinstance(src, str)
        assert src

    def test_has_import(self, src):
        assert "import zuspec.dataclasses as zdc" in src

    def test_has_class_declaration(self, src):
        assert "class Blink" in src

    def test_is_valid_python(self, src):
        try:
            compile(src, "<rotate_roundtrip>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Rotate round-trip output has syntax error: {exc}")

    def test_all_leds_present(self, src):
        for p in ("L1", "L2", "L3", "L4"):
            assert p in src

    def test_state_register_present(self, src):
        assert "_state" in src

    def test_wait_counter_present(self, src):
        assert "_S_3_cnt" in src

    def test_inferred_loop_var(self, src):
        assert "_i" in src

    def test_no_sv_keywords(self, src):
        for kw in ("always_ff", "always_comb", "endmodule", "logic ", "wire "):
            assert kw not in src, f"SV keyword {kw!r} found in Python output"

    def test_no_tick_calls(self, src):
        assert "tick()" not in src


class TestLowerToPythonTopOverride:
    def test_top_rename_blinky(self):
        from zuspec.synth import lower_to_python
        import blink as spl_blink
        src = lower_to_python(spl_blink.Blink, top="MyBlinker")
        assert "class MyBlinker" in src
        assert "class Blink" not in src

    def test_top_rename_rotate(self):
        from zuspec.synth import lower_to_python
        import rotate as spl_rotate
        src = lower_to_python(spl_rotate.Blink, top="RotateLEDs")
        assert "class RotateLEDs" in src
        assert "class Blink" not in src

    def test_top_none_keeps_original_name(self):
        from zuspec.synth import lower_to_python
        import blink as spl_blink
        src = lower_to_python(spl_blink.Blink, top=None)
        assert "class Blink" in src
