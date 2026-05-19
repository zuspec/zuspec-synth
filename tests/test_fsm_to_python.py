# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for FSMToPythonPass — Phase 2 of IR→Python round-trip."""
import sys
import os
import pytest

_this_dir = os.path.dirname(os.path.abspath(__file__))
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
_blinky_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "blinky")
_rotate_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "rotate")
_uart_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "uart")

for _p in (_synth_src, _dc_src, _blinky_dir, _rotate_dir, _uart_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zuspec.dataclasses.data_model_factory import DataModelFactory
from zuspec.synth.passes.component_fields import ComponentFieldsPass
from zuspec.synth.passes.process_to_fsm import ProcessToFSMPass
from zuspec.synth.passes.fsm_to_python import FSMToPythonPass
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig


def _build_ir(component_cls, *extra_passes):
    """Build SynthIR up to (and including) FSMToPythonPass for a component."""
    ctx = DataModelFactory().build(component_cls)
    ir = SynthIR(component=component_cls, model_context=ctx)
    cfg = SynthConfig()
    ir = ComponentFieldsPass(cfg).run(ir)
    ir = ProcessToFSMPass(cfg).run(ir)
    ir = FSMToPythonPass(cfg).run(ir)
    return ir


# ---------------------------------------------------------------------------
# Single-state (blinky) tests
# ---------------------------------------------------------------------------

class TestSingleStateFSM:
    @pytest.fixture(scope="class")
    def ir(self):
        import blink as spl_blink
        return _build_ir(spl_blink.Blink)

    def test_single_state_produces_sync_method(self, ir):
        """Single-state FSM should populate ``py/module/sync``."""
        src = ir.lowered_py.get("py/module/sync", "")
        assert src, "py/module/sync should not be empty for single-state FSM"

    def test_single_state_has_zdc_sync_decorator(self, ir):
        src = ir.lowered_py["py/module/sync"]
        assert "@zdc.sync" in src

    def test_single_state_no_top_key(self, ir):
        """Single-state path must NOT set py/module/top."""
        assert "py/module/top" not in ir.lowered_py

    def test_single_state_no_tick_in_output(self, ir):
        """Clock ticks should not appear in the Python body."""
        src = ir.lowered_py["py/module/sync"]
        assert "tick()" not in src

    def test_single_state_counter_assignment(self, ir):
        """Blinky body should contain a counter increment."""
        src = ir.lowered_py["py/module/sync"]
        assert "_counter" in src

    def test_single_state_augmented_assign(self, ir):
        """Counter increment should use ``+=``."""
        src = ir.lowered_py["py/module/sync"]
        assert "+=" in src


# ---------------------------------------------------------------------------
# Multi-state (rotate) tests
# ---------------------------------------------------------------------------

class TestMultiStateFSM:
    @pytest.fixture(scope="class")
    def ir(self):
        import rotate as spl_rotate
        return _build_ir(spl_rotate.Blink)

    def test_multi_state_produces_top_key(self, ir):
        """Multi-state FSM should populate ``py/module/top``."""
        src = ir.lowered_py.get("py/module/top", "")
        assert src, "py/module/top should not be empty for multi-state FSM"

    def test_multi_state_skips_lowered_py_module_sync(self, ir):
        """Multi-state path must NOT set py/module/sync."""
        assert "py/module/sync" not in ir.lowered_py

    def test_multi_state_has_class_declaration(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "class Blink" in src

    def test_multi_state_has_zdc_dataclass(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "@zdc.dataclass" in src

    def test_multi_state_has_sync_component(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "zdc.SyncComponent" in src

    def test_multi_state_has_import(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "import zuspec.dataclasses as zdc" in src

    def test_multi_state_contains_state_register(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "_state" in src
        assert "zdc.field(" in src

    def test_multi_state_contains_wait_counter(self, ir):
        """WAIT_CYCLES state requires a cycle counter register."""
        src = ir.lowered_py["py/module/top"]
        assert "_S_3_cnt" in src

    def test_multi_state_contains_inferred_register(self, ir):
        """Loop variable ``i`` should become an auto-inferred ``_i`` register."""
        src = ir.lowered_py["py/module/top"]
        assert "_i" in src

    def test_multi_state_contains_all_state_branches(self, ir):
        """All three states (IDLE, LOOP_I_CHK, S_3) should appear as branches."""
        src = ir.lowered_py["py/module/top"]
        assert "# IDLE" in src
        assert "# LOOP_I_CHK" in src
        assert "# S_3" in src

    def test_multi_state_contains_ports(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "BTN_N" in src
        for port in ("L1", "L2", "L3", "L4"):
            assert port in src

    def test_multi_state_port_directions(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "zdc.input()" in src   # BTN_N
        assert "zdc.output()" in src  # L1-L4

    def test_multi_state_has_sync_method(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "@zdc.sync" in src
        assert "def _fsm(self):" in src

    def test_wait_cycles_state_has_counter_decrement(self, ir):
        """S_3 state should decrement the counter in the else branch."""
        src = ir.lowered_py["py/module/top"]
        assert "_S_3_cnt = self._S_3_cnt - 1" in src

    def test_wait_cycles_state_no_double_counter_check(self, ir):
        """The counter-zero check should appear exactly once in the S_3 branch.

        A duplicate check would cause a correctness bug (the counter was
        already modified by the else branch on the previous cycle).
        """
        src = ir.lowered_py["py/module/top"]
        # Count occurrences of the S_3_cnt == 0 test
        count = src.count("self._S_3_cnt == 0")
        assert count == 1, (
            f"Expected 1 counter-zero test, found {count} — "
            "duplicate would be a correctness bug"
        )

    def test_loop_i_chk_else_branch(self, ir):
        """LOOP_I_CHK conditional transition should have an else fallback."""
        src = ir.lowered_py["py/module/top"]
        # After the "if self._i < 4:" branch there must be an else
        assert "else:" in src

    def test_loop_counter_init_on_entry(self, ir):
        """When entering S_3, counter should be initialised to wait_cycles-1."""
        src = ir.lowered_py["py/module/top"]
        # 4194303 == 4194304 - 1  (rotate uses ~1 second at 4MHz-ish)
        assert "4194303" in src

    def test_reset_domain_none_propagated(self, ir):
        """rotate uses ResetDomain(style='none') — should appear in output."""
        src = ir.lowered_py["py/module/top"]
        assert "reset_domain" in src
        assert "none" in src

    def test_output_is_valid_python(self, ir):
        """The generated source should be syntactically valid Python."""
        src = ir.lowered_py["py/module/top"]
        try:
            compile(src, "<generated>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Generated Python has syntax error: {exc}")

    def test_led_assignments_use_comparison(self, ir):
        """LED outputs should use comparison expressions (e.g., ``i == 0``)."""
        src = ir.lowered_py["py/module/top"]
        assert "self._i == 0" in src or "self._i==" in src.replace(" ", "")

    def test_i_incremented_in_s3_body(self, ir):
        """``_i`` should be incremented inside the S_3 state body."""
        src = ir.lowered_py["py/module/top"]
        assert "self._i = self._i + 1" in src or "self._i += " in src


class TestUartRxLowering:
    @pytest.fixture(scope="class")
    def ir(self):
        import uart_rx as spl_uart
        return _build_ir(spl_uart.UartRx)

    def test_uart_output_is_valid_python(self, ir):
        src = ir.lowered_py["py/module/top"]
        try:
            compile(src, "<generated-uart>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Generated UART Python has syntax error: {exc}")

    def test_uart_exprcall_nodes_are_lowered(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "ExprCall(" not in src

    def test_uart_references_keep_self_prefix(self, ir):
        src = ir.lowered_py["py/module/top"]
        assert "if (self.rx == 1):" in src
        assert "self._data = (self._data | (self.rx << self._bit_i))" in src
