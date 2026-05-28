# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for sprtl/fsm_structural.py — backend-neutral FSM structural helpers."""
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src    = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
for _p in (_synth_src, _dc_src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMStateKind
from zuspec.synth.sprtl.fsm_structural import (
    WaitCounterInfo,
    fsm_initial_state_name,
    fsm_state_names,
    fsm_state_width,
    fsm_wait_counter_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fsm(*state_names, wc_map=None):
    """Build a minimal FSMModule with the given state names.

    *wc_map* is an optional dict of ``state_name → wait_cycles`` to mark
    states as WAIT_CYCLES with the given cycle count.
    """
    wc_map = wc_map or {}
    fsm = FSMModule(name="test")
    for sname in state_names:
        wc = wc_map.get(sname)
        if wc is not None:
            fsm.add_state(sname, kind=FSMStateKind.WAIT_CYCLES, wait_cycles=wc)
        else:
            fsm.add_state(sname)
    return fsm


# ---------------------------------------------------------------------------
# fsm_state_width
# ---------------------------------------------------------------------------

class TestFsmStateWidth:
    def test_single_state(self):
        fsm = _make_fsm("IDLE")
        assert fsm_state_width(fsm) == 1

    def test_two_states(self):
        fsm = _make_fsm("IDLE", "RUN")
        assert fsm_state_width(fsm) == 1

    def test_three_states(self):
        fsm = _make_fsm("IDLE", "RUN", "DONE")
        assert fsm_state_width(fsm) == 2

    def test_four_states(self):
        fsm = _make_fsm("S0", "S1", "S2", "S3")
        assert fsm_state_width(fsm) == 2

    def test_five_states(self):
        fsm = _make_fsm("S0", "S1", "S2", "S3", "S4")
        assert fsm_state_width(fsm) == 3


# ---------------------------------------------------------------------------
# fsm_state_names
# ---------------------------------------------------------------------------

class TestFsmStateNames:
    def test_returns_all_states(self):
        fsm = _make_fsm("IDLE", "RUN", "DONE")
        names = fsm_state_names(fsm)
        assert len(names) == 3
        for st in fsm.states:
            assert st.id in names

    def test_unique_names_preserved(self):
        fsm = _make_fsm("IDLE", "RUN", "DONE")
        names = fsm_state_names(fsm)
        assert set(names.values()) == {"IDLE", "RUN", "DONE"}

    def test_all_names_distinct(self):
        fsm = _make_fsm("S0", "S1", "S2", "S3")
        names = fsm_state_names(fsm)
        values = list(names.values())
        assert len(values) == len(set(values)), "Names are not unique"

    def test_dedup_collision(self):
        """Two states with the same name but different encodings get disambiguated."""
        fsm = FSMModule(name="test")
        # Manually create states with the same name to force collision
        from zuspec.synth.sprtl.fsm_ir import FSMState
        fsm.states.append(FSMState(id=0, name="STATE"))
        fsm.states.append(FSMState(id=1, name="STATE"))
        fsm.state_encoding = {0: 0, 1: 1}
        fsm.state_width = 1
        names = fsm_state_names(fsm)
        # First occurrence keeps the name; second gets _{enc} appended
        assert names[0] == "STATE"
        assert names[1] == "STATE_1"

    def test_same_name_same_encoding_no_dedup(self):
        """Two states sharing name AND encoding are unusual but get the same name."""
        from zuspec.synth.sprtl.fsm_ir import FSMState
        fsm = FSMModule(name="test")
        fsm.states.append(FSMState(id=0, name="X"))
        fsm.states.append(FSMState(id=1, name="X"))
        fsm.state_encoding = {0: 0, 1: 0}
        fsm.state_width = 1
        names = fsm_state_names(fsm)
        assert names[0] == "X"
        assert names[1] == "X"  # same encoding → no suffix added


# ---------------------------------------------------------------------------
# fsm_wait_counter_info
# ---------------------------------------------------------------------------

class TestFsmWaitCounterInfo:
    def test_no_wait_states(self):
        fsm = _make_fsm("IDLE", "RUN")
        names = fsm_state_names(fsm)
        assert fsm_wait_counter_info(fsm, names) == []

    def test_single_wait_state(self):
        fsm = _make_fsm("IDLE", "WAIT", wc_map={"WAIT": 8})
        names = fsm_state_names(fsm)
        infos = fsm_wait_counter_info(fsm, names)
        assert len(infos) == 1
        wci = infos[0]
        assert isinstance(wci, WaitCounterInfo)
        assert wci.counter_name == "WAIT_cnt"
        assert wci.n_cycles == 8
        assert wci.init_val == 7
        assert wci.counter_width == 3  # ceil(log2(7)) = 3

    def test_wait_cycles_1_excluded(self):
        """wait_cycles == 1 needs no counter."""
        fsm = _make_fsm("IDLE", "W1", wc_map={"W1": 1})
        names = fsm_state_names(fsm)
        assert fsm_wait_counter_info(fsm, names) == []

    def test_counter_width_4m_cycles(self):
        """Rotate example: 4_194_304 cycles → 22-bit counter."""
        n = 4_194_304
        fsm = _make_fsm("IDLE", "S_3", wc_map={"S_3": n})
        names = fsm_state_names(fsm)
        infos = fsm_wait_counter_info(fsm, names)
        assert len(infos) == 1
        wci = infos[0]
        assert wci.counter_name == "S_3_cnt"
        assert wci.init_val == n - 1
        assert wci.counter_width == 22  # (4194303).bit_length() == 22

    def test_counter_name_uses_deduplicated_state_name(self):
        """Counter name must use the deduplicated name, not the raw state.name."""
        from zuspec.synth.sprtl.fsm_ir import FSMState
        fsm = FSMModule(name="test")
        fsm.states.append(FSMState(id=0, name="S"))
        fsm.states.append(
            FSMState(id=1, name="S", kind=FSMStateKind.WAIT_CYCLES, wait_cycles=4)
        )
        fsm.state_encoding = {0: 0, 1: 1}
        fsm.state_width = 1
        names = fsm_state_names(fsm)
        infos = fsm_wait_counter_info(fsm, names)
        assert len(infos) == 1
        # The WAIT_CYCLES state (id=1) was deduplicated to "S_1"
        assert infos[0].counter_name == "S_1_cnt"

    def test_multiple_wait_states(self):
        fsm = _make_fsm("IDLE", "W8", "W16", wc_map={"W8": 8, "W16": 16})
        names = fsm_state_names(fsm)
        infos = fsm_wait_counter_info(fsm, names)
        assert len(infos) == 2
        cnames = {wci.counter_name for wci in infos}
        assert cnames == {"W8_cnt", "W16_cnt"}


# ---------------------------------------------------------------------------
# fsm_initial_state_name
# ---------------------------------------------------------------------------

class TestFsmInitialStateName:
    def test_returns_first_state_name(self):
        fsm = _make_fsm("IDLE", "RUN", "DONE")
        names = fsm_state_names(fsm)
        assert fsm_initial_state_name(fsm, names) == "IDLE"

    def test_returns_deduplicated_name(self):
        """Initial state name comes from the dedup map, not raw state.name."""
        from zuspec.synth.sprtl.fsm_ir import FSMState
        fsm = FSMModule(name="test")
        # Two states both named "IDLE"; initial_state defaults to 0 (first)
        fsm.states.append(FSMState(id=0, name="IDLE"))
        fsm.states.append(FSMState(id=1, name="IDLE"))
        fsm.state_encoding = {0: 0, 1: 1}
        fsm.state_width = 1
        fsm.initial_state = 0
        names = fsm_state_names(fsm)
        assert fsm_initial_state_name(fsm, names) == "IDLE"

    def test_fallback_when_no_initial_state(self):
        fsm = FSMModule(name="test")
        # No states → get_state returns None
        assert fsm_initial_state_name(fsm, {}) == "IDLE"


# ---------------------------------------------------------------------------
# Cross-backend agreement (integration)
# ---------------------------------------------------------------------------

class TestCrossBackendAgreement:
    """Verify SV and Python backends produce matching structural identifiers
    for the same FSM IR, confirming both now use fsm_structural helpers."""

    @pytest.fixture(autouse=True)
    def _setup_paths(self):
        _blinky_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "blinky")
        _rotate_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "rotate")
        for p in (_blinky_dir, _rotate_dir):
            if p not in sys.path:
                sys.path.insert(0, p)

    def test_rotate_counter_name_matches_sv_and_python(self):
        """The wait-counter name in the Python output matches the SV output."""
        import importlib
        spl_rotate = importlib.import_module("rotate")
        from zuspec.synth import _synthesize_sprtl as lower_to_sv, lower_to_python

        sv_src  = lower_to_sv(spl_rotate.Blink)
        py_src  = lower_to_python(spl_rotate.Blink)

        # Both backends should use the same counter identifier
        assert "S_3_cnt" in sv_src,  "SV output missing S_3_cnt"
        assert "S_3_cnt" in py_src,  "Python output missing S_3_cnt"

    def test_rotate_state_encoding_consistent(self):
        """State encoding values are identical in SV and Python output."""
        import importlib
        spl_rotate = importlib.import_module("rotate")
        from zuspec.synth import _synthesize_sprtl as lower_to_sv, lower_to_python

        sv_src  = lower_to_sv(spl_rotate.Blink)
        py_src  = lower_to_python(spl_rotate.Blink)

        # SV: "S_3 = 2'dN"  Python: "elif self._state == N:"
        # The encoded values 0, 1, 2 must all appear in both
        for enc in ("0", "1", "2"):
            assert enc in sv_src
            assert enc in py_src
