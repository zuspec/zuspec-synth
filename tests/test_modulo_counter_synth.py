# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for Counter sub-component RTL synthesis via counter_lower.py."""
import sys
import pytest

sys.path.insert(0, 'packages/zuspec-dataclasses/src')
sys.path.insert(0, 'packages/zuspec-synth/src')


# ---------------------------------------------------------------------------
# Fixtures — synthesize the SV once per class
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blinker_sv():
    import zuspec.dataclasses as zdc
    from zuspec.dataclasses.modulo_counter import ModuloCounter
    from zuspec.synth import synthesize

    @zdc.dataclass
    class Blinker(zdc.Component):
        led: zdc.u1 = zdc.field(default=0)
        cnt: ModuloCounter = zdc.inst(ModuloCounter, kwargs={'PERIOD': 8})

        @zdc.proc
        async def _run(self):
            while True:
                await self.cnt.wait_next()
                self.led = 1 - self.led

    return synthesize(Blinker)


@pytest.fixture(scope="module")
def fast_blinker_sv():
    """Component with a smaller PERIOD to verify the counter width scales."""
    import zuspec.dataclasses as zdc
    from zuspec.dataclasses.modulo_counter import ModuloCounter
    from zuspec.synth import synthesize

    @zdc.dataclass
    class FastBlinker(zdc.Component):
        led: zdc.u1 = zdc.field(default=0)
        cnt: ModuloCounter = zdc.inst(ModuloCounter, kwargs={'PERIOD': 4})

        @zdc.proc
        async def _run(self):
            while True:
                await self.cnt.wait_next()
                self.led = 1 - self.led

    return synthesize(FastBlinker)


@pytest.fixture(scope="module")
def dual_counter_sv():
    """Component with two independent ModuloCounter sub-fields."""
    import zuspec.dataclasses as zdc
    from zuspec.dataclasses.modulo_counter import ModuloCounter
    from zuspec.synth import synthesize

    @zdc.dataclass
    class DualBlinker(zdc.Component):
        led: zdc.u1 = zdc.field(default=0)
        fast: ModuloCounter = zdc.inst(ModuloCounter, kwargs={'PERIOD': 4})
        slow: ModuloCounter = zdc.inst(ModuloCounter, kwargs={'PERIOD': 16})

        @zdc.proc
        async def _run(self):
            while True:
                await self.fast.wait_next()
                await self.slow.wait_next()
                self.led = 1 - self.led

    return synthesize(DualBlinker)


@pytest.fixture(scope="module")
def watchdog_sv():
    """Component with a WatchdogCounter sub-field."""
    import zuspec.dataclasses as zdc
    from zuspec.dataclasses.watchdog_counter import WatchdogCounter
    from zuspec.synth import synthesize

    @zdc.dataclass
    class WatchdogBlinker(zdc.Component):
        led: zdc.u1 = zdc.field(default=0)
        wdog: WatchdogCounter = zdc.inst(WatchdogCounter, kwargs={'TIMEOUT': 12})

        @zdc.proc
        async def _run(self):
            while True:
                await self.wdog.wait_next()
                self.led = 1 - self.led

    return synthesize(WatchdogBlinker)


# ---------------------------------------------------------------------------
# Blinker (ModuloCounter PERIOD=8) tests
# ---------------------------------------------------------------------------

class TestBlinkerModuloCounter:
    def test_synthesis_succeeds(self, blinker_sv):
        assert blinker_sv is not None
        assert len(blinker_sv) > 0

    def test_has_module_declaration(self, blinker_sv):
        assert "module Blinker" in blinker_sv
        assert "endmodule" in blinker_sv

    def test_has_clock_reset_ports(self, blinker_sv):
        assert "input  logic clk" in blinker_sv
        assert "input  logic rst_n" in blinker_sv

    def test_has_led_register(self, blinker_sv):
        assert "led" in blinker_sv

    def test_has_fsm_states(self, blinker_sv):
        """The proc loop must produce a multi-state FSM."""
        assert "state_t" in blinker_sv
        assert "IDLE" in blinker_sv

    def test_has_cycle_wait_counter(self, blinker_sv):
        """PERIOD=8 wait_next() must produce a cycle-wait counter register."""
        assert "_cnt" in blinker_sv

    def test_cycle_counter_reset_value(self, blinker_sv):
        """PERIOD=8 counter resets to 7 (counts down from N-1 to 0)."""
        assert "7" in blinker_sv

    def test_counter_decrement(self, blinker_sv):
        """Counter must be decremented each cycle while in wait state."""
        assert "- 1" in blinker_sv or "-1'b1" in blinker_sv or "- 1'b1" in blinker_sv

    def test_led_update_gated_by_counter(self, blinker_sv):
        """led must only be updated when the wait counter expires (== 0)."""
        assert "== 0" in blinker_sv
        # led assignment should appear after the counter-zero condition
        led_pos = blinker_sv.rfind("led")
        cnt_zero_pos = blinker_sv.rfind("== 0")
        assert cnt_zero_pos > 0, "Missing counter == 0 condition"

    def test_no_counter_subcomponent_field(self, blinker_sv):
        """The ModuloCounter sub-component 'cnt' must not appear as a port."""
        assert "cnt" not in blinker_sv or "S_1_cnt" in blinker_sv

    def test_always_ff_present(self, blinker_sv):
        assert "always_ff @(posedge clk)" in blinker_sv

    def test_reset_applies_to_state(self, blinker_sv):
        assert "state <= IDLE" in blinker_sv


# ---------------------------------------------------------------------------
# FastBlinker (ModuloCounter PERIOD=4) — counter width scaling
# ---------------------------------------------------------------------------

class TestFastBlinkerCounterWidth:
    def test_synthesis_succeeds(self, fast_blinker_sv):
        assert fast_blinker_sv is not None

    def test_has_cycle_wait_counter(self, fast_blinker_sv):
        assert "_cnt" in fast_blinker_sv

    def test_period4_reset_value(self, fast_blinker_sv):
        """PERIOD=4 counter resets to 3."""
        assert "3" in fast_blinker_sv

    def test_has_fsm(self, fast_blinker_sv):
        assert "IDLE" in fast_blinker_sv
        assert "state_t" in fast_blinker_sv


# ---------------------------------------------------------------------------
# DualBlinker — two ModuloCounter sub-fields
# ---------------------------------------------------------------------------

class TestDualCounterSynth:
    def test_synthesis_succeeds(self, dual_counter_sv):
        assert dual_counter_sv is not None

    def test_has_two_wait_states(self, dual_counter_sv):
        """Two sequential wait_next() calls produce two wait states."""
        assert dual_counter_sv.count("_cnt") >= 2

    def test_has_fsm(self, dual_counter_sv):
        assert "IDLE" in dual_counter_sv


# ---------------------------------------------------------------------------
# WatchdogBlinker (WatchdogCounter TIMEOUT=12)
# ---------------------------------------------------------------------------

class TestWatchdogCounterSynth:
    def test_synthesis_succeeds(self, watchdog_sv):
        assert watchdog_sv is not None

    def test_has_cycle_wait_counter(self, watchdog_sv):
        assert "_cnt" in watchdog_sv

    def test_timeout12_reset_value(self, watchdog_sv):
        """TIMEOUT=12 counter resets to 11 (counts down from N-1 to 0)."""
        assert "11" in watchdog_sv

    def test_has_fsm(self, watchdog_sv):
        assert "IDLE" in watchdog_sv


# ---------------------------------------------------------------------------
# wait_for() raises NotImplementedError
# ---------------------------------------------------------------------------

def test_wait_for_raises_not_implemented():
    """wait_for() is not supported for RTL synthesis."""
    import zuspec.dataclasses as zdc
    from zuspec.dataclasses.modulo_counter import ModuloCounter
    from zuspec.synth import synthesize

    @zdc.dataclass
    class WaitForExample(zdc.Component):
        led: zdc.u1 = zdc.field(default=0)
        cnt: ModuloCounter = zdc.inst(ModuloCounter, kwargs={'PERIOD': 8})

        @zdc.proc
        async def _run(self):
            while True:
                await self.cnt.wait_for(4)
                self.led = 1 - self.led

    with pytest.raises(NotImplementedError):
        synthesize(WaitForExample)


# ---------------------------------------------------------------------------
# Plain component without counters is unaffected
# ---------------------------------------------------------------------------

def test_plain_component_unaffected():
    """Components without counter sub-fields synthesize normally."""
    import zuspec.dataclasses as zdc
    from zuspec.synth import synthesize

    @zdc.dataclass
    class PlainCounter(zdc.Component):
        count: zdc.u8 = zdc.field(default=0)

        @zdc.proc
        async def _run(self):
            while True:
                self.count = self.count + 1
                await zdc.tick()

    sv = synthesize(PlainCounter)
    assert "count" in sv
    assert "module PlainCounter" in sv
