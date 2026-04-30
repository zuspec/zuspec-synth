"""Integration tests for port-method call lowering with multiple ports.

This test exercises synthesis of a component with *two* ProtocolPort fields —
one awaited (fetch), one non-awaited (monitor) — to verify that the synthesis
engine handles multiple ports correctly in a single component.

All test data (component definition + interfaces) is inline; no external
project paths are required.
"""

import typing
import pytest

import zuspec.dataclasses as zdc


# ---------------------------------------------------------------------------
# Inline test component: FetchMonitor
#
# Models a minimal fetch unit that reads from a memory port and reports each
# fetch to a monitor port. Structurally similar to RV32ICore's _run loop but
# without any ISA-specific logic.
# ---------------------------------------------------------------------------

class FetchIface(typing.Protocol):
    async def read_word(self, addr: zdc.u32) -> zdc.u32: ...


class MonitorIface(typing.Protocol):
    def on_fetch(self, addr: zdc.u32, data: zdc.u32) -> None: ...


@zdc.dataclass
class FetchMonitor(zdc.Component):
    """Component with one awaited port (fetch) and one non-awaited port (monitor)."""
    fetch:   FetchIface   = zdc.port()
    monitor: MonitorIface = zdc.port()

    @zdc.proc
    async def _run(self):
        addr = zdc.u32(0)
        while True:
            data = await self.fetch.read_word(addr)
            self.monitor.on_fetch(addr, data)
            addr = zdc.u32(addr + 4)


# ---------------------------------------------------------------------------
# Second inline component: DualReader
#
# Two independently-awaited port calls in sequence to verify that each gets
# its own FSM state and distinct signal names.
# ---------------------------------------------------------------------------

class PortA(typing.Protocol):
    async def load(self, addr: zdc.u32) -> zdc.u32: ...


class PortB(typing.Protocol):
    async def store(self, addr: zdc.u32, val: zdc.u32) -> zdc.u32: ...


@zdc.dataclass
class DualReader(zdc.Component):
    """Component with two distinct awaited port calls."""
    porta: PortA = zdc.port()
    portb: PortB = zdc.port()

    @zdc.proc
    async def _run(self):
        while True:
            x = await self.porta.load(0)
            y = await self.portb.store(0, x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fetch_monitor_sv():
    from zuspec.synth import synthesize
    sv = synthesize(FetchMonitor)
    assert sv, "synthesize(FetchMonitor) returned empty"
    return sv


@pytest.fixture(scope="module")
def dual_reader_sv():
    from zuspec.synth import synthesize
    sv = synthesize(DualReader)
    assert sv, "synthesize(DualReader) returned empty"
    return sv


# ---------------------------------------------------------------------------
# Tests: FetchMonitor (awaited + non-awaited ports in one component)
# ---------------------------------------------------------------------------

class TestFetchMonitorMultiPort:
    def test_synthesizes(self, fetch_monitor_sv):
        assert len(fetch_monitor_sv) > 50

    def test_module_name(self, fetch_monitor_sv):
        assert "module FetchMonitor" in fetch_monitor_sv

    def test_fsm_style(self, fetch_monitor_sv):
        assert "typedef enum" in fetch_monitor_sv

    # Awaited port: fetch.read_word
    def test_fetch_valid(self, fetch_monitor_sv):
        assert "fetch_read_word_valid" in fetch_monitor_sv

    def test_fetch_arg(self, fetch_monitor_sv):
        assert "fetch_read_word_arg0" in fetch_monitor_sv

    def test_fetch_ack(self, fetch_monitor_sv):
        assert "fetch_read_word_ack" in fetch_monitor_sv

    def test_fetch_rdata(self, fetch_monitor_sv):
        assert "fetch_read_word_rdata" in fetch_monitor_sv

    def test_fetch_wait_state(self, fetch_monitor_sv):
        assert "FETCH_READ_WORD_REQ" in fetch_monitor_sv

    # Non-awaited port: monitor.on_fetch
    def test_monitor_valid(self, fetch_monitor_sv):
        assert "monitor_on_fetch_valid" in fetch_monitor_sv

    def test_monitor_arg0(self, fetch_monitor_sv):
        assert "monitor_on_fetch_arg0" in fetch_monitor_sv

    def test_monitor_arg1(self, fetch_monitor_sv):
        assert "monitor_on_fetch_arg1" in fetch_monitor_sv

    def test_monitor_no_ack(self, fetch_monitor_sv):
        assert "monitor_on_fetch_ack" not in fetch_monitor_sv

    def test_monitor_no_wait_state(self, fetch_monitor_sv):
        assert "MONITOR_ON_FETCH_REQ" not in fetch_monitor_sv

    def test_both_ports_distinct(self, fetch_monitor_sv):
        # Signals for the two ports must not collide
        assert "fetch_read_word_valid" in fetch_monitor_sv
        assert "monitor_on_fetch_valid" in fetch_monitor_sv


# ---------------------------------------------------------------------------
# Tests: DualReader (two distinct awaited ports)
# ---------------------------------------------------------------------------

class TestDualReaderDistinctStates:
    def test_synthesizes(self, dual_reader_sv):
        assert len(dual_reader_sv) > 50

    def test_porta_valid(self, dual_reader_sv):
        assert "porta_load_valid" in dual_reader_sv

    def test_porta_ack(self, dual_reader_sv):
        assert "porta_load_ack" in dual_reader_sv

    def test_porta_wait_state(self, dual_reader_sv):
        assert "PORTA_LOAD_REQ" in dual_reader_sv

    def test_portb_valid(self, dual_reader_sv):
        assert "portb_store_valid" in dual_reader_sv

    def test_portb_ack(self, dual_reader_sv):
        assert "portb_store_ack" in dual_reader_sv

    def test_portb_wait_state(self, dual_reader_sv):
        assert "PORTB_STORE_REQ" in dual_reader_sv

    def test_two_distinct_wait_states(self, dual_reader_sv):
        # Each port call must produce its own WAIT_COND state
        assert "PORTA_LOAD_REQ" in dual_reader_sv
        assert "PORTB_STORE_REQ" in dual_reader_sv
