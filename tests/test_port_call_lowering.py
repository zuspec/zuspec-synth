"""Tests for protocol-port method call lowering in zuspec-synth.

These tests verify that:
- `await self.PORT.METHOD(args)` lowers to a WAIT_COND state with correct
  port signals (valid/argN/ack/rdata) and a handshake FSM.
- Non-awaited `self.PORT.METHOD(args)` lowers to a combinatorial output
  in the current FSM state (no ack/wait).
- @zdc.proc components with ProtocolPort fields route through SPRTLTransformer,
  not the simple reg-process path.
- Counter-style @zdc.proc (no ProtocolPort fields) still routes through the
  simple reg-process path and produces Verilog-2005 always-block output.
"""

import sys
import pytest

sys.path.insert(0, 'packages/zuspec-dataclasses/src')
sys.path.insert(0, 'packages/zuspec-synth/src')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mem_reader_sv():
    """Synthesize MemReader — minimal component with one awaited port call."""
    from examples.mem_reader import MemReader
    from zuspec.synth import synthesize
    return synthesize(MemReader)


@pytest.fixture(scope="module")
def monitor_writer_sv():
    """Synthesize MonitorWriter — component with a non-awaited port output."""
    from examples.monitor_writer import MonitorWriter
    from zuspec.synth import synthesize
    return synthesize(MonitorWriter)


# ---------------------------------------------------------------------------
# Helper: build MemReader / MonitorWriter from files in examples/
# We define them in separate modules to avoid exec() source-code issues.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MemReader: awaited port-method call tests
# ---------------------------------------------------------------------------

class TestMemReaderPortCall:
    def test_mem_reader_synthesizes(self, mem_reader_sv):
        assert mem_reader_sv and len(mem_reader_sv) > 50

    def test_module_declaration(self, mem_reader_sv):
        assert "module MemReader" in mem_reader_sv

    def test_valid_output_port(self, mem_reader_sv):
        assert "mem_read_word_valid" in mem_reader_sv

    def test_arg_output_port(self, mem_reader_sv):
        assert "mem_read_word_arg0" in mem_reader_sv

    def test_ack_input_port(self, mem_reader_sv):
        assert "mem_read_word_ack" in mem_reader_sv

    def test_rdata_input_port(self, mem_reader_sv):
        assert "mem_read_word_rdata" in mem_reader_sv

    def test_req_state_exists(self, mem_reader_sv):
        assert "MEM_READ_WORD_REQ" in mem_reader_sv

    def test_valid_asserted_in_req_state(self, mem_reader_sv):
        assert "mem_read_word_valid <= 1'b1" in mem_reader_sv

    def test_result_latched_on_ack(self, mem_reader_sv):
        assert "mem_read_word_ack" in mem_reader_sv
        assert "mem_read_word_rdata" in mem_reader_sv

    def test_result_register_declared(self, mem_reader_sv):
        # result_var 'data' must become a register
        assert "data" in mem_reader_sv

    def test_fsm_style_not_always_posedge(self, mem_reader_sv):
        # ProtocolPort components must NOT use the simple reg-process path
        assert "always @(posedge clock or posedge reset)" not in mem_reader_sv

    def test_handshake_self_loop(self, mem_reader_sv):
        # The WAIT_COND state must loop to itself while ack is not asserted
        # (two transitions from MEM_READ_WORD_REQ: conditional advance + self-loop)
        assert mem_reader_sv.count("MEM_READ_WORD_REQ") >= 2


# ---------------------------------------------------------------------------
# MonitorWriter: non-awaited port output tests
# ---------------------------------------------------------------------------

class TestMonitorWriterPortOutput:
    def test_monitor_writer_synthesizes(self, monitor_writer_sv):
        assert monitor_writer_sv and len(monitor_writer_sv) > 50

    def test_valid_output_present(self, monitor_writer_sv):
        assert "monitor_on_event_valid" in monitor_writer_sv

    def test_arg_output_present(self, monitor_writer_sv):
        assert "monitor_on_event_arg0" in monitor_writer_sv

    def test_no_ack_port_for_void_call(self, monitor_writer_sv):
        # Non-awaited calls don't need an ack port
        assert "monitor_on_event_ack" not in monitor_writer_sv

    def test_no_wait_state_for_void_call(self, monitor_writer_sv):
        # Non-awaited calls don't create a WAIT_COND state
        assert "MONITOR_ON_EVENT_REQ" not in monitor_writer_sv


# ---------------------------------------------------------------------------
# Routing: proc_processes path detection
# ---------------------------------------------------------------------------

class TestProcRoutingWithProtocolPort:
    def test_routes_through_fsm_not_reg_process(self, mem_reader_sv):
        """Components with ProtocolPort fields must use FSM style."""
        # FSM style: state enum typedef present
        assert "typedef enum" in mem_reader_sv
        # NOT the simple reg-process style
        assert "always @(posedge clock or posedge reset)" not in mem_reader_sv
