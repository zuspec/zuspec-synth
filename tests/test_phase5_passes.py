"""Tests for Phase 5 synthesis IR nodes and lowering passes.

Covers:
  - protocol_ir.py  (data-class construction + helpers)
  - QueueLowerPass  (populates SynthIR.queue_nodes)
  - IfProtocolLowerPass  (populates SynthIR.protocol_ports)
  - SpawnLowerPass  (populates SynthIR.spawn_nodes)
  - SelectLowerPass  (populates SynthIR.select_nodes)
  - CompletionAnalysisPass  (populates SynthIR.completion_nodes)
"""
from __future__ import annotations

import pytest

from zuspec.synth.ir.protocol_ir import (
    CompletionIR,
    IfProtocolPortIR,
    IfProtocolScenario,
    ProtocolField,
    QueueIR,
    SelectBranchIR,
    SelectIR,
    SpawnIR,
)
from zuspec.synth.ir.synth_ir import SynthIR


# ============================================================
# Helpers
# ============================================================

def _make_ir(component=None) -> SynthIR:
    return SynthIR(component=component)


# ============================================================
# protocol_ir.py — data class tests
# ============================================================

class TestQueueIR:
    def test_defaults(self):
        q = QueueIR(name="req_q")
        assert q.name == "req_q"
        assert q.elem_width == 32
        assert q.depth == 16

    def test_addr_bits(self):
        q = QueueIR(name="q", depth=8)
        assert q.addr_bits == 3

    def test_addr_bits_power_of_two_plus_one(self):
        q = QueueIR(name="q", depth=9)
        assert q.addr_bits == 4

    def test_count_bits(self):
        q = QueueIR(name="q", depth=16)
        assert q.count_bits == 5  # need to hold 0..16 (17 values)

    def test_depth_1_addr_bits(self):
        q = QueueIR(name="q", depth=1)
        assert q.addr_bits == 1


class TestIfProtocolPortIR:
    def _make_port(self, scenario, **kw) -> IfProtocolPortIR:
        return IfProtocolPortIR(name="mem", scenario=scenario, **kw)

    def test_scenario_a_signal_name(self):
        port = self._make_port(IfProtocolScenario.A)
        assert port.signal_name("req_valid") == "mem_req_valid"

    def test_scenario_b_sv_ports_contains_ready(self):
        port = IfProtocolPortIR(
            name="mem",
            scenario=IfProtocolScenario.B,
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        )
        ports = port.all_sv_ports()
        names = [p[2] for p in ports]
        assert "mem_req_valid" in names
        assert "mem_req_ready" in names
        assert "mem_resp_valid" in names
        assert "mem_req_addr" in names
        assert "mem_resp_data" in names

    def test_scenario_a_no_ready_or_valid_when_always_ready(self):
        """Scenario A with req_always_ready removes req_ready and resp_valid."""
        class _Props:
            req_always_ready = True
            resp_always_valid = True
            resp_has_backpressure = False
            max_outstanding = 1
            in_order = True

        port = IfProtocolPortIR(
            name="fast",
            scenario=IfProtocolScenario.A,
            properties=_Props(),
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        )
        ports = port.all_sv_ports()
        names = [p[2] for p in ports]
        assert "fast_req_ready" not in names
        assert "fast_resp_valid" not in names
        assert "fast_req_addr" in names

    def test_scenario_d_has_id_fields(self):
        class _Props:
            req_always_ready = False
            resp_always_valid = False
            resp_has_backpressure = False
            max_outstanding = 4
            in_order = False

        port = IfProtocolPortIR(
            name="ooo",
            scenario=IfProtocolScenario.D,
            properties=_Props(),
            id_bits=2,
        )
        ports = port.all_sv_ports()
        names = [p[2] for p in ports]
        assert "ooo_req_id" in names
        assert "ooo_resp_id" in names

    def test_export_flips_direction(self):
        port_out = IfProtocolPortIR(
            name="p",
            is_export=False,
            scenario=IfProtocolScenario.B,
        )
        port_exp = IfProtocolPortIR(
            name="p",
            is_export=True,
            scenario=IfProtocolScenario.B,
        )
        ports_out = {p[2]: p[0] for p in port_out.all_sv_ports()}
        ports_exp = {p[2]: p[0] for p in port_exp.all_sv_ports()}
        for sig_name in ports_out:
            if sig_name in ports_exp:
                assert ports_out[sig_name] != ports_exp[sig_name], (
                    f"Signal {sig_name!r} should flip direction for export"
                )


class TestSpawnIR:
    def test_defaults(self):
        s = SpawnIR(name="main_do_req")
        assert s.n_slots == 1
        assert s.slot_fields == []
        assert s.result_fields == []
        assert s.protocol_port is None

    def test_custom_fields(self):
        s = SpawnIR(
            name="main_do_req",
            n_slots=4,
            slot_fields=[ProtocolField("addr", 32)],
            protocol_port="mem",
        )
        assert s.n_slots == 4
        assert s.protocol_port == "mem"


class TestSelectIR:
    def test_defaults(self):
        sel = SelectIR(name="main_select_0")
        assert sel.branches == []
        assert sel.round_robin is False

    def test_branches(self):
        sel = SelectIR(
            name="main_select_0",
            branches=[
                SelectBranchIR("req_q", 0),
                SelectBranchIR("ack_q", 1),
            ],
        )
        assert len(sel.branches) == 2
        assert sel.branches[0].queue_name == "req_q"


class TestCompletionIR:
    def test_defaults(self):
        c = CompletionIR(name="done")
        assert c.elem_width == 32
        assert c.queue_path == []

    def test_custom(self):
        c = CompletionIR(name="result", elem_width=64, queue_path=["req_q"])
        assert c.elem_width == 64
        assert "req_q" in c.queue_path


# ============================================================
# SynthIR — new fields are present and default to empty lists
# ============================================================

class TestSynthIRNewFields:
    def test_all_new_fields_default_empty(self):
        ir = _make_ir()
        assert ir.protocol_ports == []
        assert ir.queue_nodes == []
        assert ir.spawn_nodes == []
        assert ir.select_nodes == []
        assert ir.completion_nodes == []

    def test_can_append_nodes(self):
        ir = _make_ir()
        ir.queue_nodes.append(QueueIR(name="q"))
        ir.protocol_ports.append(IfProtocolPortIR(name="mem", scenario=IfProtocolScenario.B))
        assert len(ir.queue_nodes) == 1
        assert len(ir.protocol_ports) == 1


# ============================================================
# QueueLowerPass — runs on minimal SynthIR with no model context
# ============================================================

class TestQueueLowerPass:
    def test_no_context_noop(self):
        from zuspec.synth.passes.queue_lower import QueueLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = QueueLowerPass(SynthConfig()).run(ir)
        assert result.queue_nodes == []

    def test_returns_ir(self):
        from zuspec.synth.passes.queue_lower import QueueLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = QueueLowerPass(SynthConfig()).run(ir)
        assert result is ir


# ============================================================
# IfProtocolLowerPass — no model context noop
# ============================================================

class TestIfProtocolLowerPass:
    def test_no_context_noop(self):
        from zuspec.synth.passes.if_protocol_lower import IfProtocolLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = IfProtocolLowerPass(SynthConfig()).run(ir)
        assert result.protocol_ports == []

    def test_returns_ir(self):
        from zuspec.synth.passes.if_protocol_lower import IfProtocolLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = IfProtocolLowerPass(SynthConfig()).run(ir)
        assert result is ir


# ============================================================
# SpawnLowerPass
# ============================================================

class TestSpawnLowerPass:
    def test_no_context_noop(self):
        from zuspec.synth.passes.spawn_lower import SpawnLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = SpawnLowerPass(SynthConfig()).run(ir)
        assert result.spawn_nodes == []

    def test_returns_ir(self):
        from zuspec.synth.passes.spawn_lower import SpawnLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = SpawnLowerPass(SynthConfig()).run(ir)
        assert result is ir


# ============================================================
# SelectLowerPass
# ============================================================

class TestSelectLowerPass:
    def test_no_context_noop(self):
        from zuspec.synth.passes.select_lower import SelectLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = SelectLowerPass(SynthConfig()).run(ir)
        assert result.select_nodes == []

    def test_returns_ir(self):
        from zuspec.synth.passes.select_lower import SelectLowerPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = SelectLowerPass(SynthConfig()).run(ir)
        assert result is ir


# ============================================================
# CompletionAnalysisPass
# ============================================================

class TestCompletionAnalysisPass:
    def test_no_context_noop(self):
        from zuspec.synth.passes.completion_analysis import CompletionAnalysisPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = CompletionAnalysisPass(SynthConfig()).run(ir)
        assert result.completion_nodes == []

    def test_returns_ir(self):
        from zuspec.synth.passes.completion_analysis import CompletionAnalysisPass
        from zuspec.synth.ir.synth_ir import SynthConfig
        ir = _make_ir()
        result = CompletionAnalysisPass(SynthConfig()).run(ir)
        assert result is ir


# ============================================================
# Scenario selection helper (via IfProtocolLowerPass._select_scenario)
# ============================================================

class TestScenarioSelection:
    def _sel(self, **kwargs):
        from zuspec.synth.passes.if_protocol_lower import _select_scenario
        class _P:
            pass
        p = _P()
        for k, v in kwargs.items():
            setattr(p, k, v)
        return _select_scenario(p)

    def test_none_props_gives_b(self):
        from zuspec.synth.passes.if_protocol_lower import _select_scenario
        assert _select_scenario(None) == IfProtocolScenario.B

    def test_scenario_a_when_both_ready_valid(self):
        s = self._sel(max_outstanding=1, req_always_ready=True,
                      resp_always_valid=True, in_order=True)
        assert s == IfProtocolScenario.A

    def test_scenario_b_when_mo1_no_ready(self):
        s = self._sel(max_outstanding=1, req_always_ready=False,
                      resp_always_valid=False, in_order=True)
        assert s == IfProtocolScenario.B

    def test_scenario_c_when_mo4_in_order(self):
        s = self._sel(max_outstanding=4, req_always_ready=False,
                      resp_always_valid=False, in_order=True)
        assert s == IfProtocolScenario.C

    def test_scenario_d_when_mo4_out_of_order(self):
        s = self._sel(max_outstanding=4, req_always_ready=False,
                      resp_always_valid=False, in_order=False)
        assert s == IfProtocolScenario.D
