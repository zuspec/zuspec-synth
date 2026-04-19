"""Tests for Phase 6: SV code generation for IfProtocol, Queue, Select.

Covers:
  - sprtl/protocol_sv.py  (template generators)
  - passes/protocol_sv_emit.py  (ProtocolSVEmitPass)
"""
from __future__ import annotations

import pytest

from zuspec.synth.ir.protocol_ir import (
    IfProtocolPortIR,
    IfProtocolScenario,
    ProtocolField,
    QueueIR,
    SelectBranchIR,
    SelectIR,
    SpawnIR,
)
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR


# ============================================================
# Helpers
# ============================================================

def _make_ir():
    return SynthIR(component=None)


# ============================================================
# generate_ifprotocol_port_decls
# ============================================================

class TestGenerateIfProtocolPortDecls:
    def test_scenario_b_contains_req_valid(self):
        from zuspec.synth.sprtl.protocol_sv import generate_ifprotocol_port_decls
        port = IfProtocolPortIR(
            name="mem",
            scenario=IfProtocolScenario.B,
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        )
        sv = generate_ifprotocol_port_decls(port)
        assert "mem_req_valid" in sv
        assert "mem_req_ready" in sv
        assert "mem_resp_valid" in sv
        assert "mem_req_addr" in sv
        assert "mem_resp_data" in sv

    def test_scenario_b_directions(self):
        from zuspec.synth.sprtl.protocol_sv import generate_ifprotocol_port_decls
        port = IfProtocolPortIR(
            name="mem",
            scenario=IfProtocolScenario.B,
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        )
        sv = generate_ifprotocol_port_decls(port)
        # req_valid is output from initiator
        assert "output logic" in sv
        assert "input logic" in sv

    def test_width_1_no_brackets(self):
        from zuspec.synth.sprtl.protocol_sv import generate_ifprotocol_port_decls
        port = IfProtocolPortIR(name="p", scenario=IfProtocolScenario.B)
        sv = generate_ifprotocol_port_decls(port)
        # Width-1 signals should not have [0:0] bracket
        lines = [l for l in sv.splitlines() if "req_valid" in l]
        assert len(lines) == 1
        assert "[" not in lines[0]

    def test_wide_field_has_brackets(self):
        from zuspec.synth.sprtl.protocol_sv import generate_ifprotocol_port_decls
        port = IfProtocolPortIR(
            name="bus",
            scenario=IfProtocolScenario.B,
            req_fields=[ProtocolField("addr", 64)],
        )
        sv = generate_ifprotocol_port_decls(port)
        assert "[63:0]" in sv

    def test_export_flips_req_valid_to_input(self):
        from zuspec.synth.sprtl.protocol_sv import generate_ifprotocol_port_decls
        port = IfProtocolPortIR(
            name="mem",
            is_export=True,
            scenario=IfProtocolScenario.B,
        )
        sv = generate_ifprotocol_port_decls(port)
        lines = {l.split()[-1].rstrip(","): " ".join(l.split()[:2])
                 for l in sv.splitlines() if l.strip()}
        # For an export req_valid comes from outside → input
        assert "input" in lines.get("mem_req_valid", "")

    def test_scenario_d_includes_id_ports(self):
        from zuspec.synth.sprtl.protocol_sv import generate_ifprotocol_port_decls

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
        sv = generate_ifprotocol_port_decls(port)
        assert "ooo_req_id" in sv
        assert "ooo_resp_id" in sv


# ============================================================
# generate_fifo_sv
# ============================================================

class TestGenerateFifoSV:
    def test_module_name(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        q = QueueIR(name="req_q", elem_width=32, depth=16)
        sv = generate_fifo_sv(q, module_prefix="dut_")
        assert "module dut_req_q_fifo" in sv

    def test_default_module_name(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        q = QueueIR(name="myqueue", depth=8)
        sv = generate_fifo_sv(q)
        assert "module myqueue_fifo" in sv

    def test_has_clk_rst(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        sv = generate_fifo_sv(QueueIR(name="q"))
        assert "clk" in sv
        assert "rst" in sv

    def test_has_wr_rd_ports(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        sv = generate_fifo_sv(QueueIR(name="q"))
        for sig in ("wr_en", "wr_data", "rd_en", "rd_data", "full", "empty", "count"):
            assert sig in sv, f"Missing port: {sig}"

    def test_correct_data_width(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        sv = generate_fifo_sv(QueueIR(name="q", elem_width=64))
        assert "[63:0]" in sv

    def test_correct_depth(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        sv = generate_fifo_sv(QueueIR(name="q", depth=32))
        assert "0:31" in sv

    def test_endmodule_present(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        sv = generate_fifo_sv(QueueIR(name="q"))
        assert "endmodule" in sv

    def test_always_ff_present(self):
        from zuspec.synth.sprtl.protocol_sv import generate_fifo_sv
        sv = generate_fifo_sv(QueueIR(name="q"))
        assert "always_ff" in sv


# ============================================================
# generate_priority_arbiter_sv
# ============================================================

class TestGeneratePriorityArbiterSV:
    def _sel(self, n: int, rr: bool = False) -> SelectIR:
        return SelectIR(
            name="main_select_0",
            branches=[SelectBranchIR(f"q{i}", i) for i in range(n)],
            round_robin=rr,
        )

    def test_empty_sel_is_comment(self):
        from zuspec.synth.sprtl.protocol_sv import generate_priority_arbiter_sv
        sel = SelectIR(name="s", branches=[])
        sv = generate_priority_arbiter_sv(sel)
        assert "module" not in sv
        assert "//" in sv

    def test_module_name(self):
        from zuspec.synth.sprtl.protocol_sv import generate_priority_arbiter_sv
        sv = generate_priority_arbiter_sv(self._sel(3))
        assert "module main_select_0_arb" in sv

    def test_req_gnt_ports_present(self):
        from zuspec.synth.sprtl.protocol_sv import generate_priority_arbiter_sv
        sv = generate_priority_arbiter_sv(self._sel(2))
        assert "req_i" in sv
        assert "gnt_o" in sv
        assert "sel_o" in sv
        assert "gnt_valid_o" in sv

    def test_endmodule(self):
        from zuspec.synth.sprtl.protocol_sv import generate_priority_arbiter_sv
        sv = generate_priority_arbiter_sv(self._sel(2))
        assert "endmodule" in sv

    def test_single_branch(self):
        from zuspec.synth.sprtl.protocol_sv import generate_priority_arbiter_sv
        sv = generate_priority_arbiter_sv(self._sel(1))
        assert "module" in sv
        assert "endmodule" in sv


# ============================================================
# generate_rr_arbiter_sv
# ============================================================

class TestGenerateRRArbiterSV:
    def _sel(self, n: int) -> SelectIR:
        return SelectIR(
            name="rr_sel",
            branches=[SelectBranchIR(f"q{i}", i) for i in range(n)],
            round_robin=True,
        )

    def test_empty_sel_is_comment(self):
        from zuspec.synth.sprtl.protocol_sv import generate_rr_arbiter_sv
        sel = SelectIR(name="s", branches=[])
        sv = generate_rr_arbiter_sv(sel)
        assert "module" not in sv

    def test_module_name(self):
        from zuspec.synth.sprtl.protocol_sv import generate_rr_arbiter_sv
        sv = generate_rr_arbiter_sv(self._sel(4))
        assert "module rr_sel_rr_arb" in sv

    def test_clk_rst_present(self):
        from zuspec.synth.sprtl.protocol_sv import generate_rr_arbiter_sv
        sv = generate_rr_arbiter_sv(self._sel(2))
        assert "clk" in sv
        assert "rst" in sv

    def test_mask_r_register(self):
        from zuspec.synth.sprtl.protocol_sv import generate_rr_arbiter_sv
        sv = generate_rr_arbiter_sv(self._sel(4))
        assert "mask_r" in sv

    def test_endmodule(self):
        from zuspec.synth.sprtl.protocol_sv import generate_rr_arbiter_sv
        sv = generate_rr_arbiter_sv(self._sel(3))
        assert "endmodule" in sv


# ============================================================
# ProtocolSVEmitPass
# ============================================================

class TestProtocolSVEmitPass:
    def _make_ir_with_queue(self, name="req_q", width=32, depth=8) -> SynthIR:
        ir = _make_ir()
        ir.queue_nodes.append(QueueIR(name=name, elem_width=width, depth=depth))
        return ir

    def _make_ir_with_port(self) -> SynthIR:
        ir = _make_ir()
        ir.protocol_ports.append(IfProtocolPortIR(
            name="mem",
            scenario=IfProtocolScenario.B,
            req_fields=[ProtocolField("addr", 32)],
            resp_fields=[ProtocolField("data", 32, is_response=True)],
        ))
        return ir

    def _make_ir_with_select(self, rr: bool = False) -> SynthIR:
        ir = _make_ir()
        ir.select_nodes.append(SelectIR(
            name="main_select_0",
            branches=[SelectBranchIR("q0", 0), SelectBranchIR("q1", 1)],
            round_robin=rr,
        ))
        return ir

    def test_empty_ir_noop(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = _make_ir()
        result = ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert result is ir
        assert result.lowered_sv == {}

    def test_queue_emits_fifo_module(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = self._make_ir_with_queue()
        ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert "sv/module/req_q_fifo" in ir.lowered_sv
        sv = "".join(ir.lowered_sv["sv/module/req_q_fifo"])
        assert "module req_q_fifo" in sv

    def test_port_emits_port_decls(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = self._make_ir_with_port()
        ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert "sv/port/mem" in ir.lowered_sv
        sv = "".join(ir.lowered_sv["sv/port/mem"])
        assert "mem_req_valid" in sv

    def test_port_emits_instantiation(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = self._make_ir_with_port()
        ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert "sv/instantiation/mem" in ir.lowered_sv
        sv = "".join(ir.lowered_sv["sv/instantiation/mem"])
        assert ".mem_req_valid(mem_req_valid)" in sv

    def test_select_priority_emits_arb(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = self._make_ir_with_select(rr=False)
        ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert "sv/module/main_select_0_arb" in ir.lowered_sv
        sv = "".join(ir.lowered_sv["sv/module/main_select_0_arb"])
        assert "module main_select_0_arb" in sv

    def test_select_rr_emits_rr_arb(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = self._make_ir_with_select(rr=True)
        ProtocolSVEmitPass(SynthConfig()).run(ir)
        key = "sv/module/main_select_0_arb"
        assert key in ir.lowered_sv
        sv = "".join(ir.lowered_sv[key])
        assert "mask_r" in sv  # round-robin specific

    def test_module_prefix_applied_to_fifo(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = self._make_ir_with_queue()
        ProtocolSVEmitPass(SynthConfig(), module_prefix="top_").run(ir)
        sv = "".join(ir.lowered_sv["sv/module/req_q_fifo"])
        assert "module top_req_q_fifo" in sv

    def test_returns_ir(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = _make_ir()
        result = ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert result is ir

    def test_multiple_queues(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = _make_ir()
        ir.queue_nodes.append(QueueIR(name="req_q", depth=8))
        ir.queue_nodes.append(QueueIR(name="resp_q", depth=4))
        ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert "sv/module/req_q_fifo" in ir.lowered_sv
        assert "sv/module/resp_q_fifo" in ir.lowered_sv

    def test_multiple_ports(self):
        from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass
        ir = _make_ir()
        ir.protocol_ports.append(IfProtocolPortIR(name="mem", scenario=IfProtocolScenario.B))
        ir.protocol_ports.append(IfProtocolPortIR(name="io", scenario=IfProtocolScenario.A))
        ProtocolSVEmitPass(SynthConfig()).run(ir)
        assert "sv/port/mem" in ir.lowered_sv
        assert "sv/port/io" in ir.lowered_sv
