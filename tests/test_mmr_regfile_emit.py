"""Tests for MmrRegFileRtlEmitter / synthesize_regfile.

Covers:
- Module name derivation (CamelCase → snake_case)
- APB4 port list
- hwif_in / hwif_out struct contents
- Per-field RTL patterns: simple RW, RO, WO, stickybit, singlepulse, HW.W, woclr
- Interrupt output for registers with stickybit fields
- Standalone synthesize_regfile() entry point
- Pass integration via MmrRegFileEmitPass
"""
import sys
import pytest

sys.path.insert(0, 'packages/zuspec-dataclasses/src')
sys.path.insert(0, 'packages/zuspec-synth/src')

import zuspec.dataclasses as zdc
from zuspec.dataclasses.mmr.base import RegisterFile


# ---------------------------------------------------------------------------
# Shared register file fixtures
# ---------------------------------------------------------------------------

@zdc.regfile
class SimpleRegs(RegisterFile):
    """A minimal register file with a single RW field."""

    @zdc.reg(offset=0x00, width=32)
    class CTRL:
        EN: zdc.u1 = zdc.reg_field(default=0)


@zdc.regfile
class DMARegs(RegisterFile):
    """Copy-engine register file matching the design doc example."""

    @zdc.reg(offset=0x00, width=32)
    class CTRL:
        START: zdc.u1 = zdc.FieldAttr.Pulse        # singlepulse
        ABORT: zdc.u1 = zdc.FieldAttr.Pulse

    @zdc.reg(offset=0x04, width=32)
    class STATUS:
        BUSY:  zdc.u1 = zdc.reg_field(sw=zdc.SW.RO, hw=zdc.HW.W, default=0)
        DONE:  zdc.u1 = zdc.FieldAttr.StickyBit
        ERROR: zdc.u1 = zdc.FieldAttr.StickyBit

    @zdc.reg(offset=0x08, width=32)
    class IRQ_EN:
        DONE_EN:  zdc.u1 = zdc.reg_field(default=0)
        ERROR_EN: zdc.u1 = zdc.reg_field(default=0)

    @zdc.reg(offset=0x0C, width=32)
    class SRC_ADDR:
        ADDR: zdc.u32 = zdc.reg_field(default=0)

    @zdc.reg(offset=0x10, width=32)
    class DST_ADDR:
        ADDR: zdc.u32 = zdc.reg_field(default=0)

    @zdc.reg(offset=0x14, width=32)
    class LENGTH:
        LEN: zdc.u32 = zdc.reg_field(default=0)


@zdc.regfile
class MixedAccessRegs(RegisterFile):
    """Register file exercising RO, WO, HW.W, and woclr fields."""

    @zdc.reg(offset=0x00, width=32)
    class STATUS:
        RO_FLAG:  zdc.u1 = zdc.reg_field(sw=zdc.SW.RO, hw=zdc.HW.W, default=0)
        WO_CMD:   zdc.u1 = zdc.reg_field(sw=zdc.SW.WO, hw=zdc.HW.R, default=0)
        CLR_IRQR: zdc.u1 = zdc.reg_field(
            sw=zdc.SW.RW, hw=zdc.HW.W,
            onwrite='woclr', stickybit=True, default=0
        )


@zdc.regfile
class WeRegs(RegisterFile):
    """HW write-enable qualified field."""

    @zdc.reg(offset=0x00, width=32)
    class DATA:
        VAL: zdc.u8 = zdc.reg_field(hw=zdc.HW.W, we=True, default=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sv_of(cls, **kwargs):
    from zuspec.synth.passes.mmr_regfile_emit import synthesize_regfile
    return synthesize_regfile(cls, **kwargs)


def pkg_of(cls, **kwargs):
    from zuspec.synth.passes.mmr_regfile_emit import MmrRegFileRtlEmitter
    return MmrRegFileRtlEmitter(cls, **kwargs).emit_package()


# ===========================================================================
# 1. Module name
# ===========================================================================

class TestModuleName:
    def test_simple_class_lowercased(self):
        sv = sv_of(SimpleRegs)
        assert "module simple_regs" in sv

    def test_dma_camel_to_snake(self):
        sv = sv_of(DMARegs)
        assert "module dma_regs" in sv

    def test_override_module_name(self):
        sv = sv_of(SimpleRegs, module_name="my_ctrl")
        assert "module my_ctrl" in sv

    def test_endmodule_present(self):
        sv = sv_of(SimpleRegs)
        assert "endmodule" in sv


# ===========================================================================
# 2. APB4 port list
# ===========================================================================

class TestApb4Ports:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(SimpleRegs)

    def test_clk_rst(self, sv):
        assert "input  logic        clk" in sv
        assert "input  logic        rst" in sv

    def test_apb4_inputs(self, sv):
        assert "input  logic        psel" in sv
        assert "input  logic        penable" in sv
        assert "input  logic        pwrite" in sv
        assert "input  logic" in sv and "paddr" in sv
        assert "input  logic" in sv and "pwdata" in sv

    def test_apb4_outputs(self, sv):
        assert "output logic" in sv and "prdata" in sv
        assert "output logic        pready" in sv
        assert "output logic        pslverr" in sv

    def test_hwif_ports(self, sv):
        assert "simple_regs__in_t  hwif_in" in sv
        assert "simple_regs__out_t hwif_out" in sv

    def test_zero_wait_state(self, sv):
        assert "assign pready  = 1'b1" in sv
        assert "assign pslverr = 1'b0" in sv


# ===========================================================================
# 3. Field storage and combo wires
# ===========================================================================

class TestFieldStorage:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(SimpleRegs)

    def test_storage_wire_declared(self, sv):
        assert "field_storage_CTRL_EN" in sv

    def test_combo_next_wire(self, sv):
        assert "field_combo_next_CTRL_EN" in sv

    def test_combo_load_wire(self, sv):
        assert "field_combo_load_CTRL_EN" in sv


# ===========================================================================
# 4. Address strobe decoder
# ===========================================================================

class TestAddressDecoder:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(DMARegs)

    def test_strobe_signals(self, sv):
        assert "decoded_reg_strb_CTRL" in sv
        assert "decoded_reg_strb_STATUS" in sv

    def test_case_addresses(self, sv):
        assert "8'h00: decoded_reg_strb_CTRL" in sv
        assert "8'h04: decoded_reg_strb_STATUS" in sv
        assert "8'h08: decoded_reg_strb_IRQ_EN" in sv

    def test_apb_qualifier(self, sv):
        assert "if (cpuif_req)" in sv


# ===========================================================================
# 5. Simple RW field always_comb / always_ff
# ===========================================================================

class TestSimpleRwField:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(SimpleRegs)

    def test_always_comb_block(self, sv):
        assert "always_comb begin : CTRL_EN_combo" in sv

    def test_sw_write_path(self, sv):
        # SW writes the bus data into the field
        assert "decoded_reg_strb_CTRL && cpuif_req_is_wr" in sv

    def test_always_ff_block(self, sv):
        assert "always_ff @(posedge clk) begin : CTRL_EN_ff" in sv

    def test_sync_reset(self, sv):
        assert "if (rst)" in sv
        assert "field_storage_CTRL_EN <= 1'b0" in sv

    def test_load_next_gate(self, sv):
        assert "else if (field_combo_load_CTRL_EN)" in sv
        assert "field_storage_CTRL_EN <= field_combo_next_CTRL_EN" in sv


# ===========================================================================
# 6. Stickybit field
# ===========================================================================

class TestStickybitField:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(DMARegs)

    def test_posedge_pipeline_register(self, sv):
        assert "field_q_STATUS_DONE" in sv

    def test_posedge_detection(self, sv):
        assert "(~field_q_STATUS_DONE) & hwif_in.STATUS.DONE_hwset" in sv

    def test_pipeline_reg_update_in_ff(self, sv):
        # Pipeline register updated in always_ff
        assert "field_q_STATUS_DONE <= hwif_in.STATUS.DONE_hwset" in sv

    def test_interrupt_output(self, sv):
        assert "assign hwif_out.STATUS.intr" in sv
        assert "field_storage_STATUS_DONE" in sv
        assert "field_storage_STATUS_ERROR" in sv


# ===========================================================================
# 7. Singlepulse field
# ===========================================================================

class TestSinglepulseField:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(DMARegs)

    def test_auto_clear_logic(self, sv):
        # When field is non-zero and SW not writing, clear it
        assert "singlepulse: auto-clear when not being written" in sv

    def test_auto_clear_condition(self, sv):
        assert "field_storage_CTRL_START != 1'b0" in sv

    def test_swmod_output(self, sv):
        # singlepulse implies swmod output
        assert "hwif_out.CTRL.START_swmod" in sv


# ===========================================================================
# 8. RO field
# ===========================================================================

class TestRoField:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(MixedAccessRegs)

    def test_hw_w_hwif_in_member(self, sv):
        # RO_FLAG has hw=HW.W so needs hwif_in.STATUS.RO_FLAG_next
        assert "hwif_in.STATUS.RO_FLAG_next" in sv

    def test_ro_field_no_sw_write(self, sv):
        # SW cannot write a RO field — the write strobe section should not assign it
        # The RO field comb block should have no SW write section
        assert "CTRL_EN_combo" in sv_of(SimpleRegs)  # sanity
        sv_block = sv.split("STATUS_RO_FLAG_combo")[1].split("end")[0]
        # There should be no cpuif_req_is_wr path assigning RO_FLAG
        assert "cpuif_req_is_wr" not in sv_block or "RO_FLAG" not in sv_block.split("cpuif_req_is_wr")[0]

    def test_ro_field_readable(self, sv):
        # RO fields must appear in the read mux (prdata)
        assert "field_storage_STATUS_RO_FLAG" in sv
        # prdata read mux should include it
        sv_after_read_mux = sv.split("Read mux")[1]
        assert "field_storage_STATUS_RO_FLAG" in sv_after_read_mux


# ===========================================================================
# 9. woclr field
# ===========================================================================

class TestWoclrField:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(MixedAccessRegs)

    def test_woclr_pattern(self, sv):
        # woclr: next = storage & ~write_data
        assert "field_storage_STATUS_CLR_IRQR & ~cpuif_wr_data" in sv


# ===========================================================================
# 10. HW.W with write-enable
# ===========================================================================

class TestHwWriteEnable:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(WeRegs)

    def test_we_signal_in_hwif_in(self):
        pkg = pkg_of(WeRegs)
        assert "VAL_we" in pkg

    def test_we_gates_hw_write(self, sv):
        assert "hwif_in.DATA.VAL_we" in sv


# ===========================================================================
# 11. hwif package struct contents
# ===========================================================================

class TestHwifPackage:
    @pytest.fixture(scope="class")
    def pkg(self):
        return pkg_of(DMARegs)

    def test_package_declaration(self, pkg):
        assert "package dma_regs_pkg" in pkg
        assert "endpackage" in pkg

    def test_hwif_in_struct(self, pkg):
        assert "dma_regs__in_t" in pkg

    def test_hwif_out_struct(self, pkg):
        assert "dma_regs__out_t" in pkg

    def test_busy_next_in_hwif_in(self, pkg):
        # STATUS.BUSY has hw=HW.W
        assert "BUSY_next" in pkg

    def test_done_hwset_in_hwif_in(self, pkg):
        # STATUS.DONE is stickybit → hwset input
        assert "DONE_hwset" in pkg

    def test_start_swmod_in_hwif_out(self, pkg):
        # CTRL.START is singlepulse → swmod output
        assert "START_swmod" in pkg

    def test_intr_in_hwif_out(self, pkg):
        # STATUS has stickybit fields → intr output
        assert "intr" in pkg

    def test_no_hwif_in_for_ctrl(self, pkg):
        # CTRL.START is singlepulse with sw=RW, hw=R — no hwif_in member
        assert "START_next" not in pkg


# ===========================================================================
# 12. Read mux
# ===========================================================================

class TestReadMux:
    @pytest.fixture(scope="class")
    def sv(self):
        return sv_of(DMARegs)

    def test_prdata_initialized_zero(self, sv):
        assert "prdata = " in sv and "0" in sv

    def test_read_on_no_write(self, sv):
        assert "cpuif_req && !cpuif_req_is_wr" in sv

    def test_src_addr_readable(self, sv):
        assert "field_storage_SRC_ADDR_ADDR" in sv

    def test_wo_field_not_in_read_mux(self):
        sv = sv_of(MixedAccessRegs)
        read_mux_block = sv.split("Read mux")[1]
        assert "WO_CMD" not in read_mux_block


# ===========================================================================
# 13. Pass integration
# ===========================================================================

class TestMmrRegFileEmitPass:
    def test_pass_populates_lowered_sv(self):
        import zuspec.dataclasses as zdc
        from zuspec.dataclasses.mmr.base import RegisterFile
        from zuspec.synth.elab.elaborator import Elaborator
        from zuspec.synth.elab.elab_ir import ComponentSynthMeta
        from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
        from zuspec.synth.passes import MmrRegFileEmitPass

        @zdc.regfile
        class Regs(RegisterFile):
            @zdc.reg(offset=0x00)
            class CTRL:
                EN: zdc.u1 = zdc.reg_field(default=0)

        @zdc.dataclass
        class MyComp(zdc.Component):
            regs: Regs = zdc.inst()

        ir = SynthIR(component=MyComp, config=SynthConfig())
        from zuspec.synth.passes.elaborate import ElaboratePass
        ElaboratePass(MyComp, {}).run(ir)

        pass_ = MmrRegFileEmitPass(SynthConfig())
        pass_.run(ir)

        assert "sv/regfile/regs" in ir.lowered_sv
        sv = ir.lowered_sv["sv/regfile/regs"]
        assert "module regs" in sv

    def test_results_dict_populated(self):
        from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
        from zuspec.synth.passes import MmrRegFileEmitPass
        from zuspec.synth.passes.elaborate import ElaboratePass

        @zdc.regfile
        class Regs2(RegisterFile):
            @zdc.reg(offset=0x00)
            class STATUS:
                OK: zdc.u1 = zdc.reg_field(sw=zdc.SW.RO, hw=zdc.HW.W)

        @zdc.dataclass
        class Comp2(zdc.Component):
            regs: Regs2 = zdc.inst()

        ir = SynthIR(component=Comp2, config=SynthConfig())
        ElaboratePass(Comp2, {}).run(ir)
        pass_ = MmrRegFileEmitPass(SynthConfig())
        pass_.run(ir)

        assert "regs" in pass_.results
        assert "module regs2" in pass_.results["regs"]
