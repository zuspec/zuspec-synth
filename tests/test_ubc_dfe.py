"""
Tests for UnitBodyCompiler (UBC) and DecodeFieldExtractor (DFE).

These tests pin the SV output produced from the RISC-V behavioral model so
that future refactors don't silently change the generated RTL.

Each test checks:
  * the correct number of output lines
  * specific semantically-critical lines (correct encoding, guards, etc.)
  * the overall structure (case header/footer, wire declarations)
"""

import os
import sys
import pytest

_this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_this_dir, '..', 'src'))
sys.path.insert(0, os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src'))
sys.path.insert(0, os.path.join(_this_dir, '..', '..', '..', 'src'))

from org.zuspec.example.mls.riscv.rv_units import ALUUnit, MulDivUnit
from org.zuspec.example.mls.riscv.rv_core import RVCore
from zuspec.synth.sprtl.unit_body_compiler import UnitBodyCompiler
from org.zuspec.example.mls.riscv.sprtl.decode_field_extractor import DecodeFieldExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_decode_cls():
    import importlib
    mod = importlib.import_module('org.zuspec.example.mls.riscv.rv_core')
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name == 'Decode':
            return obj
    raise RuntimeError("Decode class not found in rv_core module")


# ---------------------------------------------------------------------------
# UnitBodyCompiler — ALUUnit
# ---------------------------------------------------------------------------

class TestUBCAlU:
    """UnitBodyCompiler compiled from ALUUnit.execute()."""

    @pytest.fixture(scope='class')
    def ubc(self):
        return UnitBodyCompiler(ALUUnit, 'execute')

    def test_emit_alu_case_inner_line_count(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='e',
                                         signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'},
                                         indent='      ')
        # 1 case header + 9 arms (0-8 + default) + 1 endcase = 12 lines
        assert len(lines) == 12

    def test_emit_alu_case_header_footer(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='e',
                                         signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'},
                                         indent='      ')
        assert lines[0].strip() == 'case (e_alu_op)'
        assert lines[-1].strip() == 'endcase'

    def test_add_operation(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='e',
                                         signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'})
        add_line = next(l for l in lines if l.strip().startswith('0:'))
        assert 'e_alu_a + e_alu_b' in add_line

    def test_sub_operation(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='e',
                                         signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'})
        sub_line = next(l for l in lines if l.strip().startswith('1:'))
        assert 'e_alu_a - e_alu_b' in sub_line

    def test_slt_signed(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='e',
                                         signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'})
        slt_line = next(l for l in lines if l.strip().startswith('3:'))
        assert '$signed(e_alu_a)' in slt_line
        assert '$signed(e_alu_b)' in slt_line

    def test_sltu_unsigned(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='e',
                                         signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'})
        sltu_line = next(l for l in lines if l.strip().startswith('4:'))
        assert '$signed' not in sltu_line
        assert 'e_alu_a < e_alu_b' in sltu_line

    def test_sra_arithmetic_shift(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='e',
                                         signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'})
        sra_line = next(l for l in lines if l.strip().startswith('7:'))
        assert '>>>' in sra_line
        assert '$signed' in sra_line

    def test_full_always_block(self, ubc):
        lines = ubc.emit_alu_case(32, pfx='e',
                                   signal_map={'rs1': 'e_alu_a', 'rs2': 'e_alu_b'})
        full = '\n'.join(lines)
        assert 'always @(*)' in full
        assert 'case (e_alu_op)' in full
        assert 'endcase' in full

    def test_prefix_propagated(self, ubc):
        lines = ubc.emit_alu_case_inner(32, pfx='x',
                                         signal_map={'rs1': 'x_alu_a', 'rs2': 'x_alu_b'})
        for l in lines:
            if ':' in l and l.strip()[0].isdigit():
                assert 'x_alu_result' in l


# ---------------------------------------------------------------------------
# UnitBodyCompiler — MulDivUnit
# ---------------------------------------------------------------------------

class TestUBCMulDiv:
    """UnitBodyCompiler compiled from MulDivUnit.execute()."""

    @pytest.fixture(scope='class')
    def ubc(self):
        return UnitBodyCompiler(MulDivUnit, 'execute')

    # --- emit_muldiv_wires ---

    def test_wires_count(self, ubc):
        wires = ubc.emit_muldiv_wires(32, pfx='e', a='e_alu_a', b='e_alu_b')
        assert len(wires) == 3

    def test_wires_width(self, ubc):
        wires = ubc.emit_muldiv_wires(32, pfx='e', a='e_alu_a', b='e_alu_b')
        for w in wires:
            assert '[63:0]' in w

    def test_wire_mul_uu(self, ubc):
        wires = ubc.emit_muldiv_wires(32, pfx='e', a='e_alu_a', b='e_alu_b')
        uu = next(w for w in wires if 'mul_uu' in w)
        assert 'e_alu_a * e_alu_b' in uu
        assert '$signed' not in uu

    def test_wire_mul_ss(self, ubc):
        wires = ubc.emit_muldiv_wires(32, pfx='e', a='e_alu_a', b='e_alu_b')
        ss = next(w for w in wires if 'mul_ss' in w)
        assert '$signed(e_alu_a) * $signed(e_alu_b)' in ss

    def test_wire_mul_su(self, ubc):
        wires = ubc.emit_muldiv_wires(32, pfx='e', a='e_alu_a', b='e_alu_b')
        su = next(w for w in wires if 'mul_su' in w)
        assert '$signed(e_alu_a) * $unsigned(e_alu_b)' in su

    # --- emit_muldiv_case_inner ---

    def test_case_header_footer(self, ubc):
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        assert lines[0].strip() == 'case (e_funct3)'
        assert lines[-1].strip() == 'endcase'

    def test_mul_lower(self, ubc):
        """MUL (funct3=0): lower 32 bits of signed×signed product."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        mul = next(l for l in lines if "3'd0:" in l)
        assert 'e_mul_ss[31:0]' in mul

    def test_mulh_upper_ss(self, ubc):
        """MULH (funct3=1): upper 32 bits of signed×signed product."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        mulh = next(l for l in lines if "3'd1:" in l)
        assert 'e_mul_ss[63:32]' in mulh

    def test_mulhsu_upper_su(self, ubc):
        """MULHSU (funct3=2): upper 32 bits of signed×unsigned product."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        mulhsu = next(l for l in lines if "3'd2:" in l)
        assert 'e_mul_su[63:32]' in mulhsu

    def test_mulhu_upper_uu(self, ubc):
        """MULHU (funct3=3): upper 32 bits of unsigned×unsigned product."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        mulhu = next(l for l in lines if "3'd3:" in l)
        assert 'e_mul_uu[63:32]' in mulhu

    def test_div_signed_with_divzero(self, ubc):
        """DIV (funct3=4): signed division with div-by-zero → all-ones."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        div = next(l for l in lines if "3'd4:" in l)
        assert "{32{1'b0}}" in div       # div-by-zero check
        assert "{32{1'b1}}" in div       # all-ones result
        assert '$signed(e_alu_a) / $signed(e_alu_b)' in div

    def test_divu_unsigned_with_divzero(self, ubc):
        """DIVU (funct3=5): unsigned division with div-by-zero → all-ones."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        divu = next(l for l in lines if "3'd5:" in l)
        assert "{32{1'b1}}" in divu
        assert '$signed' not in divu
        assert 'e_alu_a / e_alu_b' in divu

    def test_rem_signed_with_divzero(self, ubc):
        """REM (funct3=6): signed remainder, div-by-zero → dividend."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        rem = next(l for l in lines if "3'd6:" in l)
        assert 'e_alu_a' in rem          # dividend returned on div-by-zero
        assert '$signed(e_alu_a) % $signed(e_alu_b)' in rem

    def test_remu_unsigned_default(self, ubc):
        """REMU (default/else): unsigned remainder, div-by-zero → dividend."""
        lines = ubc.emit_muldiv_case_inner(32, pfx='e', a='e_alu_a', b='e_alu_b')
        remu = next(l for l in lines if 'default:' in l)
        assert 'e_alu_a % e_alu_b' in remu
        assert '$signed' not in remu

    def test_prefix_propagated(self, ubc):
        wires = ubc.emit_muldiv_wires(32, pfx='m', a='m_alu_a', b='m_alu_b')
        for w in wires:
            assert 'm_mul_' in w

    def test_xlen64_wire_width(self, ubc):
        wires = ubc.emit_muldiv_wires(64, pfx='e', a='e_alu_a', b='e_alu_b')
        for w in wires:
            assert '[127:0]' in w


# ---------------------------------------------------------------------------
# DecodeFieldExtractor — Decode class from rv_core
# ---------------------------------------------------------------------------

class TestDFEDecode:
    """DecodeFieldExtractor compiled from rv_core.Decode."""

    @pytest.fixture(scope='class')
    def dfe(self):
        Decode = _find_decode_cls()
        return DecodeFieldExtractor(Decode, insn_signal='e_insn', prefix='e', xlen=32)

    # --- imm wires ---

    def test_imm_wires_count(self, dfe):
        wires = dfe.emit_imm_wires()
        assert len(wires) == 5

    def test_imm_i_sign_extend(self, dfe):
        wires = dfe.emit_imm_wires()
        imm_i = next(w for w in wires if 'imm_i' in w)
        # sign-extended from bit 31
        assert 'e_insn[31]' in imm_i
        assert 'e_insn[31:20]' in imm_i

    def test_imm_s_bits(self, dfe):
        wires = dfe.emit_imm_wires()
        imm_s = next(w for w in wires if 'imm_s' in w)
        assert 'e_insn[31:25]' in imm_s
        assert 'e_insn[11:7]' in imm_s

    def test_imm_b_scrambled(self, dfe):
        wires = dfe.emit_imm_wires()
        imm_b = next(w for w in wires if 'imm_b' in w)
        # B-type: bits 31, 7, 30:25, 11:8 reassembled
        assert 'e_insn[31]' in imm_b
        assert 'e_insn[7]' in imm_b
        assert 'e_insn[30:25]' in imm_b
        assert 'e_insn[11:8]' in imm_b

    def test_imm_u_upper(self, dfe):
        wires = dfe.emit_imm_wires()
        imm_u = next(w for w in wires if 'imm_u' in w)
        assert 'e_insn[31:12]' in imm_u

    def test_imm_j_scrambled(self, dfe):
        wires = dfe.emit_imm_wires()
        imm_j = next(w for w in wires if 'imm_j' in w)
        assert 'e_insn[31]' in imm_j
        assert 'e_insn[19:12]' in imm_j
        assert 'e_insn[20]' in imm_j
        assert 'e_insn[30:21]' in imm_j

    # --- alu_op wire ---

    def test_alu_op_wire_structure(self, dfe):
        lines = dfe.emit_alu_op_wire(rtype_guard='e_is_op')
        text = '\n'.join(lines)
        assert 'always @(*)' in text
        assert 'case (e_funct3)' in text
        assert 'endcase' in text

    def test_alu_op_reg_declaration(self, dfe):
        lines = dfe.emit_alu_op_wire(rtype_guard='e_is_op')
        assert any('reg' in l and 'e_alu_op' in l for l in lines)

    def test_alu_op_funct3_0_add_sub(self, dfe):
        """funct3=0: ADD (op) vs SUB (r-type with funct7[5])."""
        lines = dfe.emit_alu_op_wire(rtype_guard='e_is_op')
        arm0 = next(l for l in lines if "3'd0:" in l)
        assert 'e_is_op' in arm0
        assert 'e_funct7[5]' in arm0

    def test_alu_op_funct3_5_srl_sra(self, dfe):
        """funct3=5: SRL vs SRA distinguished by funct7[5]."""
        lines = dfe.emit_alu_op_wire(rtype_guard='e_is_op')
        arm5 = next(l for l in lines if "3'd5:" in l)
        assert 'e_funct7[5]' in arm5

    def test_alu_op_nine_cases(self, dfe):
        """Should produce 7 explicit funct3 arms + 1 default = 8 case arms total."""
        lines = dfe.emit_alu_op_wire(rtype_guard='e_is_op')
        case_arms = [l for l in lines if ':' in l and (
            any(f"3'd{i}:" in l for i in range(7)) or 'default:' in l)]
        assert len(case_arms) == 8

    def test_has_method_alu_op(self, dfe):
        assert dfe.has_method('_alu_op')
