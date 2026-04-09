"""Tests for AccessorLowering — generic sync-method-to-wire lowering.

Exercises the class against the RISC-V ``Decode`` action to verify that:
  * Wire declarations are produced for all simple bit-field extractors
  * Sign-extended immediates are correctly encoded
  * Methods that are not pure input-field expressions are silently skipped
  * No name matching is used — detection is structural only

The expected SV strings are derived from the RISC-V ISA encoding, not
from the Python method names.
"""

import os
import sys
import pytest

_this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_this_dir, '..', 'src'))
sys.path.insert(0, os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src'))
sys.path.insert(0, os.path.join(_this_dir, '..', '..', '..', 'src'))

from org.zuspec.example.mls.riscv.rv_core import Decode
from zuspec.synth.sprtl.accessor_lowering import (
    AccessorLowering,
    _detect_sext_body,
    _find_input_field_names,
    _discover_accessor_methods,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def al():
    return AccessorLowering(Decode, input_signal='d_insn', prefix='d', xlen=32)


@pytest.fixture(scope='module')
def wires(al):
    return al.emit_wires()


@pytest.fixture(scope='module')
def wire_map(wires):
    """Map wire name → full declaration line."""
    result = {}
    for line in wires:
        # e.g. "wire [4:0] d_rd = d_insn[11:7];"
        # strip "wire " prefix and split on " = "
        body = line[len('wire '):]
        name_part = body.split(' = ')[0].split()[-1]  # last token before ' = '
        result[name_part] = line
    return result


# ---------------------------------------------------------------------------
# Input-field discovery
# ---------------------------------------------------------------------------

class TestInputFieldDetection:
    def test_insn_detected_as_input(self):
        fields = _find_input_field_names(Decode)
        assert 'insn' in fields

    def test_output_fields_not_included(self):
        fields = _find_input_field_names(Decode)
        # 'rd', 'rs1', etc. are output() fields — must NOT be in input set
        assert 'rd' not in fields
        assert 'rs1' not in fields
        assert 'imm_i' not in fields


# ---------------------------------------------------------------------------
# Method discovery
# ---------------------------------------------------------------------------

class TestMethodDiscovery:
    def test_staticmethod_excluded(self):
        """_sext is @staticmethod — must not appear in accessor list."""
        methods = dict(_discover_accessor_methods(Decode))
        assert '_sext' not in methods

    def test_instance_accessor_included(self):
        methods = dict(_discover_accessor_methods(Decode))
        assert '_opcode' in methods
        assert '_rd' in methods
        assert '_imm_i' in methods

    def test_constraint_methods_included_as_candidates(self):
        """c_alu etc. are (self)-only instance methods; discovery includes them.
        They will be skipped during translation (not filtered here)."""
        methods = dict(_discover_accessor_methods(Decode))
        assert 'c_alu' in methods


# ---------------------------------------------------------------------------
# Structural sext detection
# ---------------------------------------------------------------------------

class TestSextDetection:
    def test_sext_detected_by_body_structure(self):
        import inspect, textwrap
        raw = vars(Decode).get('_sext')
        assert raw is not None, "_sext not found on Decode"
        assert _detect_sext_body(raw), "_sext body not detected as sign-extension"

    def test_non_sext_method_not_detected(self):
        raw = vars(Decode).get('_rd')
        assert raw is not None
        assert not _detect_sext_body(raw)


# ---------------------------------------------------------------------------
# Simple bit-field wires
# ---------------------------------------------------------------------------

class TestSimpleFieldWires:
    """Bit-field extractors should become single-slice wire declarations."""

    def test_opcode_wire(self, wire_map):
        line = wire_map.get('d_opcode')
        assert line is not None, "d_opcode wire not generated"
        assert 'd_insn[6:0]' in line
        assert '[6:0]' in line

    def test_rd_wire(self, wire_map):
        line = wire_map.get('d_rd')
        assert line is not None, "d_rd wire not generated"
        assert 'd_insn[11:7]' in line
        assert '[4:0]' in line

    def test_rs1_wire(self, wire_map):
        line = wire_map.get('d_rs1')
        assert line is not None
        assert 'd_insn[19:15]' in line
        assert '[4:0]' in line

    def test_rs2_wire(self, wire_map):
        line = wire_map.get('d_rs2')
        assert line is not None
        assert 'd_insn[24:20]' in line
        assert '[4:0]' in line

    def test_funct3_wire(self, wire_map):
        line = wire_map.get('d_funct3')
        assert line is not None
        assert 'd_insn[14:12]' in line
        assert '[2:0]' in line

    def test_funct7_wire(self, wire_map):
        line = wire_map.get('d_funct7')
        assert line is not None
        assert 'd_insn[31:25]' in line
        assert '[6:0]' in line


# ---------------------------------------------------------------------------
# Sign-extended immediate wires
# ---------------------------------------------------------------------------

class TestImmediateWires:
    """Immediates require sign extension from the input field."""

    def test_imm_i_width(self, wire_map):
        line = wire_map.get('d_imm_i')
        assert line is not None, "d_imm_i wire not generated"
        assert '[31:0]' in line

    def test_imm_i_bits(self, wire_map):
        # I-type: insn[31:20], sign-extended to 32 bits
        line = wire_map['d_imm_i']
        assert 'd_insn[31:20]' in line
        assert 'd_insn[31]' in line  # sign bit

    def test_imm_s_generated(self, wire_map):
        line = wire_map.get('d_imm_s')
        assert line is not None, "d_imm_s wire not generated"
        assert '[31:0]' in line
        # S-type: insn[31:25] and insn[11:7]
        assert 'd_insn[31:25]' in line
        assert 'd_insn[11:7]' in line

    def test_imm_b_generated(self, wire_map):
        line = wire_map.get('d_imm_b')
        assert line is not None, "d_imm_b wire not generated"
        assert '[31:0]' in line
        # B-type: bit 0 is always 0
        assert "1'b0" in line

    def test_imm_u_generated(self, wire_map):
        line = wire_map.get('d_imm_u')
        assert line is not None, "d_imm_u wire not generated"
        assert '[31:0]' in line
        # U-type: insn[31:12] kept at position, lower 12 bits are zero
        assert 'd_insn[31:12]' in line
        assert "12'b0" in line

    def test_imm_j_generated(self, wire_map):
        line = wire_map.get('d_imm_j')
        assert line is not None, "d_imm_j wire not generated"
        assert '[31:0]' in line
        # J-type: bit 0 is always 0
        assert "1'b0" in line


# ---------------------------------------------------------------------------
# Non-translatable methods silently skipped
# ---------------------------------------------------------------------------

class TestSkipping:
    def test_constraint_methods_skipped(self, wire_map):
        """Constraint dispatch methods must NOT appear as wires."""
        assert 'd_c_alu' not in wire_map
        assert 'c_alu' not in wire_map

    def test_alu_op_skipped(self, wire_map):
        """_alu_op calls other instance methods and uses match — not translatable."""
        assert 'd_alu_op' not in wire_map


# ---------------------------------------------------------------------------
# No-prefix mode
# ---------------------------------------------------------------------------

class TestNoPrefix:
    def test_no_prefix(self):
        al = AccessorLowering(Decode, input_signal='insn', prefix='', xlen=32)
        wires = al.emit_wires()
        names = [w.split('=')[0].split()[-1] for w in wires]
        assert 'rd' in names
        assert 'funct3' in names
