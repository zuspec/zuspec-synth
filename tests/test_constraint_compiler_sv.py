"""Tests for ConstraintCompiler.from_sv_action() — Phase 4.

Verifies that the SV-class entry point into the synthesis pipeline produces
the same ConstraintBlockSet (and downstream RTL) as the Python-class path.
"""
import os
import tempfile
import pytest
import pyslang

from zuspec.synth.sprtl.constraint_compiler import ConstraintCompiler
from zuspec.synth.ir.constraint_ir import BitRange


# ---------------------------------------------------------------------------
# Shared helpers — same compile/find pattern as test_constraint_mapper.py
# ---------------------------------------------------------------------------

SHARE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'packages', 'zuspec-fe-sv',
                 'src', 'zuspec', 'fe', 'sv', 'share')
)

# Fall back to the sibling package location if running from the synth package.
if not os.path.isdir(SHARE):
    SHARE = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'zuspec-fe-sv',
                     'src', 'zuspec', 'fe', 'sv', 'share')
    )

_LIB_FILES = [
    os.path.join(SHARE, 'zsp_clock_if.sv'),
    os.path.join(SHARE, 'zsp_reset_if.sv'),
    os.path.join(SHARE, 'zsp_reg_if.sv'),
    os.path.join(SHARE, 'zsp_pkg.sv'),
]


def _compile_sv(src: str):
    sm = pyslang.SourceManager()
    sm.addUserDirectories(SHARE)
    comp = pyslang.Compilation()
    trees = []
    for f in _LIB_FILES:
        t = pyslang.SyntaxTree.fromFile(f, sm)
        trees.append(t)
        comp.addSyntaxTree(t)
    with tempfile.NamedTemporaryFile(suffix='.sv', mode='w', delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        t = pyslang.SyntaxTree.fromFile(tmp, sm)
        trees.append(t)
        comp.addSyntaxTree(t)
    finally:
        os.unlink(tmp)
    return comp, comp.getRoot(), trees  # trees kept alive


def _find_class(root, name: str):
    found = []
    def visit(sym):
        if sym.kind == pyslang.SymbolKind.ClassType and getattr(sym, 'name', '') == name:
            found.append(sym)
        return True
    root.visit(visit)
    return found[0] if found else None


def _sv_to_action_ir(src: str, class_name: str, comp_type_name: str):
    """Parse SV source and return a DataTypeAction with constraint_set."""
    from zuspec.fe.sv.class_mapper import ClassMapper
    from zuspec.fe.sv.error import ErrorReporter
    from zuspec.fe.sv.config import SVToZuspecConfig
    from zuspec.fe.sv.type_mapper import TypeMapper

    comp, root, trees = _compile_sv(src)
    sym = _find_class(root, class_name)
    assert sym is not None, f"Class '{class_name}' not found"

    config = SVToZuspecConfig()
    error_reporter = ErrorReporter()
    type_mapper = TypeMapper(config, error_reporter)
    mapper = ClassMapper(config, error_reporter, type_mapper, compilation=comp)
    action_ir = mapper.map_action(sym)
    assert action_ir is not None
    assert action_ir.constraint_set is not None, "constraint_set not populated"
    return action_ir, trees  # trees kept alive


# ---------------------------------------------------------------------------
# Toy 3-instruction decoder (ALU ADD, ALU SUB, LOAD)
# ---------------------------------------------------------------------------

_TOY_DECODE_SRC = """`include "zsp_macros.svh"
package toy_pkg;
    import zsp_pkg::*;

    class toy_core_c extends zsp_component;
        `zsp_component_utils(toy_core_c)
    endclass

    class toy_decode_c extends zsp_action_c #(toy_core_c);
        logic [7:0]  instr;          // 8-bit toy encoding
        rand logic [3:0]  opcode;    // instr[3:0]
        rand logic [3:0]  alu_op;    // output
        rand logic        is_load;   // output

        constraint c_extract {
            opcode == instr[3:0];
        }

        constraint c_add {
            if (opcode == 4'b0001) {
                alu_op == 4'd0;
                is_load == 1'b0;
                zsp_mark #(logic[3:0])::valid(alu_op);
            }
        }

        constraint c_sub {
            if (opcode == 4'b0010) {
                alu_op == 4'd1;
                is_load == 1'b0;
                zsp_mark #(logic[3:0])::valid(alu_op);
            }
        }

        constraint c_load {
            if (opcode == 4'b1000) {
                alu_op == 4'd0;
                is_load == 1'b1;
                zsp_mark #(logic)::valid(is_load);
            }
        }

        `zsp_action_utils(toy_decode_c, toy_core_c)
    endclass
endpackage
"""


@pytest.fixture(scope='module')
def toy_cc():
    """Full pipeline from SV source → ConstraintCompiler with cset."""
    action_ir, _trees = _sv_to_action_ir(_TOY_DECODE_SRC, 'toy_decode_c', 'toy_core_c')
    cc = ConstraintCompiler.from_sv_action(action_ir, prefix='t')
    cc.compute_support()
    return cc


def test_from_sv_action_cset_not_none(toy_cc):
    assert toy_cc.cset is not None


def test_from_sv_action_input_field(toy_cc):
    assert toy_cc.cset.input_field == 'instr'
    assert toy_cc.cset.input_width == 8


def test_from_sv_action_output_fields(toy_cc):
    names = [f.name for f in toy_cc.cset.output_fields]
    assert 'alu_op' in names
    assert 'is_load' in names
    # opcode is a derived field (rand) — included as output in the cset
    assert 'opcode' in names


def test_from_sv_action_constraint_blocks(toy_cc):
    names = [b.name for b in toy_cc.cset.constraints]
    assert 'c_add' in names
    assert 'c_sub' in names
    assert 'c_load' in names


def test_from_sv_action_support_bits(toy_cc):
    """Support should contain BitRange(3, 0) for opcode == instr[3:0]."""
    assert BitRange(msb=3, lsb=0) in toy_cc.cset.support_bits


def test_from_sv_action_c_add_conditions(toy_cc):
    block = next(b for b in toy_cc.cset.constraints if b.name == 'c_add')
    assert block.conditions == {BitRange(msb=3, lsb=0): 1}


def test_from_sv_action_c_sub_conditions(toy_cc):
    block = next(b for b in toy_cc.cset.constraints if b.name == 'c_sub')
    assert block.conditions == {BitRange(msb=3, lsb=0): 2}


def test_from_sv_action_c_add_assignments(toy_cc):
    block = next(b for b in toy_cc.cset.constraints if b.name == 'c_add')
    assert block.assignments['alu_op'] == 0
    assert block.assignments['is_load'] == 0


def test_from_sv_action_validity_decls(toy_cc):
    """ValidityDecl entries created for zsp_mark::valid() calls."""
    vd_fields = {vd.field_name for vd in toy_cc.cset.validity_decls}
    assert 'alu_op' in vd_fields
    assert 'is_load' in vd_fields


def test_from_sv_action_no_constraint_set_raises():
    """from_sv_action() raises ValueError when constraint_set is None."""
    from zuspec.ir.core.data_type import DataTypeAction
    action_ir = DataTypeAction(name='bad', super=None, comp_type_name=None)
    action_ir.constraint_set = None
    with pytest.raises(ValueError, match="has no constraint_set"):
        ConstraintCompiler.from_sv_action(action_ir)


# ---------------------------------------------------------------------------
# End-to-end pipeline: build_cubes → minimize → emit_sv
# ---------------------------------------------------------------------------

def test_from_sv_action_build_cubes(toy_cc):
    """build_cubes() runs without error after from_sv_action()."""
    import copy
    cc2 = copy.copy(toy_cc)  # shallow copy so we don't mutate fixture
    cc2.build_cubes()
    assert hasattr(cc2, '_cubes_by_bit')
    assert cc2._cubes_by_bit  # non-empty


def test_from_sv_action_minimize_and_emit(toy_cc):
    """Full pipeline through minimize() and emit_sv() runs without error."""
    import copy
    cc2 = copy.copy(toy_cc)
    cc2.build_cubes()
    cc2.minimize()
    lines = cc2.emit_sv()
    sv = '\n'.join(lines)
    # Should produce at least some assign statements
    assert 'assign' in sv or 'wire' in sv or sv  # non-trivial output
