"""Unit tests for ConstraintCompiler extensions.

Tests cover:
  1. Input field detection via zdc.input() marker
  2. Derived-field guard resolution (self.opcode == X → BitRange)
  3. Bit-extraction pre-pass: shift-mask pattern
  4. Bit-extraction pre-pass: subscript pattern
  5. Full pipeline with named derived-field guards
  6. Full pipeline with bit-slice guards (identical output)
"""
import dataclasses
import pytest

import sys, pathlib
_pkg = pathlib.Path(__file__).parents[2] / 'zuspec-dataclasses' / 'src'
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

import zuspec.dataclasses as zdc
from zuspec.dataclasses.decorators import constraint, Input
from zuspec.synth.sprtl.constraint_compiler import ConstraintCompiler
from zuspec.synth.ir.constraint_ir import BitRange


# ---------------------------------------------------------------------------
# Minimal test fixtures
# ---------------------------------------------------------------------------

@zdc.dataclass
class _MinimalNamed:
    """Three-instruction mini-decode using named derived-field guards."""
    instr    : zdc.u8 = zdc.input()

    opcode   : zdc.u4 = zdc.rand()   # instr[3:0]
    funct    : zdc.u1 = zdc.rand()   # instr[4]

    out_a    : zdc.u1 = zdc.rand()
    out_b    : zdc.u1 = zdc.rand()
    out_c    : zdc.u2 = zdc.rand()

    @constraint
    def c_extract(self):
        assert self.opcode == (self.instr & 0xF)
        assert self.funct  == ((self.instr >> 4) & 0x1)

    @constraint
    def c_instr0(self):
        if self.opcode == 0 and self.funct == 0:
            assert self.out_a == 1
            assert self.out_b == 0
            assert self.out_c == 1

    @constraint
    def c_instr1(self):
        if self.opcode == 1 and self.funct == 0:
            assert self.out_a == 0
            assert self.out_b == 1
            assert self.out_c == 2

    @constraint
    def c_instr2(self):
        if self.opcode == 2 and self.funct == 1:
            assert self.out_a == 1
            assert self.out_b == 1
            assert self.out_c == 3


@zdc.dataclass
class _MinimalBitSlice:
    """Same decode, but using bit-slice guards in the if conditions."""
    instr    : zdc.u8 = zdc.input()

    out_a    : zdc.u1 = zdc.rand()
    out_b    : zdc.u1 = zdc.rand()
    out_c    : zdc.u2 = zdc.rand()

    @constraint
    def c_instr0(self):
        if self.instr[3:0] == 0 and self.instr[4:4] == 0:
            assert self.out_a == 1
            assert self.out_b == 0
            assert self.out_c == 1

    @constraint
    def c_instr1(self):
        if self.instr[3:0] == 1 and self.instr[4:4] == 0:
            assert self.out_a == 0
            assert self.out_b == 1
            assert self.out_c == 2

    @constraint
    def c_instr2(self):
        if self.instr[3:0] == 2 and self.instr[4:4] == 1:
            assert self.out_a == 1
            assert self.out_b == 1
            assert self.out_c == 3


@zdc.dataclass
class _NoInputMarker:
    """Legacy class: input field detected via subscript heuristic, no zdc.input()."""
    insn     : zdc.u8 = zdc.rand()   # deliberately rand — exercises legacy path

    out_x    : zdc.u1 = zdc.rand()

    @constraint
    def c_x(self):
        if self.insn[3:0] == 5:
            assert self.out_x == 1


# ---------------------------------------------------------------------------
# Test 1: Input field detected via zdc.input() marker
# ---------------------------------------------------------------------------

def test_input_field_detected_from_zdc_input():
    cc = ConstraintCompiler(_MinimalNamed, prefix='')
    cc.extract()
    assert cc.cset.input_field == 'instr', (
        f"Expected input_field='instr', got '{cc.cset.input_field}'"
    )
    assert cc.cset.input_width == 8, (
        f"Expected input_width=8, got {cc.cset.input_width}"
    )
    # 'instr' must not appear in output_fields
    output_names = {f.name for f in cc.cset.output_fields}
    assert 'instr' not in output_names


# ---------------------------------------------------------------------------
# Test 2: Derived-field guard resolves to correct BitRange
# ---------------------------------------------------------------------------

def test_derived_field_guard_resolves_to_bitrange():
    cc = ConstraintCompiler(_MinimalNamed, prefix='')
    cc.extract()
    # After extract(), _derived_to_bitrange should map opcode → BitRange(3,0)
    assert 'opcode' in cc._derived_to_bitrange, "opcode not in derived map"
    assert 'funct' in cc._derived_to_bitrange, "funct not in derived map"
    br_opcode = cc._derived_to_bitrange['opcode']
    assert br_opcode == BitRange(msb=3, lsb=0), f"Got {br_opcode}"
    br_funct = cc._derived_to_bitrange['funct']
    assert br_funct == BitRange(msb=4, lsb=4), f"Got {br_funct}"


# ---------------------------------------------------------------------------
# Test 3: Pre-pass parses shift-mask extraction pattern
# ---------------------------------------------------------------------------

def test_extract_fields_mask_shift_parsed():
    """(self.instr >> 4) & 0x1 → BitRange(4, 4)."""
    cc = ConstraintCompiler(_MinimalNamed, prefix='')
    cc.extract()
    br = cc._derived_to_bitrange.get('funct')
    assert br is not None, "funct not in derived_to_bitrange"
    assert br.lsb == 4 and br.msb == 4, f"Expected BitRange(4,4), got {br}"


# ---------------------------------------------------------------------------
# Test 4: Pre-pass parses subscript extraction pattern
# ---------------------------------------------------------------------------

@zdc.dataclass
class _SubscriptExtract:
    instr    : zdc.u8 = zdc.input()
    nibble   : zdc.u4 = zdc.rand()
    out_x    : zdc.u1 = zdc.rand()

    @constraint
    def c_extract(self):
        assert self.nibble == self.instr[3:0]

    @constraint
    def c_x(self):
        if self.nibble == 7:
            assert self.out_x == 1


def test_extract_fields_subscript_parsed():
    """self.instr[3:0] → BitRange(3, 0)."""
    cc = ConstraintCompiler(_SubscriptExtract, prefix='')
    cc.extract()
    br = cc._derived_to_bitrange.get('nibble')
    assert br is not None, "nibble not in derived_to_bitrange"
    assert br.msb == 3 and br.lsb == 0, f"Expected BitRange(3,0), got {br}"


# ---------------------------------------------------------------------------
# Test 5: Full pipeline with named derived-field guards produces valid SV
# ---------------------------------------------------------------------------

def _run_full_pipeline(cls, prefix=''):
    cc = ConstraintCompiler(cls, prefix=prefix)
    cc.extract()
    cc.compute_support()
    cc.validate(warn_only=True)
    cc.build_cubes()
    cc.minimize()
    return cc.emit_sv()


def test_full_pipeline_named_fields():
    lines = _run_full_pipeline(_MinimalNamed)
    sv = '\n'.join(lines)
    # Should emit wire declarations for out_a, out_b, out_c
    assert 'out_a' in sv, "out_a missing from SV output"
    assert 'out_b' in sv, "out_b missing from SV output"
    assert 'out_c' in sv, "out_c missing from SV output"
    # Should reference input bit-ranges in wire assignments
    assert 'instr[' in sv or 'in_' in sv, "No input bit references found"


# ---------------------------------------------------------------------------
# Test 6: Full pipeline with bit-slice guards — output must agree with Test 5
# ---------------------------------------------------------------------------

def test_full_pipeline_bitslice_fields():
    lines_named = _run_full_pipeline(_MinimalNamed)
    lines_slice = _run_full_pipeline(_MinimalBitSlice)

    # Both pipelines should produce the same number of assignment lines for
    # the common outputs (out_a, out_b, out_c).  The named pipeline also emits
    # assignments for the derived fields opcode and funct, so we compare only
    # the shared signal names.
    def assigns_for(lines, signals):
        return {s: any(s in l and l.strip().startswith('assign') for l in lines)
                for s in signals}

    common = ['out_a', 'out_b', 'out_c']
    named_assigns = assigns_for(lines_named, common)
    slice_assigns = assigns_for(lines_slice, common)
    assert named_assigns == slice_assigns, (
        f"Named-field assigns: {named_assigns}\n"
        f"Bit-slice  assigns: {slice_assigns}"
    )
    # Both should emit all three common outputs.
    for sig in common:
        assert named_assigns[sig], f"Named pipeline missing assign for {sig}"
        assert slice_assigns[sig], f"Bit-slice pipeline missing assign for {sig}"


# ---------------------------------------------------------------------------
# Test 7: Constraint inheritance — subclass picks up parent @constraint methods
# ---------------------------------------------------------------------------

@zdc.dataclass
class _InheritBase:
    """Base class with one constraint."""
    instr : zdc.u8 = zdc.input()
    out_a : zdc.u1 = zdc.rand()
    out_b : zdc.u1 = zdc.rand()

    @constraint
    def c_base(self):
        if self.instr[3:0] == 0:
            assert self.out_a == 1
            assert self.out_b == 0


@zdc.dataclass
class _InheritChild(_InheritBase):
    """Subclass adds a second constraint; inherits c_base from parent."""

    @constraint
    def c_child(self):
        if self.instr[3:0] == 1:
            assert self.out_a == 0
            assert self.out_b == 1


@zdc.dataclass
class _InheritOverride(_InheritBase):
    """Subclass overrides the parent constraint with a different encoding."""

    @constraint
    def c_base(self):  # same name — should replace parent version
        if self.instr[3:0] == 7:
            assert self.out_a == 1
            assert self.out_b == 0


def test_inheritance_picks_up_parent_constraints():
    """extract() must collect @constraint methods from base classes via MRO."""
    cc = ConstraintCompiler(_InheritChild)
    cc.extract()
    names = {c.name for c in cc.cset.constraints}
    assert 'c_base'  in names, "Parent constraint c_base not collected"
    assert 'c_child' in names, "Child constraint c_child not collected"


def test_inheritance_child_has_two_encodings():
    """Full pipeline on _InheritChild produces output for both encodings."""
    cc = ConstraintCompiler(_InheritChild)
    cc.extract(); cc.compute_support(); cc.build_cubes(); cc.minimize()
    sv = '\n'.join(cc.emit_sv())
    assert 'out_a' in sv
    # Both encodings contribute cubes — total terms > 0
    total = sum(len(f.cubes) for f in cc.cset.sop_functions)
    assert total > 0, "Expected non-empty SOP after inheriting two constraints"


def test_inheritance_override_replaces_parent():
    """When subclass re-declares same constraint name, only the override is used."""
    cc_base     = ConstraintCompiler(_InheritBase)
    cc_override = ConstraintCompiler(_InheritOverride)
    cc_base.extract()
    cc_override.extract()

    # Both have exactly one constraint named 'c_base'.
    base_blocks     = [c for c in cc_base.cset.constraints     if c.name == 'c_base']
    override_blocks = [c for c in cc_override.cset.constraints if c.name == 'c_base']
    assert len(base_blocks)     == 1, "Base should have exactly one c_base"
    assert len(override_blocks) == 1, "Override should have exactly one c_base (not two)"

    # The override encodes a different value: instr[3:0] == 7 vs == 0.
    base_val     = list(base_blocks[0].conditions.values())
    override_val = list(override_blocks[0].conditions.values())
    assert base_val != override_val, (
        f"Base and override should encode different values: {base_val} vs {override_val}")


# ---------------------------------------------------------------------------
# OR-guard splitting tests
# ---------------------------------------------------------------------------

@zdc.dataclass
class _OrGuardAction:
    """Instruction class with OR-condition guards (e.g. immediate decode)."""
    instr  : zdc.u8 = zdc.input()

    is_ld  : zdc.u1 = zdc.rand()
    is_st  : zdc.u1 = zdc.rand()
    kind   : zdc.u2 = zdc.rand()

    @constraint
    def c_ld_st(self):
        # Opcode 1 or opcode 2 → is_ld=1
        if self.instr[3:2] == 1 or self.instr[3:2] == 2:
            assert self.is_ld == 1

    @constraint
    def c_store(self):
        if self.instr[3:2] == 3:
            assert self.is_st == 1

    @constraint
    def c_kind(self):
        if self.instr[1:0] == 0:
            assert self.kind == 0
        if self.instr[1:0] == 1:
            assert self.kind == 1
        if self.instr[1:0] == 2:
            assert self.kind == 2


def test_or_guard_splits_into_two_blocks():
    """OR-condition guard produces two ConstraintBlocks, one per arm."""
    cc = ConstraintCompiler(_OrGuardAction)
    cc.extract()
    ld_blocks = [b for b in cc.cset.constraints if b.name.startswith('c_ld_st')]
    assert len(ld_blocks) == 2, f"Expected 2 blocks for OR guard, got {len(ld_blocks)}"
    # Each arm should have a different condition value.
    vals = sorted([list(b.conditions.values())[0] for b in ld_blocks])
    assert vals == [1, 2], f"Expected condition values [1, 2], got {vals}"


def test_or_guard_blocks_have_same_assignment():
    """All blocks from an OR expansion share the same assignments."""
    cc = ConstraintCompiler(_OrGuardAction)
    cc.extract()
    ld_blocks = [b for b in cc.cset.constraints if b.name.startswith('c_ld_st')]
    for b in ld_blocks:
        assert b.assignments.get('is_ld') == 1


def test_or_guard_full_pipeline():
    """OR-guard action runs through full pipeline without errors."""
    cc = ConstraintCompiler(_OrGuardAction)
    cc.extract()
    cc.compute_support()
    cc.build_cubes()
    cc.minimize()
    sv = '\n'.join(cc.emit_sv())
    assert 'is_ld' in sv
    assert 'is_st' in sv
    assert 'kind' in sv


# ---------------------------------------------------------------------------
# Tuple parsing tests
# ---------------------------------------------------------------------------

def test_tuple_parses_in_constraint_parser():
    """ast.Tuple nodes are parsed to {'type': 'tuple', ...} without raising."""
    import ast
    from zuspec.dataclasses.constraint_parser import ConstraintParser

    src = "x == zdc.concat((0, 4), y)"
    tree = ast.parse(src, mode='eval')
    # The tuple (0, 4) is an argument to concat; parse_expr should not raise.
    parser = ConstraintParser()
    # Extract the concat call argument
    call = tree.body.comparators[0]
    result = parser.parse_expr(call.args[0])  # (0, 4)
    assert result['type'] == 'tuple', f"Expected 'tuple', got {result['type']}"
    assert len(result['elts']) == 2


# ---------------------------------------------------------------------------
# Expression assignment (sext/concat) tests
# ---------------------------------------------------------------------------

MASK32 = 0xFFFFFFFF


@zdc.dataclass
class _ImmDecodeAction:
    """Minimal immediate-decode pattern: sext with subscript RHS."""
    instr : zdc.u32 = zdc.input()
    imm   : zdc.u32 = zdc.rand()

    @constraint
    def c_itype(self):
        if self.instr[6:0] == 0x13:
            assert self.imm == zdc.sext(self.instr[31:20], 12) & MASK32

    @constraint
    def c_utype(self):
        if self.instr[6:0] == 0x37:
            assert self.imm == 0  # constant block alongside expression blocks


def test_expr_assignment_detected_as_expr_field():
    """imm field is detected as an expression field (bypasses SOP)."""
    cc = ConstraintCompiler(_ImmDecodeAction)
    cc.extract()
    assert 'imm' in cc._expr_fields, "imm should be in _expr_fields"


def test_constant_block_alongside_expr_block_coerced_to_expr_path():
    """Block with constant assignment for an expr field is treated as expr path."""
    cc = ConstraintCompiler(_ImmDecodeAction)
    cc.extract()
    # Both blocks should be in constraints
    itype_blocks = [b for b in cc.cset.constraints if 'itype' in b.name]
    utype_blocks = [b for b in cc.cset.constraints if 'utype' in b.name]
    assert len(itype_blocks) == 1
    assert len(utype_blocks) == 1
    # The constant block should still have its assignment recorded
    assert utype_blocks[0].assignments.get('imm') == 0


def test_expr_field_excluded_from_sop_cubes():
    """imm field produces no cubes in the SOP pipeline."""
    cc = ConstraintCompiler(_ImmDecodeAction)
    cc.extract()
    cc.compute_support()
    cc.build_cubes()
    # imm is an expr field — no cubes should exist for it
    imm_cube_keys = [k for k in cc._cubes_by_bit if 'imm' in k]
    for key in imm_cube_keys:
        assert cc._cubes_by_bit[key] == [], f"Expected empty cubes for {key}"


def test_expr_field_emits_structural_mux():
    """emit_sv generates structural priority-mux wires for imm."""
    cc = ConstraintCompiler(_ImmDecodeAction)
    cc.extract()
    cc.compute_support()
    cc.build_cubes()
    cc.minimize()
    sv = '\n'.join(cc.emit_sv())
    # Should have per-block wires
    assert 'd_imm_c0' in sv, f"Expected d_imm_c0 wire in:\n{sv}"
    assert 'd_imm_c1' in sv, f"Expected d_imm_c1 wire in:\n{sv}"
    # Should have assign d_imm = ...
    assert 'assign d_imm' in sv, f"Expected assign d_imm in:\n{sv}"
    # Should contain sext-style bit-replication pattern
    assert '{' in sv


def test_sext_renders_correct_sv():
    """_dict_expr_to_sv renders sext(instr[31:20], 12) to the correct SV."""
    cc = ConstraintCompiler(_ImmDecodeAction)
    cc.extract()
    cc.compute_support()  # populates cset.input_field
    # Manually render the sext expression
    sext_expr = {
        'type': 'call',
        'func': 'sext',
        'args': [
            {'type': 'subscript',
             'value': {'type': 'attribute', 'attr': 'instr'},
             'slice': {'type': 'slice',
                       'lower': {'type': 'constant', 'value': 31},
                       'upper': {'type': 'constant', 'value': 20}}},
            {'type': 'constant', 'value': 12}
        ]
    }
    sv = cc._dict_expr_to_sv(sext_expr)
    assert sv is not None, "sext expression should render to non-None"
    # Should contain sign-extension: {20{bit[11]}} pattern
    assert '20{' in sv, f"Expected sign extension fill in: {sv}"
    assert '11' in sv, f"Expected bit 11 (n-1=12-1) in: {sv}"


# ---------------------------------------------------------------------------
# OR-guard + expression RHS combined (realistic immediate decode)
# ---------------------------------------------------------------------------

@zdc.dataclass
class _FullImmDecodeAction:
    """More realistic immediate decode with OR guards + expression RHS."""
    instr : zdc.u32 = zdc.input()
    imm   : zdc.u32 = zdc.rand()
    kind  : zdc.u2  = zdc.rand()

    @constraint
    def c_itype_imm(self):
        # OR guard: opcodes 0x03 or 0x13
        if self.instr[6:0] == 0x03 or self.instr[6:0] == 0x13:
            assert self.imm == zdc.sext(self.instr[31:20], 12) & MASK32

    @constraint
    def c_utype_imm(self):
        if self.instr[6:0] == 0x37:
            assert self.imm == 0
            assert self.kind == 0

    @constraint
    def c_itype_kind(self):
        if self.instr[6:0] == 0x03:
            assert self.kind == 1

    @constraint
    def c_rtype_kind(self):
        if self.instr[6:0] == 0x33:
            assert self.kind == 2


def test_full_imm_decode_pipeline():
    """Full pipeline with OR guards + expression RHS runs without errors."""
    cc = ConstraintCompiler(_FullImmDecodeAction)
    cc.extract()
    cc.compute_support()
    cc.build_cubes()
    cc.minimize()
    sv = '\n'.join(cc.emit_sv())
    # imm uses structural mux path
    assert 'imm' in cc._expr_fields
    assert 'assign d_imm' in sv
    # kind uses SOP path (no expression assignments for kind)
    assert 'kind' not in cc._expr_fields
    assert 'd_kind' in sv


def test_or_guard_expr_produces_two_itype_blocks():
    """c_itype_imm with two OR arms produces two ConstraintBlocks."""
    cc = ConstraintCompiler(_FullImmDecodeAction)
    cc.extract()
    itype_blocks = [b for b in cc.cset.constraints if 'itype_imm' in b.name]
    assert len(itype_blocks) == 2, (
        f"Expected 2 blocks for c_itype_imm OR guard, got {len(itype_blocks)}")
    cond_vals = sorted([list(b.conditions.values())[0] for b in itype_blocks])
    assert cond_vals == [0x03, 0x13]
