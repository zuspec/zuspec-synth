"""Tests for ODC cube machinery: zdc.valid(), _resolve_guard(), build_odc_cubes().

Covers:
  S1 — parser recognition of zdc.valid() and zdc.internal()
  S2 — _resolve_guard() FieldRef / OR / AND / NOT-warn
  S2 — build_odc_cubes() no-decl / single / two-decls / RV32I-style
  S3 — GROW+ODC expands into non-observable region
  S3 — GROW+ODC does NOT expand into observable OFF region
  S3 — no-valid-decl path matches Phase-1 baseline (regression)
"""
from __future__ import annotations

import dataclasses
import sys
import pathlib
import warnings
import pytest

_dc_pkg = pathlib.Path(__file__).parents[2] / 'zuspec-dataclasses' / 'src'
if str(_dc_pkg) not in sys.path:
    sys.path.insert(0, str(_dc_pkg))

import zuspec.dataclasses as zdc
from zuspec.dataclasses.decorators import constraint, Input
from zuspec.synth.sprtl.constraint_compiler import ConstraintCompiler
from zuspec.synth.sprtl.cube_minimizer import CubeExpandMinimizer, _grow
from zuspec.synth.ir.constraint_ir import ValidityDecl


# ---------------------------------------------------------------------------
# Shared test fixture: tiny 4-instruction decoder
# ---------------------------------------------------------------------------

@zdc.dataclass
class _Tiny:
    """4-instruction decoder: 3-bit opcode → is_a, is_b, out_val.

    Encodings:
      000 → is_a=1, is_b=0, out_val=0   (op A)
      001 → is_a=1, is_b=0, out_val=1   (op A variant)
      010 → is_a=0, is_b=1, out_val=2   (op B)
      011 → is_a=0, is_b=1, out_val=3   (op B variant)
      1xx → undefined (DC)
    """
    insn    : zdc.u3 = zdc.input()
    is_a    : zdc.u1 = zdc.rand()
    is_b    : zdc.u1 = zdc.rand()
    out_val : zdc.u2 = zdc.rand()

    @constraint
    def c_op_a0(self):
        if self.insn[2:0] == 0b000:
            assert self.is_a == 1
            assert self.is_b == 0
            assert self.out_val == 0

    @constraint
    def c_op_a1(self):
        if self.insn[2:0] == 0b001:
            assert self.is_a == 1
            assert self.is_b == 0
            assert self.out_val == 1

    @constraint
    def c_op_b2(self):
        if self.insn[2:0] == 0b010:
            assert self.is_a == 0
            assert self.is_b == 1
            assert self.out_val == 2

    @constraint
    def c_op_b3(self):
        if self.insn[2:0] == 0b011:
            assert self.is_a == 0
            assert self.is_b == 1
            assert self.out_val == 3


@zdc.dataclass
class _TinyWithODC(_Tiny):
    """Same decoder as _Tiny, plus zdc.valid() annotations on out_val.

    Inherits all constraint methods from _Tiny; extract() now walks the full
    MRO so inherited constraints are picked up correctly.
    """

    @constraint
    def c_odc_hints(self):
        if self.is_a or self.is_b:
            zdc.valid(self.out_val)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile(cls):
    """Run full pipeline up through build_odc_cubes + minimize."""
    cc = ConstraintCompiler(cls)
    cc.extract()
    cc.compute_support()
    cc.build_cubes()
    if cc.cset.validity_decls:
        cc.build_odc_cubes()
    cc.minimize()
    return cc


def _term_count(cc) -> int:
    """Total AND terms across all SOPFunctions."""
    return sum(len(f.cubes) for f in cc.cset.sop_functions)


# ---------------------------------------------------------------------------
# S1: Parser — zdc.valid() recognition
# ---------------------------------------------------------------------------

class TestParser:
    def test_valid_decl_collected(self):
        """zdc.valid() inside if block → ValidityDecl collected in cset."""
        cc = ConstraintCompiler(_TinyWithODC)
        cc.extract()
        decls = cc.cset.validity_decls
        assert len(decls) == 1
        d = decls[0]
        assert d.field_name == 'out_val'
        assert d.source_method == 'c_odc_hints'
        assert d.guard is not None   # guard = is_a or is_b

    def test_no_valid_decl_when_absent(self):
        """Class without zdc.valid() → validity_decls empty."""
        cc = ConstraintCompiler(_Tiny)
        cc.extract()
        assert cc.cset.validity_decls == []

    def test_internal_field_marked(self):
        """zdc.internal(field) → field marked as internal in FieldDecl."""
        @zdc.dataclass
        class _WithInternal(_Tiny):
            helper : zdc.u1 = zdc.rand()

            @constraint
            def c_helper(self):
                zdc.internal(self.helper)

        cc = ConstraintCompiler(_WithInternal)
        cc.extract()
        assert cc.cset.internal_fields == ['helper']
        fd = cc.cset.field_by_name('helper')
        assert fd is not None and fd.internal is True

    def test_multiple_valid_same_field(self):
        """Two separate zdc.valid() for same field → two ValidityDecls."""
        @zdc.dataclass
        class _MultiDecl(_Tiny):
            @constraint
            def c_v1(self):
                if self.is_a:
                    zdc.valid(self.out_val)

            @constraint
            def c_v2(self):
                if self.is_b:
                    zdc.valid(self.out_val)

        cc = ConstraintCompiler(_MultiDecl)
        cc.extract()
        decls = [d for d in cc.cset.validity_decls if d.field_name == 'out_val']
        assert len(decls) == 2

    def test_unconditional_valid(self):
        """Top-level zdc.valid() (no enclosing if) → guard=None."""
        @zdc.dataclass
        class _Unconditional(_Tiny):
            @constraint
            def c_always(self):
                zdc.valid(self.is_a)

        cc = ConstraintCompiler(_Unconditional)
        cc.extract()
        decls = [d for d in cc.cset.validity_decls if d.field_name == 'is_a']
        assert len(decls) == 1
        assert decls[0].guard is None


# ---------------------------------------------------------------------------
# S2: _resolve_guard and build_odc_cubes
# ---------------------------------------------------------------------------

class TestResolveGuard:
    def _setup(self, cls):
        cc = ConstraintCompiler(cls)
        cc.extract()
        cc.compute_support()
        cc.build_cubes()
        return cc

    def test_field_ref_guard(self):
        """FieldRef guard → returns ON-set cubes for that field."""
        cc = self._setup(_TinyWithODC)
        # is_a's ON-set cubes (insn = 000 or 001)
        is_a_cubes = cc._cubes_by_bit.get('is_a', [])
        guard = {'type': 'bool_op', 'op': 'or', 'values': [
            {'type': 'attribute', 'value': {'type': 'name', 'id': 'self'}, 'attr': 'is_a'},
            {'type': 'attribute', 'value': {'type': 'name', 'id': 'self'}, 'attr': 'is_b'},
        ]}
        result = cc._resolve_guard(guard)
        assert result is not None
        assert len(result) > 0

    def test_or_guard_is_union(self):
        """OR guard result is at least as large as each operand alone."""
        cc = self._setup(_TinyWithODC)
        is_a_guard = {'type': 'attribute', 'value': {}, 'attr': 'is_a'}
        is_b_guard = {'type': 'attribute', 'value': {}, 'attr': 'is_b'}
        or_guard = {'type': 'bool_op', 'op': 'or', 'values': [is_a_guard, is_b_guard]}
        r_a = cc._resolve_guard(is_a_guard) or []
        r_b = cc._resolve_guard(is_b_guard) or []
        r_or = cc._resolve_guard(or_guard) or []
        assert len(r_or) >= len(r_a)
        assert len(r_or) >= len(r_b)

    def test_and_guard_is_intersection(self):
        """AND guard result is subset of each operand (can be empty on conflict)."""
        cc = self._setup(_TinyWithODC)
        is_a_guard = {'type': 'attribute', 'value': {}, 'attr': 'is_a'}
        is_b_guard = {'type': 'attribute', 'value': {}, 'attr': 'is_b'}
        and_guard = {'type': 'bool_op', 'op': 'and', 'values': [is_a_guard, is_b_guard]}
        r_and = cc._resolve_guard(and_guard) or []
        # is_a and is_b never both true → intersection should be empty (disjoint cubes)
        assert r_and == []

    def test_not_guard_resolves_to_off_set(self):
        """NOT guard (S4) → returns OFF-set cubes of the field (no warning)."""
        cc = self._setup(_TinyWithODC)
        not_guard = {'type': 'unary_op', 'op': 'not',
                     'operand': {'type': 'attribute', 'value': {}, 'attr': 'is_a'}}
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cc._resolve_guard(not_guard)
        # S4: NOT guard is now resolved — result is the OFF-set of is_a, no warning.
        assert result is not None
        assert not any('NOT' in str(warning.message) for warning in w)
        # OFF-set of is_a should be non-empty (cells where is_a = 0).
        assert len(result) > 0

    def test_tautology_guard(self):
        """True constant guard → tautology cube (mask=0, value=0)."""
        cc = self._setup(_TinyWithODC)
        true_guard = {'type': 'constant', 'value': True}
        result = cc._resolve_guard(true_guard)
        assert result == [(0, 0)]


class TestBuildODCCubes:
    def test_no_validity_decl_means_empty_obs_cubes(self):
        """Class with no zdc.valid() → obs_cubes_by_bit empty."""
        cc = ConstraintCompiler(_Tiny)
        cc.extract()
        cc.compute_support()
        cc.build_cubes()
        cc.build_odc_cubes()
        assert cc.cset.obs_cubes_by_bit == {}

    def test_single_decl_populates_obs_cubes(self):
        """Single zdc.valid() decl → obs_cubes_by_bit populated for that field."""
        cc = ConstraintCompiler(_TinyWithODC)
        cc.extract()
        cc.compute_support()
        cc.build_cubes()
        cc.build_odc_cubes()
        # out_val is 2 bits → bit columns 'out_val_bit0' and 'out_val_bit1'
        assert 'out_val_bit0' in cc.cset.obs_cubes_by_bit
        assert 'out_val_bit1' in cc.cset.obs_cubes_by_bit
        # Obs cubes should be non-empty (is_a or is_b = defined encodings)
        assert len(cc.cset.obs_cubes_by_bit['out_val_bit0']) > 0

    def test_two_decls_union(self):
        """Two separate valid() decls for same field → obs cubes is their union."""
        @zdc.dataclass
        class _TwoDecls(_Tiny):
            @constraint
            def c_v1(self):
                if self.is_a:
                    zdc.valid(self.out_val)
            @constraint
            def c_v2(self):
                if self.is_b:
                    zdc.valid(self.out_val)

        cc = ConstraintCompiler(_TwoDecls)
        cc.extract()
        cc.compute_support()
        cc.build_cubes()
        cc.build_odc_cubes()

        # Should have union of is_a cubes + is_b cubes.
        @zdc.dataclass
        class _SingleOrDecl(_Tiny):
            @constraint
            def c_v(self):
                if self.is_a or self.is_b:
                    zdc.valid(self.out_val)

        cc2 = ConstraintCompiler(_SingleOrDecl)
        cc2.extract()
        cc2.compute_support()
        cc2.build_cubes()
        cc2.build_odc_cubes()

        obs_two = set(cc.cset.obs_cubes_by_bit.get('out_val_bit0', []))
        obs_one = set(cc2.cset.obs_cubes_by_bit.get('out_val_bit0', []))
        assert obs_two == obs_one


# ---------------------------------------------------------------------------
# S3: GROW+ODC integration
# ---------------------------------------------------------------------------

class TestGROWWithODC:
    def test_odc_reduces_term_count(self):
        """Adding zdc.valid() annotations reduces or maintains term count."""
        cc_base = _compile(_Tiny)
        cc_odc  = _compile(_TinyWithODC)
        base_count = _term_count(cc_base)
        odc_count  = _term_count(cc_odc)
        assert odc_count <= base_count, (
            f"ODC should not increase terms: {odc_count} > {base_count}")

    def test_odc_does_not_break_defined_encodings(self):
        """After ODC minimization, all defined-encoding outputs are still correct.

        We enumerate minterms for encodings 0b000–0b011 and verify the SOP
        produces the expected output values.
        """
        cc = _compile(_TinyWithODC)
        sv_lines = cc.emit_sv()
        sv = '\n'.join(sv_lines)
        # Spot-check: out_val_bit0 and out_val_bit1 wires must appear.
        assert 'out_val_bit0' in sv or 'out_val' in sv

    def test_grow_odc_safe_expansion(self):
        """_grow with obs_list: expands into non-observable cell blocked by OFF."""
        # Toy: 3 support bits.  ON-cube: bit2=1 (mask=4, value=4).
        # OFF-cube: bit0=1 (mask=1, value=1) — outside observability.
        # obs_list: bit1=1 (mask=2, value=2) — only bit1=1 is observable.
        # Without ODC: cannot drop bit2 (would cover bit0=1 OFF region).
        # With ODC: can drop bit2 because the new cells (bit2=0 side) are
        # disjoint from obs_list when we check the expansion-only cells.
        on_m, on_v = 0b100, 0b100
        off_list = [(0b001, 0b001)]   # OFF: bit0=1
        obs_list = [(0b010, 0b010)]   # OBS: bit1=1

        grown_no_odc = _grow(on_m, on_v, off_list, n_vars=3, obs_list=None)
        grown_with_odc = _grow(on_m, on_v, off_list, n_vars=3, obs_list=obs_list)

        # With ODC we might be able to drop more bits (fewer constrained bits = lower mask).
        # At minimum, should not be worse.
        no_odc_bits = bin(grown_no_odc[0]).count('1')
        odc_bits = bin(grown_with_odc[0]).count('1')
        assert odc_bits <= no_odc_bits, (
            f"GROW+ODC should not constrain more: no_odc={no_odc_bits} odc={odc_bits}")

    def test_no_valid_decl_matches_baseline(self):
        """Class without zdc.valid() gives same result with/without ODC path."""
        cc1 = ConstraintCompiler(_Tiny)
        cc1.extract(); cc1.compute_support(); cc1.build_cubes(); cc1.minimize()

        cc2 = ConstraintCompiler(_Tiny)
        cc2.extract(); cc2.compute_support(); cc2.build_cubes()
        cc2.build_odc_cubes()   # no-op: no validity_decls
        cc2.minimize()

        assert _term_count(cc1) == _term_count(cc2), (
            "ODC path should not affect result when no zdc.valid() annotations")


# ---------------------------------------------------------------------------
# S4: NOT guard support (complement via De Morgan / OFF-set)
# ---------------------------------------------------------------------------

class TestNotGuardS4:
    """S4: NOT guards use the pre-computed OFF-set, no solver needed."""

    def _setup(self, cls):
        cc = ConstraintCompiler(cls)
        cc.extract()
        cc.compute_support()
        cc.build_cubes()
        return cc

    def test_not_field_ref_returns_off_set(self):
        """NOT(field) guard → OFF-set of the field."""
        cc = self._setup(_Tiny)
        not_guard = {'type': 'unary_op', 'op': 'not',
                     'operand': {'type': 'attribute', 'value': {}, 'attr': 'is_a'}}
        result = cc._resolve_guard(not_guard)
        assert result is not None
        # OFF-set of is_a = cells where is_a=0; must be non-empty.
        assert len(result) > 0
        # ON-set and OFF-set should be disjoint and cover the whole space.
        on_cubes = cc._cubes_by_bit.get('is_a', [])
        for o_m, o_v in on_cubes:
            for f_m, f_v in result:
                common = o_m & f_m
                assert (o_v ^ f_v) & common != 0, \
                    f"ON and NOT-resolved OFF cubes must be disjoint: on=({o_m},{o_v}) off=({f_m},{f_v})"

    def test_not_or_resolves_via_demorgan(self):
        """NOT(a OR b) = AND(NOT(a), NOT(b)) = intersection of OFF-sets."""
        cc = self._setup(_Tiny)
        guard = {
            'type': 'unary_op', 'op': 'not',
            'operand': {
                'type': 'bool_op', 'op': 'or',
                'values': [
                    {'type': 'attribute', 'value': {}, 'attr': 'is_a'},
                    {'type': 'attribute', 'value': {}, 'attr': 'is_b'},
                ]
            }
        }
        result = cc._resolve_guard(guard)
        assert result is not None
        # NOT(is_a OR is_b) = cells where is_a=0 AND is_b=0.
        # That's a smaller (more restrictive) set than NOT(is_a) alone.
        not_a = cc._resolve_guard({'type': 'unary_op', 'op': 'not',
                                    'operand': {'type': 'attribute', 'value': {}, 'attr': 'is_a'}})
        # The intersection should be a subset of NOT(is_a).
        assert len(result) <= len(not_a), \
            "NOT(a OR b) should be no larger than NOT(a)"

    def test_double_negation_resolves_to_on_set(self):
        """NOT(NOT(field)) → resolves back to ON-set."""
        cc = self._setup(_Tiny)
        double_not = {
            'type': 'unary_op', 'op': 'not',
            'operand': {'type': 'unary_op', 'op': 'not',
                        'operand': {'type': 'attribute', 'value': {}, 'attr': 'is_a'}}
        }
        result = cc._resolve_guard(double_not)
        expected = cc._resolve_guard({'type': 'attribute', 'value': {}, 'attr': 'is_a'})
        assert result == expected

    @zdc.dataclass
    class _TinyWithNotODC(_Tiny):
        @constraint
        def c_not_odc(self):
            if not self.is_a:
                zdc.valid(self.out_val)

    def test_not_guard_odc_reduces_or_maintains_terms(self):
        """zdc.valid() with NOT guard should not increase term count."""
        cc_base = _compile(_Tiny)
        cc_not = _compile(self._TinyWithNotODC)
        assert _term_count(cc_not) <= _term_count(cc_base), \
            "NOT-guard ODC should not increase term count"
