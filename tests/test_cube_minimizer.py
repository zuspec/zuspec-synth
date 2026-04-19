"""Unit tests for CubeExpandMinimizer (cube GROW minimizer).

Tests cover:
  1. _grow: drop bit when enlarged cube is disjoint from all OFF-cubes
  2. _grow: tautology cube when OFF-list is empty
  3. _grow: cannot drop bit when enlarged cube intersects an OFF-cube
  4. _minimize_one: 3-instruction toy decoder reduces product terms
  5. _cse: shared terms detected across outputs
  6. Full pipeline (build_cubes + minimize) on _MinimalNamed: valid SV,
     product term count ≤ QM
  7. Correctness: cover does not hit any OFF-set minterm (exhaustive)
"""
from __future__ import annotations

import dataclasses
import sys
import pathlib
import pytest

# Add zuspec-dataclasses to path (mirrors test_constraint_compiler.py setup).
_pkg = pathlib.Path(__file__).parents[2] / 'zuspec-dataclasses' / 'src'
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

import zuspec.dataclasses as zdc
from zuspec.dataclasses.decorators import constraint
from zuspec.synth.sprtl.cube_minimizer import (
    CubeExpandMinimizer,
    _disjoint,
    _grow,
    _subsumes,
)
from zuspec.synth.sprtl.constraint_compiler import ConstraintCompiler
from zuspec.synth.sprtl.qm_minimizer import MultiOutputQM
from zuspec.synth.ir.constraint_ir import SOPCube


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_cover_correct(
    on_cubes:    list,
    off_cubes:   list,
    cover_cubes: list,
    n_vars:      int,
) -> None:
    """Assert that the cover agrees with the ON/OFF sets on every minterm.

    For each minterm in 0 .. 2**n_vars - 1:
      - If the minterm is in an ON-cube, the cover must cover it.
      - If the minterm is in an OFF-cube, the cover must NOT cover it.

    Args:
        on_cubes:    List of (mask, value) ON-set cubes.
        off_cubes:   List of (mask, value) OFF-set cubes.
        cover_cubes: List of SOPCube objects forming the cover.
        n_vars:      Number of support bits.
    """
    def in_cube(m: int, mask: int, value: int) -> bool:
        return (m & mask) == (value & mask)

    for m in range(1 << n_vars):
        covered = any(cube.covers(m) for cube in cover_cubes)
        in_on  = any(in_cube(m, mask, val) for mask, val in on_cubes)
        in_off = any(in_cube(m, mask, val) for mask, val in off_cubes)

        if in_on and not in_off:
            assert covered, f"ON minterm {m:0{n_vars}b} not covered"
        if in_off:
            assert not covered, f"OFF minterm {m:0{n_vars}b} erroneously covered"


# ---------------------------------------------------------------------------
# Test 1: _grow drops a bit when safe
# ---------------------------------------------------------------------------

def test_grow_drops_safe_bit():
    """Drop bit 1 when doing so stays disjoint from the OFF-cube.

    2-bit input.  ON-cube: bit1=1, bit0=0  → (mask=0b11, value=0b10).
    OFF-cube:              bit1=1, bit0=1  → (mask=0b11, value=0b11).

    Trying bit 0: new cube (0b10, 0b10) = "bit1=1".
      Check against OFF (0b11, 0b11): m1&m2&(v1^v2) = 0b10&0b11&(0b10^0b11)
        = 0b10 & 0b01 = 0 → NOT disjoint → cannot drop bit 0.

    Trying bit 1: new cube (0b01, 0b00) = "bit0=0".
      Check against OFF (0b11, 0b11): 0b01&0b11&(0b00^0b11) = 0b01&0b11 = 0b01 ≠ 0
        → disjoint → safe to drop bit 1.

    Result: (0b01, 0b00).
    """
    result = _grow(0b11, 0b10, [(0b11, 0b11)], n_vars=2)
    assert result == (0b01, 0b00), f"Expected (0b01, 0b00), got {result}"


# ---------------------------------------------------------------------------
# Test 2: _grow produces tautology when OFF-list is empty
# ---------------------------------------------------------------------------

def test_grow_tautology_when_no_off_cubes():
    """With no OFF-cubes every bit can be dropped → tautology (0, 0)."""
    result = _grow(0b111, 0b101, [], n_vars=3)
    assert result == (0, 0), f"Expected (0, 0), got {result}"


# ---------------------------------------------------------------------------
# Test 3: _grow cannot drop a bit that would cover an OFF minterm
# ---------------------------------------------------------------------------

def test_grow_preserves_bit_conflicting_with_off():
    """Bit 0 cannot be dropped when enlarging intersects the OFF-cube.

    1-bit input.
    ON-cube:  (0b1, 0b1) = "bit0=1 → output=1".
    OFF-cube: (0b1, 0b0) = "bit0=0 → output=0".
    Dropping bit 0 would give (0, 0) = tautology, which covers the OFF minterm.
    """
    result = _grow(0b1, 0b1, [(0b1, 0b0)], n_vars=1)
    # Cannot grow at all — OFF-cube covers the complement.
    assert result == (0b1, 0b1), f"Expected (0b1, 0b1), got {result}"


# ---------------------------------------------------------------------------
# Test 4: _minimize_one on a toy 3-instruction decoder
# ---------------------------------------------------------------------------

def test_minimize_one_output_reduces_terms():
    """3-instruction decoder: GROW should collapse two ON-cubes.

    4-bit opcode input.
      LOAD  (opcode=0b0000): out_x = 1  → ON-cube  (0b1111, 0b0000)
      STORE (opcode=0b0001): out_x = 0  → OFF-cube (0b1111, 0b0001)
      ALU   (opcode=0b0010): out_x = 1  → ON-cube  (0b1111, 0b0010)

    The two ON-cubes differ only in bit 1 (0b0000 vs 0b0010) while the
    OFF-cube has bit 0 = 1.  After GROW, both ON-cubes can drop bits 2 and 3
    (safe, not constrained by OFF-cube).  The resulting PIs should be fewer
    than the 2 raw ON-cubes (or equal at worst).
    """
    on_list  = [(0b1111, 0b0000), (0b1111, 0b0010)]
    off_list = [(0b1111, 0b0001)]
    minimizer = CubeExpandMinimizer()
    result = minimizer._minimize_one(on_list, off_list, n_vars=4)

    # Must cover all ON-cubes.
    for mask, value in on_list:
        assert any(cube.covers(value) for cube in result), \
            f"ON minterm {value:04b} not covered by result"

    # Must not cover any OFF minterm.
    for mask, value in off_list:
        for m in range(1 << 4):
            if (m & mask) == value:
                assert not any(cube.covers(m) for cube in result), \
                    f"OFF minterm {m:04b} erroneously covered"

    # Result should be no worse than 2 product terms.
    assert len(result) <= 2, f"Expected ≤2 product terms, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 5: CSE detects shared terms across outputs
# ---------------------------------------------------------------------------

def test_cse_finds_shared_term():
    """Identical product term in two outputs → one SharedTerm entry."""
    # Two outputs with the same single cube: bit0=1 only.
    cube = SOPCube(literals={0: 1, 1: None})
    per_output = {
        'out_a': [cube],
        'out_b': [cube],
        'out_c': [SOPCube(literals={0: None, 1: 1})],  # different cube
    }
    shared = CubeExpandMinimizer._cse(per_output)
    assert len(shared) == 1, f"Expected 1 shared term, got {len(shared)}"
    assert set(shared[0].used_by) == {'out_a', 'out_b'}, \
        f"Wrong used_by: {shared[0].used_by}"


# ---------------------------------------------------------------------------
# Fixture: minimal 3-instruction test decode action
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


def _run_full_pipeline(cls, prefix=''):
    cc = ConstraintCompiler(cls, prefix=prefix)
    cc.extract()
    cc.compute_support()
    cc.validate(warn_only=True)
    cc.build_cubes()
    cc.minimize()
    return cc


# ---------------------------------------------------------------------------
# Test 6: Full pipeline — GROW produces valid SV, ≤ QM product terms
# ---------------------------------------------------------------------------

def test_full_pipeline_grow_beats_or_matches_qm():
    """GROW product term count must be ≤ QM's count on the toy decoder."""
    # Run GROW pipeline.
    cc_grow = _run_full_pipeline(_MinimalNamed)
    sv_lines = cc_grow.emit_sv()
    sv = '\n'.join(sv_lines)

    # Basic sanity: all output names appear in emitted SV.
    for sig in ('out_a', 'out_b', 'out_c'):
        assert sig in sv, f"Signal {sig} missing from SV output"

    # Count product terms produced by GROW (total `assign` lines for outputs).
    grow_pt = sum(1 for l in sv_lines
                  if l.strip().startswith('assign') and 'out_' in l)

    # Run QM pipeline for comparison (bypass GROW by using old MultiOutputQM
    # directly on the same cube sets).
    cc_qm = ConstraintCompiler(_MinimalNamed, prefix='')
    cc_qm.extract()
    cc_qm.compute_support()
    cc_qm.validate(warn_only=True)
    cc_qm.build_cubes()
    n = cc_qm._n_vars
    qm_per_output, _ = MultiOutputQM().minimize_from_cube_sets(
        cc_qm._cubes_by_bit, n
    )
    qm_pt = sum(len(cubes) for cubes in qm_per_output.values())

    grow_pt_total = sum(
        len(fn.cubes) for fn in cc_grow.cset.sop_functions
    )

    assert grow_pt_total <= qm_pt, (
        f"GROW produced more product terms than QM: {grow_pt_total} > {qm_pt}"
    )


# ---------------------------------------------------------------------------
# Test 7: Correctness — cover never hits an OFF-set minterm (toy decoder)
# ---------------------------------------------------------------------------

def test_cover_correctness_no_off_minterm_covered():
    """Exhaustive verification: no OFF minterm is covered by the GROW result."""
    cc = _run_full_pipeline(_MinimalNamed)
    n = cc._n_vars

    for fn in cc.cset.sop_functions:
        name = fn.output_name
        on_list  = cc._cubes_by_bit.get(name, [])
        off_list = cc._off_cubes_by_bit.get(name, [])
        _verify_cover_correct(on_list, off_list, fn.cubes, n)


# ---------------------------------------------------------------------------
# Test 8: Full pipeline on RV32I decode (integration)
# ---------------------------------------------------------------------------

def test_full_pipeline_rv32i():
    """Integration test: RV32I decoder produces ≤ 50 product terms total."""
    _examples = pathlib.Path(__file__).parents[3] / 'examples' / '04_constraints'
    if str(_examples) not in sys.path:
        sys.path.insert(0, str(_examples))

    try:
        from rv32i_decode import RV32IDecode
    except ImportError as e:
        pytest.skip(f"RV32IDecode not importable: {e}")

    cc = ConstraintCompiler(RV32IDecode, prefix='d')
    cc.extract()
    cc.compute_support()
    cc.validate(warn_only=True)
    cc.build_cubes()
    cc.minimize()

    total_pt = sum(len(fn.cubes) for fn in cc.cset.sop_functions)
    assert total_pt <= 50, (
        f"RV32I GROW produced {total_pt} product terms; expected ≤ 50"
    )

    # Correctness: no OFF minterm covered.
    n = cc._n_vars
    for fn in cc.cset.sop_functions:
        name = fn.output_name
        on_list  = cc._cubes_by_bit.get(name, [])
        off_list = cc._off_cubes_by_bit.get(name, [])
        _verify_cover_correct(on_list, off_list, fn.cubes, n)
