"""Phase 0 – structural property analysis of a ConstraintBlockSet.

Analyses a :class:`~zuspec.synth.ir.constraint_ir.ConstraintBlockSet` and
returns a :class:`SynthesisStrategy` that downstream pipeline phases read
instead of assuming a particular implementation style (flat SOP, BDD,
priority encoding, etc.).

Checks performed
----------------
1. **Global support size** – union of all BitRange keys across all block
   guards; if > 20 bits, BDD-based synthesis is flagged.
2. **Per-field support size** – same union but restricted to blocks that
   actually assign the field.
3. **Mutual exclusion** – structural check (no solver required) for each
   pair (Ci, Cj): if they share a BitRange with conflicting required values
   they are trivially exclusive; otherwise they may overlap.
4. **Coverage check** – truth-table enumeration (only when
   global_support_bits ≤ ``_TRUTH_TABLE_BIT_LIMIT``).
5. **Passthrough detection** – identifies output bits that are exact
   copies of a support bit on all "care" input combinations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from ..ir.constraint_ir import BitRange, ConstraintBlock, ConstraintBlockSet

_LOG = logging.getLogger(__name__)

# Truth-table enumeration is O(2^n); skip it above this threshold.
_TRUTH_TABLE_BIT_LIMIT = 16


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class SynthesisStrategy:
    """Structural properties of a ConstraintBlockSet selected by PropertyAnalyzer.

    All boolean flags default to the *conservative* value so callers can
    check the flag without worrying about whether analysis was run.
    """

    # True if every pair (Ci, Cj) of blocks has an unsatisfiable combined
    # guard.  False → overlap exists → priority encoding is needed.
    mutually_exclusive: bool = False

    # True if every possible support-vector combination is covered by at
    # least one block's guard.  False → uncovered region exists and can be
    # exploited as a don't-care during minimisation.
    total_coverage: bool = False

    # Total number of distinct support bits across all block guards
    # (sum of BitRange widths in the union).
    global_support_bits: int = 0

    # For each output field name: number of support bits in the union of
    # guards of blocks that assign that field.
    per_field_support: Dict[str, int] = field(default_factory=dict)

    # True if at least one output bit is a direct wire copy of one support
    # bit on all care rows.
    has_passthrough: bool = False

    # True if global_support_bits > 20; truth-table enumeration is
    # infeasible and the compiler should use BDD-based synthesis or emit
    # a case statement instead.
    needs_bdd: bool = False

    # Populated when has_passthrough is True.
    # Maps field_name → {output_bit_index: support_bit_index}.
    passthrough_map: Dict[str, Dict[int, int]] = field(default_factory=dict)

    # Block-name pairs that are NOT structurally mutually exclusive.
    # Populated by analyze(); empty when mutually_exclusive is True.
    non_exclusive_pairs: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_support_bits(constraints: List[ConstraintBlock]) -> List[BitRange]:
    """Return the ordered union of all BitRanges used in any block guard.

    Ordering is insertion order (first appearance across constraints in list
    order) so the result is deterministic.
    """
    seen: set[BitRange] = set()
    ordered: List[BitRange] = []
    for cb in constraints:
        for br in cb.conditions:
            if br not in seen:
                seen.add(br)
                ordered.append(br)
    return ordered


def _build_br_offset(support_bits: List[BitRange]) -> Dict[BitRange, int]:
    """Map each BitRange to its LSB position in the flattened support vector.

    The support vector is a compact integer whose bits are laid out as::

        [  support_bits[0]  |  support_bits[1]  | … ]
         bits 0 .. w0-1      bits w0 .. w0+w1-1

    This allows a single integer ``sv_idx ∈ [0, 2^n)`` to represent one
    complete assignment of all support variables.
    """
    offsets: Dict[BitRange, int] = {}
    offset = 0
    for br in support_bits:
        offsets[br] = offset
        offset += br.width()
    return offsets


def _block_covers_sv(
    block: ConstraintBlock,
    sv_idx: int,
    br_offset: Dict[BitRange, int],
) -> bool:
    """Return True if every guard condition of *block* is satisfied by *sv_idx*.

    *sv_idx* is a support-vector index: bit ``br_offset[br] + k`` encodes
    bit ``k`` of BitRange ``br``.
    """
    for br, required_val in block.conditions.items():
        offset = br_offset[br]
        mask = (1 << br.width()) - 1
        actual_val = (sv_idx >> offset) & mask
        if actual_val != required_val:
            return False
    return True


def _structurally_exclusive(ci: ConstraintBlock, cj: ConstraintBlock) -> bool:
    """Return True if *ci* and *cj* are structurally mutually exclusive.

    Two blocks are structurally exclusive when there exists at least one
    BitRange that appears in **both** guards with **different** required
    values.  This is a sufficient (but not necessary) condition for mutual
    exclusion and requires no SAT/SMT solver call.
    """
    for br, val_i in ci.conditions.items():
        if br in cj.conditions and cj.conditions[br] != val_i:
            return True
    return False


# ---------------------------------------------------------------------------
# Public analyser
# ---------------------------------------------------------------------------

class PropertyAnalyzer:
    """Analyses a ConstraintBlockSet and selects the synthesis strategy.

    Usage::

        analyzer = PropertyAnalyzer()
        strategy = analyzer.analyze(cblock_set)
        if strategy.needs_bdd:
            ...
    """

    def analyze(self, cblock_set: ConstraintBlockSet) -> SynthesisStrategy:
        """Analyse *cblock_set* and return a populated :class:`SynthesisStrategy`.

        The analysis is purely structural (no solver calls).  Steps:

        1. Collect global support bits; set ``global_support_bits`` and
           ``needs_bdd``.
        2. Compute ``per_field_support`` for every output field.
        3. Check pairwise mutual exclusion using the structural heuristic.
        4. If ``global_support_bits ≤ _TRUTH_TABLE_BIT_LIMIT``: enumerate
           the truth table to determine ``total_coverage`` and detect
           passthrough wires.
        """
        strategy = SynthesisStrategy()
        constraints = cblock_set.constraints

        if not constraints:
            _LOG.warning("PropertyAnalyzer: ConstraintBlockSet has no constraints.")
            return strategy

        # ------------------------------------------------------------------
        # 1. Global support bits
        # ------------------------------------------------------------------
        # Prefer support_bits already populated by a prior pipeline phase
        # (Phase B / compute_support); derive from conditions otherwise.
        support_bits: List[BitRange] = (
            cblock_set.support_bits
            if cblock_set.support_bits
            else _collect_support_bits(constraints)
        )

        strategy.global_support_bits = sum(br.width() for br in support_bits)
        strategy.needs_bdd = strategy.global_support_bits > 20

        # ------------------------------------------------------------------
        # 2. Per-field support bits
        # ------------------------------------------------------------------
        for fd in cblock_set.output_fields:
            fname = fd.name
            field_brs: set[BitRange] = set()
            for cb in constraints:
                if fname in cb.assignments:
                    field_brs.update(cb.conditions.keys())
            strategy.per_field_support[fname] = sum(
                br.width() for br in field_brs
            )

        # ------------------------------------------------------------------
        # 3. Mutual exclusion check (structural, no solver)
        # ------------------------------------------------------------------
        if len(constraints) < 2:
            # A single block has no peers; trivially mutually exclusive.
            strategy.mutually_exclusive = True
        else:
            non_exclusive: List[Tuple[str, str]] = []
            for ci, cj in combinations(constraints, 2):
                if not _structurally_exclusive(ci, cj):
                    # Cannot confirm exclusion structurally; assume overlap.
                    non_exclusive.append((ci.name, cj.name))
                    _LOG.debug(
                        "PropertyAnalyzer: blocks %r and %r are not "
                        "structurally mutually exclusive — may need "
                        "priority encoding.",
                        ci.name,
                        cj.name,
                    )
            strategy.mutually_exclusive = len(non_exclusive) == 0
            strategy.non_exclusive_pairs = non_exclusive

        # ------------------------------------------------------------------
        # 4. Coverage and passthrough (truth-table enumeration)
        # ------------------------------------------------------------------
        n = strategy.global_support_bits
        if n > _TRUTH_TABLE_BIT_LIMIT:
            _LOG.warning(
                "PropertyAnalyzer: global_support_bits=%d > %d; "
                "skipping coverage and passthrough checks.",
                n,
                _TRUTH_TABLE_BIT_LIMIT,
            )
            return strategy

        br_offset = _build_br_offset(support_bits)
        num_minterms = 1 << n

        # --- 4a. Build the output table ---
        # output_table[sv_idx] is None if uncovered, otherwise a dict
        # mapping field_name → int value (merged across covering blocks;
        # None entry means the field has conflicting assignments).
        output_table: List[Optional[Dict[str, Optional[int]]]] = [
            None
        ] * num_minterms

        uncovered = False
        for sv_idx in range(num_minterms):
            covering = [
                cb
                for cb in constraints
                if _block_covers_sv(cb, sv_idx, br_offset)
            ]
            if not covering:
                uncovered = True
                continue  # output_table[sv_idx] remains None (don't-care)

            # Merge assignments from all covering blocks.  When two blocks
            # disagree on a field value, record None (ambiguous).
            merged: Dict[str, Optional[int]] = {}
            for cb in covering:
                for fname, val in cb.assignments.items():
                    if fname not in merged:
                        merged[fname] = val
                    elif merged[fname] != val:
                        merged[fname] = None  # conflict
            output_table[sv_idx] = merged

        strategy.total_coverage = not uncovered

        # --- 4b. Passthrough detection ---
        # Output bit *out_bit* of field *fname* is a passthrough of support
        # bit *sup_bit* when, for every care minterm, the two bits agree.
        passthrough_map: Dict[str, Dict[int, int]] = {}
        for fd in cblock_set.output_fields:
            fname = fd.name
            bit_map: Dict[int, int] = {}
            for out_bit in range(fd.width):
                for sup_bit in range(n):
                    is_pass = True
                    found_care = False
                    for sv_idx in range(num_minterms):
                        row = output_table[sv_idx]
                        if row is None or fname not in row or row[fname] is None:
                            # Uncovered / no assignment / ambiguous → skip.
                            continue
                        found_care = True
                        out_val = (row[fname] >> out_bit) & 1  # type: ignore[operator]
                        in_val = (sv_idx >> sup_bit) & 1
                        if out_val != in_val:
                            is_pass = False
                            break
                    if found_care and is_pass:
                        bit_map[out_bit] = sup_bit
                        _LOG.debug(
                            "PropertyAnalyzer: passthrough detected: "
                            "%s[%d] ← support_bit[%d]",
                            fname,
                            out_bit,
                            sup_bit,
                        )
            if bit_map:
                passthrough_map[fname] = bit_map

        if passthrough_map:
            strategy.has_passthrough = True
            strategy.passthrough_map = passthrough_map

        return strategy
