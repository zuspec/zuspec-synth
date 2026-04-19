"""Cube GROW minimizer for constraint-derived logic functions.

Implements Espresso-EXPAND-style prime implicant generation using cube algebra
rather than minterm enumeration.  This allows exploiting the large don't-care
space of decode functions (94%+ for RV32I) without enumerating DC minterms.

Key advantage over Quine-McCluskey:
  - QM merges adjacent ON-minterms; adding 1927 DC minterms causes exponential
    prime implicant explosion.
  - GROW works entirely in cube space: for each ON-cube, iteratively drop
    constrained bits as long as the enlarged cube remains disjoint from all
    explicit OFF-cubes.  The DC minterms are implicit and never enumerated.

No external solver dependency.  Pure Python cube arithmetic.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..ir.constraint_ir import SOPCube, SharedTerm
from .qm_minimizer import _cube_key, _mask_value_to_cube, _popcount


# ---------------------------------------------------------------------------
# Cube-algebra helpers
# ---------------------------------------------------------------------------

def _disjoint(m1: int, v1: int, m2: int, v2: int) -> bool:
    """Return True iff two cubes share no minterm.

    Two cubes are disjoint iff there exists a bit constrained in both cubes
    but to opposite values.  Compact formula: ``bool(m1 & m2 & (v1 ^ v2))``.

    Args:
        m1, v1: mask and value of the first cube.
        m2, v2: mask and value of the second cube.
    """
    return bool(m1 & m2 & (v1 ^ v2))


def _subsumes(pi_m: int, pi_v: int, c_m: int, c_v: int) -> bool:
    """Return True iff cube PI covers all minterms of cube C (PI subsumes C).

    PI subsumes C iff for every bit b where PI is constrained (pi_m[b]=1):
      - C is also constrained on that bit (c_m[b]=1), AND
      - they agree on the value (pi_v[b] == c_v[b]).

    In other words: PI must not constrain any bit that C leaves as don't-care,
    and on shared constrained bits the values must agree.

    Compact formula: ``(pi_m & (~c_m | (pi_v ^ c_v))) == 0``.

    Args:
        pi_m, pi_v: mask and value of the (larger) prime implicant candidate.
        c_m, c_v:   mask and value of the cube to be covered.
    """
    return (pi_m & (~c_m | (pi_v ^ c_v))) == 0


def _grow(
    mask: int,
    value: int,
    off_list: List[Tuple[int, int]],
    n_vars: int,
    obs_list: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    """Expand cube (mask, value) to its maximal prime implicant.

    Iterates over each constrained bit in LSB-to-MSB order.  If dropping
    the bit (making it don't-care) keeps the enlarged cube disjoint from
    every OFF-set cube, the bit is dropped permanently.  A dropped bit will
    never be re-considered, so the result depends on iteration order but is
    always a valid prime implicant.

    When ``obs_list`` is provided, the ODC (Observability Don't-Care) region
    is exploited: a cube may expand into any area that is either (a) disjoint
    from ``off_list`` as normal, OR (b) outside the observability region
    (``obs_list``).  Expansion into a non-observable region can never produce
    a wrong observable output, so it is always valid.

    Concretely: a candidate expanded cube is safe if it is disjoint from
    all OFF-cubes, OR if the cells it newly covers are entirely outside the
    observability region.  We check the latter by testing whether the expanded
    cube intersects any obs_list cube in the "new" cells it adds; if not, the
    expansion is into pure ODC.

    When ``off_list`` is empty and ``obs_list`` is None, every bit can be
    dropped, yielding the tautology cube ``(0, 0)``.

    Args:
        mask:     Constrained-bit mask (1 = bit is specified).
        value:    Bit values for constrained positions.
        off_list: OFF-set cubes ``(mask, value)`` for this output bit.
        n_vars:   Total number of support bits (iteration bound).
        obs_list: Observability ON-set cubes.  Cells outside this set are ODC.
                  ``None`` or empty list → always observable (standard GROW).

    Returns:
        ``(new_mask, new_value)`` — the grown prime implicant.
    """
    for bit in range(n_vars):
        if not (mask >> bit) & 1:
            continue                          # bit already don't-care
        new_mask  = mask  & ~(1 << bit)
        new_value = value & ~(1 << bit)       # bit becomes don't-care

        # Standard safety check: disjoint from all OFF cubes.
        safe = all(_disjoint(new_mask, new_value, om, ov) for om, ov in off_list)

        if not safe and obs_list:
            # ODC check: if the expansion only adds cells that are entirely
            # outside the observability region, it is safe regardless of
            # the OFF-set (the OFF values are don't-cares there).
            # "New cells" = cells covered by (new_mask, new_value) but NOT
            # by (mask, value).  These are the cells where the dropped bit
            # equals the complement of its original value.  Use XOR to flip
            # the bit so new_cell_value has the bit set to the complement.
            new_cell_mask  = new_mask  | (1 << bit)   # == mask (bit still constrained)
            new_cell_value = value ^ (1 << bit)        # flip bit → complement cells
            # Check: is every new cell outside all obs cubes?
            # i.e. is (new_cell_mask, new_cell_value) disjoint from every obs cube?
            outside_obs = all(_disjoint(new_cell_mask, new_cell_value, om, ov)
                              for om, ov in obs_list)
            if outside_obs:
                safe = True

        if safe:
            mask, value = new_mask, new_value  # safe to drop
    return mask, value


# ---------------------------------------------------------------------------
# Multi-output minimizer
# ---------------------------------------------------------------------------

class CubeExpandMinimizer:
    """Two-level SOP minimizer based on cube GROW (Espresso EXPAND).

    Unlike Quine-McCluskey, this minimizer exploits the explicit OFF-set cubes
    available from constraint blocks.  Undefined input encodings remain genuine
    don't-cares and are never enumerated.

    Interface mirrors ``MultiOutputQM.minimize_from_cube_sets()``.

    Typical usage::

        minimizer = CubeExpandMinimizer()
        per_output, shared_terms = minimizer.minimize(
            on_cubes, off_cubes, n_vars
        )
    """

    def minimize(
        self,
        on_cubes:  Dict[str, List[Tuple[int, int]]],
        off_cubes: Dict[str, List[Tuple[int, int]]],
        n_vars: int,
        obs_cubes: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> Tuple[Dict[str, List[SOPCube]], List[SharedTerm]]:
        """Minimize all outputs and return ``(per_output_cubes, shared_terms)``.

        Args:
            on_cubes:  Per-output ON-set cube lists (from ``build_cubes()``).
            off_cubes: Per-output OFF-set cube lists (from ``build_cubes()``).
            n_vars:    Number of support bits.
            obs_cubes: Optional per-output observability cube lists
                       (from ``build_odc_cubes()``).  When provided, the GROW
                       algorithm exploits the ODC region: a cube may expand
                       beyond its OFF-set boundary if the new cells are entirely
                       outside the observability region.  ``None`` disables ODC
                       (standard GROW behaviour).

        Returns:
            A tuple ``(per_output_cubes, shared_terms)`` where
            ``per_output_cubes`` maps each output name to its ``SOPCube`` list
            and ``shared_terms`` lists cubes shared by two or more outputs.
        """
        per_output: Dict[str, List[SOPCube]] = {}
        for name, on_list in on_cubes.items():
            off_list = off_cubes.get(name, [])
            obs_list = obs_cubes.get(name) if obs_cubes else None
            per_output[name] = self._minimize_one(on_list, off_list, n_vars, obs_list)

        shared_terms = self._cse(per_output)
        return per_output, shared_terms

    # ------------------------------------------------------------------
    # Per-output minimization
    # ------------------------------------------------------------------

    def _minimize_one(
        self,
        on_list:  List[Tuple[int, int]],
        off_list: List[Tuple[int, int]],
        n_vars:   int,
        obs_list: Optional[List[Tuple[int, int]]] = None,
    ) -> List[SOPCube]:
        """Minimize one output bit: GROW → deduplicate → cover.

        Steps:
          1a. Phase-1 GROW without ODC: each ON minterm expands to a prime
              implicant using only the standard OFF-set constraint.
          1b. Phase-2 GROW with ODC (if obs_list provided): each Phase-1 PI
              is grown further by allowing expansion into non-observable cells.
              Running Phase 2 on already-merged PIs avoids the pitfall where
              different minterms would expand in incompatible directions.
          2. Deduplicate identical grown PIs.
          3. Remove dominated PIs (subsumed by a larger sibling).
          4. Select a minimum cover (essential PIs first, then greedy).

        Args:
            on_list:  ON-set cubes for this output bit.
            off_list: OFF-set cubes for this output bit.
            n_vars:   Total support bits.
            obs_list: Observability cubes for this output bit.  ``None`` →
                      always observable (standard GROW, no ODC exploitation).

        Returns:
            List of ``SOPCube`` objects forming a minimal SOP cover.
        """
        if not on_list:
            return []

        # Step 1a: Phase-1 GROW without ODC → stable prime implicants.
        # This ensures different ON minterms converge to the same cube before
        # ODC expansion, avoiding incompatible multi-directional expansions.
        phase1_pis: List[Tuple[int, int]] = [
            _grow(m, v, off_list, n_vars, None) for m, v in on_list
        ]

        # Step 1b: If ODC is enabled, grow each Phase-1 PI further using ODC.
        if obs_list:
            grown: List[Tuple[int, int]] = [
                _grow(m, v, off_list, n_vars, obs_list) for m, v in phase1_pis
            ]
        else:
            grown = phase1_pis

        # Step 2: Deduplicate.
        seen: set = set()
        unique_pis: List[Tuple[int, int]] = []
        for g in grown:
            if g not in seen:
                seen.add(g)
                unique_pis.append(g)

        # Step 3: Remove dominated PIs.
        unique_pis = self._remove_dominated(unique_pis)

        # Step 4: Greedy essential-first cover.
        selected = self._greedy_cover(unique_pis, on_list)

        return [_mask_value_to_cube(m, v, n_vars) for m, v in selected]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_dominated(
        pis: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Remove PIs dominated by a strictly larger sibling.

        PI_i is dominated iff there exists PI_j such that:
          - PI_j subsumes PI_i (every minterm of PI_i is also in PI_j), and
          - PI_j is strictly larger (fewer constrained bits).

        Removing dominated PIs reduces cover candidates without loss —
        a dominating PI is always at least as good a cover choice.

        Args:
            pis: List of ``(mask, value)`` prime implicants.

        Returns:
            Filtered list with dominated PIs removed.
        """
        n = len(pis)
        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            mi, vi = pis[i]
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                mj, vj = pis[j]
                # PI_j subsumes PI_i and is strictly larger (less constrained).
                if _subsumes(mj, vj, mi, vi) and _popcount(mj) < _popcount(mi):
                    dominated[i] = True
                    break
        return [p for k, p in enumerate(pis) if not dominated[k]]

    @staticmethod
    def _greedy_cover(
        pis:     List[Tuple[int, int]],
        on_list: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Select a minimal cover of ``on_list`` using ``pis``.

        Each original ON-cube must be subsumed by at least one selected PI.
        Uses essential-PI-first selection then greedy set-cover for any
        remaining uncovered ON-cubes.

        Tiebreaker: among PIs with equal coverage gain, prefer the one with
        fewer constrained bits (wider cube → simpler AND gate in RTL).

        Args:
            pis:     Prime implicant candidates (grown cubes).
            on_list: Original ON-set cubes that must all be covered.

        Returns:
            Subset of ``pis`` forming the selected cover.
        """
        n = len(on_list)
        if n == 0:
            return []

        # Pre-compute which ON-cubes each PI covers.
        pi_covers: List[List[int]] = []
        for pm, pv in pis:
            covers = [
                i for i, (om, ov) in enumerate(on_list)
                if _subsumes(pm, pv, om, ov)
            ]
            pi_covers.append(covers)

        uncovered = set(range(n))
        selected: List[Tuple[int, int]] = []

        # Build reverse map: ON-cube index → list of PI indices that cover it.
        cube_to_pis: Dict[int, List[int]] = {i: [] for i in range(n)}
        for pi_idx, covers in enumerate(pi_covers):
            for cube_idx in covers:
                cube_to_pis[cube_idx].append(pi_idx)

        # Essential PIs: the sole cover for at least one ON-cube.
        essential: set = set()
        for cube_idx, pi_list in cube_to_pis.items():
            if len(pi_list) == 1:
                essential.add(pi_list[0])

        for pi_idx in sorted(essential):  # sorted for deterministic output
            pm, pv = pis[pi_idx]
            selected.append((pm, pv))
            uncovered -= set(pi_covers[pi_idx])

        # Greedy remainder.
        while uncovered:
            best_idx = -1
            best_gain = -1
            best_lits = 999

            for pi_idx, (pm, pv) in enumerate(pis):
                gain = len(set(pi_covers[pi_idx]) & uncovered)
                if gain == 0:
                    continue
                lits = _popcount(pm)
                if gain > best_gain or (gain == best_gain and lits < best_lits):
                    best_idx  = pi_idx
                    best_gain = gain
                    best_lits = lits

            if best_idx == -1:
                break  # should not happen for a valid (consistent) problem
            pm, pv = pis[best_idx]
            selected.append((pm, pv))
            uncovered -= set(pi_covers[best_idx])

        return selected

    @staticmethod
    def _cse(
        per_output: Dict[str, List[SOPCube]],
    ) -> List[SharedTerm]:
        """Find cubes appearing in ≥2 outputs and build SharedTerm list.

        Identical product terms across multiple output bits can be computed
        once and shared (one AND gate, multiple OR inputs).

        Args:
            per_output: Mapping from output name to its ``SOPCube`` list.

        Returns:
            List of ``SharedTerm`` objects for cubes used by 2+ outputs.
        """
        cube_to_outputs: Dict[tuple, List[str]] = {}
        for out_name, cubes in per_output.items():
            for cube in cubes:
                key = _cube_key(cube)
                cube_to_outputs.setdefault(key, [])
                if out_name not in cube_to_outputs[key]:
                    cube_to_outputs[key].append(out_name)

        shared: List[SharedTerm] = []
        wire_counter = 0
        for key, names in cube_to_outputs.items():
            if len(names) >= 2:
                shared.append(SharedTerm(
                    wire_name=f"w_sh{wire_counter}",
                    cube=SOPCube(literals=dict(key)),
                    used_by=list(names),
                ))
                wire_counter += 1
        return shared
