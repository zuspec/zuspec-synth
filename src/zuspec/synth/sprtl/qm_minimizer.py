"""Quine-McCluskey two-level SOP minimizer.

Self-contained implementation with no external dependencies.
Uses IR types from ``..ir.constraint_ir``.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set, Tuple

from ..ir.constraint_ir import SOPCube, SharedTerm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _popcount(x: int) -> int:
    """Return the number of set bits in x."""
    return bin(x).count('1')


def _cube_key(cube: SOPCube) -> tuple:
    """Return a hashable key for a SOPCube (for deduplication / CSE)."""
    return tuple(sorted(cube.literals.items()))


def _mask_value_to_cube(mask: int, value: int, n_vars: int) -> SOPCube:
    """Convert internal (mask, value) representation to a SOPCube.

    Args:
        mask:   Bit mask; a 1-bit means the position is *not* a don't-care.
        value:  Bit values for non-don't-care positions.
        n_vars: Total number of support bits.

    Returns:
        SOPCube whose ``literals`` maps bit index → 0 | 1 | None.
    """
    literals: Dict[int, Optional[int]] = {}
    for i in range(n_vars):
        if (mask >> i) & 1:
            literals[i] = (value >> i) & 1
        else:
            literals[i] = None
    return SOPCube(literals=literals)


# ---------------------------------------------------------------------------
# Single-output minimizer
# ---------------------------------------------------------------------------

class QMMinimizer:
    """Single-output Quine-McCluskey SOP minimizer."""

    def minimize(
        self,
        ones: Set[int],
        dontcares: Set[int],
        n_vars: int,
    ) -> List[SOPCube]:
        """Return a minimal SOP cover for the given on-set.

        Args:
            ones:      Minterms where the output is 1.
            dontcares: Minterms where the output is don't-care.
            n_vars:    Number of support bits.

        Returns:
            List of SOPCubes forming a minimal (or near-minimal) SOP cover.
        """
        if not ones:
            return []

        # Safety valve: skip minimization for very large problems.
        if n_vars > 20:
            warnings.warn(
                f"QMMinimizer: n_vars={n_vars} > 20; skipping minimization "
                "and returning raw minterms.",
                stacklevel=2,
            )
            full_mask = (1 << n_vars) - 1
            return [
                _mask_value_to_cube(full_mask, m, n_vars) for m in sorted(ones)
            ]

        prime_implicants = self._generate_prime_implicants(ones, dontcares, n_vars)
        cover = self._select_cover(prime_implicants, ones, n_vars)
        return cover

    # ------------------------------------------------------------------
    # Phase 1: Prime implicant generation
    # ------------------------------------------------------------------

    def _generate_prime_implicants(
        self,
        ones: Set[int],
        dontcares: Set[int],
        n_vars: int,
    ) -> List[Tuple[int, int, frozenset]]:
        """Generate all prime implicants via iterated merging.

        Each term is represented as ``(mask, value, covered_ones)`` where:
          - ``mask``  — 1-bits for non-don't-care positions.
          - ``value`` — bit values at non-don't-care positions.
          - ``covered_ones`` — frozenset of original ``ones`` minterms covered.

        Returns:
            List of prime implicant tuples ``(mask, value, covered_ones)``.
        """
        all_minterms = ones | dontcares

        # Initialise: each minterm is its own cube.
        # Group by popcount for efficient merging.
        full_mask = (1 << n_vars) - 1

        def covered_ones_for(m: int) -> frozenset:
            return frozenset({m}) if m in ones else frozenset()

        # current_group: dict mapping (mask,value) → covered_ones set
        current: Dict[Tuple[int, int], Set[int]] = {
            (full_mask, m): set(covered_ones_for(m)) for m in all_minterms
        }

        prime_implicants: Dict[Tuple[int, int], Set[int]] = {}

        while current:
            next_round: Dict[Tuple[int, int], Set[int]] = {}
            merged: Set[Tuple[int, int]] = set()

            items = list(current.items())
            # Compare every pair; merging is O(n^2) but n is small for QM.
            for i in range(len(items)):
                (m1, v1), cov1 = items[i]
                for j in range(i + 1, len(items)):
                    (m2, v2), cov2 = items[j]
                    if m1 != m2:
                        continue
                    diff = v1 ^ v2
                    if _popcount(diff) != 1:
                        continue
                    # Merge: the differing bit becomes don't-care.
                    new_mask = m1 & ~diff
                    new_value = v1 & ~diff
                    key = (new_mask, new_value)
                    if key not in next_round:
                        next_round[key] = set()
                    next_round[key].update(cov1)
                    next_round[key].update(cov2)
                    merged.add((m1, v1))
                    merged.add((m2, v2))

            # Terms that could not be merged are prime implicants.
            for (mask, value), cov in items:
                if (mask, value) not in merged:
                    key = (mask, value)
                    if key not in prime_implicants:
                        prime_implicants[key] = set()
                    prime_implicants[key].update(cov)

            current = next_round

        # Filter: keep only PIs that cover at least one real minterm (not just DCs).
        result: List[Tuple[int, int, frozenset]] = [
            (mask, value, frozenset(cov))
            for (mask, value), cov in prime_implicants.items()
            if cov  # cov contains only ones-minterms
        ]
        return result

    # ------------------------------------------------------------------
    # Phase 2: Cover selection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cube-based entry point (avoids minterm explosion)
    # ------------------------------------------------------------------

    def minimize_from_cubes(
        self,
        input_cubes: List[Tuple[int, int]],
        n_vars: int,
    ) -> List[SOPCube]:
        """Return a minimal SOP cover from an explicit list of product-term cubes.

        This avoids minterm expansion entirely.  Each input cube is a
        ``(mask, value)`` pair where a 0-bit in *mask* means "don't-care at
        this bit position".  Input cubes that are fully subsumed by another
        input cube may be dropped; overlapping but non-subsuming cubes are
        kept (they represent distinct on-set regions).

        Args:
            input_cubes: List of ``(mask, value)`` pairs — the initial cubes.
            n_vars:      Number of support bits.

        Returns:
            List of SOPCubes forming a minimal (or near-minimal) SOP cover.
        """
        if not input_cubes:
            return []

        # Remove dominated cubes (C1 is dominated by C2 if C2 subsumes C1
        # and C2 has at least as many don't-cares).  Keeping only the
        # largest cubes reduces QM work.
        dominated: List[bool] = [False] * len(input_cubes)
        for i in range(len(input_cubes)):
            if dominated[i]:
                continue
            m1, v1 = input_cubes[i]
            for j in range(len(input_cubes)):
                if i == j or dominated[j]:
                    continue
                m2, v2 = input_cubes[j]
                # C1 is dominated by C2 iff C2 subsumes C1:
                #   every bit specified in C1 is also specified in C2 to the same value
                #   and C2 has at least as many don't-cares (i.e. is "larger or equal")
                if (m1 & m2) == m1 and (v1 & m1) == (v2 & m1) and m2 != m1:
                    dominated[i] = True
                    break

        unique: List[Tuple[int, int]] = [
            c for i, c in enumerate(input_cubes) if not dominated[i]
        ]
        # Deduplicate.
        seen: set = set()
        deduped: List[Tuple[int, int]] = []
        for c in unique:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        prime_implicants = self._generate_prime_implicants_from_cubes(deduped, n_vars)
        cover = self._select_cover_from_cubes(prime_implicants, deduped, n_vars)
        return cover

    @staticmethod
    def _cube_subsumes(pi_mask: int, pi_value: int, c_mask: int, c_value: int) -> bool:
        """Return True iff PI cube subsumes input cube C.

        PI subsumes C iff every bit specified in C is either don't-care in PI
        or has the same value as in C.
        """
        return (c_mask & pi_mask & (pi_value ^ c_value)) == 0

    def _generate_prime_implicants_from_cubes(
        self,
        cubes: List[Tuple[int, int]],
        n_vars: int,
    ) -> List[Tuple[int, int, List[int]]]:
        """Generate prime implicants starting from an explicit cube list.

        Returns list of ``(mask, value, covered_cube_indices)``.
        """
        n = len(cubes)
        if n == 0:
            return []

        # Initialise: each cube is its own term; track which input cubes it
        # covers (by index into the original `cubes` list).
        current: Dict[Tuple[int, int], Set[int]] = {}
        for idx, (mask, value) in enumerate(cubes):
            key = (mask, value)
            if key not in current:
                current[key] = set()
            current[key].add(idx)
            # Also mark any other input cube subsumed by this term.
            for jdx, (mj, vj) in enumerate(cubes):
                if self._cube_subsumes(mask, value, mj, vj):
                    current[key].add(jdx)

        prime_implicants: Dict[Tuple[int, int], Set[int]] = {}

        while current:
            next_round: Dict[Tuple[int, int], Set[int]] = {}
            merged: Set[Tuple[int, int]] = set()
            items = list(current.items())

            for i in range(len(items)):
                (m1, v1), cov1 = items[i]
                for j in range(i + 1, len(items)):
                    (m2, v2), cov2 = items[j]
                    if m1 != m2:
                        continue
                    diff = v1 ^ v2
                    if _popcount(diff) != 1:
                        continue
                    new_mask = m1 & ~diff
                    new_value = v1 & ~diff
                    key = (new_mask, new_value)
                    if key not in next_round:
                        next_round[key] = set()
                    next_round[key].update(cov1)
                    next_round[key].update(cov2)
                    # The merged cube may also subsume other input cubes.
                    for idx, (mc, vc) in enumerate(cubes):
                        if self._cube_subsumes(new_mask, new_value, mc, vc):
                            next_round[key].add(idx)
                    merged.add((m1, v1))
                    merged.add((m2, v2))

            for (mask, value), cov in items:
                if (mask, value) not in merged:
                    key = (mask, value)
                    if key not in prime_implicants:
                        prime_implicants[key] = set()
                    prime_implicants[key].update(cov)

            current = next_round

        return [
            (mask, value, sorted(cov))
            for (mask, value), cov in prime_implicants.items()
        ]

    def _select_cover_from_cubes(
        self,
        pis: List[Tuple[int, int, List[int]]],
        input_cubes: List[Tuple[int, int]],
        n_vars: int,
    ) -> List[SOPCube]:
        """Select a minimal cover ensuring every input cube is subsumed."""
        n = len(input_cubes)
        uncovered = set(range(n))
        selected: List[Tuple[int, int]] = []

        # Each PI carries a list of input-cube indices it covers.
        # Find essential PIs: the only PI covering some input cube.
        cover_table: Dict[int, List[int]] = {i: [] for i in range(n)}
        for pi_idx, (_, _, cov) in enumerate(pis):
            for cube_idx in cov:
                if cube_idx in cover_table:
                    cover_table[cube_idx].append(pi_idx)

        essential: Set[int] = set()
        for cube_idx, pi_list in cover_table.items():
            if len(pi_list) == 1:
                essential.add(pi_list[0])

        for pi_idx in essential:
            mask, value, cov = pis[pi_idx]
            selected.append((mask, value))
            uncovered -= set(cov)

        # Greedy remainder.
        while uncovered:
            best_idx = -1
            best_count = -1
            best_literals = n_vars + 1

            for pi_idx, (mask, value, cov) in enumerate(pis):
                gain = len(set(cov) & uncovered)
                if gain == 0:
                    continue
                n_lit = _popcount(mask)
                if gain > best_count or (gain == best_count and n_lit < best_literals):
                    best_idx = pi_idx
                    best_count = gain
                    best_literals = n_lit

            if best_idx == -1:
                break
            mask, value, cov = pis[best_idx]
            selected.append((mask, value))
            uncovered -= set(cov)

        return [_mask_value_to_cube(mask, value, n_vars) for mask, value in selected]

    def _select_cover(
        self,
        pis: List[Tuple[int, int, frozenset]],
        ones: Set[int],
        n_vars: int,
    ) -> List[SOPCube]:
        """Select a minimal cover from the prime implicants.

        Uses essential PI selection followed by greedy set-cover for any
        remaining uncovered minterms.

        Args:
            pis:    List of ``(mask, value, covered_ones)`` prime implicants.
            ones:   The on-set minterms that must all be covered.
            n_vars: Number of support bits (used for cube conversion).

        Returns:
            List of SOPCubes forming the selected cover.
        """
        uncovered = set(ones)
        selected: List[Tuple[int, int]] = []

        # Build cover table: minterm → list of PI indices that cover it.
        cover_table: Dict[int, List[int]] = {m: [] for m in uncovered}
        for idx, (mask, value, cov) in enumerate(pis):
            for m in cov:
                if m in cover_table:
                    cover_table[m].append(idx)

        # Find essential PIs: only PI covering a minterm.
        essential_indices: Set[int] = set()
        for m, pi_list in cover_table.items():
            if len(pi_list) == 1:
                essential_indices.add(pi_list[0])

        for idx in essential_indices:
            mask, value, cov = pis[idx]
            selected.append((mask, value))
            uncovered -= cov

        # Greedy set-cover for the remainder.
        while uncovered:
            best_idx = -1
            best_count = -1
            best_literals = n_vars + 1  # fewer literals is better (tiebreak)

            for idx, (mask, value, cov) in enumerate(pis):
                gain = len(cov & uncovered)
                if gain == 0:
                    continue
                n_literals = _popcount(mask)
                if gain > best_count or (gain == best_count and n_literals < best_literals):
                    best_idx = idx
                    best_count = gain
                    best_literals = n_literals

            if best_idx == -1:
                # Should not happen for a valid truth table.
                break

            mask, value, cov = pis[best_idx]
            selected.append((mask, value))
            uncovered -= cov

        return [_mask_value_to_cube(mask, value, n_vars) for mask, value in selected]


# ---------------------------------------------------------------------------
# Multi-output minimizer with CSE
# ---------------------------------------------------------------------------

class MultiOutputQM:
    """Multi-output SOP minimizer with common subexpression elimination."""

    def minimize_from_cube_sets(
        self,
        outputs: Dict[str, List[Tuple[int, int]]],
        n_vars: int,
    ) -> Tuple[Dict[str, List[SOPCube]], List[SharedTerm]]:
        """Minimize all outputs from per-output cube lists (no minterm expansion).

        Args:
            outputs: Mapping from output name to list of ``(mask, value)`` cubes.
            n_vars:  Number of support bits.

        Returns:
            Same shape as :meth:`minimize`.
        """
        minimizer = QMMinimizer()
        per_output: Dict[str, List[SOPCube]] = {}
        for name, cubes in outputs.items():
            per_output[name] = minimizer.minimize_from_cubes(cubes, n_vars)

        # CSE
        cube_to_outputs: Dict[tuple, List[str]] = {}
        for out_name, cubes in per_output.items():
            for cube in cubes:
                key = _cube_key(cube)
                if key not in cube_to_outputs:
                    cube_to_outputs[key] = []
                if out_name not in cube_to_outputs[key]:
                    cube_to_outputs[key].append(out_name)

        shared_terms: List[SharedTerm] = []
        wire_counter = 0
        for key, out_names in cube_to_outputs.items():
            if len(out_names) >= 2:
                cube = SOPCube(literals=dict(key))
                shared_terms.append(SharedTerm(
                    wire_name=f"w_sh{wire_counter}",
                    cube=cube,
                    used_by=list(out_names),
                ))
                wire_counter += 1

        return per_output, shared_terms

    def minimize(
    ) -> Tuple[Dict[str, List[SOPCube]], List[SharedTerm]]:
        """Minimise all outputs jointly and extract shared prime implicants.

        Args:
            outputs: Mapping from output name to ``(ones, dontcares)`` sets.
            n_vars:  Number of support bits shared by all outputs.

        Returns:
            A tuple ``(per_output_cubes, shared_terms)`` where:
              - ``per_output_cubes`` maps each output name to its SOPCube list.
              - ``shared_terms`` is a list of SharedTerm objects for cubes that
                appear in two or more outputs (CSE candidates).
        """
        minimizer = QMMinimizer()

        # Step 1: Minimise each output independently.
        per_output: Dict[str, List[SOPCube]] = {}
        for name, (ones, dcs) in outputs.items():
            per_output[name] = minimizer.minimize(ones, dcs, n_vars)

        # Step 2: CSE — find cubes that appear in ≥2 outputs.
        # Map canonical cube key → list of output names containing it.
        cube_to_outputs: Dict[tuple, List[str]] = {}
        for out_name, cubes in per_output.items():
            for cube in cubes:
                key = _cube_key(cube)
                if key not in cube_to_outputs:
                    cube_to_outputs[key] = []
                if out_name not in cube_to_outputs[key]:
                    cube_to_outputs[key].append(out_name)

        # Build SharedTerm list for shared cubes.
        shared_terms: List[SharedTerm] = []
        wire_counter = 0
        # Map cube key → wire name for already-seen shared cubes.
        shared_key_to_wire: Dict[tuple, str] = {}

        for key, out_names in cube_to_outputs.items():
            if len(out_names) >= 2:
                wire_name = f"w_sh{wire_counter}"
                wire_counter += 1
                # Reconstruct the SOPCube from the key.
                cube = SOPCube(literals=dict(key))
                shared_terms.append(SharedTerm(
                    wire_name=wire_name,
                    cube=cube,
                    used_by=list(out_names),
                ))
                shared_key_to_wire[key] = wire_name

        return per_output, shared_terms


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # 2-variable function: f(a,b) = a OR b  (minterms 1,2,3; minterm 0 is off)
    # Expected minimal cover: two prime implicants — a=1 (cube {1:1,0:X}) and b=1.
    ones = {1, 2, 3}
    dcs: Set[int] = set()

    minimizer = QMMinimizer()
    cover = minimizer.minimize(ones=ones, dontcares=dcs, n_vars=2)

    print("f(a,b) = a OR b")
    print(f"  Minterms (ones): {sorted(ones)}")
    print(f"  Prime implicant cover ({len(cover)} cube(s)):")
    for cube in cover:
        parts = []
        for bit, val in sorted(cube.literals.items()):
            if val is None:
                parts.append(f"x{bit}=-")
            else:
                parts.append(f"x{bit}={val}")
        print(f"    [{', '.join(parts)}]")

    # Multi-output example: f0 = a, f1 = b, f2 = a OR b (shares PIs with f0/f1)
    print()
    mo = MultiOutputQM()
    outputs = {
        'f0': ({2, 3}, set()),   # a=1 regardless of b → minterm 2 (10) and 3 (11)
        'f1': ({1, 3}, set()),   # b=1 regardless of a → minterm 1 (01) and 3 (11)
        'f2': ({1, 2, 3}, set()),  # a OR b
    }
    per_out, shared = mo.minimize(outputs=outputs, n_vars=2)
    print("Multi-output: f0=a, f1=b, f2=a|b")
    for name, cubes in per_out.items():
        print(f"  {name}: {len(cubes)} cube(s)")
        for c in cubes:
            parts = [f"x{b}={v if v is not None else '-'}" for b, v in sorted(c.literals.items())]
            print(f"    [{', '.join(parts)}]")
    print(f"  Shared terms: {len(shared)}")
    for st in shared:
        parts = [f"x{b}={v if v is not None else '-'}" for b, v in sorted(st.cube.literals.items())]
        print(f"    {st.wire_name}: [{', '.join(parts)}]  used_by={st.used_by}")
