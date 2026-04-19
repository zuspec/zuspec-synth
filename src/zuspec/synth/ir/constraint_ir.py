"""Constraint compiler IR dataclasses.

Pure data structures used throughout the constraint-to-RTL pipeline.
No logic lives here — only the shared vocabulary between pipeline phases.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FieldDecl:
    """Declaration of one output field on the class under compilation."""
    name: str
    width: int
    soft_default: Optional[int] = None  # None → always computed; int → ODC when gating flag=0
    # DEPRECATED: soft_default is superseded by ValidityDecl / zdc.valid().
    # Retained for backward compatibility only; do not use in new code.
    internal: bool = False  # set by zdc.internal() — field excluded from port list


@dataclass
class ValidityDecl:
    """One ``zdc.valid(field)`` declaration from a constraint method.

    Records that *field_name* is observable (has a meaningful value) when
    *guard* is true.  Multiple ValidityDecls for the same field are unioned
    to form the observability ON-set; the complement is the ODC region.

    The *guard* is a parsed expression dict (same format as constraint
    antecedents) or ``None`` to indicate "always observable" (equivalent to
    no ValidityDecl but explicit).
    """
    field_name: str                   # output field this declaration applies to
    guard: Optional[Dict]             # antecedent expression dict, or None (always)
    source_method: str                # constraint method name (for diagnostics)


@dataclass(frozen=True, eq=True)
class BitRange:
    """A contiguous bit slice of the class input (e.g. insn[14:12])."""
    msb: int
    lsb: int

    def width(self) -> int:
        return self.msb - self.lsb + 1

    def var_name(self) -> str:
        """Canonical RTL wire name for this slice."""
        return f"in_{self.msb}_{self.lsb}"

    def extract(self, value: int) -> int:
        """Extract this bit range from an integer value."""
        mask = (1 << self.width()) - 1
        return (value >> self.lsb) & mask

    def __repr__(self) -> str:
        if self.msb == self.lsb:
            return f"[{self.msb}]"
        return f"[{self.msb}:{self.lsb}]"


@dataclass
class ConstraintBlock:
    """Normalised form of one @zdc.constraint method.

    ``conditions`` maps each BitRange that appears in the guard to the
    integer value it must equal.  ``assignments`` maps output field names
    to the values assigned when the guard is satisfied.
    """
    name: str                            # e.g. "c_add" — method name
    conditions: Dict[BitRange, int]      # {bit_range: required_value}
    assignments: Dict[str, int]          # {field_name: value}


@dataclass
class SOPCube:
    """One product term in a sum-of-products expression.

    ``literals`` maps a support-bit index (position in the support vector)
    to 0, 1, or None (don't-care).
    """
    literals: Dict[int, Optional[int]]   # {support_bit_index: 0 | 1 | None}

    def covers(self, minterm: int) -> bool:
        """Return True if this cube covers the given minterm index."""
        for bit_idx, val in self.literals.items():
            if val is None:
                continue
            if ((minterm >> bit_idx) & 1) != val:
                return False
        return True


@dataclass
class SOPFunction:
    """Minimised SOP for one output field (or one bit of a multi-bit field)."""
    output_name: str       # e.g. "is_alu" or "alu_op_bit2"
    cubes: List[SOPCube]


@dataclass
class SharedTerm:
    """A prime-implicant cube shared by two or more output columns (CSE)."""
    wire_name: str         # generated name, e.g. "w_sh0"
    cube: SOPCube
    used_by: List[str]     # output field names that reference this term


@dataclass
class ConstraintBlockSet:
    """All ConstraintBlocks for one class, plus supporting metadata.

    Fields are populated incrementally by pipeline phases:
      - ``constraints``      filled by Phase A (extract)
      - ``validity_decls``   filled by Phase A (extract) — from zdc.valid() calls
      - ``internal_fields``  filled by Phase A (extract) — from zdc.internal() calls
      - ``support_bits``     filled by Phase B (compute_support)
      - ``obs_cubes_by_bit`` filled by Phase D-ODC (build_odc_cubes)
      - ``sop_functions``    filled by Phase E (minimize)
      - ``shared_terms``     filled by Phase E (minimize)
    """
    input_field: str                       # name of the single input field
    input_width: int                       # bit-width of that field
    output_fields: List[FieldDecl]
    constraints: List[ConstraintBlock] = field(default_factory=list)
    validity_decls: List[ValidityDecl] = field(default_factory=list)
    internal_fields: List[str] = field(default_factory=list)
    support_bits: List[BitRange] = field(default_factory=list)
    # Per-output-bit observability cubes (mask, value) — filled by build_odc_cubes().
    # Key: bit-column name (e.g. "alu_op_bit2"); value: list of (mask, value) cubes.
    # Empty list means "always observable" (no ODC exploitation for that bit).
    obs_cubes_by_bit: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    sop_functions: List[SOPFunction] = field(default_factory=list)
    shared_terms: List[SharedTerm] = field(default_factory=list)

    def support_index(self, br: BitRange) -> int:
        """Return the position of a BitRange in the support vector."""
        return self.support_bits.index(br)

    def support_size(self) -> int:
        """Total number of support bits (sum of widths)."""
        return sum(br.width() for br in self.support_bits)

    def field_by_name(self, name: str) -> Optional[FieldDecl]:
        for f in self.output_fields:
            if f.name == name:
                return f
        return None
