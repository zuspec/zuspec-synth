"""Constraint-to-RTL pipeline: orchestrates Phases A–F.

Phase A  extract()         — parse @constraint methods → ConstraintBlockSet
Phase B  compute_support() — union BitRanges → support_bits
Phase C  validate()        — mutual-exclusion / coverage checks
Phase D  build_table()     — enumerate minterms → per-output truth tables
Phase E  minimize()        — SOP minimization via MultiOutputQM
Phase F  emit_sv()         — emit SystemVerilog wire assignments
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from zuspec.dataclasses.constraint_parser import ConstraintParser, extract_rand_fields
from ..ir.constraint_ir import (
    BitRange, ConstraintBlock, ConstraintBlockSet, FieldDecl,
    SOPCube, SOPFunction, SharedTerm,
)
from .qm_minimizer import MultiOutputQM

log = logging.getLogger(__name__)


class ConstraintValidationError(Exception):
    pass


class ConstraintCompiler:
    """Orchestrate the constraint-to-RTL pipeline for one action class.

    Typical usage::

        cc = ConstraintCompiler(MyDecodeAction, prefix='d')
        cc.extract()
        cc.compute_support()
        cc.validate(warn_only=True)
        cc.build_table()
        cc.minimize()
        lines = cc.emit_sv()
    """

    def __init__(self, action_cls: type, prefix: str = 'd'):
        """
        Args:
            action_cls: Python class with @zdc.constraint methods and rand() fields.
            prefix:     Wire name prefix for emitted RTL (e.g. 'd' → wires named d_is_alu).
        """
        self._cls = action_cls
        self._prefix = prefix
        self.cset: Optional[ConstraintBlockSet] = None
        self._per_field_support: Dict[str, List[BitRange]] = {}
        self._ones_by_bit: Dict[str, Set[int]] = {}
        self._dontcares_by_bit: Dict[str, Set[int]] = {}
        self._n_vars: int = 0
        self.strategy = None

    # ------------------------------------------------------------------
    # Phase A — extract
    # ------------------------------------------------------------------

    def extract(self) -> None:
        """Walk @_is_constraint methods, build ConstraintBlockSet."""
        parser = ConstraintParser()
        blocks: List[ConstraintBlock] = []
        cond_field_names: Set[str] = set()

        for attr_name, value in vars(self._cls).items():
            if not (callable(value) and getattr(value, '_is_constraint', False)):
                continue
            try:
                parsed = parser.parse_constraint(value)
            except Exception as exc:
                log.warning("Skipping constraint %s.%s: %s", self._cls.__name__, attr_name, exc)
                continue
            for expr in parsed.get('exprs', []):
                block = self._build_block(attr_name, expr)
                if block is not None:
                    blocks.append(block)
                for fname in self._collect_subscript_fields(
                        expr.get('antecedent', {}) if expr.get('type') == 'implies' else {}):
                    cond_field_names.add(fname)

        # Determine output fields (rand-marked) and detect the input field.
        output_fields: List[FieldDecl] = []
        input_field_name: Optional[str] = None
        input_field_width: int = 32  # sensible default for instruction words

        if dataclasses.is_dataclass(self._cls):
            for f in dataclasses.fields(self._cls):
                meta = f.metadata
                # extract_rand_fields also handles metadata stored in field.type
                if not meta and isinstance(getattr(f, 'type', None), dataclasses.Field):
                    meta = f.type.metadata  # type: ignore[union-attr]

                if meta.get('rand') or meta.get('randc'):
                    width = meta.get('width', 1)
                    soft = meta.get('soft_default', None)
                    output_fields.append(FieldDecl(name=f.name, width=width, soft_default=soft))
                else:
                    # Candidate input: the non-rand field used in subscript conditions.
                    if f.name in cond_field_names and input_field_name is None:
                        input_field_name = f.name
                        w = meta.get('width', None) if meta else None
                        if isinstance(w, int):
                            input_field_width = w

        # Fall back: pick the most commonly referenced field name from conditions.
        if input_field_name is None and cond_field_names:
            input_field_name = next(iter(sorted(cond_field_names)))

        if input_field_name is None:
            input_field_name = 'insn'

        # Infer input width from the highest MSB seen if still at the default.
        if input_field_width == 32:
            max_msb = max((br.msb for b in blocks for br in b.conditions), default=31)
            if max_msb > 0:
                input_field_width = max_msb + 1

        self.cset = ConstraintBlockSet(
            input_field=input_field_name,
            input_width=input_field_width,
            output_fields=output_fields,
            constraints=blocks,
        )

    # -- Phase A helpers ---------------------------------------------------

    def _build_block(self, method_name: str, expr: Dict[str, Any]) -> Optional[ConstraintBlock]:
        """Convert a single 'implies' expression dict into a ConstraintBlock."""
        if expr.get('type') != 'implies':
            return None

        antecedent = expr.get('antecedent', {})
        consequent = expr.get('consequent', [])
        # consequent may be a list (from if-statement) or a single dict (from implies() call)
        if isinstance(consequent, dict):
            consequent = [consequent]

        conditions = self._parse_conditions(antecedent)
        if not conditions:
            return None

        assignments: Dict[str, int] = {}
        for cons in consequent:
            fname, val = self._parse_assignment(cons)
            if fname is not None:
                assignments[fname] = val

        if not assignments:
            return None

        return ConstraintBlock(name=method_name, conditions=conditions, assignments=assignments)

    def _parse_conditions(self, node: Dict[str, Any]) -> Dict[BitRange, int]:
        """Recursively parse an antecedent AST node → {BitRange: required_value}."""
        t = node.get('type')
        if t == 'compare':
            ops = node.get('ops', [])
            comps = node.get('comparators', [])
            if ops == ['=='] and comps:
                br = self._extract_bitrange(node.get('left', {}))
                val = self._extract_int(comps[0])
                if br is not None and val is not None:
                    return {br: val}
        elif t == 'bool_op' and node.get('op') == 'and':
            result: Dict[BitRange, int] = {}
            for child in node.get('values', []):
                result.update(self._parse_conditions(child))
            return result
        return {}

    def _extract_bitrange(self, node: Dict[str, Any]) -> Optional[BitRange]:
        """Extract a BitRange from a subscript AST node.

        For hardware bit-slice notation written as ``insn[6:0]``, Python's AST
        gives ``ast.Slice(lower=6, upper=0)`` — the *left* operand (lower) is
        the MSB and the *right* operand (upper) is the LSB.
        """
        if node.get('type') != 'subscript':
            return None
        sl = node.get('slice', {})
        if sl.get('type') == 'slice':
            lower = sl.get('lower')
            upper = sl.get('upper')
            # In insn[MSB:LSB], Python slice lower=MSB, upper=LSB.
            msb = self._extract_int(lower) if lower else None
            lsb = self._extract_int(upper) if upper else None
            if msb is None and lsb is None:
                return None
            if msb is None:
                msb = lsb
            if lsb is None:
                lsb = msb
            return BitRange(msb=msb, lsb=lsb)
        elif sl.get('type') == 'index':
            idx = self._extract_int(sl.get('value', {}))
            if idx is not None:
                return BitRange(msb=idx, lsb=idx)
        return None

    def _extract_int(self, node: Dict[str, Any]) -> Optional[int]:
        """Extract a Python int from a constant or name AST node.

        Handles:
          - ``{'type': 'constant', 'value': N}``  — inline integer literal.
          - ``{'type': 'name', 'id': 'SOME_CONST'}`` — module-level name;
            resolved via the action class's module globals.
        """
        if node.get('type') == 'constant':
            v = node.get('value')
            if isinstance(v, int):
                return v
        if node.get('type') == 'name':
            name = node.get('id', '')
            # Try to resolve via the action class's module globals.
            import sys
            mod = sys.modules.get(getattr(self._cls, '__module__', ''), None)
            if mod is not None:
                v = getattr(mod, name, None)
                if isinstance(v, int):
                    return v
            # Fallback: search calling frames for the name.
            import inspect
            frame = inspect.currentframe()
            while frame is not None:
                if name in frame.f_globals and isinstance(frame.f_globals[name], int):
                    return frame.f_globals[name]
                if name in frame.f_locals and isinstance(frame.f_locals[name], int):
                    return frame.f_locals[name]
                frame = frame.f_back
        return None

    def _parse_assignment(self, node: Dict[str, Any]) -> Tuple[Optional[str], int]:
        """Parse `self.field == value` → (field_name, value)."""
        if node.get('type') != 'compare':
            return None, 0
        ops = node.get('ops', [])
        comps = node.get('comparators', [])
        if ops != ['=='] or not comps:
            return None, 0
        val = self._extract_int(comps[0])
        if val is None:
            return None, 0
        left = node.get('left', {})
        if left.get('type') == 'attribute':
            return left.get('attr'), val
        if left.get('type') == 'name':
            return left.get('id'), val
        return None, 0

    def _collect_subscript_fields(self, node: Dict[str, Any]) -> List[str]:
        """Return field names referenced in subscript nodes (recursively)."""
        t = node.get('type')
        if t == 'subscript':
            val = node.get('value', {})
            vt = val.get('type')
            if vt == 'attribute':
                return [val.get('attr', '')]
            if vt == 'name':
                return [val.get('id', '')]
        elif t in ('bool_op',):
            names: List[str] = []
            for child in node.get('values', []):
                names.extend(self._collect_subscript_fields(child))
            return names
        elif t == 'compare':
            return self._collect_subscript_fields(node.get('left', {}))
        return []

    # ------------------------------------------------------------------
    # Phase B — compute_support
    # ------------------------------------------------------------------

    def compute_support(self) -> None:
        """Compute union of all BitRanges appearing in block conditions."""
        assert self.cset is not None, "Call extract() first"

        seen: Set[BitRange] = set()
        for block in self.cset.constraints:
            seen.update(block.conditions.keys())
        self.cset.support_bits = sorted(seen, key=lambda br: (br.lsb, br.msb))

        # Per-output-field support: which BitRanges guard blocks that assign that field.
        for fd in self.cset.output_fields:
            field_brs: Set[BitRange] = set()
            for block in self.cset.constraints:
                if fd.name in block.assignments:
                    field_brs.update(block.conditions.keys())
            self._per_field_support[fd.name] = sorted(
                field_brs, key=lambda br: (br.lsb, br.msb))

    # ------------------------------------------------------------------
    # Phase C — validate
    # ------------------------------------------------------------------

    def validate(self, warn_only: bool = False) -> List[str]:
        """Mutual-exclusion and coverage checks via PropertyAnalyzer.

        Returns a list of issue strings.  Raises ConstraintValidationError
        unless warn_only=True.
        """
        assert self.cset is not None, "Call extract() first"
        issues: List[str] = []

        try:
            from .property_analysis import PropertyAnalyzer  # type: ignore[import]
            analyzer = PropertyAnalyzer()
            self.strategy = analyzer.analyze(self.cset)
            if hasattr(self.strategy, 'issues'):
                issues.extend(self.strategy.issues)
            if hasattr(self.strategy, 'mutually_exclusive') and not self.strategy.mutually_exclusive:
                issues.append("Constraint blocks are not mutually exclusive")
        except ImportError:
            log.warning("property_analysis not available — skipping formal validation")
            return issues

        for issue in issues:
            log.warning("Constraint issue: %s", issue)
        if issues and not warn_only:
            raise ConstraintValidationError(
                "Constraint validation failed:\n" + "\n".join(issues))
        return issues

    # ------------------------------------------------------------------
    # Phase D — build_table
    # ------------------------------------------------------------------

    def build_table(self) -> None:
        """Build per-bit truth tables using sparse minterm enumeration.

        Instead of iterating all 2^n minterms (impractical for n > ~16),
        we enumerate only the minterms that each block's conditions can
        produce.  For a block with conditions on k bits out of n total
        support bits, there are 2^(n-k) matching minterms — one for each
        combination of the *unconstrained* support bits.

        Everything not covered by any block is implicitly don't-care; the
        QM minimizer receives an empty dontcares set (it will produce a
        minimal cover for the ones set, treating everything else as DC).
        """
        assert self.cset is not None, "Call compute_support() first"
        support = self.cset.support_bits

        # Flat support vector: one entry per individual bit, LSB-first within
        # each BitRange, then in support order.
        flat_bits: List[Tuple[BitRange, int]] = []
        for br in support:
            for bit_offset in range(br.width()):
                flat_bits.append((br, bit_offset))

        n = len(flat_bits)
        self._n_vars = n

        # Pre-build a lookup {(BitRange, bit_offset): flat_index} for speed.
        flat_bit_map: Dict[Tuple[BitRange, int], int] = {
            entry: i for i, entry in enumerate(flat_bits)
        }

        def bit_col(fname: str, bit: int, width: int) -> str:
            return fname if width == 1 else f"{fname}_bit{bit}"

        # Initialise per-bit truth-table sets (dontcares always empty here).
        ones_by_bit: Dict[str, Set[int]] = {}
        dcs_by_bit: Dict[str, Set[int]] = {}
        for fd in self.cset.output_fields:
            for b in range(fd.width):
                col = bit_col(fd.name, b, fd.width)
                ones_by_bit[col] = set()
                dcs_by_bit[col] = set()

        # For each block, compute the base minterm index (constrained bits set)
        # and the mask of free (unconstrained) bit positions.  Then iterate
        # over all 2^(#free) combinations to produce all matching minterms.
        for block in self.cset.constraints:
            # Build base index and constrained-bit mask.
            base_idx = 0
            constrained_mask = 0
            for br, req_val in block.conditions.items():
                for offset in range(br.width()):
                    flat_idx = flat_bit_map[(br, offset)]
                    req_bit = (req_val >> offset) & 1
                    base_idx |= req_bit << flat_idx
                    constrained_mask |= 1 << flat_idx

            free_bits = [i for i in range(n) if not (constrained_mask >> i) & 1]
            n_free = len(free_bits)

            for free_combo in range(1 << n_free):
                idx = base_idx
                for k, fi in enumerate(free_bits):
                    if (free_combo >> k) & 1:
                        idx |= 1 << fi

                for fd in self.cset.output_fields:
                    assigned_val = block.assignments.get(fd.name)
                    for b in range(fd.width):
                        col = bit_col(fd.name, b, fd.width)
                        if assigned_val is None:
                            # Field not assigned by this block — don't-care.
                            dcs_by_bit[col].add(idx)
                        elif (assigned_val >> b) & 1:
                            ones_by_bit[col].add(idx)
                        # else: zero — leave out of ones (zero is the default)

        self._ones_by_bit = ones_by_bit
        self._dontcares_by_bit = dcs_by_bit

    # ------------------------------------------------------------------
    # Phase D-alt — build_cubes  (fast path, replaces build_table)
    # ------------------------------------------------------------------

    def build_cubes(self) -> None:
        """Build per-output cube lists directly from constraint blocks.

        This avoids all minterm enumeration.  Each block's conditions are
        converted to a single (mask, value) cube over the flat support vector.
        The cubes are grouped by output bit; for each output bit we record:
          - ``_cubes_by_bit[col]`` — cubes for which this bit is driven to 1.
          - ``_zero_cubes_by_bit[col]`` — cubes that explicitly drive this bit
            to 0 (not currently used by QM, kept for future ODC).

        Unconstrained outputs are naturally don't-care by omission.
        """
        assert self.cset is not None, "Call compute_support() first"
        support = self.cset.support_bits

        flat_bits: List[Tuple[BitRange, int]] = []
        for br in support:
            for bit_offset in range(br.width()):
                flat_bits.append((br, bit_offset))

        n = len(flat_bits)
        self._n_vars = n

        flat_bit_map: Dict[Tuple[BitRange, int], int] = {
            entry: i for i, entry in enumerate(flat_bits)
        }

        def bit_col(fname: str, bit: int, width: int) -> str:
            return fname if width == 1 else f"{fname}_bit{bit}"

        cubes_by_bit: Dict[str, List[Tuple[int, int]]] = {}
        for fd in self.cset.output_fields:
            for b in range(fd.width):
                cubes_by_bit[bit_col(fd.name, b, fd.width)] = []

        for block in self.cset.constraints:
            # Build the (mask, value) cube for this block's conditions.
            cube_mask = 0
            cube_value = 0
            for br, req_val in block.conditions.items():
                for offset in range(br.width()):
                    flat_idx = flat_bit_map[(br, offset)]
                    req_bit = (req_val >> offset) & 1
                    cube_mask |= 1 << flat_idx
                    cube_value |= req_bit << flat_idx

            # Distribute to output bits.
            for fd in self.cset.output_fields:
                assigned_val = block.assignments.get(fd.name)
                if assigned_val is None:
                    continue  # Not assigned — don't add a cube.
                for b in range(fd.width):
                    if (assigned_val >> b) & 1:
                        col = bit_col(fd.name, b, fd.width)
                        cubes_by_bit[col].append((cube_mask, cube_value))

        self._cubes_by_bit = cubes_by_bit

    # ------------------------------------------------------------------
    # Phase E — minimize
    # ------------------------------------------------------------------

    def minimize(self) -> None:
        """Run SOP minimization via MultiOutputQM and store results in cset."""
        n = self._n_vars or self.cset.support_size()

        if hasattr(self, '_cubes_by_bit') and self._cubes_by_bit is not None:
            # Fast cube-based path (preferred): no minterm expansion.
            per_output_cubes, shared_terms = MultiOutputQM().minimize_from_cube_sets(
                self._cubes_by_bit, n
            )
        else:
            # Fallback: minterm-based path (for small support sizes).
            assert hasattr(self, '_ones_by_bit'), "Call build_table() or build_cubes() first"
            outputs: Dict[str, Tuple[Set[int], Set[int]]] = {
                name: (ones, self._dontcares_by_bit[name])
                for name, ones in self._ones_by_bit.items()
            }
            per_output_cubes, shared_terms = MultiOutputQM().minimize(outputs, n)

        self.cset.sop_functions = [
            SOPFunction(output_name=name, cubes=cubes)
            for name, cubes in per_output_cubes.items()
        ]
        self.cset.shared_terms = shared_terms

    # ------------------------------------------------------------------
    # Phase F — emit_sv
    # ------------------------------------------------------------------

    def emit_sv(self) -> List[str]:
        """Emit a list of SystemVerilog lines for the compiled constraint logic."""
        assert self.cset is not None, "Call minimize() first"
        cs = self.cset
        lines: List[str] = []

        lines.append(
            f"// Generated by ConstraintCompiler — {len(cs.constraints)} blocks, "
            f"{len(cs.support_bits)} support bits"
        )

        # Support wire declarations.
        p = self._prefix
        input_sig = f"{p}_{cs.input_field}" if p else cs.input_field
        for br in cs.support_bits:
            vn = br.var_name()
            if br.msb == br.lsb:
                lines.append(f"wire {vn} = {input_sig}[{br.msb}];")
            else:
                lines.append(
                    f"wire [{br.msb}:{br.lsb}] {vn} = "
                    f"{input_sig}[{br.msb}:{br.lsb}];"
                )

        # Rebuild flat_bits (same ordering as build_table).
        flat_bits: List[Tuple[BitRange, int]] = [
            (br, b_offset)
            for br in cs.support_bits
            for b_offset in range(br.width())
        ]

        def cube_to_sv(cube: SOPCube) -> str:
            """Render one product term to a SV expression string."""
            terms: List[str] = []
            for bit_idx, val in sorted(cube.literals.items()):
                if val is None:
                    continue
                br, b_within = flat_bits[bit_idx]
                wire_ref = br.var_name() if br.width() == 1 else f"{br.var_name()}[{b_within}]"
                terms.append(f"~{wire_ref}" if val == 0 else wire_ref)
            return " & ".join(terms) if terms else "1'b1"

        # Map (frozen literal items) → shared wire name for CSE substitution.
        shared_key_to_wire: Dict[tuple, str] = {
            tuple(sorted(st.cube.literals.items())): st.wire_name
            for st in cs.shared_terms
        }

        def sop_expr(cubes: List[SOPCube]) -> str:
            """Render a list of cubes to a SV SOP expression."""
            if not cubes:
                return "1'b0"
            parts: List[str] = []
            for cube in cubes:
                key = tuple(sorted(cube.literals.items()))
                if key in shared_key_to_wire:
                    parts.append(shared_key_to_wire[key])
                else:
                    parts.append(f"({cube_to_sv(cube)})")
            return " | ".join(parts)

        # Shared-term wire declarations (CSE).
        for st in cs.shared_terms:
            lines.append(f"wire {st.wire_name} = ({cube_to_sv(st.cube)});")

        # SOP function index.
        sop_by_name: Dict[str, SOPFunction] = {
            sf.output_name: sf for sf in cs.sop_functions
        }

        def bit_col(fname: str, bit: int, width: int) -> str:
            return fname if width == 1 else f"{fname}_bit{bit}"

        # Build a set of 1-bit output field names for gating-flag lookup.
        _flag_names = {fd.name for fd in cs.output_fields if fd.width == 1}

        def gating_flag(fd: FieldDecl) -> Optional[str]:
            """Return the RTL name of the gating flag for a soft-default field, or None.

            1-bit flags gate themselves (no ODC wrapping needed — they ARE the flags).
            For a multi-bit field named e.g. 'alu_op', look for a 1-bit field named
            'is_alu' or 'is_alu_op' in the output fields.  If none found, return None
            and the field is emitted without ODC gating.
            """
            if fd.width == 1:
                return None  # type flags don't ODC-gate themselves
            # Try progressively shorter prefixes of the field name.
            for candidate in (f"is_{fd.name}", f"is_{fd.name.split('_')[0]}"):
                if candidate in _flag_names:
                    return f"{p}_{candidate}"
            return None

        # Output field assignments.
        for fd in cs.output_fields:
            # Passthrough override from synthesis strategy.
            if (self.strategy is not None
                    and hasattr(self.strategy, 'passthroughs')
                    and fd.name in self.strategy.passthroughs):
                wire = self.strategy.passthroughs[fd.name]
                lines.append(f"wire {p}_{fd.name};")
                lines.append(f"assign {p}_{fd.name} = {wire};")
                continue

            if fd.width == 1:
                sop_fn = sop_by_name.get(fd.name)
                expr = sop_expr(sop_fn.cubes if sop_fn else [])
                lines.append(f"wire {p}_{fd.name};")
                lines.append(f"assign {p}_{fd.name} = {expr};")
            else:
                # Multi-bit field: declare the wire, then assign.
                lines.append(f"wire [{fd.width-1}:0] {p}_{fd.name};")
                # Decompose into bit columns, MSB first for concatenation.
                bit_exprs: List[str] = []
                for b in range(fd.width - 1, -1, -1):
                    col = bit_col(fd.name, b, fd.width)
                    sop_fn = sop_by_name.get(col)
                    bit_exprs.append(sop_expr(sop_fn.cubes if sop_fn else []))
                concat = "{" + ", ".join(bit_exprs) + "}"
                flag = gating_flag(fd) if fd.soft_default is not None else None
                if flag:
                    lines.append(
                        f"assign {p}_{fd.name} = {flag} ? "
                        f"{concat} : {fd.width}'bx;"
                    )
                else:
                    lines.append(f"assign {p}_{fd.name} = {concat};")

        return lines
