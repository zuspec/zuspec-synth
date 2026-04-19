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
import typing
from typing import Any, Dict, List, Optional, Set, Tuple

from zuspec.dataclasses.constraint_parser import ConstraintParser, extract_rand_fields
from ..ir.constraint_ir import (
    BitRange, ConstraintBlock, ConstraintBlockSet, FieldDecl,
    SOPCube, SOPFunction, SharedTerm, ValidityDecl,
)
from .qm_minimizer import MultiOutputQM
from .cube_minimizer import CubeExpandMinimizer

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
        # Map from derived-field name → BitRange in the input word (built by pre-pass).
        self._derived_to_bitrange: Dict[str, BitRange] = {}

    # ------------------------------------------------------------------
    # Phase A-alt — from_sv_action
    # ------------------------------------------------------------------

    @classmethod
    def from_sv_action(cls, action_ir: Any, prefix: str = 'd') -> 'ConstraintCompiler':
        """Create a ConstraintCompiler pre-populated from a DataTypeAction IR node.

        Bypasses ``extract()`` by converting an ``ActionConstraintSet``
        (produced by the fe-sv pipeline) directly into a ``ConstraintBlockSet``.
        Phases B–F (compute_support, validate, build_cubes/build_table, minimize,
        emit_sv) can then be run unchanged.

        Args:
            action_ir: A ``DataTypeAction`` whose ``constraint_set`` is a
                populated ``ActionConstraintSet`` from ``constraint_mapper.py``.
            prefix:    Wire-name prefix for emitted RTL (e.g. ``'d'``).

        Returns:
            A ``ConstraintCompiler`` with ``cset`` populated; ready for
            ``compute_support()``.

        Raises:
            ValueError: if ``action_ir.constraint_set`` is ``None``.
        """
        # Lazy import — avoids a hard build-time dep of zuspec-synth on zuspec-fe-sv.
        from zuspec.fe.sv.constraint_mapper import (  # type: ignore[import]
            ActionConstraintSet, ActionConstraintBlock, SVConditionalBody,
            BitRangeExtraction, ValidMarker, DontCareMarker,
        )

        acs = getattr(action_ir, 'constraint_set', None)
        if acs is None:
            raise ValueError(
                f"DataTypeAction '{getattr(action_ir, 'name', '?')}' has no constraint_set; "
                "ensure zuspec-fe-sv parsed the file with ConstraintMapper enabled."
            )

        # Bypass __init__ — we populate everything manually.
        cc: 'ConstraintCompiler' = cls.__new__(cls)
        cc._cls = None
        cc._prefix = prefix
        cc.cset = None
        cc._per_field_support = {}
        cc._ones_by_bit = {}
        cc._dontcares_by_bit = {}
        cc._n_vars = 0
        cc.strategy = None

        # Build the derived-field → BitRange map from extraction constraints.
        cc._derived_to_bitrange = {
            e.field_name: BitRange(msb=e.msb, lsb=e.lsb)
            for e in acs.extractions
        }

        # Build output FieldDecl list from rand fields.
        output_fields: List[FieldDecl] = []
        for f in acs.fields:
            if f.is_rand:
                output_fields.append(FieldDecl(name=f.name, width=f.width))

        # Convert ActionConstraintBlocks → ConstraintBlocks + ValidityDecls.
        constraints: List[ConstraintBlock] = []
        validity_decls: List[ValidityDecl] = []

        for sv_block in acs.constraint_blocks:
            cond = sv_block.conditional
            if cond is None:
                continue  # extraction-only blocks don't produce synthesis blocks

            # Guard: map SVConstraintExpr(field_name, value) → {BitRange: value}.
            conditions: Dict[BitRange, int] = {}
            for sc in cond.guard_conditions:
                br = cc._derived_to_bitrange.get(sc.field_name)
                if br is not None:
                    conditions[br] = sc.value

            if not conditions:
                continue  # unmappable guard (unknown derived field) — skip

            assignments = dict(cond.assignments)
            if not assignments:
                continue  # no assignments → nothing to synthesize

            constraints.append(ConstraintBlock(
                name=sv_block.name,
                conditions=conditions,
                assignments=assignments,
            ))

            # Valid markers → ValidityDecl (guard=None: conservative, always observable).
            # The guard is already encoded in the ConstraintBlock conditions; treating
            # the field as always observable is safe (conservative for ODC purposes).
            for vm in cond.valid_markers:
                validity_decls.append(ValidityDecl(
                    field_name=vm.field_name,
                    guard=None,
                    source_method=sv_block.name,
                ))

        cc.cset = ConstraintBlockSet(
            input_field=acs.input_field or 'instr',
            input_width=acs.input_width,
            output_fields=output_fields,
            constraints=constraints,
            validity_decls=validity_decls,
        )

        return cc

    # ------------------------------------------------------------------
    # Phase A — extract
    # ------------------------------------------------------------------

    def extract(self) -> None:
        """Walk @_is_constraint methods, build ConstraintBlockSet."""
        parser = ConstraintParser()

        # ------------------------------------------------------------------ #
        # Step 1: Scan dataclass fields to identify input field and outputs.  #
        # This must run before block-building so we know the input field name  #
        # for the derived-field pre-pass.                                      #
        # ------------------------------------------------------------------ #
        output_fields: List[FieldDecl] = []
        input_field_name: Optional[str] = None
        input_field_width: Optional[int] = None  # None = not yet determined
        cond_field_names_legacy: Set[str] = set()  # used only for legacy fallback

        if dataclasses.is_dataclass(self._cls):
            try:
                from zuspec.dataclasses.decorators import Input as _InputMarker
            except ImportError:
                _InputMarker = None

            # Use __annotations__ directly to preserve Annotated metadata without
            # triggering get_type_hints's forward-reference resolver (which fails on
            # generic Action[T] base classes).
            ann: Dict[str, Any] = {}
            for klass in reversed(self._cls.__mro__):
                ann.update(getattr(klass, '__annotations__', {}))

            def _ann_width(name: str) -> Optional[int]:
                hint = ann.get(name)
                if hint is not None and hasattr(hint, '__metadata__') and hint.__metadata__:
                    return getattr(hint.__metadata__[0], 'width', None)
                return None

            for f in dataclasses.fields(self._cls):
                meta = f.metadata
                if not meta and isinstance(getattr(f, 'type', None), dataclasses.Field):
                    meta = f.type.metadata  # type: ignore[union-attr]

                if meta.get('rand') or meta.get('randc'):
                    width = meta.get('width', None)
                    if not isinstance(width, int):
                        width = _ann_width(f.name)
                    if not isinstance(width, int):
                        width = 1
                    soft = meta.get('soft_default', None)
                    # Auto-detect internal fields: leading '_' convention OR
                    # explicit metadata["internal"] = True (from zdc.field(internal=True)).
                    is_internal = f.name.startswith('_') or bool(meta.get('internal', False))
                    output_fields.append(FieldDecl(name=f.name, width=width, soft_default=soft,
                                                   internal=is_internal))
                elif _InputMarker is not None and f.default_factory is _InputMarker:
                    if input_field_name is None:
                        input_field_name = f.name
                        w = meta.get('width', None) if meta else None
                        if not isinstance(w, int):
                            w = _ann_width(f.name)
                        if isinstance(w, int):
                            input_field_width = w

        # ------------------------------------------------------------------ #
        # Step 2: Build derived-field map (pre-pass) using known input name.  #
        # ------------------------------------------------------------------ #
        if input_field_name is None:
            # Temporary fallback: need at least a name for the pre-pass.
            # Will be refined in Step 3 using legacy subscript detection.
            input_field_name = 'insn'

        self._build_derived_field_map(input_field_name, parser)

        # ------------------------------------------------------------------ #
        # Step 3: Build constraint blocks AND validity/internal decls.
        # ------------------------------------------------------------------ #
        blocks: List[ConstraintBlock] = []
        validity_decls: List[ValidityDecl] = []
        internal_fields: List[str] = []

        # Collect constraint methods from the full MRO (base classes first so
        # subclass overrides take precedence).  Using a dict keyed by attr_name
        # means the last write wins — i.e. the most-derived definition is used.
        _constraint_methods: Dict[str, Any] = {}
        for klass in reversed(self._cls.__mro__):
            for attr_name, value in vars(klass).items():
                if callable(value) and getattr(value, '_is_constraint', False):
                    _constraint_methods[attr_name] = value

        for attr_name, value in _constraint_methods.items():
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
                # Extract validity_decl / internal_decl from consequents of implies.
                if expr.get('type') == 'implies':
                    guard = expr.get('antecedent')
                    for cons in (expr.get('consequent') or []):
                        if isinstance(cons, dict):
                            if cons.get('type') == 'validity_decl':
                                fname = self._field_name_from_expr(cons.get('field', {}))
                                if fname:
                                    validity_decls.append(
                                        ValidityDecl(field_name=fname, guard=guard,
                                                     source_method=attr_name))
                            elif cons.get('type') == 'internal_decl':
                                fname = self._field_name_from_expr(cons.get('field', {}))
                                if fname and fname not in internal_fields:
                                    internal_fields.append(fname)
                # Top-level validity_decl (unconditionally observable — explicit)
                elif expr.get('type') == 'validity_decl':
                    fname = self._field_name_from_expr(expr.get('field', {}))
                    if fname:
                        validity_decls.append(
                            ValidityDecl(field_name=fname, guard=None,
                                         source_method=attr_name))
                elif expr.get('type') == 'internal_decl':
                    fname = self._field_name_from_expr(expr.get('field', {}))
                    if fname and fname not in internal_fields:
                        internal_fields.append(fname)
                # Legacy path: collect subscript-referenced field names.
                for fname in self._collect_subscript_fields(
                        expr.get('antecedent', {}) if expr.get('type') == 'implies' else {}):
                    cond_field_names_legacy.add(fname)

        # Mark internal fields on their FieldDecl objects (from zdc.internal() calls).
        for fname in internal_fields:
            fd = next((f for f in output_fields if f.name == fname), None)
            if fd is not None:
                fd.internal = True

        # Collect internal fields that were auto-detected from '_' prefix or
        # metadata["internal"] during field construction above, and add them to
        # the internal_fields list so ConstraintBlockSet.internal_fields is complete.
        for fd in output_fields:
            if fd.internal and fd.name not in internal_fields:
                internal_fields.append(fd.name)

        # ------------------------------------------------------------------ #
        # Step 4: Refine input field detection using legacy fallback if needed.
        # ------------------------------------------------------------------ #
        if input_field_name == 'insn' and cond_field_names_legacy:
            # No zdc.input() marker found; try subscript-referenced names.
            if dataclasses.is_dataclass(self._cls):
                dc_field_names = {f.name for f in dataclasses.fields(self._cls)}
                for fname in sorted(cond_field_names_legacy):
                    if fname in dc_field_names:
                        input_field_name = fname
                        # Re-run derived map with the correct name.
                        self._build_derived_field_map(input_field_name, parser)
                        break

        if input_field_name == 'insn' and cond_field_names_legacy:
            input_field_name = next(iter(sorted(cond_field_names_legacy)))

        # Infer input width from highest MSB seen only when not determined from annotations.
        if input_field_width is None:
            max_msb = max((br.msb for b in blocks for br in b.conditions), default=31)
            input_field_width = max_msb + 1 if max_msb > 0 else 32

        self.cset = ConstraintBlockSet(
            input_field=input_field_name,
            input_width=input_field_width,
            output_fields=output_fields,
            constraints=blocks,
            validity_decls=validity_decls,
            internal_fields=internal_fields,
        )

    # -- Phase A helpers ---------------------------------------------------

    def _field_name_from_expr(self, expr: Dict[str, Any]) -> Optional[str]:
        """Extract an output field name from a parsed attribute/name expression."""
        if expr.get('type') == 'attribute':
            return expr.get('attr')
        if expr.get('type') == 'name':
            return expr.get('id')
        return None

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
                left = node.get('left', {})
                # Case 1: subscript guard — self.instr[6:0] == X
                if left.get('type') == 'subscript':
                    br = self._extract_bitrange(left)
                    val = self._extract_int(comps[0])
                    if br is not None and val is not None:
                        return {br: val}
                # Case 2: named derived-field guard — self.opcode == X
                elif left.get('type') == 'attribute':
                    field_name = left.get('attr', '')
                    br = self._derived_to_bitrange.get(field_name)
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

    def _build_derived_field_map(self, input_field_name: str, parser: Any) -> None:
        """Pre-pass: scan constraints for bit-extraction patterns, build _derived_to_bitrange.

        Recognizes patterns of the form::

            assert self.<derived> == (self.<input> >> lsb) & mask   # shift-mask
            assert self.<derived> == (self.<input> & mask)           # mask only (lsb=0)
            assert self.<derived> == self.<input>[msb:lsb]           # subscript

        The map is used by _parse_conditions() to resolve named-field guards
        (``self.opcode == X``) into BitRange conditions on the input word.
        """
        self._derived_to_bitrange = {}

        _methods: Dict[str, Any] = {}
        for klass in reversed(self._cls.__mro__):
            for attr_name, value in vars(klass).items():
                if callable(value) and getattr(value, '_is_constraint', False):
                    _methods[attr_name] = value

        for attr_name, value in _methods.items():
            if not (callable(value) and getattr(value, '_is_constraint', False)):
                continue
            try:
                parsed = parser.parse_constraint(value)
            except Exception:
                continue

            for expr in parsed.get('exprs', []):
                if expr.get('type') != 'compare':
                    continue
                if expr.get('ops') != ['==']:
                    continue
                left = expr.get('left', {})
                if left.get('type') != 'attribute':
                    continue
                derived_name = left.get('attr', '')
                if not derived_name:
                    continue
                comparators = expr.get('comparators', [])
                if not comparators:
                    continue
                rhs = comparators[0]
                br = self._try_parse_extraction(rhs, input_field_name)
                if br is not None:
                    self._derived_to_bitrange[derived_name] = br

    def _try_parse_extraction(self, node: Dict[str, Any], input_name: str) -> Optional[BitRange]:
        """Try to parse an extraction RHS into a BitRange.

        Handles:
          - ``self.<input>[msb:lsb]``                   → BitRange(msb, lsb)
          - ``(self.<input> >> lsb) & mask``             → BitRange(lsb + width(mask) - 1, lsb)
          - ``self.<input> & mask``                      → BitRange(width(mask) - 1, 0)
        """
        t = node.get('type')

        # Subscript form: self.<input>[msb:lsb]
        if t == 'subscript':
            val = node.get('value', {})
            field = val.get('attr') if val.get('type') == 'attribute' else val.get('id')
            if field != input_name:
                return None
            return self._extract_bitrange(node)

        # Mask form: expr & mask
        if t == 'bin_op' and node.get('op') == '&':
            left = node.get('left', {})
            right = node.get('right', {})
            mask_val = self._extract_int(right)
            if mask_val is None or mask_val <= 0:
                return None
            width = mask_val.bit_length()

            # Plain mask: self.<input> & mask → lsb=0
            if left.get('type') == 'attribute' and left.get('attr') == input_name:
                return BitRange(msb=width - 1, lsb=0)

            # Shift-mask: (self.<input> >> lsb) & mask
            if left.get('type') == 'bin_op' and left.get('op') == '>>':
                inner_left = left.get('left', {})
                inner_right = left.get('right', {})
                field = (inner_left.get('attr')
                         if inner_left.get('type') == 'attribute'
                         else inner_left.get('id'))
                if field != input_name:
                    return None
                lsb = self._extract_int(inner_right)
                if lsb is None:
                    return None
                return BitRange(msb=lsb + width - 1, lsb=lsb)

        return None

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
          - ``_cubes_by_bit[col]``     — cubes for which this bit is driven to 1
            (ON-set).
          - ``_off_cubes_by_bit[col]`` — cubes that explicitly drive this bit to 0
            (OFF-set).  Used by CubeExpandMinimizer for don't-care exploitation.

        Blocks that do not mention a field at all leave it as don't-care by
        omission and contribute nothing to either cube list.
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
        off_cubes_by_bit: Dict[str, List[Tuple[int, int]]] = {}
        for fd in self.cset.output_fields:
            for b in range(fd.width):
                col = bit_col(fd.name, b, fd.width)
                cubes_by_bit[col] = []
                off_cubes_by_bit[col] = []

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
                    continue  # Not assigned — DC by omission; no cube added.
                for b in range(fd.width):
                    col = bit_col(fd.name, b, fd.width)
                    if (assigned_val >> b) & 1:
                        cubes_by_bit[col].append((cube_mask, cube_value))
                    else:
                        off_cubes_by_bit[col].append((cube_mask, cube_value))

        self._cubes_by_bit = cubes_by_bit
        self._off_cubes_by_bit = off_cubes_by_bit

    # ------------------------------------------------------------------
    # Phase D-ODC — build_odc_cubes
    # ------------------------------------------------------------------

    def build_odc_cubes(self) -> None:
        """Build per-output-bit observability cube lists from ValidityDecls.

        Must be called after ``build_cubes()`` (requires ``_cubes_by_bit``
        to be populated so that guard field names can be resolved to cube
        lists).

        For each output field that has at least one ``ValidityDecl``:
          1. Resolve each guard to a list of (mask, value) cubes (the guard's
             ON-set in the support space).
          2. Union the resolved cubes across all decls for that field.
          3. Store the union in ``cset.obs_cubes_by_bit`` (one entry per
             output bit column, same key format as ``_cubes_by_bit``).

        Fields with *no* ValidityDecl are skipped; they remain always
        observable (no ODC exploitation).
        """
        assert self.cset is not None, "Call build_cubes() first"
        assert hasattr(self, '_cubes_by_bit'), "Call build_cubes() first"

        import warnings

        def bit_col(fname: str, bit: int, width: int) -> str:
            return fname if width == 1 else f"{fname}_bit{bit}"

        # Group validity_decls by field name.
        from collections import defaultdict
        decls_by_field: Dict[str, List] = defaultdict(list)
        for vd in self.cset.validity_decls:
            decls_by_field[vd.field_name].append(vd)

        obs_cubes_by_bit: Dict[str, List[Tuple[int, int]]] = {}

        for fname, decls in decls_by_field.items():
            # Union of resolved guard cubes.
            obs_cubes: List[Tuple[int, int]] = []
            for vd in decls:
                if vd.guard is None:
                    # Unconditionally observable — tautology cube (mask=0, value=0).
                    obs_cubes.append((0, 0))
                else:
                    resolved = self._resolve_guard(vd.guard)
                    if resolved is not None:
                        obs_cubes.extend(resolved)
                    else:
                        warnings.warn(
                            f"ODC: could not resolve guard for zdc.valid({fname}) "
                            f"in {vd.source_method} — field treated as always observable",
                            stacklevel=2,
                        )
                        obs_cubes.append((0, 0))  # conservative: always observable

            fd = self.cset.field_by_name(fname)
            if fd is None:
                log.warning("ODC: zdc.valid() references unknown field '%s' — skipped", fname)
                continue
            for b in range(fd.width):
                col = bit_col(fname, b, fd.width)
                obs_cubes_by_bit[col] = obs_cubes

        self.cset.obs_cubes_by_bit = obs_cubes_by_bit

    def _resolve_guard(self, expr: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        """Convert a guard expression to a list of (mask, value) support-space cubes.

        Returns the ON-set of the guard as a cube list, or ``None`` if the
        guard contains constructs that cannot be resolved (NOT, comparisons
        with constants not in the cube algebra, etc.).

        Supported patterns:
          FieldRef (attribute/name)  → _cubes_by_bit[field]  (direct lookup)
          BoolOp OR(a, b, ...)       → union of resolved lists  (free)
          BoolOp AND(a, b, ...)      → pairwise cube intersection  (O(|A|×|B|))
          Constant True / bool True  → [(0, 0)]  (tautology)

        Unsupported (returns None with warning):
          NOT expr                   → complement deferred to Sprint S4
          Comparison                 → constant guard not supported in v1
        """
        import warnings
        t = expr.get('type')

        if t == 'attribute':
            fname = expr.get('attr', '')
            return self._get_field_cubes(fname)

        if t == 'name':
            fname = expr.get('id', '')
            return self._get_field_cubes(fname)

        if t == 'constant':
            v = expr.get('value')
            if v is True or v == 1:
                return [(0, 0)]   # tautology
            if v is False or v == 0:
                return []         # empty (never)
            return None

        if t == 'bool_op':
            op = expr.get('op')
            children = expr.get('values', [])
            if op == 'or':
                result: List[Tuple[int, int]] = []
                for child in children:
                    sub = self._resolve_guard(child)
                    if sub is None:
                        return None
                    result.extend(sub)
                return result
            if op == 'and':
                result = [(0, 0)]   # start with tautology
                for child in children:
                    sub = self._resolve_guard(child)
                    if sub is None:
                        return None
                    result = self._cube_list_intersect(result, sub)
                return result

        if t == 'unary_op' and expr.get('op') == 'not':
            # S4: NOT guard — compute complement via De Morgan's laws.
            # NOT(FieldRef) = OFF-set of the field (no solver needed).
            # NOT(OR(a,b))  = AND(NOT(a), NOT(b)) = intersection.
            # NOT(AND(a,b)) = OR(NOT(a), NOT(b))  = union.
            return self._complement_guard(expr.get('operand', {}))

        if t == 'compare':
            warnings.warn(
                "ODC: comparison guard in zdc.valid() not yet supported in v1 "
                "— field treated as always observable",
                stacklevel=3,
            )
            return None

        return None

    def _complement_guard(self, expr: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        """Return the complement (NOT) of a guard expression as cube list.

        Uses De Morgan's laws to push negation to the leaves, where the
        complement of a boolean field is its OFF-set (already computed by
        build_cubes()).  No external solver is required.

        Rules:
          NOT(FieldRef)      → _off_cubes_by_bit[field]
          NOT(NOT(x))        → _resolve_guard(x)   (double negation)
          NOT(OR(a, b, ...)) → intersection of NOT(a), NOT(b), ...
          NOT(AND(a,b, ...)) → union of NOT(a), NOT(b), ...
          NOT(True)          → []     (empty / never observable)
          NOT(False)         → [(0,0)] (tautology / always observable)
        """
        import warnings
        t = expr.get('type')

        if t in ('attribute', 'name'):
            fname = expr.get('attr') or expr.get('id', '')
            return self._get_field_off_cubes(fname)

        if t == 'unary_op' and expr.get('op') == 'not':
            # NOT NOT x → x
            return self._resolve_guard(expr.get('operand', {}))

        if t == 'bool_op':
            op = expr.get('op')
            children = expr.get('values', [])
            if op == 'or':
                # NOT(a OR b) = NOT(a) AND NOT(b)
                result: List[Tuple[int, int]] = [(0, 0)]  # tautology seed
                for child in children:
                    sub = self._complement_guard(child)
                    if sub is None:
                        return None
                    result = self._cube_list_intersect(result, sub)
                return result
            if op == 'and':
                # NOT(a AND b) = NOT(a) OR NOT(b)
                result = []
                for child in children:
                    sub = self._complement_guard(child)
                    if sub is None:
                        return None
                    result.extend(sub)
                return result

        if t == 'constant':
            v = expr.get('value')
            if v is True or v == 1:
                return []        # NOT True = False
            if v is False or v == 0:
                return [(0, 0)]  # NOT False = True
            return None

        warnings.warn(
            f"ODC: complement of '{t}' guard not supported "
            "— field treated as always observable",
            stacklevel=3,
        )
        return None

    def _get_field_off_cubes(self, fname: str) -> Optional[List[Tuple[int, int]]]:
        """Return the OFF-set cube list for a boolean field used as a NOT guard."""
        cubes = self._off_cubes_by_bit.get(fname)
        if cubes is None:
            log.warning("ODC: guard field '%s' not found in OFF cube set", fname)
            return None
        return list(cubes)

    def _get_field_cubes(self, fname: str) -> Optional[List[Tuple[int, int]]]:
        """Return the ON-set cube list for a boolean field used as a guard.

        Looks up `_cubes_by_bit[fname]` (for 1-bit fields the column name
        equals the field name).  Returns None if the field is unknown.
        """
        cubes = self._cubes_by_bit.get(fname)
        if cubes is None:
            log.warning("ODC: guard field '%s' not found in cube set", fname)
            return None
        return list(cubes)

    @staticmethod
    def _cube_list_intersect(
        a: List[Tuple[int, int]],
        b: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Pairwise intersection of two cube lists.

        For each pair (m1, v1) from *a* and (m2, v2) from *b*:
          conflict = m1 & m2 & (v1 ^ v2)   # bits constrained to different values
          if conflict == 0:
            result cube: mask = m1 | m2, value = v1 | v2
        """
        result: List[Tuple[int, int]] = []
        for m1, v1 in a:
            for m2, v2 in b:
                conflict = m1 & m2 & (v1 ^ v2)
                if not conflict:
                    result.append((m1 | m2, v1 | v2))
        return result

    # ------------------------------------------------------------------
    # Phase E — minimize
    # ------------------------------------------------------------------

    def minimize(self) -> None:
        """Run SOP minimization and store results in cset.

        Selects the best available minimizer:
          1. CubeExpandMinimizer (GROW): when both ON- and OFF-cubes are
             available (i.e. after ``build_cubes()``).  Exploits the large
             don't-care space without enumerating it.
          2. MultiOutputQM (cube path): fallback when only ON-cubes are
             available.
          3. MultiOutputQM (minterm path): legacy fallback after
             ``build_table()``.
        """
        n = self._n_vars or self.cset.support_size()

        if (hasattr(self, '_cubes_by_bit') and self._cubes_by_bit is not None
                and hasattr(self, '_off_cubes_by_bit')):
            # Preferred path: GROW minimizer with explicit OFF-cubes.
            # Pass obs_cubes if available (from build_odc_cubes).
            obs = self.cset.obs_cubes_by_bit if self.cset.obs_cubes_by_bit else None
            per_output_cubes, shared_terms = CubeExpandMinimizer().minimize(
                self._cubes_by_bit,
                self._off_cubes_by_bit,
                n,
                obs_cubes=obs,
            )
        elif hasattr(self, '_cubes_by_bit') and self._cubes_by_bit is not None:
            # Cube-based QM (no OFF-set available).
            per_output_cubes, shared_terms = MultiOutputQM().minimize_from_cube_sets(
                self._cubes_by_bit, n
            )
        else:
            # Legacy: minterm-based path (after build_table()).
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
                # Declare with zero-based width so that [0], [1], ... accesses are
                # in-range.  The RHS slice preserves the correct bit mapping.
                w = br.msb - br.lsb + 1
                lines.append(
                    f"wire [{w-1}:0] {vn} = "
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

            # Derived fields are direct wire aliases to bit-slices of the input.
            if fd.name in self._derived_to_bitrange:
                br = self._derived_to_bitrange[fd.name]
                pfx = f"{p}_" if p else ""
                if fd.width == 1:
                    lines.append(f"wire {pfx}{fd.name} = {input_sig}[{br.msb}];")
                else:
                    lines.append(
                        f"wire [{fd.width-1}:0] {pfx}{fd.name} = "
                        f"{input_sig}[{br.msb}:{br.lsb}];"
                    )
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
