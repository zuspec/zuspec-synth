"""Generic accessor-method-to-wire lowering.

Discovers all synchronous instance methods on an action class whose
signature is ``(self)`` only, analyzes their AST to determine whether
they are pure combinational expressions over ``zdc.input()`` fields,
and emits equivalent SV ``wire`` declarations.

No name matching is performed at any point.  Method names are used only
as wire identifiers (leading underscores stripped).  Helper patterns such
as sign extension are detected from the method body structure alone.
"""
from __future__ import annotations

import ast
import dataclasses as dc
import inspect
import logging
import textwrap
from typing import List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input-field detection
# ---------------------------------------------------------------------------

def _find_input_field_names(action_cls: type) -> Set[str]:
    """Return the names of all ``zdc.input()`` fields on *action_cls*."""
    try:
        fields = dc.fields(action_cls)
    except TypeError:
        return set()
    from zuspec.dataclasses.decorators import Input  # type: ignore[import]
    return {f.name for f in fields
            if getattr(f, 'default_factory', dc.MISSING) is Input}


# ---------------------------------------------------------------------------
# Method discovery
# ---------------------------------------------------------------------------

def _discover_accessor_methods(cls: type) -> List[Tuple[str, object]]:
    """Return ``(name, fn)`` pairs for instance methods taking only ``self``.

    Excludes staticmethods, classmethods, dunder methods, and any method
    whose parameter list contains anything beyond ``self``.
    """
    results = []
    for name, raw in vars(cls).items():
        if name.startswith('__'):
            continue
        if isinstance(raw, (staticmethod, classmethod)):
            continue
        if not callable(raw):
            continue
        try:
            sig = inspect.signature(raw)
        except (ValueError, TypeError):
            continue
        if list(sig.parameters) == ['self']:
            results.append((name, raw))
    return results


# ---------------------------------------------------------------------------
# Sign-extension structural detection
# ---------------------------------------------------------------------------

def _detect_sext_body(fn_or_raw) -> bool:
    """Return True if the function's body structurally implements sign extension.

    Recognises the pattern::

        if v & (1 << (bits - 1)):
            v |= -(1 << bits)
        return v & MASK

    Detection is purely structural — the function name is never examined.
    """
    fn = fn_or_raw.__func__ if isinstance(fn_or_raw, staticmethod) else fn_or_raw
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        has_sign_if = False
        has_mask_return = False
        for stmt in node.body:
            if isinstance(stmt, ast.If):
                cond = stmt.test
                if isinstance(cond, ast.BinOp) and isinstance(cond.op, ast.BitAnd):
                    for sub in stmt.body:
                        if isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.BitOr):
                            has_sign_if = True
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                v = stmt.value
                if isinstance(v, ast.BinOp) and isinstance(v.op, ast.BitAnd):
                    if isinstance(v.right, ast.Constant):
                        has_mask_return = True
        if has_sign_if and has_mask_return:
            return True
    return False


# ---------------------------------------------------------------------------
# Helper: look up a name through a class's MRO
# ---------------------------------------------------------------------------

def _lookup_on_class(cls: type, name: str):
    for klass in cls.__mro__:
        if name in vars(klass):
            return vars(klass)[name]
    return None


# ---------------------------------------------------------------------------
# OR-chain analysis
# ---------------------------------------------------------------------------

def _mask_width(mask: int) -> int:
    """Number of bits set in a contiguous-from-zero all-1s mask (0x1F → 5)."""
    n = 0
    while (1 << n) - 1 < mask:
        n += 1
    return n


def _collect_or_terms(node: ast.expr) -> List[ast.expr]:
    """Flatten a nested BitOr tree into a list of leaf terms."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _collect_or_terms(node.left) + _collect_or_terms(node.right)
    return [node]


def _parse_single_term(node: ast.expr,
                       input_field_names: Set[str],
                       insn_var: str,
                       insn_width: int = 32) -> Optional[Tuple[int, int, int]]:
    """Parse one bit-placement term → ``(dst_lo, src_lo, n_bits)`` or None.

    Accepted forms (where *var* is either the local insn alias or
    ``int(self.FIELD)`` for any ``FIELD`` in *input_field_names*):

    - ``(var >> src_lo) & mask << dst_lo``
    - ``(var >> src_lo) << dst_lo``   — no mask: *n_bits* = insn_width - src_lo
    - Same forms without the left-shift (dst_lo = 0)
    """
    def _is_input(n: ast.expr) -> bool:
        if isinstance(n, ast.Name) and n.id == insn_var:
            return True
        return (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == 'int'
                and len(n.args) == 1
                and isinstance(n.args[0], ast.Attribute)
                and isinstance(n.args[0].value, ast.Name)
                and n.args[0].value.id == 'self'
                and n.args[0].attr in input_field_names)

    def _src_nbits(inner: ast.expr) -> Optional[Tuple[int, int]]:
        """Return (src_lo, n_bits) or None."""
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitAnd):
            shift = inner.left
            mask_node = inner.right
            if not isinstance(mask_node, ast.Constant):
                return None
            nbits = _mask_width(int(mask_node.value))
            if isinstance(shift, ast.BinOp) and isinstance(shift.op, ast.RShift):
                if isinstance(shift.right, ast.Constant) and _is_input(shift.left):
                    return int(shift.right.value), nbits
            return None

        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.RShift):
            if isinstance(inner.right, ast.Constant) and _is_input(inner.left):
                return int(inner.right.value), insn_width - int(inner.right.value)

        return None

    # Pattern: inner << dst_lo
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift):
        if not isinstance(node.right, ast.Constant):
            return None
        dst_lo = int(node.right.value)
        r = _src_nbits(node.left)
        return (dst_lo, r[0], r[1]) if r else None

    # No left-shift → dst_lo = 0
    r = _src_nbits(node)
    return (0, r[0], r[1]) if r else None


def _or_chain_to_sv(or_chain: ast.expr,
                    nbits: int,
                    input_field_names: Set[str],
                    insn_var: str,
                    input_signal: str,
                    xlen: int,
                    insn_width: int = 32) -> Optional[str]:
    """Translate a bit-OR chain plus sign-extension width to an SV concat.

    *nbits* is the total signed width before zero-extension to *xlen*.
    Gaps in coverage (e.g. an implicit 0 at bit 0 for branch immediates)
    are filled with ``1'b0`` or ``N'b0``.
    """
    terms_raw = _collect_or_terms(or_chain)
    assignments: List[Tuple[int, int, int]] = []  # (dst_lo, src_lo, n_bits)
    for t in terms_raw:
        if isinstance(t, ast.Constant) and t.value == 0:
            continue
        result = _parse_single_term(t, input_field_names, insn_var, insn_width)
        if result is not None:
            assignments.append(result)
        else:
            log.debug("AccessorLowering: cannot parse OR-chain term: %s", ast.dump(t))
            return None

    if not assignments:
        return f"{xlen}'b0"

    # Sort by dst_lo descending (MSB first for SV concat)
    assignments.sort(key=lambda x: x[0], reverse=True)

    # Find the sign bit: the insn bit at position (nbits-1) in the assembled value
    sign_src_insn_bit: Optional[int] = None
    for dst_lo, src_lo, n in assignments:
        dst_hi = dst_lo + n - 1
        if dst_hi == nbits - 1:
            sign_src_insn_bit = src_lo + n - 1
            break
        if dst_lo <= nbits - 1 <= dst_hi:
            sign_src_insn_bit = src_lo + (nbits - 1 - dst_lo)
            break
    if sign_src_insn_bit is None and assignments:
        dst_lo, src_lo, n = assignments[0]
        sign_src_insn_bit = src_lo + n - 1

    parts: List[str] = []
    n_sign_ext = xlen - nbits
    if n_sign_ext > 0 and sign_src_insn_bit is not None:
        sign_bit = f'{input_signal}[{sign_src_insn_bit}]'
        parts.append('{' + str(n_sign_ext) + '{' + sign_bit + '}}')

    # Build concatenation, inserting zero gaps for uncovered positions
    prev_hi = nbits - 1
    for dst_lo, src_lo, n in assignments:
        dst_hi = dst_lo + n - 1
        gap = prev_hi - dst_hi
        if gap > 0:
            parts.append(f"{gap}'b0")
        src_hi = src_lo + n - 1
        if n == 1:
            parts.append(f'{input_signal}[{src_hi}]')
        else:
            parts.append(f'{input_signal}[{src_hi}:{src_lo}]')
        prev_hi = dst_lo - 1

    if prev_hi >= 0:
        parts.append(f"{prev_hi + 1}'b0")

    if len(parts) == 1:
        return parts[0]
    return '{' + ', '.join(parts) + '}'


# ---------------------------------------------------------------------------
# Upper-mask detection (e.g. 0xFFFFF000 for U-type immediate)
# ---------------------------------------------------------------------------

def _is_upper_mask(mask: int) -> bool:
    """Return True if *mask* is a contiguous run of 1s with trailing zeros."""
    if mask == 0:
        return False
    lo_zeros = (mask & -mask).bit_length() - 1
    if lo_zeros == 0:
        return False
    shifted = mask >> lo_zeros
    return shifted != 0 and (shifted & (shifted + 1)) == 0


def _upper_mask_to_sv(mask: int, input_signal: str, xlen: int) -> str:
    """Convert e.g. ``0xFFFFF000`` → ``{input[31:12], 12'b0}``."""
    lo = (mask & -mask).bit_length() - 1
    hi = mask.bit_length() - 1
    core = '{' + f'{input_signal}[{hi}:{lo}], {lo}\'b0' + '}'
    if xlen > 32:
        n_sign = xlen - 32
        sign_bit = f'{input_signal}[31]'
        return '{' + str(n_sign) + '{' + sign_bit + '}}, ' + core
    return core


# ---------------------------------------------------------------------------
# Per-method translation
# ---------------------------------------------------------------------------

def _is_input_expr(node: ast.expr,
                   input_field_names: Set[str],
                   insn_var: Optional[str]) -> bool:
    """Return True if *node* refers to the action's input field."""
    if insn_var and isinstance(node, ast.Name) and node.id == insn_var:
        return True
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'int'
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == 'self'
            and node.args[0].attr in input_field_names)


def _translate_method(method,
                      input_field_names: Set[str],
                      input_signal: str,
                      cls: type,
                      xlen: int = 32,
                      insn_width: int = 32) -> Optional[Tuple[str, str]]:
    """Translate *method* to ``(width_spec, sv_expr)`` or ``None``.

    *width_spec* is the SV bit-range string such as ``'[4:0]'``, or ``''``
    for a single-bit wire.  Returns ``None`` when the method body cannot be
    reduced to a combinational expression over the action's input fields.
    """
    try:
        src = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(src)
    except Exception as exc:
        log.debug("AccessorLowering: cannot parse source: %s", exc)
        return None

    fn_def: Optional[ast.FunctionDef] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fn_def = node
            break
    if fn_def is None:
        return None

    # Detect local variable aliasing the input field: ``i = int(self.FIELD)``
    insn_var: Optional[str] = None
    for stmt in fn_def.body:
        if (isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and _is_input_expr(stmt.value, input_field_names, None)):
            insn_var = stmt.targets[0].id

    for stmt in fn_def.body:
        if not isinstance(stmt, ast.Return) or stmt.value is None:
            continue
        val = stmt.value

        # ----------------------------------------------------------------
        # Case 1: BinOp with BitAnd — ``int(self.FIELD) & mask``
        #         or ``(int(self.FIELD) >> N) & mask``
        # ----------------------------------------------------------------
        if isinstance(val, ast.BinOp) and isinstance(val.op, ast.BitAnd):
            if not isinstance(val.right, ast.Constant):
                continue
            mask = int(val.right.value)
            left = val.left

            # Upper mask (trailing zeros): {input[hi:lo], lo'b0}
            if _is_upper_mask(mask):
                sv = _upper_mask_to_sv(mask, input_signal, xlen)
                return f'[{xlen-1}:0]', sv

            # Shift + mask: (int(self.FIELD) >> N) & mask
            n_bits = _mask_width(mask)
            if (isinstance(left, ast.BinOp) and isinstance(left.op, ast.RShift)
                    and isinstance(left.right, ast.Constant)
                    and _is_input_expr(left.left, input_field_names, insn_var)):
                src_lo = int(left.right.value)
                hi, lo = src_lo + n_bits - 1, src_lo
                if n_bits == 1:
                    return '', f'{input_signal}[{hi}]'
                return f'[{n_bits-1}:0]', f'{input_signal}[{hi}:{lo}]'

            # No shift: int(self.FIELD) & mask
            if _is_input_expr(left, input_field_names, insn_var):
                if n_bits == 1:
                    return '', f'{input_signal}[0]'
                return f'[{n_bits-1}:0]', f'{input_signal}[{n_bits-1}:0]'

        # ----------------------------------------------------------------
        # Case 2: Direct right shift — ``int(self.FIELD) >> N``
        # ----------------------------------------------------------------
        if (isinstance(val, ast.BinOp) and isinstance(val.op, ast.RShift)
                and isinstance(val.right, ast.Constant)
                and _is_input_expr(val.left, input_field_names, insn_var)):
            src_lo = int(val.right.value)
            n_bits = insn_width - src_lo
            hi = insn_width - 1
            if n_bits == 1:
                return '', f'{input_signal}[{hi}]'
            return f'[{n_bits-1}:0]', f'{input_signal}[{hi}:{src_lo}]'

        # ----------------------------------------------------------------
        # Case 3: Call to a helper whose body structurally implements sext
        #         ``return self.HELPER(expr, nbits)``
        # ----------------------------------------------------------------
        if (isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and isinstance(val.func.value, ast.Name)
                and val.func.value.id == 'self'
                and len(val.args) == 2
                and isinstance(val.args[1], ast.Constant)):
            raw_helper = _lookup_on_class(cls, val.func.attr)
            if raw_helper is None or not _detect_sext_body(raw_helper):
                continue
            nbits = int(val.args[1].value)
            arg0 = val.args[0]

            # Case 3a: plain right-shift (no OR chain) — e.g. ``_imm_i``
            if (isinstance(arg0, ast.BinOp) and isinstance(arg0.op, ast.RShift)
                    and isinstance(arg0.right, ast.Constant)
                    and _is_input_expr(arg0.left, input_field_names, insn_var)):
                src_lo = int(arg0.right.value)
                hi = src_lo + nbits - 1
                lo = src_lo
                n_sign = xlen - nbits
                inner = f'{input_signal}[{hi}:{lo}]'
                if n_sign <= 0:
                    return f'[{xlen-1}:0]', inner
                sign_bit = f'{input_signal}[{hi}]'
                sv = '{' + str(n_sign) + '{' + sign_bit + '}}, ' + inner
                return f'[{xlen-1}:0]', '{' + sv + '}'

            # Case 3b: OR chain — e.g. ``_imm_s``, ``_imm_b``, ``_imm_j``
            sv = _or_chain_to_sv(arg0, nbits, input_field_names,
                                 insn_var or '', input_signal, xlen, insn_width)
            if sv is not None:
                return f'[{xlen-1}:0]', sv

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AccessorLowering:
    """Lower synchronous accessor methods on an action class to SV wire declarations.

    Discovers all instance methods whose signature is ``(self)`` only,
    analyzes their AST to determine whether they are pure combinational
    expressions over ``zdc.input()`` fields, and emits equivalent SV
    ``wire`` declarations.

    No name matching is performed — method names are used only as wire
    identifiers (leading underscores stripped).  Helper patterns such as
    sign extension are detected from the method body structure alone.

    Usage::

        al = AccessorLowering(Decode, input_signal='d_insn', prefix='d', xlen=32)
        for line in al.emit_wires():
            print(line)
    """

    def __init__(self, action_cls: type, input_signal: str,
                 prefix: str = '', xlen: int = 32):
        """
        Args:
            action_cls:   Action class with accessor methods.
            input_signal: SV signal name for the packed input bus (e.g. ``'d_insn'``).
            prefix:       Optional wire-name prefix (e.g. ``'d'`` → ``d_rd``).
            xlen:         Width of sign-extended results (default 32).
        """
        self._cls = action_cls
        self._input_signal = input_signal
        self._prefix = prefix
        self._xlen = xlen
        self._input_fields = _find_input_field_names(action_cls)

    def emit_wires(self) -> List[str]:
        """Return SV ``wire`` declarations for all translatable accessor methods."""
        lines: List[str] = []
        for name, fn in _discover_accessor_methods(self._cls):
            result = _translate_method(fn, self._input_fields,
                                       self._input_signal, self._cls, self._xlen)
            if result is None:
                log.debug("AccessorLowering: %s.%s — not translatable, skipping",
                          self._cls.__name__, name)
                continue
            width_spec, sv_expr = result
            wire_name = name.lstrip('_')
            full_name = (f'{self._prefix}_{wire_name}'
                         if self._prefix else wire_name)
            w = (width_spec + ' ') if width_spec else ''
            lines.append(f'wire {w}{full_name} = {sv_expr};')
        return lines
