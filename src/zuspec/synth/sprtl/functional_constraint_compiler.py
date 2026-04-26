"""Compile @zdc.constraint methods that use match/case dispatch to SystemVerilog.

Unlike the SOP-based ConstraintCompiler (which handles Decode-style
``if opcode == X: assert field == Y`` patterns), this module handles
Execute-style constraints that:

  * dispatch on enum-valued, non-instruction fields via ``match/case``
  * compute outputs via arbitrary arithmetic expressions
  * define constraint-local witnesses (``name: zdc.uN = zdc.rand()``)
  * may have nested ``match`` statements or ``if/else`` inside case arms

The compiler emits one ``always_comb`` block per constraint method.

Usage example::

    signal_map = {
        'dec.t.kind':    'dec_kind',
        'dec.t.alu_op':  'dec_alu_op',
        'dec.t.funct3':  'dec_funct3',
        'dec.t.rs1_val': 'dec_rs1_val',
        'dec.t.rs2_val': 'dec_rs2_val',
        'dec.t.pc':      'dec_pc',
        'dec.t.imm':     'dec_imm',
        'out.t.alu_out': 'exe_alu_out',
        'out.t.next_pc': 'exe_next_pc',
        'out.t.rd_wen':  'exe_rd_wen',
        'pc.t':          'pc_q',
        '_taken':        'exe_taken',
    }
    fcc = FunctionalConstraintCompiler(Execute, signal_map)
    sv_lines = fcc.emit_sv()
"""
from __future__ import annotations

import ast
import enum
import inspect
import logging
import sys
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expression translator
# ---------------------------------------------------------------------------

class _FuncExprTranslator(ast.NodeVisitor):
    """Translate an Execute-style Python expression AST node to SV string.

    Args:
        signal_map:  dotted-path (without 'self.') → SV signal name, e.g.
                     ``{'dec.t.kind': 'dec_kind'}``.  Single-level names
                     (witness locals, class-level rand fields) are also
                     looked up here.
        local_vars:  witness-local name → SV expression, populated as the
                     compiler walks the case arm body.
        enum_map:    dict mapping ``EnumClass.MEMBER`` dotted string to int,
                     used for case label resolution.  Populated lazily.
        module_globals: globals dict of the action class's defining module,
                        used to resolve module-level constants.
    """

    def __init__(
        self,
        signal_map: Dict[str, str],
        local_vars: Optional[Dict[str, str]] = None,
        module_globals: Optional[Dict[str, Any]] = None,
    ):
        self._signal_map = signal_map
        self._locals = local_vars or {}
        self._globals = module_globals or {}

    def translate(self, node: ast.expr) -> str:
        result = self.visit(node)
        if result is None:
            return f'/* unsupported:{ast.dump(node)} */'
        return result

    # ------------------------------------------------------------------
    # Terminal nodes
    # ------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        if name in self._locals:
            return self._locals[name]
        if name in self._signal_map:
            return self._signal_map[name]
        if name == 'MASK32':
            v = self._globals.get('MASK32', 0xFFFFFFFF)
            return f"32'h{v:08X}"
        v = self._globals.get(name)
        if isinstance(v, int):
            return str(v)
        return name

    def visit_Constant(self, node: ast.Constant) -> str:
        v = node.value
        if isinstance(v, int):
            return str(v)
        return repr(v)

    def visit_Attribute(self, node: ast.Attribute) -> str:
        """Resolve a (possibly multi-level) attribute chain.

        Chains like ``self.dec.t.kind`` are resolved through the signal_map
        using the path *without* the leading ``self.``.  If not found, the
        chain is resolved for enum literals like ``InstrKind.ALU_REG``.
        """
        path = self._attr_path(node)
        if path.startswith('self.'):
            short = path[5:]  # strip 'self.'
        else:
            short = path

        if short in self._signal_map:
            return self._signal_map[short]

        # Try enum resolution: EnumClass.MEMBER → integer literal
        parts = path.rsplit('.', 1)
        if len(parts) == 2:
            cls_name, member = parts
            cls = self._globals.get(cls_name)
            if cls is None:
                for mod in sys.modules.values():
                    cls = getattr(mod, cls_name, None)
                    if cls is not None:
                        break
            if cls is not None and issubclass(cls, enum.IntEnum):
                try:
                    return str(int(cls[member]))
                except (KeyError, TypeError):
                    pass

        return path  # fallback: render as-is

    @staticmethod
    def _attr_path(node: ast.expr) -> str:
        """Return dotted path string for an Attribute chain."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_FuncExprTranslator._attr_path(node.value)}.{node.attr}"
        return ast.dump(node)

    # ------------------------------------------------------------------
    # Compound nodes
    # ------------------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left = self.translate(node.left)
        right = self.translate(node.right)

        op_map = {
            ast.Add:     '+',
            ast.Sub:     '-',
            ast.BitAnd:  '&',
            ast.BitOr:   '|',
            ast.BitXor:  '^',
            ast.LShift:  '<<',
            ast.RShift:  '>>',
            ast.Mult:    '*',
            ast.FloorDiv: '/',
            ast.Mod:     '%',
        }
        op = op_map.get(type(node.op), '?')

        if isinstance(node.op, ast.RShift):
            # Arithmetic right-shift via int(x) >> int(y)
            if self._is_int_call(node.left) and self._is_int_call(node.right):
                il = self.translate(node.left.args[0])
                ir = self.translate(node.right.args[0])
                return f'($signed({il}) >>> {ir})'
            # Arithmetic right-shift via zdc.signed(x) >> y
            if self._is_zdc_signed_call(node.left):
                inner = self.translate(node.left.args[0])
                return f'($signed({inner}) >>> {right})'

        # Mask idiom: expr & 0xFFFFFFFF → no-op in 32-bit context
        if isinstance(node.op, ast.BitAnd):
            v = self._const_int(node.right)
            if v == 0xFFFFFFFF:
                return left
            if v is not None:
                return f"({left} & 32'h{v:08X})"

        # XOR with constant (e.g. MASK32 ^ 1)
        if isinstance(node.op, ast.BitXor):
            v = self._const_int(node.right)
            if v is not None:
                return f"({left} ^ 32'h{v:08X})"

        return f'({left} {op} {right})'

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.translate(node.operand)
        if isinstance(node.op, ast.USub):
            return f'(-{operand})'
        if isinstance(node.op, ast.Invert):
            return f'(~{operand})'
        if isinstance(node.op, ast.Not):
            return f'(!{operand})'
        return f'(?{operand})'

    def visit_IfExp(self, node: ast.IfExp) -> str:
        test   = self.translate(node.test)
        body   = self.translate(node.body)
        orelse = self.translate(node.orelse)
        return f'(({test}) ? {body} : {orelse})'

    def visit_Compare(self, node: ast.Compare) -> str:
        left = self.translate(node.left)
        op_map = {
            ast.Lt:   '<', ast.LtE:  '<=',
            ast.Gt:   '>', ast.GtE:  '>=',
            ast.Eq:   '==', ast.NotEq: '!=',
        }
        parts = []
        for op, comp in zip(node.ops, node.comparators):
            right = self.translate(comp)
            if isinstance(op, (ast.In, ast.NotIn)):
                # Membership test: val in (A, B, C) → (val == A || val == B || ...)
                elts = []
                if isinstance(comp, ast.Tuple):
                    for elt in comp.elts:
                        parts.append(f'{left} == {self.translate(elt)}')
                    joined = ' || '.join(parts)
                    return f'({joined})' if isinstance(op, ast.In) else f'(!({joined}))'
                return f'/* in-check not supported */'
            parts.append(f'{left} {op_map.get(type(op), "??")} {right}')
        return ' && '.join(parts)

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = '&&' if isinstance(node.op, ast.And) else '||'
        parts = [self.translate(v) for v in node.values]
        return f' {op} '.join(f'({p})' for p in parts)

    def visit_Subscript(self, node: ast.Subscript) -> str:
        base = self.translate(node.value)
        sl = node.slice
        if isinstance(sl, ast.Slice):
            msb = self._const_int(sl.lower)
            lsb = self._const_int(sl.upper)
            if msb is not None and lsb is not None:
                return f'{base}[{msb}:{lsb}]'
            if msb is not None and lsb is None:
                return f'{base}[{msb}]'
        elif isinstance(sl, ast.Constant) and isinstance(sl.value, int):
            return f'{base}[{sl.value}]'
        return f'{base}[/*??*/]'

    def visit_Call(self, node: ast.Call) -> str:
        func = node.func

        # int(x) → transparent
        if isinstance(func, ast.Name) and func.id == 'int':
            return self.translate(node.args[0])

        # bool(x) → x
        if isinstance(func, ast.Name) and func.id == 'bool':
            return self.translate(node.args[0])

        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            ns = func.value.id
            fn = func.attr

            if ns == 'zdc':
                # zdc.cbit(expr) → single-bit comparison result
                if fn == 'cbit' and node.args:
                    inner = self.translate(node.args[0])
                    return f"({inner} ? 1'b1 : 1'b0)"

                # zdc.signed(expr) → $signed(expr)
                if fn == 'signed' and node.args:
                    inner = self.translate(node.args[0])
                    return f'$signed({inner})'

                # zdc.sext(inner, n) → {{{{W-n}}{{inner[n-1]}}, inner[n-1:0]}}
                if fn == 'sext' and len(node.args) >= 2:
                    inner = self.translate(node.args[0])
                    n = self._const_int(node.args[1])
                    if n is not None:
                        W = 32
                        return f'{{{{{W-n}{{({inner})[{n-1}]}}}}, ({inner})[{n-1}:0]}}'

                # zdc.concat(a, b, ...) → {a, b, ...}
                if fn == 'concat':
                    parts = []
                    for arg in node.args:
                        if isinstance(arg, ast.Tuple) and len(arg.elts) == 2:
                            # (value, width) zero-fill pair
                            val = self._const_int(arg.elts[0])
                            width = self._const_int(arg.elts[1])
                            if val is not None and width is not None:
                                parts.append(f"{width}'h{val & ((1 << width) - 1):X}")
                                continue
                        parts.append(self.translate(arg))
                    return '{' + ', '.join(parts) + '}'

                # zdc.uN(val) → transparent
                if fn.startswith('u') and node.args:
                    return self.translate(node.args[0])

        # Fallback
        return f'/* call:{ast.dump(func)} */'

    def _is_int_call(self, node: ast.expr) -> bool:
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'int')

    def _is_zdc_signed_call(self, node: ast.expr) -> bool:
        """Return True if node is a ``zdc.signed(...)`` call."""
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'zdc'
                and node.func.attr == 'signed'
                and node.args)

    def _const_int(self, node: ast.expr) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            v = self._globals.get(node.id)
            if isinstance(v, int):
                return v
        if isinstance(node, ast.BinOp):
            l = self._const_int(node.left)
            r = self._const_int(node.right)
            if l is not None and r is not None:
                op = type(node.op)
                if op == ast.BitXor:
                    return l ^ r
                if op == ast.BitAnd:
                    return l & r
                if op == ast.BitOr:
                    return l | r
                if op == ast.Sub:
                    return l - r
                if op == ast.Add:
                    return l + r
        return None


# ---------------------------------------------------------------------------
# Witness discovery
# ---------------------------------------------------------------------------

def _find_witnesses(func_def: ast.FunctionDef) -> Dict[str, int]:
    """Scan a constraint function body for local ``name: zdc.uN = zdc.rand()`` declarations.

    Returns a dict ``{name: bit_width}``.  Only annotated assignments where the
    annotation is a ``zdc.uN`` type (N integer) and the value is a ``zdc.rand()``
    call are recognised.
    """
    witnesses: Dict[str, int] = {}
    for node in ast.walk(func_def):
        if not isinstance(node, ast.AnnAssign):
            continue
        ann = node.annotation
        if not (isinstance(ann, ast.Attribute)
                and isinstance(ann.value, ast.Name)
                and ann.value.id == 'zdc'
                and ann.attr.startswith('u')):
            continue
        try:
            width = int(ann.attr[1:])
        except ValueError:
            continue
        if not isinstance(node.target, ast.Name):
            continue
        # Value must be zdc.rand()
        val = node.value
        if (isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and isinstance(val.func.value, ast.Name)
                and val.func.value.id == 'zdc'
                and val.func.attr == 'rand'):
            witnesses[node.target.id] = width
    return witnesses


# ---------------------------------------------------------------------------
# Case label helpers
# ---------------------------------------------------------------------------

def _resolve_pattern_labels(pattern, module_globals: Dict[str, Any]) -> Optional[str]:
    """Return a comma-separated SV case label string for an AST match pattern.

    Handles:
    - ``MatchValue(Attribute(EnumClass, 'MEMBER'))`` → int value
    - ``MatchValue(Constant(N))`` → N
    - ``MatchOr([p1, p2, ...])`` → "v1, v2, ..."
    - ``MatchAs(pattern=None)`` → default (return None)
    """
    if isinstance(pattern, ast.MatchAs) and pattern.name is None:
        return None  # default

    if isinstance(pattern, ast.MatchValue):
        val_node = pattern.value
        if isinstance(val_node, ast.Constant):
            return str(int(val_node.value))
        if isinstance(val_node, ast.Attribute):
            int_val = _resolve_enum_attr(val_node, module_globals)
            if int_val is not None:
                return str(int_val)
            # Fallback: emit symbolic name
            return val_node.attr
        return None

    if isinstance(pattern, ast.MatchOr):
        parts = []
        for sub in pattern.patterns:
            label = _resolve_pattern_labels(sub, module_globals)
            if label is None:
                return None  # default inside OR — treat whole as default
            parts.append(label)
        return ', '.join(parts)

    return None  # unsupported pattern → treat as default


def _resolve_enum_attr(node: ast.Attribute, module_globals: Dict[str, Any]) -> Optional[int]:
    """Resolve ``EnumClass.MEMBER`` to its integer value."""
    cls_name = (node.value.id
                if isinstance(node.value, ast.Name) else None)
    if cls_name is None:
        return None

    cls = module_globals.get(cls_name)
    if cls is None:
        for mod in sys.modules.values():
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                break

    if cls is None or not (isinstance(cls, type) and issubclass(cls, enum.IntEnum)):
        return None
    try:
        return int(cls[node.attr])
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main compiler
# ---------------------------------------------------------------------------

class FunctionalConstraintCompiler:
    """Compile an action class's @zdc.constraint methods to always_comb SV.

    This compiler handles the Execute-stage pattern where constraints dispatch
    on enum-valued fields and compute outputs with arbitrary arithmetic.  It
    produces one ``always_comb begin … end`` block per constraint method.

    Parameters
    ----------
    cls:
        Action class whose ``@zdc.constraint`` methods are to be compiled.
    signal_map:
        Maps dotted attribute paths (without the leading ``self.``) to SV
        signal names.  Both input fields and output fields are listed here::

            {
                'dec.t.kind':    'dec_kind',   # input
                'out.t.alu_out': 'exe_alu_out', # output
                '_taken':        'exe_taken',   # class-level rand field
            }
    indent:
        Indentation string for emitted SV lines (default: two spaces).
    """

    def __init__(self, cls: type, signal_map: Dict[str, str], indent: str = '  '):
        self._cls = cls
        self._signal_map = signal_map
        self._indent = indent
        mod_name = getattr(cls, '__module__', '')
        mod = sys.modules.get(mod_name, None)
        self._globals: Dict[str, Any] = vars(mod) if mod is not None else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit_sv(self) -> List[str]:
        """Return SV lines for all @zdc.constraint methods of the class."""
        lines: List[str] = []
        for name, value in self._cls.__dict__.items():
            if callable(value) and getattr(value, '_is_constraint', False):
                try:
                    lines.extend(self._emit_constraint(name, value))
                    lines.append('')
                except Exception as exc:
                    log.warning(
                        'FunctionalConstraintCompiler: skipping %s.%s: %s',
                        self._cls.__name__, name, exc,
                    )
        return lines

    # ------------------------------------------------------------------
    # Per-constraint emission
    # ------------------------------------------------------------------

    def _emit_constraint(self, name: str, method) -> List[str]:
        source = inspect.getsource(method)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise ValueError(f'Expected FunctionDef, got {type(func_def).__name__}')

        witnesses = _find_witnesses(func_def)
        local_vars: Dict[str, str] = {}
        tr = _FuncExprTranslator(self._signal_map, local_vars, self._globals)

        I = self._indent
        lines: List[str] = []
        lines.append(f'{I}// {self._cls.__name__}.{name}')
        lines.append(f'{I}always_comb begin')

        # Declare witness wires
        for wname, wwidth in witnesses.items():
            lines.append(f'{I}  logic [{wwidth-1}:0] {wname};')
            lines.append(f'{I}  {wname} = {wwidth}\'b0;')
            local_vars[wname] = wname  # map local name → SV name (self)

        # Emit body statements
        for stmt in func_def.body:
            # Skip docstrings
            if (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                continue
            # Skip witness declarations (already emitted above)
            if isinstance(stmt, ast.AnnAssign) and stmt.target.id in witnesses:
                continue
            lines.extend(self._emit_stmt(stmt, tr, local_vars, indent=I + '  '))

        lines.append(f'{I}end')
        return lines

    # ------------------------------------------------------------------
    # Statement emitters
    # ------------------------------------------------------------------

    def _emit_stmt(
        self,
        stmt: ast.stmt,
        tr: _FuncExprTranslator,
        local_vars: Dict[str, str],
        indent: str,
    ) -> List[str]:
        """Emit SV lines for one statement."""
        if isinstance(stmt, ast.Assert):
            return self._emit_assert(stmt, tr, indent)

        if isinstance(stmt, ast.If):
            return self._emit_if(stmt, tr, local_vars, indent)

        if isinstance(stmt, ast.Match):
            return self._emit_match(stmt, tr, local_vars, indent)

        if isinstance(stmt, ast.AnnAssign):
            # Already handled (witness declarations) — skip
            return []

        if isinstance(stmt, ast.Assign):
            # Local variable assignment (non-witness)
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    sv_val = tr.translate(stmt.value)
                    local_vars[tgt.id] = sv_val
            return []

        if isinstance(stmt, ast.Expr):
            # Bare expression (e.g. await call) — ignore
            return []

        return [f'{indent}/* unsupported stmt: {type(stmt).__name__} */']

    def _emit_assert(
        self,
        stmt: ast.Assert,
        tr: _FuncExprTranslator,
        indent: str,
    ) -> List[str]:
        """Emit SV for ``assert lhs == rhs``."""
        test = stmt.test
        if not isinstance(test, ast.Compare):
            return [f'{indent}/* assert: non-compare test */']
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq) or not test.comparators:
            return [f'{indent}/* assert: unsupported op */']

        lhs_node = test.left
        rhs_node = test.comparators[0]

        lhs_sv = tr.translate(lhs_node)
        rhs_sv = tr.translate(rhs_node)

        return [f'{indent}{lhs_sv} = {rhs_sv};']

    def _emit_if(
        self,
        stmt: ast.If,
        tr: _FuncExprTranslator,
        local_vars: Dict[str, str],
        indent: str,
    ) -> List[str]:
        """Emit SV for ``if cond: body [else: orelse]``."""
        cond_sv = tr.translate(stmt.test)
        lines = [f'{indent}if ({cond_sv}) begin']
        for s in stmt.body:
            lines.extend(self._emit_stmt(s, tr, local_vars, indent + '  '))
        if stmt.orelse:
            lines.append(f'{indent}end else begin')
            for s in stmt.orelse:
                lines.extend(self._emit_stmt(s, tr, local_vars, indent + '  '))
        lines.append(f'{indent}end')
        return lines

    def _emit_match(
        self,
        stmt: ast.Match,
        tr: _FuncExprTranslator,
        local_vars: Dict[str, str],
        indent: str,
    ) -> List[str]:
        """Emit SV for ``match subject: case X: body``."""
        subject_sv = tr.translate(stmt.subject)
        lines = [f'{indent}case ({subject_sv})']
        for case in stmt.cases:
            label = _resolve_pattern_labels(case.pattern, self._globals)
            if label is None:
                lines.append(f'{indent}  default: begin')
            else:
                lines.append(f'{indent}  {label}: begin')
            for s in case.body:
                lines.extend(self._emit_stmt(s, tr, local_vars, indent + '    '))
            lines.append(f'{indent}  end')
        lines.append(f'{indent}endcase')
        return lines
