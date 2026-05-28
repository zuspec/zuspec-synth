# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""ir_to_python — shared IR-to-Python expression/statement helpers.

These helpers convert DC IR expression and statement trees to Python source
strings using ``zdc`` API idioms.  They are the Python mirror of
``ir_to_sv.py`` and are consumed by ``FSMToPythonPass``,
``CombToPythonPass``, and any other Python-backend passes.

Place in pass chain
-------------------
These are *helper functions*, not passes.  They are imported by:

- ``passes/fsm_to_python.py``   (single-state and multi-state FSM bodies)
- ``passes/comb_to_python.py``  (combinational process bodies)

Inputs (function arguments)
----------------------------
- ``expr`` / ``stmts`` — IR expression/statement AST nodes.
- ``idx_to_name``       — ``{field_index: signal_name}`` dict from
  ``ComponentFields.idx_to_name`` or ``FSMModule.body_idx_to_name``.

Outputs (return values)
-----------------------
- Strings (``ir_expr_to_python``) or lists of strings (``ir_stmts_to_python``).

Usage example
-------------
::

    from zuspec.synth.sprtl.ir_to_python import ir_expr_to_python, ir_stmts_to_python

    lines = ir_stmts_to_python(fsm.body_stmts, fsm.body_idx_to_name, indent=2)
"""
from __future__ import annotations

_BINOP_PY = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "//", "Mod": "%",
    "BitAnd": "&", "BitOr": "|", "BitXor": "^",
    "LShift": "<<", "RShift": ">>",
    "And": "and", "Or": "or",
    "Eq": "==", "NotEq": "!=",
    "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
}
_UNOP_PY = {"Not": "not ", "USub": "-", "Invert": "~"}

# Constants whose magnitude needs zdc.bv() wrapping in Python source.
_BV_THRESHOLD = 0x10000


def ir_expr_to_python(expr, idx_to_name: dict, *, self_prefix: bool = True) -> str:
    """Recursively convert a DC IR expression node to a Python source string.

    Args:
        expr:          An IR expression AST node (any ``Expr*`` type).
        idx_to_name:   Mapping from ``ExprRefField.index`` to signal name.
        self_prefix:   When True (default), component fields get ``self.``
                       prefix; local variables do not.

    Returns:
        A Python source string representing *expr*.
    """
    t = type(expr).__name__

    if t == "ExprRefField":
        name = idx_to_name.get(expr.index, f"_f{expr.index}")
        return f"self.{name}" if self_prefix else name

    if t == "ExprAttribute":
        base = ir_expr_to_python(expr.value, idx_to_name, self_prefix=self_prefix)
        if base == "" or type(expr.value).__name__ == "TypeExprRefSelf":
            return f"self.{expr.attr}" if self_prefix else expr.attr
        return f"{base}.{expr.attr}"

    if t == "ExprSubscript":
        base = ir_expr_to_python(expr.value, idx_to_name, self_prefix=self_prefix)
        sl = expr.slice
        if type(sl).__name__ == "ExprConstant":
            return f"{base}[{sl.value}]"
        return f"{base}[{ir_expr_to_python(sl, idx_to_name, self_prefix=self_prefix)}]"

    if t == "ExprConstant":
        v = expr.value
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, int) and (v > _BV_THRESHOLD or v < -0x8000):
            return f"zdc.bv(0x{v & 0xFFFFFFFF:X})"
        return str(v)

    if t == "ExprSext":
        val = ir_expr_to_python(expr.value, idx_to_name, self_prefix=self_prefix)
        return f"zdc.sext({val}, bits={expr.bits})"

    if t == "ExprZext":
        val = ir_expr_to_python(expr.value, idx_to_name, self_prefix=self_prefix)
        return f"zdc.zext({val}, bits={expr.bits})"

    if t == "ExprCbit":
        inner = ir_expr_to_python(expr.value, idx_to_name, self_prefix=self_prefix)
        if type(expr.value).__name__ == "ExprCompare":
            return inner
        return f"bool({inner})"

    if t == "ExprSigned":
        val = ir_expr_to_python(expr.value, idx_to_name, self_prefix=self_prefix)
        return f"zdc.signed({val})"

    if t == "ExprCall":
        func = expr.func
        if func is not None and type(func).__name__ == "ExprAttribute":
            # zdc.bv(x), zdc.bit(x), zdc.u32(x), … — pass through args unchanged
            _ZDC_CTOR_ATTRS = ('bv', 'bit', 'u1', 'u2', 'u4', 'u8', 'u16', 'u32', 'u64',
                               's8', 's16', 's32', 's64')
            if func.attr in _ZDC_CTOR_ATTRS:
                base_val = func.value
                if (base_val is not None
                        and type(base_val).__name__ == "ExprRefUnresolved"
                        and base_val.name == "zdc"):
                    args = getattr(expr, 'args', [])
                    if len(args) == 1:
                        arg_py = ir_expr_to_python(args[0], idx_to_name, self_prefix=self_prefix)
                        return f"zdc.{func.attr}({arg_py})"
            # port.read() → self.port
            if func.attr == "read":
                base = func.value
                if base is not None and type(base).__name__ == "ExprRefField":
                    return idx_to_name.get(base.index, f"_f{base.index}")
                if base is not None:
                    return ir_expr_to_python(base, idx_to_name, self_prefix=self_prefix)

        if func is not None and type(func).__name__ == "ExprRefUnresolved":
            fname = func.name
            args = getattr(expr, 'args', [])
            if fname == "zdc.sext" and len(args) == 2:
                val_py = ir_expr_to_python(args[0], idx_to_name, self_prefix=self_prefix)
                bits_val = getattr(args[1], 'value', None)
                bits_str = str(bits_val) if isinstance(bits_val, int) else ir_expr_to_python(
                    args[1], idx_to_name, self_prefix=self_prefix)
                return f"zdc.sext({val_py}, bits={bits_str})"
            if fname == "zdc.zext" and len(args) == 2:
                val_py = ir_expr_to_python(args[0], idx_to_name, self_prefix=self_prefix)
                bits_val = getattr(args[1], 'value', None)
                bits_str = str(bits_val) if isinstance(bits_val, int) else ir_expr_to_python(
                    args[1], idx_to_name, self_prefix=self_prefix)
                return f"zdc.zext({val_py}, bits={bits_str})"
            if fname == "zdc.cbit" and len(args) == 1:
                inner = args[0]
                inner_py = ir_expr_to_python(inner, idx_to_name, self_prefix=self_prefix)
                if type(inner).__name__ == "ExprCompare":
                    return inner_py
                return f"bool({inner_py})"
            if fname == "zdc.signed" and len(args) == 1:
                val_py = ir_expr_to_python(args[0], idx_to_name, self_prefix=self_prefix)
                return f"zdc.signed({val_py})"
            if fname == "_illegal":
                return "# illegal"

        if func is not None:
            fn_str = ir_expr_to_python(func, idx_to_name, self_prefix=self_prefix)
            args_str = ", ".join(
                ir_expr_to_python(a, idx_to_name, self_prefix=self_prefix)
                for a in getattr(expr, 'args', [])
            )
            return f"{fn_str}({args_str})"

    if t == "ExprBool":
        op = getattr(expr, 'op', None)
        op_name = getattr(op, 'name', str(op)) if op else 'And'
        py_op = 'and' if 'And' in str(op_name) else 'or'
        values = getattr(expr, 'values', [])
        parts = [ir_expr_to_python(v, idx_to_name, self_prefix=self_prefix) for v in values]
        return f"({f' {py_op} '.join(parts)})"

    if t == "ExprBin":
        lhs = ir_expr_to_python(expr.lhs, idx_to_name, self_prefix=self_prefix)
        rhs = ir_expr_to_python(expr.rhs, idx_to_name, self_prefix=self_prefix)
        op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
        py_op = _BINOP_PY.get(op_name, op_name)
        return f"({lhs} {py_op} {rhs})"

    if t == "ExprUnary":
        operand = ir_expr_to_python(expr.operand, idx_to_name, self_prefix=self_prefix)
        op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
        py_op = _UNOP_PY.get(op_name, op_name)
        return f"({py_op}{operand})"

    if t == "ExprCompare":
        parts = [ir_expr_to_python(expr.left, idx_to_name, self_prefix=self_prefix)]
        for op, comparator in zip(expr.ops, expr.comparators):
            op_name = op.name if hasattr(op, "name") else str(op)
            py_op = _BINOP_PY.get(op_name, op_name)
            parts.append(py_op)
            parts.append(ir_expr_to_python(comparator, idx_to_name, self_prefix=self_prefix))
        return "(" + " ".join(parts) + ")"

    if t == "ExprRefUnresolved":
        return getattr(expr, 'name', str(expr))

    if hasattr(expr, "name"):
        return expr.name
    if hasattr(expr, "value"):
        return str(expr.value)
    return str(expr)


def _ir_if_to_python(stmt, idx_to_name: dict, indent: int, lines: list,
                     *, self_prefix: bool = True):
    """Emit an if/elif/else chain from a StmtIf IR node."""
    sp = " " * indent
    cond = ir_expr_to_python(stmt.test, idx_to_name, self_prefix=self_prefix)
    lines.append(f"{sp}if {cond}:")
    body_lines = ir_stmts_to_python(
        stmt.body, idx_to_name, indent=indent + 4, self_prefix=self_prefix
    )
    if not body_lines:
        lines.append(f"{sp}    pass")
    else:
        lines.extend(body_lines)

    orelse = stmt.orelse
    while orelse:
        if len(orelse) == 1 and type(orelse[0]).__name__ == "StmtIf":
            inner = orelse[0]
            inner_cond = ir_expr_to_python(inner.test, idx_to_name, self_prefix=self_prefix)
            lines.append(f"{sp}elif {inner_cond}:")
            inner_body = ir_stmts_to_python(
                inner.body, idx_to_name, indent=indent + 4, self_prefix=self_prefix
            )
            if not inner_body:
                lines.append(f"{sp}    pass")
            else:
                lines.extend(inner_body)
            orelse = inner.orelse
        else:
            lines.append(f"{sp}else:")
            else_body = ir_stmts_to_python(
                orelse, idx_to_name, indent=indent + 4, self_prefix=self_prefix
            )
            if not else_body:
                lines.append(f"{sp}    pass")
            else:
                lines.extend(else_body)
            orelse = []


def ir_stmts_to_python(
    stmts: list,
    idx_to_name: dict,
    *,
    indent: int = 8,
    self_prefix: bool = True,
) -> list[str]:
    """Convert a list of DC IR statement nodes to Python source lines.

    Args:
        stmts:       List of IR statement AST nodes.
        idx_to_name: Mapping from ``ExprRefField.index`` to signal name.
        indent:      Number of leading spaces on each output line.
        self_prefix: When True, component-level field refs get ``self.``
                     prefix.

    Returns:
        List of Python source strings (no trailing newline per line).
    """
    lines: list[str] = []
    sp = " " * indent

    for stmt in stmts:
        t = type(stmt).__name__

        if t == "StmtAssign":
            for target in stmt.targets:
                tgt = ir_expr_to_python(target, idx_to_name, self_prefix=self_prefix)
                val = ir_expr_to_python(stmt.value, idx_to_name, self_prefix=self_prefix)
                lines.append(f"{sp}{tgt} = {val}")

        elif t == "StmtAugAssign":
            tgt = ir_expr_to_python(stmt.target, idx_to_name, self_prefix=self_prefix)
            val = ir_expr_to_python(stmt.value, idx_to_name, self_prefix=self_prefix)
            op_name = stmt.op.name if hasattr(stmt.op, "name") else str(stmt.op)
            py_op = _BINOP_PY.get(op_name, op_name)
            lines.append(f"{sp}{tgt} {py_op}= {val}")

        elif t == "StmtIf":
            _ir_if_to_python(stmt, idx_to_name, indent, lines, self_prefix=self_prefix)

        elif t == "StmtExpr":
            expr = stmt.expr
            if expr is not None and type(expr).__name__ == "ExprAwait":
                # Skip tick() awaits — these become clock edges implicitly in @zdc.sync
                inner = expr.value
                if inner is not None and type(inner).__name__ == "ExprCall":
                    func = inner.func
                    if (func is not None
                            and type(func).__name__ == "ExprAttribute"
                            and func.attr == "write"):
                        base = func.value
                        if base is not None and type(base).__name__ == "ExprRefField":
                            field_name = idx_to_name.get(base.index, f"_f{base.index}")
                            args = getattr(inner, 'args', [])
                            val_py = (ir_expr_to_python(args[0], idx_to_name, self_prefix=self_prefix)
                                      if args else "0")
                            if self_prefix:
                                lines.append(f"{sp}self.{field_name} = {val_py}")
                            else:
                                lines.append(f"{sp}{field_name} = {val_py}")

        elif t == "StmtWhile":
            # Inline while body (typically the loop body minus the tick)
            lines.extend(
                ir_stmts_to_python(
                    stmt.body, idx_to_name, indent=indent, self_prefix=self_prefix
                )
            )

        elif t == "StmtFor":
            # Emit for i in range(n): body
            target = ir_expr_to_python(stmt.target, idx_to_name, self_prefix=False)
            iter_expr = stmt.iter
            iter_str = ir_expr_to_python(iter_expr, idx_to_name, self_prefix=False)
            lines.append(f"{sp}for {target} in {iter_str}:")
            body_lines = ir_stmts_to_python(
                stmt.body, idx_to_name, indent=indent + 4, self_prefix=self_prefix
            )
            if not body_lines:
                lines.append(f"{sp}    pass")
            else:
                lines.extend(body_lines)

        else:
            lines.append(f"{sp}# <unsupported: {t}>")

    return lines
