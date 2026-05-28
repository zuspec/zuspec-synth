# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""ir_to_sv — shared IR-to-SystemVerilog expression/statement helpers.

These helpers convert DC IR expression and statement trees to SV strings.
They are used by both the legacy ``_synthesize_simple_sync`` path (imported
via ``zuspec.synth``) and by the new pass-chain passes (``CombLowerPass``,
``SingleStateStrategy``).

Keeping them in this module avoids importing the heavy ``zuspec.synth``
package from within a pass, which would create circular imports.
"""
from __future__ import annotations

_BINOP_SV = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
    "BitAnd": "&", "BitOr": "|", "BitXor": "^",
    "LShift": "<<", "RShift": ">>",
    "And": "&&", "Or": "||",
    "Eq": "==", "NotEq": "!=",
    "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
}
_UNOP_SV = {"Not": "!", "USub": "-", "Invert": "~"}


def ir_expr_to_sv(expr, idx_to_name: dict) -> str:
    """Recursively convert a DC IR expression to a SV string."""
    t = type(expr).__name__
    if t == "ExprRefField":
        return idx_to_name.get(expr.index, f"_f{expr.index}")
    if t == "ExprAttribute":
        base = ir_expr_to_sv(expr.value, idx_to_name)
        if base == "" or type(expr.value).__name__ == "TypeExprRefSelf":
            return expr.attr
        return f"{base}_{expr.attr}"
    if t == "ExprSubscript":
        base = ir_expr_to_sv(expr.value, idx_to_name)
        sl = expr.slice
        if type(sl).__name__ == "ExprConstant":
            return f"{base}[{sl.value}]"
        return f"{base}[{ir_expr_to_sv(sl, idx_to_name)}]"
    if t == "ExprConstant":
        v = expr.value
        if isinstance(v, int) and not isinstance(v, bool) and (v > 0xFFFF or v < -0x8000):
            return f"32'h{v & 0xFFFFFFFF:08X}"
        return str(v)
    if t == "ExprSext":
        val_sv = ir_expr_to_sv(expr.value, idx_to_name)
        n = expr.bits
        shift = 32 - n
        return f"($signed(({val_sv}) << {shift}) >>> {shift})"
    if t == "ExprZext":
        val_sv = ir_expr_to_sv(expr.value, idx_to_name)
        return f"{val_sv}[{expr.bits - 1}:0]"
    if t == "ExprCbit":
        inner_sv = ir_expr_to_sv(expr.value, idx_to_name)
        if type(expr.value).__name__ == "ExprCompare":
            return inner_sv
        return f"({inner_sv}[0])"
    if t == "ExprSigned":
        return f"$signed({ir_expr_to_sv(expr.value, idx_to_name)})"
    if t == "ExprCall":
        func = expr.func
        if func is not None and type(func).__name__ == "ExprAttribute":
            if func.attr == "read":
                base = func.value
                if base is not None and type(base).__name__ == "ExprRefField":
                    return idx_to_name.get(base.index, f"_f{base.index}")
            _ZDC_CTOR_ATTRS = ('bv', 'bit', 'u1', 'u2', 'u4', 'u8', 'u16', 'u32', 'u64',
                               's8', 's16', 's32', 's64')
            if func.attr in _ZDC_CTOR_ATTRS:
                base_val = func.value
                if (base_val is not None
                        and type(base_val).__name__ == "ExprRefUnresolved"
                        and base_val.name == "zdc"):
                    args = getattr(expr, 'args', [])
                    if len(args) == 1:
                        return ir_expr_to_sv(args[0], idx_to_name)
        if func is not None and type(func).__name__ == "ExprRefUnresolved":
            fname = func.name
            args = getattr(expr, 'args', [])
            if fname == "zdc.sext" and len(args) == 2:
                val_sv = ir_expr_to_sv(args[0], idx_to_name)
                bits_val = getattr(args[1], 'value', None)
                if isinstance(bits_val, int) and bits_val > 0:
                    n = bits_val
                    shift = 32 - n
                    return f"($signed(({val_sv}) << {shift}) >>> {shift})"
                bits_sv = ir_expr_to_sv(args[1], idx_to_name)
                return f"($signed(({val_sv}) << (32-{bits_sv})) >>> (32-{bits_sv}))"
            if fname == "zdc.zext" and len(args) == 2:
                val_sv = ir_expr_to_sv(args[0], idx_to_name)
                bits_val = getattr(args[1], 'value', None)
                if isinstance(bits_val, int) and bits_val > 0:
                    return f"{val_sv}[{bits_val-1}:0]"
                bits_sv = ir_expr_to_sv(args[1], idx_to_name)
                return f"{val_sv}[{bits_sv}-1:0]"
            if fname == "zdc.cbit" and len(args) == 1:
                inner = args[0]
                inner_sv = ir_expr_to_sv(inner, idx_to_name)
                if type(inner).__name__ == "ExprCompare":
                    return inner_sv
                return f"({inner_sv}[0])"
            if fname == "zdc.signed" and len(args) == 1:
                val_sv = ir_expr_to_sv(args[0], idx_to_name)
                return f"$signed({val_sv})"
            if fname == "_illegal":
                return "/* illegal */"
        if func is not None:
            fn_str = ir_expr_to_sv(func, idx_to_name)
            args_str = ", ".join(ir_expr_to_sv(a, idx_to_name) for a in getattr(expr, 'args', []))
            return f"{fn_str}({args_str})"
    if t == "ExprBool":
        op = getattr(expr, 'op', None)
        op_name = getattr(op, 'name', str(op)) if op else 'And'
        sv_op = '&&' if 'And' in str(op_name) else '||'
        values = getattr(expr, 'values', [])
        parts = [ir_expr_to_sv(v, idx_to_name) for v in values]
        return f"({f' {sv_op} '.join(parts)})"
    if t == "ExprBin":
        lhs = ir_expr_to_sv(expr.lhs, idx_to_name)
        rhs = ir_expr_to_sv(expr.rhs, idx_to_name)
        op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
        sv_op = _BINOP_SV.get(op_name, op_name)
        return f"({lhs} {sv_op} {rhs})"
    if t == "ExprUnary":
        operand = ir_expr_to_sv(expr.operand, idx_to_name)
        op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
        sv_op = _UNOP_SV.get(op_name, op_name)
        if op_name == "Invert":
            inner = getattr(expr, 'operand', None)
            if inner is not None and type(inner).__name__ == "ExprConstant":
                v = getattr(inner, 'value', None)
                if isinstance(v, int) and not isinstance(v, bool):
                    return f"32'h{(~v) & 0xFFFFFFFF:08X}"
        return f"({sv_op}{operand})"
    if t == "ExprCompare":
        parts = [ir_expr_to_sv(expr.left, idx_to_name)]
        for op, comparator in zip(expr.ops, expr.comparators):
            op_name = op.name if hasattr(op, "name") else str(op)
            sv_op = _BINOP_SV.get(op_name, op_name)
            parts.append(sv_op)
            parts.append(ir_expr_to_sv(comparator, idx_to_name))
        return "(" + " ".join(parts) + ")"
    if hasattr(expr, "name"):
        return expr.name
    if hasattr(expr, "value"):
        return str(expr.value)
    return str(expr)


def _ir_if_to_sv(stmt, idx_to_name: dict, indent: int, lines: list):
    """Emit an if/elif/else chain from a StmtIf IR node."""
    sp = " " * indent
    cond = ir_expr_to_sv(stmt.test, idx_to_name)
    lines.append(f"{sp}if ({cond}) begin")
    lines.extend(ir_stmts_to_sv(stmt.body, idx_to_name, indent + 2))

    orelse = stmt.orelse
    while orelse:
        if len(orelse) == 1 and type(orelse[0]).__name__ == "StmtIf":
            inner = orelse[0]
            inner_cond = ir_expr_to_sv(inner.test, idx_to_name)
            lines.append(f"{sp}end else if ({inner_cond}) begin")
            lines.extend(ir_stmts_to_sv(inner.body, idx_to_name, indent + 2))
            orelse = inner.orelse
        else:
            lines.append(f"{sp}end else begin")
            lines.extend(ir_stmts_to_sv(orelse, idx_to_name, indent + 2))
            orelse = []
    lines.append(f"{sp}end")


def ir_stmts_to_sv(stmts, idx_to_name: dict, indent: int = 2) -> list:
    """Convert a list of IR statements to SV lines (non-blocking assignments)."""
    lines = []
    sp = " " * indent
    for stmt in stmts:
        t = type(stmt).__name__
        if t == "StmtAssign":
            for target in stmt.targets:
                tgt = ir_expr_to_sv(target, idx_to_name)
                val = ir_expr_to_sv(stmt.value, idx_to_name)
                lines.append(f"{sp}{tgt} <= {val};")
        elif t == "StmtAugAssign":
            tgt = ir_expr_to_sv(stmt.target, idx_to_name)
            val = ir_expr_to_sv(stmt.value, idx_to_name)
            op_name = stmt.op.name if hasattr(stmt.op, "name") else str(stmt.op)
            sv_op = _BINOP_SV.get(op_name, op_name)
            lines.append(f"{sp}{tgt} <= {tgt} {sv_op} {val};")
        elif t == "StmtIf":
            _ir_if_to_sv(stmt, idx_to_name, indent, lines)
        elif t == "StmtExpr":
            expr = stmt.expr
            if expr is not None and type(expr).__name__ == "ExprAwait":
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
                            val_sv = ir_expr_to_sv(args[0], idx_to_name) if args else "'0"
                            lines.append(f"{sp}{field_name} <= {val_sv};")
                            continue
        elif t == "StmtWhile":
            lines.extend(ir_stmts_to_sv(stmt.body, idx_to_name, indent))
    return lines


def _ir_if_to_sv_comb(stmt, idx_to_name: dict, indent: int, lines: list):
    """Emit an if/elif/else chain with blocking assignments for always_comb."""
    sp = " " * indent
    cond = ir_expr_to_sv(stmt.test, idx_to_name)
    lines.append(f"{sp}if ({cond}) begin")
    lines.extend(ir_stmts_to_sv_comb(stmt.body, idx_to_name, indent + 2))

    orelse = stmt.orelse
    while orelse:
        if len(orelse) == 1 and type(orelse[0]).__name__ == "StmtIf":
            inner = orelse[0]
            inner_cond = ir_expr_to_sv(inner.test, idx_to_name)
            lines.append(f"{sp}end else if ({inner_cond}) begin")
            lines.extend(ir_stmts_to_sv_comb(inner.body, idx_to_name, indent + 2))
            orelse = inner.orelse
        else:
            lines.append(f"{sp}end else begin")
            lines.extend(ir_stmts_to_sv_comb(orelse, idx_to_name, indent + 2))
            orelse = []
    lines.append(f"{sp}end")


def ir_stmts_to_sv_comb(stmts, idx_to_name: dict, indent: int = 2) -> list:
    """Convert IR statements to SV lines using blocking (=) assignments for always_comb."""
    lines = []
    sp = " " * indent
    for stmt in stmts:
        t = type(stmt).__name__
        if t == "StmtAssign":
            for target in stmt.targets:
                tgt = ir_expr_to_sv(target, idx_to_name)
                val = ir_expr_to_sv(stmt.value, idx_to_name)
                lines.append(f"{sp}{tgt} = {val};")
        elif t == "StmtAugAssign":
            tgt = ir_expr_to_sv(stmt.target, idx_to_name)
            val = ir_expr_to_sv(stmt.value, idx_to_name)
            op_name = stmt.op.name if hasattr(stmt.op, "name") else str(stmt.op)
            sv_op = _BINOP_SV.get(op_name, op_name)
            lines.append(f"{sp}{tgt} = {tgt} {sv_op} {val};")
        elif t == "StmtIf":
            _ir_if_to_sv_comb(stmt, idx_to_name, indent, lines)
    return lines


def field_bits(f) -> int:
    """Return the bit width for a field, defaulting to 32 for unbounded int."""
    bits = getattr(getattr(f, "datatype", None), "bits", 1)
    if bits is None or bits < 1:
        return 32
    return bits


def width_str(bits: int) -> str:
    return f"[{bits - 1}:0] " if bits > 1 else "       "
