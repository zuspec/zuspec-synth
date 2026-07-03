"""Helper constructors for :mod:`zuspec.ir.core` IR — the single chokepoint that
encodes the *RTL IR contract* the be-sv emitter expects.

Two traps these helpers make unrepresentable:

1. **Internal registers must be plain ``Field(is_reg=True)``, not ``FieldInOut``.**
   A ``FieldInOut`` is *always* a module port, so an internal reg modeled as one
   wrongly emits as an ``input``.  Use :func:`reg` for internal state and
   :func:`in_port` / :func:`out_reg` for pins.
2. **Equality/relational operators must be ``ExprCompare``, not
   ``ExprBin(BinOp.Eq)``.**  ``be-sv``'s ``ExprBin`` op-map omits comparisons and
   renders ``"?"``; its ``ExprCompare`` path renders ``==``/``<``/...  Use
   :func:`cmp` / :func:`cmp_eq` — there is deliberately no ``ExprBin`` comparison
   constructor here.

Clock/reset on a synchronous process are conveyed via ``Function.metadata`` as
``ExprRefField`` into the component's ``fields`` (what ``be-sv`` reads).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import zuspec.ir.core as ir

# Comparison operators that must NOT appear inside an ``ExprBin`` (use ExprCompare).
_COMPARE_BINOPS = frozenset(
    op for name in ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE")
    if (op := getattr(ir.BinOp, name, None)) is not None
)


# -- types -------------------------------------------------------------------
def int_t(bits: int, signed: bool = False) -> ir.DataTypeInt:
    return ir.DataTypeInt(bits=bits, signed=signed)


# -- fields ------------------------------------------------------------------
def in_port(name: str, bits: int = 1, signed: bool = False) -> ir.FieldInOut:
    """An input pin."""
    return ir.FieldInOut(name=name, datatype=int_t(bits, signed), is_out=False)


def out_reg(name: str, bits: int = 1, signed: bool = False) -> ir.FieldInOut:
    """A registered output pin."""
    return ir.FieldInOut(name=name, datatype=int_t(bits, signed), is_out=True)


def reg(name: str, bits: int = 1, signed: bool = False) -> ir.Field:
    """An internal register (NOT a port)."""
    return ir.Field(name=name, datatype=int_t(bits, signed), is_reg=True)


# -- expressions -------------------------------------------------------------
def sref(index: int) -> ir.ExprRefField:
    """Reference to ``self.fields[index]``."""
    return ir.ExprRefField(base=ir.TypeExprRefSelf(), index=index)


def const(value: int) -> ir.ExprConstant:
    return ir.ExprConstant(value=value)


def binop(op: ir.BinOp, lhs: ir.Expr, rhs: ir.Expr) -> ir.ExprBin:
    if op in _COMPARE_BINOPS:
        raise ValueError(
            f"refusing to build ExprBin with comparison op {op!r}; use cmp()/cmp_eq() "
            "(be-sv's ExprBin op-map omits comparisons — see module docstring)"
        )
    return ir.ExprBin(lhs=lhs, op=op, rhs=rhs)


def add(lhs: ir.Expr, rhs: ir.Expr) -> ir.ExprBin:
    return binop(ir.BinOp.Add, lhs, rhs)


def cmp(op: ir.CmpOp, lhs: ir.Expr, rhs: ir.Expr) -> ir.ExprCompare:
    return ir.ExprCompare(left=lhs, ops=[op], comparators=[rhs])


def cmp_eq(lhs: ir.Expr, rhs: ir.Expr) -> ir.ExprCompare:
    return cmp(ir.CmpOp.Eq, lhs, rhs)


def inv(operand: ir.Expr) -> ir.ExprUnary:
    """Bitwise NOT (``~``)."""
    return ir.ExprUnary(op=ir.UnaryOp.Invert, operand=operand)


def lognot(operand: ir.Expr) -> ir.ExprUnary:
    """Logical NOT (``!``)."""
    return ir.ExprUnary(op=ir.UnaryOp.Not, operand=operand)


# -- statements --------------------------------------------------------------
def assign(target: ir.Expr, value: ir.Expr) -> ir.StmtAssign:
    return ir.StmtAssign(targets=[target], value=value)


def if_(test: ir.Expr,
        body: Sequence[ir.Stmt],
        orelse: Sequence[ir.Stmt] = ()) -> ir.StmtIf:
    return ir.StmtIf(test=test, body=list(body), orelse=list(orelse))


# -- processes & components --------------------------------------------------
def sync_proc(name: str,
              body: Sequence[ir.Stmt],
              *,
              clock_index: int,
              reset_index: Optional[int] = None) -> ir.Function:
    """A clocked process.  Clock/reset are passed as ``metadata`` refs (be-sv reads
    ``metadata['clock'|'reset']`` as ``ExprRefField`` into ``component.fields``)."""
    metadata: Dict[str, ir.Expr] = {"clock": sref(clock_index)}
    if reset_index is not None:
        metadata["reset"] = sref(reset_index)
    return ir.Function(name=name, body=list(body), metadata=metadata)


def comb_proc(name: str, body: Sequence[ir.Stmt]) -> ir.Function:
    """A combinational process (be-sv emits ``always @(*)``).  Used for Mealy FSM
    outputs — e.g. the register-update ``hwif`` ``we``/``next`` pulse (D1), driven
    combinationally so ``we`` is high exactly one cycle."""
    return ir.Function(name=name, body=list(body))


def component(name: str,
              fields: Sequence[ir.Field],
              *,
              sync_processes: Sequence[ir.Function] = (),
              comb_processes: Sequence[ir.Function] = (),
              proc_processes: Sequence[ir.Function] = ()) -> ir.DataTypeComponent:
    return ir.DataTypeComponent(
        name=name,
        super=None,
        fields=list(fields),
        sync_processes=list(sync_processes),
        comb_processes=list(comb_processes),
        proc_processes=list(proc_processes),
    )


def context(components: Iterable[ir.DataTypeComponent]) -> ir.Context:
    return ir.Context(type_m={c.name: c for c in components})


# -- contract validation -----------------------------------------------------
def field_index(comp: ir.DataTypeComponent, name: str) -> int:
    for i, f in enumerate(comp.fields):
        if f.name == name:
            return i
    raise KeyError(f"no field named {name!r} in component {comp.name!r}")


def validate_rtl_component(comp: ir.DataTypeComponent) -> List[str]:
    """Return a list of RTL-contract violations (empty == clean)."""
    problems: List[str] = []

    def walk_expr(e) -> None:
        if isinstance(e, ir.ExprBin) and e.op in _COMPARE_BINOPS:
            problems.append(
                f"ExprBin uses comparison op {e.op!r}; must be ExprCompare"
            )
        for child in _expr_children(e):
            walk_expr(child)

    def walk_stmt(s) -> None:
        for e in _stmt_exprs(s):
            walk_expr(e)
        for sub in _stmt_children(s):
            walk_stmt(sub)

    for proc in list(comp.sync_processes) + list(comp.comb_processes):
        for s in proc.body:
            walk_stmt(s)

    return problems


def _expr_children(e):
    for attr in ("lhs", "rhs", "operand", "left", "value", "base"):
        child = getattr(e, attr, None)
        if isinstance(child, ir.Expr):
            yield child
    for attr in ("comparators", "args"):
        for child in getattr(e, attr, []) or []:
            if isinstance(child, ir.Expr):
                yield child


def _stmt_exprs(s):
    for attr in ("test", "value"):
        e = getattr(s, attr, None)
        if isinstance(e, ir.Expr):
            yield e
    for t in getattr(s, "targets", []) or []:
        if isinstance(t, ir.Expr):
            yield t


def _stmt_children(s):
    for attr in ("body", "orelse"):
        for sub in getattr(s, attr, []) or []:
            yield sub
