# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""CounterLowerPass — erase Counter sub-components for RTL synthesis.

For each field whose type is ``Counter``, ``ModuloCounter``, or ``WatchdogCounter``,
this pass:

1. **Rewrites** every ``await self.<ctr>.wait_next()`` call in ``@proc`` processes
   to ``await zdc.cycles(PERIOD)``, which the SPRTL transformer handles natively
   with a built-in cycle-counter register.
2. **Neutralises** the counter sub-component field (sets its datatype to ``None``)
   so that ``_is_hierarchical`` does not route the component to the hierarchical
   synthesis path.

``await self.<ctr>.wait_for(v)`` is not supported for RTL synthesis and raises
``NotImplementedError``.

This pass must run **before** the ``_is_hierarchical`` check in
``_synthesize_sprtl`` so that Counter sub-component references are erased before
the synthesis pipeline sees them.

Recognition relies on ``DataTypeRef.ref_name in _COUNTER_TYPE_NAMES`` together
with the Python class field metadata (``'kwargs'`` dict) to extract PERIOD /
WIDTH values.
"""
from __future__ import annotations

import dataclasses as _dc_mod
import logging
from typing import Dict, List

from zuspec.ir.core.expr import (
    ExprAttribute,
    ExprAwait,
    ExprCall,
    ExprConstant,
    ExprRefUnresolved,
)
from zuspec.ir.core.stmt import StmtExpr

_log = logging.getLogger(__name__)

# Counter type names recognised by this pass.
_COUNTER_TYPE_NAMES: frozenset = frozenset(
    ("Counter", "ModuloCounter", "WatchdogCounter")
)


# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------

class _CounterFieldInfo:
    """Metadata captured for one Counter sub-component field."""

    __slots__ = ("field_name", "field_index", "period")

    def __init__(self, field_name, field_index, period):
        self.field_name  = field_name
        self.field_index = field_index
        self.period      = period


# ---------------------------------------------------------------------------
# Main entry point (called from _synthesize_sprtl)
# ---------------------------------------------------------------------------

def lower_counter_fields(component_ir, model_context, py_cls) -> bool:
    """Lower all Counter sub-component fields in *component_ir* in place.

    Parameters
    ----------
    component_ir:
        ``DataTypeComponent`` IR node for the component being synthesised.
    model_context:
        ``DataModel`` context produced by ``DataModelFactory`` (provides the
        ``type_m`` dict for counter type lookup).
    py_cls:
        The Python class being synthesised; used to read ``zdc.inst`` kwargs.

    Returns
    -------
    bool
        ``True`` if at least one counter field was lowered (caller should
        re-run the ``_is_hierarchical`` check).
    """
    infos = _collect_counter_fields(component_ir, model_context, py_cls)
    if not infos:
        return False

    _rewrite_proc_bodies(component_ir, infos)

    for info in infos:
        _neutralize_field(component_ir, info)

    return True


# ---------------------------------------------------------------------------
# Step 1 — collect Counter field metadata
# ---------------------------------------------------------------------------

def _collect_counter_fields(component_ir, ctx, py_cls) -> List[_CounterFieldInfo]:
    """Return metadata for every Counter sub-component field."""
    py_meta: Dict[str, dict] = {}
    if py_cls is not None and hasattr(py_cls, "__dataclass_fields__"):
        try:
            for df in _dc_mod.fields(py_cls):
                py_meta[df.name] = dict(df.metadata)
        except Exception:
            pass

    infos = []
    for idx, f in enumerate(component_ir.fields):
        dt = getattr(f, "datatype", None)
        if dt is None or type(dt).__name__ != "DataTypeRef":
            continue
        ref_name = getattr(dt, "ref_name", None)
        if ref_name not in _COUNTER_TYPE_NAMES:
            continue
        ref_ir = ctx.type_m.get(ref_name)
        if ref_ir is None or type(ref_ir).__name__ != "DataTypeComponent":
            continue

        is_modulo = ref_name in ("ModuloCounter", "WatchdogCounter")
        meta = py_meta.get(f.name, {})
        kwargs = meta.get("kwargs", {})

        if is_modulo:
            period = int(kwargs.get("PERIOD") or kwargs.get("TIMEOUT") or 256)
        else:
            width = int(kwargs.get("WIDTH", 8))
            period = 1 << width

        infos.append(_CounterFieldInfo(
            field_name=f.name,
            field_index=idx,
            period=period,
        ))

    return infos


# ---------------------------------------------------------------------------
# Step 2 — rewrite proc bodies: counter waits → await zdc.cycles(PERIOD)
# ---------------------------------------------------------------------------

def _rewrite_proc_bodies(component_ir, infos: List[_CounterFieldInfo]) -> None:
    """Rewrite counter wait calls in all @proc processes."""
    idx_to_info: Dict[int, _CounterFieldInfo] = {
        info.field_index: info for info in infos
    }
    for proc in getattr(component_ir, "proc_processes", []):
        new_body = _rewrite_stmts(proc.body, idx_to_info)
        proc.body.clear()
        proc.body.extend(new_body)


def _rewrite_stmts(stmts, idx_to_info: Dict[int, _CounterFieldInfo]) -> list:
    out = []
    for stmt in stmts:
        replaced = _try_rewrite_counter_wait(stmt, idx_to_info)
        if replaced is not None:
            out.append(replaced)
            continue

        t = type(stmt).__name__
        if t == "StmtWhile":
            new_body = _rewrite_stmts(stmt.body, idx_to_info)
            stmt.body.clear()
            stmt.body.extend(new_body)
        elif t == "StmtIf":
            stmt.body[:] = _rewrite_stmts(getattr(stmt, "body", []), idx_to_info)
            stmt.orelse[:] = _rewrite_stmts(getattr(stmt, "orelse", []), idx_to_info)

        out.append(stmt)
    return out


def _try_rewrite_counter_wait(stmt, idx_to_info):
    """Return a ``StmtExpr(await zdc.cycles(N))`` if *stmt* is a counter wait call.

    Matches: ``StmtExpr(ExprAwait(ExprCall(ExprAttribute(ExprRefField, 'wait_next'|'wait_for'))))``
    """
    if type(stmt).__name__ != "StmtExpr":
        return None
    expr = getattr(stmt, "expr", None)
    if type(expr).__name__ != "ExprAwait":
        return None
    call = getattr(expr, "value", None)
    if type(call).__name__ != "ExprCall":
        return None
    func = getattr(call, "func", None)
    if type(func).__name__ != "ExprAttribute":
        return None
    method_name = getattr(func, "attr", None)
    if method_name not in ("wait_next", "wait_for"):
        return None

    field_ref = getattr(func, "value", None)
    if type(field_ref).__name__ != "ExprRefField":
        return None
    field_idx = getattr(field_ref, "index", None)
    info = idx_to_info.get(field_idx)
    if info is None:
        return None

    if method_name == "wait_for":
        raise NotImplementedError(
            f"CounterLowerPass: await self.{info.field_name}.wait_for() is not "
            "supported for RTL synthesis. Use wait_next() instead."
        )

    # Replace with await zdc.cycles(PERIOD)
    cycles_call = ExprCall(
        func=ExprAttribute(
            value=ExprRefUnresolved(name="zdc"),
            attr="cycles",
        ),
        args=[ExprConstant(value=info.period)],
        keywords=[],
    )
    _log.debug(
        "CounterLowerPass: rewrote await %s.wait_next() → await zdc.cycles(%d)",
        info.field_name, info.period,
    )
    return StmtExpr(expr=ExprAwait(value=cycles_call))


# ---------------------------------------------------------------------------
# Step 3 — neutralise the counter field so _is_hierarchical ignores it
# ---------------------------------------------------------------------------

def _neutralize_field(component_ir, info: _CounterFieldInfo) -> None:
    """Set the counter field's datatype to None so it is ignored by downstream passes."""
    f = component_ir.fields[info.field_index]
    f.datatype = None
    _log.debug(
        "CounterLowerPass: neutralised field %r (index %d)",
        info.field_name, info.field_index,
    )
