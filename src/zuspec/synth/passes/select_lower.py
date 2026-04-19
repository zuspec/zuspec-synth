"""SelectLowerPass — lower zdc.select() call sites to SelectIR synthesis nodes.

Scans the top-level component's function bodies for ``SelectStmt`` IR nodes
and creates a ``SelectIR`` entry in ``SynthIR.select_nodes`` for each one.

Each select statement has one or more ``(queue_name, handler)`` branches.
The pass records the queue names and assigns each an integer tag value (its
index in the branch list) for use by the arbiter in SV codegen.

Policy: by default the arbiter uses a fixed priority (branch 0 wins).
If more than one branch exists and the component type's ``select_round_robin``
attribute is truthy, a round-robin arbiter is requested instead.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from zuspec.dataclasses import ir as zdc_ir
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR
from zuspec.synth.ir.protocol_ir import SelectBranchIR, SelectIR
from zuspec.synth.passes.synth_pass import SynthPass

_log = logging.getLogger(__name__)


def _attr_name(expr: Any) -> Optional[str]:
    if isinstance(expr, zdc_ir.ExprAttribute):
        return expr.attr
    if isinstance(expr, (zdc_ir.ExprRefLocal, zdc_ir.ExprRefUnresolved)):
        return expr.name
    return None


def _scan_stmts_for_selects(stmts: list, found: list) -> None:
    """DFS scan for SelectStmt."""
    if not hasattr(zdc_ir, "SelectStmt"):
        return
    for stmt in stmts:
        if isinstance(stmt, zdc_ir.SelectStmt):
            found.append(stmt)
        for attr in ("body", "orelse", "cases", "stmts", "branches"):
            children = getattr(stmt, attr, None)
            if isinstance(children, list):
                _scan_stmts_for_selects(children, found)


class SelectLowerPass(SynthPass):
    """Produce ``SelectIR`` nodes from ``SelectStmt`` occurrences."""

    def __init__(self, config: SynthConfig = None) -> None:
        super().__init__(config or SynthConfig())

    @property
    def name(self) -> str:
        return "select_lower"

    def run(self, ir: SynthIR) -> SynthIR:
        ctxt = ir.model_context
        if ctxt is None:
            return ir

        comp_cls = ir.component
        comp_name = getattr(comp_cls, "__name__", None) if comp_cls else None
        if comp_name is None:
            return ir

        comp_dtype = None
        if hasattr(ctxt, "type_m"):
            for name, dtype in ctxt.type_m.items():
                if name == comp_name and isinstance(dtype, zdc_ir.DataTypeComponent):
                    comp_dtype = dtype
                    break
        if comp_dtype is None:
            return ir

        round_robin_default = bool(getattr(comp_cls, "select_round_robin", False))

        select_counter = 0
        for func in getattr(comp_dtype, "functions", []):
            select_stmts: list = []
            body = func.body if isinstance(func.body, list) else []
            _scan_stmts_for_selects(body, select_stmts)

            for stmt in select_stmts:
                branches_raw = getattr(stmt, "branches", [])
                branches: List[SelectBranchIR] = []
                for tag, branch in enumerate(branches_raw):
                    q_name = None
                    # Branch may have a queue_expr attribute, or be a tuple
                    if hasattr(branch, "queue_expr"):
                        q_name = _attr_name(branch.queue_expr)
                    elif isinstance(branch, (list, tuple)) and len(branch) >= 1:
                        q_name = _attr_name(branch[0])

                    if q_name is None:
                        q_name = f"<unknown_queue_{tag}>"
                    branches.append(SelectBranchIR(queue_name=q_name, tag_value=tag))

                node = SelectIR(
                    name=f"{func.name}_select_{select_counter}",
                    branches=branches,
                    round_robin=round_robin_default,
                )
                ir.select_nodes.append(node)
                select_counter += 1
                _log.debug(
                    "[SelectLowerPass] %s.%s: select(%d branches) → SelectIR('%s')",
                    comp_name, func.name, len(branches), node.name,
                )

        return ir
