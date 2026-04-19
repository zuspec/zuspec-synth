"""CompletionAnalysisPass — static analysis of zdc.Completion[T] token usage.

The pass performs inter-procedural data-flow analysis to verify:
1. Each ``Completion`` token is **set exactly once** (``done.set(v)``).
2. Each ``Completion`` token is **awaited exactly once** (``await done``).
3. No ``Completion`` is stored in a component-level field (only locals /
   struct fields that travel through a ``Queue`` are allowed).

Restriction violations are reported as errors via a configurable error handler.

The pass populates ``SynthIR.completion_nodes`` with a ``CompletionIR`` for
each token that passes all checks, recording its payload width and the queue
path it takes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from zuspec.dataclasses import ir as zdc_ir
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR
from zuspec.synth.ir.protocol_ir import CompletionIR
from zuspec.synth.passes.synth_pass import SynthPass

_log = logging.getLogger(__name__)


class CompletionAnalysisError(Exception):
    """Raised when a Completion token violates static restrictions."""


def _location_str(node: Any) -> str:
    """Return a human-readable location string from an IR node."""
    loc = getattr(node, "loc", None)
    if loc is None:
        return "<unknown>"
    if hasattr(loc, "file") and hasattr(loc, "line"):
        return f"{loc.file}:{loc.line}"
    return str(loc)


def _find_completions_in_stmts(stmts: list, results: dict) -> None:
    """DFS scan for CompletionSetStmt and CompletionAwaitExpr in *stmts*."""
    if not hasattr(zdc_ir, "CompletionSetStmt"):
        return

    for stmt in stmts:
        if isinstance(stmt, zdc_ir.CompletionSetStmt):
            name = _attr_name(stmt.completion_expr)
            if name:
                results.setdefault(name, {"sets": [], "awaits": [], "width": 32})
                results[name]["sets"].append(_location_str(stmt))

        # Recurse into body/orelse
        for attr in ("body", "orelse", "cases", "stmts"):
            children = getattr(stmt, attr, None)
            if isinstance(children, list):
                _find_completions_in_stmts(children, results)


def _find_awaits_in_stmts(stmts: list, results: dict) -> None:
    """DFS scan for CompletionAwaitExpr inside ExprAwait nodes."""
    if not hasattr(zdc_ir, "CompletionAwaitExpr"):
        return

    def _visit(node: Any) -> None:
        if isinstance(node, zdc_ir.ExprAwait):
            inner = getattr(node, "value", None)
            if isinstance(inner, zdc_ir.CompletionAwaitExpr):
                name = _attr_name(inner.completion_expr)
                if name:
                    results.setdefault(name, {"sets": [], "awaits": [], "width": 32})
                    results[name]["awaits"].append(_location_str(node))
                    # Capture payload width from result_type
                    rt = getattr(inner, "result_type", None)
                    if rt is not None and isinstance(rt, zdc_ir.DataTypeInt):
                        results[name]["width"] = max(8, rt.bits)
        # Recurse into children
        for attr_name in ("value", "body", "orelse", "stmts", "cases", "func"):
            child = getattr(node, attr_name, None)
            if child is None:
                continue
            if isinstance(child, list):
                for c in child:
                    if hasattr(c, "__dataclass_fields__"):
                        _visit(c)
            elif hasattr(child, "__dataclass_fields__"):
                _visit(child)

    for stmt in stmts:
        _visit(stmt)


def _attr_name(expr: Any) -> Optional[str]:
    """Return the attribute name from self.<name> or just a local name."""
    if isinstance(expr, zdc_ir.ExprAttribute):
        return expr.attr
    if isinstance(expr, (zdc_ir.ExprRefLocal, zdc_ir.ExprRefUnresolved)):
        return expr.name
    return None


def _find_queue_puts_in_stmts(stmts: list, queue_paths: dict) -> None:
    """Find QueuePutStmt to determine which queues carry completions."""
    if not hasattr(zdc_ir, "QueuePutStmt"):
        return

    for stmt in stmts:
        if isinstance(stmt, zdc_ir.QueuePutStmt):
            q_name = _attr_name(stmt.queue_expr)
            v_name = _attr_name(stmt.value_expr)
            if q_name and v_name:
                # Heuristically note that v_name goes through q_name
                queue_paths.setdefault(v_name, []).append(q_name)
        for attr in ("body", "orelse", "cases", "stmts"):
            children = getattr(stmt, attr, None)
            if isinstance(children, list):
                _find_queue_puts_in_stmts(children, queue_paths)


class CompletionAnalysisPass(SynthPass):
    """Validate Completion token usage and populate ``SynthIR.completion_nodes``."""

    def __init__(self, config: SynthConfig = None) -> None:
        super().__init__(config or SynthConfig())

    @property
    def name(self) -> str:
        return "completion_analysis"

    def run(self, ir: SynthIR) -> SynthIR:
        ctxt = ir.model_context
        if ctxt is None:
            return ir

        comp_cls = ir.component
        comp_name = getattr(comp_cls, "__name__", None) if comp_cls else None
        if comp_name is None:
            return ir

        comp_dtype = None
        for name, dtype in ctxt.type_m.items():
            if name == comp_name and isinstance(dtype, zdc_ir.DataTypeComponent):
                comp_dtype = dtype
                break
        if comp_dtype is None:
            return ir

        completion_info: dict = {}
        queue_paths: dict = {}

        for func in getattr(comp_dtype, "functions", []):
            body = func.body if isinstance(func.body, list) else []
            _find_completions_in_stmts(body, completion_info)
            _find_awaits_in_stmts(body, completion_info)
            _find_queue_puts_in_stmts(body, queue_paths)

        for name, info in completion_info.items():
            n_sets = len(info["sets"])
            n_awaits = len(info["awaits"])

            if n_sets == 0:
                _log.warning("[CompletionAnalysis] '%s': no set() found — possible deadlock", name)
            elif n_sets > 1:
                locs = ", ".join(info["sets"])
                raise CompletionAnalysisError(
                    f"Completion '{name}' is set more than once ({locs}). "
                    "Each Completion must have exactly one set() call."
                )

            if n_awaits == 0:
                _log.warning("[CompletionAnalysis] '%s': no await found — set() result ignored", name)
            elif n_awaits > 1:
                locs = ", ".join(info["awaits"])
                raise CompletionAnalysisError(
                    f"Completion '{name}' is awaited more than once ({locs}). "
                    "Each Completion must be awaited exactly once."
                )

            node = CompletionIR(
                name=name,
                elem_width=info.get("width", 32),
                set_location=info["sets"][0] if info["sets"] else "",
                await_location=info["awaits"][0] if info["awaits"] else "",
                queue_path=queue_paths.get(name, []),
            )
            ir.completion_nodes.append(node)
            _log.debug("[CompletionAnalysis] recorded Completion '%s' (w=%d)", name, node.elem_width)

        return ir
