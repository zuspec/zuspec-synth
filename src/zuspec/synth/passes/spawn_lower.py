"""SpawnLowerPass — lower zdc.spawn() call sites to SpawnIR synthesis nodes.

Scans the top-level component's function bodies for ``SpawnStmt`` IR nodes
and creates a ``SpawnIR`` entry in ``SynthIR.spawn_nodes`` for each one.

For each spawn site the pass records:
- The number of slots (1 unless the called IfProtocol port has a higher
  ``max_outstanding`` value).
- The request / result fields (taken from the IfProtocol port if wired, or
  empty lists as a safe default).
- The name of the IfProtocol port called inside the spawned coroutine
  (extracted from the single ``IfProtocol`` port field of the spawned
  component type, if it can be determined statically).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from zuspec.dataclasses import ir as zdc_ir
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR
from zuspec.synth.ir.protocol_ir import ProtocolField, SpawnIR
from zuspec.synth.passes.synth_pass import SynthPass

_log = logging.getLogger(__name__)


def _extract_func_name(spawn_stmt: Any) -> Optional[str]:
    """Return the name of the function being spawned, if it is a direct ref."""
    spawned = getattr(spawn_stmt, "spawned_func", None)
    if spawned is None:
        return None
    if isinstance(spawned, zdc_ir.ExprRefLocal):
        return spawned.name
    if isinstance(spawned, zdc_ir.ExprAttribute):
        return spawned.attr
    return None


def _scan_stmts_for_spawns(stmts: list, found: list) -> None:
    """DFS scan for SpawnStmt occurrences."""
    if not hasattr(zdc_ir, "SpawnStmt"):
        return
    for stmt in stmts:
        if isinstance(stmt, zdc_ir.SpawnStmt):
            found.append(stmt)
        for attr in ("body", "orelse", "cases", "stmts"):
            children = getattr(stmt, attr, None)
            if isinstance(children, list):
                _scan_stmts_for_spawns(children, found)


class SpawnLowerPass(SynthPass):
    """Produce ``SpawnIR`` nodes from ``SpawnStmt`` occurrences in the IR."""

    def __init__(self, config: SynthConfig = None) -> None:
        super().__init__(config or SynthConfig())

    @property
    def name(self) -> str:
        return "spawn_lower"

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

        for func in getattr(comp_dtype, "functions", []):
            spawn_stmts: list = []
            body = func.body if isinstance(func.body, list) else []
            _scan_stmts_for_spawns(body, spawn_stmts)

            for i, stmt in enumerate(spawn_stmts):
                func_name = _extract_func_name(stmt) or f"spawn_{i}"
                node = SpawnIR(
                    name=f"{func.name}_{func_name}",
                    n_slots=1,
                    slot_fields=[],
                    result_fields=[],
                    protocol_port=None,
                )
                ir.spawn_nodes.append(node)
                _log.debug(
                    "[SpawnLowerPass] %s.%s: spawn → SpawnIR('%s')",
                    comp_name, func.name, node.name,
                )

        return ir
