#****************************************************************************
# Copyright 2019-2026 Matthew Ballance and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#****************************************************************************
"""
ProcessBodyWalker — extracts pipeline stage structure from a @zdc.process body.

Uses the Python AST of the @zdc.process method (and each action's body() method)
directly, because DataModelFactory flattens async-with blocks when it inlines
action bodies, losing the concurrency structure needed for RTL synthesis.

The walker identifies these canonical patterns:

  FETCH:      ``await self.comp.<icache-like port>(addr)``
  DECODE:     ``await ActionType(args)(comp=self.comp)`` inside an action body
  REG_READ:   ``async with self.comp.<sched>.share(rs)`` and
              ``await self.comp.<regfile>.read_all(...)``
  EXECUTE:    ``async with self.comp.<alu/mul/fpu_pool>.lock() as claim``
              ``await claim.t.execute(...)``
  MEM_ACCESS: ``async with self.comp.<lsu_pool>.lock() as claim``
              ``await claim.t.access(...)``
  WRITEBACK:  (implicit — written outputs of the ExecuteInstruction action)
"""
from __future__ import annotations

import ast
import inspect
import logging
import sys
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

class PipelineStage(Enum):
    """Pipeline stage inferred from hardware operation patterns."""
    FETCH      = "fetch"
    DECODE     = "decode"
    REG_READ   = "reg_read"
    EXECUTE    = "execute"
    MEM_ACCESS = "mem_access"
    WRITEBACK  = "writeback"
    UNKNOWN    = "unknown"


@dataclass
class HwOp:
    """A single hardware operation found inside an action body."""
    stage: PipelineStage
    kind: str     # 'port_call' | 'action_call' | 'regfile_read' |
                  # 'pool_lock' | 'pool_exec' | 'hazard_check'
    resource: str # human-readable resource name (e.g. 'icache', 'alu_pool')
    detail: str = ""


@dataclass
class ActionInfo:
    """One action invocation found at the top level of a @zdc.process loop."""
    name: str                           # action class name, e.g. 'FetchNext'
    cls: Optional[type]                 # resolved Python class, or None
    ops: List[HwOp] = field(default_factory=list)
    stages: List[PipelineStage] = field(default_factory=list)  # ordered unique stages


@dataclass
class ProcessInfo:
    """All pipeline information extracted from a single @zdc.process method."""
    component_cls: type
    process_name: str
    actions: List[ActionInfo] = field(default_factory=list)
    all_stages: List[PipelineStage] = field(default_factory=list)  # global ordered unique stages


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

class ProcessBodyWalker:
    """
    Extracts pipeline stage structure from a @zdc.process method using Python AST.

    Usage::

        info = ProcessBodyWalker().walk(MyProcessor)
        # info.all_stages  → [PipelineStage.FETCH, DECODE, REG_READ, EXECUTE, ...]
        # info.actions[0].name  → 'FetchNext'
    """

    # Port names containing these substrings → stage mapping
    _FETCH_PORT_HINTS: tuple = ('icache', 'instr', 'fetch', 'imem')
    _MEM_PORT_HINTS:   tuple = ('dcache', 'data', 'dmem', 'lsu')

    # Pool field-name substrings → stage mapping
    _EXEC_POOL_HINTS: tuple = ('alu', 'mul', 'fpu', 'mul_div', 'execute')
    _MEM_POOL_HINTS:  tuple = ('lsu', 'dmem', 'dcache')

    # ---- public API -------------------------------------------------------

    def walk(self, component_cls: type) -> ProcessInfo:
        """
        Walk all @zdc.process descriptors on *component_cls*.

        Returns a :class:`ProcessInfo` describing the pipeline stages.
        """
        info = ProcessInfo(component_cls=component_cls, process_name='')

        for attr_name in dir(component_cls):
            try:
                attr = getattr(component_cls, attr_name, None)
            except Exception:
                continue
            if attr is not None and type(attr).__name__ == 'ExecProc':
                info.process_name = attr_name
                self._walk_process(info, attr.method, component_cls)
                break  # only the first @zdc.process

        # Compute ordered unique stages across all actions
        seen: set = set()
        for action in info.actions:
            for stage in action.stages:
                if stage not in seen:
                    seen.add(stage)
                    info.all_stages.append(stage)

        return info

    # ---- process level ----------------------------------------------------

    def _walk_process(self, info: ProcessInfo, method, component_cls: type) -> None:
        """Parse the process method AST, find action invocations in the loop body."""
        try:
            src = textwrap.dedent(inspect.getsource(method))
            tree = ast.parse(src)
        except Exception as exc:
            _log.warning("ProcessBodyWalker: cannot parse %s: %s", info.process_name, exc)
            return

        func_node = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)),
            None,
        )
        if func_node is None:
            return

        # Prefer the body of the ``while True:`` loop; fall back to whole body
        while_node = next(
            (n for n in func_node.body if isinstance(n, ast.While) and _is_const_true(n.test)),
            None,
        )
        loop_body = while_node.body if while_node is not None else func_node.body

        ns = self._namespace(component_cls)
        for node in loop_body:
            action = self._try_extract_top_level_action(node, ns)
            if action is not None:
                info.actions.append(action)

    # ---- action invocation detection --------------------------------------

    def _try_extract_top_level_action(
        self, node: ast.AST, ns: dict
    ) -> Optional[ActionInfo]:
        """
        Detect ``result = await ActionType(field=val)(comp=self)``
        or       ``await ActionType(field=val)(comp=self)``.

        AST shape::

            (Assign|Expr)(
              value=Await(
                Call(
                  func=Call(func=Name(id='ActionType'), keywords=[...]),
                  keywords=[keyword(arg='comp')])))
        """
        expr = _unwrap_assign_expr(node)
        if expr is None or not isinstance(expr, ast.Await):
            return None

        outer_call = expr.value
        if not isinstance(outer_call, ast.Call):
            return None
        if not any(k.arg == 'comp' for k in outer_call.keywords):
            return None

        inner_call = outer_call.func
        if not isinstance(inner_call, ast.Call):
            return None
        if not isinstance(inner_call.func, ast.Name):
            return None

        action_name = inner_call.func.id
        action_cls = ns.get(action_name)

        action_info = ActionInfo(name=action_name, cls=action_cls)
        if action_cls is not None:
            self._walk_action_body(action_info, action_cls, ns)
        return action_info

    # ---- action body walking ----------------------------------------------

    def _walk_action_body(
        self, action_info: ActionInfo, action_cls: type, ns: dict
    ) -> None:
        """Parse ``action_cls.body()`` AST and collect :class:`HwOp` items."""
        body_method = getattr(action_cls, 'body', None)
        if body_method is None:
            return
        try:
            src = textwrap.dedent(inspect.getsource(body_method))
            tree = ast.parse(src)
        except Exception as exc:
            _log.warning(
                "ProcessBodyWalker: cannot parse %s.body: %s", action_info.name, exc
            )
            return

        func_node = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
            ),
            None,
        )
        if func_node is None:
            return

        self._collect_ops(action_info, func_node.body, ns, depth=0)

        # Compute ordered unique stages
        seen: set = set()
        for op in action_info.ops:
            if op.stage not in seen:
                seen.add(op.stage)
                action_info.stages.append(op.stage)

    def _collect_ops(
        self,
        action_info: ActionInfo,
        stmts: list,
        ns: dict,
        depth: int,
    ) -> None:
        """Recursively walk a statement list and collect :class:`HwOp` items."""
        for stmt in stmts:
            # --- await expressions ---
            expr_val = _unwrap_assign_expr(stmt)
            if isinstance(expr_val, ast.Await):
                op = self._classify_await(expr_val.value)
                if op is not None:
                    action_info.ops.append(op)

            # --- async with ---
            elif isinstance(stmt, ast.AsyncWith):
                for item in stmt.items:
                    op = self._classify_async_with(item.context_expr)
                    if op is not None:
                        action_info.ops.append(op)
                self._collect_ops(action_info, stmt.body, ns, depth + 1)

            # --- control flow ---
            elif isinstance(stmt, ast.If):
                self._collect_ops(action_info, stmt.body, ns, depth + 1)
                self._collect_ops(action_info, stmt.orelse, ns, depth + 1)

            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    self._collect_ops(action_info, case.body, ns, depth + 1)

    # ---- operation classification -----------------------------------------

    def _classify_await(self, call_node: ast.AST) -> Optional[HwOp]:
        """Classify ``await <call_node>`` → :class:`HwOp` or ``None``."""
        if not isinstance(call_node, ast.Call):
            return None

        func = call_node.func

        # Pattern A: await self.comp.<port>(args)  — direct port call
        port_name = _extract_comp_field_call(func, via='comp')
        if port_name is not None:
            stage = self._stage_for_port(port_name)
            return HwOp(stage=stage, kind='port_call', resource=port_name)

        # Pattern B: await ActionType(args)(comp=self.comp) — nested action
        if (
            isinstance(func, ast.Call)
            and isinstance(func.func, ast.Name)
            and any(k.arg == 'comp' for k in call_node.keywords)
        ):
            return HwOp(
                stage=PipelineStage.DECODE,
                kind='action_call',
                resource=func.func.id,
            )

        # Pattern C: await self.comp.<regfile>.read_all(...)
        chain = _attr_chain(func)
        if chain is not None and 'read_all' in chain:
            idx = chain.index('read_all')
            resource = chain[idx - 1] if idx > 0 else 'regfile'
            return HwOp(stage=PipelineStage.REG_READ, kind='regfile_read', resource=resource)

        # Pattern D: await claim.t.execute(...)  or  await claim.t.access(...)
        if chain is not None and chain and chain[-1] in ('execute', 'access'):
            # Determine stage from the outer async-with (we don't have that context here,
            # so default to EXECUTE; MEM_ACCESS is set from the enclosing pool_lock op).
            return HwOp(stage=PipelineStage.EXECUTE, kind='pool_exec', resource='.'.join(chain))

        return None

    def _classify_async_with(self, ctx_expr: ast.AST) -> Optional[HwOp]:
        """Classify ``async with <ctx_expr>`` → :class:`HwOp` or ``None``."""
        # ctx_expr may be a Call or an attribute chain
        if isinstance(ctx_expr, ast.Call):
            chain = _attr_chain(ctx_expr.func)
        else:
            chain = _attr_chain(ctx_expr)

        if chain is None or not chain:
            return None

        method = chain[-1]

        if method == 'share':
            resource = _after(chain, 'comp') or 'sched'
            return HwOp(stage=PipelineStage.REG_READ, kind='hazard_check', resource=resource)

        if method == 'lock':
            resource = _after(chain, 'comp') or 'pool'
            lower = resource.lower()
            if any(h in lower for h in self._MEM_POOL_HINTS):
                return HwOp(stage=PipelineStage.MEM_ACCESS, kind='pool_lock', resource=resource)
            if any(h in lower for h in self._EXEC_POOL_HINTS):
                return HwOp(stage=PipelineStage.EXECUTE, kind='pool_lock', resource=resource)
            # rd_sched.lock — write-hazard reservation, part of EXECUTE setup
            return HwOp(stage=PipelineStage.EXECUTE, kind='hazard_check', resource=resource)

        return None

    # ---- helpers ----------------------------------------------------------

    def _stage_for_port(self, port_name: str) -> PipelineStage:
        lower = port_name.lower()
        if any(h in lower for h in self._FETCH_PORT_HINTS):
            return PipelineStage.FETCH
        if any(h in lower for h in self._MEM_PORT_HINTS):
            return PipelineStage.MEM_ACCESS
        return PipelineStage.UNKNOWN

    def _namespace(self, component_cls: type) -> dict:
        """Build a name→class mapping for action resolution."""
        mod = sys.modules.get(component_cls.__module__, None)
        return dict(vars(mod)) if mod else {}


# ---------------------------------------------------------------------------
# Private AST helpers
# ---------------------------------------------------------------------------

def _is_const_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _unwrap_assign_expr(node: ast.AST) -> Optional[ast.AST]:
    """Return the value expression of an Assign or Expr node, else None."""
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, (ast.AnnAssign, ast.Expr)):
        return node.value
    return None


def _attr_chain(node: ast.AST) -> Optional[List[str]]:
    """
    Extract ``a.b.c.d`` → ``['a', 'b', 'c', 'd']``.
    Returns ``None`` if the node is not a pure attribute chain.
    """
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _extract_comp_field_call(func: ast.AST, via: str) -> Optional[str]:
    """
    Detect ``self.<via>.<field>`` as the function of a call.

    Requires exactly 3 chain elements so that deeper chains like
    ``self.comp.regfile.read_all`` are not misidentified as port calls.
    Returns ``<field>`` or ``None``.
    """
    chain = _attr_chain(func)
    if chain is None or len(chain) != 3:
        return None
    if chain[0] == 'self' and chain[1] == via:
        return chain[2]
    return None


def _after(chain: List[str], marker: str) -> Optional[str]:
    """Return the element immediately after *marker* in *chain*."""
    try:
        idx = chain.index(marker)
        if idx + 1 < len(chain):
            return chain[idx + 1]
    except ValueError:
        pass
    return None
