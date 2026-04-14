"""SyncBodyLowerPass — substitute DSL queries in ``@zdc.sync`` method bodies.

For each ``SyncIR`` in ``pip.sync_irs``, this pass rewrites ``zdc.stage.*``
query calls in the body AST with the corresponding Verilog signal names:

* ``zdc.stage.valid(self.X)``   → string ``"{x}_valid_q"``
* ``zdc.stage.ready(self.X)``   → string ``"(~{x}_valid_q | ~{x}_stalled)"``
* ``zdc.stage.stalled(self.X)`` → string ``"{x}_stalled"``
* ``zdc.stage.flush(self.X, cond)`` → record ``FlushDeclNode`` in
  ``SyncIR.flush_decls`` (the stall-gen pass picks these up later)

The same substitution is also applied inside ``@zdc.stage`` bodies so that
cross-stage queries work uniformly.

The lowered signal-name strings are stored as ``zdc.stage.*`` AST
:class:`ast.Constant` nodes replacing the original call nodes (i.e. the
body AST is mutated in-place).
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional

from .synth_pass import SynthPass
from zuspec.synth.ir.pipeline_ir import PipelineIR, SyncIR
from zuspec.synth.ir.synth_ir import SynthIR

_log = logging.getLogger(__name__)


def _stage_lower(name: str) -> str:
    return name.lower()


class _DslQueryRewriter(ast.NodeTransformer):
    """Replace ``zdc.stage.valid/ready/stalled(self.X)`` with string constants."""

    def __init__(self, stage_names: List[str], flush_decls: list) -> None:
        self._stage_names = {s.upper(): s for s in stage_names}
        self._flush_decls = flush_decls  # mutable list; we append FlushDeclNode here

    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)
        # Match:  <receiver>.valid(self.X)
        #         <receiver>.ready(self.X)
        #         <receiver>.stalled(self.X)
        #         <receiver>.flush(self.X, cond)
        if not isinstance(node.func, ast.Attribute):
            return node
        method = node.func.attr
        if method not in ('valid', 'ready', 'stalled', 'flush'):
            return node
        receiver = node.func.value
        # Accept zdc.stage.X or stage.X
        receiver_ok = (
            isinstance(receiver, ast.Attribute) and receiver.attr == 'stage'
        ) or (
            isinstance(receiver, ast.Name) and receiver.id == 'stage'
        )
        if not receiver_ok:
            return node

        if method in ('valid', 'ready', 'stalled'):
            # First arg should be self.STAGE
            target = node.args[0] if node.args else None
            stage_name = None
            if isinstance(target, ast.Attribute):
                stage_name = target.attr.lower()
            elif isinstance(target, ast.Name):
                stage_name = target.id.lower()
            if stage_name is None:
                return node
            if method == 'valid':
                sig = f"{stage_name}_valid_q"
            elif method == 'stalled':
                sig = f"{stage_name}_stalled"
            else:  # ready
                sig = f"(~{stage_name}_valid_q | ~{stage_name}_stalled)"
            return ast.copy_location(ast.Constant(value=sig), node)

        if method == 'flush':
            # flush(self.X, cond)
            target_ast = node.args[0] if node.args else None
            cond_ast = node.args[1] if len(node.args) > 1 else (
                node.keywords[0].value if node.keywords else None)
            target_name = None
            if isinstance(target_ast, ast.Attribute):
                target_name = target_ast.attr
            try:
                from zuspec.dataclasses.ir.pipeline import FlushDeclNode
                self._flush_decls.append(FlushDeclNode(
                    target_stage=target_name,
                    cond_ast=cond_ast,
                ))
            except ImportError:
                pass
            # Replace with a no-op constant (flush is handled elsewhere)
            return ast.copy_location(ast.Constant(value=0), node)

        return node


class SyncBodyLowerPass(SynthPass):
    """Substitute DSL query calls in sync/stage method bodies.

    This pass is a no-op when ``ir.pipeline_ir`` is ``None`` or has no sync IRs.
    """

    @property
    def name(self) -> str:
        return "sync_body_lower"

    def run(self, ir: SynthIR) -> SynthIR:
        """Rewrite DSL queries in sync and stage bodies.

        :param ir: Synthesis IR with ``ir.pipeline_ir`` set.
        :type ir: SynthIR
        :return: Updated IR (body ASTs mutated in-place).
        :rtype: SynthIR
        """
        if ir.pipeline_ir is None:
            _log.debug("[SyncBodyLowerPass] no pipeline_ir — skipping")
            return ir

        pip = ir.pipeline_ir
        if not pip.sync_irs:
            _log.debug("[SyncBodyLowerPass] no sync_irs — skipping")
            return ir

        stage_names = [s.name for s in pip.stages]

        for sync in pip.sync_irs:
            if sync.body_ast is None:
                continue
            rewriter = _DslQueryRewriter(stage_names, sync.flush_decls)
            sync.body_ast = rewriter.visit(sync.body_ast)
            ast.fix_missing_locations(sync.body_ast)
            _log.debug("[SyncBodyLowerPass] lowered sync body '%s'", sync.name)

        _log.info("[SyncBodyLowerPass] lowered %d sync method(s) in %s",
                  len(pip.sync_irs), pip.module_name)
        return ir
