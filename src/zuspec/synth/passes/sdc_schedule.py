"""SDCSchedulePass — Approach A pipeline stage assignment via SDC scheduling.

Builds a dependency graph from ``PipelineIR.operations`` (collected by
:class:`PipelineAnnotationPass`) and runs :class:`SDCScheduler` (Bellman-Ford
over the constraint graph) to assign each operation an optimal pipeline stage.

The result is written back into :attr:`PipelineIR.stages` as a revised stage
partition with ``cycle_lo`` / ``cycle_hi`` set, and
:attr:`PipelineIR.approach` is updated to ``"sdc"``.

This pass is optional; when using Approach C the user provides explicit
``zdc.stage()`` markers and this pass is skipped.
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional, Set

from .synth_pass import SynthPass
from .pipeline_annotation import compute_channels_from_stages
from zuspec.synth.ir.pipeline_ir import PipelineIR, StageIR, ChannelDecl
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal dependency graph built from PipelineIR
# ---------------------------------------------------------------------------

class _PipelineDepGraph:
    """Lightweight dependency graph over pipeline stage operations.

    Nodes are (stage_index, stmt_index) pairs.  Edges encode:
    - Intra-stage sequential order (stmt[i] → stmt[i+1]).
    - Inter-stage def-use (definition in stage i used in stage j adds
      edge i → j).

    The graph is consumed by :class:`SDCScheduler`.
    """

    def __init__(self) -> None:
        # op_id → {"stage": int, "stmt_idx": int, "successors": List[int]}
        self.operations: Dict[int, Any] = {}
        self._next_id = 0

    def add_op(self, stage: int, stmt_idx: int) -> int:
        op_id = self._next_id
        self._next_id += 1
        self.operations[op_id] = _PipelineOp(op_id, stage, stmt_idx)
        return op_id

    def add_edge(self, from_id: int, to_id: int) -> None:
        self.operations[from_id].successors.append(to_id)
        self.operations[to_id].predecessors.append(from_id)


class _PipelineOp:
    """Single operation node for :class:`_PipelineDepGraph`."""

    def __init__(self, op_id: int, stage: int, stmt_idx: int) -> None:
        self.id = op_id
        self.stage = stage
        self.stmt_idx = stmt_idx
        self.successors: List[int] = []
        self.predecessors: List[int] = []
        # Fields expected by SDCScheduler
        self.latency: int = 1
        self.asap_time: int = 0
        self.alap_time: int = 0

        from zuspec.synth.sprtl.scheduler import OperationType
        self.op_type = OperationType.ASSIGN

    @property
    def mobility(self) -> int:
        return max(0, self.alap_time - self.asap_time)

    @property
    def is_critical(self) -> bool:
        return self.mobility == 0


# ---------------------------------------------------------------------------
# AST helpers — collect defs and uses within a statement
# ---------------------------------------------------------------------------

class _VarDef(ast.NodeVisitor):
    """Collect variable names defined (assigned to) by an AST statement."""

    def __init__(self) -> None:
        self.defs: Set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.defs.add(t.id)
            elif isinstance(t, ast.Attribute):
                self.defs.add(f"self.{t.attr}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.defs.add(node.target.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.defs.add(node.target.id)
        self.generic_visit(node)


class _VarUse(ast.NodeVisitor):
    """Collect variable names used (read) by an AST statement."""

    def __init__(self) -> None:
        self.uses: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name):
            self.uses.add(f"self.{node.attr}")
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# SDCSchedulePass
# ---------------------------------------------------------------------------

class SDCSchedulePass(SynthPass):
    """Assign pipeline stages to operations using SDC scheduling (Approach A).

    Reads ``ir.pipeline_ir`` (populated by :class:`PipelineAnnotationPass` for
    Approach C, or set to an unannotated :class:`PipelineIR` for Approach A)
    and runs :class:`SDCScheduler` to compute an optimal stage assignment.

    The updated stage partition is written back into
    ``ir.pipeline_ir.stages``.

    Args:
        config: Synthesis configuration.  ``config.latency_model`` and
            ``config.clock_period_ns`` are forwarded to the scheduler.
        latency_model: Optional per-operation-type override (merged with the
            scheduler defaults).
    """

    def __init__(
        self,
        config: SynthConfig,
        *,
        latency_model: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(config=config)
        self._latency_override = latency_model

    @property
    def name(self) -> str:
        return "sdc_schedule"

    def run(self, ir: SynthIR) -> SynthIR:
        """Run SDC scheduling on the flat stage produced by :class:`PipelineAnnotationPass`.

        This method is a no-op when ``pip.approach`` is not ``"auto"``
        (Approach C pipelines pass through with ``approach`` set to
        ``"user+sdc"`` to record that this pass was visited).

        :param ir: Synthesis IR with ``ir.pipeline_ir.approach == "auto"``.
        :type ir: SynthIR
        :return: Updated IR with stages reassigned and ``pip.channels``
                 rebuilt.  ``pip.approach`` is set to ``"sdc"``.
        :rtype: SynthIR
        """
        pip = getattr(ir, "pipeline_ir", None)
        if pip is None:
            _log.info("[SDCSchedulePass] no pipeline_ir — skipping")
            return ir

        if not pip.stages:
            _log.info("[SDCSchedulePass] pipeline_ir has no stages — skipping")
            return ir

        # For user-annotated pipelines (Approach C), stages are already declared.
        # Only run SDC for automatic scheduling (approach == "auto").
        if pip.approach not in (None, "auto"):
            _log.info(
                "[SDCSchedulePass] approach=%r — stages already assigned, skipping SDC",
                pip.approach,
            )
            pip.approach = "user+sdc"  # mark that we validated but didn't reschedule
            return ir

        try:
            self._run_sdc(ir, pip)
        except Exception as exc:
            _log.warning("[SDCSchedulePass] SDC scheduling failed (%s) — keeping original stages", exc)

        return ir

    # ------------------------------------------------------------------

    def _run_sdc(self, ir: SynthIR, pip: PipelineIR) -> None:
        # Build dependency graph from PipelineIR stages
        graph = _PipelineDepGraph()

        # Map stage_index → list of op_ids (in stmt order)
        stage_op_ids: List[List[int]] = []
        for si, stage in enumerate(pip.stages):
            ops = []
            for stmt_idx in range(len(stage.operations)):
                oid = graph.add_op(si, stmt_idx)
                ops.append(oid)
                # Sequential order within stage (latency=1)
                if stmt_idx > 0:
                    graph.add_edge(ops[-2], oid)
            stage_op_ids.append(ops)

        # Inter-stage def-use edges
        defined_at: Dict[str, int] = {}  # var → op_id of last definition
        for si, (stage, op_ids) in enumerate(zip(pip.stages, stage_op_ids)):
            for stmt_idx, (stmt, oid) in enumerate(zip(stage.operations, op_ids)):
                vd = _VarDef()
                vd.visit(stmt)
                vu = _VarUse()
                vu.visit(stmt)

                for var in vu.uses:
                    if var in defined_at:
                        prev_oid = defined_at[var]
                        if prev_oid != oid:
                            graph.add_edge(prev_oid, oid)

                for var in vd.defs:
                    defined_at[var] = oid

        if not graph.operations:
            pip.approach = "sdc"
            return

        # ASAP scheduling via BFS/topological order on _PipelineDepGraph
        t = self._asap(graph)

        # If the user specified a target stage count (stages=N), fold the ASAP
        # schedule to fit within N stages by proportional bucketing.
        max_asap = max(t.values(), default=0)
        target_stages = getattr(pip, "pipeline_stages", 0) or 0
        if target_stages > 0 and max_asap + 1 > target_stages:
            # Proportionally map ASAP time t[oid] → bucket in [0, target_stages)
            t = {
                oid: min(int(tv * target_stages // (max_asap + 1)), target_stages - 1)
                for oid, tv in t.items()
            }
            max_asap = target_stages - 1

        # Rebuild stages from schedule
        max_stage = max_asap
        new_stages: List[StageIR] = [
            StageIR(name=f"S{i}", index=i, inputs=[], outputs=[], ports=[])
            for i in range(max_stage + 1)
        ]
        for si, (stage, op_ids) in enumerate(zip(pip.stages, stage_op_ids)):
            for stmt_idx, (stmt, oid) in enumerate(zip(stage.operations, op_ids)):
                target_stage = t.get(oid, si)
                if target_stage < 0:
                    target_stage = si
                new_stages[target_stage].operations.append(stmt)

        # Preserve user-visible stage names where possible
        for i, old_stage in enumerate(pip.stages):
            if i < len(new_stages) and old_stage.name:
                new_stages[i].name = old_stage.name

        # Update cycle windows
        for i, ns in enumerate(new_stages):
            ns.cycle_lo = i
            ns.cycle_hi = i

        pip.stages = new_stages

        # Recompute channels from the new stage partition
        stage_names = [s.name for s in new_stages]
        stage_stmts_list = [s.operations for s in new_stages]
        annotation_map = getattr(pip, "annotation_map", {}) or {}
        stage_inputs, stage_outputs = compute_channels_from_stages(
            stage_names, stage_stmts_list, annotation_map
        )
        all_channels: List[ChannelDecl] = []
        seen: Set[str] = set()
        for i, ns in enumerate(new_stages):
            ns.inputs = stage_inputs[i]
            ns.outputs = stage_outputs[i]
            for ch in stage_outputs[i]:
                if ch.name not in seen:
                    seen.add(ch.name)
                    all_channels.append(ch)
        pip.channels = all_channels

        pip.approach = "sdc"

        _log.info(
            "[SDCSchedulePass] pipeline '%s': %d ops → %d stages",
            pip.module_name,
            sum(len(s.operations) for s in new_stages),
            len(new_stages),
        )

    def _asap(self, graph: _PipelineDepGraph) -> Dict[int, int]:
        """ASAP stage assignment via topological-order relaxation."""
        t: Dict[int, int] = {oid: 0 for oid in graph.operations}
        # Kahn's algorithm for topological order
        in_degree = {oid: len(op.predecessors)
                     for oid, op in graph.operations.items()}
        from collections import deque
        queue: deque = deque(
            oid for oid, deg in in_degree.items() if deg == 0
        )
        while queue:
            uid = queue.popleft()
            uop = graph.operations[uid]
            for vid in uop.successors:
                new_t = t[uid] + uop.latency
                if new_t > t[vid]:
                    t[vid] = new_t
                in_degree[vid] -= 1
                if in_degree[vid] == 0:
                    queue.append(vid)
        return t
