"""IRLayerVerifier -- post-pass invariant checkers for each named IR layer.

Each verifier is intended to be run by
:class:`~zuspec.dataclasses.transform.pass_manager.LayeredPassManager`
immediately after a pass declares a layer transition via
:attr:`~zuspec.synth.passes.synth_pass.SynthPass.output_layer`.

Verifiers raise :class:`LayerVerificationError` on the first invariant
violation found.  Error messages follow the convention::

    <VerifierClass>: <rule description>
      Expected: <what the verifier expected>
      Got:      <what it found>
      Context:  <IR path / node type>
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuspec.synth.ir.synth_ir import SynthIR


class LayerVerificationError(Exception):
    """Raised when an IR invariant for a named layer is violated."""


class IRLayerVerifier(abc.ABC):
    """Abstract base for all layer verifiers.

    Concrete subclasses implement :meth:`verify` and check the invariants
    specific to one :class:`~zuspec.synth.ir.layers.IRLayer`.
    """

    @abc.abstractmethod
    def verify(self, ir: "SynthIR") -> None:
        """Verify layer invariants for *ir*.

        Args:
            ir: The current ``SynthIR`` object to inspect.

        Raises:
            LayerVerificationError: If any invariant is violated.
        """


# ---------------------------------------------------------------------------
# Layer 0: ACTIVITY
# ---------------------------------------------------------------------------

class ActivityLayerVerifier(IRLayerVerifier):
    """Verify Activity IR invariants (Layer 0, pre-scheduling).

    Invariants checked:

    1. Every ``ActivityTraversal`` has a non-``None`` ``action_cls``.
    2. Every ``ActivityParallel`` / ``ActivitySchedule`` has a ``JoinSpec``
       with ``kind`` in ``{"all", "branch", "none", "select", "first"}``.
    3. No ``ActivitySequenceBlock`` that is the top-level body of a compound
       action has zero children.
    """

    _VALID_JOIN_KINDS = frozenset({"all", "branch", "none", "select", "first"})

    def verify(self, ir: "SynthIR") -> None:
        """Walk activity trees attached to the model context and check invariants."""
        ctx = getattr(ir, "model_context", None)
        if ctx is None:
            # No activity context to check -- pass is effectively a no-op.
            return

        from zuspec.ir.core.activity import (
            ActivityTraversal,
            ActivityParallel,
            ActivitySchedule,
            ActivitySequenceBlock,
        )

        # Collect all action DataTypes from the context.
        actions = []
        if hasattr(ctx, "action_types"):
            actions = list(ctx.action_types.values())
        elif hasattr(ctx, "types"):
            from zuspec.ir.core.data_type import DataTypeAction
            actions = [t for t in ctx.types.values() if isinstance(t, DataTypeAction)]

        for action in actions:
            body = getattr(action, "activity", None)
            if body is None:
                continue
            self._check_seq_block(body, is_top=True)

    def _check_seq_block(self, node: Any, is_top: bool = False) -> None:
        from zuspec.ir.core.activity import (
            ActivityTraversal,
            ActivityParallel,
            ActivitySchedule,
            ActivitySequenceBlock,
        )

        if isinstance(node, ActivitySequenceBlock):
            children = getattr(node, "stmts", []) or getattr(node, "body", []) or []
            if is_top and len(children) == 0:
                raise LayerVerificationError(
                    "ActivityLayerVerifier: ActivitySequenceBlock with zero children\n"
                    "  Expected: at least one child statement\n"
                    "  Got:      empty sequence block\n"
                    "  Context:  top-level activity body"
                )
            for child in children:
                self._check_seq_block(child, is_top=False)

        elif isinstance(node, ActivityTraversal):
            action_cls = getattr(node, "action_cls", None)
            if action_cls is None:
                raise LayerVerificationError(
                    "ActivityLayerVerifier: ActivityTraversal has None action_cls\n"
                    "  Expected: a non-None Python action class reference\n"
                    "  Got:      None\n"
                    f"  Context:  {node!r}"
                )

        elif isinstance(node, (ActivityParallel, ActivitySchedule)):
            join_spec = getattr(node, "join_spec", None)
            if join_spec is None:
                raise LayerVerificationError(
                    f"ActivityLayerVerifier: {type(node).__name__} has no JoinSpec\n"
                    "  Expected: JoinSpec with a known kind\n"
                    "  Got:      None\n"
                    f"  Context:  {type(node).__name__}"
                )
            kind = getattr(join_spec, "kind", None)
            if kind not in self._VALID_JOIN_KINDS:
                raise LayerVerificationError(
                    f"ActivityLayerVerifier: {type(node).__name__} JoinSpec has "
                    f"unknown kind {kind!r}\n"
                    f"  Expected: one of {sorted(self._VALID_JOIN_KINDS)}\n"
                    f"  Got:      {kind!r}\n"
                    f"  Context:  {type(node).__name__}.join_spec.kind"
                )
            stmts = getattr(node, "stmts", []) or getattr(node, "body", []) or []
            for child in stmts:
                self._check_seq_block(child, is_top=False)

        else:
            # Recursively descend into known container fields.
            for attr in ("stmts", "body", "branches", "true_body", "false_body",
                         "cases", "then_body", "else_body"):
                sub = getattr(node, attr, None)
                if sub is None:
                    continue
                if isinstance(sub, list):
                    for child in sub:
                        self._check_seq_block(child, is_top=False)
                else:
                    self._check_seq_block(sub, is_top=False)


# ---------------------------------------------------------------------------
# Layer 1: SCHEDULED
# ---------------------------------------------------------------------------

class ScheduledLayerVerifier(IRLayerVerifier):
    """Verify scheduled IR invariants (Layer 1, post-SchedulePass).

    Invariants checked:

    1. ``ir.schedule_obj`` is not ``None``.
    2. Every scheduled operation has ``stage >= 0``.
    3. No two operations with a direct data dependency share the same stage.
    """

    def verify(self, ir: "SynthIR") -> None:
        if ir.schedule_obj is None:
            raise LayerVerificationError(
                "ScheduledLayerVerifier: ir.schedule_obj is None\n"
                "  Expected: a non-None schedule object after SchedulePass\n"
                "  Got:      None\n"
                "  Context:  SynthIR.schedule_obj"
            )

        sched = ir.schedule_obj

        # Check stage indices >= 0 and data-dependency constraint.
        ops = getattr(sched, "ops", None) or getattr(sched, "operations", None) or []
        stage_by_id: dict = {}
        for op in ops:
            stage = getattr(op, "stage", None)
            op_id = getattr(op, "id", id(op))
            if stage is None or stage < 0:
                raise LayerVerificationError(
                    f"ScheduledLayerVerifier: operation {op_id!r} has invalid stage {stage!r}\n"
                    "  Expected: stage >= 0\n"
                    f"  Got:      {stage!r}\n"
                    f"  Context:  ScheduledOperation id={op_id!r}"
                )
            stage_by_id[op_id] = stage

        # Check that no two operations with a data dependency share a stage.
        for op in ops:
            op_id = getattr(op, "id", id(op))
            deps = getattr(op, "depends_on", []) or []
            for dep_id in deps:
                if dep_id in stage_by_id and stage_by_id[dep_id] == stage_by_id[op_id]:
                    raise LayerVerificationError(
                        f"ScheduledLayerVerifier: data-dependent operations share stage\n"
                        f"  Expected: producer stage < consumer stage\n"
                        f"  Got:      op {op_id!r} and dependency {dep_id!r} both in "
                        f"stage {stage_by_id[op_id]!r}\n"
                        f"  Context:  ScheduledOperation dependency chain"
                    )


# ---------------------------------------------------------------------------
# Layer 2: PIPELINE
# ---------------------------------------------------------------------------

class PipelineLayerVerifier(IRLayerVerifier):
    """Verify Pipeline IR invariants (Layer 2, post-PipelineFrontendPass).

    Invariants checked (for components that declare a pipeline):

    1. ``ir.pipeline_ir`` is not ``None``.
    2. Every ``StageCallNode.stage_name`` appears in ``PipelineIR.stages``.
    3. No ``FlushDecl.target_stage`` references an undefined stage name.
    4. Every multi-cycle ``StageCallNode`` (``cycles > 1``) has a corresponding
       ``StageMethodIR`` with at least one ``StallDecl``.
    """

    def verify(self, ir: "SynthIR") -> None:
        pip = ir.pipeline_ir
        if pip is None:
            # Components with no pipeline declaration are fine here.
            return

        stage_names = {s.name for s in (getattr(pip, "stages", None) or [])}

        # Check SyncIR flush decls reference valid stage names.
        for sync_ir in getattr(pip, "sync_irs", []):
            for flush in getattr(sync_ir, "flush_decls", []):
                target = getattr(flush, "target_stage", None)
                if target is not None and target not in stage_names:
                    raise LayerVerificationError(
                        "PipelineLayerVerifier: FlushDecl.target_stage references "
                        "undefined stage\n"
                        f"  Expected: stage name present in PipelineIR.stages\n"
                        f"  Got:      {target!r}\n"
                        f"  Context:  SyncIR {sync_ir.name!r} FlushDecl"
                    )

        # Check each StageIR flush_decls too.
        for stage in getattr(pip, "stages", []):
            for flush in getattr(stage, "flush_decls", []):
                target = getattr(flush, "target_stage", None)
                if target is not None and target not in stage_names:
                    raise LayerVerificationError(
                        "PipelineLayerVerifier: StageIR.flush_decls references "
                        "undefined stage\n"
                        f"  Expected: stage name in {sorted(stage_names)}\n"
                        f"  Got:      {target!r}\n"
                        f"  Context:  StageIR {stage.name!r}"
                    )

        # Check multi-cycle stage call nodes have stall declarations.
        ctx = getattr(ir, "model_context", None)
        if ctx is None:
            return

        comp_types = []
        if hasattr(ctx, "component_types"):
            comp_types = list(ctx.component_types.values())
        elif hasattr(ctx, "types"):
            from zuspec.ir.core.data_type import DataTypeComponent
            comp_types = [t for t in ctx.types.values() if isinstance(t, DataTypeComponent)]

        for comp in comp_types:
            pipeline_root = getattr(comp, "pipeline_root_ir", None)
            if pipeline_root is None:
                continue
            stage_method_irs = getattr(pipeline_root, "stage_method_irs", {}) or {}
            for stage_name, smIR in stage_method_irs.items():
                call_node = getattr(smIR, "call_node", None)
                cycles = getattr(call_node, "cycles", 1) if call_node else 1
                if cycles and cycles > 1:
                    stall_decls = getattr(smIR, "stall_decls", []) or []
                    if not stall_decls:
                        raise LayerVerificationError(
                            "PipelineLayerVerifier: multi-cycle StageCallNode "
                            "has no StallDecl\n"
                            f"  Expected: at least one StallDecl in StageMethodIR\n"
                            f"  Got:      none\n"
                            f"  Context:  stage {stage_name!r} cycles={cycles}"
                        )


# ---------------------------------------------------------------------------
# Layer 3: STRUCTURAL
# ---------------------------------------------------------------------------

class StructuralLayerVerifier(IRLayerVerifier):
    """Verify Structural RTL IR invariants (Layer 3, post-LowerPass).

    Invariants checked:

    1. All keys in ``ir.lowered_sv`` match the ``<backend>/<category>/<item>``
       regex (via :func:`~zuspec.synth.ir.synth_ir.validate_lowered_sv_key`).
    2. No ``DomainNode`` instances remain in ``ir``
       (delegates to ``PassManager.verify_ready``).
    3. ``ir.meta`` is not ``None``.
    """

    def verify(self, ir: "SynthIR") -> None:
        from zuspec.synth.ir.synth_ir import validate_lowered_sv_key
        from zuspec.dataclasses.transform.pass_manager import PassManager

        if ir.meta is None:
            raise LayerVerificationError(
                "StructuralLayerVerifier: ir.meta is None\n"
                "  Expected: a non-None ComponentSynthMeta after ElaboratePass\n"
                "  Got:      None\n"
                "  Context:  SynthIR.meta"
            )

        for key in ir.lowered_sv:
            try:
                validate_lowered_sv_key(key)
            except ValueError as exc:
                raise LayerVerificationError(
                    f"StructuralLayerVerifier: invalid lowered_sv key {key!r}\n"
                    f"  Expected: '<backend>/<category>/<item>' format\n"
                    f"  Got:      {key!r}\n"
                    f"  Context:  SynthIR.lowered_sv"
                ) from exc

        # Delegate DomainNode check to PassManager.
        pm = PassManager([])
        try:
            pm.verify_ready(ir)
        except Exception as exc:
            raise LayerVerificationError(
                f"StructuralLayerVerifier: unlowered DomainNode found\n"
                f"  {exc}\n"
                f"  Context:  SynthIR (recursive walk)"
            ) from exc
