"""StallGenPass — generate stall and valid-chain logic for stalled hazards.

For each hazard resolved by ``"stall"`` in ``PipelineIR.hazards``, this pass
records the stall signal descriptors that ``SVEmitPass`` will later render
as Verilog:

- A ``stall`` wire predicate: driven high when the producer stage is valid
  and the producer's write-address matches the consumer's read-address
  (for ``IndexedRegFile`` accesses) or unconditionally when a scalar load-use
  is detected.
- Valid-signal freeze: all stages from stage-0 through the consumer stage
  hold their valid registers when ``stall`` is asserted.
- Bubble insertion: the stage immediately after the producer receives
  ``valid = 0`` (a pipeline bubble) while the stall is active.

The results are stored in ``PipelineIR`` via two new lists on the IR:

``stall_signals`` — list of :class:`StallSignal` descriptors, each describing
    one stall condition predicate.

``valid_chain`` — list of :class:`ValidStageEntry` in pipeline order,
    describing how each stage's ``*_valid`` register is driven (either from
    the previous stage's valid or held / cleared on stall).

These lists are consumed by ``SVEmitPass`` / ``PipelineSVCodegen`` to emit the
actual Verilog.  They are attached to ``PipelineIR`` as dynamic attributes so
that the core dataclass does not need yet another field for every future
extension.
"""
from __future__ import annotations

import ast as _ast
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .synth_pass import SynthPass
from .expr_lowerer import ExprLowerer
from zuspec.synth.ir.pipeline_ir import HazardPair, PipelineIR
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result descriptors (attached to PipelineIR as dynamic attrs)
# ---------------------------------------------------------------------------

@dataclass
class StallSignal:
    """Descriptor for a generated stall predicate wire.

    :attr:`signal_name` is the Verilog wire name (e.g. ``stall_0``).
    :attr:`producer_stage` / :attr:`consumer_stage` identify the hazard.
    :attr:`hazard_signal` is the variable that causes the stall.
    :attr:`freeze_stages` lists the stage names that must hold their state
        (i.e. not advance) while this stall is active.
    :attr:`bubble_stage` is the stage name where a bubble (``valid = 0``)
        must be inserted while the stall is active.
    """
    signal_name:    str
    producer_stage: str
    consumer_stage: str
    hazard_signal:  str
    freeze_stages:  List[str] = field(default_factory=list)
    bubble_stage:   Optional[str] = None


@dataclass
class ValidStageEntry:
    """How one pipeline stage's ``*_valid`` register is driven.

    :attr:`stage_name` is the stage name (e.g. ``"EX"``).
    :attr:`valid_reg` is the Verilog register name (e.g. ``"ex_valid_q"``).
    :attr:`source_valid` is the Verilog expression driven into this register
        when no stall is active (e.g. ``"id_valid_q"`` or ``"valid_in"``).
    :attr:`stall_signals` is the list of stall wire names that can freeze or
        bubble this stage.
    :attr:`bubble_on_stall` — if True, this stage receives a bubble (``1'b0``)
        rather than being frozen when any of its stall signals is asserted.
    :attr:`cancel_signal` — optional cancel wire name; when asserted clears
        valid without freezing upstream.
    :attr:`flush_signal` — optional aggregated flush wire name; highest priority.
    """
    stage_name:      str
    valid_reg:       str
    source_valid:    str          # what drives this reg when not stalled
    stall_signals:   List[str] = field(default_factory=list)
    bubble_on_stall: bool = False
    cancel_signal:   Optional[str] = None
    flush_signal:    Optional[str] = None


@dataclass
class DeclStallEntry:
    """A stage-declared stall condition generated from ``zdc.stage.stall(cond)``.

    :attr:`stage_name`  — name of the stage declaring the stall.
    :attr:`wire_name`   — Verilog wire: ``{stage_lower}_decl_stalled``.
    :attr:`cond_expr`   — lowered condition expression string (for ``assign``).
    """
    stage_name: str
    wire_name:  str
    cond_expr:  str


@dataclass
class CancelEntry:
    """A stage-declared cancel from ``zdc.stage.cancel(cond)``.

    :attr:`stage_name` — name of the stage declaring the cancel.
    :attr:`wire_name`  — Verilog wire: ``{stage_lower}_cancel``.
    :attr:`cond_expr`  — lowered condition expression string.
    """
    stage_name: str
    wire_name:  str
    cond_expr:  str


@dataclass
class FlushEntry:
    """A flush wire from source to target stage.

    :attr:`source_stage`  — stage (or sync method) emitting the flush.
    :attr:`target_stage`  — stage being flushed.
    :attr:`wire_name`     — Verilog wire: ``{src_lower}_flush_{tgt_lower}``.
    :attr:`cond_expr`     — lowered condition expression string.
    """
    source_stage: str
    target_stage: str
    wire_name:    str
    cond_expr:    str


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

class StallGenPass(SynthPass):
    """Generate stall-signal and valid-chain descriptors for stall-resolved hazards.

    This pass is a no-op when ``ir.pipeline_ir`` is ``None`` or has no
    stall-resolved hazards.
    """

    @property
    def name(self) -> str:
        return "stall_gen"

    def run(self, ir: SynthIR) -> SynthIR:
        """Generate stall-signal and valid-chain descriptors.

        :param ir: Synthesis IR with hazards resolved by :class:`ForwardingGenPass`.
        :type ir: SynthIR
        :return: Updated IR with ``pip.stall_signals``, ``pip.valid_chain``,
                 ``pip.decl_stalls``, ``pip.cancels``, and ``pip.flushes``
                 populated.
        :rtype: SynthIR
        """
        if ir.pipeline_ir is None:
            _log.debug("[StallGenPass] no pipeline_ir — skipping")
            return ir

        pip = ir.pipeline_ir
        stall_hazards = [h for h in pip.hazards if h.resolved_by == "stall"]

        # Build stage-declared stalls and cancels (new API)
        decl_stalls = self._build_decl_stalls(pip)
        cancels     = self._build_cancels(pip)
        flushes     = self._build_flushes(pip)
        pip.decl_stalls = decl_stalls  # type: ignore[attr-defined]
        pip.cancels     = cancels      # type: ignore[attr-defined]
        pip.flushes     = flushes      # type: ignore[attr-defined]

        # Build the valid-chain (needed for all pipelines)
        valid_chain = self._build_valid_chain(pip)
        pip.valid_chain = valid_chain  # type: ignore[attr-defined]

        # Attach cancel/flush signals to valid-chain entries
        for ve in valid_chain:
            sl = ve.stage_name.lower()
            for c in cancels:
                if c.stage_name == ve.stage_name:
                    ve.cancel_signal = c.wire_name
            # Aggregated flush: {stage_lower}_flush
            target_flushes = [f for f in flushes if f.target_stage == ve.stage_name]
            if target_flushes:
                ve.flush_signal = f"{sl}_flush"

        if not stall_hazards:
            pip.stall_signals = []  # type: ignore[attr-defined]
            _log.debug("[StallGenPass] no stall hazards — valid chain generated only")
            return ir

        _log.info("[StallGenPass] generating stall logic for %d hazard(s) in %s",
                  len(stall_hazards), pip.module_name)

        stage_name_to_idx = {s.name: s.index for s in pip.stages}

        stall_signals: List[StallSignal] = []
        for i, hazard in enumerate(stall_hazards):
            sig_name = f"stall_{i}"
            producer_idx = stage_name_to_idx.get(hazard.producer_stage, 0)
            consumer_idx = stage_name_to_idx.get(hazard.consumer_stage, 0)

            # Stages to freeze: all stages from 0 through the consumer
            freeze_up_to = max(producer_idx, consumer_idx)
            freeze_stages = [pip.stages[k].name for k in range(freeze_up_to + 1)]

            # Stage immediately after the producer receives a bubble
            bubble_idx = producer_idx + 1
            bubble_stage = (
                pip.stages[bubble_idx].name
                if bubble_idx < len(pip.stages)
                else None
            )

            ss = StallSignal(
                signal_name=sig_name,
                producer_stage=hazard.producer_stage,
                consumer_stage=hazard.consumer_stage,
                hazard_signal=hazard.signal,
                freeze_stages=freeze_stages,
                bubble_stage=bubble_stage,
            )
            stall_signals.append(ss)

            # Annotate the valid-chain entries with their stall signals
            for ve in valid_chain:
                if ve.stage_name in freeze_stages:
                    ve.stall_signals.append(sig_name)
                if ve.stage_name == bubble_stage:
                    ve.stall_signals.append(sig_name)
                    ve.bubble_on_stall = True

        # Also add decl_stall signals to valid-chain (freeze upstream + self)
        for ds in decl_stalls:
            stage_idx = stage_name_to_idx.get(ds.stage_name, 0)
            freeze_stages = [pip.stages[k].name for k in range(stage_idx + 1)]
            for ve in valid_chain:
                if ve.stage_name in freeze_stages and ds.wire_name not in ve.stall_signals:
                    ve.stall_signals.append(ds.wire_name)

        pip.stall_signals = stall_signals  # type: ignore[attr-defined]
        _log.info(
            "[StallGenPass] %s: %d stall signal(s), %d valid-chain entries",
            pip.module_name, len(stall_signals), len(valid_chain),
        )
        return ir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lower_cond(self, cond: object, stage, pip: PipelineIR) -> str:
        """Lower an AST condition node to an SV expression string."""
        if cond is None:
            return "1'b1"
        if isinstance(cond, _ast.AST):
            try:
                return ExprLowerer(stage, pip).lower_expr(cond)  # type: ignore[arg-type]
            except Exception:
                try:
                    return _ast.unparse(cond)  # type: ignore[arg-type]
                except Exception:
                    return "1'b1"
        return str(cond)

    def _build_decl_stalls(self, pip: PipelineIR) -> List[DeclStallEntry]:
        """Build DeclStallEntry list from StageIR.stall_cond fields."""
        result = []
        for stage in pip.stages:
            cond = getattr(stage, 'stall_cond', None)
            if cond is None:
                continue
            sl = stage.name.lower()
            cond_expr = self._lower_cond(cond, stage, pip)
            result.append(DeclStallEntry(
                stage_name=stage.name,
                wire_name=f"{sl}_decl_stalled",
                cond_expr=cond_expr,
            ))
        return result

    def _build_cancels(self, pip: PipelineIR) -> List[CancelEntry]:
        """Build CancelEntry list from StageIR.cancel_cond fields."""
        result = []
        for stage in pip.stages:
            cond = getattr(stage, 'cancel_cond', None)
            if cond is None:
                continue
            sl = stage.name.lower()
            cond_expr = self._lower_cond(cond, stage, pip)
            result.append(CancelEntry(
                stage_name=stage.name,
                wire_name=f"{sl}_cancel",
                cond_expr=cond_expr,
            ))
        return result

    def _build_flushes(self, pip: PipelineIR) -> List[FlushEntry]:
        """Build FlushEntry list from StageIR.flush_decls and SyncIR.flush_decls."""
        result = []
        # From stage bodies
        for stage in pip.stages:
            for fdecl in getattr(stage, 'flush_decls', []):
                target = getattr(fdecl, 'target_stage', None)
                cond   = getattr(fdecl, 'cond_ast', None)
                if target is None:
                    continue
                cond_expr = self._lower_cond(cond, stage, pip)
                src_lower = stage.name.lower()
                tgt_lower = target.lower()
                result.append(FlushEntry(
                    source_stage=stage.name,
                    target_stage=target,
                    wire_name=f"{src_lower}_flush_{tgt_lower}",
                    cond_expr=cond_expr,
                ))
        # From sync bodies
        for sync in getattr(pip, 'sync_irs', []):
            for fdecl in getattr(sync, 'flush_decls', []):
                target = getattr(fdecl, 'target_stage', None)
                cond   = getattr(fdecl, 'cond_ast', None)
                if target is None:
                    continue
                # Sync has no inputs; use first stage as a proxy for ExprLowerer
                proxy_stage = pip.stages[0] if pip.stages else None
                cond_expr = self._lower_cond(cond, proxy_stage, pip) if proxy_stage else (
                    _ast.unparse(cond) if isinstance(cond, _ast.AST) else "1'b1"
                )
                src_lower = sync.name.lower()
                tgt_lower = target.lower()
                result.append(FlushEntry(
                    source_stage=sync.name,
                    target_stage=target,
                    wire_name=f"{src_lower}_flush_{tgt_lower}",
                    cond_expr=cond_expr,
                ))
        return result

    def _build_valid_chain(self, pip: PipelineIR) -> List[ValidStageEntry]:
        """Build the ordered list of valid-register entries for all stages."""
        entries: List[ValidStageEntry] = []
        for idx, stage in enumerate(pip.stages):
            stage_lower = stage.name.lower()
            valid_reg   = f"{stage_lower}_valid_q"
            if idx == 0:
                source_valid = "valid_in"
            else:
                prev_lower   = pip.stages[idx - 1].name.lower()
                source_valid = f"{prev_lower}_valid_q"
            entries.append(ValidStageEntry(
                stage_name=stage.name,
                valid_reg=valid_reg,
                source_valid=source_valid,
            ))
        return entries
