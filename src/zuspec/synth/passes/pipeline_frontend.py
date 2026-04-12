"""PipelineFrontendPass — build PipelineIR from the new @zdc.pipeline / @zdc.stage API.

Replaces ``PipelineAnnotationPass`` for components defined with the new-style API:

.. code-block:: python

    @zdc.pipeline(clock="clk", reset="rst_n")
    def execute(self):
        pc, insn = self.IF()
        rs1, rs2, rd, imm = self.ID(insn)
        result = self.EX(rs1, rs2, imm)
        self.WB(rd, result)

This pass:

1. Locates the ``DataTypeComponent`` for the top-level component in
   ``ir.model_context``.
2. Reads ``pipeline_root_ir``, ``stage_method_irs``, and ``sync_method_irs``
   (populated by ``DataModelFactory`` / DC-6).
3. Builds an ordered list of :class:`~zuspec.synth.ir.pipeline_ir.StageIR`
   from the ``StageCallNode`` sequence.
4. Infers :class:`~zuspec.synth.ir.pipeline_ir.ChannelDecl` entries for
   variable flows: each variable produced by stage k and consumed by stage m
   gets a channel (k→m for adjacent, or k→k+1 for auto-threading prep).
5. Copies ``no_forward``, ``stall_cond``, ``cancel_cond``, and ``flush_decls``
   from each ``StageMethodIR`` into the corresponding ``StageIR``.
6. Populates ``PipelineIR.sync_irs``, ``clock_field``, and ``reset_field``.
7. Stores the result in ``ir.pipeline_ir``.

If no ``pipeline_root_ir`` is found (old-style sentinel API), the pass is a
no-op — ``PipelineAnnotationPass`` handles those components.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .synth_pass import SynthPass
from .pipeline_annotation import _width_from_type_hint
from zuspec.synth.ir.pipeline_ir import (
    ChannelDecl, PipelineIR, StageIR, SyncIR,
)
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)

_DEFAULT_WIDTH = 32


class PipelineFrontendPass(SynthPass):
    """Build ``PipelineIR`` from the new ``@zdc.pipeline`` / ``@zdc.stage`` API.

    This pass is a no-op when ``ir.model_context`` is ``None`` or the
    top-level component has no ``pipeline_root_ir`` (i.e. it uses the old
    sentinel-based API).
    """

    @property
    def name(self) -> str:
        return "pipeline_frontend"

    def run(self, ir: SynthIR) -> SynthIR:
        """Build ``ir.pipeline_ir`` from new-style pipeline IR in model context.

        :param ir: Synthesis IR with ``ir.component`` and ``ir.model_context`` set.
        :type ir: SynthIR
        :return: Updated IR with ``ir.pipeline_ir`` set (or unchanged on no-op).
        :rtype: SynthIR
        """
        if ir.model_context is None:
            _log.debug("[PipelineFrontendPass] no model_context — skipping")
            return ir

        comp_cls = ir.component
        if comp_cls is None:
            _log.debug("[PipelineFrontendPass] no component — skipping")
            return ir

        # Find DataTypeComponent in the model context
        comp_type = self._find_component_type(ir.model_context, comp_cls)
        if comp_type is None:
            _log.debug("[PipelineFrontendPass] component type not found in model_context")
            return ir

        root_ir = getattr(comp_type, 'pipeline_root_ir', None)
        if root_ir is None:
            _log.debug("[PipelineFrontendPass] no pipeline_root_ir — old-style API, skipping")
            return ir

        stage_method_irs = getattr(comp_type, 'stage_method_irs', [])
        sync_method_irs  = getattr(comp_type, 'sync_method_irs', [])

        _log.info("[PipelineFrontendPass] building PipelineIR for %s (%d stages, %d syncs)",
                  comp_cls.__name__, len(root_ir.stage_calls), len(sync_method_irs))

        pip = self._build_pipeline_ir(comp_cls, root_ir, stage_method_irs, sync_method_irs)
        ir.pipeline_ir = pip
        return ir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_component_type(self, ctx, comp_cls):
        """Return the ``DataTypeComponent`` for *comp_cls* from *ctx*, or None."""
        cls_name = comp_cls.__name__
        for name, dt in ctx.type_m.items():
            # Check by class name and by type
            simple_name = name.split('.')[-1]
            if simple_name == cls_name:
                from zuspec.dataclasses.ir.data_type import DataTypeComponent
                if isinstance(dt, DataTypeComponent):
                    return dt
        return None

    def _build_pipeline_ir(
        self,
        comp_cls,
        root_ir,
        stage_method_irs,
        sync_method_irs,
    ) -> PipelineIR:
        """Construct the full ``PipelineIR`` from the new-API IRs."""
        # Build a map from stage name → StageMethodIR
        method_map: Dict[str, Any] = {m.name: m for m in stage_method_irs}

        # Expand multi-cycle calls before building stages / inferring channels.
        expanded_calls = self._expand_stage_calls(root_ir.stage_calls, method_map)

        # Build ordered stage list from expanded call sequence
        stages: List[StageIR] = []
        for idx, call in enumerate(expanded_calls):
            sname = call.stage_name
            # Resolve original method: substage names like EX_c2 → look up EX
            orig_name = call._orig_stage_name if hasattr(call, '_orig_stage_name') else sname
            smir = method_map.get(orig_name) or method_map.get(sname)
            is_first_substage = getattr(call, '_is_first_substage', True)
            # Populate operations from stage method body AST (for hazard analysis)
            # Only the first substage carries the real logic; others are pass-through.
            operations = []
            if smir and smir.body_ast is not None and is_first_substage:
                operations = list(smir.body_ast.body)
            stage = StageIR(
                name=sname,
                index=idx,
                operations=operations,
                no_forward=smir.no_forward if (smir and is_first_substage) else False,
                stall_cond=smir.stall_decls[0].cond_ast if (smir and smir.stall_decls and is_first_substage) else None,
                cancel_cond=smir.cancel_decls[0].cond_ast if (smir and smir.cancel_decls and is_first_substage) else None,
                flush_decls=list(smir.flush_decls) if (smir and is_first_substage) else [],
            )
            stages.append(stage)

        stage_idx: Dict[str, int] = {s.name: i for i, s in enumerate(stages)}

        # Infer channels from the expanded call-sequence data flow
        channels = self._infer_channels(expanded_calls, stages, stage_idx)

        # Attach input/output channel lists to each StageIR
        for ch in channels:
            src = stage_idx.get(ch.src_stage)
            dst = stage_idx.get(ch.dst_stage)
            if src is not None:
                stages[src].outputs.append(ch)
            if dst is not None:
                stages[dst].inputs.append(ch)

        # Build SyncIR list
        sync_irs: List[SyncIR] = []
        for smir in sync_method_irs:
            sync_irs.append(SyncIR(
                name=smir.name,
                clock=smir.clock,
                reset=smir.reset,
                flush_decls=list(getattr(smir, 'flush_decls', [])),
                body_ast=getattr(smir, 'body_ast', None),
            ))

        # Forward default: if root_ir.forward is False, auto-stall; if True, auto-forward
        forward_default: Optional[bool] = root_ir.forward if hasattr(root_ir, 'forward') else None

        module_name = comp_cls.__name__
        port_widths = self._build_port_widths(comp_cls)

        return PipelineIR(
            module_name=module_name,
            stages=stages,
            channels=channels,
            meta=None,
            pipeline_stages=len(stages),
            forward_default=forward_default,
            approach="new",
            sync_irs=sync_irs,
            clock_field=root_ir.clock,
            reset_field=root_ir.reset,
            port_widths=port_widths,
        )

    def _effective_cycles(self, call, method_map: Dict[str, Any]) -> int:
        """Return the effective cycle count for *call*, combining Form A and Form B.

        Form B (context manager, ``call.cycles``) wins when it is > 1.
        Otherwise fall back to the decorator default (Form A, ``StageMethodIR.cycles``).
        """
        if call.cycles > 1:
            return call.cycles
        smir = method_map.get(call.stage_name)
        if smir is not None:
            return getattr(smir, 'cycles', 1)
        return 1

    def _expand_stage_calls(self, stage_calls, method_map: Dict[str, Any]):
        """Expand multi-cycle stage calls into N single-cycle substage calls.

        Each ``StageCallNode`` with ``cycles > 1`` (after combining Form A and
        Form B) is replaced by N synthetic call nodes — one per substage — that
        chain intermediate signals named ``_{STAGE}_c{K}_{var}`` between them.

        Single-cycle calls (the common case) pass through unchanged.

        The returned objects are plain objects with the same interface as
        ``StageCallNode`` plus two extra attributes used by the stage builder:

        - ``_orig_stage_name``  — the original method name (e.g. ``"EX"``).
        - ``_is_first_substage`` — True only for the first substage.
        """
        from zuspec.dataclasses.ir.pipeline import StageCallNode

        class _SubstageCall:
            """Lightweight substage call node."""
            __slots__ = ('stage_name', 'arg_names', 'return_names', 'cycles',
                         '_orig_stage_name', '_is_first_substage')

            def __init__(self, stage_name, arg_names, return_names,
                         orig_stage_name, is_first_substage):
                self.stage_name = stage_name
                self.arg_names = arg_names
                self.return_names = return_names
                self.cycles = 1
                self._orig_stage_name = orig_stage_name
                self._is_first_substage = is_first_substage

        expanded = []
        for call in stage_calls:
            n = self._effective_cycles(call, method_map)
            if n <= 1:
                # Single-cycle: pass through with identity attrs
                call._orig_stage_name = call.stage_name
                call._is_first_substage = True
                expanded.append(call)
                continue

            sname = call.stage_name
            ret_names = call.return_names  # final output variable names

            if n == 1:
                expanded.append(call)
                continue

            # Build N substage call nodes
            for k in range(1, n + 1):
                sub_name = f"{sname}_c{k}"
                is_first = (k == 1)
                is_last  = (k == n)

                if is_first:
                    arg_names_k = call.arg_names
                    # Intermediate outputs (not the final consumer names)
                    if n > 1:
                        ret_names_k = [f"_{sname}_c{k}_{r}" for r in ret_names]
                    else:
                        ret_names_k = ret_names
                elif is_last:
                    arg_names_k = [f"_{sname}_c{k-1}_{r}" for r in ret_names]
                    ret_names_k = ret_names
                else:
                    arg_names_k = [f"_{sname}_c{k-1}_{r}" for r in ret_names]
                    ret_names_k = [f"_{sname}_c{k}_{r}" for r in ret_names]

                expanded.append(_SubstageCall(
                    stage_name=sub_name,
                    arg_names=arg_names_k,
                    return_names=ret_names_k,
                    orig_stage_name=sname,
                    is_first_substage=is_first,
                ))

        return expanded

    def _build_port_widths(self, comp_cls) -> Dict[str, int]:
        """Return {field_name: bit_width} from the component class field annotations."""
        import typing
        widths: Dict[str, int] = {}
        if comp_cls is None:
            return widths
        try:
            hints = typing.get_type_hints(comp_cls, include_extras=True)
        except Exception:
            return widths
        for name, hint in hints.items():
            if name.startswith("_"):
                continue
            w = _width_from_type_hint(hint)
            if w is not None:
                widths[name] = w
        return widths

    def _infer_channels(
        self,
        stage_calls,
        stages: List[StageIR],
        stage_idx: Dict[str, int],
    ) -> List[ChannelDecl]:
        """Infer inter-stage channels from the call-sequence data flow.

        For each variable produced at stage k (in ``return_names``) and consumed
        at stage m > k (in ``arg_names``), create a ``ChannelDecl`` from stage
        k to stage m.  Variables consumed at the immediately following stage get
        a direct channel; variables that skip stages are handled by
        :class:`~zuspec.synth.passes.auto_thread.AutoThreadPass`.
        """
        # Map each variable → (producer_stage_idx, producer_stage_name)
        produced_at: Dict[str, Tuple[int, str]] = {}
        for call in stage_calls:
            sidx = stage_idx.get(call.stage_name)
            if sidx is None:
                continue
            for vname in call.return_names:
                produced_at[vname] = (sidx, call.stage_name)

        channels: List[ChannelDecl] = []
        seen: Set[Tuple[str, str, str]] = set()

        for call in stage_calls:
            consumer_idx = stage_idx.get(call.stage_name)
            if consumer_idx is None:
                continue
            for vname in call.arg_names:
                prod = produced_at.get(vname)
                if prod is None:
                    continue  # sourced from outside pipeline
                prod_idx, prod_stage = prod
                if prod_idx >= consumer_idx:
                    continue  # feed-forward only
                key = (prod_stage, call.stage_name, vname)
                if key in seen:
                    continue
                seen.add(key)
                ch_name = f"{vname}_{prod_stage.lower()}_to_{call.stage_name.lower()}"
                channels.append(ChannelDecl(
                    name=ch_name,
                    width=_DEFAULT_WIDTH,
                    depth=1,
                    src_stage=prod_stage,
                    dst_stage=call.stage_name,
                ))

        return channels
