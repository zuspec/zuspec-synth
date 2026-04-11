"""ForwardingGenPass — resolve pipeline hazards to forwarding muxes or stalls.

For each :class:`~zuspec.synth.ir.pipeline_ir.HazardPair` in
``PipelineIR.hazards``, this pass decides whether to insert a bypass mux
(forwarding) or to rely on stall logic:

1. If a per-signal :class:`~zuspec.synth.ir.pipeline_ir.ForwardingDecl`
   covers the hazard, its ``suppressed`` field decides.
2. Otherwise, the process-level ``forward_default`` decides:

   - ``None`` (default) → :exc:`~zuspec.dataclasses.PipelineError` listing
     all unresolved hazards.
   - ``True``  → auto-create ``ForwardingDecl(suppressed=False)`` (bypass mux).
   - ``False`` → auto-create ``ForwardingDecl(suppressed=True)`` (stall).

Updates ``HazardPair.resolved_by`` to ``"forward"`` or ``"stall"`` and
appends any auto-generated ``ForwardingDecl`` entries to
``PipelineIR.forwarding``.

WAW hazards are always resolved by suppressing the earlier write (noted in
``resolved_by="suppress"``); no forwarding declaration is needed.

WAR hazards for plain scalars are structural in a feed-forward pipeline and
resolved by reordering (``resolved_by="reorder"``).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .synth_pass import SynthPass
from zuspec.synth.ir.pipeline_ir import ForwardingDecl, HazardPair, PipelineIR
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)

_UNSET = object()  # sentinel: config field not present


class ForwardingGenPass(SynthPass):
    """Resolve each hazard to a forwarding mux or stall decision.

    This pass is a no-op when ``ir.pipeline_ir`` is ``None``.
    """

    @property
    def name(self) -> str:
        return "forwarding_gen"

    def run(self, ir: SynthIR) -> SynthIR:
        """Resolve all hazards on *pip* to forwarding or stall decisions.

        :param ir: Synthesis IR with ``ir.pipeline_ir`` set and hazards detected.
        :type ir: SynthIR
        :return: Updated IR with each hazard's ``resolved_by`` field set and
                 ``pip.forwarding`` finalised.
        :rtype: SynthIR
        """
        if ir.pipeline_ir is None:
            _log.debug("[ForwardingGenPass] no pipeline_ir — skipping")
            return ir

        pip = ir.pipeline_ir
        if not pip.hazards:
            _log.debug("[ForwardingGenPass] no hazards to resolve")
            return ir

        _log.info("[ForwardingGenPass] resolving %d hazard(s) in %s",
                  len(pip.hazards), pip.module_name)

        # Build lookup: (from_stage, to_stage, signal) → ForwardingDecl
        decl_map: Dict[Tuple[str, str, str], ForwardingDecl] = {}
        for decl in pip.forwarding:
            key = (decl.from_stage, decl.to_stage, decl.signal)
            decl_map[key] = decl

        # Config-level forward_default overrides pip-level when provided (not None).
        # When config says None (require explicit), that takes full precedence.
        cfg_fwd = getattr(self._config, 'forward_default', _UNSET)
        if cfg_fwd is _UNSET:
            forward_default = pip.forward_default
        elif cfg_fwd is None:
            # Config explicitly requires explicit declarations → override pip default
            forward_default = None
        else:
            forward_default = cfg_fwd
        unresolved: List[HazardPair] = []
        auto_decls: List[ForwardingDecl] = []

        # Build per-stage no_forward flag set (from new-API StageIR.no_forward)
        no_forward_stages = {s.name for s in pip.stages if getattr(s, 'no_forward', False)}

        for hazard in pip.hazards:
            if hazard.kind == "WAW":
                hazard.resolved_by = "suppress"
                continue
            if hazard.kind == "WAR":
                hazard.resolved_by = "reorder"
                continue

            # RAW hazard
            key = (hazard.producer_stage, hazard.consumer_stage, hazard.signal)
            decl = decl_map.get(key)

            if decl is not None:
                hazard.resolved_by = "stall" if decl.suppressed else "forward"
            elif hazard.producer_stage in no_forward_stages:
                # Stage-level no_forward: auto-stall all outputs of this stage
                auto = ForwardingDecl(
                    from_stage=hazard.producer_stage,
                    to_stage=hazard.consumer_stage,
                    signal=hazard.signal,
                    suppressed=True,
                )
                auto_decls.append(auto)
                decl_map[key] = auto
                hazard.resolved_by = "stall"
            elif forward_default is None:
                unresolved.append(hazard)
            elif forward_default:
                # Auto-forward
                auto = ForwardingDecl(
                    from_stage=hazard.producer_stage,
                    to_stage=hazard.consumer_stage,
                    signal=hazard.signal,
                    suppressed=False,
                )
                auto_decls.append(auto)
                decl_map[key] = auto
                hazard.resolved_by = "forward"
            else:
                # Auto-stall
                auto = ForwardingDecl(
                    from_stage=hazard.producer_stage,
                    to_stage=hazard.consumer_stage,
                    signal=hazard.signal,
                    suppressed=True,
                )
                auto_decls.append(auto)
                decl_map[key] = auto
                hazard.resolved_by = "stall"

        pip.forwarding.extend(auto_decls)

        if unresolved:
            from zuspec.dataclasses.decorators import PipelineError
            lines = [
                f"  {h.kind} hazard: '{h.signal}' written in stage '{h.producer_stage}', "
                f"read in stage '{h.consumer_stage}'"
                for h in unresolved
            ]
            hint = (
                "Add zdc.forward(from_stage=PRODUCER, to_stage=CONSUMER, var=SIGNAL) "
                "for bypass, or zdc.no_forward(...) for stall, "
                "or set forward=True/False on @zdc.pipeline."
            )
            raise PipelineError(
                f"Unresolved RAW hazard(s) in '{pip.module_name}' "
                f"(forward=None requires explicit declarations):\n"
                + "\n".join(lines)
                + f"\n{hint}"
            )

        # Resolve regfile hazards using the same forward_default policy
        self._resolve_regfile_hazards(pip, forward_default)

        fwd_count   = sum(1 for h in pip.hazards if h.resolved_by == "forward")
        stall_count = sum(1 for h in pip.hazards if h.resolved_by == "stall")
        rf_fwd   = sum(1 for h in pip.regfile_hazards if h.resolved_by == "forward")
        rf_stall = sum(1 for h in pip.regfile_hazards if h.resolved_by == "stall")
        _log.info(
            "[ForwardingGenPass] %s: %d forward, %d stall, %d auto-generated; "
            "regfile: %d forward, %d stall",
            pip.module_name, fwd_count, stall_count, len(auto_decls), rf_fwd, rf_stall,
        )
        return ir

    def _resolve_regfile_hazards(self, pip: "PipelineIR", forward_default) -> None:
        """Resolve each ``RegFileHazard`` to forwarding mux or stall."""
        from zuspec.synth.ir.pipeline_ir import PipelineIR
        for rfh in pip.regfile_hazards:
            if rfh.resolved_by != "unresolved":
                continue
            # Check per-signal forwarding declaration (signal = "field.result_var")
            signal_key = f"{rfh.field_name}.{rfh.read_result_var}"
            decl = None
            for d in pip.forwarding:
                if d.signal == signal_key and d.from_stage == rfh.write_stage and d.to_stage == rfh.read_stage:
                    decl = d
                    break
            if decl is not None:
                rfh.resolved_by = "stall" if decl.suppressed else "forward"
                rfh.suppressed  = decl.suppressed
            elif forward_default is True:
                rfh.resolved_by = "forward"
            elif forward_default is False:
                rfh.resolved_by = "stall"
                rfh.suppressed  = True
            else:
                # Default to forward for regfile hazards to avoid stalls
                rfh.resolved_by = "forward"
