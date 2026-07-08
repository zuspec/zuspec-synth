# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""FSMToRTLPass — convert FSMModule instances to RTL SV text bodies."""
from __future__ import annotations

import logging
from typing import Any

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthIR

_log = logging.getLogger(__name__)


class FSMToRTLPass(SynthPass):
    """Render each FSMModule to a SystemVerilog text fragment.

    For *single-state* FSMs (``fsm.single_state is True``), generates the
    ``always_ff`` body only (no header/ports) and stores it in
    ``ir.lowered_sv["sv/module/clocked"]`` for assembly by
    :class:`ModuleAssemblePass`.

    For *multi-state* FSMs (SPRTL path), generates the complete self-contained
    SV module and stores it in ``ir.lowered_sv["sv/module/top"]`` directly,
    bypassing further assembly.

    Reads:
        ir.fsm_modules: List of FSMModule instances (set by ProcessToFSMPass).

    Populates:
        ir.lowered_sv["sv/module/clocked"]: Body text for single-state modules.
        ir.lowered_sv["sv/module/top"]: Complete SV for multi-state modules.
    """

    @property
    def name(self) -> str:
        return "fsm_to_rtl"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.sprtl.sv_codegen import generate_sv

        bodies: list[str] = []
        for fsm in ir.fsm_modules:
            single = getattr(fsm, "single_state", False)
            if single:
                body = generate_sv(fsm, body_only=True)
                _log.debug(
                    "[FSMToRTLPass] single-state body %d chars for %r",
                    len(body), fsm.name,
                )
                bodies.append(body)
            else:
                # Multi-state path: emit via zuspec.be.sv (the sole SV backend),
                # falling back to the legacy generator for FSM shapes the be.sv
                # builder does not yet cover (process→FSM unification, phase U-2).
                sv = self._emit_multistate(fsm)
                _log.debug(
                    "[FSMToRTLPass] multi-state full module %d chars for %r",
                    len(sv), fsm.name,
                )
                ir.lowered_sv["sv/module/top"] = sv
                return ir  # ModuleAssemblePass will use this directly.

        ir.lowered_sv["sv/module/clocked"] = "\n".join(bodies)
        return ir

    @staticmethod
    def _emit_multistate(fsm: Any) -> str:
        """Emit a complete multi-state FSM module via be.sv, else legacy.

        Attempts the be.sv builder + serializer.  Any construct the core-shape
        builder cannot map (``FsmSVUnsupported``) or any expression the be.sv
        emitter cannot render (``NotImplementedError``) triggers a fall back to
        the legacy ``sv_codegen.generate_sv`` — so this flip is regression-safe:
        no FSM loses coverage, the be.sv path simply takes over the shapes it
        can faithfully reproduce.
        """
        from zuspec.synth.sprtl.sv_codegen import generate_sv
        try:
            from zuspec.be.sv.passes.fsm_to_sv import build_fsm_sv, FsmSVUnsupported
            from zuspec.be.sv.ir.sv_emit import SVEmitter
            nodes = build_fsm_sv(fsm)
            sv = SVEmitter().emit_all(nodes)
            _log.debug("[FSMToRTLPass] be.sv path handled %r", fsm.name)
            return sv
        except (FsmSVUnsupported, NotImplementedError) as e:
            _log.debug(
                "[FSMToRTLPass] be.sv path declined %r (%s); using legacy generator",
                fsm.name, e,
            )
            return generate_sv(fsm)
