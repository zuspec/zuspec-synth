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
                # Multi-state SPRTL path: generate complete module directly.
                sv = generate_sv(fsm)
                _log.debug(
                    "[FSMToRTLPass] multi-state full module %d chars for %r",
                    len(sv), fsm.name,
                )
                ir.lowered_sv["sv/module/top"] = sv
                return ir  # ModuleAssemblePass will use this directly.

        ir.lowered_sv["sv/module/clocked"] = "\n".join(bodies)
        return ir
