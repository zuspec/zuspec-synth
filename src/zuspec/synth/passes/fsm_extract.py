"""FSMExtractPass — build a structural FSMModule skeleton."""
from __future__ import annotations

import logging

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


class FSMExtractPass(SynthPass):
    """Build a synthetic :class:`~zuspec.synth.sprtl.fsm_ir.FSMModule` skeleton.

    Creates one :class:`~zuspec.synth.sprtl.fsm_ir.FSMState` per pipeline stage
    implied by ``config.pipeline_stages``, wired in a simple ring topology.
    Stores the result in ``ir.fsm_module``.

    A real implementation would walk the component body AST and populate
    states with actual FSM operations.
    """

    @property
    def name(self) -> str:
        return "fsm_extract"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMState, FSMStateKind

        comp_name = getattr(ir.component, "__name__", "Unknown") if ir.component else "Unknown"
        fsm = FSMModule(name=f"{comp_name}_FSM")

        n = self.config.pipeline_stages
        from zuspec.synth.mls import _STAGE_NAMES
        stage_names = _STAGE_NAMES.get(n, [f"S{i}" for i in range(n)])

        for sid, sname in enumerate(stage_names):
            state = FSMState(id=sid, name=sname, kind=FSMStateKind.NORMAL)
            state.add_transition(target=(sid + 1) % len(stage_names))
            fsm.states.append(state)

        ir.fsm_module = fsm
        _log.info("[FSMExtractPass] built FSM with %d states", len(fsm.states))
        return ir
