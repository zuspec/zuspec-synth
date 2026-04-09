"""FSMExtractPass — extract a real FSMModule from @zdc.process source."""
from __future__ import annotations

import logging

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


class FSMExtractPass(SynthPass):
    """Extract an :class:`~zuspec.synth.sprtl.fsm_ir.FSMModule` from the
    component's ``@zdc.process`` body.

    Uses :class:`~zuspec.synth.sprtl.process_body_walker.ProcessBodyWalker` to
    parse the Python AST of the process method and every referenced action
    ``body()`` method.  Each distinct :class:`PipelineStage` discovered becomes
    one :class:`~zuspec.synth.sprtl.fsm_ir.FSMState`; states are wired in the
    order they are encountered (ring topology: last → first).

    Falls back to a generic ring FSM of ``config.pipeline_stages`` states when
    the walker fails or finds no stages.
    """

    @property
    def name(self) -> str:
        return "fsm_extract"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.sprtl.fsm_ir import FSMModule, FSMState, FSMStateKind

        comp_cls = ir.component
        comp_name = getattr(comp_cls, "__name__", "Unknown") if comp_cls else "Unknown"
        fsm = FSMModule(name=f"{comp_name}_FSM")

        stages_extracted = False
        if comp_cls is not None:
            try:
                from zuspec.synth.sprtl.process_body_walker import ProcessBodyWalker
                info = ProcessBodyWalker().walk(comp_cls)
                if info.all_stages:
                    for stage in info.all_stages:
                        fsm.add_state(stage.value.upper())
                    for i, state in enumerate(fsm.states):
                        state.add_transition(target=(i + 1) % len(fsm.states))
                    _log.info(
                        "[FSMExtractPass] %s: extracted %d stages: %s",
                        comp_name,
                        len(info.all_stages),
                        [s.value for s in info.all_stages],
                    )
                    stages_extracted = True
            except Exception as exc:
                _log.warning(
                    "[FSMExtractPass] walker failed for %s (%s) — using config fallback",
                    comp_name,
                    exc,
                )

        if not stages_extracted:
            # Fallback: generic ring FSM from config.pipeline_stages
            n = self.config.pipeline_stages
            try:
                from zuspec.synth.mls import _STAGE_NAMES
                stage_names = _STAGE_NAMES.get(n, [f"S{i}" for i in range(n)])
            except Exception:
                stage_names = [f"S{i}" for i in range(n)]

            for sid, sname in enumerate(stage_names):
                state = FSMState(id=sid, name=sname, kind=FSMStateKind.NORMAL)
                state.add_transition(target=(sid + 1) % len(stage_names))
                fsm.states.append(state)
            _log.info(
                "[FSMExtractPass] %s: fallback FSM with %d states",
                comp_name,
                len(fsm.states),
            )

        ir.fsm_module = fsm
        return ir
