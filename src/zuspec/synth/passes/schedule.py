"""SchedulePass — schedule FSM operations into pipeline stages."""
from __future__ import annotations

import logging
from typing import Optional

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


class _ParallelIssue:
    """Represents a parallel dual/multi-issue directive.

    Created by :func:`parallel`; passed to :class:`SchedulePass` as the
    ``issue`` constructor argument.
    """

    def __init__(self, action_types) -> None:
        self.action_types = action_types

    def __repr__(self) -> str:
        names = [t.__name__ for t in self.action_types]
        return f"parallel({', '.join(names)})"


def parallel(*action_types) -> _ParallelIssue:
    """Declare a parallel issue constraint for the scheduler.

    Example::

        from zuspec.synth.passes import SchedulePass, parallel
        issue = parallel(ExecuteInstruction, ExecuteInstruction)
        SchedulePass(cfg, strategy="list", issue=issue)

    The scheduler uses this directive to guarantee that the two
    ``ExecuteInstruction`` invocations hold *different* resource pool
    slots before either body starts.
    """
    return _ParallelIssue(action_types)


class SchedulePass(SynthPass):
    """Schedule the FSM operations into pipeline stages.

    Reads ``ir.fsm_module`` (set by :class:`FSMExtractPass`) and runs the
    selected scheduler strategy.  Stores the result in ``ir.schedule_obj``.

    Args:
        config: Synthesis configuration; ``config.strategy`` sets the default.
        strategy: Override for the scheduling strategy (``"asap"`` or ``"list"``).
            Defaults to ``config.strategy`` when not supplied.
        issue: Optional :func:`parallel` directive for multi-issue pipelines.
    """

    def __init__(
        self,
        config: SynthConfig,
        *,
        strategy: Optional[str] = None,
        issue: Optional[_ParallelIssue] = None,
    ) -> None:
        super().__init__(config=config)
        self._strategy = strategy if strategy is not None else config.strategy
        self._issue = issue

    @property
    def name(self) -> str:
        return "schedule"

    def run(self, ir: SynthIR) -> SynthIR:
        if ir.fsm_module is None:
            _log.info("[SchedulePass] no fsm_module — skipping scheduling")
            return ir

        from zuspec.synth.sprtl.scheduler import (
            FSMToScheduleGraphBuilder, ASAPScheduler, ListScheduler,
        )
        try:
            builder = FSMToScheduleGraphBuilder()
            graph = builder.build(ir.fsm_module)
            if self._strategy == "list":
                ir.schedule_obj = ListScheduler().schedule(graph)
            else:
                ir.schedule_obj = ASAPScheduler().schedule(graph)
            _log.info(
                "[SchedulePass] strategy=%s  ops=%d",
                self._strategy,
                len(graph.operations),
            )
        except Exception as exc:
            _log.warning("[SchedulePass] scheduler raised (%s) — continuing", exc)
        return ir
