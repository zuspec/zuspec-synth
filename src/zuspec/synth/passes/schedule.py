"""SchedulePass — schedule FSM operations into pipeline stages."""
from __future__ import annotations

import logging
from typing import Dict, Optional

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
        fixed_assignments: Optional mapping from FSM ``state_id`` to required
            pipeline stage index (0-based).  Operations matching a ``state_id``
            in this dict are pinned to the specified stage after the main
            scheduler runs.  Use this to preserve stage assignments from a
            golden synthesis run when applying an ECO patch.

            Feasibility is verified against the ASAP/ALAP window for each
            operation.  If the requested stage falls outside that window
            (because a data-dependency chain makes it impossible),
            :exc:`~zuspec.synth.ir.synth_ir.ScheduleConstraintError` is raised.

            ``state_id`` values correspond to
            :attr:`~zuspec.synth.sprtl.scheduler.ScheduledOperation.state_id`,
            which is the FSM state ID assigned during
            :class:`~zuspec.synth.passes.FSMExtractPass`.

            Example::

                SchedulePass(cfg, fixed_assignments={0: 0, 3: 1, 7: 2})
    """

    def __init__(
        self,
        config: SynthConfig,
        *,
        strategy: Optional[str] = None,
        issue: Optional[_ParallelIssue] = None,
        fixed_assignments: Optional[Dict[int, int]] = None,
    ) -> None:
        super().__init__(config=config)
        self._strategy = strategy if strategy is not None else config.strategy
        self._issue = issue
        self._fixed_assignments: Dict[int, int] = fixed_assignments or {}

    @property
    def name(self) -> str:
        return "schedule"

    def run(self, ir: SynthIR) -> SynthIR:
        if ir.fsm_module is None:
            _log.info("[SchedulePass] no fsm_module — skipping scheduling")
            return ir

        from zuspec.synth.sprtl.scheduler import (
            FSMToScheduleGraphBuilder, ASAPScheduler, ALAPScheduler, ListScheduler,
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

            # --- ECO: apply fixed stage assignments -------------------------
            if self._fixed_assignments and ir.schedule_obj is not None:
                self._apply_fixed_assignments(
                    graph, ir.schedule_obj, ALAPScheduler,
                    already_have_alap=(self._strategy == "list"),
                )

        except Exception as exc:
            _log.warning("[SchedulePass] scheduler raised (%s) — continuing", exc)
        return ir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_fixed_assignments(
        self, graph, schedule, ALAPScheduler, *, already_have_alap: bool
    ) -> None:
        """Pin operations to specific stages, validating feasibility.

        For the ASAP strategy path, ALAP times are not computed by the main
        scheduler, so this method runs a separate ALAP pass to obtain the
        upper-bound window before checking feasibility.

        Args:
            graph: The dependency graph (ops have asap_time set; alap_time may
                be set if ListScheduler already ran MobilityAnalyzer).
            schedule: The :class:`~zuspec.synth.sprtl.scheduler.Schedule` to mutate.
            ALAPScheduler: The scheduler class (passed in to avoid a circular import).
            already_have_alap: ``True`` when the ListScheduler already ran
                ``MobilityAnalyzer`` (sets alap_time on all ops).

        Raises:
            :exc:`~zuspec.synth.ir.synth_ir.ScheduleConstraintError`: If a
                requested stage is outside the feasible window.
        """
        from zuspec.synth.ir.synth_ir import ScheduleConstraintError

        if not already_have_alap:
            # ASAP path: compute ALAP to get the upper-bound window.
            ALAPScheduler().schedule(graph, total_latency=schedule.total_latency)

        # Build state_id → op lookup.
        state_to_op = {
            op.state_id: op
            for op in graph.operations.values()
            if op.state_id is not None
        }

        for state_id, requested_stage in self._fixed_assignments.items():
            op = state_to_op.get(state_id)
            if op is None:
                _log.warning(
                    "[SchedulePass] fixed_assignments: state_id=%d not found — skipping",
                    state_id,
                )
                continue

            asap = op.asap_time
            alap = op.alap_time

            if asap <= requested_stage <= alap:
                # Feasible: update schedule.
                old_time = schedule.operation_times.get(op.id, -1)
                schedule.operation_times[op.id] = requested_stage
                # Keep cycle_operations consistent.
                ops_at_old = schedule.cycle_operations.get(old_time, [])
                if op.id in ops_at_old:
                    ops_at_old.remove(op.id)
                schedule.cycle_operations[requested_stage].append(op.id)
                _log.debug(
                    "[SchedulePass] fixed op %d (state_id=%d): stage %d → %d",
                    op.id, state_id, old_time, requested_stage,
                )
            else:
                raise ScheduleConstraintError(
                    op_id=op.id,
                    state_id=state_id,
                    requested_stage=requested_stage,
                    earliest_stage=asap,
                    latest_stage=alap,
                )
