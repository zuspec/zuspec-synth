"""Tests for SchedulePass.fixed_assignments hook (Item 2 — ECO Foundation).

Verifies:
- fixed_assignments defaults to an empty dict.
- fixed_assignments is accepted as a kwarg and stored correctly.
- Operations with matching state_id are pinned to the requested stage.
- ScheduleConstraintError is raised when the requested stage is infeasible.
- Passing fixed_assignments={} produces the same result as no kwarg.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import pytest

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
for _p in [_synth_src, _dc_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR, ScheduleConstraintError
from zuspec.synth.passes.schedule import SchedulePass
from zuspec.synth.sprtl.scheduler import (
    ASAPScheduler, ALAPScheduler, DependencyGraph, OperationType, Schedule,
)
from zuspec.synth.sprtl.fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMAssign,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_independent_fsm(n_states: int = 3) -> FSMModule:
    """Build an FSMModule with *n_states* states, each containing one independent
    FSMAssign.  No RAW dependencies are created between states so every
    operation is freely movable within the ASAP/ALAP window.
    """
    states = []
    for i in range(n_states):
        s = FSMState(id=i, name=f"S{i}", kind=FSMStateKind.NORMAL)
        # Simple assignment: x_i = 0  (no reads, so no RAW deps)
        s.add_operation(FSMAssign(target=f"x{i}", value=0))
        states.append(s)
    return FSMModule(name="test_fsm", states=states)


def _make_ir_with_fsm(fsm: FSMModule) -> tuple:
    """Return (SynthConfig, SynthIR) with *fsm* pre-loaded into ir.fsm_module."""
    cfg = SynthConfig()
    ir = SynthIR()
    ir.fsm_module = fsm
    return cfg, ir


def _build_chained_graph(n_ops: int = 3, latency: int = 1) -> DependencyGraph:
    """Build a linear dependency chain op0 → op1 → … → op_{n-1}.

    Each op has the given *latency* and is assigned state_id == its position.
    On a critical-path chain, ASAP == ALAP for every node.
    """
    graph = DependencyGraph()
    prev_id = None
    for i in range(n_ops):
        op = graph.add_operation(OperationType.ADD, latency=latency, state_id=i)
        if prev_id is not None:
            graph.add_dependency(prev_id, op.id)
        prev_id = op.id
    return graph


# ---------------------------------------------------------------------------
# Test: defaults
# ---------------------------------------------------------------------------

def test_fixed_assignments_default_is_empty():
    """SchedulePass without fixed_assignments kwarg stores an empty dict."""
    cfg = SynthConfig()
    sp = SchedulePass(cfg)
    assert sp._fixed_assignments == {}


def test_fixed_assignments_accepted_as_kwarg():
    """SchedulePass stores the provided fixed_assignments dict."""
    cfg = SynthConfig()
    sp = SchedulePass(cfg, fixed_assignments={0: 1, 3: 2})
    assert sp._fixed_assignments == {0: 1, 3: 2}


# ---------------------------------------------------------------------------
# Test: noop when empty
# ---------------------------------------------------------------------------

def test_fixed_assignments_noop_when_empty():
    """SchedulePass with fixed_assignments={} produces the same schedule as no kwarg."""
    fsm = _make_independent_fsm(n_states=2)
    cfg, ir_base = _make_ir_with_fsm(fsm)
    cfg2 = SynthConfig()
    ir_empty = SynthIR()
    ir_empty.fsm_module = fsm  # same fsm object — graph rebuilt from same source

    ir_base = SchedulePass(cfg).run(ir_base)
    ir_empty = SchedulePass(cfg2, fixed_assignments={}).run(ir_empty)

    base_times = dict(ir_base.schedule_obj.operation_times)
    empty_times = dict(ir_empty.schedule_obj.operation_times)
    assert base_times == empty_times


# ---------------------------------------------------------------------------
# Test: pins operation to correct stage
# ---------------------------------------------------------------------------

def test_fixed_assignments_pins_operation_to_correct_stage():
    """Pin one independent op to a non-ASAP stage; verify the schedule reflects it.

    The FSM has 2 independent ops.  Without pinning, ASAP puts both at stage 0.
    ALAP (with total_latency=1 and latency=0) allows both to be at stage 0 or 1.
    Pinning state_id=0 to stage 1 must move that op's assigned time to 1.
    """
    fsm = _make_independent_fsm(n_states=2)
    cfg, ir = _make_ir_with_fsm(fsm)

    ir = SchedulePass(cfg, fixed_assignments={0: 1}).run(ir)

    assert ir.schedule_obj is not None

    # Find the op with state_id == 0
    from zuspec.synth.sprtl.scheduler import FSMToScheduleGraphBuilder
    graph = FSMToScheduleGraphBuilder().build(fsm)
    ops_sid0 = [op for op in graph.operations.values() if op.state_id == 0]
    assert len(ops_sid0) == 1
    op_sid0 = ops_sid0[0]

    assigned = ir.schedule_obj.operation_times.get(op_sid0.id, -1)
    assert assigned == 1, (
        f"Expected op with state_id=0 at stage 1, got stage {assigned}"
    )


# ---------------------------------------------------------------------------
# Test: cycle_operations updated when stage changes
# ---------------------------------------------------------------------------

def test_fixed_assignments_updates_cycle_operations():
    """When an op is re-pinned, cycle_operations is updated consistently."""
    fsm = _make_independent_fsm(n_states=2)
    cfg, ir = _make_ir_with_fsm(fsm)

    # Pin state_id=0 from stage 0 → stage 1
    ir = SchedulePass(cfg, fixed_assignments={0: 1}).run(ir)
    sched = ir.schedule_obj

    # Every op_id in operation_times must appear exactly once across cycle_operations
    for op_id, stage in sched.operation_times.items():
        assert op_id in sched.cycle_operations.get(stage, []), (
            f"op {op_id} scheduled at stage {stage} but not in cycle_operations[{stage}]"
        )


# ---------------------------------------------------------------------------
# Test: infeasible assignment raises ScheduleConstraintError
# ---------------------------------------------------------------------------

def test_fixed_assignments_infeasible_raises_schedule_constraint_error():
    """Requesting an impossible stage raises ScheduleConstraintError.

    We test via _apply_fixed_assignments directly with a pre-built chained
    graph where op at state_id=2 is pinned at ASAP=ALAP=2 (critical path),
    but we request stage 0 (infeasible).
    """
    from zuspec.synth.sprtl.scheduler import ALAPScheduler as ALAP

    graph = _build_chained_graph(n_ops=3, latency=1)
    # Run ASAP to set asap_time
    schedule = ASAPScheduler().schedule(graph)
    # Run ALAP to set alap_time
    ALAP().schedule(graph, total_latency=schedule.total_latency)

    cfg = SynthConfig()
    sp = SchedulePass(cfg, fixed_assignments={2: 0})  # state_id=2, request stage 0

    with pytest.raises(ScheduleConstraintError) as exc_info:
        sp._apply_fixed_assignments(
            graph, schedule, ALAP, already_have_alap=True
        )

    err = exc_info.value
    assert err.state_id == 2
    assert err.requested_stage == 0
    # The feasible window for state_id=2 on a 3-op critical-path chain is [2, 2]
    assert err.earliest_stage == 2
    assert err.latest_stage == 2


def test_fixed_assignments_feasible_does_not_raise():
    """Requesting a feasible stage does not raise.

    For the 3-op chain, op at state_id=2 has ASAP=ALAP=2.
    Requesting stage 2 (the only feasible stage) must succeed silently.
    """
    from zuspec.synth.sprtl.scheduler import ALAPScheduler as ALAP

    graph = _build_chained_graph(n_ops=3, latency=1)
    schedule = ASAPScheduler().schedule(graph)
    ALAP().schedule(graph, total_latency=schedule.total_latency)

    cfg = SynthConfig()
    sp = SchedulePass(cfg, fixed_assignments={2: 2})  # state_id=2, request stage 2

    # Must not raise
    sp._apply_fixed_assignments(graph, schedule, ALAP, already_have_alap=True)

    # Find op with state_id=2 and verify it stayed at stage 2
    ops_sid2 = [op for op in graph.operations.values() if op.state_id == 2]
    assert len(ops_sid2) == 1
    assert schedule.operation_times[ops_sid2[0].id] == 2


# ---------------------------------------------------------------------------
# Test: ScheduleConstraintError attributes
# ---------------------------------------------------------------------------

def test_schedule_constraint_error_attributes():
    """ScheduleConstraintError carries the expected fields."""
    err = ScheduleConstraintError(
        op_id=7, state_id=3, requested_stage=5, earliest_stage=2, latest_stage=4
    )
    assert err.op_id == 7
    assert err.state_id == 3
    assert err.requested_stage == 5
    assert err.earliest_stage == 2
    assert err.latest_stage == 4
    assert "5" in str(err)
    assert "state_id=3" in str(err)
