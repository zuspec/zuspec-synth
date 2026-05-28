# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral FSM structural helpers.

These pure functions derive structural metadata from an ``FSMModule`` in a
single place so that every backend (SV, Python, C, …) makes identical
decisions about:

* State encoding widths
* Deduplicated state-name strings
* Wait-cycle counter names, widths, and initial values
* The initial-state name

Importing backends should call these helpers rather than re-deriving the
same logic independently, which prevents subtle divergence as the FSM IR
evolves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class WaitCounterInfo:
    """Metadata for a single WAIT_CYCLES state's cycle counter."""

    state: object       # FSMState — the WAIT_CYCLES state
    n_cycles: int       # Total cycles to wait
    counter_name: str   # e.g. ``"S_3_cnt"``
    counter_width: int  # Bit width of the counter register
    init_val: int       # Load value on entry = n_cycles - 1


def fsm_state_names(fsm) -> Dict[int, str]:
    """Return a deduplicated ``state_id → name`` mapping.

    If two states share a name but have different encoded values, the second
    occurrence gets ``_{enc}`` appended to make the name unique.  This
    matches the deduplication logic in ``SVCodeGenerator``.
    """
    result: Dict[int, str] = {}
    seen: Dict[str, int] = {}  # name → first encoding that claimed it
    for i, st in enumerate(fsm.states):
        enc = fsm.state_encoding.get(st.id, i)
        sname = st.name
        if sname in seen and seen[sname] != enc:
            sname = f"{sname}_{enc}"
        seen.setdefault(st.name, enc)
        result[st.id] = sname
    return result


def fsm_state_width(fsm) -> int:
    """Return the state register bit width (pre-computed on the FSMModule)."""
    return fsm.state_width or 1


def fsm_wait_counter_info(
    fsm, state_names: Dict[int, str]
) -> List[WaitCounterInfo]:
    """Return wait-counter metadata for every WAIT_CYCLES state.

    Counter names are derived from *state_names* (the deduplicated mapping
    returned by :func:`fsm_state_names`) so that SV and Python backends use
    the same counter identifier.

    Only states with ``wait_cycles > 1`` are included — a one-cycle "wait"
    needs no counter.
    """
    from zuspec.synth.sprtl.fsm_ir import FSMStateKind

    result: List[WaitCounterInfo] = []
    for state in fsm.states:
        if state.kind != FSMStateKind.WAIT_CYCLES:
            continue
        wc = getattr(state, "wait_cycles", 1)
        if wc <= 1:
            continue
        cname = f"{state_names[state.id]}_cnt"
        init_val = wc - 1
        width = max(1, init_val.bit_length())
        result.append(
            WaitCounterInfo(
                state=state,
                n_cycles=wc,
                counter_name=cname,
                counter_width=width,
                init_val=init_val,
            )
        )
    return result


def fsm_initial_state_name(fsm, state_names: Dict[int, str]) -> str:
    """Return the deduplicated name string of the initial (post-reset) state."""
    init_state = fsm.get_state(fsm.initial_state)
    if init_state is None:
        return "IDLE"
    return state_names.get(init_state.id, "IDLE")
