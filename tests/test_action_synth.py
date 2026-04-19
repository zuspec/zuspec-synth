"""Synthesis tests for examples/02_action — action-call positional form.

Verifies that ``await IncrCount()(self)`` inside a ``@zdc.proc`` synthesizes
to RTL identical in structure to the direct-write counter in examples/01_counter.
"""
import sys
import pytest

sys.path.insert(0, 'packages/zuspec-dataclasses/src')
sys.path.insert(0, 'packages/zuspec-synth/src')

import zuspec.dataclasses as zdc


# ---------------------------------------------------------------------------
# Domain model — defined at module level so inspect.getmodule resolves names.
# ---------------------------------------------------------------------------

@zdc.dataclass
class IncrCount(zdc.Action['_Counter']):
    async def body(self):
        await self.comp.count.write(self.comp.count.read() + 1)


@zdc.dataclass
class _Counter(zdc.Component):
    count: zdc.Reg[zdc.b32] = zdc.output()

    @zdc.proc
    async def _count(self):
        while True:
            await IncrCount()(self)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def action_sv():
    from zuspec.synth import synthesize
    return synthesize(_Counter)


# ---------------------------------------------------------------------------
# Tests — output should be structurally identical to test_counter_synth.py
# ---------------------------------------------------------------------------

def test_has_clock_reset_ports(action_sv):
    assert "input clock" in action_sv
    assert "input reset" in action_sv


def test_has_count_output_reg(action_sv):
    assert "output reg" in action_sv
    assert "count" in action_sv


def test_has_always_posedge(action_sv):
    assert "always @(posedge clock or posedge reset)" in action_sv


def test_reset_clears_count(action_sv):
    assert "count <= 0" in action_sv


def test_body_increments_count(action_sv):
    assert "count <= " in action_sv
    assert "count + 1" in action_sv or "(count + 1)" in action_sv

