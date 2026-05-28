"""Tests for await zdc.tick(N) — multi-cycle wait support.

Covers:
* Constant expression folding for tick argument (e.g. ``1 << 22``)
* WAIT_CYCLES counter register declaration and always_ff logic
* Correct next-state logic (self-loop until counter == 0)
* tick(1) still works as before (no counter emitted)
"""
import sys
import pytest

sys.path.insert(0, 'packages/zuspec-dataclasses/src')
sys.path.insert(0, 'packages/zuspec-synth/src')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blink_sv():
    """Synthesise a Blink-like component using await zdc.tick(1 << 22)."""
    import zuspec.dataclasses as zdc
    from zuspec.synth import synthesize

    @zdc.dataclass
    class Blink(zdc.Component):
        L1: zdc.bit = zdc.output()
        L2: zdc.bit = zdc.output()
        L3: zdc.bit = zdc.output()
        L4: zdc.bit = zdc.output()

        @zdc.proc
        async def _count(self):
            while True:
                await zdc.tick()
                for i in range(4):
                    self.L1 = (i == 0)
                    self.L2 = (i == 1)
                    self.L3 = (i == 2)
                    self.L4 = (i == 3)
                    await zdc.tick((1 << 22))

    return synthesize(Blink)


@pytest.fixture(scope="module")
def tick1_sv():
    """Synthesise a component using await zdc.tick(1) — single-cycle, no counter."""
    import zuspec.dataclasses as zdc
    from zuspec.synth import synthesize

    @zdc.dataclass
    class TickOne(zdc.Component):
        out: zdc.bit = zdc.output()

        @zdc.proc
        async def _run(self):
            while True:
                self.out = 1
                await zdc.tick(1)
                self.out = 0
                await zdc.tick(1)

    return synthesize(TickOne)


@pytest.fixture(scope="module")
def tick_expr_sv():
    """Synthesise a component using a simple constant expression ``2 + 3``."""
    import zuspec.dataclasses as zdc
    from zuspec.synth import synthesize

    @zdc.dataclass
    class TickExpr(zdc.Component):
        out: zdc.bit = zdc.output()

        @zdc.proc
        async def _run(self):
            while True:
                self.out = 1
                await zdc.tick(2 + 3)

    return synthesize(TickExpr)


# ---------------------------------------------------------------------------
# Blink tests
# ---------------------------------------------------------------------------

def test_blink_sv_generated(blink_sv):
    assert blink_sv, "Expected non-empty SV output"


def test_blink_counter_declared(blink_sv):
    """A cycle counter register must be declared for the long tick."""
    # The counter for wait_cycles = 4194304 (= 1<<22)
    # The exact name depends on the state name, but it ends in _cnt
    assert "_cnt" in blink_sv, "Expected a cycle-counter register (_cnt)"


def test_blink_counter_width(blink_sv):
    """Counter should be at least 22 bits wide for 1<<22 - 1 = 4194303."""
    # 4194303 needs 22 bits (bit_length() == 22)
    assert "[21:0]" in blink_sv, "Expected 22-bit counter [21:0]"


def test_blink_counter_init_value(blink_sv):
    """Counter must be initialised to wait_cycles - 1 = 4194303."""
    assert "4194303" in blink_sv, "Expected counter init value 4194303"


def test_blink_counter_decrement(blink_sv):
    """Counter must be decremented each cycle."""
    assert "- 1'b1" in blink_sv, "Expected counter decrement expression"


def test_blink_counter_advance_condition(blink_sv):
    """Next-state logic must advance only when counter == 0."""
    assert "== 0" in blink_sv, "Expected counter == 0 advance condition"


def test_blink_loop_output_assignments(blink_sv):
    """L1..L4 output assignments from the for-loop body must appear."""
    assert "L1 <=" in blink_sv
    assert "L2 <=" in blink_sv
    assert "L3 <=" in blink_sv
    assert "L4 <=" in blink_sv


# ---------------------------------------------------------------------------
# tick(1) — no counter expected
# ---------------------------------------------------------------------------

def test_tick1_no_counter(tick1_sv):
    """tick(1) must not generate a cycle-counter register."""
    assert "_cnt" not in tick1_sv, "tick(1) must NOT generate a counter"


def test_tick1_sv_generated(tick1_sv):
    assert tick1_sv, "Expected non-empty SV output"


# ---------------------------------------------------------------------------
# Constant expression folding
# ---------------------------------------------------------------------------

def test_tick_expr_constant_folded(tick_expr_sv):
    """2 + 3 = 5 cycles: counter should be initialised to 4."""
    assert "4" in tick_expr_sv, "Expected folded constant value 4 (= 5-1)"


def test_tick_expr_no_binop_in_sv(tick_expr_sv):
    """Constant expression must be folded at compile time, not emitted as SV."""
    # We should NOT see '2 + 3' in the generated SV
    assert "2 + 3" not in tick_expr_sv
