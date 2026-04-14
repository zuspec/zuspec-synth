"""Phase 8 tests — CDC analysis pass and TwoFFSync primitive.

Tests:
    T1  test_single_domain_no_crossings     — homogeneous hierarchy → no crossings
    T2  test_two_domain_wire_is_crossing    — two different domains → one crossing
    T3  test_two_ff_sync_suppresses_crossing — TwoFFSync suppresses crossing
    T4  test_cdc_unchecked_suppresses       — @cdc_unchecked suppresses crossing
    T5  test_two_ff_sync_sv_output          — TwoFFSync SV contains two-FF chain
    T6  test_two_ff_sync_sdc_false_path     — TwoFFSync SDC contains set_false_path
"""

import pytest
import zuspec.dataclasses as zdc
from zuspec.dataclasses.cdc import TwoFFSync, cdc_unchecked
from zuspec.synth import synthesize, _generate_sdc
from zuspec.synth.passes import CDCAnalysisPass, CDCCrossing


# ---------------------------------------------------------------------------
# Leaf components used across tests
# ---------------------------------------------------------------------------

@zdc.dataclass
class FastProducer(zdc.Component):
    """Runs in the 'fast' clock domain (4 ns period)."""
    clock_domain = zdc.ClockDomain(name="fast_clk", period=zdc.Time.ns(4))
    reset_domain = zdc.ResetDomain()

    data_out: zdc.bit = zdc.output(reset=0)

    @zdc.sync
    def _gen(self):
        self.data_out = ~self.data_out


@zdc.dataclass
class SlowConsumer(zdc.Component):
    """Runs in the 'slow' clock domain (10 ns period)."""
    clock_domain = zdc.ClockDomain(name="slow_clk", period=zdc.Time.ns(10))
    reset_domain = zdc.ResetDomain()

    data_in: zdc.bit = zdc.input()
    received: zdc.b8 = zdc.output(reset=0)

    @zdc.sync
    def _recv(self):
        if self.data_in:
            self.received = self.received + 1


@zdc.dataclass
class SameClockComp(zdc.Component):
    """Sibling component also in the 'fast' clock domain."""
    clock_domain = zdc.ClockDomain(name="fast_clk", period=zdc.Time.ns(4))
    reset_domain = zdc.ResetDomain()

    data_in: zdc.bit = zdc.input()

    @zdc.sync
    def _nop(self):
        pass  # no-op


# ---------------------------------------------------------------------------
# T1  Single domain — no crossings
# ---------------------------------------------------------------------------

class TestSingleDomainNoCrossings:
    def test_single_domain_no_crossings(self):
        """A hierarchy with two instances in the same clock domain → empty list."""

        @zdc.dataclass
        class SingleDomainTop(zdc.Component):
            prod: FastProducer  = zdc.inst()
            cons: SameClockComp = zdc.inst()

            def __bind__(self):
                return ((self.cons.data_in, self.prod.data_out),)

        crossings = CDCAnalysisPass.run(SingleDomainTop)
        assert crossings == [], f"Expected no crossings, got: {crossings}"


# ---------------------------------------------------------------------------
# T2  Two different domains → one crossing detected
# ---------------------------------------------------------------------------

class TestTwoDomainWireIsCrossing:
    def test_two_domain_wire_is_crossing(self):
        """Direct wire between fast and slow domain → one unsuppressed CDCCrossing."""

        @zdc.dataclass
        class UnsafeTop(zdc.Component):
            prod: FastProducer  = zdc.inst()
            cons: SlowConsumer  = zdc.inst()

            def __bind__(self):
                return ((self.cons.data_in, self.prod.data_out),)

        crossings = CDCAnalysisPass.run(UnsafeTop)
        assert len(crossings) == 1
        c = crossings[0]
        assert c.suppressed is False
        # The two domains involved should be fast_clk and slow_clk
        assert {c.src_domain, c.dst_domain} == {"fast_clk", "slow_clk"}


# ---------------------------------------------------------------------------
# T3  TwoFFSync suppresses a crossing
# ---------------------------------------------------------------------------

class TestTwoFFSyncSuppressesCrossing:
    def test_two_ff_sync_suppresses_crossing(self):
        """Inserting TwoFFSync between the two sub-instances marks crossing suppressed."""

        @zdc.dataclass
        class SafeTop(zdc.Component):
            prod: FastProducer = zdc.inst()
            sync: TwoFFSync    = zdc.inst()  # runs in slow domain
            cons: SlowConsumer = zdc.inst()

            def __bind__(self):
                return (
                    (self.sync.data_in,  self.prod.data_out),
                    (self.cons.data_in,  self.sync.data_out),
                )

        crossings = CDCAnalysisPass.run(SafeTop)
        # There may be crossings between fast_clk and slow_clk, and between
        # TwoFFSync's domain and fast_clk — all should be suppressed.
        assert len(crossings) >= 1
        for c in crossings:
            if c.src_domain != c.dst_domain:
                assert c.suppressed, (
                    f"Expected crossing {c!r} to be suppressed by TwoFFSync"
                )


# ---------------------------------------------------------------------------
# T4  cdc_unchecked suppresses a crossing
# ---------------------------------------------------------------------------

class TestCdcUncheckedSuppresses:
    def test_cdc_unchecked_suppresses(self):
        """@cdc_unchecked on a class marks all crossings involving it as suppressed."""

        @cdc_unchecked("Gray-coded counter, safe by construction")
        @zdc.dataclass
        class GraySafeProducer(zdc.Component):
            clock_domain = zdc.ClockDomain(name="fast_clk", period=zdc.Time.ns(4))
            reset_domain = zdc.ResetDomain()
            data_out: zdc.bit = zdc.output(reset=0)

            @zdc.sync
            def _gen(self):
                self.data_out = ~self.data_out

        @zdc.dataclass
        class UncheckedTop(zdc.Component):
            prod: GraySafeProducer = zdc.inst()
            cons: SlowConsumer     = zdc.inst()

            def __bind__(self):
                return ((self.cons.data_in, self.prod.data_out),)

        crossings = CDCAnalysisPass.run(UncheckedTop)
        assert len(crossings) >= 1
        for c in crossings:
            assert c.suppressed, (
                f"Expected crossing {c!r} to be suppressed by cdc_unchecked"
            )
            assert "cdc_unchecked" in (c.suppressor or "")


# ---------------------------------------------------------------------------
# T5  TwoFFSync SV output contains the two-FF chain
# ---------------------------------------------------------------------------

class TestTwoFFSyncSVOutput:
    def test_two_ff_sync_sv_output(self):
        """synthesize(TwoFFSync) emits a two-stage FF chain (_sync0 → data_out)."""
        sv = synthesize(TwoFFSync)
        assert "_sync0" in sv, "Expected '_sync0' register in SV output"
        # Verify the two assignment stages are present (non-blocking assignments)
        assert "_sync0 <= data_in" in sv or "_sync0<= data_in" in sv or \
               "_sync0 <= data_in" in sv.replace(" ", ""), \
               "Expected '_sync0 <= data_in' in SV"
        assert "data_out <= _sync0" in sv or "data_out<= _sync0" in sv or \
               "data_out <= _sync0" in sv.replace(" ", ""), \
               "Expected 'data_out <= _sync0' in SV"


# ---------------------------------------------------------------------------
# T6  TwoFFSync SDC contains set_false_path
# ---------------------------------------------------------------------------

class TestTwoFFSyncSDCFalsePath:
    def test_two_ff_sync_sdc_false_path(self):
        """_generate_sdc for a design containing TwoFFSync emits set_false_path."""

        @zdc.dataclass
        class SyncedTop(zdc.Component):
            clock_domain = zdc.ClockDomain(name="slow_clk", period=zdc.Time.ns(10))
            reset_domain = zdc.ResetDomain()

            prod: FastProducer = zdc.inst()
            sync: TwoFFSync    = zdc.inst()
            cons: SlowConsumer = zdc.inst()

            def __bind__(self):
                return (
                    (self.sync.data_in,  self.prod.data_out),
                    (self.cons.data_in,  self.sync.data_out),
                )

        sdc = _generate_sdc(SyncedTop)
        assert "set_false_path" in sdc, \
            f"Expected 'set_false_path' in SDC output:\n{sdc}"
        assert "_sync0_reg" in sdc, \
            f"Expected '_sync0_reg' target in SDC:\n{sdc}"

    def test_two_ff_sync_direct_sdc(self):
        """_generate_sdc called directly on TwoFFSync emits set_false_path."""
        # TwoFFSync has no period, so create_clock won't be emitted,
        # but set_false_path should still appear.
        sdc = _generate_sdc(TwoFFSync)
        assert "set_false_path" in sdc, \
            f"Expected 'set_false_path' in SDC for TwoFFSync:\n{sdc}"
        assert "_sync0_reg" in sdc
