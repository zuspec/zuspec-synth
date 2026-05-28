"""Tests for LayeredPassManager (R1)."""
import pytest
from zuspec.dataclasses.transform.pass_manager import LayeredPassManager, PassManager
from zuspec.synth.verify.layer_verifiers import (
    IRLayerVerifier,
    LayerVerificationError,
)
from zuspec.synth.ir.layers import IRLayer


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _NoopPass:
    """A pass that returns the IR unchanged."""
    def __init__(self, name_str: str, out_layer=None):
        self._name = name_str
        self._out_layer = out_layer

    @property
    def name(self):
        return self._name

    @property
    def output_layer(self):
        return self._out_layer

    def produces_domain_types(self):
        return []

    def run(self, ir):
        return ir


class _CountingVerifier(IRLayerVerifier):
    """Verifier that counts how many times it is called."""
    def __init__(self):
        self.call_count = 0

    def verify(self, ir):
        self.call_count += 1


class _FailingVerifier(IRLayerVerifier):
    def verify(self, ir):
        raise LayerVerificationError("Intentional failure for testing")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLayeredPassManager:
    def test_verifier_called_after_layer_transition(self):
        verifier = _CountingVerifier()
        p1 = _NoopPass("p1", out_layer=IRLayer.PIPELINE)
        pm = LayeredPassManager(
            passes=[p1],
            verifiers={IRLayer.PIPELINE: verifier},
        )
        pm.run(object())
        assert verifier.call_count == 1

    def test_verifier_not_called_when_no_output_layer(self):
        verifier = _CountingVerifier()
        p1 = _NoopPass("p1", out_layer=None)
        pm = LayeredPassManager(
            passes=[p1],
            verifiers={IRLayer.PIPELINE: verifier},
        )
        pm.run(object())
        assert verifier.call_count == 0

    def test_verifier_called_at_correct_pass(self):
        """Verifier for PIPELINE is called only after the PIPELINE pass, not before."""
        call_order = []

        class _RecordingPass:
            def __init__(self, name_str, out_layer=None):
                self._name = name_str
                self._out_layer = out_layer

            @property
            def name(self):
                return self._name

            @property
            def output_layer(self):
                return self._out_layer

            def produces_domain_types(self):
                return []

            def run(self, ir):
                call_order.append(f"pass:{self._name}")
                return ir

        class _RecordingVerifier(IRLayerVerifier):
            def __init__(self, label):
                self._label = label

            def verify(self, ir):
                call_order.append(f"verify:{self._label}")

        p1 = _RecordingPass("sched", out_layer=IRLayer.SCHEDULED)
        p2 = _RecordingPass("pipe", out_layer=IRLayer.PIPELINE)
        pm = LayeredPassManager(
            passes=[p1, p2],
            verifiers={
                IRLayer.SCHEDULED: _RecordingVerifier("SCHEDULED"),
                IRLayer.PIPELINE: _RecordingVerifier("PIPELINE"),
            },
        )
        pm.run(object())
        assert call_order == [
            "pass:sched", "verify:SCHEDULED",
            "pass:pipe",  "verify:PIPELINE",
        ]

    def test_failing_verifier_raises_and_stops_pipeline(self):
        p1 = _NoopPass("p1", out_layer=IRLayer.STRUCTURAL)
        p2 = _NoopPass("p2")

        class _CountingPass(_NoopPass):
            def __init__(self):
                super().__init__("p2")
                self.ran = False

            def run(self, ir):
                self.ran = True
                return ir

        counter_p = _CountingPass()
        pm = LayeredPassManager(
            passes=[p1, counter_p],
            verifiers={IRLayer.STRUCTURAL: _FailingVerifier()},
        )
        with pytest.raises(LayerVerificationError):
            pm.run(object())
        assert not counter_p.ran, "Pass after failing verifier must not run"

    def test_timings_recorded(self):
        p1 = _NoopPass("alpha")
        p2 = _NoopPass("beta", out_layer=IRLayer.SCHEDULED)
        pm = LayeredPassManager(passes=[p1, p2], verifiers={})
        pm.run(object())
        timing_names = [name for name, _ in pm.timings]
        assert timing_names == ["alpha", "beta"]
        for _, elapsed in pm.timings:
            assert elapsed >= 0.0

    def test_timings_reset_on_rerun(self):
        p1 = _NoopPass("p1")
        pm = LayeredPassManager(passes=[p1], verifiers={})
        pm.run(object())
        pm.run(object())
        assert len(pm.timings) == 1  # not accumulated
