"""Tests for passes/protocol_compat.py — ProtocolCompatChecker."""
from __future__ import annotations

import warnings
import pytest

from zuspec.ir.core.data_type import IfProtocolProperties, IfProtocolType
from zuspec.synth.passes.protocol_compat import ProtocolCompatChecker, ProtocolCompatError


def props(**kwargs) -> IfProtocolProperties:
    return IfProtocolProperties(**kwargs)


@pytest.fixture
def checker() -> ProtocolCompatChecker:
    return ProtocolCompatChecker()


# ---------------------------------------------------------------------------
# Standalone validation (§6.2)
# ---------------------------------------------------------------------------

class TestStandaloneValidation:
    def test_defaults_are_valid(self, checker):
        checker.validate(props(), "loc")  # no exception

    def test_max_outstanding_zero_raises(self, checker):
        with pytest.raises(ProtocolCompatError, match="max_outstanding"):
            checker.validate(props(max_outstanding=0), "loc")

    def test_max_outstanding_negative_raises(self, checker):
        with pytest.raises(ProtocolCompatError, match="max_outstanding"):
            checker.validate(props(max_outstanding=-1), "loc")

    def test_initiation_interval_zero_raises(self, checker):
        with pytest.raises(ProtocolCompatError, match="initiation_interval"):
            checker.validate(props(initiation_interval=0), "loc")

    def test_resp_always_valid_without_fixed_latency_raises(self, checker):
        with pytest.raises(ProtocolCompatError, match="fixed_latency"):
            checker.validate(props(resp_always_valid=True), "loc")

    def test_resp_always_valid_with_fixed_latency_is_ok(self, checker):
        checker.validate(props(resp_always_valid=True, fixed_latency=4), "loc")

    def test_fixed_latency_and_backpressure_mutually_exclusive(self, checker):
        with pytest.raises(ProtocolCompatError, match="mutually exclusive"):
            checker.validate(
                props(fixed_latency=2, resp_has_backpressure=True), "loc"
            )

    def test_in_order_false_with_max_outstanding_1_warns(self, checker):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checker.validate(props(in_order=False, max_outstanding=1), "loc")
        assert any("trivially in-order" in str(warning.message) for warning in w)

    def test_in_order_false_with_max_outstanding_gt1_no_warn(self, checker):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checker.validate(props(in_order=False, max_outstanding=4), "loc")
        trivial = [x for x in w if "trivially in-order" in str(x.message)]
        assert not trivial

    def test_accepts_ifprotocoltype(self, checker):
        ipt = IfProtocolType(properties=props(max_outstanding=2))
        checker.validate(ipt, "loc")  # should not raise


# ---------------------------------------------------------------------------
# Cross-side compatibility (§6.1)
# ---------------------------------------------------------------------------

class TestCompatibilityChecks:

    # max_outstanding
    def test_outstanding_export_less_than_port_raises(self, checker):
        with pytest.raises(ProtocolCompatError, match="max_outstanding"):
            checker.check(
                props(max_outstanding=4),
                props(max_outstanding=2),
                "MyComp.port",
            )

    def test_outstanding_export_equal_is_ok(self, checker):
        checker.check(
            props(max_outstanding=4),
            props(max_outstanding=4),
            "MyComp.port",
        )

    def test_outstanding_export_greater_is_ok(self, checker):
        checker.check(
            props(max_outstanding=2),
            props(max_outstanding=8),
            "MyComp.port",
        )

    # in_order
    def test_port_inorder_export_ooo_warns(self, checker):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checker.check(
                props(in_order=True),
                props(in_order=False, max_outstanding=4),
                "MyComp.port",
            )
        assert any("reorder buffer" in str(warning.message) for warning in w)

    def test_port_ooo_export_inorder_is_ok(self, checker):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checker.check(
                props(in_order=False, max_outstanding=4),
                props(in_order=True, max_outstanding=4),  # export can serve same depth
                "MyComp.port",
            )
        reorder = [x for x in w if "reorder buffer" in str(x.message)]
        assert not reorder

    # resp_always_valid (fixed latency)
    def test_port_fixed_latency_export_variable_raises(self, checker):
        with pytest.raises(ProtocolCompatError, match="fixed-latency"):
            checker.check(
                props(resp_always_valid=True, fixed_latency=4),
                props(resp_always_valid=False),
                "MyComp.port",
            )

    def test_both_fixed_latency_is_ok(self, checker):
        checker.check(
            props(resp_always_valid=True, fixed_latency=4),
            props(resp_always_valid=True, fixed_latency=4),
            "MyComp.port",
        )

    def test_port_variable_export_fixed_is_ok(self, checker):
        checker.check(
            props(resp_always_valid=False),
            props(resp_always_valid=True, fixed_latency=2),
            "MyComp.port",
        )

    # resp_has_backpressure
    def test_port_backpressure_export_none_warns(self, checker):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checker.check(
                props(resp_has_backpressure=True),
                props(resp_has_backpressure=False),
                "MyComp.port",
            )
        assert any("resp_ready" in str(warning.message) for warning in w)

    def test_both_backpressure_is_ok(self, checker):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checker.check(
                props(resp_has_backpressure=True),
                props(resp_has_backpressure=True),
                "MyComp.port",
            )
        backpressure_warns = [
            x for x in w if "resp_ready" in str(x.message)
        ]
        assert not backpressure_warns

    # Location string appears in messages
    def test_location_in_error_message(self, checker):
        with pytest.raises(ProtocolCompatError, match="Foo.bar"):
            checker.check(
                props(max_outstanding=4),
                props(max_outstanding=1),
                "Foo.bar",
            )

    # Accepts IfProtocolType on either side
    def test_accepts_ifprotocoltype_on_both_sides(self, checker):
        port_ipt  = IfProtocolType(properties=props(max_outstanding=2))
        export_ipt = IfProtocolType(properties=props(max_outstanding=4))
        checker.check(port_ipt, export_ipt, "MyComp.port")


# ---------------------------------------------------------------------------
# ElaboratePass integration smoke test
# ---------------------------------------------------------------------------

class TestElaboratePassIntegration:
    """Smoke test: ElaboratePass runs without error on a simple component."""

    def test_elaborate_does_not_break_existing_components(self):
        import zuspec.dataclasses as zdc
        from zuspec.synth.passes.elaborate import ElaboratePass
        from zuspec.synth.ir.synth_ir import SynthIR

        @zdc.dataclass
        class SimpleComp(zdc.Component):
            x: zdc.u32 = zdc.field(default=0)

        ir = SynthIR()
        pass_ = ElaboratePass(SimpleComp, None)
        result = pass_.run(ir)
        assert result is ir
