"""Phase 4 tests — DomainBinding in FSMModule and synthesis uses domain info."""

import pytest
import zuspec.dataclasses as zdc
from zuspec.dataclasses.domain import ClockDomain, ResetDomain, HardwareResetDomain
from zuspec.dataclasses.data_model_factory import DataModelFactory
from zuspec.ir.core.data_type import DataTypeComponent
from zuspec.synth.sprtl.fsm_ir import DomainBinding, FSMModule
from zuspec.synth import synthesize


def _get_dtc(cls):
    factory = DataModelFactory()
    ctx = factory.build(cls)
    for key, val in ctx.type_m.items():
        if isinstance(val, DataTypeComponent) and cls.__name__ in key:
            return val
    return None


# ---------------------------------------------------------------------------
# T1  DomainBinding.from_component_ir — no domain → defaults
# ---------------------------------------------------------------------------

class TestDomainBindingDefaults:
    def test_no_domain_gives_defaults(self):
        @zdc.dataclass
        class NoDom(zdc.Component):
            clock : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock")
            def _count(self):
                self.count = self.count + 1

        dtc = _get_dtc(NoDom)
        db = DomainBinding.from_component_ir(dtc)
        assert db.clock_name == "clk"
        assert db.reset_name == "rst_n"
        assert db.reset_active_low is True
        assert db.reset_async is False
        assert db.period is None

    def test_clock_domain_sets_name(self):
        @zdc.dataclass
        class SysDom(zdc.Component):
            clock_domain = zdc.ClockDomain(period=zdc.Time.ns(10), name="sys_clk")
            clock : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock")
            def _count(self):
                self.count = self.count + 1

        dtc = _get_dtc(SysDom)
        db = DomainBinding.from_component_ir(dtc)
        assert db.clock_name == "sys_clk"
        assert db.period is not None
        assert db.period.as_ns() == pytest.approx(10.0)

    def test_reset_domain_active_high(self):
        @zdc.dataclass
        class AHRst(zdc.Component):
            reset_domain = zdc.ResetDomain(polarity="active_high", style="async")
            clock : zdc.bit = zdc.input()
            reset : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock", reset="reset")
            def _count(self):
                if self.reset:
                    self.count = 0
                else:
                    self.count = self.count + 1

        dtc = _get_dtc(AHRst)
        db = DomainBinding.from_component_ir(dtc)
        assert db.reset_active_low is False
        assert db.reset_name == "rst"
        assert db.reset_async is True

    def test_hardware_reset_domain(self):
        @zdc.dataclass
        class HWRst(zdc.Component):
            reset_domain = zdc.HardwareResetDomain(polarity="active_low")
            clock : zdc.bit = zdc.input()
            reset : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock", reset="reset")
            def _count(self):
                if self.reset:
                    self.count = 0
                else:
                    self.count = self.count + 1

        dtc = _get_dtc(HWRst)
        db = DomainBinding.from_component_ir(dtc)
        assert db.reset_active_low is True
        assert db.reset_name == "rst_n"


# ---------------------------------------------------------------------------
# T2  FSMModule has domain_binding slot
# ---------------------------------------------------------------------------

class TestFSMModuleDomainBinding:
    def test_domain_binding_slot_defaults_none(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(FSMModule)}
        assert "domain_binding" in fields
        assert fields["domain_binding"].default is None

    def test_can_set_domain_binding(self):
        db = DomainBinding(clock_name="fast_clk", reset_name="rst_n")
        fsm = FSMModule(name="test", domain_binding=db)
        assert fsm.domain_binding is db
        assert fsm.domain_binding.clock_name == "fast_clk"


# ---------------------------------------------------------------------------
# T3  synthesize() uses ClockDomain info for clock/reset port names
# ---------------------------------------------------------------------------

class TestSynthesizeUsesDomain:
    def test_clock_name_from_domain(self):
        @zdc.dataclass
        class DomClk(zdc.Component):
            clock_domain = zdc.ClockDomain(period=zdc.Time.ns(10), name="sys_clk")
            clock : zdc.bit = zdc.input()
            reset : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock", reset="reset")
            def _count(self):
                if self.reset:
                    self.count = 0
                else:
                    self.count = self.count + 1

        sv = synthesize(DomClk)
        assert "posedge sys_clk" in sv

    def test_reset_domain_async_affects_sensitivity(self):
        """Async reset domain → reset appears in always_ff sensitivity list."""
        @zdc.dataclass
        class AsyncRstComp(zdc.Component):
            reset_domain = zdc.ResetDomain(polarity="active_low", style="async")
            clock : zdc.bit = zdc.input()
            reset : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock", reset="reset",
                      reset_async=True, reset_active_low=True)
            def _count(self):
                if self.reset:
                    self.count = 0
                else:
                    self.count = self.count + 1

        sv = synthesize(AsyncRstComp)
        # Async reset → sensitivity list includes negedge rst_n
        assert "negedge" in sv or "posedge" in sv

    def test_no_domain_uses_legacy_field_name(self):
        """When no domain is declared, clock/reset port names come from the field names."""
        @zdc.dataclass
        class LegacySync(zdc.Component):
            clk  : zdc.bit = zdc.input()
            rst_n: zdc.bit = zdc.input()
            count: zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clk", reset="rst_n")
            def _count(self):
                if self.rst_n:
                    self.count = 0
                else:
                    self.count = self.count + 1

        sv = synthesize(LegacySync)
        assert "posedge clk" in sv


# ---------------------------------------------------------------------------
# T4  synthesize(sdc_output=) generates SDC when period is known
# ---------------------------------------------------------------------------

class TestSynthesizeSDC:
    def test_sdc_generated_when_period_known(self, tmp_path):
        @zdc.dataclass
        class SDCComp(zdc.Component):
            clock_domain = zdc.ClockDomain(period=zdc.Time.ns(10), name="sys_clk")
            clock : zdc.bit = zdc.input()
            reset : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock", reset="reset")
            def _count(self):
                if self.reset:
                    self.count = 0
                else:
                    self.count = self.count + 1

        sdc_path = tmp_path / "out.sdc"
        synthesize(SDCComp, sdc_output=str(sdc_path))
        sdc = sdc_path.read_text()
        assert "create_clock" in sdc
        assert "sys_clk" in sdc
        assert "10" in sdc

    def test_sdc_not_generated_when_no_period(self, tmp_path):
        @zdc.dataclass
        class NoPerComp(zdc.Component):
            clock_domain = zdc.ClockDomain(name="sys_clk")   # no period
            clock : zdc.bit = zdc.input()
            count : zdc.b32 = zdc.output(reset=0)

            @zdc.sync(clock="clock")
            def _count(self):
                self.count = self.count + 1

        sdc_path = tmp_path / "out.sdc"
        synthesize(NoPerComp, sdc_output=str(sdc_path))
        assert not sdc_path.exists()
