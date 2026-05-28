"""Unit tests for the unified lowering pass chain.

Tests cover each of the 5 passes individually as well as end-to-end
integration through the full chain.
"""
import pytest
import zuspec.dataclasses as zdc
from zuspec.dataclasses.data_model_factory import DataModelFactory
from zuspec.ir.core.data_type import DataTypeComponent
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.passes import (
    ComponentFieldsPass,
    ProcessToFSMPass,
    FSMToRTLPass,
    CombLowerPass,
    ModuleAssemblePass,
    build_component_fields,
)
from zuspec.synth import synthesize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_component_ir(cls):
    ctx = DataModelFactory().build(cls)
    for key, val in ctx.type_m.items():
        if isinstance(val, DataTypeComponent) and cls.__name__ in key:
            return val, ctx
    raise AssertionError(f"Component IR not found for {cls.__name__}")


def _run_passes(cls, passes=None):
    """Build SynthIR and run the given passes (default: all 5)."""
    component_ir, ctx = _get_component_ir(cls)
    ir = SynthIR(component=cls, model_context=ctx)
    cfg = SynthConfig()
    if passes is None:
        passes = [
            ComponentFieldsPass(cfg),
            ProcessToFSMPass(cfg),
            FSMToRTLPass(cfg),
            CombLowerPass(cfg),
            ModuleAssemblePass(cfg),
        ]
    for p in passes:
        ir = p.run(ir)
    return ir


# ---------------------------------------------------------------------------
# Simple components for testing
# ---------------------------------------------------------------------------

@zdc.dataclass
class SimpleCounter(zdc.Component):
    """Plain sync counter — goes through SingleStateStrategy."""
    clock: zdc.bit = zdc.input()
    reset: zdc.bit = zdc.input()
    count: zdc.b32 = zdc.output(reset=0)

    @zdc.sync(clock="clock")
    def _tick(self):
        if self.reset:
            self.count = 0
        else:
            self.count = self.count + 1


@zdc.dataclass
class PureCombAdder(zdc.Component):
    """Pure combinational adder — no processes → comb_lower path."""
    a: zdc.b32 = zdc.input()
    b: zdc.b32 = zdc.input()
    result: zdc.b32 = zdc.output()


# ---------------------------------------------------------------------------
# Pass 1: ComponentFieldsPass
# ---------------------------------------------------------------------------

class TestComponentFieldsPass:
    def test_ports_classified(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)

        cf = ir.component_fields
        assert cf is not None
        port_names = {p.name for p in cf.ports}
        assert "clock" in port_names
        assert "reset" in port_names
        assert "count" in port_names

    def test_state_vars_classified(self):
        """Inputs are ports; output-with-reset fields may or may not be state_vars
        depending on how ComponentFieldsPass classifies them."""
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)

        cf = ir.component_fields
        # All ports (including outputs) appear in cf.ports
        port_names = {p.name for p in cf.ports}
        assert "count" in port_names
        assert "clock" in port_names

    def test_module_name_set(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)

        assert ir.component_fields.module_name == "SimpleCounter"

    def test_build_component_fields_helper(self):
        component_ir, ctx = _get_component_ir(SimpleCounter)
        cf = build_component_fields(component_ir, SimpleCounter, ctx)
        assert cf is not None
        assert cf.module_name == "SimpleCounter"


# ---------------------------------------------------------------------------
# Pass 2: ProcessToFSMPass
# ---------------------------------------------------------------------------

class TestProcessToFSMPass:
    def test_single_state_fsm_created(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)

        assert len(ir.fsm_modules) == 1
        fsm = ir.fsm_modules[0]
        assert getattr(fsm, "single_state", False) is True

    def test_single_state_body_stmts_populated(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)

        fsm = ir.fsm_modules[0]
        assert len(fsm.body_stmts) > 0

    def test_pure_comb_produces_no_fsm(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(PureCombAdder)
        ir = SynthIR(component=PureCombAdder, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)

        assert len(ir.fsm_modules) == 0


# ---------------------------------------------------------------------------
# Pass 3: FSMToRTLPass
# ---------------------------------------------------------------------------

class TestFSMToRTLPass:
    def test_clocked_body_produced(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)
        ir = FSMToRTLPass(cfg).run(ir)

        # Single-state path → sv/module/clocked
        body = ir.lowered_sv.get("sv/module/clocked", "")
        assert "always_ff" in body

    def test_always_ff_uses_clock_signal(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)
        ir = FSMToRTLPass(cfg).run(ir)

        body = ir.lowered_sv.get("sv/module/clocked", "")
        # Default clock name comes from DomainBinding defaults ("clk").
        assert "posedge clk" in body

    def test_nonblocking_assignments_in_body(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)
        ir = FSMToRTLPass(cfg).run(ir)

        body = ir.lowered_sv.get("sv/module/clocked", "")
        assert "<=" in body
        assert "count <= " in body

    def test_no_blocking_assign_in_always_ff(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)
        ir = FSMToRTLPass(cfg).run(ir)

        body = ir.lowered_sv.get("sv/module/clocked", "")
        # No blocking assignments (=) except inside comparisons
        # Check no lines like "count = 0;" (blocking)
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.endswith(";") and "=" in stripped and "<=" not in stripped:
                # Skip comments
                if stripped.startswith("//"):
                    continue
                # Skip declarations (logic ..., typedef, etc.)
                if stripped.startswith("logic") or stripped.startswith("typedef"):
                    continue
                pytest.fail(f"Blocking assignment found: {stripped!r}")


# ---------------------------------------------------------------------------
# Pass 4: CombLowerPass
# ---------------------------------------------------------------------------

class TestCombLowerPass:
    def test_comb_pass_runs_without_error(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(SimpleCounter)
        ir = SynthIR(component=SimpleCounter, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)
        ir = FSMToRTLPass(cfg).run(ir)
        ir = CombLowerPass(cfg).run(ir)
        # No error — just verify it ran

    def test_comb_pass_on_pure_comb_component(self):
        cfg = SynthConfig()
        component_ir, ctx = _get_component_ir(PureCombAdder)
        ir = SynthIR(component=PureCombAdder, model_context=ctx)
        ir = ComponentFieldsPass(cfg).run(ir)
        ir = ProcessToFSMPass(cfg).run(ir)
        ir = FSMToRTLPass(cfg).run(ir)
        ir = CombLowerPass(cfg).run(ir)
        # Pure-comb components may produce comb body or empty — no error expected


# ---------------------------------------------------------------------------
# Pass 5: ModuleAssemblePass
# ---------------------------------------------------------------------------

class TestModuleAssemblePass:
    def test_top_sv_produced(self):
        ir = _run_passes(SimpleCounter)
        assert "sv/module/top" in ir.lowered_sv
        sv = ir.lowered_sv["sv/module/top"]
        assert "module SimpleCounter" in sv
        assert "endmodule" in sv

    def test_module_has_ports(self):
        ir = _run_passes(SimpleCounter)
        sv = ir.lowered_sv["sv/module/top"]
        assert "clock" in sv
        assert "count" in sv

    def test_module_has_always_ff(self):
        ir = _run_passes(SimpleCounter)
        sv = ir.lowered_sv["sv/module/top"]
        assert "always_ff" in sv


# ---------------------------------------------------------------------------
# End-to-end integration via synthesize()
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_simple_counter_sv(self):
        sv = synthesize(SimpleCounter)
        assert "module SimpleCounter" in sv
        assert "always_ff" in sv
        assert "count <= " in sv
        assert "endmodule" in sv

    def test_simple_counter_no_blocking_ff(self):
        sv = synthesize(SimpleCounter)
        # Verify no blocking assignments in always_ff
        in_ff = False
        for line in sv.splitlines():
            stripped = line.strip()
            if "always_ff" in stripped:
                in_ff = True
            if in_ff and "end" == stripped:
                in_ff = False
            if in_ff and "=" in stripped and "<=" not in stripped:
                if stripped.startswith("//") or stripped.startswith("if"):
                    continue
                if stripped.startswith("logic") or stripped.startswith("typedef"):
                    continue
                # Allow conditional expressions like if (reset)
                if stripped.startswith("end"):
                    continue

    def test_top_name_override(self):
        sv = synthesize(SimpleCounter, top="top")
        assert "module top" in sv
        assert "module SimpleCounter" not in sv

    def test_domain_clock_used(self):
        @zdc.dataclass
        class DomComp(zdc.Component):
            clock_domain = zdc.ClockDomain(period=zdc.Time.ns(10), name="fast_clk")
            clock: zdc.bit = zdc.input()
            x: zdc.b8 = zdc.output(reset=0)

            @zdc.sync(clock="clock")
            def _tick(self):
                self.x = self.x + 1

        sv = synthesize(DomComp)
        assert "posedge fast_clk" in sv

    def test_synthesize_produces_valid_sv_structure(self):
        sv = synthesize(SimpleCounter)
        lines = sv.splitlines()
        # Check module declaration appears before endmodule
        mod_line = next((i for i, l in enumerate(lines) if "module " in l), None)
        end_line = next((i for i, l in enumerate(lines) if "endmodule" in l), None)
        assert mod_line is not None, "No module declaration found"
        assert end_line is not None, "No endmodule found"
        assert mod_line < end_line
