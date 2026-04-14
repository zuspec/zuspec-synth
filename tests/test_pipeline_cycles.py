"""Integration tests for pipeline cycles=N support.

Exercises both Form A (@zdc.stage(cycles=N)) and Form B
(with zdc.stage.cycles(N):) through the full synthesis pass chain.

All component classes MUST be defined at module level so that
``inspect.getsource`` works correctly inside DataModelFactory.
"""
from __future__ import annotations

import sys
import os

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
if "" in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

import pytest
import zuspec.dataclasses as zdc
from zuspec.dataclasses.data_model_factory import DataModelFactory
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.passes import (
    PipelineFrontendPass,
    AutoThreadPass,
    HazardAnalysisPass,
    ForwardingGenPass,
    StallGenPass,
    SyncBodyLowerPass,
    SVEmitPass,
)


# ---------------------------------------------------------------------------
# Shared synth helper
# ---------------------------------------------------------------------------

def run_synth(component_cls):
    """Run full pass chain → (PipelineIR, sv_text)."""
    cfg = SynthConfig(forward_default=True)
    ir = SynthIR()
    ir.component = component_cls
    ir.model_context = DataModelFactory().build(component_cls)

    for pass_cls in [
        PipelineFrontendPass,
        AutoThreadPass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
        SyncBodyLowerPass,
    ]:
        ir = pass_cls(cfg).run(ir)

    ir = SVEmitPass(cfg).run(ir)
    sv = ir.lowered_sv.get("pipeline_sv", "")
    return ir.pipeline_ir, sv


# ---------------------------------------------------------------------------
# Baseline 3-stage component (cycles=1 everywhere — existing behaviour)
# ---------------------------------------------------------------------------

@zdc.dataclass
class _BaselinePipe(zdc.Component):
    clk:    zdc.clock
    rst_n:  zdc.reset
    insn:   zdc.u32
    result: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.insn,)

    @zdc.stage
    def EX(self, insn: zdc.u32) -> (zdc.u32,):
        return (self.result,)

    @zdc.stage
    def WB(self, result: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock='clk', reset='rst_n')
    def execute(self):
        (insn,)   = self.IF()
        (result,) = self.EX(insn)
        self.WB(result)


# ---------------------------------------------------------------------------
# Form A: @zdc.stage(cycles=2) on EX
# ---------------------------------------------------------------------------

@zdc.dataclass
class _DecoratorTwoCyclePipe(zdc.Component):
    clk:    zdc.clock
    rst_n:  zdc.reset
    insn:   zdc.u32
    result: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.insn,)

    @zdc.stage(cycles=2)
    def EX(self, insn: zdc.u32) -> (zdc.u32,):
        return (self.result,)

    @zdc.stage
    def WB(self, result: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock='clk', reset='rst_n')
    def execute(self):
        (insn,)   = self.IF()
        (result,) = self.EX(insn)
        self.WB(result)


# ---------------------------------------------------------------------------
# Form B: with zdc.stage.cycles(2): — context manager
# ---------------------------------------------------------------------------

@zdc.dataclass
class _CMTwoCyclePipe(zdc.Component):
    clk:    zdc.clock
    rst_n:  zdc.reset
    insn:   zdc.u32
    result: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.insn,)

    @zdc.stage
    def EX(self, insn: zdc.u32) -> (zdc.u32,):
        return (self.result,)

    @zdc.stage
    def WB(self, result: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock='clk', reset='rst_n')
    def execute(self):
        (insn,) = self.IF()
        with zdc.stage.cycles(2):
            (result,) = self.EX(insn)
        self.WB(result)


# ---------------------------------------------------------------------------
# Form B: cycles=3
# ---------------------------------------------------------------------------

@zdc.dataclass
class _CMThreeCyclePipe(zdc.Component):
    clk:    zdc.clock
    rst_n:  zdc.reset
    insn:   zdc.u32
    result: zdc.u32

    @zdc.stage
    def IF(self) -> (zdc.u32,):
        return (self.insn,)

    @zdc.stage
    def EX(self, insn: zdc.u32) -> (zdc.u32,):
        return (self.result,)

    @zdc.stage
    def WB(self, result: zdc.u32) -> ():
        pass

    @zdc.pipeline(clock='clk', reset='rst_n')
    def execute(self):
        (insn,) = self.IF()
        with zdc.stage.cycles(3):
            (result,) = self.EX(insn)
        self.WB(result)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineCycles:
    def test_baseline_has_three_stages(self):
        """Baseline 3-stage pipe → 3 stages in PipelineIR."""
        pip, sv = run_synth(_BaselinePipe)
        assert pip is not None
        assert len(pip.stages) == 3

    def test_baseline_stage_names(self):
        """Baseline stage names are IF, EX, WB."""
        pip, sv = run_synth(_BaselinePipe)
        names = [s.name for s in pip.stages]
        assert 'IF' in names
        assert 'EX' in names
        assert 'WB' in names


class TestDecoratorCycles:
    def test_decorator_two_cycle_has_four_stages(self):
        """@zdc.stage(cycles=2) on EX → 4 stages total (IF, EX_c1, EX_c2, WB)."""
        pip, sv = run_synth(_DecoratorTwoCyclePipe)
        assert pip is not None
        assert len(pip.stages) == 4

    def test_decorator_two_cycle_substage_names(self):
        """EX with cycles=2 → stages named EX_c1 and EX_c2."""
        pip, sv = run_synth(_DecoratorTwoCyclePipe)
        names = [s.name for s in pip.stages]
        assert 'EX_c1' in names
        assert 'EX_c2' in names
        assert 'EX' not in names

    def test_decorator_two_cycle_sv_contains_substages(self):
        """Generated SV mentions EX_c1 and EX_c2."""
        pip, sv = run_synth(_DecoratorTwoCyclePipe)
        sv_lower = sv.lower()
        assert 'ex_c1' in sv_lower or 'EX_c1' in sv
        assert 'ex_c2' in sv_lower or 'EX_c2' in sv


class TestContextManagerCycles:
    def test_cm_two_cycle_has_four_stages(self):
        """with zdc.stage.cycles(2): on EX → 4 stages total."""
        pip, sv = run_synth(_CMTwoCyclePipe)
        assert pip is not None
        assert len(pip.stages) == 4

    def test_cm_two_cycle_substage_names(self):
        """CM form: stages named EX_c1 and EX_c2."""
        pip, sv = run_synth(_CMTwoCyclePipe)
        names = [s.name for s in pip.stages]
        assert 'EX_c1' in names
        assert 'EX_c2' in names

    def test_cm_matches_decorator_stage_count(self):
        """Form A and Form B produce same number of stages for cycles=2."""
        pip_a, _ = run_synth(_DecoratorTwoCyclePipe)
        pip_b, _ = run_synth(_CMTwoCyclePipe)
        assert len(pip_a.stages) == len(pip_b.stages)

    def test_cm_three_cycle_has_five_stages(self):
        """cycles=3 → IF, EX_c1, EX_c2, EX_c3, WB = 5 stages."""
        pip, sv = run_synth(_CMThreeCyclePipe)
        assert pip is not None
        assert len(pip.stages) == 5

    def test_cm_three_cycle_all_substages_present(self):
        """cycles=3 produces EX_c1, EX_c2, EX_c3."""
        pip, sv = run_synth(_CMThreeCyclePipe)
        names = [s.name for s in pip.stages]
        assert 'EX_c1' in names
        assert 'EX_c2' in names
        assert 'EX_c3' in names

    def test_cm_three_cycle_order(self):
        """Stage order: IF < EX_c1 < EX_c2 < EX_c3 < WB."""
        pip, sv = run_synth(_CMThreeCyclePipe)
        names = [s.name for s in pip.stages]
        assert names.index('IF') < names.index('EX_c1')
        assert names.index('EX_c1') < names.index('EX_c2')
        assert names.index('EX_c2') < names.index('EX_c3')
        assert names.index('EX_c3') < names.index('WB')

    def test_cm_runtime_no_op(self):
        """The pipeline body with a with-block does not raise at Python runtime."""
        # Verify zdc.stage.cycles() context manager is a no-op
        executed = []
        @zdc.pipeline(clock='clk', reset='rst_n')
        def execute(self):
            with zdc.stage.cycles(2):
                executed.append(True)

        # Calling execute at runtime should not raise; the CM is transparent
        class _Stub:
            pass
        execute(_Stub())
        assert executed == [True]


class TestChannelPassthrough:
    def test_two_cycle_channels_exist(self):
        """Multi-cycle expansion creates channels connecting substages."""
        pip, sv = run_synth(_CMTwoCyclePipe)
        # There should be channels between substages
        assert len(pip.channels) >= 1

    def test_three_cycle_channels_count(self):
        """Three-cycle expansion: at least 2 inter-substage channels."""
        pip, sv = run_synth(_CMThreeCyclePipe)
        assert len(pip.channels) >= 2
