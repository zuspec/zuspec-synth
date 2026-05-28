"""Tests for IRLayerVerifier implementations (R1)."""
import pytest
import dataclasses as dc
from typing import Optional

from zuspec.synth.verify.layer_verifiers import (
    ActivityLayerVerifier,
    ScheduledLayerVerifier,
    PipelineLayerVerifier,
    StructuralLayerVerifier,
    LayerVerificationError,
)
from zuspec.synth.ir.synth_ir import SynthIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ir(**kwargs) -> SynthIR:
    """Return a minimal SynthIR with given keyword overrides."""
    return SynthIR(**{k: v for k, v in kwargs.items() if k in
                      {f.name for f in dc.fields(SynthIR)}})


# ---------------------------------------------------------------------------
# ActivityLayerVerifier
# ---------------------------------------------------------------------------

class TestActivityLayerVerifier:
    def test_passes_with_no_model_context(self):
        ir = _make_ir()
        # Should not raise when model_context is None.
        ActivityLayerVerifier().verify(ir)

    def test_passes_with_empty_context(self):
        """A context with zero action types should pass."""
        class FakeCtx:
            type_m = {}
            action_types = {}
        ir = SynthIR(model_context=FakeCtx())
        ActivityLayerVerifier().verify(ir)


# ---------------------------------------------------------------------------
# ScheduledLayerVerifier
# ---------------------------------------------------------------------------

class TestScheduledLayerVerifier:
    def test_raises_when_schedule_obj_is_none(self):
        ir = _make_ir()
        with pytest.raises(LayerVerificationError, match="schedule_obj"):
            ScheduledLayerVerifier().verify(ir)

    def test_passes_with_minimal_schedule_obj(self):
        class FakeSched:
            ops = []
            operations = []
        ir = SynthIR(schedule_obj=FakeSched())
        ScheduledLayerVerifier().verify(ir)

    def test_raises_on_negative_stage(self):
        @dc.dataclass
        class FakeOp:
            id: int
            stage: int
            depends_on: list = dc.field(default_factory=list)

        class FakeSched:
            ops = [FakeOp(id=0, stage=-1)]
        ir = SynthIR(schedule_obj=FakeSched())
        with pytest.raises(LayerVerificationError, match="stage"):
            ScheduledLayerVerifier().verify(ir)

    def test_raises_on_dependency_same_stage(self):
        @dc.dataclass
        class FakeOp:
            id: int
            stage: int
            depends_on: list = dc.field(default_factory=list)

        class FakeSched:
            ops = [
                FakeOp(id=0, stage=1),
                FakeOp(id=1, stage=1, depends_on=[0]),
            ]
        ir = SynthIR(schedule_obj=FakeSched())
        with pytest.raises(LayerVerificationError, match="dependency"):
            ScheduledLayerVerifier().verify(ir)

    def test_passes_with_valid_dependency_chain(self):
        @dc.dataclass
        class FakeOp:
            id: int
            stage: int
            depends_on: list = dc.field(default_factory=list)

        class FakeSched:
            ops = [
                FakeOp(id=0, stage=0),
                FakeOp(id=1, stage=1, depends_on=[0]),
            ]
        ir = SynthIR(schedule_obj=FakeSched())
        ScheduledLayerVerifier().verify(ir)  # should not raise


# ---------------------------------------------------------------------------
# PipelineLayerVerifier
# ---------------------------------------------------------------------------

class TestPipelineLayerVerifier:
    def test_passes_with_no_pipeline(self):
        ir = _make_ir()
        PipelineLayerVerifier().verify(ir)

    def test_passes_with_valid_pipeline(self):
        from zuspec.synth.ir.pipeline_ir import PipelineIR, StageIR
        pip = PipelineIR(
            module_name="Top",
            stages=[StageIR(name="IF", index=0), StageIR(name="EX", index=1)],
            channels=[],
            meta=None,
            pipeline_stages=2,
        )
        ir = SynthIR(pipeline_ir=pip)
        PipelineLayerVerifier().verify(ir)

    def test_raises_on_bad_flush_target_in_sync_ir(self):
        from zuspec.synth.ir.pipeline_ir import PipelineIR, StageIR, SyncIR
        from zuspec.ir.core.pipeline import FlushDecl

        flush = FlushDecl(target_stage="NonExistentStage")
        sync = SyncIR(name="on_flush", flush_decls=[flush])
        pip = PipelineIR(
            module_name="Top",
            stages=[StageIR(name="IF", index=0)],
            channels=[],
            meta=None,
            pipeline_stages=1,
            sync_irs=[sync],
        )
        ir = SynthIR(pipeline_ir=pip)
        with pytest.raises(LayerVerificationError, match="FlushDecl"):
            PipelineLayerVerifier().verify(ir)


# ---------------------------------------------------------------------------
# StructuralLayerVerifier
# ---------------------------------------------------------------------------

class TestStructuralLayerVerifier:
    def test_raises_when_meta_is_none(self):
        ir = _make_ir()
        with pytest.raises(LayerVerificationError, match="meta"):
            StructuralLayerVerifier().verify(ir)

    def test_raises_on_invalid_lowered_sv_key(self):
        class FakeMeta:
            pass
        ir = SynthIR(meta=FakeMeta(), lowered_sv={"bad_key": "sv text"})
        with pytest.raises(LayerVerificationError, match="lowered_sv"):
            StructuralLayerVerifier().verify(ir)

    def test_passes_with_valid_lowered_sv_keys(self):
        class FakeMeta:
            pass
        ir = SynthIR(meta=FakeMeta(), lowered_sv={"sv/pipeline/top": "module foo();"})
        StructuralLayerVerifier().verify(ir)

    def test_raises_on_unlowered_domain_node(self):
        from zuspec.ir.core.domain_node import DomainNode
        import dataclasses as _dc

        @_dc.dataclass(kw_only=True)
        class _TestDomain(DomainNode):
            # Implement required abstract methods.
            def inputs(self):
                return []
            def outputs(self):
                return []

        class FakeMeta:
            pass
        ir = SynthIR(meta=FakeMeta(), lowered_sv={})
        # Embed a DomainNode in ir so verify_ready fails.
        ir.rtl_modules = [_TestDomain()]
        with pytest.raises(LayerVerificationError):
            StructuralLayerVerifier().verify(ir)
