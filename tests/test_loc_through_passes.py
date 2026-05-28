"""Tests for loc propagation through synthesis passes (R2)."""
import pytest
from zuspec.ir.core.base import Loc


class TestCopyLocOnSynthPassNodes:
    """Verify copy_loc() and propagate_loc() helpers work as expected."""

    def test_stage_ir_carries_loc_field(self):
        """StageIR must have a loc field (added for R2)."""
        from zuspec.synth.ir.pipeline_ir import StageIR
        import dataclasses as dc
        field_names = {f.name for f in dc.fields(StageIR)}
        assert "loc" in field_names, "StageIR must have a loc field (R2 requirement)"

    def test_forwarding_decl_carries_loc_field(self):
        """ForwardingDecl must have a loc field (added for R2)."""
        from zuspec.synth.ir.pipeline_ir import ForwardingDecl
        import dataclasses as dc
        field_names = {f.name for f in dc.fields(ForwardingDecl)}
        assert "loc" in field_names, "ForwardingDecl must have a loc field (R2 requirement)"

    def test_synth_pass_propagate_loc_is_callable(self):
        """SynthPass.propagate_loc must be a static method."""
        from zuspec.synth.passes.synth_pass import SynthPass
        assert callable(SynthPass.propagate_loc)

    def test_propagate_loc_copies_loc_to_base_node(self):
        """propagate_loc should copy loc from a Base node with loc to another."""
        import dataclasses as dc
        from zuspec.ir.core.base import Base
        from zuspec.synth.passes.synth_pass import SynthPass

        @dc.dataclass(kw_only=True)
        class _N(Base):
            v: int = 0

        src = _N(v=1, loc=Loc(file="src.py", line=5, pos=0))
        dst = _N(v=2)
        SynthPass.propagate_loc(src, dst)
        assert dst.loc is not None
        assert dst.loc.line == 5

    def test_propagate_loc_no_error_when_dst_has_no_copy_loc(self):
        """propagate_loc should not raise when dst is a plain dataclass without copy_loc."""
        import dataclasses as dc
        from zuspec.ir.core.base import Base
        from zuspec.synth.passes.synth_pass import SynthPass

        @dc.dataclass(kw_only=True)
        class _SrcNode(Base):
            v: int = 0

        @dc.dataclass
        class _PlainDst:
            v: int = 0

        src = _SrcNode(v=1, loc=Loc(file="a.py", line=3, pos=0))
        dst = _PlainDst(v=2)
        # Should not raise even though _PlainDst has no copy_loc.
        SynthPass.propagate_loc(src, dst)

    def test_pipeline_frontend_sets_loc_on_stage_ir(self):
        """PipelineFrontendPass._build_pipeline_ir should attach loc to StageIR from body_ast."""
        import ast
        from zuspec.synth.ir.pipeline_ir import PipelineIR
        from zuspec.ir.core.pipeline import PipelineRootIR, StageMethodIR, StageCallNode

        # Build a minimal AST body.
        func_src = "def EX(self, a, b):\n    return a + b\n"
        tree = ast.parse(func_src)
        func_def = tree.body[0]  # ast.FunctionDef with lineno=1

        smir = StageMethodIR(name="EX", body_ast=func_def)
        call = StageCallNode(stage_name="EX", arg_names=["a", "b"], return_names=["result"])
        root_ir = PipelineRootIR(stage_calls=[call])

        from zuspec.synth.passes.pipeline_frontend import PipelineFrontendPass
        from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

        cfg = SynthConfig()
        pass_ = PipelineFrontendPass(cfg)
        pip = pass_._build_pipeline_ir(
            comp_cls=type("FakeComp", (), {"__name__": "FakeComp"})(),
            root_ir=root_ir,
            stage_method_irs=[smir],
            sync_method_irs=[],
        )
        assert isinstance(pip, PipelineIR)
        assert len(pip.stages) == 1
        stage = pip.stages[0]
        # loc should be set from body_ast.lineno
        assert stage.loc is not None, "StageIR.loc should be set from body_ast"
        assert stage.loc.line == func_def.lineno
