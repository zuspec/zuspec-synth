"""zuspec.synth.passes — synthesis pass classes."""
from .synth_pass import SynthPass
from .elaborate import ElaboratePass
from .async_pipeline_elaborate import AsyncPipelineElaboratePass
from .async_pipeline_to_ir import AsyncPipelineToIrPass
from .fsm_extract import FSMExtractPass
from .schedule import SchedulePass, parallel, _ParallelIssue
from .lower import LowerPass
from .cert_emit import CertEmitPass
from .pipeline_annotation import PipelineAnnotationPass
from .pipeline_frontend import PipelineFrontendPass
from .auto_thread import AutoThreadPass
from .hazard_analysis import HazardAnalysisPass
from .forwarding_gen import ForwardingGenPass
from .stall_gen import StallGenPass
from .sync_body_lower import SyncBodyLowerPass
from .sdc_schedule import SDCSchedulePass
from .pipeline_sv_emit import PipelineSVCodegen, SVEmitPass

from .pipeline_to_source import PipelineToSource, PipelineToSourcePass
from .expr_lowerer import ExprLowerer, collect_ports
from .cdc_analysis import CDCAnalysisPass, CDCCrossing

__all__ = [
    "SynthPass",
    "ElaboratePass",
    "FSMExtractPass",
    "SchedulePass",
    "LowerPass",
    "CertEmitPass",
    "parallel",
    "_ParallelIssue",
    # Pipeline passes (old API — Approach C)
    "PipelineAnnotationPass",
    # Pipeline passes (new API)
    "PipelineFrontendPass",
    "AutoThreadPass",
    "SyncBodyLowerPass",
    # Async pipeline passes
    "AsyncPipelineElaboratePass",
    "AsyncPipelineToIrPass",
    # Shared pipeline passes
    "HazardAnalysisPass",
    "ForwardingGenPass",
    "StallGenPass",
    "SDCSchedulePass",
    "PipelineSVCodegen",
    "SVEmitPass",
    "PipelineToSource",
    "PipelineToSourcePass",
    "ExprLowerer",
    "collect_ports",
    "CDCAnalysisPass",
    "CDCCrossing",
]
