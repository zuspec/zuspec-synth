"""IRLayer -- named IR abstraction layers for the Zuspec synthesis pipeline.

Each layer corresponds to a stable point in the pass sequence where the IR
satisfies a well-defined set of invariants.  See the rationale docs under
``docs/rationale/`` for the full description of each layer.
"""
from __future__ import annotations

from enum import Enum, auto


class IRLayer(Enum):
    """Named abstraction layers of the Zuspec synthesis IR.

    Layers are ordered: ``ACTIVITY < SCHEDULED < PIPELINE < STRUCTURAL``.
    A higher-numbered layer is always a more concrete representation than a
    lower-numbered one.

    Attributes:
        ACTIVITY:   Post-``DataModelFactory``, pre-scheduling.  The IR contains
                    Activity IR nodes (``ActivitySequenceBlock``, traversals,
                    etc.) but no schedule assignment.
        SCHEDULED:  Post-``SchedulePass`` / ``SDCSchedulePass``.  Operations
                    have been assigned to pipeline stages; ``ir.schedule_obj``
                    is non-``None``.
        PIPELINE:   Post-``PipelineFrontendPass`` / ``AsyncPipelineToIrPass``.
                    A concrete ``PipelineIR`` with ``StageIR`` entries exists;
                    ``ir.pipeline_ir`` is non-``None``.
        STRUCTURAL: Post-``LowerPass`` / ``SyncBodyLowerPass``, pre-SV emit.
                    All ``DomainNode`` instances are lowered; ``ir.meta`` is
                    non-``None`` and ``ir.lowered_sv`` keys are validated.
    """

    ACTIVITY = auto()
    SCHEDULED = auto()
    PIPELINE = auto()
    STRUCTURAL = auto()
