"""
Lowering pass: converts ComponentSynthMeta + pipeline_stages into PipelineIR.

``Lowerer.lower()`` partitions the logical stage list (from ``_STAGE_NAMES``)
into ``StageIR`` nodes and builds ``ChannelDecl`` edges for each adjacent
stage pair.  This is a structural lowering; full schedule-driven partitioning
(where FSM states are assigned to stages) is a Phase 3 item.
"""
from __future__ import annotations

from typing import List, Optional

from ..ir.pipeline_ir import ChannelDecl, PipelineIR, StageIR
from .elab_ir import ComponentSynthMeta

# Re-use the same stage-key lists as mls.py
_STAGE_NAMES = {
    1: ["EXECUTE"],
    2: ["FETCH", "EXECUTE"],
    3: ["FETCH", "DECODE", "EXECUTE"],
    4: ["FETCH", "DECODE", "EXECUTE", "WRITEBACK"],
    5: ["FETCH", "DECODE", "REG_READ", "EXECUTE", "WRITEBACK"],
    6: ["FETCH", "DECODE", "REG_READ", "EXECUTE", "MEM_ACCESS", "WRITEBACK"],
    7: ["FETCH", "DECODE", "REG_READ", "ISSUE_A", "ISSUE_B", "EXECUTE", "WRITEBACK"],
}

# Override module names for keys whose natural capitalisation is wrong
_STAGE_MODULE_OVERRIDES = {
    "WRITEBACK": "WriteBackStage",
}

# Payload widths per channel (PC[XLEN-1:0] + insn[31:0])
_CHANNEL_WIDTH = 64   # conservative: always use 64 bits
_CHANNEL_DEPTH = 2


def _stage_module_name(key: str) -> str:
    """Convert a stage key ('REG_READ') to a Verilog module name ('RegReadStage')."""
    if key in _STAGE_MODULE_OVERRIDES:
        return _STAGE_MODULE_OVERRIDES[key]
    parts = key.split("_")
    return "".join(p.capitalize() for p in parts) + "Stage"


def _channel_name(src_key: str, dst_key: str) -> str:
    """Build a channel identifier from two stage keys, e.g. 'fetch_to_decode'."""
    return f"{src_key.lower()}_to_{dst_key.lower()}"


class Lowerer:
    """Structural pipeline lowering pass."""

    def lower(
        self,
        meta: Optional[ComponentSynthMeta],
        pipeline_stages: int,
        module_name: str,
    ) -> PipelineIR:
        """Return a ``PipelineIR`` for the given stage count."""
        stage_keys = _STAGE_NAMES.get(
            pipeline_stages,
            [f"S{i}" for i in range(pipeline_stages)],
        )

        stages: List[StageIR] = [
            StageIR(name=_stage_module_name(k), index=i)
            for i, k in enumerate(stage_keys)
        ]

        channels: List[ChannelDecl] = []
        for i in range(len(stages) - 1):
            ch = ChannelDecl(
                name=_channel_name(stage_keys[i], stage_keys[i + 1]),
                width=_CHANNEL_WIDTH,
                depth=_CHANNEL_DEPTH,
                src_stage=stages[i].name,
                dst_stage=stages[i + 1].name,
            )
            channels.append(ch)
            stages[i].outputs.append(ch)
            stages[i + 1].inputs.append(ch)

        return PipelineIR(
            module_name=module_name,
            stages=stages,
            channels=channels,
            meta=meta,
            pipeline_stages=pipeline_stages,
        )
