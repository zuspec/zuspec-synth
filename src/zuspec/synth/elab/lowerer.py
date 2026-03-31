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
# Actual width is resolved at lower() time from meta.config.xlen.
_CHANNEL_WIDTH_DEFAULT = 64   # fallback when xlen is not available
_CHANNEL_DEPTH = 2
_CHANNEL_DEPTH_DEEP = 4        # depth used for stages with variable-latency ops
# Stages whose downstream channel should use the deep-buffered skid buffer
# (variable-latency units like multiply/divide/FP stall for multiple cycles)
_DEEP_CHANNEL_AFTER = {"EXECUTE", "MEM_ACCESS"}


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
        # Derive channel width from config.xlen (fall back to 64 if unavailable).
        # meta.config may be a dict (from dataclass-style elaborator) or an object.
        xlen = _CHANNEL_WIDTH_DEFAULT
        if meta is not None and meta.config is not None:
            cfg = meta.config
            if isinstance(cfg, dict):
                xlen = cfg.get("xlen", _CHANNEL_WIDTH_DEFAULT)
            else:
                xlen = getattr(cfg, "xlen", _CHANNEL_WIDTH_DEFAULT)
        # Payload = PC (xlen bits) + insn (32 bits), but keep as xlen for simplicity
        channel_width = xlen

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
            # Use a deeper channel after variable-latency stages so the
            # skid buffer template is selected in the emitter.
            src_key = stage_keys[i]
            is_deep = src_key in _DEEP_CHANNEL_AFTER
            depth = _CHANNEL_DEPTH_DEEP if is_deep else _CHANNEL_DEPTH
            ch = ChannelDecl(
                name=_channel_name(stage_keys[i], stage_keys[i + 1]),
                width=channel_width,
                depth=depth,
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
