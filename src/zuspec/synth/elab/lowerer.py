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

_CHANNEL_DEPTH = 2
_CHANNEL_DEPTH_DEEP = 4        # depth used for stages with variable-latency ops
# Stages whose downstream channel should use the deep-buffered skid buffer
# (variable-latency units like multiply/divide/FP stall for multiple cycles)
_DEEP_CHANNEL_AFTER = {"EXECUTE", "MEM_ACCESS"}

# Maps bundle field name → preferred stage key; first match in the stage_keys
# list wins.  For dcache, MEM_ACCESS is preferred over EXECUTE so that the
# data-cache interface is exposed on the right stage in 6/7-stage pipelines.
_BUNDLE_STAGE_PREFERENCE = {
    "icache":       ["FETCH"],
    "dcache":       ["MEM_ACCESS", "EXECUTE"],
    "dcache_load":  ["MEM_ACCESS", "EXECUTE"],
    "dcache_store": ["MEM_ACCESS", "EXECUTE"],
}


def channel_payload_width(src_key: str, xlen: int) -> int:
    """Return the packed-bus width for the channel produced by *src_key*.

    Payload layouts (LSB first):

    **FETCH output** — ``{insn[31:0], pc[xlen-1:0]}``::

        bits[xlen-1:0]      = PC
        bits[xlen+31:xlen]  = INSN (32-bit instruction word)

        total = xlen + 32

    **DECODE output** — fetch payload + decoded fields::

        bits[+4:+0]          = rd[4:0]
        bits[+9:+5]          = rs1[4:0]
        bits[+14:+10]        = rs2[4:0]
        bits[+17:+15]        = funct3[2:0]
        bits[+24:+18]        = funct7[6:0]
        bits[+34:+25]        = optype[9:0]
        bits[+34+xlen:+35]   = imm[xlen-1:0]  (sign-extended)

        total = (xlen + 32) + 5+5+5+3+7+10+xlen = 2*xlen + 67

    **REG_READ output** — decode payload + register-file values::

        bits[+xlen-1:+0]     = rs1_val[xlen-1:0]
        bits[+2*xlen-1:+xlen]= rs2_val[xlen-1:0]

        total = (2*xlen + 67) + 2*xlen = 4*xlen + 67

    **EXECUTE / MEM_ACCESS / ISSUE_* output** — writeback bundle::

        bits[4:0]             = rd[4:0]
        bits[xlen+4:5]        = result[xlen-1:0]
        bits[xlen+5]          = we         (register write enable)
        bits[xlen+6]          = branch_taken
        bits[2*xlen+6:xlen+7] = branch_target[xlen-1:0]

        total = 2*xlen + 7
    """
    if src_key == "FETCH":
        return xlen + 32
    elif src_key == "DECODE":
        return 2 * xlen + 67
    elif src_key == "REG_READ":
        return 4 * xlen + 67
    elif src_key in ("ISSUE_A", "ISSUE_B"):
        # Issue slots pass the full decode+regread payload through to EXECUTE
        return 4 * xlen + 67
    elif src_key in ("EXECUTE", "MEM_ACCESS"):
        return 2 * xlen + 7
    else:
        # Unknown stage key — fall back to xlen
        return xlen


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
        # Derive xlen from config (fall back to 32 if unavailable).
        xlen = 32
        if meta is not None and meta.config is not None:
            cfg = meta.config
            if isinstance(cfg, dict):
                xlen = cfg.get("xlen", 32)
            else:
                xlen = getattr(cfg, "xlen", 32)

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
            src_key = stage_keys[i]
            is_deep = src_key in _DEEP_CHANNEL_AFTER
            depth = _CHANNEL_DEPTH_DEEP if is_deep else _CHANNEL_DEPTH
            ch = ChannelDecl(
                name=_channel_name(stage_keys[i], stage_keys[i + 1]),
                width=channel_payload_width(src_key, xlen),
                depth=depth,
                src_stage=stages[i].name,
                dst_stage=stages[i + 1].name,
            )
            channels.append(ch)
            stages[i].outputs.append(ch)
            stages[i + 1].inputs.append(ch)

        # Assign bundle ports (icache, dcache, …) from meta to the appropriate
        # stage based on _BUNDLE_STAGE_PREFERENCE.
        meta_ports = getattr(meta, 'ports', None) if meta is not None else None
        if meta_ports:
            # Build lookup: stage_key → StageIR
            key_to_stage = {k: stages[i] for i, k in enumerate(stage_keys)}
            # Group PortDecls by bundle name
            bundles: dict[str, list] = {}
            for pd in meta_ports:
                bundles.setdefault(pd.bundle, []).append(pd)
            for bundle_name, port_list in bundles.items():
                preferred_keys = _BUNDLE_STAGE_PREFERENCE.get(bundle_name, [stage_keys[0]])
                target_stage = None
                for key in preferred_keys:
                    if key in key_to_stage:
                        target_stage = key_to_stage[key]
                        break
                if target_stage is None:
                    # Fall back to first stage
                    target_stage = stages[0]
                target_stage.ports.extend(port_list)

        return PipelineIR(
            module_name=module_name,
            stages=stages,
            channels=channels,
            meta=meta,
            pipeline_stages=pipeline_stages,
        )
