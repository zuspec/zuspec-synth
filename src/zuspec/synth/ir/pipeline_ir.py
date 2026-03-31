"""
Pipeline intermediate representation produced by the Lowerer pass.

``PipelineIR`` is the central data structure for Phase 2+.  It describes a
multi-stage pipeline as a list of named ``StageIR`` nodes connected by
``ChannelDecl`` (FIFO) edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..elab.elab_ir import ComponentSynthMeta, PortDecl


@dataclass
class ChannelDecl:
    """An inter-stage FIFO channel."""
    name: str        # e.g. "fetch_to_decode"
    width: int       # total data width in bits (e.g. 64 for PC+INSN)
    depth: int       # FIFO depth (typically 2)
    src_stage: str   # name of the producing StageIR (e.g. "FetchStage")
    dst_stage: str   # name of the consuming StageIR (e.g. "DecodeStage")


@dataclass
class StageIR:
    """One pipeline stage."""
    name: str                                          # e.g. "FetchStage"
    index: int                                         # 0-based
    inputs:  List[ChannelDecl] = field(default_factory=list)
    outputs: List[ChannelDecl] = field(default_factory=list)
    ports:   List["PortDecl"]  = field(default_factory=list)  # memory / external ports


@dataclass
class PipelineIR:
    """Lowered pipeline topology for a single component configuration."""
    module_name:     str                    # top-level Verilog module name
    stages:          List[StageIR]
    channels:        List[ChannelDecl]
    meta:            Optional["ComponentSynthMeta"]
    pipeline_stages: int
