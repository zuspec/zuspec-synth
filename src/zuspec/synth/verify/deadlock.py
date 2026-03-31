"""
Static deadlock freedom check for a lowered PipelineIR.

A linear pipeline (no back-edges) with all channels having positive depth is
structurally deadlock-free: every stage can always make forward progress once
its input channel is non-empty, because the output channel has spare capacity
(depth ≥ 1) and there are no circular dependencies.

This check is intentionally conservative — it returns PASS only when the
structural conditions are demonstrably satisfied.  Full model-checking is a
future Phase 4 item.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..ir.pipeline_ir import PipelineIR, ChannelDecl


@dataclass
class DeadlockDiagnostic:
    channel: ChannelDecl
    reason: str


def check_deadlock_freedom(
    pipeline_ir: PipelineIR,
) -> Tuple[bool, str, List[DeadlockDiagnostic]]:
    """Return (is_free, method_name, diagnostics).

    ``method_name`` is always ``"static_graph_analysis"``; it is included in
    the certificate so consumers know how the proof was produced.

    A pipeline is considered deadlock-free when:
    1. Every inter-stage channel has depth ≥ 1 (no zero-capacity edges).
    2. The channel graph is acyclic (no back-edges from later stages to
       earlier ones by index).
    """
    diagnostics: List[DeadlockDiagnostic] = []
    stage_index = {s.name: s.index for s in pipeline_ir.stages}

    for ch in pipeline_ir.channels:
        if ch.depth < 1:
            diagnostics.append(DeadlockDiagnostic(ch, "channel depth < 1"))
            continue
        src_idx = stage_index.get(ch.src_stage, -1)
        dst_idx = stage_index.get(ch.dst_stage, -1)
        if src_idx >= dst_idx:
            diagnostics.append(
                DeadlockDiagnostic(
                    ch,
                    f"back-edge detected: {ch.src_stage}[{src_idx}] "
                    f"→ {ch.dst_stage}[{dst_idx}]",
                )
            )

    return (len(diagnostics) == 0, "static_graph_analysis", diagnostics)
