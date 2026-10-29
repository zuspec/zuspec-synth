"""Architecture scheduling: arch spec + algorithm dataflow graph -> schedule
(node -> step / S-box instance, latency, II) -> V3 stage-TLM SV classes.

Modules: ``xls_ir`` (XLS text IR reader, specialization), ``spec`` (arch-spec
YAML), ``sched`` (list scheduler, binding, loop folding, report),
``emit_v3`` (SV classes), ``task`` (dv-flow ``zuspec.synth.ArchSched``).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from . import emit_v3, sched, spec, xls_ir


def run(ir_text: str, arch: "spec.ArchSpec", consts: Optional[Dict[str, int]] = None,
        data_inputs: Sequence[str] = ("block_",)):
    """Parse, specialize, classify and schedule; returns (original fn, Schedule)."""
    orig = xls_ir.parse(ir_text)
    f = xls_ir.specialize(xls_ir.parse(ir_text), dict(consts or {}))
    xls_ir.classify(f)
    return orig, sched.schedule(f, arch, data_inputs)
