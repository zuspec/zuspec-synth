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
    s = sched.schedule(f, arch, data_inputs)
    s.ir_sha256 = sched.ir_hash(ir_text)
    s.consts = {k: int(v) for k, v in dict(consts or {}).items()}
    s.data_inputs = list(data_inputs)
    return orig, s


def load(json_text: str, ir_text: str, arch: Optional["spec.ArchSpec"] = None,
         consts: Optional[Dict[str, int]] = None,
         data_inputs: Optional[Sequence[str]] = None):
    """(original fn, Schedule) from ``sched.to_json`` output (``ArchSched``'s
    ``<m>.<arch>.sched.full.json``) instead of scheduling; checks the IR (and
    the spec / constants / data inputs, when given) hash to what it was made
    from."""
    return sched.from_json(json_text, ir_text, arch, consts, data_inputs)
