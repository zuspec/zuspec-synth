# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""CombLowerPass — lower combinational processes to SV always_comb blocks."""
from __future__ import annotations

import logging

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthIR

_log = logging.getLogger(__name__)


class CombLowerPass(SynthPass):
    """Emit ``always_comb`` blocks for all comb_processes on the component.

    Reads:
        ir.component_fields: Provides ``idx_to_name`` and output-port names
            needed for default-zero assignments.
        ir.model_context: Used to look up the component IR.

    Populates:
        ir.lowered_sv["sv/module/comb"]: SV text for combinational blocks
            (empty string when no comb processes exist).
    """

    @property
    def name(self) -> str:
        return "comb_lower"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.sprtl.ir_to_sv import ir_stmts_to_sv_comb

        cf = ir.component_fields
        ctx = ir.model_context

        # Resolve component IR.
        cls = ir.component
        component_ir = None
        if ctx is not None and cls is not None:
            component_ir = (
                ctx.type_m.get(getattr(cls, "__qualname__", None))
                or ctx.type_m.get(cls.__name__)
            )

        if component_ir is None:
            ir.lowered_sv["sv/module/comb"] = ""
            return ir

        comb_processes = getattr(component_ir, "comb_processes", [])
        if not comb_processes:
            ir.lowered_sv["sv/module/comb"] = ""
            return ir

        idx_to_name = cf.idx_to_name if cf is not None else {}
        output_names = [p.name for p in cf.ports if getattr(p, 'direction', None) == "output"]

        lines: list[str] = []
        for comb_proc in comb_processes:
            lines.append("always_comb begin")
            # Latch-free: default all outputs to zero.
            for name in output_names:
                lines.append(f"  {name} = '0;")
            lines.extend(
                ir_stmts_to_sv_comb(
                    getattr(comb_proc, "body", []),
                    idx_to_name,
                    indent=2,
                )
            )
            lines.append("end")
            lines.append("")

        ir.lowered_sv["sv/module/comb"] = "\n".join(lines)
        _log.debug("[CombLowerPass] emitted %d comb blocks", len(comb_processes))
        return ir
