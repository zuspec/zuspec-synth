# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""CombToPythonPass — lower combinational processes to ``@zdc.comb`` methods.

Place in pass chain
-------------------
Runs after ``FSMToPythonPass``; before ``ModuleAssemblePythonPass``.

Inputs
------
- ``ir.component_fields`` — Provides ``idx_to_name`` and output-port names.
- ``ir.model_context``    — Used to look up the component IR.
- ``ir.component``        — The Python component class.

Outputs
-------
- ``ir.lowered_py["py/module/comb"]`` — Python text for ``@zdc.comb``
  method(s), or empty string when no comb processes exist.
"""
from __future__ import annotations

import logging

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthIR

_log = logging.getLogger(__name__)


class CombToPythonPass(SynthPass):
    """Emit ``@zdc.comb`` methods for all combinational processes.

    Each ``@zdc.comb`` process in the component model becomes a Python
    method decorated with ``@zdc.comb``.  Output ports default to zero
    at the top of the method body to avoid latches (mirroring
    ``CombLowerPass``'s ``always_comb`` default-zero strategy).
    """

    @property
    def name(self) -> str:
        return "comb_to_python"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.sprtl.ir_to_python import ir_stmts_to_python

        cf = ir.component_fields
        ctx = ir.model_context

        # Resolve component IR (same lookup as CombLowerPass)
        cls = ir.component
        component_ir = None
        if ctx is not None and cls is not None:
            component_ir = (
                ctx.type_m.get(getattr(cls, "__qualname__", None))
                or ctx.type_m.get(cls.__name__)
            )

        if component_ir is None:
            ir.lowered_py["py/module/comb"] = ""
            return ir

        comb_processes = getattr(component_ir, "comb_processes", [])
        if not comb_processes:
            ir.lowered_py["py/module/comb"] = ""
            return ir

        idx_to_name = cf.idx_to_name if cf is not None else {}
        output_names = [
            p.name for p in cf.ports if getattr(p, "direction", None) == "output"
        ]

        method_blocks: list[str] = []
        for proc_idx, comb_proc in enumerate(comb_processes):
            method_name = getattr(comb_proc, "name", None) or f"_comb_{proc_idx}"
            lines = [f"    @zdc.comb", f"    def {method_name}(self):"]

            # Default outputs to zero (latch-free)
            for name in output_names:
                lines.append(f"        self.{name} = 0")

            # Emit body
            body_lines = ir_stmts_to_python(
                getattr(comb_proc, "body", []),
                idx_to_name,
                indent=8,
            )
            if body_lines:
                lines.extend(body_lines)
            else:
                lines.append("        pass")

            method_blocks.append("\n".join(lines))

        ir.lowered_py["py/module/comb"] = "\n\n".join(method_blocks)
        _log.debug(
            "[CombToPythonPass] emitted %d comb method(s)", len(comb_processes)
        )
        return ir
