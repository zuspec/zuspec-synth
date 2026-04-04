"""LowerPass — lower the scheduled IR into an explicit pipeline topology."""
from __future__ import annotations

import logging

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


class LowerPass(SynthPass):
    """Lower the scheduled IR into a concrete :class:`~zuspec.synth.ir.pipeline_ir.PipelineIR`.

    Calls :class:`~zuspec.synth.elab.lowerer.Lowerer` with
    ``config.pipeline_stages`` and the component name derived from
    ``ir.component`` and ``ir.config``.  Stores the result in ``ir.pipeline_ir``.
    """

    @property
    def name(self) -> str:
        return "lower"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.elab.lowerer import Lowerer

        comp_name = getattr(ir.component, "__name__", "Component") if ir.component else "Component"
        isa_str = ""
        if ir.config is not None and hasattr(ir.config, "isa_spec"):
            isa_str = f"_{ir.config.isa_spec()}"
        module_name = f"{comp_name}{isa_str}"

        try:
            ir.pipeline_ir = Lowerer().lower(ir.meta, self.config.pipeline_stages, module_name)
            _log.info(
                "[LowerPass] lowered to %d-stage pipeline (%s)",
                self.config.pipeline_stages,
                module_name,
            )
        except Exception as exc:
            _log.warning("[LowerPass] Lowerer failed (%s) — pipeline_ir not set", exc)
        return ir
