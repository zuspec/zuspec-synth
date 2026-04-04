"""ElaboratePass — run the elaborator and populate ir.meta."""
from __future__ import annotations

import logging
from typing import Any

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


class ElaboratePass(SynthPass):
    """Run the structural elaborator on *component_cls*.

    Stores the resulting ``ComponentSynthMeta`` in ``ir.meta`` and sets
    ``ir.component`` / ``ir.config`` for use by downstream passes.

    Args:
        component_cls: The top-level ``@zdc.dataclass`` component class.
        component_config: Component-level configuration (e.g. ``RVConfig``).
    """

    def __init__(self, component_cls: Any, component_config: Any) -> None:
        super().__init__(config=SynthConfig())
        self._cls = component_cls
        self._component_config = component_config

    @property
    def name(self) -> str:
        return "elaborate"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.elab.elaborator import Elaborator

        ir.component = self._cls
        ir.config = self._component_config
        try:
            ir.meta = Elaborator().elaborate(self._cls, self._component_config)
            _log.info("[ElaboratePass] elaborated %s", self._cls.__name__)
        except Exception as exc:
            _log.warning("[ElaboratePass] elaborator failed (%s) — meta not set", exc)
        return ir
