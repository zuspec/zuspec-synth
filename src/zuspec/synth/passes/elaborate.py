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
        from zuspec.synth.passes.protocol_compat import ProtocolCompatChecker

        ir.component = self._cls
        ir.config = self._component_config
        try:
            ir.meta = Elaborator().elaborate(self._cls, self._component_config)
            _log.info("[ElaboratePass] elaborated %s", self._cls.__name__)
        except Exception as exc:
            _log.warning("[ElaboratePass] elaborator failed (%s) — meta not set", exc)

        try:
            from zuspec.dataclasses.data_model_factory import DataModelFactory
            ir.model_context = DataModelFactory().build(self._cls)
            _log.info("[ElaboratePass] built model context (%d types)",
                      len(ir.model_context.type_m))
        except Exception as exc:
            _log.warning("[ElaboratePass] DataModelFactory failed (%s) — model_context not set", exc)

        self._run_protocol_compat(ir)
        return ir

    def _run_protocol_compat(self, ir: SynthIR) -> None:
        """Validate IfProtocol fields on the top-level component class."""
        import dataclasses as dc
        try:
            from zuspec.ir.core.data_type import IfProtocolProperties
            from zuspec.synth.passes.protocol_compat import ProtocolCompatChecker, ProtocolCompatError
        except ImportError:
            return

        checker = ProtocolCompatChecker()

        if not dc.is_dataclass(self._cls):
            return

        for f in dc.fields(self._cls):
            kind = f.metadata.get('kind')
            if kind not in ('port', 'export'):
                continue
            props = f.metadata.get('protocol_props')
            if props is None:
                continue
            location = f"{self._cls.__name__}.{f.name}"
            try:
                checker.validate(props, location)
            except ProtocolCompatError as exc:
                _log.error("[ElaboratePass] Protocol error: %s", exc)
                raise
