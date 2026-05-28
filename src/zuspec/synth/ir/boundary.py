"""IRBoundarySerializer / IRBoundaryDeserializer -- the post-elaboration IR snapshot.

The "post-elaboration boundary" is the ``SynthIR`` object after
``ElaboratePass``, ``FSMExtractPass``, and ``SchedulePass`` have run, and
before any SV-emission pass.  This module serializes and reloads that snapshot
so that:

- SV-emission passes can be replayed in isolation (``--load-ir`` CLI flag).
- IR-level regression tests can diff the scheduler output without running the
  full frontend.
- A future CIRCT bridge can read the snapshot directly.

Fields that reference live Python class objects or contain non-serializable
closures are **omitted or stubbed**; the reconstructed ``SynthIR`` is
sufficient for the emission passes but cannot re-run elaboration or scheduling.
"""
from __future__ import annotations

import dataclasses as dc
from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from zuspec.synth.ir.synth_ir import SynthIR
    from zuspec.synth.ir.layers import IRLayer

from zuspec.ir.core.serializer import IRSerializer, _SCHEMA_VERSION
from zuspec.ir.core.deserializer import IRDeserializer, IRDeserializeError

import yaml


class IRBoundarySerializer(IRSerializer):
    """Serialize a post-elaboration ``SynthIR`` snapshot to YAML.

    Extends :class:`~zuspec.ir.core.serializer.IRSerializer` with ``SynthIR``-
    specific handling: live Python class references (``component``,
    ``decode_cls``, etc.) are omitted; plain-data fields are included.
    """

    #: Fields of ``SynthIR`` that reference live Python objects or are not
    #: needed by emission passes and are therefore excluded from the snapshot.
    _EXCLUDED_FIELDS = frozenset({
        "component",
        "config",
        "decode_cls",
        "decode_c_cls",
        "execute_cls",
    })

    def serialize_synth_ir(self, ir: "SynthIR", layer: "IRLayer") -> str:
        """Serialize *ir* to a YAML boundary snapshot.

        Args:
            ir: The ``SynthIR`` to serialize.
            layer: The :class:`~zuspec.synth.ir.layers.IRLayer` at which the
                snapshot is taken (typically ``IRLayer.SCHEDULED``).

        Returns:
            YAML text suitable for storage and later reload.
        """
        self._seen: set = set()
        layer_name = layer.name if hasattr(layer, "name") else str(layer)

        # Build a filtered dict of SynthIR fields.
        payload: dict = {"_type": type(ir).__name__}
        if dc.is_dataclass(ir):
            for f in dc.fields(ir):
                if f.name in self._EXCLUDED_FIELDS:
                    continue
                if f.name.startswith("_"):
                    continue
                payload[f.name] = self._to_dict(getattr(ir, f.name))

        root = {
            "_schema_version": _SCHEMA_VERSION,
            "_layer": layer_name,
            **payload,
        }
        return yaml.dump(root, default_flow_style=False, allow_unicode=True, sort_keys=False)


class IRBoundaryDeserializer(IRDeserializer):
    """Reconstruct a ``SynthIR`` from a boundary snapshot YAML.

    The reconstructed object is sufficient for emission passes (``SVEmitPass``,
    ``CertEmitPass``) to run, but the excluded fields (``component``, etc.)
    are ``None``.
    """

    def deserialize_synth_ir(self, yaml_text: str) -> Tuple["SynthIR", "IRLayer"]:
        """Load a boundary snapshot and return a ``(SynthIR, IRLayer)`` pair.

        Args:
            yaml_text: YAML text produced by :class:`IRBoundarySerializer`.

        Returns:
            ``(ir, layer)`` where *ir* is a partially reconstructed ``SynthIR``.

        Raises:
            IRDeserializeError: If the YAML is malformed or contains unknown type tags.
        """
        from zuspec.synth.ir.synth_ir import SynthIR
        from zuspec.synth.ir.layers import IRLayer

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise IRDeserializeError("Expected a YAML mapping at the root")

        layer_name = data.get("_layer", "SCHEDULED")
        try:
            layer = IRLayer[layer_name]
        except KeyError:
            layer = layer_name  # type: ignore[assignment]

        # Register SynthIR and pipeline IR types.
        self._ensure_registered()

        # Build SynthIR from the non-header, non-excluded fields.
        fields_data = {
            k: v for k, v in data.items()
            if k not in ("_schema_version", "_layer", "_type")
        }

        field_names = {f.name for f in dc.fields(SynthIR)}
        kwargs: dict = {}
        for k, v in fields_data.items():
            if k in field_names:
                try:
                    kwargs[k] = self._from_dict(v, path=k)
                except IRDeserializeError:
                    # Unknown sub-types are skipped; emission passes degrade gracefully.
                    kwargs[k] = None

        ir = SynthIR(**kwargs)
        return ir, layer

    def _ensure_registered(self) -> None:
        """Auto-register all known pipeline and synth IR types."""
        from zuspec.synth.ir import pipeline_ir as pip_mod
        from zuspec.synth.ir import protocol_ir as proto_mod
        from zuspec.synth.ir import synth_ir as si_mod

        for mod in (pip_mod, proto_mod, si_mod):
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and dc.is_dataclass(obj):
                    self._registry.setdefault(obj.__name__, obj)
