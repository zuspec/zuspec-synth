"""IfProtocolLowerPass — lower IfProtocol ports/exports to IfProtocolPortIR nodes.

Traverses the top-level component's fields; for each field whose type is an
``IfProtocolType`` (or ``IfProtocolExport``), computes the synthesis scenario
(A, B, C, or D), builds the list of SV signals, and appends an
``IfProtocolPortIR`` to ``SynthIR.protocol_ports``.

Scenario selection rules:
  A → ``req_always_ready=True`` AND ``resp_always_valid=True`` AND ``max_outstanding==1``
  B → ``max_outstanding==1`` (default)
  C → ``max_outstanding > 1`` AND ``in_order=True``
  D → ``max_outstanding > 1`` AND ``in_order=False``
"""
from __future__ import annotations

import inspect
import logging
import math
import typing
from typing import Any, List, Optional

from zuspec.dataclasses import ir as zdc_ir
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR
from zuspec.synth.ir.protocol_ir import (
    IfProtocolPortIR,
    IfProtocolScenario,
    ProtocolField,
)
from zuspec.synth.passes.synth_pass import SynthPass

_log = logging.getLogger(__name__)


def _annotation_bits(annotation: Any) -> int:
    """Return bit width for a zdc integer type annotation (e.g. zdc.u32 → 32).

    Falls back to 32 for unknown types.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return 32
    # zdc types are Annotated[int, U(width=N, signed=B)]
    meta = getattr(annotation, "__metadata__", None)
    if meta:
        for m in meta:
            w = getattr(m, "width", None)
            if w is not None:
                return int(w)
    # Plain Python int → 32 bits
    if annotation is int:
        return 32
    return 32


def _fields_from_signatures(proto_cls: Any) -> tuple[List[ProtocolField], List[ProtocolField]]:
    """Extract req and resp ProtocolField lists from an IfProtocol class's method signatures.

    Iterates every public async method declared on *proto_cls* (bodies ``...``).
    Parameters other than ``self`` become request fields; non-None return types
    become response fields (named ``<method>_ret`` when multiple methods exist,
    or ``data`` when there is only one method).
    """
    req: List[ProtocolField] = []
    resp: List[ProtocolField] = []

    methods = [
        (name, member)
        for name, member in inspect.getmembers(proto_cls, predicate=inspect.isfunction)
        if not name.startswith("_") and inspect.iscoroutinefunction(member)
    ]

    for _mname, method in methods:
        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            continue

        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            width = _annotation_bits(param.annotation)
            req.append(ProtocolField(name=pname, width=width, is_response=False))

        ret = sig.return_annotation
        if ret is not inspect.Parameter.empty and ret is not type(None) and ret is not None:
            width = _annotation_bits(ret)
            resp.append(ProtocolField(name="data", width=width, is_response=True))

    return req, resp


def _is_ifprotocol_cls(t: Any) -> bool:
    """Return True if *t* is an IfProtocol subclass (not the base itself)."""
    try:
        from zuspec.dataclasses.if_protocol import IfProtocol
        return (
            isinstance(t, type)
            and issubclass(t, IfProtocol)
            and t is not IfProtocol
        )
    except ImportError:
        return False


def _select_scenario(props: Any) -> IfProtocolScenario:
    """Return the synthesis scenario based on IfProtocolProperties."""
    if props is None:
        return IfProtocolScenario.B

    mo = getattr(props, "max_outstanding", 1) or 1
    rar = getattr(props, "req_always_ready", False)
    rav = getattr(props, "resp_always_valid", False)
    in_order = getattr(props, "in_order", True)

    if mo == 1:
        if rar and rav:
            return IfProtocolScenario.A
        return IfProtocolScenario.B
    else:
        if in_order:
            return IfProtocolScenario.C
        return IfProtocolScenario.D


def _id_bits(props: Any) -> int:
    """Number of ID bits needed for out-of-order protocols."""
    if props is None:
        return 0
    mo = getattr(props, "max_outstanding", 1) or 1
    in_order = getattr(props, "in_order", True)
    if mo > 1 and not in_order:
        return max(1, (mo - 1).bit_length())
    return 0


def _protocol_fields_from_cls(proto_cls: Any, is_response: bool) -> List[ProtocolField]:
    """Extract ProtocolField list from an IfProtocol class.

    First tries the legacy ``req_fields``/``resp_fields`` dict attributes,
    then falls back to parsing method signatures via :func:`_fields_from_signatures`.
    """
    if proto_cls is None:
        return []

    # Legacy dict attributes (set by earlier code paths)
    fields_attr = "resp_fields" if is_response else "req_fields"
    raw = getattr(proto_cls, fields_attr, None)
    if raw is not None:
        return [ProtocolField(name=fname, width=fwidth, is_response=is_response)
                for fname, fwidth in raw.items()]

    # Extract from method signatures
    req, resp = _fields_from_signatures(proto_cls)
    return resp if is_response else req


def _guess_fields_from_dtype(
    proto_type: Any, is_export: bool, is_response: bool
) -> List[ProtocolField]:
    """Fallback: infer fields from an IfProtocolType IR node's struct."""
    if proto_type is None:
        return []

    struct = getattr(proto_type, "resp_struct" if is_response else "req_struct", None)
    if struct is None:
        return []

    result = []
    for field in getattr(struct, "fields", []):
        w = 32
        dt = getattr(field, "datatype", None)
        if isinstance(dt, zdc_ir.DataTypeInt):
            w = max(8, dt.bits)
        result.append(ProtocolField(name=field.name, width=w, is_response=is_response))
    return result


class IfProtocolLowerPass(SynthPass):
    """Populate ``SynthIR.protocol_ports`` from component IfProtocol fields.

    Handles two cases:

    1. **Model-context path**: ``DataModelFactory`` has stored an
       ``IfProtocolType`` node for the field — original behaviour.
    2. **Annotation fallback**: ``DataModelFactory`` stored a plain
       ``DataTypeRef`` (the typical result today).  The pass falls back to
       inspecting ``comp_cls.__annotations__`` directly, resolving each
       annotation that is an ``IfProtocol`` subclass.
    """

    def __init__(self, config: SynthConfig = None) -> None:
        super().__init__(config or SynthConfig())

    @property
    def name(self) -> str:
        return "if_protocol_lower"

    def run(self, ir: SynthIR) -> SynthIR:
        ctxt = ir.model_context
        if ctxt is None:
            return ir

        comp_cls = ir.component
        comp_name = getattr(comp_cls, "__name__", None) if comp_cls else None
        if comp_name is None:
            return ir

        comp_dtype = None
        if hasattr(ctxt, "type_m"):
            for name, dtype in ctxt.type_m.items():
                if name == comp_name and isinstance(dtype, zdc_ir.DataTypeComponent):
                    comp_dtype = dtype
                    break

        if comp_dtype is None:
            _log.debug("[IfProtocolLowerPass] component '%s' not found in type_m", comp_name)
            return ir

        # Build a lookup from field name → Python annotation for the fallback path.
        py_annotations: dict = {}
        if comp_cls is not None:
            try:
                py_annotations = typing.get_type_hints(comp_cls)
            except Exception:
                py_annotations = getattr(comp_cls, "__annotations__", {})

        for field in getattr(comp_dtype, "fields", []):
            dtype = field.datatype

            # --- Path 1: model_context stores an IfProtocolType node --------
            is_proto = hasattr(zdc_ir, "IfProtocolType") and isinstance(dtype, zdc_ir.IfProtocolType)
            if is_proto:
                props = getattr(dtype, "properties", None)
                proto_cls = getattr(dtype, "protocol_cls", None)
                is_export = getattr(dtype, "is_export", False)
                scenario = _select_scenario(props)

                req_fields = _protocol_fields_from_cls(proto_cls, is_response=False)
                resp_fields = _protocol_fields_from_cls(proto_cls, is_response=True)

                if not req_fields:
                    req_fields = _guess_fields_from_dtype(dtype, is_export, is_response=False)
                if not resp_fields:
                    resp_fields = _guess_fields_from_dtype(dtype, is_export, is_response=True)

                node = IfProtocolPortIR(
                    name=field.name,
                    is_export=is_export,
                    scenario=scenario,
                    properties=props,
                    req_fields=req_fields,
                    resp_fields=resp_fields,
                    protocol_cls=proto_cls,
                    id_bits=_id_bits(props),
                )
                ir.protocol_ports.append(node)
                _log.debug(
                    "[IfProtocolLowerPass] (IR path) %s.%s → scenario=%s is_export=%s",
                    comp_name, field.name, scenario.value, is_export,
                )
                continue

            # --- Path 2: annotation fallback (DataTypeRef / plain DataType) --
            py_type = py_annotations.get(field.name)
            if py_type is None or not _is_ifprotocol_cls(py_type):
                continue

            proto_cls = py_type
            props = proto_cls._get_ir_properties()
            is_export = False  # annotation path cannot distinguish export vs port yet
            scenario = _select_scenario(props)
            req_fields, resp_fields = _fields_from_signatures(proto_cls)

            node = IfProtocolPortIR(
                name=field.name,
                is_export=is_export,
                scenario=scenario,
                properties=props,
                req_fields=req_fields,
                resp_fields=resp_fields,
                protocol_cls=proto_cls,
                id_bits=_id_bits(props),
            )
            ir.protocol_ports.append(node)
            _log.debug(
                "[IfProtocolLowerPass] (annotation path) %s.%s → scenario=%s",
                comp_name, field.name, scenario.value,
            )

        return ir
