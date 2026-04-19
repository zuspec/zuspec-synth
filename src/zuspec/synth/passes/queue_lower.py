"""QueueLowerPass — lower zdc.Queue[T] fields to QueueIR synthesis nodes.

Traverses component fields in the model context looking for ``QueueType``
IR nodes and emits a ``QueueIR`` entry in ``SynthIR.queue_nodes`` for each.

The bit width of the element type is inferred from the ``DataTypeInt`` or
via a simple heuristic (default 32 bits) when a precise type is unavailable.

When ``DataModelFactory`` stores a field as a plain ``DataType`` (not
``QueueType``), the pass falls back to inspecting ``comp_cls.__annotations__``
and ``__dataclass_fields__`` for ``Queue[T]`` annotations.
"""
from __future__ import annotations

import logging
import math
import typing
from typing import Any

from zuspec.dataclasses import ir as zdc_ir
from zuspec.synth.ir.synth_ir import SynthIR
from zuspec.synth.ir.protocol_ir import QueueIR
from zuspec.synth.ir.synth_ir import SynthConfig
from zuspec.synth.passes.synth_pass import SynthPass

_log = logging.getLogger(__name__)


def _elem_width(dtype) -> int:
    """Infer element bit width from a DataType."""
    if dtype is None:
        return 32
    if isinstance(dtype, zdc_ir.DataTypeInt):
        bits = dtype.bits
        if bits <= 0:
            return 32
        for w in (8, 16, 32, 64):
            if bits <= w:
                return w
        return bits
    if isinstance(dtype, zdc_ir.DataTypeRef):
        return 32
    return 32


def _annotation_bits(annotation: Any) -> int:
    """Return bit width for a zdc integer type annotation (e.g. zdc.u32 → 32)."""
    if annotation is None:
        return 32
    meta = getattr(annotation, "__metadata__", None)
    if meta:
        for m in meta:
            w = getattr(m, "width", None)
            if w is not None:
                return int(w)
    return 32


def _is_queue_alias(t: Any) -> bool:
    """Return True if *t* is a ``Queue[T]`` alias."""
    try:
        from zuspec.dataclasses.queue_type import _QueueAlias
        return isinstance(t, _QueueAlias)
    except ImportError:
        return False


class QueueLowerPass(SynthPass):
    """Produces ``QueueIR`` nodes for every ``Queue[T]`` field in the top-level component."""

    def __init__(self, config: SynthConfig = None) -> None:
        super().__init__(config or SynthConfig())

    @property
    def name(self) -> str:
        return "queue_lower"

    def run(self, ir: SynthIR) -> SynthIR:
        ctxt = ir.model_context
        if ctxt is None:
            return ir

        comp_cls = ir.component
        comp_name = getattr(comp_cls, "__name__", None) if comp_cls else None
        if comp_name is None:
            return ir

        comp_dtype = None
        for name, dtype in ctxt.type_m.items():
            if name == comp_name and isinstance(dtype, zdc_ir.DataTypeComponent):
                comp_dtype = dtype
                break

        if comp_dtype is None:
            return ir

        # Build annotation + dataclass_fields lookup for the fallback path.
        py_annotations: dict = {}
        dc_fields: dict = {}
        if comp_cls is not None:
            try:
                py_annotations = typing.get_type_hints(comp_cls)
            except Exception:
                py_annotations = getattr(comp_cls, "__annotations__", {})
            dc_fields = getattr(comp_cls, "__dataclass_fields__", {})

        for field in getattr(comp_dtype, "fields", []):
            dtype = field.datatype

            # --- Path 1: model_context stores a QueueType node ---------------
            if hasattr(zdc_ir, "QueueType") and isinstance(dtype, zdc_ir.QueueType):
                elem_w = _elem_width(getattr(dtype, "element_type", None))
                depth = getattr(dtype, "depth", 16) or 16
                node = QueueIR(name=field.name, elem_width=elem_w, depth=depth)
                ir.queue_nodes.append(node)
                _log.debug("[QueueLowerPass] (IR path) %s.%s → QueueIR(w=%d, d=%d)",
                           comp_name, field.name, elem_w, depth)
                continue

            # --- Path 2: annotation fallback (Queue[T] alias) ----------------
            py_type = py_annotations.get(field.name)
            if py_type is None or not _is_queue_alias(py_type):
                continue

            elem_w = _annotation_bits(py_type._item)

            # Depth from the default value (e.g. QueueRT(depth=4))
            dc_field = dc_fields.get(field.name)
            depth = 16
            if dc_field is not None:
                default = dc_field.default
                depth = int(getattr(default, "depth", 16) or 16)

            node = QueueIR(name=field.name, elem_width=elem_w, depth=depth)
            ir.queue_nodes.append(node)
            _log.debug("[QueueLowerPass] (annotation path) %s.%s → QueueIR(w=%d, d=%d)",
                       comp_name, field.name, elem_w, depth)

        return ir
