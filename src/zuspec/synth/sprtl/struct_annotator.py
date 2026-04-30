"""Struct annotator: build FSMStructDef / FSMStructInstance metadata from Python types.

Walks every DataTypeAction in ``ctx.type_m``, finds actions that have an
``activity()`` method calling sub-actions (via ActivityBodyWalker), and for
each action step inspects its ``Buffer[T]`` fields.  Returns:

  - a list of ``FSMStructDef`` objects (one per unique payload class T)
  - a dict mapping *flat register prefix* (e.g. ``_action_result_dec_out``)
    to the ``FSMStructInstance`` it should become (e.g. ``dec_out`` of type
    ``DecodeResult_t``).

The flat prefix is the root of all registers that currently represent that
buffer: ``_action_result_{var}_{buf_field_name}``.  Individual field regs
follow the pattern ``{flat_prefix}_{payload_field}``.
"""
from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import Dict, List, Optional, Tuple

from .buffer_elab import _buffer_generic_arg, _field_width
from .activity_body_walker import ActivityBodyWalker
from .fsm_ir import FSMStructDef, FSMStructInstance


def _resolve_buffer_type(cls: type, field_name: str) -> Optional[type]:
    """Resolve ``Buffer[T]`` → ``T`` for a named field on a dataclass."""
    ann: Dict = {}
    for klass in reversed(cls.__mro__):
        ann.update(getattr(klass, "__annotations__", {}))
    hint = ann.get(field_name)
    if hint is None:
        return None
    return _buffer_generic_arg(hint)


def _struct_def_for(T: type) -> Optional[FSMStructDef]:
    """Build an FSMStructDef from payload dataclass ``T``, or None on failure."""
    if not dataclasses.is_dataclass(T):
        return None
    try:
        t_hints = typing.get_type_hints(T, include_extras=True)
    except Exception:
        t_hints = {}

    fields: List[Tuple[str, int]] = []
    for f in dataclasses.fields(T):
        hint = t_hints.get(f.name, f.type)

        class _MockField:
            type = hint

        w = _field_width(_MockField())
        fields.append((f.name, w))

    return FSMStructDef(name=f"{T.__name__}_t", fields=fields)


def build_struct_metadata(ctx) -> Tuple[List[FSMStructDef], Dict[str, FSMStructInstance]]:
    """Return ``(struct_defs, flat_to_struct)`` for all Buffer[T] fields found.

    Parameters
    ----------
    ctx:
        DataModelFactory build result (has ``type_m`` dict mapping type names
        to IR nodes).

    Returns
    -------
    struct_defs:
        Unique ``FSMStructDef`` objects (one per payload class T).
    flat_to_struct:
        Maps ``_action_result_{var}_{buf_field}`` → ``FSMStructInstance``.
    """
    try:
        from zuspec.ir.core.data_type import DataTypeAction
    except ImportError:
        return [], {}

    struct_defs: Dict[str, FSMStructDef] = {}   # T.__name__ → FSMStructDef
    flat_to_struct: Dict[str, FSMStructInstance] = {}

    for _type_name, type_ir in ctx.type_m.items():
        if not isinstance(type_ir, DataTypeAction):
            continue
        py_cls = getattr(type_ir, "py_type", None)
        if py_cls is None:
            continue

        act_method = getattr(py_cls, "activity", None)
        if act_method is None:
            continue

        try:
            namespace = vars(inspect.getmodule(py_cls))
            walker = ActivityBodyWalker(act_method, namespace=namespace)
        except Exception:
            continue

        if not walker.steps:
            continue

        for step in walker.steps:
            if not step.var_name or step.action_cls is None:
                continue
            if not dataclasses.is_dataclass(step.action_cls):
                continue

            for f in dataclasses.fields(step.action_cls):
                T = _resolve_buffer_type(step.action_cls, f.name)
                if T is None:
                    continue

                # Ensure we have a struct def for T
                if T.__name__ not in struct_defs:
                    sd = _struct_def_for(T)
                    if sd is None:
                        continue
                    struct_defs[T.__name__] = sd

                flat_prefix = f"_action_result_{step.var_name}_{f.name}"
                instance_name = f"{step.var_name}_{f.name}"
                struct_type = f"{T.__name__}_t"
                flat_to_struct[flat_prefix] = FSMStructInstance(
                    instance_name=instance_name,
                    struct_type=struct_type,
                )

    return list(struct_defs.values()), flat_to_struct
