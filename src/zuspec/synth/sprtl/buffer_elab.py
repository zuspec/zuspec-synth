"""Buffer elaboration: flatten ``Buffer[T]`` fields to logic wire declarations.

For a single-cycle CPU (all stages run in one clock cycle), inter-stage
buffers are *combinational wires* — no pipeline registers are needed.  This
pass introspects each stage's output buffer type ``T`` and emits a
``logic [W-1:0] <prefix>_<field>;`` declaration for every field of ``T``.

Usage::

    from zuspec.synth.sprtl.buffer_elab import BufferElaborationPass
    from zuspec.synth.sprtl.activity_body_walker import ActivityBodyWalker

    walker = ActivityBodyWalker(RV32ISingleCycleA.activity, namespace)
    elab = BufferElaborationPass(walker)
    wire_lines = elab.emit_wire_declarations()
"""
from __future__ import annotations

import dataclasses
import typing
from typing import List, Optional, Tuple

from .activity_body_walker import ActivityBodyWalker, ActionStep


def _field_width(field: dataclasses.Field) -> int:  # type: ignore[type-arg]
    """Extract bit width from a dataclass field's type annotation.

    Handles:
    - ``typing.Annotated[int, U(width=N, signed=…)]`` → N
    - ``bvN`` class (e.g. ``bv32._width`` attribute) → N
    - Fallback → 32
    """
    ftype = field.type
    # Annotated form: Annotated[int, U(width=N)]
    if hasattr(ftype, '__metadata__') and ftype.__metadata__:
        meta = ftype.__metadata__[0]
        w = getattr(meta, 'width', None)
        if isinstance(w, int):
            return w
    # bvN class with _width attribute (e.g. bv32._width == 32)
    if isinstance(ftype, type) and hasattr(ftype, '_width'):
        w = ftype._width
        if isinstance(w, int):
            return w
    return 32  # safe default for u32


def _buffer_generic_arg(hint) -> Optional[type]:
    """Extract ``T`` from ``zdc.Buffer[T]`` type hint.

    Returns ``None`` if the hint is not a Buffer generic.
    """
    # hint is something like Buffer[FetchResult]; __args__[0] is FetchResult.
    args = getattr(hint, '__args__', None)
    if args:
        return args[0]
    return None


class BufferElaborationPass:
    """Emit wire declarations for all inter-stage buffer signals.

    Parameters
    ----------
    walker:
        A completed ``ActivityBodyWalker`` instance.
    stage_var_to_prefix:
        Optional override mapping stage variable names (e.g. ``'fetch'``) to
        signal prefixes (e.g. ``'fetch'``).  If None, the variable name is
        used as the prefix.
    """

    def __init__(
        self,
        walker: ActivityBodyWalker,
        stage_var_to_prefix: Optional[dict] = None,
    ):
        self._walker = walker
        self._prefix_map = stage_var_to_prefix or {}

    def _prefix_for(self, step: ActionStep) -> str:
        return self._prefix_map.get(step.var_name, step.var_name)

    def emit_wire_declarations(self) -> List[str]:
        """Return ``logic`` declaration lines for all inter-stage wires.

        Each output buffer field of each stage becomes one or more wire lines.
        For a single-cycle CPU, every wire is a plain ``logic`` (combinational).
        """
        lines: List[str] = []
        for step in self._walker.steps:
            cls = step.action_cls
            if cls is None:
                continue
            prefix = self._prefix_for(step)
            wires = self._wires_for_output(cls, prefix)
            if wires:
                lines.append(f'// {cls.__name__} output wires')
                lines.extend(wires)
        return lines

    def _wires_for_output(self, cls: type, prefix: str) -> List[str]:
        """Get wire declarations for the output buffer fields of *cls*."""
        if not dataclasses.is_dataclass(cls):
            return []

        try:
            from zuspec.dataclasses.decorators import Output as _OutputMarker
        except ImportError:
            _OutputMarker = None

        lines: List[str] = []
        for f in dataclasses.fields(cls):
            is_output = (
                (_OutputMarker is not None and f.default_factory is _OutputMarker)
                or f.metadata.get('output')
                or f.metadata.get('zdc_field_kind') == 'output'
            )
            if is_output:
                T = self._resolve_buffer_type(cls, f)
                if T is not None:
                    lines.extend(self._wires_for_type(T, prefix))
        return lines

    def _resolve_buffer_type(self, cls: type, field: dataclasses.Field) -> Optional[type]:  # type: ignore[type-arg]
        """Resolve ``Buffer[T]`` → ``T`` from class annotations."""
        import typing as _typing
        ann: dict = {}
        for klass in reversed(cls.__mro__):
            ann.update(getattr(klass, '__annotations__', {}))
        hint = ann.get(field.name)
        if hint is None:
            return None
        # Unwrap typing.get_type_hints raises on generics, use __args__ directly.
        return _buffer_generic_arg(hint)

    def _wires_for_type(self, T: type, prefix: str) -> List[str]:
        """Return ``logic`` lines for every field of dataclass *T*."""
        if not dataclasses.is_dataclass(T):
            return []
        lines: List[str] = []
        for f in dataclasses.fields(T):
            w = _field_width(f)
            if w == 1:
                lines.append(f'logic {prefix}_{f.name};')
            else:
                lines.append(f'logic [{w-1}:0] {prefix}_{f.name};')
        return lines

    def build_signal_map_for_stage(self, step: ActionStep) -> dict:
        """Build the ``signal_map`` dict for ``FunctionalConstraintCompiler``.

        Maps ``<buf_field>.t.<payload_field>`` → ``<stage_prefix>_<field>``.
        Also maps output buffer fields as ``out.t.<field>`` → ``<prefix>_<field>``.
        """
        cls = step.action_cls
        if cls is None or not dataclasses.is_dataclass(cls):
            return {}

        try:
            from zuspec.dataclasses.decorators import Input as _InputMarker, Output as _OutputMarker
        except ImportError:
            _InputMarker = _OutputMarker = None

        prefix = self._prefix_for(step)
        sig_map: dict = {}

        for f in dataclasses.fields(cls):
            meta = f.metadata
            is_output = (
                (_OutputMarker is not None and f.default_factory is _OutputMarker)
                or meta.get('output')
                or meta.get('zdc_field_kind') == 'output'
            )
            is_input = (
                (_InputMarker is not None and f.default_factory is _InputMarker)
                or meta.get('input')
                or meta.get('zdc_field_kind') == 'input'
            )

            if is_output:
                T = self._resolve_buffer_type(cls, f)
                if T is not None and dataclasses.is_dataclass(T):
                    for tf in dataclasses.fields(T):
                        sig_map[f'out.t.{tf.name}'] = f'{prefix}_{tf.name}'
                        sig_map[f'{f.name}.t.{tf.name}'] = f'{prefix}_{tf.name}'

            elif is_input:
                # The field name matches the buffer-producing stage var name.
                T = self._resolve_buffer_type(cls, f)
                src_prefix = f.name  # e.g. 'fetch', 'dec', 'exe', 'mem'
                if T is not None and dataclasses.is_dataclass(T):
                    for tf in dataclasses.fields(T):
                        sig_map[f'{f.name}.t.{tf.name}'] = f'{src_prefix}_{tf.name}'

        return sig_map
