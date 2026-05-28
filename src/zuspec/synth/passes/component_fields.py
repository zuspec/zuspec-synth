# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""ComponentFieldsPass — classify component fields into ports, state vars, and domain info."""
from __future__ import annotations

import dataclasses as _dc
import logging
from typing import Any

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


def _field_bits(f) -> int:
    """Return the bit width for a field, defaulting to 32 for unbounded int."""
    bits = getattr(getattr(f, "datatype", None), "bits", 1)
    if bits is None or bits < 1:
        return 32
    return bits


def _get_component_ir(ir: SynthIR):
    """Resolve the DataTypeComponent IR for ir.component from ir.model_context."""
    cls = ir.component
    ctx = ir.model_context
    if ctx is None or cls is None:
        return None
    return (
        ctx.type_m.get(getattr(cls, "__qualname__", None))
        or ctx.type_m.get(cls.__name__)
    )


def build_component_fields(component_ir, py_cls, ctx) -> "ComponentFields":
    """Classify component fields from a DataTypeComponent IR.

    This function is the canonical implementation used by both
    ``ComponentFieldsPass`` and the direct-call path in tests.

    Args:
        component_ir: A ``DataTypeComponent`` IR node.
        py_cls:       The Python class being synthesised (may be ``None``).
        ctx:          The ``DataModel`` context from ``DataModelFactory``.

    Returns:
        A fully populated :class:`~zuspec.synth.sprtl.fsm_ir.ComponentFields`.
    """
    from zuspec.synth.sprtl.fsm_ir import (
        ComponentFields, PortDecl, StateVarDecl, DomainBinding,
    )

    fields = list(getattr(component_ir, "fields", []))
    idx_to_name: dict = {i: f.name for i, f in enumerate(fields)}

    # Python class field metadata (for bundle/mirror detection) and defaults
    # (used as fallback reset values when the IR field has no reset_value).
    py_fields_meta: dict = {}
    py_fields_default: dict = {}
    if py_cls is not None and hasattr(py_cls, "__dataclass_fields__"):
        for pf in _dc.fields(py_cls):
            py_fields_meta[pf.name] = pf.metadata
            if pf.default is not _dc.MISSING and pf.default is not None:
                try:
                    py_fields_default[pf.name] = int(pf.default)
                except (TypeError, ValueError):
                    pass

    # Clock / reset domain — computed once for all paths.
    domain_binding = DomainBinding.from_component_ir(component_ir)

    if getattr(component_ir, "clock_domain", None) is not None:
        clock_name = domain_binding.clock_name
    else:
        clock_name = "clk"

    if getattr(component_ir, "reset_domain", None) is not None:
        reset_name = domain_binding.reset_name
        reset_active_low = domain_binding.reset_active_low
        reset_async = domain_binding.reset_async
    else:
        reset_name = "rst_n"
        reset_active_low = True
        reset_async = False

    ports: list = []
    state_vars: list = []
    clock_port_injected = False

    # Inject clock port unless an explicit clock field already exists.
    # All synthesisable components with processes need a clock port; even
    # components without an explicit ClockDomain declaration use 'clk' by default.
    has_clock_field = any(getattr(f, "name", "") == clock_name for f in fields)
    if not has_clock_field:
        ports.append(PortDecl(name=clock_name, direction="input", width=1))
        clock_port_injected = True

    # Similarly inject a reset port unless there is already a field for it.
    has_reset_field = any(getattr(f, "name", "") == reset_name for f in fields)
    if not has_reset_field:
        ports.append(PortDecl(name=reset_name, direction="input", width=1))

    for f in fields:
        is_out = getattr(f, "is_out", None)
        width = _field_bits(f)
        dt = getattr(f, "datatype", None)

        # Bundle / struct port expansion.
        if ctx is not None and dt is not None and type(dt).__name__ == "DataTypeRef":
            struct_ir = ctx.type_m.get(dt.ref_name)
            if struct_ir is not None and type(struct_ir).__name__ == "DataTypeStruct":
                is_mirror = py_fields_meta.get(f.name, {}).get("kind") == "mirror"
                for sf in struct_ir.fields:
                    sf_is_out = bool(getattr(sf, "is_out", True))
                    if is_mirror:
                        sf_is_out = not sf_is_out
                    sf_width = _field_bits(sf)
                    direction = "output" if sf_is_out else "input"
                    ports.append(PortDecl(
                        name=f"{f.name}_{sf.name}",
                        direction=direction,
                        width=sf_width,
                    ))
                continue  # Bundle handled; skip normal classification.

        if is_out is None:
            # Internal state variable.
            rv = getattr(f, "reset_value", None)
            state_vars.append(StateVarDecl(
                name=f.name,
                width=width,
                reset_value=rv,
            ))
        else:
            direction = "output" if is_out else "input"
            ports.append(PortDecl(
                name=f.name,
                direction=direction,
                width=width,
            ))

    # Auto-generated reset clauses from fields with explicit reset_value.
    # Falls back to the Python field's default value (from zdc.field(default=N))
    # when the IR field has no reset_value set.
    reset_clauses: list = []
    for f in fields:
        rv = getattr(f, "reset_value", None)
        if rv is None:
            rv = py_fields_default.get(f.name)
        if rv is not None:
            reset_clauses.append((f.name, int(rv)))

    return ComponentFields(
        ports=ports,
        state_vars=state_vars,
        idx_to_name=idx_to_name,
        clock_name=clock_name,
        reset_name=reset_name,
        reset_active_low=reset_active_low,
        reset_async=reset_async,
        reset_clauses=reset_clauses,
        clock_port_injected=clock_port_injected,
        module_name=getattr(py_cls, "__name__", "") if py_cls is not None else "",
    )


class ComponentFieldsPass(SynthPass):
    """Classify every component field into ports, state vars, and domain info.

    Reads:
        ir.component: The Python class being synthesised.
        ir.model_context: DataModel context from DataModelFactory.

    Populates:
        ir.component_fields: A fully populated ComponentFields instance.

    Raises:
        ValueError: If the component IR cannot be found in model_context.
    """

    @property
    def name(self) -> str:
        return "component_fields"

    def run(self, ir: SynthIR) -> SynthIR:
        component_ir = _get_component_ir(ir)
        if component_ir is None:
            raise ValueError(
                f"ComponentFieldsPass: could not find IR for "
                f"{getattr(ir.component, '__name__', ir.component)!r}."
            )

        ir.component_fields = build_component_fields(
            component_ir, ir.component, ir.model_context
        )
        _log.debug(
            "[ComponentFieldsPass] %d ports, %d state_vars",
            len(ir.component_fields.ports),
            len(ir.component_fields.state_vars),
        )
        return ir
