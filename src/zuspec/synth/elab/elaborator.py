"""
Elaborator: Component class + config → ComponentSynthMeta.
"""
from __future__ import annotations
import dataclasses as dc
import typing
from typing import Any, Dict, Optional, Type, get_args, get_origin

from .elab_ir import ComponentSynthMeta, ResourcePoolDecl, ArbiterDecl, InstanceDecl, PortDecl, RegFileDeclIR, IndexedPoolDeclIR


def _get_pool_element_type(hint) -> Optional[type]:
    """Extract T from ClaimPool[T] or similar generic."""
    args = get_args(hint)
    if args:
        return args[0]
    return None


def _bit_width(type_hint) -> int:
    """Extract the integer bit width from a zdc.uN / zdc.iN annotated type.

    Handles:
    * ``Annotated[int, U(N)]``  → N
    * ``Annotated[int, S(N)]``  → N
    * Bare ``int`` or unknown   → 32 (safe default)
    """
    if type_hint is None:
        return 32
    args = get_args(type_hint)
    for a in args:
        w = getattr(a, 'width', None)
        if w is not None:
            return int(w)
    return 32


def _expand_bundle_ports(bundle_field_name: str, bundle_type: type) -> list[PortDecl]:
    """Expand a zdc.Bundle subclass into a flat list of PortDecl.

    Each field of the bundle contributes one PortDecl.  Direction is read from
    the field's ``default_factory``:
    - ``Output`` (or subclass with 'Output' in name) → direction='output'
    - ``Input``  (or subclass with 'Input' in name)  → direction='input'

    Bit width is read from the ``U.width`` (or ``S.width``) annotation on the
    field's type hint (via ``typing.get_args``); falls back to 1.
    """
    try:
        fields = dc.fields(bundle_type)
    except TypeError:
        return []

    try:
        hints = typing.get_type_hints(bundle_type, include_extras=True)
    except Exception:
        hints = {}

    ports: list[PortDecl] = []
    for f in fields:
        # Determine direction from default_factory class name
        factory = f.default_factory  # type: ignore[misc]
        if factory is dc.MISSING:
            continue
        fname = getattr(factory, '__name__', '')
        if 'Output' in fname:
            direction = 'output'
        elif 'Input' in fname:
            direction = 'input'
        else:
            continue  # unknown direction — skip

        # Determine bit width from annotated type
        width = 1
        hint = hints.get(f.name)
        if hint is not None:
            args = typing.get_args(hint)
            if len(args) >= 2:
                ann = args[1]
                w = getattr(ann, 'width', None)
                if w is not None:
                    width = int(w)

        ports.append(PortDecl(
            name=f'{bundle_field_name}_{f.name}',
            direction=direction,
            width=width,
            bundle=bundle_field_name,
        ))
    return ports


class Elaborator:
    """Walks a @zdc.dataclass Component class and produces ComponentSynthMeta."""

    def elaborate(self, component_cls: type, config: Any = None) -> ComponentSynthMeta:
        """Elaborate a component class into ComponentSynthMeta.

        Args:
            component_cls: The @zdc.dataclass Component subclass (e.g. RVCore).
            config: Optional config object (e.g. RVConfig instance) or dict.
                    If an object, its __dict__ is used.
        """
        # Resolve config to a plain dict
        if config is None:
            config_dict: Dict[str, Any] = {}
        elif isinstance(config, dict):
            config_dict = config
        else:
            # Object with attributes (e.g. RVConfig dataclass instance)
            try:
                config_dict = {
                    f.name: getattr(config, f.name)
                    for f in dc.fields(config)
                }
            except TypeError:
                config_dict = vars(config) if hasattr(config, '__dict__') else {}

        # Get type hints for the component class
        try:
            hints = typing.get_type_hints(component_cls, include_extras=True)
        except Exception:
            hints = {}

        instances: list[InstanceDecl] = []
        pools: list[ResourcePoolDecl] = []
        arbiters: list[ArbiterDecl] = []
        port_decls: list[PortDecl] = []
        regfiles: list[RegFileDeclIR] = []
        indexed_pools: list[IndexedPoolDeclIR] = []

        pipeline_width = config_dict.get('pipeline_width', 1)

        for f in dc.fields(component_cls):
            kind = f.metadata.get('kind')
            hint = hints.get(f.name)

            if kind == 'instance':
                # Sub-component instance
                comp_type = hint if isinstance(hint, type) else None
                instances.append(InstanceDecl(
                    name=f.name,
                    comp_type=comp_type,
                    is_present=True,  # Phase 1: always present; Phase 2 prunes
                ))

            elif kind == 'pool':
                # ClaimPool[T] field — synthesize a resource pool + arbiter
                elem_type = _get_pool_element_type(hint) if hint is not None else None
                pool = ResourcePoolDecl(
                    resource_type=elem_type,
                    capacity=pipeline_width,
                    pool_field_name=f.name,
                )
                pools.append(pool)
                arbiters.append(ArbiterDecl(
                    name=f"{f.name[:-5]}_arbiter" if f.name.endswith('_pool') else f"{f.name}_arbiter",
                    pool=pool,
                ))

            elif kind == 'port':
                # Bundle port field — expand into flat PortDecl list
                bundle_type = hint if isinstance(hint, type) else None
                if bundle_type is not None:
                    port_decls.extend(_expand_bundle_ports(f.name, bundle_type))

            elif kind == 'indexed_regfile':
                # IndexedRegFile[TIdx, TData] field
                args = get_args(hint) if hint is not None else ()
                idx_type  = args[0] if len(args) > 0 else None
                data_type = args[1] if len(args) > 1 else None
                idx_width  = _bit_width(idx_type)
                data_width = _bit_width(data_type)
                # Scale data_width to xlen: a regfile declared as u32 should be
                # 64 bits wide when synthesizing a 64-bit core.
                xlen = config_dict.get('xlen', 32)
                if xlen > data_width:
                    data_width = xlen
                regfiles.append(RegFileDeclIR(
                    field_name  = f.name,
                    depth       = 2 ** idx_width,
                    idx_width   = idx_width,
                    data_width  = data_width,
                    read_ports  = f.metadata.get('read_ports',  2),
                    write_ports = f.metadata.get('write_ports', 1),
                    shared_port = f.metadata.get('shared_port', False),
                ))

            elif kind == 'indexed_pool':
                # IndexedPool[TIdx] field — scoreboard / data-hazard tracker
                args      = get_args(hint) if hint is not None else ()
                idx_type  = args[0] if len(args) > 0 else None
                idx_width = _bit_width(idx_type)
                indexed_pools.append(IndexedPoolDeclIR(
                    field_name = f.name,
                    depth      = f.metadata.get('depth',    2 ** idx_width),
                    idx_width  = idx_width,
                    noop_idx   = f.metadata.get('noop_idx', None),
                ))

        return ComponentSynthMeta(
            component_class=component_cls,
            config=config_dict,
            instances=instances,
            resource_pools=pools,
            arbiters=arbiters,
            ports=port_decls,
            regfiles=regfiles,
            indexed_pools=indexed_pools,
        )
