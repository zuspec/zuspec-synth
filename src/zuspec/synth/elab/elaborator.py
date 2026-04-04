"""
Elaborator: Component class + config → ComponentSynthMeta.
"""
from __future__ import annotations
import dataclasses as dc
import typing
from typing import Any, Dict, Optional, Type, get_args, get_origin

from .elab_ir import ComponentSynthMeta, ResourcePoolDecl, ArbiterDecl, InstanceDecl, PortDecl, MethodPortDecl, RegFileDeclIR, IndexedPoolDeclIR


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


def _expand_callable_port(
    channel_name: str,
    callable_hint,
    is_export: bool,
) -> tuple:
    """Expand a Callable[[...], Awaitable[...]] annotation into a MethodPortDecl
    and a list of flat PortDecls.

    Returns ``(MethodPortDecl, [PortDecl, ...])``.

    For a *port* (``is_export=False``) the component is the **requester**:
      - req data/valid are outputs, req ready is input
      - resp data/valid are inputs, resp ready is output

    For an *export* (``is_export=True``) the component is the **provider**:
      all directions are reversed.
    """
    import inspect
    import collections.abc

    args = get_args(callable_hint)
    # args = ([param_types...], return_type)
    param_types = args[0] if args else []
    ret_type = args[1] if len(args) > 1 else None

    # Get resp width from Awaitable[T] return
    resp_width = 0
    if ret_type is not None:
        ret_args = get_args(ret_type)
        inner = ret_args[0] if ret_args else None
        if inner is not None and inner is not type(None):
            resp_width = _bit_width(inner)

    # Build req_fields — one entry per positional param
    req_fields = []
    for i, pt in enumerate(param_types):
        param_name = f"arg{i}"
        req_fields.append((param_name, _bit_width(pt)))

    from .elab_ir import MethodPortDecl
    mdecl = MethodPortDecl(
        name=channel_name,
        req_fields=req_fields,
        resp_width=resp_width,
        is_export=is_export,
    )

    # Build flat PortDecls — direction depends on is_export
    # port():   req→output, req_ready→input, resp→input, resp_ready→output
    # export(): req→input,  req_ready→output, resp→output, resp_ready→input
    req_data_dir  = 'input'  if is_export else 'output'
    req_ready_dir = 'output' if is_export else 'input'
    resp_data_dir = 'output' if is_export else 'input'
    resp_ready_dir= 'input'  if is_export else 'output'

    port_decls = []
    for pname, pwidth in req_fields:
        port_decls.append(PortDecl(
            name=f'{channel_name}_req_{pname}',
            direction=req_data_dir,
            width=pwidth,
            bundle=channel_name,
        ))
    port_decls.append(PortDecl(name=f'{channel_name}_req_valid',  direction=req_data_dir,  width=1, bundle=channel_name))
    port_decls.append(PortDecl(name=f'{channel_name}_req_ready',  direction=req_ready_dir, width=1, bundle=channel_name))
    if resp_width > 0:
        port_decls.append(PortDecl(name=f'{channel_name}_resp_data',  direction=resp_data_dir,  width=resp_width, bundle=channel_name))
    port_decls.append(PortDecl(name=f'{channel_name}_resp_valid', direction=resp_data_dir,  width=1, bundle=channel_name))
    port_decls.append(PortDecl(name=f'{channel_name}_resp_ready', direction=resp_ready_dir, width=1, bundle=channel_name))

    return mdecl, port_decls


def _expand_protocol_port(
    field_name: str,
    protocol_type,
    is_export: bool,
) -> tuple:
    """Expand a Protocol subclass port into MethodPortDecls + flat PortDecls.

    Each async method in the Protocol becomes one channel pair.
    Method names are sorted for deterministic port ordering.

    Returns ``([MethodPortDecl, ...], [PortDecl, ...])``.
    """
    import inspect
    import collections.abc
    from typing import get_type_hints

    all_method_decls = []
    all_port_decls = []

    method_names = sorted(getattr(protocol_type, '__protocol_attrs__', set()))
    for mname in method_names:
        method = getattr(protocol_type, mname, None)
        if method is None:
            continue
        # Build a Callable hint from the method's signature
        try:
            hints = get_type_hints(method, include_extras=True)
            sig = inspect.signature(method)
            param_types = []
            for pname, p in sig.parameters.items():
                if pname == 'self':
                    continue
                param_types.append(hints.get(pname))
            ret = hints.get('return')
        except Exception:
            continue

        # Reconstruct as Callable[[params...], ret] for reuse of _expand_callable_port
        from typing import Callable, Awaitable
        callable_hint = Callable[param_types, ret if ret is not None else type(None)]

        channel_name = f'{field_name}_{mname}'
        mdecl, pdecls = _expand_callable_port(channel_name, callable_hint, is_export)
        # Override req_fields with named params (not arg0, arg1, ...)
        named_req = []
        for i, (pname, _) in enumerate(zip(
            [p for p in inspect.signature(method).parameters if p != 'self'],
            mdecl.req_fields,
        )):
            named_req.append((pname, mdecl.req_fields[i][1]))
            # Fix the PortDecl name to use the real param name
            pdecls[i] = PortDecl(
                name=f'{channel_name}_req_{pname}',
                direction=pdecls[i].direction,
                width=pdecls[i].width,
                bundle=pdecls[i].bundle,
            )
        mdecl = MethodPortDecl(
            name=mdecl.name,
            req_fields=named_req,
            resp_width=mdecl.resp_width,
            is_export=mdecl.is_export,
        )

        all_method_decls.append(mdecl)
        all_port_decls.extend(pdecls)

    return all_method_decls, all_port_decls


class Elaborator:
    """Walks a @zdc.dataclass Component class and produces ComponentSynthMeta."""

    def elaborate(self, component_cls: type, config: Any = None) -> ComponentSynthMeta:
        """Elaborate a component class into ComponentSynthMeta.

        Args:
            component_cls: The @zdc.dataclass Component subclass (e.g. MyProcessor).
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
        method_port_decls: list = []
        regfiles: list[RegFileDeclIR] = []
        indexed_pools: list[IndexedPoolDeclIR] = []

        pipeline_width = config_dict.get('pipeline_width', 1)

        import collections.abc

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

            elif kind in ('port', 'export'):
                is_export = kind == 'export'
                hint_origin = get_origin(hint)
                if hint_origin is collections.abc.Callable:
                    # Callable port/export — req/resp channel pair
                    mdecl, pdecls = _expand_callable_port(f.name, hint, is_export)
                    method_port_decls.append(mdecl)
                    port_decls.extend(pdecls)
                elif hint is not None and getattr(hint, '_is_protocol', False):
                    # Protocol port/export — one channel pair per method
                    mdecls, pdecls = _expand_protocol_port(f.name, hint, is_export)
                    method_port_decls.extend(mdecls)
                    port_decls.extend(pdecls)
                elif isinstance(hint, type):
                    # Bundle subclass — existing flat expansion (port only)
                    if not is_export:
                        port_decls.extend(_expand_bundle_ports(f.name, hint))

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
            method_ports=method_port_decls,
            regfiles=regfiles,
            indexed_pools=indexed_pools,
        )
