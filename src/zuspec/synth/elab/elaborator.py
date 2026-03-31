"""
Elaborator: Component class + config → ComponentSynthMeta.
"""
from __future__ import annotations
import dataclasses as dc
import typing
from typing import Any, Dict, Optional, Type, get_args, get_origin

from .elab_ir import ComponentSynthMeta, ResourcePoolDecl, ArbiterDecl, InstanceDecl


def _get_pool_element_type(hint) -> Optional[type]:
    """Extract T from ClaimPool[T] or similar generic."""
    args = get_args(hint)
    if args:
        return args[0]
    return None


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

        return ComponentSynthMeta(
            component_class=component_cls,
            config=config_dict,
            instances=instances,
            resource_pools=pools,
            arbiters=arbiters,
        )
