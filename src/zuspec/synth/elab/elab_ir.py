"""
Synthesis elaboration IR: lightweight metadata alongside the zdc DataTypeComponent.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class PortDecl:
    """A flat RTL port signal expanded from a Component port bundle field.

    ``direction`` is from the component's perspective:
    ``'output'`` means the component drives the signal (top-level output);
    ``'input'``  means the component receives the signal (top-level input).

    ``bundle`` is the field name on the component (e.g. ``'icache'``).
    ``name``   is the flattened RTL port name (e.g. ``'icache_addr'``).
    """
    name:      str    # flattened RTL signal name, e.g. "icache_addr"
    direction: str    # "input" or "output" (from component perspective)
    width:     int    # bit width (1 = single bit)
    bundle:    str    # source bundle field name, e.g. "icache"


@dataclass
class ResourcePoolDecl:
    """A ClaimPool field on a Component."""
    resource_type: type        # e.g. ALUUnit (type param of ClaimPool[T])
    capacity: int              # 1 = scalar, N = dual/multi-issue
    pool_field_name: str       # e.g. "alu_pool"


@dataclass
class ArbiterDecl:
    """One round-robin arbiter synthesized per ResourcePoolDecl."""
    name: str                  # e.g. "alu_arbiter"
    pool: ResourcePoolDecl


@dataclass
class InstanceDecl:
    """A sub-component instance (kind='instance' field)."""
    name: str                  # e.g. "alu"
    comp_type: type            # e.g. ALUUnit
    is_present: bool = True    # False when pruned by config


@dataclass
class ComponentSynthMeta:
    """Synthesis-specific metadata for a Component class."""
    component_class: type                       # e.g. RVCore
    config: Dict[str, Any]                      # resolved const() fields
    instances: List[InstanceDecl] = field(default_factory=list)
    resource_pools: List[ResourcePoolDecl] = field(default_factory=list)
    arbiters: List[ArbiterDecl] = field(default_factory=list)
    actions: List[type] = field(default_factory=list)  # Action subclasses in scope
    ports: List[PortDecl] = field(default_factory=list)  # flat RTL port declarations
