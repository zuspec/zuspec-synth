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
class MethodPortDecl:
    """One req/resp ready-valid channel pair from a Callable or Protocol method port.

    A ``Callable[[u32], Awaitable[u32]]`` port produces one ``MethodPortDecl``.
    A ``Protocol`` port produces one ``MethodPortDecl`` per method.

    ``name``       — flattened channel name, e.g. ``'icache'`` or ``'dcache_load'``.
    ``req_fields`` — list of ``(param_name, bit_width)`` for each request parameter.
    ``resp_width`` — response bit width; 0 means void return (no ``resp_data`` port).
    ``is_export``  — True when this is an ``export()`` field (provider side): all
                     signal directions are reversed relative to a ``port()`` field.
    """
    name:       str
    req_fields: List[tuple]   # [(param_name: str, bit_width: int), ...]
    resp_width: int           # 0 = void return
    is_export:  bool = False



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
    component_class: type                       # e.g. MyProcessor
    config: Dict[str, Any]                      # resolved const() fields
    instances: List[InstanceDecl] = field(default_factory=list)
    resource_pools: List[ResourcePoolDecl] = field(default_factory=list)
    arbiters: List[ArbiterDecl] = field(default_factory=list)
    actions: List[type] = field(default_factory=list)  # Action subclasses in scope
    ports: List[PortDecl] = field(default_factory=list)  # flat RTL port declarations
    method_ports: List[MethodPortDecl] = field(default_factory=list)  # callable/protocol ports
    regfiles: List['RegFileDeclIR'] = field(default_factory=list)
    indexed_pools: List['IndexedPoolDeclIR'] = field(default_factory=list)


@dataclass
class RegFileDeclIR:
    """An IndexedRegFile field on a Component, resolved for synthesis.

    Produced by the Elaborator when it encounters a field with
    ``kind='indexed_regfile'`` metadata.

    Attributes
    ----------
    field_name:
        The attribute name on the Component class (e.g. ``'regfile'``).
    depth:
        Total number of registers (e.g. 32 for RV32/RV64).
    idx_width:
        Address bus width in bits: ``ceil(log2(depth))`` (e.g. 5 for depth=32).
    data_width:
        Data bus width in bits (e.g. 32 for RV32, 64 for RV64).
    read_ports:
        Number of independent read port wire groups to generate.
    write_ports:
        Number of independent write port wire groups to generate.
    shared_port:
        ``True``  → single shared address/data bus; reads and writes cannot
                    overlap; maps to a true single-port BRAM primitive.
        ``False`` → separate read and write buses; reads and writes can
                    proceed concurrently; RAW forwarding muxes are generated
                    inside the register file module.
    """
    field_name:  str
    depth:       int
    idx_width:   int
    data_width:  int
    read_ports:  int
    write_ports: int
    shared_port: bool


@dataclass
class IndexedPoolDeclIR:
    """An IndexedPool field on a Component, resolved for synthesis.

    Produced by the Elaborator when it encounters a field with
    ``kind='indexed_pool'`` metadata.

    Attributes
    ----------
    field_name:
        The attribute name on the Component class (e.g. ``'rd_sched'``).
    depth:
        Total number of addressable slots.
    idx_width:
        Address width in bits (e.g. 5 for depth=32).
    noop_idx:
        Optional index value that is a structural no-op (e.g. 0 for x0).
        MLS omits hazard comparators involving this index.
    """
    field_name: str
    depth:      int
    idx_width:  int
    noop_idx:   int | None
