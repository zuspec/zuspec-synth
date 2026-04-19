"""protocol_ir.py — Synthesis IR nodes for IfProtocol, Queue, Completion, Spawn, and Select.

These nodes are produced by the lowering passes (Phase 5) and consumed by the
SV code-generation layer (Phase 6) and the SV emitter.  They are stored in
new fields on :class:`~zuspec.synth.ir.synth_ir.SynthIR`.
"""
from __future__ import annotations

import dataclasses as dc
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scenario enum
# ---------------------------------------------------------------------------

class IfProtocolScenario(Enum):
    """Protocol synthesis scenario, in increasing complexity order."""
    A = "A"  # max_outstanding=1, req_always_ready=True, resp_always_valid=True
    B = "B"  # max_outstanding=1 (default handshake)
    C = "C"  # max_outstanding=N, in_order=True (in-order FIFO response)
    D = "D"  # max_outstanding=N, in_order=False (out-of-order, ROB)


# ---------------------------------------------------------------------------
# Request/response field descriptors
# ---------------------------------------------------------------------------

@dc.dataclass
class ProtocolField:
    """One data field on the request or response channel of an IfProtocol port.

    Attributes:
        name: Field name (e.g. ``"addr"``, ``"data"``, ``"be"``).
        width: Bit width.
        is_response: ``True`` → response channel; ``False`` → request channel.
    """
    name: str
    width: int = 32
    is_response: bool = False


# ---------------------------------------------------------------------------
# IfProtocolPortIR — synthesis IR for one IfProtocol port or export
# ---------------------------------------------------------------------------

@dc.dataclass
class IfProtocolPortIR:
    """Synthesis IR node representing a single IfProtocol port or export.

    Produced by :class:`~zuspec.synth.passes.if_protocol_lower.IfProtocolLowerPass`.
    Consumed by the SV code generator to emit port declarations and FSM states.

    Attributes:
        name:           Port field name in the component (e.g. ``"mem"``).
        is_export:      ``True`` if this is an export (provider side).
        scenario:       Which synthesis scenario applies.
        properties:     ``IfProtocolProperties`` from the DSL.
        req_fields:     Data fields on the request channel.
        resp_fields:    Data fields on the response channel.
        protocol_cls:   The original ``IfProtocol`` subclass (for type info).
        id_bits:        Bit width of the ID field (0 for scenarios A/B).
    """
    name: str
    is_export: bool = False
    scenario: IfProtocolScenario = IfProtocolScenario.B
    properties: Optional[Any] = None   # IfProtocolProperties
    req_fields: List[ProtocolField] = dc.field(default_factory=list)
    resp_fields: List[ProtocolField] = dc.field(default_factory=list)
    protocol_cls: Optional[Any] = None
    id_bits: int = 0

    def signal_name(self, suffix: str) -> str:
        """Return the full SV signal name for a port signal."""
        return f"{self.name}_{suffix}"

    def all_sv_ports(self) -> List[tuple]:
        """Return ``(direction, width, signal_name)`` triples for all SV ports.

        Direction is from the component's perspective (``"output"`` / ``"input"``).
        For an export the directions are flipped relative to a port.
        """
        flip = self.is_export
        def _dir(out_is_output: bool) -> str:
            if out_is_output ^ flip:
                return "output"
            return "input"

        ports = []
        props = self.properties

        req_always_ready = getattr(props, "req_always_ready", False) if props else False
        resp_always_valid = getattr(props, "resp_always_valid", False) if props else False
        resp_has_backpressure = getattr(props, "resp_has_backpressure", False) if props else False

        if self.scenario != IfProtocolScenario.A or not req_always_ready:
            ports.append((_dir(True),  1, self.signal_name("req_valid")))
            if not req_always_ready:
                ports.append((_dir(False), 1, self.signal_name("req_ready")))

        for f in self.req_fields:
            ports.append((_dir(True), f.width, self.signal_name(f"req_{f.name}")))

        if self.id_bits > 0:
            ports.append((_dir(True), self.id_bits, self.signal_name("req_id")))

        if not resp_always_valid:
            ports.append((_dir(False), 1, self.signal_name("resp_valid")))
        if resp_has_backpressure:
            ports.append((_dir(True), 1, self.signal_name("resp_ready")))

        for f in self.resp_fields:
            ports.append((_dir(False), f.width, self.signal_name(f"resp_{f.name}")))

        if self.id_bits > 0:
            ports.append((_dir(False), self.id_bits, self.signal_name("resp_id")))

        return ports


# ---------------------------------------------------------------------------
# QueueIR — synthesis IR node for a zdc.Queue[T] field
# ---------------------------------------------------------------------------

@dc.dataclass
class QueueIR:
    """Synthesis IR for a zdc.Queue[T] field.

    Produced by :class:`~zuspec.synth.passes.queue_lower.QueueLowerPass`.
    Emitted as a synchronous FIFO module or inline FIFO logic by the SV backend.

    Attributes:
        name:        Field name (e.g. ``"requests"``).
        elem_width:  Bit width of one element.
        depth:       FIFO depth (number of entries).
    """
    name: str
    elem_width: int = 32
    depth: int = 16

    @property
    def addr_bits(self) -> int:
        """Number of bits needed to address the FIFO entries."""
        return max(1, (self.depth - 1).bit_length())

    @property
    def count_bits(self) -> int:
        """Number of bits for the count register (can hold 0…depth)."""
        return (self.depth).bit_length()


# ---------------------------------------------------------------------------
# SpawnIR — synthesis IR node for a zdc.spawn() call
# ---------------------------------------------------------------------------

@dc.dataclass
class SpawnSlotIR:
    """One slot in a spawn slot array (analogous to a pipeline slot or ROB entry)."""
    slot_fields: List[ProtocolField] = dc.field(default_factory=list)
    result_fields: List[ProtocolField] = dc.field(default_factory=list)


@dc.dataclass
class SpawnIR:
    """Synthesis IR for a zdc.spawn() call.

    Attributes:
        name:           Spawn site name (derived from spawned function name).
        n_slots:        Number of parallel slots (= max_outstanding of called port).
        slot_fields:    Fields stored per slot (from spawned coroutine arg struct).
        result_fields:  Fields of the response result captured per slot.
        protocol_port:  Name of the IfProtocol port called inside the spawned coro.
    """
    name: str
    n_slots: int = 1
    slot_fields: List[ProtocolField] = dc.field(default_factory=list)
    result_fields: List[ProtocolField] = dc.field(default_factory=list)
    protocol_port: Optional[str] = None


# ---------------------------------------------------------------------------
# SelectIR — synthesis IR node for a zdc.select() call
# ---------------------------------------------------------------------------

@dc.dataclass
class SelectBranchIR:
    """One branch of a select: a queue name and tag value."""
    queue_name: str
    tag_value: int = 0


@dc.dataclass
class SelectIR:
    """Synthesis IR for a zdc.select() call.

    Attributes:
        name:       Select site name.
        branches:   Queue + tag pairs.
        round_robin: If True, emit a round-robin arbiter; otherwise priority.
    """
    name: str
    branches: List[SelectBranchIR] = dc.field(default_factory=list)
    round_robin: bool = False


# ---------------------------------------------------------------------------
# CompletionIR — synthesis IR for a Completion token analysis result
# ---------------------------------------------------------------------------

@dc.dataclass
class CompletionIR:
    """Records a statically-analyzed Completion token.

    Produced by :class:`~zuspec.synth.passes.completion_analysis.CompletionAnalysisPass`.

    Attributes:
        name:           Variable name of the Completion token.
        elem_width:     Bit width of the payload type T.
        set_location:   Human-readable source location of ``done.set(value)``.
        await_location: Human-readable source location of ``await done``.
        queue_path:     Chain of queue field names the token travels through.
    """
    name: str
    elem_width: int = 32
    set_location: str = ""
    await_location: str = ""
    queue_path: List[str] = dc.field(default_factory=list)
