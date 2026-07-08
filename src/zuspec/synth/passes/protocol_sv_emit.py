"""ProtocolSVEmitPass — emit SV fragments for IfProtocol, Queue, Select, Spawn nodes.

This pass reads ``SynthIR.protocol_ports``, ``queue_nodes``, ``select_nodes``,
and ``spawn_nodes`` (populated by the Phase 5 lowering passes) and writes SV
source fragments into ``SynthIR.lowered_sv`` under the following key scheme:

============================================  ================================
Key pattern                                   Content
============================================  ================================
``sv/port/{name}``                            IfProtocol port declaration lines
``sv/module/{name}_fifo``                     Synchronous FIFO module
``sv/module/{name}_arb``                      Priority or round-robin arbiter
``sv/instantiation/{name}``                   Port instantiation connection snippet
============================================  ================================

The pass is idempotent — if a key already exists in ``lowered_sv``, it is
overwritten with freshly generated content.
"""
from __future__ import annotations

import logging
from typing import Optional

from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR
from zuspec.synth.ir.protocol_ir import IfProtocolScenario
from zuspec.synth.passes.synth_pass import SynthPass
from zuspec.synth.sprtl.protocol_sv import (
    generate_ifprotocol_port_decls,
    generate_fifo_sv,
    generate_port_instantiation,
    generate_priority_arbiter_sv,
    generate_rr_arbiter_sv,
)

_log = logging.getLogger(__name__)


def _sv_key(*parts: str) -> str:
    """Build a lowered_sv key."""
    return "/".join(parts)


class ProtocolSVEmitPass(SynthPass):
    """Emit SV fragments for all Phase 5 IR nodes into ``SynthIR.lowered_sv``."""

    def __init__(self, config: SynthConfig = None, module_prefix: str = "") -> None:
        super().__init__(config or SynthConfig())
        self._prefix = module_prefix

    @property
    def name(self) -> str:
        return "protocol_sv_emit"

    def run(self, ir: SynthIR) -> SynthIR:
        self._emit_ifprotocol_ports(ir)
        self._emit_queues(ir)
        self._emit_arbiters(ir)
        return ir

    # ------------------------------------------------------------------
    # IfProtocol ports
    # ------------------------------------------------------------------

    def _emit_ifprotocol_ports(self, ir: SynthIR) -> None:
        for port in ir.protocol_ports:
            decls = self._besv_port_fragment(
                "decls", port, generate_ifprotocol_port_decls)
            ir.lowered_sv[_sv_key("sv", "port", port.name)] = [decls]

            inst = self._besv_port_fragment(
                "inst", port, generate_port_instantiation)
            ir.lowered_sv[_sv_key("sv", "instantiation", port.name)] = [inst]

            _log.debug(
                "[ProtocolSVEmitPass] emitted SV for port '%s' (scenario=%s)",
                port.name, port.scenario.value,
            )

    # ------------------------------------------------------------------
    # Queue FIFOs
    # ------------------------------------------------------------------

    def _emit_queues(self, ir: SynthIR) -> None:
        for q in ir.queue_nodes:
            sv = self._besv_or_legacy("fifo", q, generate_fifo_sv)
            ir.lowered_sv[_sv_key("sv", "module", f"{q.name}_fifo")] = [sv]
            _log.debug(
                "[ProtocolSVEmitPass] emitted FIFO module for queue '%s'", q.name
            )

    # ------------------------------------------------------------------
    # Arbiters for select sites
    # ------------------------------------------------------------------

    def _emit_arbiters(self, ir: SynthIR) -> None:
        for sel in ir.select_nodes:
            if sel.round_robin:
                sv = self._besv_or_legacy("rr_arb", sel,
                                          generate_rr_arbiter_sv)
            else:
                sv = self._besv_or_legacy("priority_arb", sel,
                                          generate_priority_arbiter_sv)
            key = _sv_key("sv", "module", f"{sel.name}_arb")
            ir.lowered_sv[key] = [sv]
            arb_type = "round-robin" if sel.round_robin else "priority"
            _log.debug(
                "[ProtocolSVEmitPass] emitted %s arbiter for select '%s'",
                arb_type, sel.name,
            )

    # ------------------------------------------------------------------
    # be.sv structural emit (with legacy fallback) — process→FSM unification U-3
    # ------------------------------------------------------------------

    def _besv_or_legacy(self, kind: str, node, legacy_fn) -> str:
        """Emit *node*'s SV via ``zuspec.be.sv`` (sole SV backend), else legacy.

        The be.sv builder produces structured RTL IR that ``SVEmitter`` serialises;
        constructs it doesn't cover (e.g. a round-robin arbiter, empty select)
        raise ``ProtocolSVUnsupported`` and fall back to the legacy generator —
        so this flip is regression-safe (no node loses coverage).
        """
        try:
            from zuspec.be.sv.passes.protocol_to_sv import (
                build_fifo_sv, build_priority_arbiter_sv, build_rr_arbiter_sv,
                ProtocolSVUnsupported,
            )
            from zuspec.be.sv.ir.sv_emit import SVEmitter
            builder = {"fifo": build_fifo_sv,
                       "priority_arb": build_priority_arbiter_sv,
                       "rr_arb": build_rr_arbiter_sv}[kind]
            nodes = builder(node, module_prefix=self._prefix)
            return SVEmitter().emit_all(nodes) + "\n"
        except ProtocolSVUnsupported as e:
            _log.debug("[ProtocolSVEmitPass] be.sv declined %s (%s); using legacy",
                       kind, e)
            return legacy_fn(node, module_prefix=self._prefix)

    def _besv_port_fragment(self, which: str, port, legacy_fn) -> str:
        """Emit an IfProtocol port-decl / instantiation *fragment* via ``be.sv``.

        Unlike modules, these are port-list / connection fragments (no wrapper),
        so they go through ``RTLEmitter``'s fragment renderers rather than
        ``SVEmitter.emit_all``.  The be.sv output is byte-identical to the legacy
        generator for every scenario; the bare-``Exception`` fallback is purely
        defensive during the flip (no construct is expected to be unmapped).
        """
        try:
            from zuspec.be.sv.passes.protocol_to_sv import (
                build_ifprotocol_port_decls, build_ifprotocol_port_instantiation,
            )
            builder = {"decls": build_ifprotocol_port_decls,
                       "inst": build_ifprotocol_port_instantiation}[which]
            return builder(port)
        except Exception as e:  # pragma: no cover - defensive
            _log.debug("[ProtocolSVEmitPass] be.sv declined port %s (%s); using legacy",
                       which, e)
            return legacy_fn(port)
