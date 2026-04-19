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
            decls = generate_ifprotocol_port_decls(port)
            ir.lowered_sv[_sv_key("sv", "port", port.name)] = [decls]

            inst = generate_port_instantiation(port)
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
            sv = generate_fifo_sv(q, module_prefix=self._prefix)
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
                sv = generate_rr_arbiter_sv(sel, module_prefix=self._prefix)
            else:
                sv = generate_priority_arbiter_sv(sel, module_prefix=self._prefix)
            key = _sv_key("sv", "module", f"{sel.name}_arb")
            ir.lowered_sv[key] = [sv]
            arb_type = "round-robin" if sel.round_robin else "priority"
            _log.debug(
                "[ProtocolSVEmitPass] emitted %s arbiter for select '%s'",
                arb_type, sel.name,
            )
