"""PipelineSVCodegen — emit Verilog 2005 RTL from a PipelineIR.

Generates a standalone, synthesisable Verilog 2005 module from a
:class:`~zuspec.synth.ir.pipeline_ir.PipelineIR` that has been processed by
the full pipeline-pass chain:

  PipelineAnnotationPass (or SDCSchedulePass)
    → HazardAnalysisPass
    → ForwardingGenPass
    → StallGenPass

The emitted RTL structure:

1. Module header — clock, reset, and all component I/O ports.
2. Inter-stage pipeline registers — one ``reg`` per ``ChannelDecl``.
3. Valid-signal chain — one ``reg`` per stage; driven from ``valid_in`` /
   previous stage; frozen or bubbled when a stall is active.
4. Forwarding mux network — one ``always @(*)`` block per non-suppressed
   ``ForwardingDecl``; priority-encoded when multiple writers exist.
5. Stall signals — one ``wire`` per ``StallSignal``.
6. Stage combinational logic — one ``always @(*)`` per stage, gated by
   ``stage_valid_q``.
7. Module footer.

The output is Verilog 2005 compatible (no ``always_ff`` / ``always_comb``,
no ``logic`` keyword), consistent with existing ``mls.py`` conventions.
"""
from __future__ import annotations

import ast
import logging
import textwrap
from io import StringIO
from typing import Dict, List, Optional, Tuple

from zuspec.synth.ir.pipeline_ir import (
    ChannelDecl, ForwardingDecl, PipelineIR, RegFileAccess, RegFileDeclInfo,
    RegFileHazard, StageIR,
)
from .expr_lowerer import ExprLowerer, collect_ports, _get_sv_width

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code-generation helpers
# ---------------------------------------------------------------------------

class _SV(StringIO):
    """StringIO with indentation helpers."""

    def __init__(self, indent: str = "    ") -> None:
        super().__init__()
        self._indent = indent
        self._level = 0

    def line(self, text: str = "") -> "_SV":
        if text:
            self.write(self._indent * self._level + text + "\n")
        else:
            self.write("\n")
        return self

    def indent(self) -> "_SV":
        self._level += 1
        return self

    def dedent(self) -> "_SV":
        self._level = max(0, self._level - 1)
        return self


def _stage_lower(name: str) -> str:
    return name.lower()


def _reg_name(ch: ChannelDecl) -> str:
    """Verilog register name for a pipeline channel register."""
    return f"{ch.name}_q"


def _resolve_stage_var(var: str, stage: StageIR) -> str:
    """Resolve a variable name to its SV signal name in *stage*.

    If the variable arrives via a pipeline register (i.e., it is in one of the
    stage's input channels), the channel's flop name is returned.  Otherwise
    the variable is assumed to be stage-local and its stage-suffixed name is
    returned (e.g. ``rs1_id`` for ``rs1`` in the ``ID`` stage).
    """
    sl = stage.name.lower()
    for ch in stage.inputs:
        suffix = f"_{ch.src_stage.lower()}_to_{ch.dst_stage.lower()}"
        ch_var = ch.name[: -len(suffix)] if ch.name.endswith(suffix) else ch.name
        if ch_var == var:
            return f"{ch.name}_q"
    return f"{var}_{sl}"


def _valid_reg(stage: StageIR) -> str:
    return f"{_stage_lower(stage.name)}_valid_q"


# ---------------------------------------------------------------------------
# Main emitter
# ---------------------------------------------------------------------------

class PipelineSVCodegen:
    """Emit Verilog 2005 from a ``PipelineIR``.

    Usage::

        sv_text = PipelineSVCodegen().emit(pipeline_ir, clock_name="clk",
                                           reset_name="rst_n", reset_active_low=True)
    """

    def emit(
        self,
        pip: PipelineIR,
        *,
        clock_name: str = "clk",
        reset_name: str = "rst_n",
        reset_active_low: bool = True,
    ) -> str:
        """Return a complete Verilog 2005 module string for *pip*.

        The emitted RTL contains (in order):

        1. Module header (ports).
        2. Register-file memory arrays (if any ``IndexedRegFile`` fields).
        3. Inter-stage pipeline registers.
        4. Register-file clocked write ports.
        5. Valid-signal chain.
        6. Stall-signal wires.
        7. Forwarding mux networks.
        8. Register-file combinational read+bypass muxes.
        9. Stage-local intermediate signal declarations.
        10. Per-stage ``always @(*)`` combinational blocks.
        11. ``endmodule``.

        :param pip: Completed :class:`PipelineIR` (all passes run).
        :type pip: PipelineIR
        :param clock_name: Name of the clock port (default ``"clk"``).
        :type clock_name: str
        :param reset_name: Name of the reset port (default ``"rst_n"``).
        :type reset_name: str
        :param reset_active_low: ``True`` if reset is active-low (default).
        :type reset_active_low: bool
        :return: Synthesisable Verilog 2005 module source string.
        :rtype: str
        """
        sv = _SV()
        reset_cond = f"!{reset_name}" if reset_active_low else reset_name

        self._emit_header(sv, pip, clock_name, reset_name)
        self._emit_regfile_arrays(sv, pip)
        self._emit_pipeline_registers(sv, pip, clock_name, reset_name, reset_cond)
        self._emit_regfile_writes(sv, pip, clock_name, reset_name, reset_cond)
        self._emit_valid_chain(sv, pip, clock_name, reset_name, reset_cond)
        self._emit_stall_signals(sv, pip)
        self._emit_forwarding_muxes(sv, pip)
        self._emit_regfile_read_muxes(sv, pip)
        self._emit_stage_signals(sv, pip)
        self._emit_stage_logic(sv, pip)
        self._emit_footer(sv, pip)

        return sv.getvalue()

    # ------------------------------------------------------------------
    # Section emitters
    # ------------------------------------------------------------------

    def _emit_header(self, sv: _SV, pip: PipelineIR,
                     clock_name: str, reset_name: str) -> None:
        sv.line(f"// Pipeline module: {pip.module_name}")
        sv.line(f"// Stages: {[s.name for s in pip.stages]}")
        sv.line(f"// Approach: {pip.approach}  II={pip.initiation_interval}")
        sv.line(f"// Auto-generated by PipelineSVCodegen — do not edit directly.")
        sv.line()
        # Collect all ports first so we can handle commas correctly
        in_ports, out_ports = collect_ports(pip)
        in_port_names = {name for name, _ in in_ports}
        port_lines: List[str] = [
            f"input  wire {clock_name}",
            f"input  wire {reset_name}",
        ]
        # Only add the global valid_in port if the component doesn't already
        # declare a field with that name (which would be included via in_ports).
        if "valid_in" not in in_port_names:
            port_lines.append("input  wire valid_in")
        for name, width in in_ports:
            port_lines.append(f"input  wire [{width - 1}:0] {name}")
        for name, width in out_ports:
            port_lines.append(f"output reg  [{width - 1}:0] {name}")
        sv.line(f"module {pip.module_name} (")
        sv.indent()
        for i, pl in enumerate(port_lines):
            comma = "," if i < len(port_lines) - 1 else ""
            sv.line(pl + comma)
        sv.dedent()
        sv.line(");")
        sv.line()
        sv.line()

    def _emit_pipeline_registers(
        self, sv: _SV, pip: PipelineIR,
        clock_name: str, reset_name: str, reset_cond: str,
    ) -> None:
        if not pip.channels:
            return
        sv.line("// ── Inter-stage pipeline registers ──────────────────────────────")
        for ch in pip.channels:
            sv.line(f"reg [{ch.width-1}:0] {_reg_name(ch)};")
        sv.line()
        sv.line(f"always @(posedge {clock_name}) begin")
        sv.indent()
        sv.line(f"if ({reset_cond}) begin")
        sv.indent()
        for ch in pip.channels:
            sv.line(f"{_reg_name(ch)} <= {ch.width}'b0;")
        sv.dedent()
        sv.line("end else begin")
        sv.indent()
        # Stall wires — build combined stall expression if any stall signals exist
        stall_sigs: List[str] = []
        if hasattr(pip, "stall_signals") and pip.stall_signals:  # type: ignore[attr-defined]
            stall_sigs = [ss.signal_name for ss in pip.stall_signals]  # type: ignore[attr-defined]
        stall_expr = " | ".join(stall_sigs) if stall_sigs else ""

        for ch in pip.channels:
            suffix = f"_{ch.src_stage.lower()}_to_{ch.dst_stage.lower()}"
            if ch.name.endswith(suffix):
                var_name = ch.name[: -len(suffix)]
            else:
                # AutoThreadPass uses _thru_{dst_lower} naming
                thru_suffix = f"_thru_{ch.dst_stage.lower()}"
                if ch.name.endswith(thru_suffix):
                    var_name = ch.name[: -len(thru_suffix)]
                else:
                    var_name = ch.name
            # The RHS is the value of var_name at the src stage:
            # if it arrives via an input channel, read from that register;
            # otherwise it comes directly from the input port.
            src_stage_obj = next((s for s in pip.stages if s.name == ch.src_stage), None)
            rhs = var_name  # default: input port
            if src_stage_obj is not None:
                for in_ch in src_stage_obj.inputs:
                    in_suffix = f"_{in_ch.src_stage.lower()}_to_{in_ch.dst_stage.lower()}"
                    in_thru = f"_thru_{in_ch.dst_stage.lower()}"
                    if in_ch.name.endswith(in_suffix):
                        in_var = in_ch.name[: -len(in_suffix)]
                    elif in_ch.name.endswith(in_thru):
                        in_var = in_ch.name[: -len(in_thru)]
                    else:
                        in_var = in_ch.name
                    if in_var == var_name:
                        rhs = _reg_name(in_ch)
                        break
            if stall_expr:
                sv.line(f"if (!({stall_expr})) {_reg_name(ch)} <= {rhs};")
            else:
                sv.line(f"{_reg_name(ch)} <= {rhs};")
        sv.dedent()
        sv.line("end")
        sv.dedent()
        sv.line("end")
        sv.line()

    def _emit_valid_chain(
        self, sv: _SV, pip: PipelineIR,
        clock_name: str, reset_name: str, reset_cond: str,
    ) -> None:
        sv.line("// ── Valid-signal chain ───────────────────────────────────────────")
        for stage in pip.stages:
            sv.line(f"reg {_valid_reg(stage)};")
        sv.line()
        sv.line(f"always @(posedge {clock_name}) begin")
        sv.indent()
        sv.line(f"if ({reset_cond}) begin")
        sv.indent()
        for stage in pip.stages:
            sv.line(f"{_valid_reg(stage)} <= 1'b0;")
        sv.dedent()
        sv.line("end else begin")
        sv.indent()

        valid_chain = getattr(pip, "valid_chain", [])
        stall_sigs_all: List[str] = []
        if hasattr(pip, "stall_signals"):
            stall_sigs_all = [ss.signal_name for ss in pip.stall_signals]  # type: ignore[attr-defined]
        # Also include decl_stall wire names
        for ds in getattr(pip, "decl_stalls", []):
            stall_sigs_all.append(ds.wire_name)
        stall_expr = " | ".join(stall_sigs_all) if stall_sigs_all else ""

        for idx, stage in enumerate(pip.stages):
            vr = _valid_reg(stage)
            src = "valid_in" if idx == 0 else _valid_reg(pip.stages[idx - 1])

            ve = valid_chain[idx] if idx < len(valid_chain) else None

            # Priority: flush > cancel > stall-freeze > normal
            if ve and (ve.flush_signal or ve.cancel_signal or ve.stall_signals):
                flush_cond = ve.flush_signal if ve.flush_signal else "1'b0"
                sv.line(f"if ({flush_cond}) begin")
                sv.indent()
                sv.line(f"{vr} <= 1'b0;  // flush wins")
                sv.dedent()
                if ve.cancel_signal:
                    sv.line(f"end else if ({ve.cancel_signal}) begin")
                    sv.indent()
                    sv.line(f"{vr} <= 1'b0;  // cancel (no upstream freeze)")
                    sv.dedent()
                if ve.stall_signals:
                    freeze_cond = " | ".join(ve.stall_signals)
                    sv.line(f"end else if ({freeze_cond}) begin")
                    sv.indent()
                    sv.line(f"{vr} <= {vr};  // frozen by stall")
                    sv.dedent()
                sv.line("end else begin")
                sv.indent()
                if ve.bubble_on_stall and ve.stall_signals:
                    bubble_cond = " | ".join(ve.stall_signals)
                    sv.line(f"{vr} <= ({bubble_cond}) ? 1'b0 : {src};")
                else:
                    sv.line(f"{vr} <= {src};")
                sv.dedent()
                sv.line("end")
            elif ve and ve.bubble_on_stall and ve.stall_signals:
                bubble_cond = " | ".join(ve.stall_signals)
                sv.line(f"{vr} <= ({bubble_cond}) ? 1'b0 : {src};")
            elif ve and ve.stall_signals and stall_expr:
                sv.line(f"if (!({stall_expr})) {vr} <= {src};")
            else:
                sv.line(f"{vr} <= {src};")
        sv.dedent()
        sv.line("end")
        sv.dedent()
        sv.line("end")
        sv.line()

    def _emit_stall_signals(self, sv: _SV, pip: PipelineIR) -> None:
        stall_sigs = getattr(pip, "stall_signals", [])
        decl_stalls = getattr(pip, "decl_stalls", [])
        cancels     = getattr(pip, "cancels", [])
        flushes     = getattr(pip, "flushes", [])

        if not stall_sigs and not decl_stalls and not cancels and not flushes:
            return
        sv.line("// ── Stall / Cancel / Flush signals ──────────────────────────────")
        for ss in stall_sigs:
            # Conservative stall: producer stage valid (no address comparison
            # without more type info; full compare generated when type info available)
            prod_valid = f"{_stage_lower(ss.producer_stage)}_valid_q"
            sv.line(f"wire {ss.signal_name} = {prod_valid};  "
                    f"// stall: {ss.hazard_signal} hazard "
                    f"{ss.producer_stage}→{ss.consumer_stage}")
        # Combined stall (hazard-based)
        if len(stall_sigs) > 1:
            combined = " | ".join(ss.signal_name for ss in stall_sigs)
            sv.line(f"wire stall = {combined};")
        elif stall_sigs:
            sv.line(f"wire stall = {stall_sigs[0].signal_name};")

        # Stage-declared stalls
        for ds in decl_stalls:
            sv.line(f"wire {ds.wire_name} = {ds.cond_expr};  // declared stall ({ds.stage_name})")

        # Cancel wires
        for c in cancels:
            sv.line(f"wire {c.wire_name} = {c.cond_expr};  // cancel ({c.stage_name})")

        # Per-source flush wires
        for f in flushes:
            sv.line(f"wire {f.wire_name} = {f.cond_expr};  "
                    f"// flush from {f.source_stage} → {f.target_stage}")

        # Aggregated flush per target
        from collections import defaultdict
        by_target = defaultdict(list)
        for f in flushes:
            by_target[f.target_stage].append(f.wire_name)
        for target, wires in by_target.items():
            tgt_lower = target.lower()
            sv.line(f"wire {tgt_lower}_flush = {' | '.join(wires)};  // aggregated flush → {target}")

        sv.line()

    def _emit_forwarding_muxes(self, sv: _SV, pip: PipelineIR) -> None:
        # Exclude regfile signals (identified by "." in the name — e.g. "regfile.rdata1")
        # since those are handled by _emit_regfile_read_muxes.
        active = [d for d in pip.forwarding if not d.suppressed and "." not in d.signal]
        if not active:
            return
        sv.line("// ── Forwarding mux network ───────────────────────────────────────")
        # Group by (to_stage, signal) — multiple producers need priority mux
        from collections import defaultdict
        groups: Dict = defaultdict(list)
        for d in active:
            groups[(d.to_stage, d.signal)].append(d)

        # Build a map from signal name → width via channels
        ch_width: Dict[str, int] = {}
        for ch in pip.channels:
            suffix = f"_{ch.src_stage.lower()}_to_{ch.dst_stage.lower()}"
            var = ch.name[:-len(suffix)] if ch.name.endswith(suffix) else ch.name
            ch_width.setdefault(var, ch.width)
        # Also check port_widths
        port_w = getattr(pip, "port_widths", {})

        for (to_stage, signal), decls in groups.items():
            width = ch_width.get(signal, port_w.get(signal, 32))
            fwd_wire = f"{signal}_{_stage_lower(to_stage)}_fwd"
            sv.line(f"reg [{width - 1}:0] {fwd_wire};")
            sv.line(f"always @(*) begin")
            sv.indent()
            # Default: register value
            sv.line(f"{fwd_wire} = {signal}_{_stage_lower(to_stage)}_q;  // default: pipeline reg")
            # Bypasses (priority: closest stage first — sorted by stage index)
            stage_idx = {s.name: s.index for s in pip.stages}
            sorted_decls = sorted(decls, key=lambda d: -stage_idx.get(d.from_stage, 0))
            for d in sorted_decls:
                prod_valid = f"{_stage_lower(d.from_stage)}_valid_q"
                sv.line(f"if ({prod_valid})")
                sv.indent()
                sv.line(f"{fwd_wire} = {_stage_lower(d.from_stage)}_{d.signal};  "
                        f"// bypass from {d.from_stage}")
                sv.dedent()
            sv.dedent()
            sv.line("end")
            sv.line()

    def _emit_stage_signals(self, sv: _SV, pip: PipelineIR) -> None:
        """Declare all stage-local intermediate signals at module scope."""
        all_sigs: List[Tuple[str, int]] = []
        for stage in pip.stages:
            if stage.operations:
                lowerer = ExprLowerer(stage, pip)
                all_sigs.extend(lowerer.collect_signals(stage.operations))
        if not all_sigs:
            return
        sv.line("// ── Stage-local intermediate signals ────────────────────────────")
        for sig_name, width in all_sigs:
            sv.line(f"reg [{width - 1}:0] {sig_name};")
        sv.line()

    def _emit_stage_logic(self, sv: _SV, pip: PipelineIR) -> None:
        sv.line("// ── Stage combinational blocks ───────────────────────────────────")
        for stage in pip.stages:
            sv.line(f"// Stage {stage.name}")
            sv.line(f"always @(*) begin")
            sv.indent()
            sv.line(f"if ({_valid_reg(stage)}) begin")
            sv.indent()
            if stage.operations:
                lowerer = ExprLowerer(stage, pip)
                lines = lowerer.lower_stmts_procedural(stage.operations)
                for ln in lines:
                    sv.line(ln)
            else:
                sv.line("// (empty stage)")
            sv.dedent()
            sv.line("end else begin")
            sv.indent()
            # Default assignments prevent latches
            if stage.operations:
                lowerer_d = ExprLowerer(stage, pip)
                for sig_name, width in lowerer_d.collect_signals(stage.operations):
                    sv.line(f"{sig_name} = {width}'b0;")
                for port_name, width in lowerer_d.collect_output_ports(stage.operations):
                    sv.line(f"{port_name} = {width}'b0;")
            sv.dedent()
            sv.line("end")
            sv.dedent()
            sv.line("end")
            sv.line()

    def _emit_regfile_arrays(self, sv: _SV, pip: PipelineIR) -> None:
        """Declare register-file memory arrays (one per IndexedRegFile field)."""
        decls = getattr(pip, "regfile_decls", [])
        if not decls:
            return
        sv.line("// ── Register-file memory arrays ──────────────────────────────────")
        for d in decls:
            sv.line(
                f"reg [{d.data_width - 1}:0] {d.field_name}_mem [0:{d.depth - 1}];"
            )
        sv.line()

    def _emit_regfile_writes(
        self, sv: _SV, pip: PipelineIR,
        clock_name: str, reset_name: str, reset_cond: str,
    ) -> None:
        """Emit clocked always blocks for regfile write ports.

        One ``always @(posedge clk)`` block is emitted per write access.  The
        write is guarded by the write-stage's valid register so that bubbles
        do not spuriously write.
        """
        writes = [a for a in getattr(pip, "regfile_accesses", []) if a.kind == "write"]
        if not writes:
            return
        sv.line("// ── Register-file write ports ─────────────────────────────────────")
        # Build stage name → StageIR map
        stage_map = {s.name: s for s in pip.stages}
        for wr in writes:
            stage_obj = stage_map.get(wr.stage)
            sl = wr.stage.lower()
            valid_sig = f"{sl}_valid_q"
            # Resolve signal names through pipeline register mapping
            addr_sig = _resolve_stage_var(wr.addr_var, stage_obj) if stage_obj else f"{wr.addr_var}_{sl}"
            data_sig = _resolve_stage_var(wr.data_var, stage_obj) if stage_obj else f"{wr.data_var}_{sl}"
            sv.line(f"// Regfile write: {wr.field_name} in stage {wr.stage}")
            sv.line(f"always @(posedge {clock_name}) begin")
            sv.indent()
            sv.line(f"if ({valid_sig})")
            sv.indent()
            sv.line(
                f"{wr.field_name}_mem[{addr_sig}] <= {data_sig};"
            )
            sv.dedent()
            sv.dedent()
            sv.line("end")
            sv.line()

    def _emit_regfile_read_muxes(self, sv: _SV, pip: PipelineIR) -> None:
        """Emit combinational read-with-forwarding muxes for regfile read ports.

        For each read access, emit::

            reg [DW-1:0] result_var;
            always @(*) begin
                if (wr_valid && wr_addr == rd_addr)
                    result_var = wr_data;     // forwarded from write stage
                else
                    result_var = FIELD_mem[rd_addr];
            end

        The forwarding is only emitted when a matching ``RegFileHazard`` exists
        (i.e. a write in a later stage can alias the read address).
        """
        reads = [a for a in getattr(pip, "regfile_accesses", []) if a.kind == "read"]
        if not reads:
            return

        hazards = getattr(pip, "regfile_hazards", [])
        stage_map = {s.name: s for s in pip.stages}

        sv.line("// ── Register-file read ports (combinational, with forwarding) ─────")
        for rd in reads:
            stage_obj = stage_map.get(rd.stage)
            sl = rd.stage.lower()
            addr_sig = _resolve_stage_var(rd.addr_var, stage_obj) if stage_obj else f"{rd.addr_var}_{sl}"
            result_sig = f"{rd.result_var}_{sl}"

            # Determine data width from pip.port_widths or default
            pw = getattr(pip, "port_widths", {})
            dw = pw.get(rd.result_var, 32)

            sv.line(f"// Regfile read: {rd.field_name} in stage {rd.stage}")
            sv.line(f"reg [{dw - 1}:0] {result_sig};")
            sv.line(f"always @(*) begin")
            sv.indent()

            # Find matching hazard (same field + result_var has a write in later stage)
            matching = [
                h for h in hazards
                if h.field_name == rd.field_name
                and h.read_result_var == rd.result_var
                and h.resolved_by == "forward"
            ]

            if matching:
                for h in matching:
                    wsl = h.write_stage.lower()
                    wr_stage_obj = stage_map.get(h.write_stage)
                    wr_valid = f"{wsl}_valid_q"
                    wr_addr = _resolve_stage_var(h.write_addr_var, wr_stage_obj) if wr_stage_obj else f"{h.write_addr_var}_{wsl}"
                    wr_data = _resolve_stage_var(h.write_data_var, wr_stage_obj) if wr_stage_obj else f"{h.write_data_var}_{wsl}"
                    sv.line(f"if ({wr_valid} && {wr_addr} == {addr_sig})")
                    sv.indent()
                    sv.line(f"{result_sig} = {wr_data};  // bypass from {h.write_stage}")
                    sv.dedent()
                    sv.line(f"else")
                    sv.indent()
                    sv.line(f"{result_sig} = {rd.field_name}_mem[{addr_sig}];")
                    sv.dedent()
            else:
                # No forwarding — direct memory read
                sv.line(f"{result_sig} = {rd.field_name}_mem[{addr_sig}];")

            sv.dedent()
            sv.line("end")
            sv.line()

    def _emit_footer(self, sv: _SV, pip: PipelineIR) -> None:
        sv.line(f"endmodule  // {pip.module_name}")


_DEFAULT_WIDTH_COMMENT = "31:0"  # placeholder width comment for forwarding mux regs


# ---------------------------------------------------------------------------
# SVEmitPass wrapper
# ---------------------------------------------------------------------------

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR


class SVEmitPass(SynthPass):
    """Emit Verilog 2005 from ``ir.pipeline_ir`` using :class:`PipelineSVCodegen`.

    Stores the result in ``ir.lowered_sv["pipeline_sv"]`` and, if
    ``output_path`` is set, writes it to a file.

    This pass is a no-op when ``ir.pipeline_ir`` is ``None``.
    """

    def __init__(
        self,
        config: SynthConfig,
        *,
        output_path: Optional[str] = None,
        clock_name: str = "clk",
        reset_name: str = "rst_n",
        reset_active_low: bool = True,
    ) -> None:
        super().__init__(config=config)
        self._output_path    = output_path
        self._clock_name     = clock_name
        self._reset_name     = reset_name
        self._reset_active_low = reset_active_low

    @property
    def name(self) -> str:
        return "sv_emit"

    def run(self, ir: SynthIR) -> SynthIR:
        """Emit Verilog 2005 RTL and store it in ``ir.lowered_sv["pipeline_sv"]``.

        :param ir: Synthesis IR with all passes completed.
        :type ir: SynthIR
        :return: Updated IR with ``ir.lowered_sv["pipeline_sv"]`` set.
            If ``output_path`` was specified, the file is written as a side
            effect.
        :rtype: SynthIR
        """
        if ir.pipeline_ir is None:
            _log.debug("[SVEmitPass] no pipeline_ir — skipping")
            return ir

        pip = ir.pipeline_ir
        # SY-10: honour string clock/reset from @zdc.pipeline(clock=..., reset=...)
        clock_name = getattr(pip, 'clock_field', None) or self._clock_name
        reset_name = getattr(pip, 'reset_field', None) or self._reset_name

        sv_text = PipelineSVCodegen().emit(
            pip,
            clock_name=clock_name,
            reset_name=reset_name,
            reset_active_low=self._reset_active_low,
        )
        ir.lowered_sv["pipeline_sv"] = sv_text

        if self._output_path:
            with open(self._output_path, "w") as f:
                f.write(sv_text)
            ir.sv_path = self._output_path
            _log.info("[SVEmitPass] wrote %s (%d bytes)", self._output_path, len(sv_text))
        else:
            _log.debug("[SVEmitPass] pipeline SV generated (%d bytes)", len(sv_text))

        return ir
