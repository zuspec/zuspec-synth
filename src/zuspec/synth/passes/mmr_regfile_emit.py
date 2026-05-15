"""MmrRegFileEmitPass — SystemVerilog RTL emitter for MMR RegisterFile classes.

Generates a self-contained SystemVerilog module for each
``@zdc.regfile``-decorated ``RegisterFile`` subclass, following the
``always_comb`` / ``always_ff`` field-FF pattern described in §6.3 of
``design/abstract-mmr.md``.

Standalone usage::

    from zuspec.synth.passes.mmr_regfile_emit import synthesize_regfile
    sv_text = synthesize_regfile(MyRegs)

Pass usage (reads ``ir.meta.mmr_regfiles`` set by ElaboratePass)::

    pass_ = MmrRegFileEmitPass(config)
    pass_.run(ir)
    # results in pass_.results: dict[field_name -> sv_text]
"""
from __future__ import annotations

import re
import textwrap
from typing import Dict, List, Optional, Tuple

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _snake(name: str) -> str:
    """CamelCase → snake_case."""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    return s.lower()


def _ones(width: int) -> str:
    """SV all-ones literal for given width."""
    if width == 1:
        return "1'b1"
    return f"{width}'h{(1 << width) - 1:X}"


def _zeros(width: int) -> str:
    """SV all-zeros literal for given width."""
    if width == 1:
        return "1'b0"
    return f"{width}'h0"


def _field_bits(width: int, lsb: int) -> str:
    """SV bit-select expression, e.g. ``[3:2]`` or ``[0]``."""
    if width == 1:
        return f"[{lsb}]"
    return f"[{lsb + width - 1}:{lsb}]"


# ---------------------------------------------------------------------------
# Per-field hwif membership helpers
# ---------------------------------------------------------------------------

def _hwif_in_members(fname: str, fd) -> List[Tuple[str, int]]:
    """Return list of (signal_name, width) for hwif_in for this field."""
    from zuspec.dataclasses.mmr.enums import HW
    members: List[Tuple[str, int]] = []
    hw = fd.hw
    # Stickybit and hwset both need a 1-bit hwset input
    if fd.stickybit or fd.hwset:
        members.append((f"{fname}_hwset", 1))
    elif hw in (HW.W, HW.RW):
        members.append((f"{fname}_next", fd._width))
    if fd.hwclr:
        members.append((f"{fname}_hwclr", 1))
    if fd.we:
        members.append((f"{fname}_we", 1))
    if fd.wel:
        members.append((f"{fname}_wel", 1))
    return members


def _hwif_out_members(fname: str, fd) -> List[Tuple[str, int]]:
    """Return list of (signal_name, width) for hwif_out for this field."""
    from zuspec.dataclasses.mmr.enums import HW
    members: List[Tuple[str, int]] = []
    hw = fd.hw
    if hw in (HW.R, HW.RW):
        members.append((f"{fname}_value", fd._width))
    if fd.singlepulse or getattr(fd, 'swmod', False):
        members.append((f"{fname}_swmod", 1))
    return members


def _reg_has_intr(reg_fields) -> bool:
    """True if any field in the register has stickybit set."""
    return any(fd.stickybit for _, fd in reg_fields)


# ---------------------------------------------------------------------------
# Core emitter
# ---------------------------------------------------------------------------

class MmrRegFileRtlEmitter:
    """Emits SystemVerilog RTL for a single ``RegisterFile`` subclass.

    Parameters
    ----------
    regfile_cls:
        A ``@zdc.regfile``-decorated ``RegisterFile`` subclass.
    data_width:
        Bus data width in bits (default 32).
    addr_width:
        Bus address width in bits (default 8).
    module_name:
        Override the generated module name (default: snake_case of class name).
    """

    def __init__(
        self,
        regfile_cls,
        data_width: int = 32,
        addr_width: int = 8,
        module_name: Optional[str] = None,
    ):
        self._cls = regfile_cls
        self._data_width = data_width
        self._addr_width = addr_width
        self._module_name = module_name or _snake(regfile_cls.__name__)
        self._pkg_name = f"{self._module_name}_pkg"
        self._reg_classes = getattr(regfile_cls, '_mmr_reg_classes', [])

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def emit(self) -> str:
        """Return the complete SystemVerilog module as a string."""
        lines: List[str] = []
        lines += self._emit_header()
        lines += self._emit_module_decl()
        lines += self._emit_field_storage()
        lines += self._emit_field_combo()
        lines += self._emit_posedge_regs()
        lines += self._emit_bus_decode()
        lines += self._emit_field_logic()
        lines += self._emit_hwif_out_assigns()
        lines += self._emit_read_mux()
        lines += ["", "endmodule"]
        return "\n".join(lines)

    def emit_package(self) -> str:
        """Return the SystemVerilog package containing hwif struct typedefs."""
        lines: List[str] = []
        lines.append(f"package {self._pkg_name};")
        lines += self._emit_hwif_in_typedef(indent=4)
        lines += self._emit_hwif_out_typedef(indent=4)
        lines.append(f"endpackage : {self._pkg_name}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Module header and port list
    # ------------------------------------------------------------------

    def _emit_header(self) -> List[str]:
        mn = self._module_name
        return [
            f"// Generated by zuspec MmrRegFileRtlEmitter",
            f"// Source: {self._cls.__name__}",
            f"`include \"{self._pkg_name}.svh\"",
            "",
        ]

    def _emit_module_decl(self) -> List[str]:
        mn = self._module_name
        aw = self._addr_width
        dw = self._data_width
        lines = [
            f"module {mn} (",
            f"    input  logic        clk,",
            f"    input  logic        rst,",
            f"    // APB4 bus interface",
            f"    input  logic        psel,",
            f"    input  logic        penable,",
            f"    input  logic        pwrite,",
            f"    input  logic [{aw-1}:0]  paddr,",
            f"    input  logic [{dw-1}:0] pwdata,",
            f"    output logic [{dw-1}:0] prdata,",
            f"    output logic        pready,",
            f"    output logic        pslverr,",
            f"    // Hardware interface",
            f"    input  {mn}__in_t  hwif_in,",
            f"    output {mn}__out_t hwif_out",
            f");",
            f"",
            f"    // APB4 zero-wait-state",
            f"    assign pready  = 1'b1;",
            f"    assign pslverr = 1'b0;",
            f"",
            f"    logic cpuif_req;",
            f"    logic cpuif_req_is_wr;",
            f"    logic [{aw-1}:0] cpuif_addr;",
            f"    logic [{dw-1}:0] cpuif_wr_data;",
            f"    assign cpuif_req       = psel & penable;",
            f"    assign cpuif_req_is_wr = pwrite;",
            f"    assign cpuif_addr      = paddr;",
            f"    assign cpuif_wr_data   = pwdata;",
            f"",
        ]
        return lines

    # ------------------------------------------------------------------
    # Storage and combo structs
    # ------------------------------------------------------------------

    def _emit_field_storage(self) -> List[str]:
        lines = ["    // Field storage flip-flop values"]
        for reg_name, reg_cls in self._reg_classes:
            fields = getattr(reg_cls, '_mmr_fields', [])
            for fname, fd in fields:
                w = fd._width
                decl = f"logic" if w == 1 else f"logic [{w-1}:0]"
                lines.append(f"    {decl} field_storage_{reg_name}_{fname};")
        lines.append("")
        return lines

    def _emit_field_combo(self) -> List[str]:
        lines = ["    // Combinational next-value wires"]
        for reg_name, reg_cls in self._reg_classes:
            fields = getattr(reg_cls, '_mmr_fields', [])
            for fname, fd in fields:
                w = fd._width
                decl = f"logic" if w == 1 else f"logic [{w-1}:0]"
                lines.append(f"    {decl} field_combo_next_{reg_name}_{fname};")
                lines.append(f"    logic        field_combo_load_{reg_name}_{fname};")
        lines.append("")
        return lines

    def _emit_posedge_regs(self) -> List[str]:
        """Emit pipeline registers for stickybit posedge detection."""
        lines = []
        for reg_name, reg_cls in self._reg_classes:
            for fname, fd in getattr(reg_cls, '_mmr_fields', []):
                if fd.stickybit in (True, 'posedge', 'negedge', 'bothedge'):
                    lines.append(
                        f"    logic field_q_{reg_name}_{fname}; "
                        f"// posedge-detect pipeline register"
                    )
        if lines:
            lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Address strobe decoder
    # ------------------------------------------------------------------

    def _emit_bus_decode(self) -> List[str]:
        lines = ["    // Register address strobes"]
        for reg_name, _ in self._reg_classes:
            lines.append(f"    logic decoded_reg_strb_{reg_name};")
        lines += [
            "",
            "    always_comb begin",
        ]
        for reg_name, _ in self._reg_classes:
            lines.append(f"        decoded_reg_strb_{reg_name} = 1'b0;")
        lines += [
            "        if (cpuif_req) begin",
            f"            unique case (cpuif_addr)",
        ]
        for reg_name, reg_cls in self._reg_classes:
            off = reg_cls._mmr_offset
            lines.append(
                f"                {self._addr_width}'h{off:02X}: "
                f"decoded_reg_strb_{reg_name} = 1'b1;"
            )
        lines += [
            "                default: ;",
            "            endcase",
            "        end",
            "    end",
            "",
        ]
        return lines

    # ------------------------------------------------------------------
    # Per-field always_comb / always_ff logic
    # ------------------------------------------------------------------

    def _emit_field_logic(self) -> List[str]:
        lines = []
        for reg_name, reg_cls in self._reg_classes:
            for fname, fd in getattr(reg_cls, '_mmr_fields', []):
                lines += self._emit_one_field_comb(reg_name, fname, fd)
                lines += self._emit_one_field_ff(reg_name, fname, fd)
        return lines

    def _emit_one_field_comb(self, reg_name: str, fname: str, fd) -> List[str]:
        from zuspec.dataclasses.mmr.enums import SW, HW
        w   = fd._width
        sv  = f"field_storage_{reg_name}_{fname}"
        nxt = f"field_combo_next_{reg_name}_{fname}"
        ldn = f"field_combo_load_{reg_name}_{fname}"
        strb = f"decoded_reg_strb_{reg_name}"
        wi  = f"cpuif_wr_data{_field_bits(w, fd.lsb)}"

        lines = [f"    always_comb begin : {reg_name}_{fname}_combo"]
        lines.append(f"        {nxt}      = {sv};")
        lines.append(f"        {ldn} = 1'b0;")

        sw_can_write = fd.sw not in (SW.RO, SW.NA)
        hw_can_write = fd.hw in (HW.W, HW.RW) or fd.hwset or fd.hwclr or fd.stickybit

        # ------------------------------------------------------------------
        # SW write block (precedence='sw' → first; precedence='hw' → else if)
        # ------------------------------------------------------------------
        sw_block = self._sw_write_lines(reg_name, fname, fd, strb, wi, nxt, ldn, sv)
        hw_block = self._hw_write_lines(reg_name, fname, fd, nxt, ldn, sv)
        sp_block = self._singlepulse_lines(fname, fd, strb, nxt, ldn, sv)

        if fd.precedence == 'hw':
            lines += hw_block
            if sw_can_write:
                # Turn the `if` in sw_block into `else if`
                sw_else = _prefix_first_if(sw_block, "else ")
                lines += sw_else
        else:  # precedence='sw' (default)
            if sw_can_write:
                lines += sw_block
            if hw_can_write:
                lines += hw_block
        lines += sp_block

        lines.append(f"    end")
        lines.append("")
        return lines

    def _sw_write_lines(self, reg_name, fname, fd, strb, wi, nxt, ldn, sv) -> List[str]:
        ow = fd.onwrite
        lines = [
            f"        if ({strb} && cpuif_req_is_wr) begin",
        ]
        if ow == 'woclr':
            lines.append(f"            {nxt}      = {sv} & ~{wi};")
        elif ow == 'woset':
            lines.append(f"            {nxt}      = {sv} | {wi};")
        elif ow == 'wot':
            lines.append(f"            {nxt}      = {sv} ^ {wi};")
        elif ow == 'wzs':
            lines.append(f"            {nxt}      = {wi} ? {sv} : {_ones(fd._width)};")
        elif ow == 'wzc':
            lines.append(f"            {nxt}      = {wi} ? {sv} : {_zeros(fd._width)};")
        elif ow == 'wzt':
            lines.append(f"            {nxt}      = {wi} ? {sv} : ~{sv};")
        elif ow == 'wclr':
            lines.append(f"            {nxt}      = {_zeros(fd._width)};")
        elif ow == 'wset':
            lines.append(f"            {nxt}      = {_ones(fd._width)};")
        else:
            lines.append(f"            {nxt}      = {wi};")
        lines.append(f"            {ldn} = 1'b1;")
        lines.append(f"        end")
        return lines

    def _hw_write_lines(self, reg_name, fname, fd, nxt, ldn, sv) -> List[str]:
        from zuspec.dataclasses.mmr.enums import HW
        lines = []
        sb = fd.stickybit
        if sb in (True, 'posedge'):
            # Posedge detection: fires when pipeline reg was 0 and hwset is 1
            hws = f"hwif_in.{reg_name}.{fname}_hwset"
            q   = f"field_q_{reg_name}_{fname}"
            lines.append(f"        else if ((~{q}) & {hws}) begin")
            lines.append(f"            {nxt}      = {sv} | {hws};")
            lines.append(f"            {ldn} = 1'b1;")
            lines.append(f"        end")
        elif sb == 'negedge':
            hws = f"hwif_in.{reg_name}.{fname}_hwset"
            q   = f"field_q_{reg_name}_{fname}"
            lines.append(f"        else if ({q} & ~{hws}) begin")
            lines.append(f"            {nxt}      = {sv} | 1'b1;")
            lines.append(f"            {ldn} = 1'b1;")
            lines.append(f"        end")
        elif sb == 'bothedge':
            hws = f"hwif_in.{reg_name}.{fname}_hwset"
            q   = f"field_q_{reg_name}_{fname}"
            lines.append(f"        else if ({q} ^ {hws}) begin")
            lines.append(f"            {nxt}      = {sv} | 1'b1;")
            lines.append(f"            {ldn} = 1'b1;")
            lines.append(f"        end")
        elif fd.hwset:
            hws = f"hwif_in.{reg_name}.{fname}_hwset"
            lines.append(f"        else if ({hws}) begin")
            lines.append(f"            {nxt}      = {sv} | {hws};")
            lines.append(f"            {ldn} = 1'b1;")
            lines.append(f"        end")
        elif fd.hwclr:
            hwc = f"hwif_in.{reg_name}.{fname}_hwclr"
            lines.append(f"        else if ({hwc}) begin")
            lines.append(f"            {nxt}      = {sv} & ~{hwc};")
            lines.append(f"            {ldn} = 1'b1;")
            lines.append(f"        end")
        elif fd.hw in (HW.W, HW.RW):
            hw_next = f"hwif_in.{reg_name}.{fname}_next"
            if fd.we:
                we_sig = f"hwif_in.{reg_name}.{fname}_we"
                lines.append(f"        else if ({we_sig}) begin")
            elif fd.wel:
                wel_sig = f"hwif_in.{reg_name}.{fname}_wel"
                lines.append(f"        else if (!{wel_sig}) begin")
            else:
                lines.append(f"        else begin")
            lines.append(f"            {nxt}      = {hw_next};")
            lines.append(f"            {ldn} = 1'b1;")
            lines.append(f"        end")
        return lines

    def _singlepulse_lines(self, fname, fd, strb, nxt, ldn, sv) -> List[str]:
        """Auto-clear logic for singlepulse fields."""
        if not fd.singlepulse:
            return []
        # After any clock where the field is non-zero and SW is not writing,
        # clear back to zero next cycle.
        w = fd._width
        lines = [
            f"        // singlepulse: auto-clear when not being written",
            f"        if ({sv} != {_zeros(w)} && !({strb} && cpuif_req_is_wr)) begin",
            f"            {nxt}      = {_zeros(w)};",
            f"            {ldn} = 1'b1;",
            f"        end",
        ]
        return lines

    def _emit_one_field_ff(self, reg_name: str, fname: str, fd) -> List[str]:
        sv   = f"field_storage_{reg_name}_{fname}"
        nxt  = f"field_combo_next_{reg_name}_{fname}"
        ldn  = f"field_combo_load_{reg_name}_{fname}"
        rst_val = self._reset_literal(fd)
        w    = fd._width

        lines = [f"    always_ff @(posedge clk) begin : {reg_name}_{fname}_ff"]
        lines.append(f"        if (rst)")
        lines.append(f"            {sv} <= {rst_val};")
        lines.append(f"        else if ({ldn})")
        lines.append(f"            {sv} <= {nxt};")
        # Pipeline register for posedge/negedge/bothedge stickybit detection
        sb = fd.stickybit
        if sb in (True, 'posedge', 'negedge', 'bothedge'):
            hws = f"hwif_in.{reg_name}.{fname}_hwset"
            q   = f"field_q_{reg_name}_{fname}"
            lines.append(f"        {q} <= {hws};")
        lines.append(f"    end")
        lines.append("")
        return lines

    def _reset_literal(self, fd) -> str:
        w = fd._width
        v = fd.default & ((1 << w) - 1)
        if w == 1:
            return f"1'b{v}"
        return f"{w}'h{v:X}"

    # ------------------------------------------------------------------
    # hwif_out assigns
    # ------------------------------------------------------------------

    def _emit_hwif_out_assigns(self) -> List[str]:
        from zuspec.dataclasses.mmr.enums import HW
        lines = ["    // hwif_out assignments"]
        for reg_name, reg_cls in self._reg_classes:
            sticky_fields = []
            for fname, fd in getattr(reg_cls, '_mmr_fields', []):
                sv = f"field_storage_{reg_name}_{fname}"
                if fd.hw in (HW.R, HW.RW):
                    lines.append(
                        f"    assign hwif_out.{reg_name}.{fname}_value = {sv};"
                    )
                if fd.singlepulse or getattr(fd, 'swmod', False):
                    # swmod: high whenever SW writes this cycle
                    strb = f"decoded_reg_strb_{reg_name}"
                    lines.append(
                        f"    assign hwif_out.{reg_name}.{fname}_swmod = "
                        f"{strb} & cpuif_req_is_wr;"
                    )
                if fd.stickybit:
                    sticky_fields.append(fname)
            if sticky_fields:
                intr_expr = " | ".join(
                    f"field_storage_{reg_name}_{fn}" for fn in sticky_fields
                )
                lines.append(
                    f"    assign hwif_out.{reg_name}.intr = {intr_expr};"
                )
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Read mux (prdata)
    # ------------------------------------------------------------------

    def _emit_read_mux(self) -> List[str]:
        from zuspec.dataclasses.mmr.enums import SW
        dw = self._data_width
        aw = self._addr_width
        lines = [
            "    // Read mux",
            f"    always_comb begin",
            f"        prdata = {dw}'h0;",
            f"        if (cpuif_req && !cpuif_req_is_wr) begin",
            f"            unique case (cpuif_addr)",
        ]
        for reg_name, reg_cls in self._reg_classes:
            off = reg_cls._mmr_offset
            rdata_parts = []
            for fname, fd in getattr(reg_cls, '_mmr_fields', []):
                if fd.sw in (SW.RO, SW.RW):
                    sv = f"field_storage_{reg_name}_{fname}"
                    rdata_parts.append((fname, fd, sv))
            if rdata_parts:
                # Build bit-positioned read value
                # Start with zeros; OR in each readable field
                lines.append(
                    f"                {aw}'h{off:02X}: begin"
                )
                for fname, fd, sv in rdata_parts:
                    bs = _field_bits(fd._width, fd.lsb)
                    lines.append(
                        f"                    prdata{bs} = {sv};"
                    )
                    # Apply onread side-effects inline (handled via load_next)
                lines.append(f"                end")
            else:
                lines.append(
                    f"                {aw}'h{off:02X}: ; // write-only register"
                )
        lines += [
            "                default: ;",
            "            endcase",
            "        end",
            "    end",
            "",
        ]
        return lines

    # ------------------------------------------------------------------
    # hwif typedef emitters (for the package)
    # ------------------------------------------------------------------

    def _emit_hwif_in_typedef(self, indent: int = 4) -> List[str]:
        pad = " " * indent
        mn  = self._module_name
        lines = [f"{pad}typedef struct packed {{"]
        for reg_name, reg_cls in self._reg_classes:
            fields = getattr(reg_cls, '_mmr_fields', [])
            members = []
            for fname, fd in fields:
                members += _hwif_in_members(fname, fd)
            if members:
                lines.append(f"{pad}    struct packed {{")
                for sig, w in members:
                    decl = "logic" if w == 1 else f"logic [{w-1}:0]"
                    lines.append(f"{pad}        {decl} {sig};")
                lines.append(f"{pad}    }} {reg_name};")
        lines.append(f"{pad}}} {mn}__in_t;")
        lines.append("")
        return lines

    def _emit_hwif_out_typedef(self, indent: int = 4) -> List[str]:
        pad = " " * indent
        mn  = self._module_name
        lines = [f"{pad}typedef struct packed {{"]
        for reg_name, reg_cls in self._reg_classes:
            fields = getattr(reg_cls, '_mmr_fields', [])
            members = []
            for fname, fd in fields:
                members += _hwif_out_members(fname, fd)
            has_intr = _reg_has_intr(fields)
            if has_intr:
                members.append(("intr", 1))
            if members:
                lines.append(f"{pad}    struct packed {{")
                for sig, w in members:
                    decl = "logic" if w == 1 else f"logic [{w-1}:0]"
                    lines.append(f"{pad}        {decl} {sig};")
                lines.append(f"{pad}    }} {reg_name};")
        lines.append(f"{pad}}} {mn}__out_t;")
        lines.append("")
        return lines


# ---------------------------------------------------------------------------
# Utility: prefix first 'if' in a block with a keyword (e.g. "else ")
# ---------------------------------------------------------------------------

def _prefix_first_if(lines: List[str], prefix: str) -> List[str]:
    """Return a copy of *lines* with the first ``if`` statement prefixed."""
    result = []
    done = False
    for line in lines:
        if not done and line.lstrip().startswith("if "):
            result.append(line.replace("if ", prefix + "if ", 1))
            done = True
        else:
            result.append(line)
    return result


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def synthesize_regfile(
    regfile_cls,
    data_width: int = 32,
    addr_width: int = 8,
    module_name: Optional[str] = None,
    include_package: bool = False,
) -> str:
    """Generate SystemVerilog RTL for a ``RegisterFile`` subclass.

    Parameters
    ----------
    regfile_cls:
        A ``@zdc.regfile``-decorated ``RegisterFile`` subclass.
    data_width:
        Bus data width in bits (default 32).
    addr_width:
        Bus address width in bits (default 8).
    module_name:
        Override the generated module name (default: snake_case of class name).
    include_package:
        If True, prepend the hwif package definition to the output.

    Returns
    -------
    str
        The complete SystemVerilog text.
    """
    emitter = MmrRegFileRtlEmitter(
        regfile_cls,
        data_width=data_width,
        addr_width=addr_width,
        module_name=module_name,
    )
    sv = emitter.emit()
    if include_package:
        pkg = emitter.emit_package()
        sv = pkg + "\n\n" + sv
    return sv


# ---------------------------------------------------------------------------
# SynthPass wrapper
# ---------------------------------------------------------------------------

class MmrRegFileEmitPass(SynthPass):
    """Synthesis pass that emits SystemVerilog for MMR RegisterFile sub-components.

    Reads ``ir.meta.mmr_regfiles`` (populated by ``ElaboratePass``) and
    generates one SV module per discovered MMR register file.

    Results are stored in ``ir.lowered_sv`` under the key pattern
    ``sv/regfile/<field_name>`` (module SV) and
    ``sv/regfile/<field_name>_pkg`` (package SV).

    Also populates ``self.results`` as a plain ``{field_name: sv_text}`` dict
    for easy access in tests.
    """

    def __init__(self, config: SynthConfig) -> None:
        super().__init__(config)
        self.results: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "mmr_regfile_emit"

    def run(self, ir: SynthIR) -> None:
        if ir.meta is None:
            return
        for decl in ir.meta.mmr_regfiles:
            emitter = MmrRegFileRtlEmitter(
                decl.regfile_cls,
                module_name=decl.module_name,
            )
            sv_text  = emitter.emit()
            pkg_text = emitter.emit_package()
            ir.lowered_sv[f"sv/regfile/{decl.field_name}"]     = sv_text
            ir.lowered_sv[f"sv/regfile/{decl.field_name}_pkg"] = pkg_text
            self.results[decl.field_name] = sv_text
