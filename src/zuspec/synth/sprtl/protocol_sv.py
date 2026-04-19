"""protocol_sv.py — SV fragment generators for IfProtocol, Queue, Select, and Spawn.

All functions return synthesizable SystemVerilog source strings.  They are
called from ``ProtocolSVEmitPass`` and can also be used stand-alone.
"""
from __future__ import annotations

from io import StringIO
from typing import List, Optional

from zuspec.synth.ir.protocol_ir import (
    IfProtocolPortIR,
    IfProtocolScenario,
    QueueIR,
    SelectIR,
    SpawnIR,
)


# ============================================================
# IfProtocol port declaration snippets
# ============================================================

def generate_ifprotocol_port_decls(port: IfProtocolPortIR, indent: str = "  ") -> str:
    """Return SV port declaration lines for one IfProtocol port.

    The result is meant to be injected into a ``module`` declaration's port
    list.  Each line ends with a comma; callers must strip the trailing comma
    from the last port or handle it as part of the surrounding module text.

    Args:
        port:   ``IfProtocolPortIR`` node produced by the lowering pass.
        indent: Whitespace prefix for each line.

    Returns:
        Multi-line string of SV port declarations.
    """
    out = StringIO()
    for direction, width, name in port.all_sv_ports():
        w_str = f"[{width - 1}:0] " if width > 1 else ""
        out.write(f"{indent}{direction} logic {w_str}{name},\n")
    return out.getvalue()


# ============================================================
# Synchronous FIFO module template
# ============================================================

def generate_fifo_sv(q: QueueIR, module_prefix: str = "") -> str:
    """Return a complete synchronous FIFO SV module for *q*.

    The generated FIFO:
    - Uses a registered-output (fall-through) style.
    - Exposes standard ``wr_en``, ``wr_data``, ``rd_en``, ``rd_data``,
      ``full``, ``empty``, and ``count`` ports.
    - Uses a single-port synchronous memory (``logic [W-1:0] mem[0:D-1]``).
    - Requires an active-high synchronous reset.

    Args:
        q:             ``QueueIR`` node.
        module_prefix: Optional module-name prefix.

    Returns:
        Synthesizable SV source as a single string.
    """
    name = f"{module_prefix}{q.name}_fifo"
    W = q.elem_width
    D = q.depth
    A = q.addr_bits
    C = q.count_bits

    return f"""\
// Auto-generated synchronous FIFO for queue '{q.name}'
// Depth={D}, Width={W}
module {name} (
  input  logic             clk,
  input  logic             rst,
  input  logic             wr_en,
  input  logic [{W-1}:0]  wr_data,
  output logic             full,
  input  logic             rd_en,
  output logic [{W-1}:0]  rd_data,
  output logic             empty,
  output logic [{C-1}:0]  count
);
  logic [{W-1}:0] mem [0:{D-1}];
  logic [{A-1}:0] wr_ptr;
  logic [{A-1}:0] rd_ptr;
  logic [{C-1}:0] count_r;

  assign full  = (count_r == {D});
  assign empty = (count_r == 0);
  assign count = count_r;

  // Read data (registered output)
  always_ff @(posedge clk) begin
    if (rd_en && !empty)
      rd_data <= mem[rd_ptr];
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      wr_ptr  <= '0;
      rd_ptr  <= '0;
      count_r <= '0;
    end else begin
      if (wr_en && !full) begin
        mem[wr_ptr] <= wr_data;
        wr_ptr      <= wr_ptr + 1'b1;
      end
      if (rd_en && !empty) begin
        rd_ptr <= rd_ptr + 1'b1;
      end
      case ({{wr_en && !full, rd_en && !empty}})
        2'b10:  count_r <= count_r + 1'b1;
        2'b01:  count_r <= count_r - 1'b1;
        default: count_r <= count_r;
      endcase
    end
  end
endmodule
"""


# ============================================================
# Priority arbiter (for zdc.select with priority=True)
# ============================================================

def generate_priority_arbiter_sv(sel: SelectIR, module_prefix: str = "") -> str:
    """Return a fixed-priority arbiter module for *sel*.

    Branch 0 has the highest priority.  Each branch has a ``req_i`` input
    and a ``gnt_o`` output.  A single ``gnt_valid_o`` indicates that at least
    one grant was issued.  ``sel_o`` encodes the winning branch index.

    Args:
        sel:           ``SelectIR`` node.
        module_prefix: Optional module-name prefix.

    Returns:
        Synthesizable SV source.
    """
    N = len(sel.branches)
    if N == 0:
        return f"// Empty select '{sel.name}' — no arbiter generated\n"

    name = f"{module_prefix}{sel.name}_arb"
    idx_bits = max(1, (N - 1).bit_length()) if N > 1 else 1

    lines = [
        f"// Auto-generated priority arbiter for select '{sel.name}' ({N} branches)",
        f"module {name} (",
        f"  input  logic [{N-1}:0]      req_i,",
        f"  output logic [{N-1}:0]      gnt_o,",
        f"  output logic [{idx_bits-1}:0]    sel_o,",
        f"  output logic                 gnt_valid_o",
        f");",
        f"  always_comb begin",
        f"    gnt_o       = '0;",
        f"    sel_o       = '0;",
        f"    gnt_valid_o = 1'b0;",
    ]
    for i in range(N):
        branch = sel.branches[i]
        cond = " && ".join(f"!req_i[{j}]" for j in range(i)) if i > 0 else ""
        full_cond = f"req_i[{i}]" + (f" && {cond}" if cond else "")
        lines += [
            f"    if ({full_cond}) begin",
            f"      gnt_o[{i}]  = 1'b1;",
            f"      sel_o       = {idx_bits}'d{branch.tag_value};",
            f"      gnt_valid_o = 1'b1;",
            f"    end else",
        ]
    # close the else chain
    lines[-1] = lines[-1].rstrip(" else")
    lines += ["  end", "endmodule"]
    return "\n".join(lines) + "\n"


# ============================================================
# Round-robin arbiter
# ============================================================

def generate_rr_arbiter_sv(sel: SelectIR, module_prefix: str = "") -> str:
    """Return a round-robin arbiter module for *sel*.

    Uses a simple rotating mask (log-rounds-robin) approach suitable for
    synthesis.  Clocked on ``posedge clk``.

    Args:
        sel:           ``SelectIR`` node.
        module_prefix: Optional module-name prefix.

    Returns:
        Synthesizable SV source.
    """
    N = len(sel.branches)
    if N == 0:
        return f"// Empty select '{sel.name}' — no arbiter generated\n"

    name = f"{module_prefix}{sel.name}_rr_arb"
    idx_bits = max(1, (N - 1).bit_length()) if N > 1 else 1

    return f"""\
// Auto-generated round-robin arbiter for select '{sel.name}' ({N} branches)
module {name} (
  input  logic             clk,
  input  logic             rst,
  input  logic [{N-1}:0]  req_i,
  output logic [{N-1}:0]  gnt_o,
  output logic [{idx_bits-1}:0] sel_o,
  output logic             gnt_valid_o
);
  logic [{N-1}:0] mask_r;

  // Grant: req_i masked, then unmasked as fallback
  logic [{N-1}:0] masked_req;
  logic [{N-1}:0] unmasked_gnt;
  logic [{N-1}:0] masked_gnt;

  assign masked_req = req_i & mask_r;

  // Priority-encode masked
  always_comb begin : blk_masked
    masked_gnt = '0;
    for (int i = 0; i < {N}; i++) begin
      if (masked_req[i] && masked_gnt == '0)
        masked_gnt[i] = 1'b1;
    end
  end

  // Priority-encode unmasked (fallback)
  always_comb begin : blk_unmasked
    unmasked_gnt = '0;
    for (int i = 0; i < {N}; i++) begin
      if (req_i[i] && unmasked_gnt == '0)
        unmasked_gnt[i] = 1'b1;
    end
  end

  assign gnt_o       = (masked_gnt != '0) ? masked_gnt : unmasked_gnt;
  assign gnt_valid_o = (req_i != '0);

  // Encode sel_o
  always_comb begin
    sel_o = '0;
    for (int i = 0; i < {N}; i++) begin
      if (gnt_o[i]) sel_o = {idx_bits}'(i);
    end
  end

  // Advance mask after each grant
  always_ff @(posedge clk) begin
    if (rst) begin
      mask_r <= '1;
    end else if (gnt_valid_o) begin
      // Rotate mask left past the winner
      for (int i = 0; i < {N}; i++) begin
        if (gnt_o[i])
          mask_r <= ({N}'d1 << ((i + 1) % {N})) - 1'b1;
      end
    end
  end
endmodule
"""


# ============================================================
# IfProtocol port-bundle wrapper (instantiation snippet)
# ============================================================

def generate_port_instantiation(port: IfProtocolPortIR,
                                 instance_name: Optional[str] = None) -> str:
    """Return an SV module-instantiation connection snippet for *port*.

    Generates the ``.signal_name(signal_name)`` connection lines that
    connect the SV ports of an IfProtocol port bundle to signals of the
    same name in the enclosing module.

    Args:
        port:          ``IfProtocolPortIR`` node.
        instance_name: Unused (reserved for future use).

    Returns:
        Multi-line SV connection snippet.
    """
    out = StringIO()
    for _dir, _width, sig_name in port.all_sv_ports():
        out.write(f"  .{sig_name}({sig_name}),\n")
    return out.getvalue()
