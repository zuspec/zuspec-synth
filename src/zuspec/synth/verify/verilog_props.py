"""Embed formal properties into a DUT Verilog file for Yosys/sby verification.

Yosys does NOT support hierarchical references (``dut.signal``) in separate
property modules — they become escaped identifiers (free wires) in the SMT
model, causing assertions to fail trivially.  The correct approach is to
embed ``ifdef FORMAL`` blocks directly inside the DUT module so that signal
references resolve naturally.

The ``f_past_valid`` register guards all ``$past``-based assertions against
the undefined initial step (step 0) where the "one-cycle-ago" state does not
yet exist in the base-case trace.

Naming conventions match ``pipeline_sv_emit.py`` exactly, sourced from
``PipelineIR.valid_chain``:

- Stage valid FF:     ``valid_chain[i].valid_reg``
- Stage flush signal: ``valid_chain[i].flush_signal``
- Stage cancel:       ``valid_chain[i].cancel_signal``
- Channel register:   ``{ch.name}_q``
"""
from __future__ import annotations

import re
from io import StringIO

from ..ir.pipeline_ir import PipelineIR


class VerilogPropertyWrapper:
    """Embed formal properties into a DUT Verilog module.

    Call :meth:`generate_formal_dut` to insert an ``ifdef FORMAL`` block
    with the following properties before the ``endmodule`` line:

    - **P1** — after a reset cycle all ``valid_reg`` registers are 0.
    - **P2** — ``source_valid`` propagates to ``valid_reg`` when not flushed/stalled.
    - **P3** — stall freezes channel registers (only when stall signals exist).
    - **P5** — flush clears the target stage's valid bit.
    - **P6** — cancel clears the stage's valid bit.

    :param pip: Lowered :class:`~zuspec.synth.ir.pipeline_ir.PipelineIR`.
    """

    def __init__(self, pip: PipelineIR) -> None:
        self._pip = pip

    def generate_formal_dut(self, dut_verilog: str) -> str:
        """Insert an ``ifdef FORMAL`` block into *dut_verilog*.

        The block is inserted immediately before the final ``endmodule``
        statement so that all signal references resolve without a ``dut.``
        hierarchy prefix.

        :param dut_verilog: Verilog source of the synthesised DUT.
        :returns: Modified Verilog source with embedded formal properties.
        """
        block = self._build_formal_block()
        # Insert before the last `endmodule` line
        return re.sub(r"(\bendmodule\b)", block + r"\n\1", dut_verilog, count=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_formal_block(self) -> str:
        buf = StringIO()
        pip = self._pip
        vc = pip.valid_chain

        buf.write("`ifdef FORMAL\n")

        # f_past_valid: becomes 1 after the first clock edge, guarding all
        # $past()-based assertions from the undefined initial step.
        buf.write("  reg f_past_valid;\n")
        buf.write("  initial f_past_valid = 0;\n")
        buf.write("  always @(posedge clk) f_past_valid <= 1;\n\n")

        # Constrain the solver to start from reset state
        buf.write("  initial assume (!rst_n);\n\n")

        # P1: after a reset cycle all valid registers are 0
        buf.write("  // P1: after a reset cycle all valid registers are 0\n")
        buf.write("  always @(posedge clk) begin\n")
        buf.write("    if (f_past_valid && $past(!rst_n, 1)) begin\n")
        for entry in vc:
            buf.write(f"      assert ({entry.valid_reg} == 1'b0);\n")
        buf.write("    end\n")
        buf.write("  end\n\n")

        # P2: valid_reg takes source_valid when not flushed/stalled/cancelled
        buf.write("  // P2: valid propagates when not flushed/stalled/cancelled\n")
        for entry in vc:
            guards = ["f_past_valid", "$past(rst_n, 1)"]
            if entry.flush_signal:
                guards.append(f"!$past({entry.flush_signal}, 1)")
            if entry.cancel_signal:
                guards.append(f"!$past({entry.cancel_signal}, 1)")
            for ss in entry.stall_signals:
                guards.append(f"!$past({ss}, 1)")
            cond = " && ".join(guards)
            buf.write("  always @(posedge clk) begin\n")
            buf.write(f"    if ({cond})\n")
            buf.write(
                f"      assert ({entry.valid_reg} == $past({entry.source_valid}, 1));\n"
            )
            buf.write("  end\n\n")

        # P5: flush clears the target stage's valid bit
        has_flush = any(e.flush_signal for e in vc)
        if has_flush:
            buf.write("  // P5: flush clears the target stage valid bit\n")
            for entry in vc:
                if not entry.flush_signal:
                    continue
                buf.write("  always @(posedge clk) begin\n")
                buf.write(
                    f"    if (f_past_valid && $past(rst_n, 1) && $past({entry.flush_signal}, 1))\n"
                )
                buf.write(f"      assert ({entry.valid_reg} == 1'b0);\n")
                buf.write("  end\n\n")

        # P6: cancel clears this stage's valid bit
        has_cancel = any(e.cancel_signal for e in vc)
        if has_cancel:
            buf.write("  // P6: cancel clears this stage valid bit\n")
            for entry in vc:
                if not entry.cancel_signal:
                    continue
                buf.write("  always @(posedge clk) begin\n")
                buf.write(
                    f"    if (f_past_valid && $past(rst_n, 1) && $past({entry.cancel_signal}, 1))\n"
                )
                buf.write(f"      assert ({entry.valid_reg} == 1'b0);\n")
                buf.write("  end\n\n")

        # P3: stall freezes channel registers (when stall signals exist)
        if pip.stall_signals:
            buf.write("  // P3: stall freezes channel registers\n")
            for ch in pip.channels:
                ch_q = f"{ch.name}_q"
                dst_entry = next(
                    (e for e in vc if e.stage_name.lower() == ch.dst_stage.lower()),
                    None,
                )
                if not dst_entry or not dst_entry.stall_signals:
                    continue
                stall_cond = " || ".join(
                    f"$past({ss}, 1)" for ss in dst_entry.stall_signals
                )
                buf.write("  always @(posedge clk) begin\n")
                buf.write(
                    f"    if (f_past_valid && $past(rst_n, 1) && ({stall_cond}))\n"
                )
                buf.write(f"      assert ({ch_q} == $past({ch_q}, 1));\n")
                buf.write("  end\n\n")

        buf.write("`endif  // FORMAL\n")
        return buf.getvalue()

