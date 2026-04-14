########
Examples
########

.. contents:: Contents
   :local:
   :depth: 2

Example 1 — Simple 3-Stage ALU (Approach C)
============================================

A classic three-stage pipeline: instruction-fetch, execute, write-back.

.. code-block:: python

   import zuspec.dataclasses as zdc
   from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
   from zuspec.synth.passes import (
       PipelineAnnotationPass, HazardAnalysisPass,
       ForwardingGenPass, StallGenPass, SVEmitPass,
   )

   class AluPipe:
       a: zdc.u32
       b: zdc.u32
       out: zdc.u32

       @zdc.pipeline(clock="clk", reset="rst_n", stages=["IF", "EX", "WB"])
       def execute(self):
           IF = zdc.stage()
           a: zdc.u32 = self.a
           b: zdc.u32 = self.b
           EX = zdc.stage()
           result: zdc.u32 = a + b
           WB = zdc.stage()
           self.out = result

   cfg = SynthConfig(forward_default=True)
   ir = SynthIR(); ir.component = AluPipe
   for P in [PipelineAnnotationPass, HazardAnalysisPass,
             ForwardingGenPass, StallGenPass]:
       ir = P(cfg).run(ir)
   ir = SVEmitPass(cfg).run(ir)
   print(ir.lowered_sv["pipeline_sv"])

Generated Verilog (abridged):

.. code-block:: verilog

   module AluPipe (
       input  wire clk,
       input  wire rst_n,
       input  wire valid_in,
       input  wire [31:0] a,
       input  wire [31:0] b,
       output reg  [31:0] out
   );

   // ── Inter-stage pipeline registers ──────────────────────────────
   reg [31:0] a_if_to_ex_q;
   reg [31:0] b_if_to_ex_q;
   reg [31:0] result_ex_to_wb_q;

   always @(posedge clk) begin
       if (!rst_n) begin
           a_if_to_ex_q    <= 32'b0;
           b_if_to_ex_q    <= 32'b0;
           result_ex_to_wb_q <= 32'b0;
       end else begin
           a_if_to_ex_q    <= a_if;
           b_if_to_ex_q    <= b_if;
           result_ex_to_wb_q <= result_ex;
       end
   end

   // ── Stage combinational blocks ───────────────────────────────────
   // Stage IF
   always @(*) begin
       if (if_valid_q) begin
           a_if = a;
           b_if = b;
       end else begin
           a_if = 32'b0;
           b_if = 32'b0;
       end
   end

   // Stage EX
   always @(*) begin
       if (ex_valid_q) begin
           result_ex = (a_if_to_ex_q + b_if_to_ex_q);
       end else begin
           result_ex = 32'b0;
       end
   end

   // Stage WB
   always @(*) begin
       if (wb_valid_q) begin
           out = result_ex_to_wb_q;
       end else begin
           out = 32'b0;
       end
   end

   endmodule  // AluPipe

Example 2 — Automatic Scheduling (Approach A)
==============================================

Use ``stages=N`` (or ``stages=True``) with no ``zdc.stage()`` markers to let
the SDC scheduler partition operations automatically.

.. code-block:: python

   from zuspec.synth.passes import SDCSchedulePass

   class AluAutoA:
       a: zdc.u32
       b: zdc.u32
       out: zdc.u32

       @zdc.pipeline(clock="clk", reset="rst_n", stages=2, forward=True)
       def execute(self):
           # No zdc.stage() markers — the scheduler decides
           a: zdc.u32 = self.a
           b: zdc.u32 = self.b
           result: zdc.u32 = a + b
           self.out = result

   cfg = SynthConfig(forward_default=True)
   ir = SynthIR(); ir.component = AluAutoA
   for P in [PipelineAnnotationPass, SDCSchedulePass,
             HazardAnalysisPass, ForwardingGenPass, StallGenPass]:
       ir = P(cfg).run(ir)
   ir = SVEmitPass(cfg).run(ir)

With ``stages=True`` the scheduler chooses the minimum number of stages needed
to avoid all RAW hazards.

Example 3 — Register-File Pipeline (RISC-V style)
===================================================

A three-stage pipeline using ``zdc.IndexedRegFile`` for a 32-entry × 32-bit
integer register file.  The write-back stage feeds back to the decode stage
with a forwarding bypass.

.. code-block:: python

   class RfPipe:
       rs1:  zdc.u5
       rd:   zdc.u5
       imm:  zdc.u32
       out:  zdc.u32
       regfile: zdc.IndexedRegFile[zdc.u32, 32]

       @zdc.pipeline(
           clock="clk", reset="rst_n",
           stages=["ID", "EX", "WB"],
           forward=[zdc.forward(signal="regfile.rdata1",
                                from_stage="WB", to_stage="ID")],
       )
       def execute(self):
           ID = zdc.stage()
           rs1: zdc.u5 = self.rs1
           rdata1: zdc.u32 = self.regfile.read(rs1)
           EX = zdc.stage()
           rd: zdc.u5 = self.rd
           imm: zdc.u32 = self.imm
           result: zdc.u32 = rdata1 + imm
           WB = zdc.stage()
           self.regfile.write(rd, result)
           self.out = result

Generated Verilog (key sections):

.. code-block:: verilog

   // ── Register-file memory arrays ──────────────────────────────────
   reg [31:0] regfile_mem [0:31];

   // ── Register-file write ports ─────────────────────────────────────
   always @(posedge clk) begin
       if (wb_valid_q)
           regfile_mem[rd_ex_to_wb_q] <= result_ex_to_wb_q;
   end

   // ── Register-file read ports (combinational, with forwarding) ─────
   reg [31:0] rdata1_id;
   always @(*) begin
       if (wb_valid_q && rd_ex_to_wb_q == rs1_id)
           rdata1_id = result_ex_to_wb_q;  // bypass from WB
       else
           rdata1_id = regfile_mem[rs1_id];
   end

Key points:

* The ``regfile`` field does **not** appear as a module port — it is
  inlined as a ``reg`` array.
* ``self.regfile.read(rs1)`` is **not** emitted as a procedural statement in
  the ID stage block; the combinational read mux above handles it.
* ``self.regfile.write(rd, result)`` is **not** emitted in the WB stage block;
  the clocked write block handles it.
* The WB→ID bypass is emitted automatically because
  ``forward=[zdc.forward(signal="regfile.rdata1", ...)]`` resolves the RAW
  hazard to ``"forward"``.

Example 4 — Explicit Forwarding and Stalling
=============================================

Mix forwarding and stalling for different signals:

.. code-block:: python

   class MixedHazardPipe:
       a: zdc.u32
       b: zdc.u32
       c: zdc.u32
       out: zdc.u32

       @zdc.pipeline(
           clock="clk", reset="rst_n",
           stages=["IF", "EX", "WB"],
           forward=[
               zdc.forward(signal="result", from_stage="EX", to_stage="IF"),
               zdc.no_forward(signal="b"),   # stall on b instead
           ],
       )
       def execute(self):
           IF = zdc.stage()
           a: zdc.u32 = self.a
           b: zdc.u32 = self.b
           EX = zdc.stage()
           result: zdc.u32 = a + b
           WB = zdc.stage()
           self.out = result

* ``result`` is bypassed from EX back to IF via a forwarding mux.
* ``b`` generates a stall signal that freezes the pipeline for one cycle.
