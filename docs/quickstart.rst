##########
Quickstart
##########

This guide shows how to synthesise a simple pipeline from a Python component
description to a Verilog 2005 module in under 10 lines of code.

Installation
============

.. code-block:: bash

   pip install zuspec-synth zuspec-dataclasses

Or for development (from the repository root):

.. code-block:: bash

   pip install -e packages/zuspec-synth -e packages/zuspec-dataclasses

Basic Example
=============

The following example synthesises a 3-stage ALU pipeline (IF → EX → WB):

.. code-block:: python

   import zuspec.dataclasses as zdc
   from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
   from zuspec.synth.passes import (
       PipelineAnnotationPass,
       HazardAnalysisPass,
       ForwardingGenPass,
       StallGenPass,
       SVEmitPass,
   )

   class AluPipe:
       """Simple 3-stage ALU pipeline."""
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

   # Build the synthesis IR
   cfg = SynthConfig(forward_default=True)
   ir = SynthIR()
   ir.component = AluPipe

   # Run the pass chain
   for pass_cls in [
       PipelineAnnotationPass,
       HazardAnalysisPass,
       ForwardingGenPass,
       StallGenPass,
   ]:
       ir = pass_cls(cfg).run(ir)
   ir = SVEmitPass(cfg).run(ir)

   # Print the generated Verilog
   print(ir.lowered_sv["pipeline_sv"])

The pass chain produces::

   // Pipeline module: AluPipe
   // Stages: ['IF', 'EX', 'WB']
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
   ...

Writing output to a file
========================

Use :class:`~zuspec.synth.passes.SVEmitPass` with ``output_path``:

.. code-block:: python

   ir = SVEmitPass(cfg, output_path="alu_pipe.v").run(ir)

Next Steps
==========

* :doc:`pipeline` — All pipeline synthesis options explained.
* :doc:`examples` — More complete worked examples.
* :doc:`api` — Full API reference.
