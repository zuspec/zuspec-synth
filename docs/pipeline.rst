####################
Pipeline Synthesis
####################

``zuspec-synth`` supports two approaches for synthesising a
``@zdc.pipeline``-decorated method into a Verilog 2005 pipeline module.

.. contents:: Contents
   :local:
   :depth: 2

Overview
========

The synthesis pass chain converts a Python method annotated with
``@zdc.pipeline`` into a fully synthesisable Verilog 2005 module.

The pipeline module follows a single-issue, in-order pipeline structure:

* One ``always @(*)`` combinational block per stage.
* One ``reg`` per inter-stage live variable (the *pipeline register*).
* One ``always @(posedge clk)`` block per inter-stage register set.
* A ``*_valid_q`` register per stage (the *valid chain*).
* Optional forwarding mux networks and stall-signal wires.
* Optional inlined register-file arrays for ``IndexedRegFile`` fields.

---

Method-per-Stage API (New)
===========================

The preferred API defines each stage as a ``@zdc.stage`` method on the
component class.  The pipeline root method describes data flow via a sequence
of ``self.STAGE(args)`` calls.

.. code-block:: python

   from zuspec.synth.passes import (
       PipelineFrontendPass,
       AutoThreadPass,
       HazardAnalysisPass,
       ForwardingGenPass,
       StallGenPass,
       SyncBodyLowerPass,
       SVEmitPass,
   )

   cfg = SynthConfig(forward_default=True)
   ir = SynthIR()
   ir.component = MyPipelineClass
   ir.model_context = DataModelFactory().build(MyPipelineClass)

   for pass_cls in [
       PipelineFrontendPass,   # build PipelineIR from @zdc.stage methods
       AutoThreadPass,         # insert threading registers for skipped stages
       HazardAnalysisPass,     # detect RAW/WAW/WAR hazards
       ForwardingGenPass,      # resolve hazards to forward or stall
       StallGenPass,           # generate stall/cancel/flush wires
       SyncBodyLowerPass,      # lower zdc.stage.* queries in @zdc.sync bodies
   ]:
       ir = pass_cls(cfg).run(ir)

   ir = SVEmitPass(cfg).run(ir)
   sv_text = ir.lowered_sv["pipeline_sv"]

Example component:

.. code-block:: python

   @zdc.dataclass
   class TwoStage(zdc.Component):
       clk:   zdc.clock
       rst_n: zdc.reset
       x:     zdc.u32

       @zdc.pipeline(clock="clk", reset="rst_n")
       def execute(self):
           (x,) = self.S1()
           self.S2(x)

       @zdc.stage
       def S1(self) -> (zdc.u32,):
           return (self.x,)

       @zdc.stage
       def S2(self, x: zdc.u32) -> ():
           pass

See the ``zuspec-dataclasses`` :ref:`pipeline-api` documentation for the full
user-facing API reference, including ``@zdc.stage``, ``@zdc.sync``, stall,
cancel, flush, and stage-query DSL calls.

New Pass Descriptions
----------------------

:class:`~zuspec.synth.passes.PipelineFrontendPass`
    Reads ``PipelineRootIR`` and ``StageMethodIR`` objects produced by
    ``DataModelFactory`` and builds an ordered list of
    :class:`~zuspec.synth.ir.pipeline_ir.StageIR` objects.  Infers
    ``ChannelDecl`` entries from stage arguments and return values.  Copies
    ``no_forward``, ``stall_cond``, ``cancel_cond``, and ``flush_decls`` from
    each ``StageMethodIR`` into the corresponding ``StageIR``.

:class:`~zuspec.synth.passes.AutoThreadPass`
    Automatically inserts threading pipeline registers for variables that
    skip intermediate stages.  For each variable produced at stage *k* and
    consumed at stage *m > k+1*, it inserts ``ChannelDecl`` entries at every
    intermediate stage boundary.  Users never write threading wires; the pass
    infers them from the call-sequence graph.

:class:`~zuspec.synth.passes.StallGenPass`
    Generates stall, cancel, and flush wires.  Emits the priority-encoded
    valid-chain flip-flop template per stage (flush > cancel > stall-hold >
    normal propagation).

:class:`~zuspec.synth.passes.SyncBodyLowerPass`
    Substitutes ``zdc.stage.valid/ready/stalled(self.X)`` query calls in
    ``@zdc.sync`` bodies with the corresponding Verilog signal names
    (``X_valid``, ``~X_valid | ~X_stalled``, ``X_stalled``).  Also records
    ``zdc.stage.flush(...)`` calls in sync bodies as ``FlushDeclNode`` entries
    for ``StallGenPass``.

---

Sentinel-Based API (Deprecated)
=================================

.. deprecated::
   This API is superseded by the method-per-stage API above.  Migrate by
   replacing ``IF = zdc.stage()`` sentinels with ``@zdc.stage`` methods.
   See the migration guide in the ``zuspec-dataclasses``
   :ref:`pipeline-api` documentation.

In the old API you place ``zdc.stage()`` markers inside the pipeline body to
define stage boundaries.  Operations between two consecutive markers belong to
the stage named by the *first* marker in that interval.

.. code-block:: python

   class FourStagePipe:
       a: zdc.u32
       b: zdc.u32
       out: zdc.u32

       @zdc.pipeline(clock=lambda s: s.clk, reset=lambda s: s.rst_n,
                     stages=["IF", "ID", "EX", "WB"])
       def execute(self):
           IF = zdc.stage()
           a: zdc.u32 = self.a
           b: zdc.u32 = self.b
           ID = zdc.stage()
           EX = zdc.stage()
           result: zdc.u32 = a + b
           WB = zdc.stage()
           self.out = result

Use ``PipelineAnnotationPass`` (deprecated) for components defined with this API:

.. code-block:: python

   from zuspec.synth.passes import (
       PipelineAnnotationPass,   # Deprecated: sentinel-based extraction
       HazardAnalysisPass,
       ForwardingGenPass,
       StallGenPass,
       SVEmitPass,
   )

Approach A — Automatic Scheduling
===================================

In Approach A, omit all ``zdc.stage()`` markers and let the SDC scheduler
assign operations to stages.

Pass ``stages=N`` (a specific stage count) or ``stages=True`` (let the
scheduler choose):

.. code-block:: python

   class AluAutoA:
       a: zdc.u32
       b: zdc.u32
       out: zdc.u32

       @zdc.pipeline(clock="clk", reset="rst_n", stages=2, forward=True)
       def execute(self):
           # No zdc.stage() markers — scheduler decides
           a: zdc.u32 = self.a
           b: zdc.u32 = self.b
           result: zdc.u32 = a + b
           self.out = result

Include :class:`~zuspec.synth.passes.SDCSchedulePass` *after*
:class:`~zuspec.synth.passes.PipelineAnnotationPass` in the chain:

.. code-block:: python

   for pass_cls in [
       PipelineAnnotationPass,
       SDCSchedulePass,        # required for Approach A
       HazardAnalysisPass,
       ForwardingGenPass,
       StallGenPass,
   ]:
       ir = pass_cls(cfg).run(ir)

Hazard Detection
================

:class:`~zuspec.synth.passes.HazardAnalysisPass` detects three hazard kinds:

* **RAW** (Read-After-Write): a stage reads a value that a later stage writes.
* **WAW** (Write-After-Write): two stages both write the same variable.
* **WAR** (Write-After-Read): a stage reads, and a later stage writes, the
  same variable.

All detected hazards are stored in :attr:`PipelineIR.hazards`.

IndexedRegFile hazards (RAW between a memory read in an early stage and a
memory write in a later stage) are stored separately in
:attr:`PipelineIR.regfile_hazards`.

Forwarding and Stalling
========================

:class:`~zuspec.synth.passes.ForwardingGenPass` resolves each hazard:

* **Forward**: a bypass mux carries the fresh value directly from the
  producing stage to the consuming stage without waiting for the pipeline
  register.
* **Stall**: the pipeline is frozen until the producing stage has committed.

Resolution priority (highest first):

1. ``@zdc.stage(no_forward=True)`` on the writing stage — forces stall for
   all outputs of that stage (new API).
2. Explicit ``no_forward=[...]`` list on ``@zdc.pipeline`` — per-signal stall
   override.
3. ``SynthConfig.forward_default`` (``True`` → forward, ``False`` → stall).

For ``IndexedRegFile`` hazards, specify the forwarding signal as
``"field_name.result_var"``:

.. code-block:: python

   @zdc.pipeline(
       clock="clk", reset="rst_n",
       stages=["ID", "EX", "WB"],
       forward=[zdc.forward(signal="regfile.rdata1",
                            from_stage="WB", to_stage="ID")],
   )
   def execute(self): ...

IndexedRegFile Support
======================

``zdc.IndexedRegFile`` fields are treated as a *modeling type*: the synthesis
pass inlines a ``reg`` array directly into the pipeline module rather than
generating a separate submodule.

The emitted RTL consists of three sections:

1. **Memory array declaration** (module scope):

   .. code-block:: verilog

      reg [31:0] regfile_mem [0:31];

2. **Clocked write port** (one per write access, gated by stage valid):

   .. code-block:: verilog

      always @(posedge clk) begin
          if (wb_valid_q)
              regfile_mem[rd_ex_to_wb_q] <= result_ex_to_wb_q;
      end

3. **Combinational read with optional forwarding bypass**:

   .. code-block:: verilog

      reg [31:0] rdata1_id;
      always @(*) begin
          if (wb_valid_q && rd_ex_to_wb_q == rs1_id)
              rdata1_id = result_ex_to_wb_q;  // bypass from WB
          else
              rdata1_id = regfile_mem[rs1_id];
      end

In the pipeline body, ``self.regfile.read(addr)`` and
``self.regfile.write(addr, data)`` statements are **not** emitted in the stage
``always @(*)`` blocks; they are replaced by the dedicated sections above.

Signal Naming Conventions
==========================

Understanding signal names makes it easier to trace generated RTL.

+-------------------------------+-------------------------------------------+
| SV signal                     | Meaning                                   |
+===============================+===========================================+
| ``a_if_to_ex``                | Wire: variable ``a`` leaving stage IF     |
+-------------------------------+-------------------------------------------+
| ``a_if_to_ex_q``              | Reg: pipeline flop for ``a`` (IF→EX)     |
+-------------------------------+-------------------------------------------+
| ``a_if``                      | Stage-local signal for ``a`` in stage IF  |
+-------------------------------+-------------------------------------------+
| ``if_valid_q``                | Stage IF valid register                   |
+-------------------------------+-------------------------------------------+
| ``IF_valid``                  | Stage IF valid (new API naming)           |
+-------------------------------+-------------------------------------------+
| ``IF_stalled``                | Stage IF stall signal (new API)           |
+-------------------------------+-------------------------------------------+
| ``IF_ready``                  | Stage IF ready: ``~IF_valid | ~IF_stalled``|
+-------------------------------+-------------------------------------------+
| ``EX_flush_IF``               | EX stage flushing IF stage                |
+-------------------------------+-------------------------------------------+
| ``regfile_mem``               | Inlined register-file array               |
+-------------------------------+-------------------------------------------+
| ``rdata1_id``                 | Regfile read result for stage ID          |
+-------------------------------+-------------------------------------------+
| ``stall_result``              | Stall signal for hazard on ``result``     |
+-------------------------------+-------------------------------------------+
