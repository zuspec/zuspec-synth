.. _async_pipeline_synthesis:

Async Pipeline Synthesis
========================

This guide describes how ``zuspec-synth`` converts a Python ``@zdc.pipeline``
behavioral model — written using ``async``/``await`` stage notation — into
Verilog 2005 RTL.  The transformation is purely static: no simulation is run;
the synthesiser walks the Python AST.

.. contents:: Contents
   :local:
   :depth: 2


Overview
--------

A developer writes a pipeline as a coroutine whose body is partitioned into
stages using ``async with zdc.pipeline.stage() as NAME:`` blocks.  The
synthesiser converts this into a Verilog module with:

* One pipeline register per inter-stage boundary (``D-FF``, clocked on the
  positive edge of ``clk``).
* A ``NAME_valid_q`` register tracking token presence per stage.
* A stall / freeze network driven by RAW hazard comparators or multi-cycle
  counters.
* An optional register-file ``reg`` array with a write port and a
  combinational read+bypass mux.

The entry point is :func:`~zuspec.synth.run_async_pipeline_synth`:

.. code-block:: python

   from zuspec.synth import run_async_pipeline_synth
   sv = run_async_pipeline_synth(MyPipelineComponent)
   print(sv)  # Verilog 2005 module

Or equivalently, apply the individual passes (see `Pass chain`_ below).


Pass Chain
----------

The synthesis pipeline consists of the following ordered passes:

.. code-block:: none

   AsyncPipelineElaboratePass
       Runs the DataModelFactory on the component class to resolve field
       types.  Stores the elaborated class on SynthIR.
         ↓
   AsyncPipelineToIrPass
       Visits the pipeline coroutine's AST, identifies stage boundaries,
       auto-threads live variables across stages (ChannelDecl), lowers
       typed local variables into port_widths, and translates IrHazardOp
       nodes into RegFileAccess / RegFileDeclInfo entries.
         ↓
   HazardAnalysisPass
       Scans plain-variable live-sets to detect RAW / WAW / WAR hazard
       pairs across stage boundaries.  Also detects regfile RAW hazards
       and populates pip.regfile_hazards (separate from pip.hazards).
         ↓
   ForwardingGenPass
       Resolves each plain-variable RAW hazard as "forward" (bypass mux)
       or "stall".  Resolves each regfile hazard using the per-field
       lock_type ("bypass" → forward mux, "queue" → stall).
         ↓
   StallGenPass
       Builds the valid chain (one ValidStageEntry per stage) with
       stall_signals per entry.  Generates StallSignal IR entries for
       RAW comparators.  Propagates multi-cycle mc_stall into the chain.
         ↓
   SVEmitPass
       Emits the Verilog 2005 module: port declarations, pipeline
       registers, valid chain always blocks, stall logic, multi-cycle
       counters, bubble wires, regfile arrays, and forwarding muxes.

Each pass operates on a :class:`~zuspec.synth.ir.synth_ir.SynthIR` container
and returns a (possibly mutated) ``SynthIR``.  The passes are composable —
you can inspect the IR at any intermediate stage by stopping early.


Live-Variable Analysis and Auto-Threading
-----------------------------------------

Variables assigned in stage ``S`` and read in stage ``S+K`` are
*auto-threaded*: the synthesiser automatically generates inter-stage pipeline
registers to carry the value forward.

The mechanism is:

1. ``AsyncPipelineToIrPass`` collects Python-typed local annotations
   (``x: zdc.u32 = ...``) and infers bit widths from the type (e.g., ``u32``
   → 32 bits, ``u5`` → 5 bits).
2. Each live variable that crosses a stage boundary becomes a
   :class:`~zuspec.synth.ir.pipeline_ir.ChannelDecl` carrying
   ``(name, src_stage, dst_stages, width)``.
3. ``SVEmitPass`` emits one ``reg [W-1:0] {name}_{stage_lower}_q`` per
   stage the variable must cross, with a clocked assignment in the
   stage's pipeline register block.

Component input ports accessed as ``self.PORT`` in the first stage are
treated as external inputs (not registered on entry).


Hazard Mapping: QueueLock vs BypassLock
----------------------------------------

Resources declared with ``zdc.pipeline.resource(N, lock=...)`` produce
an inlined RTL register-file array.  The ``lock`` type controls hazard
resolution:

``BypassLock`` (default)
    A read–write alias (RAW hazard) is resolved by a **bypass forwarding
    mux**.  When the write stage is valid and the write address matches
    the read address at run time, the read result is taken from the
    write data path instead of the memory array.  This produces
    zero-stall forwarding.

    Generated Verilog::

        reg [31:0] rdata_id;
        always @(*) begin
          if (wb_valid_q && rd_wb == rs1_id)
            rdata_id = result_wb;  // bypass from WB
          else
            rdata_id = rf_mem[rs1_id];
        end

``QueueLock``
    A RAW hazard is resolved by **stall**: the read stage stalls (and all
    upstream stages freeze) until the write has committed.  No bypass mux
    is emitted; the read is always a direct memory access.

    Generated Verilog::

        reg [31:0] rdata_id;
        always @(*) begin
          rdata_id = rf_mem[rs1_id];  // direct, stall ensures no alias
        end

The lock type is stored in
:class:`~zuspec.synth.ir.pipeline_ir.RegFileDeclInfo` as ``lock_type``
(``"bypass"`` or ``"queue"``) and consumed by
``ForwardingGenPass._resolve_regfile_hazards``.

``IndexedRegFile`` typed fields (``regfile: zdc.IndexedRegFile[zdc.u5, zdc.u32]``)
follow the same mechanism; their addr/data widths are inferred from the
type parameters and they default to bypass forwarding.


Multi-Cycle Stages
------------------

A stage can span more than one clock cycle::

    async with zdc.pipeline.stage(cycles=4) as COMPUTE:
        result = await self.do_multiply(a, b)

``cycles=N`` sets ``StageIR.cycle_hi = N - 1``.  The synthesiser emits:

* A ``{stage_lower}_cycle_q`` counter register (``[$clog2(N)-1:0]``).
* An ``always @(posedge clk)`` block that counts from 0 to ``N-1`` and
  asserts ``{stage_lower}_mc_stall = (cycle_q != N-1)`` while busy.
* The ``mc_stall`` signal is added to ``stall_signals`` in all valid-chain
  entries up to and including the multi-cycle stage, so upstream stages
  freeze while it is counting.

The formal property wrapper (P2) guards ``!$past(mc_stall, 1)`` to avoid
spurious assertion failures during the count-up phase.


``bubble()`` Semantics
-----------------------

Inside a stage, a token can self-cancel::

    async with zdc.pipeline.stage() as ID:
        if should_squash:
            await zdc.pipeline.stage.bubble()

``await S.bubble()`` is lowered to an :class:`~zuspec.dataclasses.ir.pipeline_async.IrBubble`
node.  The synthesiser emits:

* A combinational ``{stage_lower}_bubble`` wire driven by the squash condition.
* In the valid-chain ``always`` block for the *next* stage, the valid
  register is cleared when ``bubble`` is asserted::

      if (id_valid_q && !id_bubble)
        ex_valid_q <= 1'b1;
      else
        ex_valid_q <= 1'b0;

Bubbles only affect the one stage where they are inserted; earlier stages
continue to flow.


Signal Naming Reference
-----------------------

The following table summarises the Verilog signal names emitted for a
stage named ``FETCH`` (lower-case: ``fetch``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Signal
     - Meaning
   * - ``fetch_valid_q``
     - 1 when a valid token occupies the FETCH stage.
   * - ``fetch_{var}_q``
     - Pipeline register carrying variable ``var`` out of FETCH.
   * - ``fetch_stall``
     - 1 when FETCH must hold its state (stall from downstream hazard).
   * - ``fetch_flush``
     - 1 when FETCH must discard its token (bubble or branch flush).
   * - ``fetch_bubble``
     - 1 when the token in FETCH self-cancels (``await S.bubble()``).
   * - ``fetch_mc_stall``
     - 1 while a multi-cycle stage upstream (or in FETCH) is counting.
   * - ``fetch_cycle_q``
     - Multi-cycle counter for FETCH (only if ``cycles>1``).
   * - ``{field}_mem[…]``
     - Register-file memory array (one per ``IndexedRegFile`` / ``PipelineResource`` field).
   * - ``{field}_{result_var}_{stage}``
     - Regfile read result wire (combinational, with optional bypass).

Stall signal propagation: a stall asserted for stage ``S`` freezes ``S``
and all stages upstream of ``S`` (lower index).  The valid-chain ``always``
block guards every stage's valid register with its cumulative stall set.


Examples
--------

3-Stage Adder (no hazards)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../zuspec-dataclasses/examples/rtl/pipeline_adder.py
   :language: python
   :lines: 24-45

Synthesise::

    python3 -c "
    from pipeline_adder import Adder
    from zuspec.synth import run_async_pipeline_synth
    print(run_async_pipeline_synth(Adder))
    "

The emitted module has three ``always @(posedge clk)`` blocks (one per
inter-stage register set), a three-entry valid chain, and no stall logic
since there are no hazards.


5-Stage RISC-V (BypassLock forwarding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../zuspec-dataclasses/examples/rtl/pipeline_riscv5.py
   :language: python
   :lines: 54-119

The ``rf`` field is a ``PipelineResource(32, lock=BypassLock())``.  The
synthesiser:

1. Emits a ``reg [31:0] rf_mem [0:31]`` array.
2. Detects the WB→ID RAW hazard on ``rf.rdata``.
3. Emits a bypass mux: ``if (wb_valid_q && rd_wb == rs1_id) rdata_id = result_wb``.

No stall is required because the bypass always produces the current value.


Limitations
-----------

The following features are **not yet supported** and are out of scope for
the current implementation:

* **Approach A (auto-scheduling)**: the synthesiser requires explicit
  ``async with zdc.pipeline.stage() as NAME:`` markers.  SDC-based
  automatic stage partitioning is planned but not implemented.
* **RenameLock**: register renaming is not modelled in RTL; only
  ``BypassLock`` and ``QueueLock`` produce hardware.
* **Out-of-order execution**: the pipeline assumes in-order token flow.
* **Multiple pipeline methods per component**: only one ``@zdc.pipeline``
  method per component class is synthesised.
* **Memory / cache stages**: ``zdc.pipeline.resource`` models an SRAM-style
  register file; true multi-cycle memory accesses use ``cycles=N`` but
  external cache/memory interfaces are not generated.
* **Formal proof of forwarding correctness**: the structural checker
  (``zuspec.synth.verify.structural``) validates basic properties but full
  formal proofs (P3/P4) for the bypass mux are not yet in the SBY flow.
