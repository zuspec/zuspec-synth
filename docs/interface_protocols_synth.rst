.. _interface-protocols-synth:

###################################
Interface Protocol Synthesis Guide
###################################

This guide covers the synthesis side of ``zdc.IfProtocol``-based ports —
how to choose properties that produce the RTL structure you want, how to
read the generated signals, and how to verify the output using the emitted
SVA assertions.

For the DSL and Python runtime side see the `zuspec-dataclasses documentation
<../zuspec-dataclasses/interface_protocols.html>`_.

.. contents:: On this page
   :depth: 3
   :local:

From Properties to RTL Structure
==================================

The synthesizer reads the ``_zdc_protocol_props`` dict attached to every
``IfProtocol`` class and selects one of five RTL templates (Scenarios A–E).
The table below summarises the mapping.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Scenario
     - Trigger properties
     - Generated RTL structures
   * - **A** Fixed latency
     - ``fixed_latency=N``
     - Request payload wire + N-stage shift-register delay line + response
       data wire.  No handshake signals.
   * - **B** Basic handshake
     - ``max_outstanding=1``, no ``fixed_latency``
     - ``req_valid``, ``req_ready``, ``req_payload``, ``resp_valid``,
       ``resp_data``.  Optional ``resp_ready`` if ``resp_has_backpressure``.
   * - **C** In-order multi-outstanding
     - ``max_outstanding=N>1``, ``in_order=True``
     - Scenario B signals + in-flight counter (prevents over-issuing) +
       response FIFO (depth N, absorbs pipelined responses).
   * - **D** Out-of-order multi-outstanding
     - ``max_outstanding=N>1``, ``in_order=False``
     - Scenario B signals + request ID field + response ID field + reorder
       buffer (ROB) indexed by ID.
   * - **E** Pipelined (II > 1)
     - ``initiation_interval=II>1``
     - Wraps any Scenario above with a down-counter that gates ``req_valid``
       for II−1 cycles after each accepted request.

Designing for a Specific Scenario
===================================

Scenario A — Synchronous ROM / pipelined operator
---------------------------------------------------

Use when the target *always* responds after a known, fixed number of cycles
and never backpressures::

    class RomIface(zdc.IfProtocol,
                   max_outstanding=1,
                   req_always_ready=True,
                   resp_always_valid=True,
                   fixed_latency=4):
        async def read(self, addr: zdc.u32) -> zdc.u32: ...

Generated SV (port fragment)::

    output [31:0] rom_req_payload,
    input  [31:0] rom_resp_data,
    // 4-deep shift register inlined in always @(posedge clk) block

Scenario B — Simple FIFO / memory-mapped register
--------------------------------------------------

Use when only one request can be in flight and the target may back-pressure::

    class RegIface(zdc.IfProtocol,
                   max_outstanding=1,
                   req_always_ready=False,
                   resp_has_backpressure=False):
        async def access(self, addr: zdc.u32, wr: zdc.u1,
                         wdata: zdc.u32) -> zdc.u32: ...

Scenario C — DDR controller / AXI read (in-order)
---------------------------------------------------

Use for any bus that allows the requester to queue several transactions and
returns results in the same order::

    class DdrIface(zdc.IfProtocol,
                   max_outstanding=8,
                   in_order=True):
        async def read(self, addr: zdc.u64) -> zdc.u64: ...

The synthesizer generates an ``inflight_cnt`` register (4 bits for 8 slots)
and a response FIFO of depth 8.  Use ``zdc.spawn()`` in the component body
to issue requests concurrently.

Scenario D — AXI4 with out-of-order responses
-----------------------------------------------

::

    class AxiIface(zdc.IfProtocol,
                   max_outstanding=16,
                   in_order=False):
        async def read(self, addr: zdc.u64, id_: zdc.u4) -> zdc.u64: ...

A 4-bit ``req_id`` / ``resp_id`` pair is emitted.  The ROB holds 16 slots.

Reading the Generated Signals
==============================

Signal naming follows the pattern ``<port_name>_<role>``.

For a port named ``mem`` with Scenario C properties::

    mem_req_valid          // 1-bit: requester → target
    mem_req_ready          // 1-bit: target → requester
    mem_req_payload        // W-bit: packed request arguments
    mem_resp_valid         // 1-bit: target → requester
    mem_resp_data          // W-bit: response value
    mem_inflight_cnt       // ⌈log₂(max_outstanding)⌉+1 bits
    mem_resp_fifo_push     // internal
    mem_resp_fifo_pop      // internal
    mem_resp_fifo_dout     // W-bit: head of response FIFO

For Scenario D, additionally::

    mem_req_id             // ⌈log₂(max_outstanding)⌉ bits
    mem_resp_id            // same width
    mem_rob_data[0:N-1]    // ROB payload array
    mem_rob_done[0:N-1]    // ROB valid bits

Verifying with SVA Assertions
===============================

The synthesis pipeline optionally appends a ``// synthesis translate_off``
block of SystemVerilog Assertions (SVA) to the generated module.  These
assertions catch protocol violations in simulation and can be formally
verified.

Enabling SVA generation
------------------------

Pass ``emit_sva=True`` to the protocol pipeline runner, or add it to the
``SynthConfig``::

    from zuspec.synth import protocol_pipeline
    sv = protocol_pipeline.run(MyComp, emit_sva=True)

Key assertions emitted
-----------------------

**Request stability** — ``req_valid`` and ``req_payload`` must not change
after the request is accepted until the next valid::

    property p_req_stable;
        @(posedge clk) (mem_req_valid && !mem_req_ready) |=>
            ($stable(mem_req_valid) && $stable(mem_req_payload));
    endproperty
    assert property (p_req_stable) else $fatal(1, "mem: request changed while stalled");

**In-flight counter bound** (Scenario C/D)::

    assert property (@(posedge clk) mem_inflight_cnt <= MAX_OUTSTANDING)
        else $fatal(1, "mem: inflight_cnt overflow");

**ROB slot re-use guard** (Scenario D)::

    assert property (@(posedge clk)
        (mem_req_valid && mem_req_ready) |-> !mem_rob_done[mem_req_id])
        else $fatal(1, "mem: ROB slot collision");

Running formal verification
----------------------------

With `SymbiYosys <https://symbiyosys.readthedocs.io/>`_::

    [options]
    mode prove
    depth 20

    [files]
    build/MyComp.sv

    [engines]
    smtbmc

With Questa (simulation-mode SVA)::

    vsim -assertdebug work.MyComp_tb

Common Synthesis Errors
========================

``ProtocolCompatError: max_outstanding mismatch``
-------------------------------------------------

The requester's ``max_outstanding`` is greater than the provider's.
Reduce the requester's value or increase the provider's.

``CompletionAnalysisError: multiple set() sites``
-------------------------------------------------

The synthesizer found more than one ``done.set(value)`` call reachable from
the ``Completion`` creation site.  Refactor to a single set site (typically
the innermost completion handler).

``SpawnLowerError: spawned coroutine calls multiple IfProtocol ports``
----------------------------------------------------------------------

``zdc.spawn()`` lowering cannot statically bound the slot count when the
spawned coroutine calls ports with different ``max_outstanding`` values.
Split the spawned coroutine into separate ones, one per port.

``ProtocolCompatError: fixed_latency mismatch``
-----------------------------------------------

One side declares ``fixed_latency=N`` and the other does not (or declares a
different value).  Both sides of a connection must agree on ``fixed_latency``.

.. seealso::

   `IfProtocol DSL guide <../zuspec-dataclasses/interface_protocols.html>`_

   `Split transactions guide <../zuspec-dataclasses/split_transactions.html>`_

   :doc:`api` — Full API reference including protocol pass classes.
