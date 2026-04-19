.. Zuspec Synthesis documentation master file

Zuspec Synthesis
================

The Zuspec Synthesis package (``zuspec-synth``) transforms Python hardware
descriptions annotated with ``zuspec-dataclasses`` decorators into synthesisable
RTL.  Its primary feature is **pipeline synthesis**: given a Python method
decorated with ``@zdc.pipeline``, it automatically emits a Verilog 2005 module
with inter-stage pipeline registers, forwarding/stall logic, and—where
applicable—inlined register-file arrays.

**Quick Links:**

* :doc:`quickstart` — Get started in 5 minutes
* :doc:`pipeline` — Pipeline synthesis deep-dive
* :doc:`api` — Full API reference
* :doc:`examples` — Worked examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart
   pipeline
   async_pipeline_synthesis
   interface_protocols_synth
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   api

Key Features
------------

* **Approach C** (user-annotated stages): place ``zdc.stage()`` markers inside
  the pipeline body to specify stage boundaries explicitly.
* **Approach A** (automatic scheduling): omit stage markers and let the SDC
  scheduler determine an optimal pipeline partition.
* **Hazard detection**: RAW, WAW, and WAR hazards are detected automatically
  across stage boundaries.
* **Forwarding / stall**: resolved per-signal via ``zdc.forward()`` /
  ``zdc.no_forward()`` hints or the ``forward=`` argument on ``@zdc.pipeline``.
* **IndexedRegFile**: ``zdc.IndexedRegFile`` fields are inlined as ``reg``
  arrays with write-enabled clocked ports and combinational read+bypass muxes.
* **Verilog 2005 output**: fully synthesisable, no ``always_ff``/``logic``
  keywords.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
