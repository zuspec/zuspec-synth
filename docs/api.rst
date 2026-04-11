##############
API Reference
##############

This page documents the public API of ``zuspec.synth``.

.. contents:: Contents
   :local:
   :depth: 2

Synthesis IR
============

The intermediate representation (IR) types are defined in
``zuspec.synth.ir.pipeline_ir`` and ``zuspec.synth.ir.synth_ir``.

SynthIR
-------

.. class:: zuspec.synth.ir.synth_ir.SynthIR

   Top-level synthesis IR container.  Passed through the pass chain;
   each pass may read and update its fields.

   **Attributes:**

   .. attribute:: component

      The Python class being synthesised.

   .. attribute:: pipeline_ir

      :class:`PipelineIR` instance, set by :class:`PipelineAnnotationPass` and
      updated by subsequent passes.  ``None`` until annotation completes.

   .. attribute:: lowered_sv

      ``Dict[str, str]`` mapping output key to generated Verilog text.
      After :class:`SVEmitPass` runs, the key ``"pipeline_sv"`` holds the
      complete Verilog module string.


SynthConfig
-----------

.. class:: zuspec.synth.ir.synth_ir.SynthConfig(forward_default=True)

   Configuration object forwarded to every pass.

   :param forward_default: Default forwarding policy when no explicit
       ``zdc.forward()`` / ``zdc.no_forward()`` hint is found.
       ``True`` → insert a forwarding bypass mux; ``False`` → stall.
   :type forward_default: bool


PipelineIR
----------

.. class:: zuspec.synth.ir.pipeline_ir.PipelineIR

   Complete lowered representation of a single ``@zdc.pipeline`` method.

   **Attributes:**

   .. attribute:: module_name
      :type: str

      Verilog module name (typically the Python class name).

   .. attribute:: stages
      :type: List[StageIR]

      Ordered list of pipeline stages, from first to last.

   .. attribute:: channels
      :type: List[ChannelDecl]

      Inter-stage pipeline register declarations.

   .. attribute:: forwarding
      :type: List[ForwardingDecl]

      Forwarding / stall declarations per signal.  Populated by
      :class:`PipelineAnnotationPass` from ``zdc.forward()`` hints and
      completed by :class:`ForwardingGenPass`.

   .. attribute:: hazards
      :type: List[HazardPair]

      Detected scalar hazards.  Populated by :class:`HazardAnalysisPass`.

   .. attribute:: regfile_accesses
      :type: List[RegFileAccess]

      All ``self.FIELD.read()`` / ``.write()`` calls found in stage bodies.
      Populated by :class:`HazardAnalysisPass`.

   .. attribute:: regfile_hazards
      :type: List[RegFileHazard]

      RAW hazards between regfile reads and writes.
      Populated by :class:`HazardAnalysisPass`.

   .. attribute:: regfile_decls
      :type: List[RegFileDeclInfo]

      One entry per unique ``IndexedRegFile`` field; contains depth/width
      information for memory array declaration.
      Populated by :class:`HazardAnalysisPass`.

   .. attribute:: approach
      :type: str

      ``"user"`` (Approach C), ``"auto"`` (Approach A before SDC), or
      ``"sdc"`` (Approach A after SDC scheduling).

   .. attribute:: pipeline_stages
      :type: int

      Number of pipeline stages (user-specified or inferred).

   .. attribute:: initiation_interval
      :type: int

      Initiation interval in clock cycles (default 1).


StageIR
-------

.. class:: zuspec.synth.ir.pipeline_ir.StageIR

   One pipeline stage.

   **Attributes:**

   .. attribute:: name
      :type: str

      Stage name (e.g. ``"IF"``, ``"EX"``, ``"WB"``).

   .. attribute:: index
      :type: int

      Zero-based stage index.

   .. attribute:: inputs
      :type: List[ChannelDecl]

      Pipeline register channels arriving at this stage.

   .. attribute:: outputs
      :type: List[ChannelDecl]

      Pipeline register channels leaving this stage.

   .. attribute:: operations
      :type: List[ast.stmt]

      AST statements belonging to this stage.


ChannelDecl
-----------

.. class:: zuspec.synth.ir.pipeline_ir.ChannelDecl

   An inter-stage pipeline register.

   **Attributes:**

   .. attribute:: name
      :type: str

      Signal name base (e.g. ``"a_if_to_ex"``).
      The ``_q`` suffix is appended for the Verilog ``reg``.

   .. attribute:: width
      :type: int

      Data width in bits.

   .. attribute:: src_stage
      :type: str

      Producing stage name.

   .. attribute:: dst_stage
      :type: str

      Consuming stage name.


ForwardingDecl
--------------

.. class:: zuspec.synth.ir.pipeline_ir.ForwardingDecl

   A forwarding or stall declaration for a single signal.

   **Attributes:**

   .. attribute:: from_stage / to_stage
      :type: str

      Producer / consumer stage names.

   .. attribute:: signal
      :type: str

      Variable name (scalar) or ``"field.result_var"`` (regfile).

   .. attribute:: suppressed
      :type: bool

      ``True`` → stall; ``False`` → forward bypass mux.


HazardPair
----------

.. class:: zuspec.synth.ir.pipeline_ir.HazardPair

   A detected data hazard.

   **Attributes:**

   .. attribute:: kind
      :type: str

      ``"RAW"``, ``"WAW"``, or ``"WAR"``.

   .. attribute:: producer_stage / consumer_stage
      :type: str

      The stages involved.

   .. attribute:: signal
      :type: str

      Variable name affected.

   .. attribute:: resolved_by
      :type: str

      ``"forward"``, ``"stall"``, or ``"unresolved"``.


RegFileDeclInfo
---------------

.. class:: zuspec.synth.ir.pipeline_ir.RegFileDeclInfo

   Describes an ``IndexedRegFile`` field for SV emission.

   **Attributes:**

   .. attribute:: field_name
      :type: str

      Python class field name (e.g. ``"regfile"``).

   .. attribute:: depth
      :type: int

      Number of entries (e.g. 32).

   .. attribute:: addr_width
      :type: int

      Address bit-width (e.g. 5 for a 32-entry file).

   .. attribute:: data_width
      :type: int

      Data bit-width (e.g. 32).


RegFileAccess
-------------

.. class:: zuspec.synth.ir.pipeline_ir.RegFileAccess

   One ``self.FIELD.read()`` or ``.write()`` call in a stage body.

   **Attributes:**

   .. attribute:: field_name
      :type: str

   .. attribute:: kind
      :type: str

      ``"read"`` or ``"write"``.

   .. attribute:: stage
      :type: str

      Stage name.

   .. attribute:: addr_var
      :type: str

      Address variable name.

   .. attribute:: data_var
      :type: str

      Data variable name (write accesses only; empty for reads).

   .. attribute:: result_var
      :type: str

      Result variable name (read accesses only; empty for writes).


RegFileHazard
-------------

.. class:: zuspec.synth.ir.pipeline_ir.RegFileHazard

   A RAW hazard between a regfile write and a later read.

   **Attributes:**

   .. attribute:: field_name / write_stage / read_stage
      :type: str

   .. attribute:: write_addr_var / read_addr_var
      :type: str

   .. attribute:: write_data_var / read_result_var
      :type: str

   .. attribute:: resolved_by
      :type: str

      ``"forward"``, ``"stall"``, or ``"unresolved"``.

   .. attribute:: suppressed
      :type: bool


Synthesis Passes
================

All passes live in ``zuspec.synth.passes``.

PipelineAnnotationPass
-----------------------

.. class:: zuspec.synth.passes.PipelineAnnotationPass(config)

   **Approach C / Approach A detection pass.**

   Locates the ``@zdc.pipeline``-decorated method on ``ir.component``,
   parses its body with the Python ``ast`` module, and constructs a
   :class:`PipelineIR`.

   For **Approach C** (body contains ``zdc.stage()`` markers):

   * Partitions operations between consecutive markers into named stages.
   * Runs live-variable analysis to determine which variables cross each
     boundary.
   * Creates :class:`ChannelDecl` entries for all live variables.
   * Copies ``zdc.forward()`` / ``zdc.no_forward()`` hints from the decorator
     into ``pip.forwarding``.

   For **Approach A** (no ``zdc.stage()`` markers):

   * Collects all operations into a single flat ``S0`` stage.
   * Sets ``pip.approach = "auto"`` and ``pip.pipeline_stages`` from the
     ``stages=`` decorator argument.
   * Does **not** create channels (deferred to :class:`SDCSchedulePass`).

   :param config: Synthesis configuration.
   :type config: SynthConfig

   .. method:: run(ir: SynthIR) -> SynthIR

      :param ir: Synthesis IR with ``ir.component`` set.
      :type ir: SynthIR
      :return: Updated IR with ``ir.pipeline_ir`` populated.
      :rtype: SynthIR
      :raises PipelineError: If the component has no ``@zdc.pipeline`` method,
          or if the body contains unsupported constructs (e.g. ``zdc.cycles``).


SDCSchedulePass
---------------

.. class:: zuspec.synth.passes.SDCSchedulePass(config)

   **Approach A: SDC-based automatic pipeline scheduling.**

   Builds a dependency graph over the flat ``S0`` stage created by
   :class:`PipelineAnnotationPass`, runs an ASAP Bellman-Ford scheduler,
   and assigns each operation to an optimal pipeline stage.

   If ``pip.pipeline_stages`` is set, the ASAP schedule is folded to fit
   into the requested number of stages using proportional bucketing.

   After rescheduling, :func:`compute_channels_from_stages` is called to
   rebuild :class:`ChannelDecl` entries.  ``pip.approach`` is updated to
   ``"sdc"``.

   This pass is a **no-op** when ``pip.approach != "auto"`` (i.e., Approach C
   pipelines pass through unchanged).

   :param config: Synthesis configuration.
   :type config: SynthConfig

   .. method:: run(ir: SynthIR) -> SynthIR

      :param ir: IR with ``pip.approach == "auto"``.
      :return: Updated IR with stages reassigned and channels populated.
      :rtype: SynthIR


HazardAnalysisPass
------------------

.. class:: zuspec.synth.passes.HazardAnalysisPass(config)

   **Detect data hazards between pipeline stages.**

   Scans the operations of each stage pair to detect:

   * **RAW** hazards (variable defined in stage W, used in earlier stage R).
   * **WAW** hazards (variable written in two stages).
   * **WAR** hazards (variable read in stage R, written in later stage W).
   * **Regfile RAW** hazards (``self.FIELD.read()`` in stage R, ``self.FIELD.write()``
     in later stage W).

   For regfile accesses, also:

   * Populates ``pip.regfile_accesses`` and ``pip.regfile_hazards``.
   * Builds ``pip.regfile_decls`` (one :class:`RegFileDeclInfo` per field).

   :param config: Synthesis configuration.
   :type config: SynthConfig

   .. method:: run(ir: SynthIR) -> SynthIR

      :return: Updated IR with ``pip.hazards``, ``pip.regfile_accesses``,
               ``pip.regfile_hazards``, and ``pip.regfile_decls`` populated.
      :rtype: SynthIR


ForwardingGenPass
-----------------

.. class:: zuspec.synth.passes.ForwardingGenPass(config)

   **Resolve hazards to forwarding muxes or stalls.**

   For each :class:`HazardPair` in ``pip.hazards``:

   1. Check ``pip.forwarding`` for an explicit ``zdc.forward()`` /
      ``zdc.no_forward()`` hint.
   2. Fall back to ``SynthConfig.forward_default``.

   Sets ``hazard.resolved_by`` to ``"forward"`` or ``"stall"`` and updates
   the corresponding :class:`ForwardingDecl` with ``suppressed`` status.

   For :class:`RegFileHazard` entries the same resolution logic applies; the
   signal key is ``"field_name.result_var"`` (e.g. ``"regfile.rdata1"``).

   :param config: Synthesis configuration.
   :type config: SynthConfig

   .. method:: run(ir: SynthIR) -> SynthIR

      :return: Updated IR with resolved hazards and ``pip.forwarding`` finalised.
      :rtype: SynthIR


StallGenPass
------------

.. class:: zuspec.synth.passes.StallGenPass(config)

   **Generate stall-signal and valid-chain descriptors.**

   For each hazard resolved to ``"stall"``, creates a
   ``StallSignal`` descriptor that is consumed by :class:`SVEmitPass` to emit:

   * A ``wire stall_<signal>`` combinational expression.
   * A modified valid-chain update that freezes the appropriate stages.

   :param config: Synthesis configuration.
   :type config: SynthConfig

   .. method:: run(ir: SynthIR) -> SynthIR

      :return: Updated IR with ``pip.stall_signals`` and ``pip.valid_chain``
               populated.
      :rtype: SynthIR


SVEmitPass
----------

.. class:: zuspec.synth.passes.SVEmitPass(config, *, output_path=None, clock_name='clk', reset_name='rst_n', reset_active_low=True)

   **Emit Verilog 2005 RTL from the completed PipelineIR.**

   Delegates to :class:`PipelineSVCodegen` and stores the result in
   ``ir.lowered_sv["pipeline_sv"]``.  Optionally writes to a file.

   This pass is a no-op when ``ir.pipeline_ir`` is ``None``.

   :param config: Synthesis configuration.
   :type config: SynthConfig
   :param output_path: If given, write the generated Verilog to this path.
   :type output_path: str, optional
   :param clock_name: Name for the clock port (default ``"clk"``).
   :type clock_name: str
   :param reset_name: Name for the reset port (default ``"rst_n"``).
   :type reset_name: str
   :param reset_active_low: Whether reset is active-low (default ``True``).
   :type reset_active_low: bool

   .. method:: run(ir: SynthIR) -> SynthIR

      :return: Updated IR with ``ir.lowered_sv["pipeline_sv"]`` set.
      :rtype: SynthIR


PipelineSVCodegen
-----------------

.. class:: zuspec.synth.passes.PipelineSVCodegen

   Low-level Verilog emitter.  Called by :class:`SVEmitPass`; can also be
   used directly when fine-grained control over emission is needed.

   .. method:: emit(pip, *, clock_name='clk', reset_name='rst_n', reset_active_low=True) -> str

      Return a complete Verilog 2005 module string for *pip*.

      :param pip: Completed pipeline IR.
      :type pip: PipelineIR
      :param clock_name: Clock port name.
      :type clock_name: str
      :param reset_name: Reset port name.
      :type reset_name: str
      :param reset_active_low: Active-low reset flag.
      :type reset_active_low: bool
      :return: Verilog 2005 module source.
      :rtype: str


Helper Functions
================

.. function:: zuspec.synth.passes.collect_ports(pip) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]

   Scan all stage operations in *pip* and return ``(inputs, outputs)``, each a
   list of ``(port_name, width_bits)`` tuples.

   ``IndexedRegFile`` fields are excluded from the returned port lists since
   they are inlined as ``reg`` arrays, not module I/O.

   :param pip: Pipeline IR (``regfile_decls`` must be populated).
   :type pip: PipelineIR
   :return: ``(input_ports, output_ports)``
   :rtype: Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]


.. function:: zuspec.synth.passes.pipeline_annotation.compute_channels_from_stages(stage_names, stage_stmts_list, annotation_map) -> List[ChannelDecl]

   Compute inter-stage :class:`ChannelDecl` entries from live-variable analysis.

   This function is also called by :class:`SDCSchedulePass` after rescheduling
   to rebuild channels following a stage-count change.

   :param stage_names: Ordered list of stage names.
   :type stage_names: List[str]
   :param stage_stmts_list: Per-stage AST statement lists (same order as *stage_names*).
   :type stage_stmts_list: List[List[ast.stmt]]
   :param annotation_map: Map of variable name → type annotation AST node.
   :type annotation_map: Dict[str, ast.expr]
   :return: Ordered list of pipeline register channel declarations.
   :rtype: List[ChannelDecl]


.. function:: zuspec.synth.passes.expr_lowerer.is_regfile_read_stmt(stmt) -> bool

   Return ``True`` if *stmt* is a ``var = self.FIELD.read(addr)`` assignment.

   :param stmt: AST statement node.
   :type stmt: ast.stmt
   :rtype: bool


.. function:: zuspec.synth.passes.expr_lowerer.is_regfile_write_stmt(stmt) -> bool

   Return ``True`` if *stmt* is a bare ``self.FIELD.write(addr, data)`` call.

   :param stmt: AST statement node.
   :type stmt: ast.stmt
   :rtype: bool
