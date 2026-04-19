"""SynthIR and SynthConfig — synthesis-time IR and configuration."""
from __future__ import annotations

import dataclasses as dc
import re
from typing import Any, Dict, List, Optional

from zuspec.synth.ir.protocol_ir import (  # noqa: F401 — re-exported for convenience
    IfProtocolPortIR,
    QueueIR,
    SpawnIR,
    SelectIR,
    CompletionIR,
)


# ---------------------------------------------------------------------------
# lowered_sv key convention
# ---------------------------------------------------------------------------

_LOWERED_SV_KEY_RE = re.compile(
    r'^[a-z][a-z0-9]*/[a-z][a-z0-9_]*/[a-zA-Z_][a-zA-Z0-9_./-]*$'
)


def validate_lowered_sv_key(key: str) -> None:
    """Raise ``ValueError`` if *key* does not follow the ``lowered_sv`` naming convention.

    All keys stored in :attr:`SynthIR.lowered_sv` must use the hierarchical
    format ``<backend>/<category>/<item>``:

    - ``backend``: lower-case ``zuspec-be-*`` suffix (e.g. ``sv``, ``sw``, ``fv``, ``trace``).
    - ``category``: structural grouping (e.g. ``pipeline``, ``stage``, ``regfile``, ``module``).
    - ``item``: specific artefact name (e.g. ``top``, ``FetchStage``, ``fifo_sync``).

    Call this during development and testing.  It is **not** called automatically on the
    hot synthesis path.

    Args:
        key: The ``lowered_sv`` dict key to validate.

    Raises:
        ValueError: If *key* does not match ``<backend>/<category>/<item>``.

    Examples::

        validate_lowered_sv_key("sv/pipeline/top")        # OK
        validate_lowered_sv_key("sw/module/bus_bridge")   # OK
        validate_lowered_sv_key("pipeline_sv")            # ValueError
    """
    if not _LOWERED_SV_KEY_RE.match(key):
        raise ValueError(
            f"lowered_sv key {key!r} does not follow the '<backend>/<category>/<item>' "
            "convention (e.g. 'sv/pipeline/top').  See SynthIR.lowered_sv docstring."
        )


class ScheduleConstraintError(Exception):
    """Raised when a ``fixed_assignments`` constraint in :class:`SchedulePass` is infeasible.

    An assignment is infeasible when the requested stage index falls outside the
    ``[earliest_stage, latest_stage]`` window computed by ASAP/ALAP analysis, meaning
    a data-dependency chain prevents the operation from being placed there.

    Args:
        op_id: The :attr:`~zuspec.synth.sprtl.scheduler.ScheduledOperation.id` of the
            operation that could not be pinned.
        state_id: The FSM state ID used as the lookup key in ``fixed_assignments``.
        requested_stage: The stage index requested by the caller.
        earliest_stage: The earliest feasible stage (ASAP bound).
        latest_stage: The latest feasible stage (ALAP bound).
    """

    def __init__(
        self,
        op_id: int,
        state_id: int,
        requested_stage: int,
        earliest_stage: int,
        latest_stage: int,
    ) -> None:
        super().__init__(
            f"fixed_assignments: op {op_id} (state_id={state_id}) cannot be pinned to "
            f"stage {requested_stage} — feasible window is "
            f"[{earliest_stage}, {latest_stage}]"
        )
        self.op_id = op_id
        self.state_id = state_id
        self.requested_stage = requested_stage
        self.earliest_stage = earliest_stage
        self.latest_stage = latest_stage


@dc.dataclass
class SynthConfig:
    """Synthesis-time configuration passed to every ``SynthPass``.

    Args:
        xlen: Instruction/data width in bits (32 or 64).
        reset_addr: Reset vector address.
        pipeline_stages: Number of pipeline stages to generate.
        strategy: Default scheduling strategy (``"asap"`` or ``"list"``).
        module_prefix: Optional prefix prepended to all emitted module names.
        forward_default: Process-level default for unresolved RAW hazards in
            ``@zdc.pipeline`` processes.  ``None`` requires explicit
            declarations; ``True`` auto-forwards; ``False`` auto-stalls.
        latency_model: Mapping from operation type name (e.g. ``"MUL"``) to
            latency in clock cycles.  Used by the SDC scheduler (Approach A).
            If empty, ``SDCScheduler.DEFAULT_LATENCY`` is used.
        clock_period_ns: Target clock period in nanoseconds.  Used by the SDC
            scheduler to check whether a stage's combinational depth fits in
            one clock cycle.
    """

    xlen: int = dc.field(default=32)
    reset_addr: int = dc.field(default=0)
    pipeline_stages: int = dc.field(default=2)
    strategy: str = dc.field(default="asap")
    module_prefix: str = dc.field(default="")
    forward_default: Optional[bool] = dc.field(default=None)
    latency_model: Dict[str, int] = dc.field(default_factory=dict)
    clock_period_ns: float = dc.field(default=10.0)


@dc.dataclass
class SynthIR:
    """Top-level synthesis IR object threaded through every pass.

    Each pass reads and/or populates fields.  The field set is a superset of
    what the old ``ComponentIR`` carried so that the existing private helpers
    in ``mls.py`` work via duck-typing.

    Args:
        component: The top-level component *class* being synthesised.
        config: Component-level configuration (e.g. ``RVConfig``).
        meta: Elaboration metadata (``ComponentSynthMeta``), set by ``ElaboratePass``.
        pipeline_ir: Explicit pipeline topology (``PipelineIR``), set by ``LowerPass``.
        fsm_module: FSM skeleton (``FSMModule``), set by ``FSMExtractPass``.
        schedule_obj: Scheduled operation graph, set by ``SchedulePass``.
        rtl_modules: Generic RTL modules (future use).
        sv_path: Path of the last emitted ``.sv`` file.
        cert_path: Path of the last emitted ``.cert.json`` file.
        decode_cls: The primary decode action class, set by ``RVSynthPass``.
            Read by ``RVStageGeneratePass`` to emit decode-stage logic.
        decode_c_cls: The constraint-compiler-compatible decode class, set by
            ``RVSynthPass``.  Read by ``RVStageGeneratePass``.
        execute_cls: The execute-dispatch action class, auto-discovered and set
            by ``RVSynthPass``.  Read by ``RVStageGeneratePass``.
        lowered_sv: SV string fragments produced by lowering passes.  Keys are
            prefixed by stage (``'d_'`` for decode, ``'e_'`` for execute) and
            named by content (e.g. ``'d_field_wires'``, ``'e_body_synth'``).
        stage_sv: Complete per-stage SV module lines produced by
            ``RVStageGeneratePass`` (or any plugin that generates stage RTL).
            Keys are stage names (e.g. ``'FetchStage'``); values are lists of
            SV text lines.  ``_generate_pipeline_sv`` prefers this dict over
            its built-in generators.
        model_context: ``zuspec.dataclasses`` ``Context`` produced by
            ``DataModelFactory``, set by ``ElaboratePass``.  Contains parsed
            ``DataTypeComponent`` / ``DataTypeAction`` IR for every type
            reachable from the top-level component class, including fully
            parsed ``@zdc.proc`` body statement trees.  Read by
            ``FSMExtractPass`` to build a real operation graph.
    """

    component: Optional[Any] = dc.field(default=None)
    config: Optional[Any] = dc.field(default=None)
    meta: Optional[Any] = dc.field(default=None)
    pipeline_ir: Optional[Any] = dc.field(default=None)
    async_pipeline_ir: Optional[Any] = dc.field(default=None)
    """IrPipeline extracted by AsyncPipelineElaboratePass.

    Populated from a ``@zdc.pipeline`` async method; consumed by
    ``AsyncPipelineToIrPass`` which converts it to a ``PipelineIR``.
    Set to ``None`` when the component has no async pipeline method.
    """
    fsm_module: Optional[Any] = dc.field(default=None)
    schedule_obj: Optional[Any] = dc.field(default=None)
    rtl_modules: List[Any] = dc.field(default_factory=list)
    sv_path: Optional[str] = dc.field(default=None)
    cert_path: Optional[str] = dc.field(default=None)
    decode_cls: Optional[Any] = dc.field(default=None)
    decode_c_cls: Optional[Any] = dc.field(default=None)
    execute_cls: Optional[Any] = dc.field(default=None)
    lowered_sv: Dict[str, Any] = dc.field(default_factory=dict)
    """SV/SW/FV string fragments produced by lowering passes.

    **Key naming convention** (mandatory for all passes)::

        <backend>/<category>/<item>

    where ``backend`` is the ``zuspec-be-*`` suffix (``sv``, ``sw``, ``fv``, ``trace``),
    ``category`` is a structural grouping (``pipeline``, ``stage``, ``regfile``, ``module``),
    and ``item`` is the specific artefact name.

    Use :func:`validate_lowered_sv_key` in tests to verify conformance.

    **Registered keys:**

    +----------------------+----------------+-------------------------------------------+
    | Key                  | Producer pass  | Description                               |
    +======================+================+===========================================+
    | ``sv/pipeline/top``  | ``SVEmitPass`` | Full pipeline SystemVerilog text          |
    +----------------------+----------------+-------------------------------------------+

    Passes that add new keys **must** register them in the table above.  Experimental or
    private keys must use a ``_`` prefix in the item segment
    (e.g. ``sv/stage/_debug``).
    """
    stage_sv: Dict[str, List[str]] = dc.field(default_factory=dict)
    model_context: Optional[Any] = dc.field(default=None)

    # --- New Phase 5 fields -------------------------------------------------

    protocol_ports: List[IfProtocolPortIR] = dc.field(default_factory=list)
    """IfProtocol port/export nodes produced by ``IfProtocolLowerPass``."""

    queue_nodes: List[QueueIR] = dc.field(default_factory=list)
    """Queue synthesis IR nodes produced by ``QueueLowerPass``."""

    spawn_nodes: List[SpawnIR] = dc.field(default_factory=list)
    """Spawn synthesis IR nodes produced by ``SpawnLowerPass``."""

    select_nodes: List[SelectIR] = dc.field(default_factory=list)
    """Select synthesis IR nodes produced by ``SelectLowerPass``."""

    completion_nodes: List[CompletionIR] = dc.field(default_factory=list)
    """Analyzed Completion tokens produced by ``CompletionAnalysisPass``."""
