"""SynthIR and SynthConfig — synthesis-time IR and configuration."""
from __future__ import annotations

import dataclasses as dc
from typing import Any, Dict, List, Optional


@dc.dataclass
class SynthConfig:
    """Synthesis-time configuration passed to every ``SynthPass``.

    Args:
        xlen: Instruction/data width in bits (32 or 64).
        reset_addr: Reset vector address.
        pipeline_stages: Number of pipeline stages to generate.
        strategy: Default scheduling strategy (``"asap"`` or ``"list"``).
        module_prefix: Optional prefix prepended to all emitted module names.
    """

    xlen: int = dc.field(default=32)
    reset_addr: int = dc.field(default=0)
    pipeline_stages: int = dc.field(default=2)
    strategy: str = dc.field(default="asap")
    module_prefix: str = dc.field(default="")


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
        decode_node: ``DecodeActionNode`` set by ``RVIdentifyPass``; carries the
            primary decode action class for ``DecodeFieldExtractor``.
        decode_c_node: ``DecodeConstraintNode`` set by ``RVIdentifyPass``; carries
            the constraint-compiler-compatible decode class.
        execute_node: ``ExecuteActionNode`` set by ``RVIdentifyPass``; carries the
            execute-dispatch action class for ``ActionBodySynthesizer``.
        domain_nodes: Active ``DomainNode`` instances introduced by
            ``RVStageIntroducePass``.  Lowering passes remove nodes from this
            list as they are lowered.  ``SVEmitPass`` raises if any remain.
        lowered_sv: SV string fragments produced by lowering passes.  Keys are
            prefixed by stage (``'d_'`` for decode, ``'e_'`` for execute) and
            named by content (e.g. ``'d_field_wires'``, ``'e_body_synth'``).
        stage_sv: Complete per-stage SV module lines produced by
            ``RVStageGeneratePass`` (or any plugin that generates stage RTL).
            Keys are stage names (e.g. ``'FetchStage'``); values are lists of
            SV text lines.  ``_generate_pipeline_sv`` prefers this dict over
            its built-in generators.
    """

    component: Optional[Any] = dc.field(default=None)
    config: Optional[Any] = dc.field(default=None)
    meta: Optional[Any] = dc.field(default=None)
    pipeline_ir: Optional[Any] = dc.field(default=None)
    fsm_module: Optional[Any] = dc.field(default=None)
    schedule_obj: Optional[Any] = dc.field(default=None)
    rtl_modules: List[Any] = dc.field(default_factory=list)
    sv_path: Optional[str] = dc.field(default=None)
    cert_path: Optional[str] = dc.field(default=None)
    decode_node: Optional[Any] = dc.field(default=None)
    decode_c_node: Optional[Any] = dc.field(default=None)
    execute_node: Optional[Any] = dc.field(default=None)
    domain_nodes: List[Any] = dc.field(default_factory=list)
    lowered_sv: Dict[str, Any] = dc.field(default_factory=dict)
    stage_sv: Dict[str, List[str]] = dc.field(default_factory=dict)
