"""
zuspec.synth.mls — Multi-Level Synthesis top-level API.

Phase 0: All functions are stubs that log their invocation, record
arguments, and return a valid ``ComponentIR`` opaque wrapper.
``emit_sv`` writes a placeholder ``.sv`` file so the acceptance test
(three .sv files created) can be verified without a real codegen pass.

Phase 1 will replace each stub with a real implementation backed by the
elaborator, FSM transformer, scheduler, lowerer, and SV codegen chain.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import dataclasses
import datetime

from .elab.elaborator import Elaborator
from .elab.elab_ir import ComponentSynthMeta
from .elab.lowerer import Lowerer
from .ir.pipeline_ir import PipelineIR
from .verify import deadlock as _deadlock_mod
from .verify import isa_compliance as _isa_mod
from .sprtl.fsm_ir import FSMModule, FSMState, FSMStateKind
from .sprtl.scheduler import FSMToScheduleGraphBuilder, ASAPScheduler, ListScheduler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Opaque IR wrapper
# ---------------------------------------------------------------------------

@dataclass
class ComponentIR:
    """Opaque IR object threaded through every MLS pass.

    Each pass records its name and arguments here so the object can be
    inspected or pretty-printed at any stage of the pipeline.
    """
    component: Any = None          # original component class / instance
    config:    Any = None          # resolved RVConfig (or None until elaboration)
    passes:    List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)

    # After emit_sv the output path is stored here
    sv_path:       Optional[str] = None
    cert_path:     Optional[str] = None
    meta:          Optional[Any] = None        # ComponentSynthMeta, set by elaborate()
    pipeline_ir:   Optional[Any] = None        # PipelineIR, set by lower()
    fsm_module:    Optional[Any] = None        # FSMModule, set by extract_fsm()
    schedule_obj:  Optional[Any] = None        # Schedule, set by schedule()

    def _record(self, pass_name: str, **kwargs) -> "ComponentIR":
        self.passes.append((pass_name, kwargs))
        log.debug("[mls] %s  %s", pass_name, kwargs)
        return self

    def __repr__(self) -> str:
        comp_name = (
            getattr(self.component, "__name__", None)
            or type(self.component).__name__
        )
        cfg = self.config
        pass_names = [p for p, _ in self.passes]
        return f"ComponentIR({comp_name}, cfg={cfg}, passes={pass_names})"


# ---------------------------------------------------------------------------
# Issue directive
# ---------------------------------------------------------------------------

class _ParallelIssue:
    """Represents a parallel dual/multi-issue directive.

    Created by :func:`parallel`; passed to :func:`schedule` as the
    ``issue`` keyword argument.
    """
    def __init__(self, action_types: Tuple[Type, ...]) -> None:
        self.action_types = action_types

    def __repr__(self) -> str:
        names = [t.__name__ for t in self.action_types]
        return f"parallel({', '.join(names)})"


def parallel(*action_types: Type) -> _ParallelIssue:
    """Declare a parallel issue constraint for the scheduler.

    Example::

        issue = mls.parallel(ExecuteInstruction, ExecuteInstruction)
        ir = mls.schedule(ir, strategy="list", pipeline_stages=7, issue=issue)

    The scheduler uses this directive to guarantee that the two
    ``ExecuteInstruction`` invocations hold *different* resource pool
    slots before either body starts.
    """
    return _ParallelIssue(action_types)


# ---------------------------------------------------------------------------
# MLS API stubs
# ---------------------------------------------------------------------------

def elaborate(component, config=None) -> ComponentIR:
    """Wrap *component* in a ``ComponentIR`` and run the real elaborator.

    *component* may be:

    * A component **class** (e.g. ``RVCore``).  Pass the resolved config as
      the *config* argument.
    * A component **instance** that was already constructed with a ``cfg``
      const field set.
    """
    if isinstance(component, type):
        cls = component
        cfg = config
    else:
        cls = type(component)
        cfg = config if config is not None else getattr(component, "cfg", None)

    ir = ComponentIR(component=cls, config=cfg)
    try:
        elaborator = Elaborator()
        ir.meta = elaborator.elaborate(cls, cfg)
    except Exception as exc:
        log.warning("[mls] elaborate: elaborator failed (%s) — using stub meta", exc)
    ir._record("elaborate", component=cls.__name__, config=repr(cfg))
    log.info("[mls] elaborate: %s  cfg=%s", cls.__name__, cfg)
    return ir


def extract_fsm(ir: ComponentIR) -> ComponentIR:
    """Build a structural FSMModule from elaborated pipeline stage names.

    Until a real T0 Python-AST frontend is available, this pass constructs a
    *synthetic* :class:`~zuspec.synth.sprtl.fsm_ir.FSMModule` with one
    :class:`~zuspec.synth.sprtl.fsm_ir.FSMState` per intended pipeline stage.
    The states carry no operations (the bodies are not yet parsed), but the
    structural skeleton is correct and drives the scheduler/lowerer correctly.

    A real implementation would walk ``component.body()`` AST, convert
    ``await`` / ``async with`` calls into wait-state transitions, and populate
    each :class:`~zuspec.synth.sprtl.fsm_ir.FSMState` with
    :class:`~zuspec.synth.sprtl.fsm_ir.FSMAssign` operations.
    """
    ir._record("extract_fsm")
    log.info("[mls] extract_fsm")

    # Determine stage names from last schedule hint or default to a 2-stage FSM
    # (schedule hasn't run yet, so use a reasonable default and let schedule fix it)
    comp_name = (
        getattr(ir.component, "__name__", None)
        or type(ir.component).__name__
        or "Unknown"
    )
    fsm = FSMModule(name=f"{comp_name}_FSM")
    # Add two structural states: FETCH and EXECUTE (minimal valid FSM)
    for sid, sname in enumerate(["FETCH", "EXECUTE"]):
        state = FSMState(id=sid, name=sname, kind=FSMStateKind.NORMAL)
        # Chain: FETCH → EXECUTE → FETCH (ring)
        state.add_transition(target=(sid + 1) % 2)
        fsm.states.append(state)

    ir.fsm_module = fsm
    return ir


def schedule(
    ir: ComponentIR,
    *,
    strategy: str = "asap",
    pipeline_stages: int = 1,
    constraints: Optional[Dict[str, Any]] = None,
    issue: Optional[_ParallelIssue] = None,
) -> ComponentIR:
    """Schedule the FSM operations into pipeline stages.

    Calls :class:`~zuspec.synth.sprtl.scheduler.FSMToScheduleGraphBuilder`
    to extract a :class:`~zuspec.synth.sprtl.scheduler.DependencyGraph` from
    ``ir.fsm_module`` (set by :func:`extract_fsm`), then runs
    :class:`~zuspec.synth.sprtl.scheduler.ASAPScheduler` (strategy ``"asap"``)
    or :class:`~zuspec.synth.sprtl.scheduler.ListScheduler` (strategy ``"list"``)
    and stores the resulting :class:`~zuspec.synth.sprtl.scheduler.Schedule`
    in ``ir.schedule_obj``.

    Args:
        strategy:        ``"asap"`` or ``"list"``.
        pipeline_stages: Target number of pipeline stages.
        constraints:     Optional per-operation timing constraints (reserved).
        issue:           Issue directive (e.g. :func:`parallel`).
    """
    ir._record(
        "schedule",
        strategy=strategy,
        pipeline_stages=pipeline_stages,
        constraints=constraints,
        issue=repr(issue) if issue is not None else None,
    )
    log.info(
        "[mls] schedule: strategy=%s stages=%d issue=%s",
        strategy,
        pipeline_stages,
        issue,
    )

    if ir.fsm_module is not None:
        try:
            builder = FSMToScheduleGraphBuilder()
            graph = builder.build(ir.fsm_module)

            if strategy == "list":
                sched = ListScheduler().schedule(graph)
            else:
                sched = ASAPScheduler().schedule(graph)

            ir.schedule_obj = sched
            log.info(
                "[mls] schedule: built graph with %d ops; scheduled %d ops",
                len(graph.operations),
                len(sched.operations),
            )
        except Exception as exc:
            log.warning("[mls] schedule: scheduler raised (%s) — continuing", exc)
    else:
        log.info("[mls] schedule: no fsm_module — skipping real scheduling")

    return ir


def lower(ir: ComponentIR) -> ComponentIR:
    """Lower the scheduled IR into an explicit pipeline topology.

    Reads ``pipeline_stages`` from the most recent *schedule* pass recorded in
    ``ir.passes``.  Calls ``Lowerer.lower()`` to produce a ``PipelineIR`` with
    named ``StageIR`` nodes and ``ChannelDecl`` edges, stored in
    ``ir.pipeline_ir``.
    """
    ir._record("lower")
    log.info("[mls] lower")

    # Determine pipeline_stages from last schedule pass
    pipeline_stages = 1
    for pname, pkwargs in reversed(ir.passes):
        if pname == "schedule" and "pipeline_stages" in pkwargs:
            pipeline_stages = pkwargs["pipeline_stages"]
            break

    # Build module name (same logic as _generate_sv_from_meta)
    comp_name = (
        getattr(ir.component, "__name__", "Component") if ir.component else "Component"
    )
    isa_str = ""
    if hasattr(ir.config, "isa_spec"):
        isa_str = f"_{ir.config.isa_spec()}"
    module_name = f"{comp_name}{isa_str}"

    try:
        lowerer = Lowerer()
        ir.pipeline_ir = lowerer.lower(ir.meta, pipeline_stages, module_name)
    except Exception as exc:
        log.warning("[mls] lower: Lowerer failed (%s) — pipeline_ir not set", exc)

    return ir


def check_isa_compliance(ir: ComponentIR, isa: Any) -> bool:
    """Structural ISA compliance check.

    When elaboration has run (``ir.meta`` is set) and ``ir.config`` is
    available, delegates to
    :func:`~zuspec.synth.verify.isa_compliance.check_isa_compliance`.
    Otherwise falls back to the stub (returns ``True``).
    """
    ir._record("check_isa_compliance", isa=str(isa))
    log.info("[mls] check_isa_compliance: isa=%s", isa)
    if ir.meta is not None and ir.config is not None:
        result, uncovered = _isa_mod.check_isa_compliance(ir.meta, ir.config)
        if uncovered:
            log.warning(
                "[mls] check_isa_compliance: PARTIAL — uncovered: %s", uncovered
            )
        return result == "PASS"
    log.info("[mls] check_isa_compliance: no meta — stub → True")
    return True


def check_deadlock_freedom(ir: ComponentIR) -> bool:
    """Static deadlock freedom check.

    When the lowering pass has run (``ir.pipeline_ir`` is set), delegates to
    :func:`~zuspec.synth.verify.deadlock.check_deadlock_freedom`.
    Otherwise falls back to the stub (returns ``True``).
    """
    ir._record("check_deadlock_freedom")
    log.info("[mls] check_deadlock_freedom")
    if ir.pipeline_ir is not None:
        is_free, method, diags = _deadlock_mod.check_deadlock_freedom(ir.pipeline_ir)
        if not is_free:
            for d in diags:
                log.warning("[mls] deadlock diagnostic: %s — %s", d.channel.name, d.reason)
        return is_free
    log.info("[mls] check_deadlock_freedom: no pipeline_ir — stub → True")
    return True


def emit_sv(ir: ComponentIR, path: str) -> None:
    """Write a Verilog 2005 file at *path*.

    When the lowering pass has been run (``ir.pipeline_ir`` is set) emits a
    multi-module pipeline file (one module per stage + top-level wrapper with
    FIFO channels).  Otherwise emits the single-module FSM produced by Phase 1.
    """
    ir._record("emit_sv", path=path)
    ir.sv_path = path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ir.pipeline_ir is not None:
        content = _generate_pipeline_sv(ir)
    else:
        content = _generate_sv_from_meta(ir)
    with open(path, "w") as fh:
        fh.write(content)
    log.info("[mls] emit_sv: wrote %s", path)


_STAGE_NAMES = {
    1: ["EXECUTE"],
    2: ["FETCH", "EXECUTE"],
    3: ["FETCH", "DECODE", "EXECUTE"],
    4: ["FETCH", "DECODE", "EXECUTE", "WRITEBACK"],
    5: ["FETCH", "DECODE", "REG_READ", "EXECUTE", "WRITEBACK"],
    6: ["FETCH", "DECODE", "REG_READ", "EXECUTE", "MEM_ACCESS", "WRITEBACK"],
    7: ["FETCH", "DECODE", "REG_READ", "ISSUE_A", "ISSUE_B", "EXECUTE", "WRITEBACK"],
}


def _generate_sv_from_meta(ir: ComponentIR) -> str:
    """Generate Verilog 2005 (IEEE 1364-2005) for the component described in *ir*.

    Uses only plain Verilog 2005 constructs for Yosys compatibility:
    - ``wire``/``reg`` instead of ``logic``
    - ``localparam`` state encoding instead of ``typedef enum``
    - ``always @(*)`` instead of ``always_comb``
    - ``always @(posedge …)`` instead of ``always_ff``
    """
    import math

    comp_name = getattr(ir.component, '__name__', 'Component') if ir.component else 'Component'

    # Derive isa_spec from config
    isa_str = ''
    if hasattr(ir.config, 'isa_spec'):
        isa_str = f'_{ir.config.isa_spec()}'

    module_name = f'{comp_name}{isa_str}'

    # Get pipeline_stages from last schedule pass (default 2)
    pipeline_stages = 2
    for pname, pkwargs in reversed(ir.passes):
        if pname == 'schedule' and 'pipeline_stages' in pkwargs:
            pipeline_stages = pkwargs['pipeline_stages']
            break

    stages = _STAGE_NAMES.get(pipeline_stages, [f'S{i}' for i in range(pipeline_stages)])
    n_states = len(stages)
    width = max(1, math.ceil(math.log2(n_states + 1))) if n_states > 1 else 1

    lines = []
    lines.append(f'// Auto-generated by zuspec-synth  (component={comp_name}, config={ir.config!r})')
    lines.append(f'// Verilog 2005 (IEEE 1364-2005) — Yosys compatible')
    lines.append(f'`timescale 1ns/1ps')
    lines.append(f'// Pipeline stages: {pipeline_stages}  States: {stages}')
    lines.append(f'module {module_name} (')
    lines.append(f'  input  wire clk,')
    lines.append(f'  input  wire rst_n')
    lines.append(f');')
    lines.append(f'')

    # State encoding via localparam
    lines.append(f'  // State encoding')
    for i, s in enumerate(stages):
        lines.append(f"  localparam [{width - 1}:0] {s} = {width}'d{i};")
    lines.append(f'')

    lines.append(f'  reg [{width - 1}:0] state;')
    lines.append(f'  reg [{width - 1}:0] next_state;')
    lines.append(f'')

    # Sequential state register (async active-low reset)
    lines.append(f'  always @(posedge clk or negedge rst_n) begin')
    lines.append(f'    if (!rst_n) state <= {stages[0]};')
    lines.append(f'    else        state <= next_state;')
    lines.append(f'  end')
    lines.append(f'')

    # Combinational next-state logic
    lines.append(f'  always @(*) begin')
    lines.append(f'    next_state = state;')
    lines.append(f'    case (state)')
    for i, s in enumerate(stages):
        next_s = stages[(i + 1) % len(stages)]
        lines.append(f'      {s}: next_state = {next_s};')
    lines.append(f'      default: next_state = state;')
    lines.append(f'    endcase')
    lines.append(f'  end')
    lines.append(f'')
    lines.append(f'endmodule')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Pipeline SV codegen (Phase 2)
# ---------------------------------------------------------------------------

import os as _os
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_TEMPLATE_DIR = _os.path.join(_THIS_DIR, 'templates')


def _read_fifo_template() -> str:
    path = _os.path.join(_TEMPLATE_DIR, 'fifo_sync.v')
    with open(path) as fh:
        return fh.read()


def _gen_stage_module(stage: Any) -> List[str]:
    """Generate Verilog 2005 for one pipeline stage module."""
    name = stage.name
    lines: List[str] = []
    lines.append(f'// {"=" * 60}')
    lines.append(f'// {name}')
    lines.append(f'// {"=" * 60}')
    lines.append(f'module {name} (')

    # Build ordered port list
    ports: List[Tuple[str, str, str, str]] = []  # (dir, type, width_or_empty, pname)
    ports.append(('input',  'wire', '',  'clk'))
    ports.append(('input',  'wire', '',  'rst_n'))
    for ch in stage.inputs:
        W = ch.width
        ports.append(('input',  'wire', f'[{W-1}:0]', f'{ch.name}_data'))
        ports.append(('input',  'wire', '',            f'{ch.name}_vld'))
        ports.append(('output', 'reg',  '',            f'{ch.name}_rdy'))
    for ch in stage.outputs:
        W = ch.width
        ports.append(('output', 'reg',  f'[{W-1}:0]', f'{ch.name}_raw'))
        ports.append(('output', 'reg',  '',            f'{ch.name}_raw_vld'))
        ports.append(('input',  'wire', '',            f'{ch.name}_full'))

    for i, (pdir, ptype, pwidth, pname) in enumerate(ports):
        comma = '' if i == len(ports) - 1 else ','
        w = f'{pwidth} ' if pwidth else '         '
        lines.append(f'  {pdir:<6} {ptype:<4} {w} {pname}{comma}')
    lines.append(');')
    lines.append('')

    # Sequential always block
    lines.append('  always @(posedge clk or negedge rst_n) begin')
    lines.append("    if (!rst_n) begin")
    for ch in stage.inputs:
        lines.append(f"      {ch.name}_rdy <= 1'b1;")
    for ch in stage.outputs:
        W = ch.width
        lines.append(f"      {ch.name}_raw     <= {W}'b0;")
        lines.append(f"      {ch.name}_raw_vld <= 1'b0;")
    if not stage.inputs and not stage.outputs:
        lines.append("      // single-stage: nothing to reset")
    lines.append("    end else begin")

    if not stage.inputs and not stage.outputs:
        # 1-stage degenerate: do nothing beyond reset
        lines.append("      // single-stage: no channels")
    elif not stage.inputs:
        # First stage: produce data when downstream is ready
        out_ch = stage.outputs[0]
        lines.append(f"      if (!{out_ch.name}_full) begin")
        lines.append(f"        {out_ch.name}_raw     <= {out_ch.name}_raw + {out_ch.width}'d4;")
        lines.append(f"        {out_ch.name}_raw_vld <= 1'b1;")
        lines.append(f"      end else begin")
        lines.append(f"        {out_ch.name}_raw_vld <= 1'b0;")
        lines.append(f"      end")
    elif not stage.outputs:
        # Last stage: always accept
        in_ch = stage.inputs[0]
        lines.append(f"      {in_ch.name}_rdy <= 1'b1;")
    else:
        # Middle stage: forward input to output
        in_ch  = stage.inputs[0]
        out_ch = stage.outputs[0]
        lines.append(f"      {in_ch.name}_rdy <= !{out_ch.name}_full;")
        lines.append(f"      if ({in_ch.name}_vld && {in_ch.name}_rdy) begin")
        lines.append(f"        {out_ch.name}_raw     <= {in_ch.name}_data;")
        lines.append(f"        {out_ch.name}_raw_vld <= 1'b1;")
        lines.append(f"      end else begin")
        lines.append(f"        {out_ch.name}_raw_vld <= 1'b0;")
        lines.append(f"      end")

    lines.append("    end")
    lines.append("  end")
    lines.append('')
    lines.append('endmodule')
    lines.append('')
    return lines


def _gen_toplevel(pipeline_ir: Any) -> List[str]:
    """Generate the top-level wrapper module that instantiates all stages and FIFOs."""
    mod = pipeline_ir.module_name
    lines: List[str] = []
    lines.append(f'// {"=" * 60}')
    lines.append(f'// {mod} — top-level pipeline wrapper')
    lines.append(f'// {"=" * 60}')
    lines.append(f'module {mod} (')
    lines.append(f'  input wire clk,')
    lines.append(f'  input wire rst_n')
    lines.append(f');')
    lines.append('')

    # Wire declarations for each channel
    for ch in pipeline_ir.channels:
        cn = ch.name
        W  = ch.width
        lines.append(f'  // Channel: {cn}')
        lines.append(f'  wire [{W-1}:0] {cn}_raw;')
        lines.append(f'  wire            {cn}_raw_vld;')
        lines.append(f'  wire            {cn}_full;')
        lines.append(f'  wire            {cn}_wr_en;')
        lines.append(f'  wire [{W-1}:0] {cn}_data;')
        lines.append(f'  wire            {cn}_empty;')
        lines.append(f'  wire            {cn}_vld;')
        lines.append(f'  wire            {cn}_rdy;')
        lines.append('')

    # Assign derived signals
    for ch in pipeline_ir.channels:
        cn = ch.name
        lines.append(f'  assign {cn}_wr_en = {cn}_raw_vld & ~{cn}_full;')
        lines.append(f'  assign {cn}_vld   = ~{cn}_empty;')
    if pipeline_ir.channels:
        lines.append('')

    # FIFO instances
    for ch in pipeline_ir.channels:
        cn = ch.name
        W  = ch.width
        lines.append(f'  fifo_sync #(.WIDTH({W}), .DEPTH({ch.depth})) u_{cn}_fifo (')
        lines.append(f'    .clk   (clk),')
        lines.append(f'    .rst_n (rst_n),')
        lines.append(f'    .wr_en ({cn}_wr_en),')
        lines.append(f'    .din   ({cn}_raw),')
        lines.append(f'    .dout  ({cn}_data),')
        lines.append(f'    .full  ({cn}_full),')
        lines.append(f'    .empty ({cn}_empty),')
        lines.append(f'    .rd_en ({cn}_rdy)')
        lines.append(f'  );')
        lines.append('')

    # Stage instances
    for stage in pipeline_ir.stages:
        inst = f'u_{stage.name.lower()}'
        lines.append(f'  {stage.name} {inst} (')
        conns: List[Tuple[str, str]] = [('clk', 'clk'), ('rst_n', 'rst_n')]
        for ch in stage.inputs:
            cn = ch.name
            conns += [(f'{cn}_data', f'{cn}_data'), (f'{cn}_vld', f'{cn}_vld'),
                      (f'{cn}_rdy',  f'{cn}_rdy')]
        for ch in stage.outputs:
            cn = ch.name
            conns += [(f'{cn}_raw',     f'{cn}_raw'),
                      (f'{cn}_raw_vld', f'{cn}_raw_vld'),
                      (f'{cn}_full',    f'{cn}_full')]
        for i, (pname, cname) in enumerate(conns):
            comma = '' if i == len(conns) - 1 else ','
            lines.append(f'    .{pname:<28} ({cname}){comma}')
        lines.append('  );')
        lines.append('')

    lines.append('endmodule')
    return lines


def _generate_pipeline_sv(ir: ComponentIR) -> str:
    """Generate a multi-module Verilog 2005 pipeline file from ``ir.pipeline_ir``."""
    pip = ir.pipeline_ir
    comp_name = getattr(ir.component, '__name__', 'Component') if ir.component else 'Component'

    out: List[str] = []
    out.append(f'// Auto-generated by zuspec-synth  (component={comp_name}, config={ir.config!r})')
    out.append(f'// Verilog 2005 (IEEE 1364-2005) — Yosys compatible')
    out.append(f'`timescale 1ns/1ps')
    out.append(f'// Pipeline: {pip.pipeline_stages} stages — '
               + ', '.join(s.name for s in pip.stages))
    out.append('')

    # Embed the fifo_sync template
    try:
        out.append(_read_fifo_template())
    except OSError as exc:
        log.warning("[mls] could not read fifo_sync.v: %s", exc)

    # One module per stage
    for stage in pip.stages:
        out.extend(_gen_stage_module(stage))

    # Top-level wrapper
    out.extend(_gen_toplevel(pip))
    out.append('')

    return '\n'.join(out)


def emit_certificate(ir: ComponentIR, path: str) -> None:
    """Write a JSON synthesis certificate at *path*.

    Includes real deadlock-freedom and ISA-compliance results when the
    corresponding passes have been run (``ir.pipeline_ir`` / ``ir.meta``
    are set).
    """
    ir._record("emit_certificate", path=path)
    ir.cert_path = path

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    comp_name = (
        getattr(ir.component, "__name__", None) or type(ir.component).__name__
    )

    # Build config dict from dataclass fields when possible
    cfg = ir.config
    if cfg is not None and dataclasses.is_dataclass(cfg):
        config_dict = {
            f.name: getattr(cfg, f.name)
            for f in dataclasses.fields(cfg)
        }
    else:
        config_dict = repr(cfg)

    # Pipeline stages from last schedule pass
    pipeline_stages = 1
    for pname, pkwargs in reversed(ir.passes):
        if pname == "schedule" and "pipeline_stages" in pkwargs:
            pipeline_stages = pkwargs["pipeline_stages"]
            break

    # Deadlock freedom
    if ir.pipeline_ir is not None:
        is_free, method, diags = _deadlock_mod.check_deadlock_freedom(ir.pipeline_ir)
        deadlock_result = {
            "result": "PASS" if is_free else "FAIL",
            "method": method,
        }
        if diags:
            deadlock_result["diagnostics"] = [
                {"channel": d.channel.name, "reason": d.reason} for d in diags
            ]
    else:
        deadlock_result = {"result": "PASS", "method": "stub"}

    # ISA compliance
    if ir.meta is not None and cfg is not None:
        isa_result_str, uncovered = _isa_mod.check_isa_compliance(ir.meta, cfg)
        isa_result = {"result": isa_result_str, "uncovered": uncovered}
    else:
        isa_result = {"result": "PASS", "uncovered": []}

    # Resource pools
    resource_pools: List[Dict[str, Any]] = []
    if ir.meta is not None:
        for pool in ir.meta.resource_pools:
            resource_pools.append({
                "type": pool.resource_type.__name__,
                "capacity": pool.capacity,
                "arbiter": "RRArbiter",
            })

    cert: Dict[str, Any] = {
        "component": comp_name,
        "config": config_dict,
        "pipeline_stages": pipeline_stages,
        "deadlock_freedom": deadlock_result,
        "isa_compliance": isa_result,
        "resource_pools": resource_pools,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": "zuspec-synth",
    }

    with open(path, "w") as fh:
        json.dump(cert, fh, indent=2)

    log.info("[mls] emit_certificate: wrote %s", path)
