"""AsyncPipelineToIrPass — translate IrPipeline to PipelineIR.

Input:  ``SynthIR.async_pipeline_ir`` (an :class:`~zuspec.ir.core.pipeline_async.IrPipeline`)
Output: ``SynthIR.pipeline_ir``   (a :class:`~zuspec.synth.ir.pipeline_ir.PipelineIR`)

The pass performs:
1. Def-use analysis across stage bodies to detect cross-stage data flow.
2. ChannelDecl generation for each cross-stage variable (with auto-threading
   so variables that skip stages get pipeline registers at each boundary).
3. StageIR construction from IrStage.
4. RegFileAccess construction from IrHazardOp (reserve/block/write/release).
5. PipelineIR assembly.

The resulting PipelineIR is then consumed by the existing shared passes:
``HazardAnalysisPass → ForwardingGenPass → StallGenPass → SVEmitPass``.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from zuspec.ir.core.pipeline_async import (
    IrBubble, IrEgressOp, IrHazardOp, IrIngressOp, IrPipeline, IrStage, IrStall
)

from zuspec.synth.ir.pipeline_ir import (
    ChannelDecl,
    HazardPair,
    PipelineIR,
    RegFileAccess,
    RegFileDeclInfo,
    StageIR,
)
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR
from .synth_pass import SynthPass

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Width inference from AST annotation nodes
# ---------------------------------------------------------------------------

_ZDC_WIDTHS: Dict[str, int] = {
    "bit": 1, "u1": 1, "u2": 2, "u3": 3, "u4": 4, "u5": 5, "u6": 6, "u7": 7,
    "u8": 8, "u16": 16, "u32": 32, "u64": 64, "u128": 128,
    "i8": 8, "i16": 16, "i32": 32, "i64": 64,
}

_REVERSE_WIDTH_MAP: Dict[int, str] = {
    1: "bit", 8: "u8", 16: "u16", 32: "u32", 64: "u64", 128: "u128",
}


def _width_to_type_name(width: int) -> str:
    """Return the zdc type name for a given bit-width (e.g. 8 → 'u8')."""
    return _REVERSE_WIDTH_MAP.get(width, f"u{width}")


def _width_from_ast_ann(ann: ast.expr, default: int = 32) -> int:
    """Return bit-width for a zdc type annotation AST node (e.g. ``zdc.u32`` → 32)."""
    attr: Optional[str] = None
    if isinstance(ann, ast.Attribute):
        attr = ann.attr
    elif isinstance(ann, ast.Name):
        attr = ann.id
    if attr:
        if attr in _ZDC_WIDTHS:
            return _ZDC_WIDTHS[attr]
        if attr.startswith(("u", "i")):
            try:
                return int(attr[1:])
            except ValueError:
                pass
    return default


def _egress_assign_from_await(node: ast.expr, egress_port_names: Set[str]) -> Optional[ast.Assign]:
    """If *node* is ``await self.PORT.put(expr)`` for a known egress port, return
    a synthetic ``self.PORT = expr`` Assign node, else None.

    If *egress_port_names* contains ``"*"``, matches any port (used for discovery).
    """
    if not isinstance(node, ast.Await):
        return None
    call = node.value
    if not isinstance(call, ast.Call):
        return None
    fn = call.func
    if not (isinstance(fn, ast.Attribute) and fn.attr == "put"):
        return None
    obj = fn.value
    if not (isinstance(obj, ast.Attribute)
            and isinstance(obj.value, ast.Name)
            and obj.value.id == "self"):
        return None
    port_name = obj.attr
    if "*" not in egress_port_names and port_name not in egress_port_names:
        return None
    value_expr = call.args[0] if call.args else ast.Constant(value=0)
    synth = ast.Assign(
        targets=[ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=port_name,
            ctx=ast.Store(),
        )],
        value=value_expr,
        lineno=0, col_offset=0,
    )
    ast.fix_missing_locations(synth)
    return synth


def _convert_egress_in_stmts(stmts: List, egress_port_names: Set[str]) -> List:
    """Recursively rewrite ``await self.PORT.put(expr)`` inside statement lists
    (including nested ast.If branches) into ``self.PORT = expr`` assignments.

    This handles the conditional-egress pattern:
        if cond:
            await self.out_port.put(result)
    → (after rewrite)
        if cond:
            self.out_port = result
    """
    out: List = []
    for stmt in stmts:
        if isinstance(stmt, ast.Expr):
            assign = _egress_assign_from_await(stmt.value, egress_port_names)
            if assign is not None:
                out.append(assign)
                continue
        if isinstance(stmt, ast.If):
            new_body = _convert_egress_in_stmts(stmt.body, egress_port_names)
            new_else = _convert_egress_in_stmts(stmt.orelse, egress_port_names)
            new_if = ast.If(
                test=stmt.test,
                body=new_body or [ast.Pass()],
                orelse=new_else,
            )
            ast.copy_location(new_if, stmt)
            ast.fix_missing_locations(new_if)
            out.append(new_if)
            continue
        out.append(stmt)
    return out



# ---------------------------------------------------------------------------
# Def-use analysis
# ---------------------------------------------------------------------------

@dataclass
class _VarInfo:
    """Information about one pipeline variable."""
    def_stage: int           # index of the stage that defines this variable
    width: int               # bit-width
    use_stages: List[int] = field(default_factory=list)  # stages that use it


class _DefUseAnalyzer:
    """Collects def/use info from pipeline stage bodies.

    For each stage body (a list of :class:`IrHazardOp`, :class:`IrBubble`,
    :class:`IrStall`, and :class:`ast.stmt` nodes), this class:
    - Detects *definitions*: ``ast.AnnAssign``, ``ast.Assign``, and
      :class:`IrHazardOp` with a ``result_var``.
    - Detects *uses*: ``ast.Name`` nodes in Load context that were previously
      defined by another stage.
    """

    def __init__(self, xlen: int = 32) -> None:
        self._xlen = xlen
        self.vars: Dict[str, _VarInfo] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_stage(self, stage_idx: int, body: List[Any]) -> None:
        """Process one stage's body nodes, recording defs then uses."""
        # Pass 1: collect definitions
        for node in body:
            self._collect_def(node, stage_idx)
        # Pass 2: collect uses (cross-stage only)
        for node in body:
            self._collect_uses(node, stage_idx)

    # ------------------------------------------------------------------
    # Definition collection
    # ------------------------------------------------------------------

    def _collect_def(self, node: Any, stage_idx: int) -> None:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            width = _width_from_ast_ann(node.annotation) if node.annotation else self._xlen
            self._record_def(name, stage_idx, width)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self._record_def(t.id, stage_idx, self._xlen)
        elif isinstance(node, IrHazardOp) and node.result_var:
            # e.g. v1: zdc.u32 = await zdc.pipeline.block(...)
            self._record_def(node.result_var, stage_idx, node.result_width)

    def _record_def(self, name: str, stage_idx: int, width: int) -> None:
        if name not in self.vars:
            self.vars[name] = _VarInfo(def_stage=stage_idx, width=width)

    # ------------------------------------------------------------------
    # Use collection
    # ------------------------------------------------------------------

    def _collect_uses(self, node: Any, stage_idx: int) -> None:
        if isinstance(node, ast.stmt):
            self._walk_for_uses(node, stage_idx, skip_annotation=True)
        elif isinstance(node, IrHazardOp):
            for expr in [node.resource_expr, node.value_expr]:
                if expr is not None:
                    self._walk_for_uses(expr, stage_idx, skip_annotation=False)
        elif isinstance(node, IrEgressOp) and node.value_expr is not None:
            # ``await self.PORT.put(expr)`` — expr may reference cross-stage vars
            self._walk_for_uses(node.value_expr, stage_idx, skip_annotation=False)

    def _walk_for_uses(self, node: Any, stage_idx: int, skip_annotation: bool) -> None:
        """Walk AST, recording cross-stage uses of known variables."""
        if skip_annotation and isinstance(node, ast.AnnAssign):
            # Walk only target and value, not annotation
            parts: List[Any] = [node.target]
            if node.value:
                parts.append(node.value)
            for p in parts:
                for n in ast.walk(p):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        self._record_use(n.id, stage_idx)
        else:
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    self._record_use(n.id, stage_idx)

    def _record_use(self, name: str, stage_idx: int) -> None:
        if name not in self.vars:
            return
        info = self.vars[name]
        if info.def_stage == stage_idx:
            return  # same-stage use, not cross-stage
        if stage_idx not in info.use_stages:
            info.use_stages.append(stage_idx)


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

class AsyncPipelineToIrPass(SynthPass):
    """Convert :class:`IrPipeline` to :class:`PipelineIR`.

    The resulting ``PipelineIR`` is compatible with the shared passes
    (``HazardAnalysisPass``, ``ForwardingGenPass``, ``StallGenPass``, ``SVEmitPass``).
    """

    @property
    def name(self) -> str:
        return "AsyncPipelineToIrPass"

    def run(self, ir: SynthIR) -> SynthIR:
        ip: Optional[IrPipeline] = ir.async_pipeline_ir
        if ip is None:
            _log.warning("AsyncPipelineToIrPass: no async_pipeline_ir on SynthIR; skipping")
            return ir

        cfg = self._config
        comp_cls = ir.component

        # 0. Extract port widths early so the analyzer can use them for ingress vars.
        port_widths = self._extract_port_widths(comp_cls)

        # 1. Def-use analysis
        analyzer = _DefUseAnalyzer(xlen=cfg.xlen)
        # Register ingress vars as "defined at stage 0" so the channel generator
        # creates FETCH→EXEC (etc.) pipeline registers to carry them forward.
        for op in ip.ingress_ops:
            if op.result_var:
                w = port_widths.get(op.port_name, op.width if op.width else cfg.xlen)
                analyzer._record_def(op.result_var, 0, w)
        # Also handle ingress ops embedded inside stage 0's body
        if ip.stages:
            for node in ip.stages[0].body:
                if isinstance(node, IrIngressOp) and node.result_var:
                    w = port_widths.get(node.port_name, node.width if node.width else cfg.xlen)
                    analyzer._record_def(node.result_var, 0, w)
        for i, stage in enumerate(ip.stages):
            analyzer.process_stage(i, stage.body)

        stage_names = [s.name for s in ip.stages]

        # 2. Generate ChannelDecls (with auto-threading for skip stages)
        channels, stage_inputs, stage_outputs = self._gen_channels(
            analyzer, stage_names
        )

        # 3. Build annotation map (var → ast.expr annotation node for ExprLowerer)
        annotation_map: Dict[str, Any] = {}
        for info_name, info in analyzer.vars.items():
            annotation_map[info_name] = ast.Attribute(
                value=ast.Name(id="zdc", ctx=ast.Load()),
                attr=_width_to_type_name(info.width),
                ctx=ast.Load(),
            )

        # Build egress port name set for conditional egress conversion.
        # We can detect OutPort fields from the component class annotations.
        egress_port_name_set: Set[str] = set()
        if comp_cls is not None:
            try:
                import typing
                hints = typing.get_type_hints(comp_cls)
            except Exception:
                hints = getattr(comp_cls, "__annotations__", {})
            for field_name, type_hint in hints.items():
                origin = getattr(type_hint, "__origin__", None)
                if origin is None:
                    # Check string name for OutPort
                    tname = getattr(type_hint, "__name__", "") or str(type_hint)
                    if "OutPort" in tname:
                        egress_port_name_set.add(field_name)
                else:
                    tname = getattr(origin, "__name__", "") or str(origin)
                    if "OutPort" in tname:
                        egress_port_name_set.add(field_name)
        # Also collect from IrEgressOp nodes (covers any detected egress)
        for iop in ip.egress_ops:
            egress_port_name_set.add(iop.port_name)
        for stage in ip.stages:
            for n in stage.body:
                if isinstance(n, IrEgressOp):
                    egress_port_name_set.add(n.port_name)

        def _scan_for_egress_ports(stmts):
            """Walk statement lists to find port names used in put() calls."""
            for stmt in stmts:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Await):
                    assign = _egress_assign_from_await(stmt.value, {"*"})
                    if assign is not None:
                        egress_port_name_set.add(assign.targets[0].attr)
                elif isinstance(stmt, ast.If):
                    _scan_for_egress_ports(stmt.body)
                    _scan_for_egress_ports(stmt.orelse)

        for stage in ip.stages:
            _scan_for_egress_ports(stage.body)

        # 4. Build StageIR list
        sir_list: List[StageIR] = []
        for i, ir_stage in enumerate(ip.stages):
            sn = ir_stage.name
            # operations: AST stmt nodes + IrEgressOp converted to self.PORT = expr stmts
            ops = []

            # 4a-pre: For stage 0, prepend synthetic capture AnnAssigns for
            # top-level ingress ops (``a = await self.a_in.get()``).  This
            # makes ExprLowerer emit ``wire [31:0] a_fetch; assign a_fetch = a_in;``
            # so the outgoing channel register captures the live port value.
            if i == 0:
                for iop in ip.ingress_ops:
                    if not iop.result_var:
                        continue
                    info = analyzer.vars.get(iop.result_var)
                    if info is None or not info.use_stages:
                        continue  # var not used downstream — no channel needed
                    width = port_widths.get(iop.port_name, iop.width if iop.width else cfg.xlen)
                    ann = ast.Attribute(
                        value=ast.Name(id="zdc", ctx=ast.Load()),
                        attr=_width_to_type_name(width),
                        ctx=ast.Load(),
                    )
                    capture = ast.AnnAssign(
                        target=ast.Name(id=iop.result_var, ctx=ast.Store()),
                        annotation=ann,
                        value=ast.Name(id=iop.port_name, ctx=ast.Load()),
                        simple=1,
                    )
                    ast.fix_missing_locations(capture)
                    ops.append(capture)

            # Convert raw ``await self.PORT.put(expr)`` inside ast.If bodies
            # (conditional egress) into ``self.PORT = expr`` assignments first,
            # before the top-level loop processes them.
            converted_body = _convert_egress_in_stmts(list(ir_stage.body), egress_port_name_set)

            for n in converted_body:
                if isinstance(n, ast.stmt):
                    ops.append(n)
                elif isinstance(n, IrEgressOp) and n.value_expr is not None:
                    # Convert ``await self.PORT.put(expr)`` →
                    # ``self.PORT = expr`` so existing SV emitter handles it.
                    synthetic = ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=n.port_name,
                                ctx=ast.Store(),
                            )
                        ],
                        value=n.value_expr,
                        lineno=0,
                        col_offset=0,
                    )
                    ast.fix_missing_locations(synthetic)
                    ops.append(synthetic)
            sir = StageIR(
                name=sn,
                index=i,
                inputs=stage_inputs.get(sn, []),
                outputs=stage_outputs.get(sn, []),
                operations=ops,
                cycle_lo=0,
                cycle_hi=max(0, ir_stage.cycles - 1),
            )
            sir_list.append(sir)

        # 4b. Add synthetic pass-through operations for auto-threaded stages.
        # A variable is "auto-threaded" through stage S if S has an output channel
        # for it but did NOT define it (def_stage != S).  In that case, ExprLowerer
        # needs a local signal (e.g. tag_id) so the pipeline register assignment
        # "tag_id_to_ex_q <= tag_id" is valid.  We emit a synthetic AnnAssign
        # "tag: zdc.u32 = tag_if_to_id_q" which ExprLowerer lowers to
        # "tag_id = tag_if_to_id_q;" inside the always @(*) block.
        for i, sir in enumerate(sir_list):
            sn = sir.name
            for out_ch in stage_outputs.get(sn, []):
                suffix = f"_{out_ch.src_stage.lower()}_to_{out_ch.dst_stage.lower()}"
                if not out_ch.name.endswith(suffix):
                    continue
                var_name = out_ch.name[: -len(suffix)]
                info = analyzer.vars.get(var_name)
                if info is None or info.def_stage == i:
                    continue  # variable is locally computed here — no pass-through needed

                # Find the input channel that carries var_name into this stage.
                in_reg: Optional[str] = None
                for in_ch in stage_inputs.get(sn, []):
                    in_suffix = f"_{in_ch.src_stage.lower()}_to_{in_ch.dst_stage.lower()}"
                    if in_ch.name.endswith(in_suffix):
                        in_var = in_ch.name[: -len(in_suffix)]
                    else:
                        in_var = in_ch.name
                    if in_var == var_name:
                        in_reg = f"{in_ch.name}_q"
                        break
                if in_reg is None:
                    continue

                # Synthetic: var: zdc.uN = <input_register_name>
                # The RHS ast.Name uses the literal register string so ExprLowerer
                # (which doesn't recognise it as a pipeline variable) emits it verbatim.
                ann = ast.Attribute(
                    value=ast.Name(id="zdc", ctx=ast.Load()),
                    attr=_width_to_type_name(info.width),
                    ctx=ast.Load(),
                )
                passthrough = ast.AnnAssign(
                    target=ast.Name(id=var_name, ctx=ast.Store()),
                    annotation=ann,
                    value=ast.Name(id=in_reg, ctx=ast.Load()),
                    simple=1,
                )
                sir.operations.append(passthrough)

        # 5. Generate RegFileAccess from IrHazardOp
        regfile_accesses, regfile_decls = self._gen_regfile_accesses(ip, comp_cls)

        # 6. Extract clock/reset fields (port_widths already extracted in step 0)
        clock_field = ip.clock_field
        reset_field = ip.reset_field

        # 7. Collect bubble stages (stages that contain IrBubble nodes)
        bubble_stages = [
            stage.name
            for stage in ip.stages
            if any(isinstance(n, IrBubble) for n in stage.body)
        ]

        # 8. Collect ingress/egress method port info from IrIngressOp/IrEgressOp.
        #    These may appear at the top level of the method body OR inside stage bodies.
        ingress_ports: List[tuple] = []
        egress_ports: List[tuple] = []
        seen_ingress: set = set()
        seen_egress: set = set()

        def _find_put_port(node: ast.expr) -> Optional[str]:
            """Return port name if *node* is ``await self.PORT.put(...)``."""
            if not isinstance(node, ast.Await):
                return None
            call = node.value
            if not isinstance(call, ast.Call):
                return None
            fn = call.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "put"):
                return None
            obj = fn.value
            if (isinstance(obj, ast.Attribute)
                    and isinstance(obj.value, ast.Name)
                    and obj.value.id == "self"):
                return obj.attr
            return None

        def _collect_ports_from_nodes(nodes):
            for node in nodes:
                if isinstance(node, IrIngressOp) and node.port_name not in seen_ingress:
                    seen_ingress.add(node.port_name)
                    # Use port_widths if we have a hint, else default 32
                    w = port_widths.get(node.port_name, node.width)
                    ingress_ports.append((node.port_name, w))
                elif isinstance(node, IrEgressOp) and node.port_name not in seen_egress:
                    seen_egress.add(node.port_name)
                    w = port_widths.get(node.port_name, node.width)
                    egress_ports.append((node.port_name, w))
                elif isinstance(node, ast.If):
                    # Recursively scan conditional branches for nested egress ops
                    _collect_ports_from_nodes(node.body)
                    _collect_ports_from_nodes(node.orelse)
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
                    # Raw ``await self.PORT.put(...)`` not yet converted to IrEgressOp
                    port_name = _find_put_port(node.value)
                    if port_name and port_name not in seen_egress:
                        seen_egress.add(port_name)
                        w = port_widths.get(port_name, cfg.xlen)
                        egress_ports.append((port_name, w))

        # Top-level ops
        _collect_ports_from_nodes(ip.ingress_ops)
        _collect_ports_from_nodes(ip.egress_ops)
        # Also scan inside stages
        for stage in ip.stages:
            _collect_ports_from_nodes(stage.body)

        # Build variable → port name map for expr lowering.
        # Only include ingress vars that are NOT threaded via channels: if a var
        # has a downstream channel (use_stages not empty), ExprLowerer will
        # resolve it from _reg_read (the channel register) in later stages, and
        # from _defined (the capture AnnAssign) in stage 0.  Keeping it in
        # ingress_var_map would incorrectly override that channel-based resolution.
        ingress_var_map: Dict[str, str] = {}
        for node in ip.ingress_ops:
            if node.result_var:
                info = analyzer.vars.get(node.result_var)
                if info is None or not info.use_stages:
                    # Not threaded — direct port substitution is fine
                    ingress_var_map[node.result_var] = node.port_name
        for stage in ip.stages:
            for node in stage.body:
                if isinstance(node, IrIngressOp) and node.result_var:
                    info = analyzer.vars.get(node.result_var)
                    if info is None or not info.use_stages:
                        ingress_var_map[node.result_var] = node.port_name

        # 9. Assemble PipelineIR
        module_name = comp_cls.__name__ if comp_cls is not None else "AsyncPipeline"
        pip = PipelineIR(
            module_name=module_name,
            stages=sir_list,
            channels=channels,
            meta=ir.meta,
            pipeline_stages=len(sir_list),
            regfile_accesses=regfile_accesses,
            regfile_decls=regfile_decls,
            forward_default=cfg.forward_default,
            clock_field=clock_field,
            reset_field=reset_field,
            clock_domain_field=ip.clock_domain_field,
            approach="async",
            annotation_map=annotation_map,
            port_widths=port_widths,
            bubble_stages=bubble_stages,
            ingress_ports=ingress_ports,
            egress_ports=egress_ports,
            ingress_var_map=ingress_var_map,
        )

        ir.pipeline_ir = pip
        return ir

    # ------------------------------------------------------------------
    # Port width extraction from component type hints
    # ------------------------------------------------------------------

    def _extract_port_widths(self, comp_cls: Any) -> Dict[str, int]:
        """Extract per-port bit-widths from the component's field annotations.

        zdc field types are ``Annotated[int, U(width=N, signed=...)]``.
        Falls back to 32 for fields whose type cannot be resolved.
        """
        if comp_cls is None:
            return {}
        try:
            import typing
            hints = typing.get_type_hints(comp_cls, include_extras=True)
            result: Dict[str, int] = {}
            for name, hint in hints.items():
                meta = getattr(hint, "__metadata__", ())
                for m in meta:
                    if hasattr(m, "width"):
                        result[name] = int(m.width)
                        break
            return result
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Channel generation
    # ------------------------------------------------------------------

    def _gen_channels(
        self,
        analyzer: _DefUseAnalyzer,
        stage_names: List[str],
    ) -> Tuple[List[ChannelDecl], Dict[str, List[ChannelDecl]], Dict[str, List[ChannelDecl]]]:
        """Generate ChannelDecl list and per-stage input/output maps."""
        channels: List[ChannelDecl] = []
        stage_inputs: Dict[str, List[ChannelDecl]] = {}
        stage_outputs: Dict[str, List[ChannelDecl]] = {}

        for var_name, info in analyzer.vars.items():
            if not info.use_stages:
                continue
            max_use = max(info.use_stages)
            def_idx = info.def_stage
            if max_use <= def_idx:
                continue  # same-stage or earlier — no channel needed

            # Thread from def_stage to max_use via consecutive ChannelDecls
            for i in range(def_idx, max_use):
                src = stage_names[i]
                dst = stage_names[i + 1]
                ch = ChannelDecl(
                    name=f"{var_name}_{src.lower()}_to_{dst.lower()}",
                    width=max(1, info.width),
                    depth=1,
                    src_stage=src,
                    dst_stage=dst,
                )
                channels.append(ch)
                stage_outputs.setdefault(src, []).append(ch)
                stage_inputs.setdefault(dst, []).append(ch)

        return channels, stage_inputs, stage_outputs

    # ------------------------------------------------------------------
    # RegFile access / declaration generation
    # ------------------------------------------------------------------

    def _gen_regfile_accesses(
        self, ip: IrPipeline, comp_cls: Any
    ) -> Tuple[List[RegFileAccess], List[RegFileDeclInfo]]:
        """Translate IrHazardOp nodes to RegFileAccess and RegFileDeclInfo."""
        accesses: List[RegFileAccess] = []
        seen_fields: Dict[str, RegFileDeclInfo] = {}

        for ir_stage in ip.stages:
            for op in ir_stage.hazard_ops:
                ra = self._translate_hazard_op(op, ir_stage.name, comp_cls, seen_fields)
                if ra is not None:
                    accesses.append(ra)

        return accesses, list(seen_fields.values())

    def _translate_hazard_op(
        self,
        op: IrHazardOp,
        stage_name: str,
        comp_cls: Any,
        seen_fields: Dict[str, "RegFileDeclInfo"],
    ) -> Optional[RegFileAccess]:
        """Convert one IrHazardOp to a RegFileAccess, or None if not applicable."""
        resource_expr = op.resource_expr
        if resource_expr is None:
            return None

        field_name, addr_var = self._parse_resource_expr(resource_expr)
        if field_name is None:
            return None

        # Register the regfile declaration
        if field_name not in seen_fields:
            decl = self._build_regfile_decl(field_name, comp_cls)
            seen_fields[field_name] = decl

        kind_map = {
            "reserve": "reserve",
            "block": "read",
            "acquire": "read",
            "write": "write",
            "release": "release",
        }
        kind = kind_map.get(op.op)
        if kind is None:
            return None

        data_var = ""
        result_var = ""
        if op.op == "write" and op.value_expr is not None:
            data_var = self._expr_to_str(op.value_expr)
        if op.op in ("block", "acquire") and op.result_var:
            result_var = op.result_var

        return RegFileAccess(
            field_name=field_name,
            kind=kind,
            stage=stage_name,
            addr_var=addr_var or "",
            data_var=data_var,
            result_var=result_var,
        )

    def _parse_resource_expr(self, expr: Any) -> Tuple[Optional[str], Optional[str]]:
        """Extract (field_name, addr_var) from ``self.field[addr]`` AST expression."""
        if isinstance(expr, ast.Subscript):
            val = expr.value
            if isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
                field_name = val.attr
                addr_var = self._expr_to_str(expr.slice)
                # Strip "self." prefix — component port references become bare names
                if addr_var.startswith("self."):
                    addr_var = addr_var[5:]
                return field_name, addr_var
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            # self.field (no index)
            return expr.attr, None
        return None, None

    def _expr_to_str(self, expr: Any) -> str:
        """Convert a simple AST expression to a string."""
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Constant):
            return str(expr.value)
        if isinstance(expr, ast.Attribute):
            return f"{self._expr_to_str(expr.value)}.{expr.attr}"
        try:
            return ast.unparse(expr)
        except Exception:
            return "<expr>"

    def _build_regfile_decl(self, field_name: str, comp_cls: Any) -> RegFileDeclInfo:
        """Build a RegFileDeclInfo for the given component field.

        Handles both ``IndexedRegFile`` (via annotated type hints) and
        ``PipelineResource`` (via runtime instance attributes).
        """
        import math
        depth = 32
        addr_width = 5
        data_width = 32
        lock_type = "bypass"  # default: forward hazards via bypass mux

        if comp_cls is not None:
            # --- PipelineResource instance (class-level attribute) ---
            try:
                from zuspec.dataclasses.pipeline_resource import PipelineResource
                from zuspec.dataclasses.pipeline_locks import BypassLock, QueueLock
                inst = getattr(comp_cls, field_name, None)
                if isinstance(inst, PipelineResource):
                    depth = inst.size
                    addr_width = max(1, int(math.ceil(math.log2(max(depth, 2)))))
                    if isinstance(inst.lock, BypassLock):
                        lock_type = "bypass"
                    elif isinstance(inst.lock, QueueLock):
                        lock_type = "queue"
                    return RegFileDeclInfo(
                        field_name=field_name,
                        depth=depth,
                        addr_width=addr_width,
                        data_width=data_width,
                        lock_type=lock_type,
                    )
            except Exception:
                pass

            # --- IndexedRegFile (annotated type hint with metadata) ---
            try:
                import typing
                hints = typing.get_type_hints(comp_cls, include_extras=True)
                hint = hints.get(field_name)
                if hint is not None:
                    meta = getattr(hint, "__metadata__", ())
                    for m in meta:
                        if hasattr(m, "depth"):
                            depth = m.depth
                        if hasattr(m, "data_width"):
                            data_width = m.data_width
                    addr_width = max(1, int(math.ceil(math.log2(max(depth, 2)))))
            except Exception:
                pass

        return RegFileDeclInfo(
            field_name=field_name,
            depth=depth,
            addr_width=addr_width,
            data_width=data_width,
            lock_type=lock_type,
        )
