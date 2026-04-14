"""PipelineAnnotationPass — Approach C pipeline stage extraction.

.. deprecated::
    This pass supports the old sentinel-based ``@zdc.pipeline`` API
    (``IF = zdc.stage()``, ``stages=[...]``).  It is superseded by
    :class:`~zuspec.synth.passes.pipeline_frontend.PipelineFrontendPass`,
    which consumes the new ``@zdc.stage`` method-per-stage API.
    Migrate to the new API; this pass will be removed in a future release.

Walks the Python AST of a ``@zdc.pipeline`` method body to:

1. Collect ``NAME = zdc.stage()`` assignments → ordered stage-name list.
2. Collect ``zdc.forward(...)`` / ``zdc.no_forward(...)`` calls →
   :class:`~zuspec.synth.ir.pipeline_ir.ForwardingDecl` list.
3. Compute the set of variables live across each stage boundary (def-use
   analysis on the body AST) → :class:`~zuspec.synth.ir.pipeline_ir.ChannelDecl`
   entries.
4. Validate the discovered stage list against the ``stages=`` argument on the
   ``@zdc.pipeline`` decorator.
5. Build and store a :class:`~zuspec.synth.ir.pipeline_ir.PipelineIR` in
   ``ir.pipeline_ir``.
"""
from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

from .synth_pass import SynthPass
from zuspec.synth.ir.pipeline_ir import (
    ChannelDecl, ForwardingDecl, HazardPair, PipelineIR, StageIR,
)
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)

# Width used when the type of a variable cannot be inferred from annotations.
_DEFAULT_WIDTH = 32


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_stage_call(node: ast.expr) -> bool:
    """Return True if *node* is a call to ``zdc.stage()``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "stage"
    if isinstance(func, ast.Name):
        return func.id == "stage"
    return False


def _is_forward_call(node: ast.expr) -> Optional[Tuple[str, str, str, bool]]:
    """Return ``(from_stage, to_stage, var, suppressed)`` if *node* is a
    ``zdc.forward()`` or ``zdc.no_forward()`` call, else ``None``.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None

    if name not in ("forward", "no_forward"):
        return None

    suppressed = name == "no_forward"
    kwargs: Dict[str, str] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        val = kw.value
        if isinstance(val, ast.Name):
            kwargs[kw.arg] = val.id
        elif isinstance(val, ast.Constant):
            kwargs[kw.arg] = str(val.value)

    from_stage = kwargs.get("from_stage", "")
    to_stage   = kwargs.get("to_stage", "")
    signal     = kwargs.get("signal", kwargs.get("var", ""))  # accept both names
    if not from_stage or not to_stage or not signal:
        return None
    return from_stage, to_stage, signal, suppressed


def _get_width_from_annotation(annotation: Optional[ast.expr]) -> int:
    """Try to extract a bit-width integer from a type annotation AST node.

    Handles simple cases: ``zdc.u32`` → 32, ``zdc.u5`` → 5, ``zdc.bit`` → 1.
    Returns ``_DEFAULT_WIDTH`` when the annotation cannot be resolved.
    """
    if annotation is None:
        return _DEFAULT_WIDTH
    # Attribute access: zdc.u32, zdc.bit
    if isinstance(annotation, ast.Attribute):
        name = annotation.attr
    elif isinstance(annotation, ast.Name):
        name = annotation.id
    else:
        return _DEFAULT_WIDTH

    if name == "bit" or name == "bit1":
        return 1
    # uN / uintN_t / bvN patterns
    for prefix in ("u", "bv", "uint", "i", "int", "s"):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            suffix = suffix.replace("_t", "")
            try:
                return int(suffix)
            except ValueError:
                pass
    return _DEFAULT_WIDTH


# ---------------------------------------------------------------------------
# Def-use analysis
# ---------------------------------------------------------------------------

class _DefUseCollector(ast.NodeVisitor):
    """Collect defined (written) and used (read) variable names from an AST subtree."""

    def __init__(self) -> None:
        self.defs: Set[str] = set()
        self.uses: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.defs.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Visit value before targets so we see uses in the RHS first.
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def generic_visit(self, node: ast.AST) -> None:
        super().generic_visit(node)


def _collect_du(stmts: List[ast.stmt]) -> Tuple[Set[str], Set[str]]:
    """Return ``(defs, uses)`` for a list of statements."""
    c = _DefUseCollector()
    for s in stmts:
        c.visit(s)
    return c.defs, c.uses


# ---------------------------------------------------------------------------
# Stage partitioner
# ---------------------------------------------------------------------------

class _StagePartition:
    """Partitions a function body into named stages."""

    def __init__(self) -> None:
        self.stage_names: List[str] = []
        # stage_stmts[i] = list of AST statements belonging to stage i
        self.stage_stmts: List[List[ast.stmt]] = []
        self.forwarding_decls: List[ForwardingDecl] = []
        # Map stage_name → index
        self._name_to_idx: Dict[str, int] = {}
        # Statements that appear before the first zdc.stage() marker (Approach A)
        self.unassigned_stmts: List[ast.stmt] = []

    def parse(self, body: List[ast.stmt]) -> None:
        """Parse the list of function body statements."""
        current_stmts: List[ast.stmt] = []
        current_name: Optional[str] = None

        for stmt in body:
            # Detect NAME = zdc.stage()
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and _is_stage_call(stmt.value):
                    # Finish previous stage (if any)
                    if current_name is not None:
                        self._add_stage(current_name, current_stmts)
                    else:
                        # Statements before the first stage marker are unassigned
                        self.unassigned_stmts.extend(current_stmts)
                    current_name = target.id
                    current_stmts = []
                    continue

            # Detect zdc.forward() / zdc.no_forward() at top level
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                result = _is_forward_call(stmt.value)
                if result is not None:
                    from_stage, to_stage, signal, suppressed = result
                    self.forwarding_decls.append(
                        ForwardingDecl(from_stage=from_stage, to_stage=to_stage,
                                       signal=signal, suppressed=suppressed)
                    )
                    continue

            current_stmts.append(stmt)

        # Flush last stage
        if current_name is not None:
            self._add_stage(current_name, current_stmts)
        else:
            # No stage markers at all — all statements are unassigned (Approach A)
            self.unassigned_stmts.extend(current_stmts)

    def _add_stage(self, name: str, stmts: List[ast.stmt]) -> None:
        idx = len(self.stage_names)
        self.stage_names.append(name)
        self.stage_stmts.append(stmts)
        self._name_to_idx[name] = idx


def _compute_channels(
    partition: "_StagePartition",
    annotation_map: Dict[str, ast.expr],
) -> Tuple[List[List[ChannelDecl]], List[List[ChannelDecl]]]:
    """Compute per-stage input/output ChannelDecls from live-variable analysis.

    A variable ``v`` defined in stage ``i`` and used in stage ``j > i``
    crosses the boundary ``i → i+1 → … → j`` and needs a pipeline register
    at each boundary.

    Returns ``(stage_inputs, stage_outputs)`` — lists parallel to partition.stage_names.
    ``stage_outputs[i]`` is the set of channels written at the end of stage ``i``;
    ``stage_inputs[j]`` is the set of channels read at the start of stage ``j``.
    """
    return compute_channels_from_stages(
        partition.stage_names, partition.stage_stmts, annotation_map
    )


def compute_channels_from_stages(
    stage_names: List[str],
    stage_stmts_list: List[List[ast.stmt]],
    annotation_map: Dict[str, ast.expr],
) -> Tuple[List[List[ChannelDecl]], List[List[ChannelDecl]]]:
    """Compute per-stage ChannelDecls from live-variable analysis.

    This is the public API used by both :class:`PipelineAnnotationPass` (Approach C)
    and :class:`SDCSchedulePass` (Approach A) after stage assignment.

    Parameters
    ----------
    stage_names:
        Ordered list of stage names (e.g. ``["IF", "EX", "WB"]``).
    stage_stmts_list:
        ``stage_stmts_list[i]`` is the list of AST statements for stage i.
    annotation_map:
        Variable name → type-annotation AST node (for width inference).

    Returns
    -------
    ``(stage_inputs, stage_outputs)`` — lists parallel to *stage_names*.
    """
    n = len(stage_names)
    stage_defs: List[Set[str]] = []
    stage_uses: List[Set[str]] = []
    for stmts in stage_stmts_list:
        d, u = _collect_du(stmts)
        stage_defs.append(d)
        stage_uses.append(u)

    seen_channels: Set[str] = set()
    stage_inputs:  List[List[ChannelDecl]] = [[] for _ in range(n)]
    stage_outputs: List[List[ChannelDecl]] = [[] for _ in range(n)]

    for def_stage_idx in range(n):
        for var in stage_defs[def_stage_idx]:
            for use_stage_idx in range(def_stage_idx + 1, n):
                if var in stage_uses[use_stage_idx]:
                    for boundary in range(def_stage_idx, use_stage_idx):
                        src = stage_names[boundary]
                        dst = stage_names[boundary + 1]
                        ch_name = f"{var}_{src.lower()}_to_{dst.lower()}"
                        if ch_name not in seen_channels:
                            seen_channels.add(ch_name)
                            width = _get_width_from_annotation(annotation_map.get(var))
                            ch = ChannelDecl(
                                name=ch_name, width=width, depth=1,
                                src_stage=src, dst_stage=dst,
                            )
                            stage_outputs[boundary].append(ch)
                            stage_inputs[boundary + 1].append(ch)
                    break  # only need the first crossing per def_stage

    return stage_inputs, stage_outputs


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

class PipelineAnnotationPass(SynthPass):
    """Approach C: extract pipeline stage structure from ``@zdc.pipeline`` body.

    Reads the ``_zdc_pipeline`` method from the component class in
    ``ir.component``, parses its AST, and builds a ``PipelineIR`` stored in
    ``ir.pipeline_ir``.

    If ``ir.component`` has no ``@zdc.pipeline`` method, this pass is a no-op.
    """

    @property
    def name(self) -> str:
        return "pipeline_annotation"

    def run(self, ir: SynthIR) -> SynthIR:  # noqa: C901 (acceptable complexity)
        """Run stage extraction on *ir*.

        :param ir: Synthesis IR with ``ir.component`` set.
        :type ir: SynthIR
        :return: Updated IR with ``ir.pipeline_ir`` populated.
        :rtype: SynthIR
        :raises PipelineError: For unsupported pipeline body constructs.
        """
        method = self._find_pipeline_method(ir)
        if method is None:
            _log.info("[PipelineAnnotationPass] no @zdc.pipeline method found — skipping")
            return ir

        comp_name = getattr(ir.component, "__name__", "Unknown")
        method_name = getattr(method, "__name__", "run")
        _log.info("[PipelineAnnotationPass] processing %s.%s", comp_name, method_name)

        # --- Parse AST ---
        try:
            src = inspect.getsource(method)
            src = textwrap.dedent(src)
            tree = ast.parse(src)
        except Exception as exc:
            _log.warning("[PipelineAnnotationPass] AST parse failed: %s", exc)
            return ir

        # Find the function def in the parsed tree
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    func_def = node
                    break
        if func_def is None:
            _log.warning("[PipelineAnnotationPass] could not locate function def for %s", method_name)
            return ir

        # Check for forbidden await zdc.cycles() in a @zdc.pipeline body
        self._check_no_cycles_await(func_def, method_name)

        # --- Partition body into stages ---
        partition = _StagePartition()
        partition.parse(func_def.body)

        if not partition.stage_names:
            # Approach A: no zdc.stage() markers — collect all body ops as one
            # flat stage; SDCSchedulePass will redistribute across stages.
            _log.info(
                "[PipelineAnnotationPass] %s.%s has no zdc.stage() markers — "
                "Approach A: collecting all ops, SDCSchedulePass will schedule",
                comp_name, method_name,
            )
            annotation_map = self._build_annotation_map(func_def)
            # Collect all non-stage-marker statements as the single S0 stage
            all_ops: List[ast.stmt] = list(partition.unassigned_stmts)
            if not all_ops:
                # Fall back: all body statements
                all_ops = list(func_def.body)

            s0 = StageIR(
                name="S0", index=0, inputs=[], outputs=[],
                operations=all_ops, cycle_lo=0, cycle_hi=0,
            )
            # Determine desired stage count from decorator (stages=N or stages=True)
            declared_stages = method._zdc_pipeline_stages
            if isinstance(declared_stages, int) and declared_stages > 0:
                desired_stages = declared_stages
            else:
                desired_stages = 1  # SDC will auto-determine
            pipeline_ir = PipelineIR(
                module_name=comp_name,
                stages=[s0],
                channels=[],
                meta=getattr(ir, "meta", None),
                pipeline_stages=desired_stages,
                forwarding=partition.forwarding_decls,
                approach="auto",
                forward_default=getattr(method, "_zdc_pipeline_forward", None),
                annotation_map=annotation_map,
                port_widths=self._build_port_widths(ir.component),
            )
            ir.pipeline_ir = pipeline_ir
            _log.info(
                "[PipelineAnnotationPass] %s: Approach A, %d ops in S0",
                comp_name, len(all_ops),
            )
            return ir

        # --- Validate against stages= decorator argument ---
        declared_stages = method._zdc_pipeline_stages
        self._validate_stages(declared_stages, partition.stage_names, method_name)

        # --- Build annotation map (var_name → type annotation AST node) ---
        annotation_map = self._build_annotation_map(func_def)

        # --- Compute live-variable channels ---
        stage_inputs, stage_outputs = _compute_channels(partition, annotation_map)

        # --- Build PipelineIR ---
        stage_irs: List[StageIR] = []
        all_channels: List[ChannelDecl] = []
        seen: Set[str] = set()

        for idx, name in enumerate(partition.stage_names):
            s = StageIR(
                name=name,
                index=idx,
                inputs=stage_inputs[idx],
                outputs=stage_outputs[idx],
                operations=partition.stage_stmts[idx],
                cycle_lo=idx,
                cycle_hi=idx,
            )
            stage_irs.append(s)
            for ch in stage_outputs[idx]:
                if ch.name not in seen:
                    seen.add(ch.name)
                    all_channels.append(ch)

        raw_forward = getattr(method, "_zdc_pipeline_forward", None)
        # If forward= is a list of _LegacyForwardingDecl, convert and merge into forwarding list
        if isinstance(raw_forward, list):
            from zuspec.dataclasses.decorators import _LegacyForwardingDecl as _ZdcFwdDecl
            extra_fwd = []
            for d in raw_forward:
                if isinstance(d, _ZdcFwdDecl):
                    extra_fwd.append(ForwardingDecl(
                        from_stage=d.from_stage or "",
                        to_stage=d.to_stage or "",
                        signal=d.signal,
                        suppressed=False,
                    ))
            all_forwarding = extra_fwd + partition.forwarding_decls
            forward_default = None  # explicit decls given; no blanket default
        else:
            all_forwarding = partition.forwarding_decls
            forward_default = raw_forward  # bool or None

        pipeline_ir = PipelineIR(
            module_name=comp_name,
            stages=stage_irs,
            channels=all_channels,
            meta=getattr(ir, "meta", None),
            pipeline_stages=len(stage_irs),
            forwarding=all_forwarding,
            approach="user",
            forward_default=forward_default,
            annotation_map=annotation_map,
            port_widths=self._build_port_widths(ir.component),
        )
        ir.pipeline_ir = pipeline_ir

        _log.info(
            "[PipelineAnnotationPass] %s: %d stages %s, %d channels, %d forwarding decls",
            comp_name,
            len(stage_irs),
            [s.name for s in stage_irs],
            len(all_channels),
            len(all_forwarding),
        )
        return ir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_pipeline_method(self, ir: SynthIR):
        """Return the first ``@zdc.pipeline``-decorated method found on
        ``ir.component``, or ``None``."""
        if ir.component is None:
            return None
        for name in dir(ir.component):
            try:
                attr = getattr(ir.component, name)
            except Exception:
                continue
            if callable(attr) and getattr(attr, "_zdc_pipeline", False):
                return attr
        return None

    def _check_no_cycles_await(self, func_def: ast.FunctionDef, method_name: str) -> None:
        """Raise ``PipelineError`` if the body contains ``await zdc.cycles(...)``."""
        try:
            from zuspec.dataclasses.decorators import PipelineError
        except ImportError:
            PipelineError = ValueError

        for node in ast.walk(func_def):
            if not isinstance(node, ast.Await):
                continue
            val = node.value
            if not isinstance(val, ast.Call):
                continue
            func = val.func
            if isinstance(func, ast.Attribute) and func.attr == "cycles":
                raise PipelineError(
                    f"@zdc.pipeline method '{method_name}' contains "
                    f"'await zdc.cycles(...)' which is not allowed in a pipeline "
                    f"process.  Use @zdc.sync for sequential processes."
                )
            if isinstance(func, ast.Name) and func.id == "cycles":
                raise PipelineError(
                    f"@zdc.pipeline method '{method_name}' contains "
                    f"'await cycles(...)' which is not allowed in a pipeline process."
                )

    def _validate_stages(
        self,
        declared: Any,
        found: List[str],
        method_name: str,
    ) -> None:
        """Validate discovered stage list against the ``stages=`` decorator arg."""
        try:
            from zuspec.dataclasses.decorators import PipelineError
        except ImportError:
            PipelineError = ValueError

        if declared is True or callable(declared):
            # Approach A or lambda — no count validation here
            return

        if isinstance(declared, int):
            if len(found) != declared:
                raise PipelineError(
                    f"@zdc.pipeline(stages={declared}) expects {declared} "
                    f"NAME = zdc.stage() markers in '{method_name}', "
                    f"but found {len(found)}: {found}"
                )
            return

        if isinstance(declared, list):
            if len(found) != len(declared):
                missing = [n for n in declared if n not in found]
                raise PipelineError(
                    f"@zdc.pipeline(stages={declared}) expects {len(declared)} "
                    f"stage markers in '{method_name}', but found {len(found)}: {found}. "
                    f"Missing: {missing}"
                )
            # Also check order / names match
            for i, (expected, actual) in enumerate(zip(declared, found)):
                if expected != actual:
                    raise PipelineError(
                        f"@zdc.pipeline stages= list declares stage '{expected}' "
                        f"at position {i} but found '{actual}' in '{method_name}'"
                    )

    def _build_annotation_map(self, func_def: ast.FunctionDef) -> Dict[str, ast.expr]:
        """Build a map of variable name → type annotation from the function body.

        Handles ``var: SomeType = ...`` and ``var: SomeType`` annotations.
        """
        ann_map: Dict[str, ast.expr] = {}
        for node in ast.walk(func_def):
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.annotation:
                    ann_map[node.target.id] = node.annotation
        return ann_map

    def _build_port_widths(self, component_cls: Any) -> Dict[str, int]:
        """Return ``{port_name: bit_width}`` from the component class's field annotations.

        Handles both plain ``name: zdc.u32`` annotations (which become
        ``typing.Annotated[int, U(width=32)]``) and ``zdc.input`` / ``zdc.output``
        field descriptors (which store the width on the ``default_factory``).
        Falls back to 32 for unrecognised types.
        """
        import typing
        widths: Dict[str, int] = {}
        if component_cls is None:
            return widths
        try:
            hints = typing.get_type_hints(component_cls, include_extras=True)
        except Exception:
            return widths
        for name, hint in hints.items():
            if name.startswith("_"):
                continue
            w = _width_from_type_hint(hint)
            if w is not None:
                widths[name] = w
        return widths


def _width_from_type_hint(hint: Any) -> Optional[int]:
    """Extract a bit-width integer from a ``zdc`` type hint, or return None."""
    import typing
    # typing.Annotated[int, U(width=W, signed=...)]
    if hasattr(hint, "__metadata__"):
        for meta in hint.__metadata__:
            if hasattr(meta, "width"):
                return int(meta.width)
    # zdc.input(zdc.u32) → dataclasses.Field; type stored on default_factory
    # The Field type is None but default_factory is the Input/Output sentinel.
    # The actual type info is on the *annotation* in the *source* — we can't get it
    # from the Field itself at runtime (it is erased). Return None; the annotation
    # pass will infer from body type comments instead.
    return None
