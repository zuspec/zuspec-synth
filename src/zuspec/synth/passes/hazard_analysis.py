"""HazardAnalysisPass — detect data hazards across pipeline stage boundaries.

Operates on a ``PipelineIR`` already populated by ``PipelineAnnotationPass``
or ``SDCSchedulePass``.  Populates ``PipelineIR.hazards`` with
:class:`~zuspec.synth.ir.pipeline_ir.HazardPair` entries.

Hazard types detected
---------------------
RAW (Read-After-Write / true dependency)
    A variable is written in stage W and read in an earlier stage R.  In a
    feed-forward pipeline this normally cannot happen for plain local
    variables.  It *does* arise for ``IndexedRegFile`` fields where a write
    in stage W can alias a read in stage R < W when the addresses match at
    runtime.

WAW (Write-After-Write / output dependency)
    The same output field is written in two different stages.  The earlier
    write is shadowed; the synthesiser suppresses it.

WAR (Write-After-Read / anti-dependency)
    A variable is read in stage R and later written in stage W > R.  For
    plain pipeline-local variables this is impossible (SSA-style); it can
    arise with ``IndexedRegFile`` fields or self.* component outputs.

For ``IndexedRegFile`` accesses the pass delegates to
:class:`~zuspec.synth.sprtl.regfile_synth.RegFileHazardAnalyzer`.
"""
from __future__ import annotations

import ast
import logging
from typing import Dict, List, Optional, Set, Tuple

from .synth_pass import SynthPass
from zuspec.synth.ir.pipeline_ir import (
    HazardPair, PipelineIR, RegFileAccess, RegFileDeclInfo, RegFileHazard, StageIR,
)
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — extract write/read sets from stage AST operations
# ---------------------------------------------------------------------------

class _WriteReadCollector(ast.NodeVisitor):
    """Collect plain-variable write and read names from a list of AST statements."""

    def __init__(self) -> None:
        self.writes: Set[str] = set()
        self.reads:  Set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)  # RHS reads
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.writes.add(t.id)
            elif isinstance(t, ast.Attribute):
                # self.field = ... → component output write
                self.writes.add(f"self.{t.attr}")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.writes.add(node.target.id)
            self.reads.add(node.target.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.writes.add(node.target.id)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.reads.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            # self.field → component input read
            self.reads.add(f"self.{node.attr}")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        super().generic_visit(node)


def _get_writes_reads(stmts: list) -> Tuple[Set[str], Set[str]]:
    """Return ``(writes, reads)`` for a list of AST statement nodes."""
    c = _WriteReadCollector()
    for s in stmts:
        c.visit(s)
    return c.writes, c.reads


# ---------------------------------------------------------------------------
# RegFile access detection
# ---------------------------------------------------------------------------

class _RegFileAccessCollector(ast.NodeVisitor):
    """Collect ``self.FIELD.read(addr)`` and ``self.FIELD.write(idx, val)`` calls.

    For each detected call, appends a :class:`RegFileAccess` to ``self.accesses``.
    The ``result_var`` for reads is filled in by the caller after examining the
    enclosing assignment target.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.accesses: List[RegFileAccess] = []
        # Track the current assignment target for read result binding
        self._current_target: str = ""

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._current_target = node.target.id
        if node.value:
            self.visit(node.value)
        self._current_target = ""

    def visit_Assign(self, node: ast.Assign) -> None:
        if node.targets and isinstance(node.targets[0], ast.Name):
            self._current_target = node.targets[0].id
        self.visit(node.value)
        self._current_target = ""
        # Also visit remaining targets
        for t in node.targets[1:]:
            self.visit(t)

    def visit_Call(self, node: ast.Call) -> None:
        # Match: self.FIELD.read(addr_expr)
        #    or: self.FIELD.write(idx_expr, data_expr)
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        ):
            field_name = func.value.attr
            method = func.attr
            if method == "read" and len(node.args) >= 1:
                addr_arg = node.args[0]
                addr_var = addr_arg.id if isinstance(addr_arg, ast.Name) else ast.unparse(addr_arg)
                self.accesses.append(RegFileAccess(
                    field_name=field_name,
                    kind="read",
                    stage=self.stage,
                    addr_var=addr_var,
                    result_var=self._current_target,
                ))
            elif method == "write" and len(node.args) >= 2:
                idx_arg  = node.args[0]
                data_arg = node.args[1]
                addr_var = idx_arg.id  if isinstance(idx_arg,  ast.Name) else ast.unparse(idx_arg)
                data_var = data_arg.id if isinstance(data_arg, ast.Name) else ast.unparse(data_arg)
                self.accesses.append(RegFileAccess(
                    field_name=field_name,
                    kind="write",
                    stage=self.stage,
                    addr_var=addr_var,
                    data_var=data_var,
                ))
        self.generic_visit(node)


def _collect_regfile_accesses(stage_name: str, stmts: List) -> List[RegFileAccess]:
    """Return all regfile read/write accesses found in *stmts*."""
    col = _RegFileAccessCollector(stage=stage_name)
    for s in stmts:
        col.visit(s)
    return col.accesses


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

class HazardAnalysisPass(SynthPass):
    """Detect data hazards between pipeline stages and populate ``PipelineIR.hazards``.

    This pass is a no-op when ``ir.pipeline_ir`` is ``None`` (i.e. the
    component has no ``@zdc.pipeline`` method).
    """

    @property
    def name(self) -> str:
        return "hazard_analysis"

    def run(self, ir: SynthIR) -> SynthIR:
        """Detect all data hazards in the pipeline and store them on *pip*.

        :param ir: Synthesis IR with ``ir.pipeline_ir`` set.
        :type ir: SynthIR
        :return: Updated IR with ``pip.hazards``, ``pip.regfile_accesses``,
                 ``pip.regfile_hazards``, and ``pip.regfile_decls`` populated.
        :rtype: SynthIR
        """
        if ir.pipeline_ir is None:
            _log.debug("[HazardAnalysisPass] no pipeline_ir — skipping")
            return ir

        pip = ir.pipeline_ir
        n = len(pip.stages)
        if n < 2:
            return ir

        _log.info("[HazardAnalysisPass] analysing %d stages in %s", n, pip.module_name)

        # Collect write/read sets per stage from operations (AST nodes)
        stage_writes: List[Set[str]] = []
        stage_reads:  List[Set[str]] = []
        for stage in pip.stages:
            if stage.operations and isinstance(stage.operations[0], ast.stmt):
                w, r = _get_writes_reads(stage.operations)
            else:
                # Operations not yet AST nodes (e.g. built from SDCSchedulePass)
                w, r = set(), set()
            stage_writes.append(w)
            stage_reads.append(r)

        hazards: List[HazardPair] = []

        # RAW: variable written in stage W, read in earlier stage R < W
        #      (unusual for plain locals but possible for component outputs /
        #       IndexedRegFile accesses — conservative check)
        for w_idx in range(n):
            for var in stage_writes[w_idx]:
                for r_idx in range(w_idx):
                    if var in stage_reads[r_idx]:
                        hazards.append(HazardPair(
                            kind="RAW",
                            producer_stage=pip.stages[w_idx].name,
                            consumer_stage=pip.stages[r_idx].name,
                            signal=var,
                        ))
                        _log.debug(
                            "[HazardAnalysisPass] RAW: %s written in %s, read in %s",
                            var, pip.stages[w_idx].name, pip.stages[r_idx].name,
                        )

        # WAW: same variable written in two different stages
        all_write_stages: Dict[str, List[int]] = {}
        for idx, w_set in enumerate(stage_writes):
            for var in w_set:
                all_write_stages.setdefault(var, []).append(idx)
        for var, idxs in all_write_stages.items():
            if len(idxs) >= 2:
                # First write is shadowed by the later one
                for earlier, later in zip(idxs, idxs[1:]):
                    hazards.append(HazardPair(
                        kind="WAW",
                        producer_stage=pip.stages[earlier].name,
                        consumer_stage=pip.stages[later].name,
                        signal=var,
                    ))
                    _log.debug(
                        "[HazardAnalysisPass] WAW: %s written in %s and %s",
                        var, pip.stages[earlier].name, pip.stages[later].name,
                    )

        # WAR: variable read in stage R, written in later stage W > R
        for r_idx in range(n):
            for var in stage_reads[r_idx]:
                for w_idx in range(r_idx + 1, n):
                    if var in stage_writes[w_idx]:
                        hazards.append(HazardPair(
                            kind="WAR",
                            producer_stage=pip.stages[r_idx].name,
                            consumer_stage=pip.stages[w_idx].name,
                            signal=var,
                        ))
                        _log.debug(
                            "[HazardAnalysisPass] WAR: %s read in %s, written in %s",
                            var, pip.stages[r_idx].name, pip.stages[w_idx].name,
                        )

        # Delegate IndexedRegFile accesses to RegFileHazardAnalyzer
        # Note: regfile hazards are tracked separately in pip.regfile_hazards
        # and resolved by ForwardingGenPass._resolve_regfile_hazards.
        # We do NOT add them to pip.hazards to avoid the plain-variable
        # forwarding logic auto-generating bypasses that conflict with lock_type.
        self._analyze_regfile_hazards(ir)

        # De-duplicate (same kind/producer/consumer/signal)
        seen: Set[Tuple[str, str, str, str]] = set()
        unique_hazards: List[HazardPair] = []
        for h in hazards:
            key = (h.kind, h.producer_stage, h.consumer_stage, h.signal)
            if key not in seen:
                seen.add(key)
                unique_hazards.append(h)

        pip.hazards = unique_hazards
        _log.info(
            "[HazardAnalysisPass] %d hazard(s) detected in %s",
            len(unique_hazards), pip.module_name,
        )
        return ir

    def _analyze_regfile_hazards(self, ir: SynthIR) -> None:
        """Detect RAW hazards for IndexedRegFile / PipelineResource accesses.

        Scans each stage's AST operations for ``self.FIELD.read(addr)`` and
        ``self.FIELD.write(idx, val)`` calls.  For every read in stage R and
        write in stage W where W > R, a ``RegFileHazard`` is appended to
        ``pip.regfile_hazards``.

        Regfile hazards are tracked separately from plain-variable hazards
        (``pip.hazards``) so that ``ForwardingGenPass`` resolves them using
        per-field ``lock_type`` rather than the global ``forward_default``.
        """
        pip = ir.pipeline_ir
        if pip is None:
            return

        # If accesses were pre-populated by AsyncPipelineToIrPass (from IrHazardOp),
        # use them directly instead of re-scanning the AST — the AST patterns differ.
        pre_populated = bool(getattr(pip, "regfile_accesses", None))
        if pre_populated:
            all_accesses = list(pip.regfile_accesses)
        else:
            # Fall back to AST scanning for sync pipelines
            all_accesses = []
            for stage in pip.stages:
                if not stage.operations or not isinstance(stage.operations[0], ast.stmt):
                    continue
                accesses = _collect_regfile_accesses(stage.name, stage.operations)
                all_accesses.extend(accesses)
            pip.regfile_accesses = all_accesses

        # Build RegFileDeclInfo entries only if not already set.
        if not getattr(pip, "regfile_decls", None):
            pip.regfile_decls = self._build_regfile_decls(all_accesses, pip)

        if not all_accesses:
            return

        # Determine width information from component annotations
        comp = ir.component
        port_widths = getattr(pip, "port_widths", {}) or {}

        # Build stage name → index map
        stage_index: Dict[str, int] = {s.name: i for i, s in enumerate(pip.stages)}

        # Detect RAW: a read in stage R, and a write in a LATER stage W > R.
        # In a pipeline, the write result from stage W "passes through" all earlier
        # stages and could alias the read address in stage R.
        reads  = [a for a in all_accesses if a.kind == "read"]
        writes = [a for a in all_accesses if a.kind == "write"]

        rf_hazards: List[RegFileHazard] = []

        for read in reads:
            r_idx = stage_index.get(read.stage, -1)
            for write in writes:
                w_idx = stage_index.get(write.stage, -1)
                # Hazard when write is in a later stage than the read
                if (
                    read.field_name == write.field_name
                    and w_idx > r_idx
                    and r_idx >= 0
                    and w_idx >= 0
                ):
                    rfh = RegFileHazard(
                        field_name=read.field_name,
                        write_stage=write.stage,
                        read_stage=read.stage,
                        write_addr_var=write.addr_var,
                        read_addr_var=read.addr_var,
                        write_data_var=write.data_var,
                        read_result_var=read.result_var,
                    )
                    rf_hazards.append(rfh)
                    _log.debug(
                        "[HazardAnalysisPass] RAW_RF: %s.read(%s) in %s, write(%s) in %s",
                        read.field_name, read.addr_var, read.stage,
                        write.addr_var, write.stage,
                    )

        pip.regfile_hazards = rf_hazards

    def _build_regfile_decls(
        self,
        accesses: List[RegFileAccess],
        pip: "PipelineIR",
    ) -> List[RegFileDeclInfo]:
        """Build one ``RegFileDeclInfo`` per unique regfile field name.

        Infers ``data_width`` from ``pip.port_widths`` (keyed on the data/result
        variable names found in the accesses) and ``addr_width`` from the address
        variables, defaulting to 32-bit data and 5-bit address.

        Preserves ``lock_type`` from any pre-existing entry in
        ``pip.regfile_decls`` (set by ``AsyncPipelineToIrPass``).
        """
        import math
        port_widths = getattr(pip, "port_widths", {}) or {}
        # Preserve lock_type from pre-populated decls (e.g. from PipelineResource)
        existing_lock = {d.field_name: d.lock_type for d in getattr(pip, "regfile_decls", [])}
        seen: Dict[str, RegFileDeclInfo] = {}
        for acc in accesses:
            if acc.field_name in seen:
                continue
            # Infer data width
            if acc.kind == "read" and acc.result_var:
                dw = port_widths.get(acc.result_var, 32)
            elif acc.kind == "write" and acc.data_var:
                dw = port_widths.get(acc.data_var, 32)
            else:
                dw = 32
            # Infer addr width
            aw = port_widths.get(acc.addr_var, 5)
            depth = 2 ** aw
            seen[acc.field_name] = RegFileDeclInfo(
                field_name=acc.field_name,
                depth=depth,
                addr_width=aw,
                data_width=dw,
                lock_type=existing_lock.get(acc.field_name, "bypass"),
            )
        return list(seen.values())
