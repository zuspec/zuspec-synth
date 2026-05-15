# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""ProcessToFSMPass — convert clocked processes to FSMModule instances."""
from __future__ import annotations

import logging
from typing import Any, List

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared predicates (mirrors of private helpers in __init__.py)
# ---------------------------------------------------------------------------

def _is_simple_tick_proc(proc) -> bool:
    """Return True if *proc* is a ``while True: await zdc.tick(); <stmts>`` loop."""
    def _is_tick_await(stmt) -> bool:
        if type(stmt).__name__ != 'StmtExpr':
            return False
        expr = getattr(stmt, 'expr', None)
        if type(expr).__name__ != 'ExprAwait':
            return False
        inner = getattr(expr, 'value', None)
        if type(inner).__name__ != 'ExprCall':
            return False
        func = getattr(inner, 'func', None)
        if type(func).__name__ != 'ExprAttribute':
            return False
        return getattr(func, 'attr', '') == 'tick' and not getattr(inner, 'args', None)

    def _has_nested_await(stmts) -> bool:
        for s in stmts or []:
            t = type(s).__name__
            if t == 'StmtExpr' and type(getattr(s, 'expr', None)).__name__ == 'ExprAwait':
                return True
            if t == 'StmtAssign' and type(getattr(s, 'value', None)).__name__ == 'ExprAwait':
                return True
            for attr in ('body', 'orelse'):
                sub = getattr(s, attr, None)
                if isinstance(sub, list) and _has_nested_await(sub):
                    return True
        return False

    body = getattr(proc, 'body', [])
    if len(body) != 1 or type(body[0]).__name__ != 'StmtWhile':
        return False
    loop = body[0]
    test = getattr(loop, 'test', None)
    if not (type(test).__name__ == 'ExprConstant' and bool(getattr(test, 'value', False))):
        return False
    loop_body = getattr(loop, 'body', [])
    tick_count = sum(1 for s in loop_body if _is_tick_await(s))
    if tick_count != 1:
        return False
    for s in loop_body:
        if _is_tick_await(s):
            continue
        t = type(s).__name__
        if t not in ('StmtAssign', 'StmtAugAssign', 'StmtIf', 'StmtAnnAssign'):
            return False
        for attr in ('body', 'orelse'):
            sub = getattr(s, attr, None)
            if isinstance(sub, list) and _has_nested_await(sub):
                return False
    return True


def _has_protocol_port(component_ir) -> bool:
    """Return True if any field is a ProtocolPort/ProtocolExport/Callable*."""
    try:
        from zuspec.ir.core.fields import FieldKind
        port_kinds = (
            FieldKind.ProtocolPort, FieldKind.ProtocolExport,
            FieldKind.CallablePort, FieldKind.CallableExport,
        )
        for f in getattr(component_ir, 'fields', []):
            if getattr(f, 'kind', None) in port_kinds:
                return True
    except ImportError:
        pass
    return False


def _proc_has_await(proc) -> bool:
    """Return True if the process body contains any ExprAwait statement."""
    def _check(stmts):
        for s in stmts or []:
            t = type(s).__name__
            if t == 'StmtExpr':
                if type(getattr(s, 'expr', None)).__name__ == 'ExprAwait':
                    return True
            elif t == 'StmtAssign':
                if type(getattr(s, 'value', None)).__name__ == 'ExprAwait':
                    return True
            for attr in ('body', 'orelse'):
                sub = getattr(s, attr, None)
                if isinstance(sub, list) and _check(sub):
                    return True
        return False
    return _check(getattr(proc, 'body', []))


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def _choose_strategy(proc, component_ir):
    """Return a Strategy instance for *proc*."""
    proc_type = type(proc).__name__
    is_async = getattr(proc, 'is_async', False)

    # Non-async @zdc.sync (type 'Function') → single-state always_ff.
    if proc_type in ('SyncProcess', 'Function') and not is_async:
        return SingleStateStrategy()

    # Async @zdc.sync → FSM transformer.
    if proc_type in ('SyncProcess', 'Function') and is_async:
        return SPRTLStrategy()

    # @zdc.proc: route based on content.
    if proc_type == 'Process':
        if _is_simple_tick_proc(proc) and not _has_protocol_port(component_ir):
            return SingleStateStrategy()
        if _has_protocol_port(component_ir) or _proc_has_await(proc):
            return SPRTLStrategy()
        return SingleStateStrategy()  # reg-process style fallback

    # Default: full FSM transformer.
    return SPRTLStrategy()


# ---------------------------------------------------------------------------
# FSMAssign/FSMCond builder for SingleStateStrategy
# ---------------------------------------------------------------------------

def _body_to_fsm_ops(stmts, idx_to_name: dict) -> list:
    """Convert IR statement list to FSMAssign/FSMCond operations.

    This is used by ``SingleStateStrategy`` to populate the single IDLE
    state without going through ``SPRTLTransformer``.
    """
    from zuspec.synth.sprtl.fsm_ir import FSMAssign, FSMCond
    from zuspec.synth.sprtl.ir_to_sv import ir_expr_to_sv

    ops = []
    for stmt in stmts:
        t = type(stmt).__name__
        if t == 'StmtAssign':
            for target in stmt.targets:
                tgt_sv = ir_expr_to_sv(target, idx_to_name)
                val_sv = ir_expr_to_sv(stmt.value, idx_to_name)
                ops.append(FSMAssign(target=tgt_sv, value=val_sv, is_nonblocking=True))
        elif t == 'StmtAugAssign':
            tgt_sv = ir_expr_to_sv(stmt.target, idx_to_name)
            val_sv = ir_expr_to_sv(stmt.value, idx_to_name)
            op_name = stmt.op.name if hasattr(stmt.op, "name") else str(stmt.op)
            from zuspec.synth.sprtl.ir_to_sv import _BINOP_SV
            sv_op = _BINOP_SV.get(op_name, op_name)
            ops.append(FSMAssign(
                target=tgt_sv,
                value=f"({tgt_sv} {sv_op} {val_sv})",
                is_nonblocking=True,
            ))
        elif t == 'StmtIf':
            cond_sv = ir_expr_to_sv(stmt.test, idx_to_name)
            then_ops = _body_to_fsm_ops(stmt.body, idx_to_name)
            else_ops = _body_to_fsm_ops(stmt.orelse, idx_to_name) if stmt.orelse else []
            ops.append(FSMCond(condition=cond_sv, then_ops=then_ops, else_ops=else_ops))
        elif t == 'StmtWhile':
            # Unwrap outer `while True:` loop (tick_proc pattern).
            inner_body = getattr(stmt, 'body', [])
            # Filter out the `await zdc.tick()` statement.
            filtered = [
                s for s in inner_body
                if not (
                    type(s).__name__ == 'StmtExpr'
                    and type(getattr(s, 'expr', None)).__name__ == 'ExprAwait'
                    and type(getattr(getattr(s, 'expr', None), 'value', None)).__name__ == 'ExprCall'
                )
            ]
            ops.extend(_body_to_fsm_ops(filtered, idx_to_name))
        elif t == 'StmtExpr':
            # Handle `await reg.write(val)` as a non-blocking assignment.
            expr = getattr(stmt, 'expr', None)
            if expr is not None and type(expr).__name__ == 'ExprAwait':
                inner = getattr(expr, 'value', None)
                if inner is not None and type(inner).__name__ == 'ExprCall':
                    func = getattr(inner, 'func', None)
                    if (func is not None
                            and type(func).__name__ == 'ExprAttribute'
                            and func.attr == 'write'):
                        base = getattr(func, 'value', None)
                        if base is not None and type(base).__name__ == 'ExprRefField':
                            field_name = idx_to_name.get(base.index, f"_f{base.index}")
                            args = getattr(inner, 'args', [])
                            val_sv = ir_expr_to_sv(args[0], idx_to_name) if args else "'0"
                            ops.append(FSMAssign(
                                target=field_name,
                                value=val_sv,
                                is_nonblocking=True,
                            ))
    return ops


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class SingleStateStrategy:
    """Build a degenerate single-state FSMModule for plain @zdc.sync / simple @zdc.proc."""

    def build(self, proc, component_ir, component_fields, model_context) -> Any:
        """Return a single-state FSMModule for *proc*."""
        from zuspec.synth.sprtl.fsm_ir import (
            FSMModule, FSMPort, FSMRegister, DomainBinding,
        )

        cf = component_fields

        fsm = FSMModule(name=cf.module_name or "module", single_state=True)

        # Apply domain binding.
        fsm.clock_signal = cf.clock_name
        fsm.reset_signal = cf.reset_name
        fsm.reset_active_low = cf.reset_active_low
        fsm.reset_async = cf.reset_async
        fsm.reset_clauses = list(cf.reset_clauses)

        # Inject ports from ComponentFields (pre-classified, bundle-expanded).
        for p in cf.ports:
            if p.name not in (cf.clock_name, cf.reset_name):
                fsm.add_port(p.name, p.direction, p.width, p.reset_value)

        # Inject state variables as registers.
        for sv in cf.state_vars:
            fsm.add_register(sv.name, sv.width, sv.reset_value)

        # Populate field_names map.
        fsm.field_names = dict(cf.idx_to_name)
        fsm.body_idx_to_name = dict(cf.idx_to_name)

        # Store raw body statements for ir_stmts_to_sv (correct SV semantics).
        body = getattr(proc, 'body', [])
        # Unwrap outer `while True:` loop for tick_proc pattern.
        if (len(body) == 1
                and type(body[0]).__name__ == 'StmtWhile'):
            inner = getattr(body[0], 'body', [])
            # Filter out the `await zdc.tick()` statements.
            body = [
                s for s in inner
                if not (
                    type(s).__name__ == 'StmtExpr'
                    and type(getattr(s, 'expr', None)).__name__ == 'ExprAwait'
                    and type(getattr(getattr(s, 'expr', None), 'value', None)).__name__ == 'ExprCall'
                )
            ]
        fsm.body_stmts = list(body)

        _log.debug(
            "[SingleStateStrategy] built single-state FSM with body_stmts=%d",
            len(fsm.body_stmts),
        )
        return fsm


class SPRTLStrategy:
    """Build an FSMModule using SPRTLTransformer (multi-state / protocol path)."""

    def build(self, proc, component_ir, component_fields, model_context) -> Any:
        """Return a multi-state FSMModule for *proc*."""
        from zuspec.synth.sprtl import (
            SPRTLTransformer, FSMOptimizer,
        )
        from zuspec.synth.sprtl.fsm_ir import DomainBinding

        cf = component_fields
        transformer = SPRTLTransformer()
        fsm = transformer.transform(component_ir, proc)
        fsm.name = cf.module_name or fsm.name

        # Apply domain clock/reset info from ComponentFields.
        fsm.clock_signal = cf.clock_name
        fsm.reset_signal = cf.reset_name
        fsm.reset_active_low = cf.reset_active_low
        fsm.reset_async = cf.reset_async

        # Trust SPRTLTransformer for ports and registers — it handles
        # bundle expansion, protocol ports, and array registers correctly.

        FSMOptimizer(minimize_states=False, merge_operations=False).optimize(fsm)

        if model_context is not None:
            _apply_struct_ir(fsm, model_context)

        _log.debug("[SPRTLStrategy] built %d-state FSM", len(fsm.states))
        return fsm


def _apply_struct_ir(fsm, ctx) -> None:
    """Best-effort struct IR annotation (silently skips on failure)."""
    try:
        from zuspec.synth.sprtl.struct_annotator import build_struct_metadata
        from zuspec.synth.sprtl.struct_ir_rewriter import rewrite_struct_ir
        struct_defs, flat_to_struct = build_struct_metadata(ctx)
        if flat_to_struct:
            rewrite_struct_ir(fsm, struct_defs, flat_to_struct)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

def _get_component_ir(ir: SynthIR):
    cls = ir.component
    ctx = ir.model_context
    if ctx is None or cls is None:
        return None
    return (
        ctx.type_m.get(getattr(cls, "__qualname__", None))
        or ctx.type_m.get(cls.__name__)
    )


class ProcessToFSMPass(SynthPass):
    """Convert every clocked process to an FSMModule.

    Iterates ``sync_processes + proc_processes`` on the component IR, selects
    the appropriate strategy for each process, and appends the resulting
    ``FSMModule`` to ``ir.fsm_modules``.

    Reads:
        ir.component_fields: Pre-classified ports, state vars, and domain info
            (set by ComponentFieldsPass).
        ir.model_context: DataModel context (set by ComponentFieldsPass's
            caller or ElaboratePass).

    Populates:
        ir.fsm_modules: One FSMModule per clocked process.

    Raises:
        ValueError: If the component IR cannot be found in model_context.
    """

    @property
    def name(self) -> str:
        return "process_to_fsm"

    def run(self, ir: SynthIR) -> SynthIR:
        component_ir = _get_component_ir(ir)
        if component_ir is None:
            raise ValueError(
                f"ProcessToFSMPass: could not find IR for "
                f"{getattr(ir.component, '__name__', ir.component)!r}."
            )

        cf = ir.component_fields
        if cf is None:
            raise ValueError(
                "ProcessToFSMPass: ir.component_fields is None — "
                "run ComponentFieldsPass first."
            )

        sync_processes = getattr(component_ir, "sync_processes", [])
        proc_processes = getattr(component_ir, "proc_processes", [])
        all_procs = list(sync_processes) + list(proc_processes)

        for proc in all_procs:
            strategy = _choose_strategy(proc, component_ir)
            fsm = strategy.build(proc, component_ir, cf, ir.model_context)
            fsm.name = cf.module_name or getattr(ir.component, "__name__", "module")
            ir.fsm_modules.append(fsm)
            _log.debug(
                "[ProcessToFSMPass] %s → %s (%d states)",
                type(proc).__name__,
                type(strategy).__name__,
                len(fsm.states),
            )

        return ir
