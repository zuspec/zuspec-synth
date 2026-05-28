# Copyright 2019-2026 Matthew Ballance and contributors
# SPDX-License-Identifier: Apache-2.0
"""FSMToPythonPass — convert FSMModule instances to Python ``@zdc.sync`` bodies.

Place in pass chain
-------------------
Runs after ``ProcessToFSMPass``; before ``CombToPythonPass`` and
``ModuleAssemblePythonPass``.

Inputs
------
- ``ir.fsm_modules``      — List of FSMModule instances.
- ``ir.component_fields`` — ComponentFields (ports, state_vars, module_name).
- ``ir.component``        — The Python component class (for reset_domain).

Outputs
-------
- ``ir.lowered_py["py/module/sync"]`` — ``@zdc.sync`` method body for the
  **single-state** path.  Assembled into a full class by
  ``ModuleAssemblePythonPass``.
- ``ir.lowered_py["py/module/top"]``  — Complete Python class for the
  **multi-state** path (bypasses ``ModuleAssemblePythonPass``).
"""
from __future__ import annotations

import logging
from io import StringIO
from typing import Any, Dict, List, Optional, Set, Tuple

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthIR

_log = logging.getLogger(__name__)

# Python binary operators mapped from IR / SV names
_BINOP_PY: Dict[str, str] = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "//", "Mod": "%",
    "BitAnd": "&", "BitOr": "|", "BitXor": "^",
    "LShift": "<<", "RShift": ">>",
    "And": "and", "Or": "or",
    "Eq": "==", "NotEq": "!=",
    "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
}

# Map tuple operator strings from the FSM transformer → Python operators
_TUPLE_OP_PY: Dict[str, str] = {
    "lt": "<", "lte": "<=", "gt": ">", "gte": ">=",
    "eq": "==", "ne": "!=",
    "+": "+", "-": "-", "*": "*",
}

# AugOp integer values → Python augmented-assignment operators
_AUGOP_PY: Dict[int, str] = {1: "+", 2: "-", 3: "*", 4: "//"}


class FSMToPythonPass(SynthPass):
    """Render each FSMModule to a Python ``@zdc.sync`` method body or full class.

    Single-state FSMs (``fsm.single_state is True``):
        Write ``@zdc.sync`` method body text to
        ``ir.lowered_py["py/module/sync"]``.
        ``ModuleAssemblePythonPass`` then wraps this into a full class.

    Multi-state FSMs:
        Write a complete ``@zdc.dataclass`` class to
        ``ir.lowered_py["py/module/top"]``.  ``ModuleAssemblePythonPass``
        is a no-op in this case.
    """

    @property
    def name(self) -> str:
        return "fsm_to_python"

    def run(self, ir: SynthIR) -> SynthIR:
        from zuspec.synth.sprtl.ir_to_python import ir_stmts_to_python

        if not ir.fsm_modules:
            ir.lowered_py["py/module/sync"] = ""
            return ir

        cf = ir.component_fields

        for fsm in ir.fsm_modules:
            if getattr(fsm, "single_state", False):
                body_lines = ir_stmts_to_python(
                    fsm.body_stmts,
                    fsm.body_idx_to_name,
                    indent=8,
                )
                method_lines = ["    @zdc.sync", "    def _count(self):"]
                if body_lines:
                    method_lines.extend(body_lines)
                else:
                    method_lines.append("        pass")
                ir.lowered_py["py/module/sync"] = "\n".join(method_lines)
                _log.debug("[FSMToPythonPass] single-state sync method emitted")

            else:
                emitter = _MultiStatePythonEmitter(fsm, cf, ir.component)
                py_src = emitter.emit()
                ir.lowered_py["py/module/top"] = py_src
                _log.debug(
                    "[FSMToPythonPass] multi-state full class %d chars for %r",
                    len(py_src),
                    fsm.name,
                )
                return ir  # ModuleAssemblePythonPass will no-op

        return ir


# ---------------------------------------------------------------------------
# Multi-state Python emitter
# ---------------------------------------------------------------------------


class _MultiStatePythonEmitter:
    """Emit a complete ``@zdc.dataclass`` class from a multi-state FSMModule.

    Follows the structural decisions of ``SVCodeGenerator`` but emits Python
    source instead of SystemVerilog.

    WAIT_CYCLES counter management
    --------------------------------
    In SV the wait counter is managed in a *separate* ``always_ff`` block:
    ``load N-1 on entry, decrement each cycle while in the state``.  In
    Python we inline this into the ``@zdc.sync`` body:

    * In states that transition **to** a WAIT_CYCLES state: inject
      ``self._CNT = N-1`` alongside the ``self._state = …`` assignment.
    * In the WAIT_CYCLES state body: the FSMCond in ``state.operations``
      already encodes ``if self._CNT == 0: …``.  We inject an ``else:``
      branch that decrements the counter.
    """

    def __init__(self, fsm, cf, component_cls):
        self._fsm = fsm
        self._cf = cf
        self._component_cls = component_cls
        self._buf = StringIO()
        self._indent_level = 0

        # Sets for name resolution
        self._port_names: Set[str] = {
            p.name for p in (getattr(fsm, "ports", None) or [])
        }
        self._reg_names: Set[str] = {
            r.name for r in (getattr(fsm, "registers", None) or [])
            if r.name != "state"
        }

        # Auto-inferred: FSMAssign targets that are not a port or register
        self._inferred_names: Set[str] = set()
        self._collect_inferred_names()

        # Deduplicated state names and encoding (shared with SV backend)
        from zuspec.synth.sprtl.fsm_structural import (
            fsm_state_names, fsm_state_width, fsm_wait_counter_info,
        )
        self._state_names: Dict[int, str] = fsm_state_names(fsm)

        # State encoding: state_id → encoded integer
        self._state_encoding: Dict[int, int] = dict(
            getattr(fsm, "state_encoding", {}) or {}
        )

        # State width
        self._state_width: int = fsm_state_width(fsm)

        # WAIT_CYCLES counter map: state_id → (counter_name, init_value, width)
        wc_infos = fsm_wait_counter_info(fsm, self._state_names)
        self._wc_counters: Dict[int, Tuple[str, int, int]] = {
            wci.state.id: (wci.counter_name, wci.init_val, wci.counter_width)
            for wci in wc_infos
        }

    # ------------------------------------------------------------------ #
    # Pre-scan helpers
    # ------------------------------------------------------------------ #

    def _collect_inferred_names(self):
        from zuspec.synth.sprtl.fsm_ir import FSMAssign, FSMCond

        def _walk(ops):
            for op in ops:
                if isinstance(op, FSMAssign) and isinstance(op.target, str):
                    name = op.target
                    if (
                        name != "state"
                        and name not in self._port_names
                        and name not in self._reg_names
                    ):
                        self._inferred_names.add(name)
                elif isinstance(op, FSMCond):
                    _walk(op.then_ops)
                    _walk(op.else_ops)

        for state in self._fsm.states:
            _walk(state.operations)

    # ------------------------------------------------------------------ #
    # Buffer helpers
    # ------------------------------------------------------------------ #

    def _emitraw(self, text: str):
        self._buf.write(text)

    def _emitln(self, line: str = "", indent: int = 0):
        sp = "    " * indent
        self._buf.write(sp + line + "\n")

    # ------------------------------------------------------------------ #
    # Signal name resolution
    # ------------------------------------------------------------------ #

    def _py_name(self, raw_name: str) -> str:
        """Map a raw FSM signal name to a Python ``self.X`` reference."""
        if raw_name == "state":
            return "self._state"
        if raw_name in self._port_names:
            return f"self.{raw_name}"
        return f"self._{raw_name}"

    # ------------------------------------------------------------------ #
    # Expression formatters
    # ------------------------------------------------------------------ #

    def _fmt_expr(self, expr: Any) -> str:
        if expr is None:
            return "0"
        if isinstance(expr, bool):
            return "True" if expr else "False"
        if isinstance(expr, (int, float)):
            return str(int(expr))
        if isinstance(expr, str):
            return self._py_name(expr)

        t = type(expr).__name__

        if t == "ExprRefLocal":
            return self._py_name(expr.name)

        if t == "ExprRefField":
            idx_map = getattr(self._fsm, "field_names", {}) or {}
            name = idx_map.get(expr.index, f"_f{expr.index}")
            return f"self.{name}"

        if t == "ExprConstant":
            v = expr.value
            if isinstance(v, bool):
                return "True" if v else "False"
            return str(v)

        if t == "ExprBin":
            lhs = self._fmt_expr(expr.lhs)
            rhs = self._fmt_expr(expr.rhs)
            op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
            py_op = _BINOP_PY.get(op_name, op_name)
            return f"({lhs} {py_op} {rhs})"

        if t == "ExprCompare":
            left = self._fmt_expr(expr.left)
            parts = [left]
            for op, comp in zip(expr.ops, expr.comparators):
                op_name = op.name if hasattr(op, "name") else str(op)
                # Strip enum wrapper if present
                if "." in op_name:
                    op_name = op_name.split(".")[-1].split(":")[0]
                py_op = _BINOP_PY.get(op_name, op_name)
                parts.extend([py_op, self._fmt_expr(comp)])
            return "(" + " ".join(parts) + ")"

        if t == "ExprUnary":
            operand = self._fmt_expr(expr.operand)
            op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
            from zuspec.synth.sprtl.ir_to_python import _UNOP_PY
            py_op = _UNOP_PY.get(op_name, op_name)
            return f"({py_op}{operand})"

        if t == "ExprSubscript":
            base = self._fmt_expr(expr.value)
            sl = expr.slice
            if type(sl).__name__ == "ExprConstant":
                return f"{base}[{sl.value}]"
            return f"{base}[{self._fmt_expr(sl)}]"

        if t == "ExprCall":
            func = expr.func
            args = getattr(expr, "args", [])

            if func is not None and type(func).__name__ == "ExprAttribute":
                _ZDC_CTOR_ATTRS = (
                    "bv", "bit", "u1", "u2", "u4", "u8", "u16", "u32", "u64",
                    "s8", "s16", "s32", "s64",
                )
                if func.attr in _ZDC_CTOR_ATTRS:
                    base_val = func.value
                    if (
                        base_val is not None
                        and type(base_val).__name__ == "ExprRefUnresolved"
                        and base_val.name == "zdc"
                        and len(args) == 1
                    ):
                        return f"zdc.{func.attr}({self._fmt_expr(args[0])})"
                if func.attr == "read" and func.value is not None:
                    return self._fmt_expr(func.value)

            if func is not None and type(func).__name__ == "ExprRefUnresolved":
                fname = func.name
                if fname == "int" and len(args) == 1:
                    return self._fmt_expr(args[0])
                if fname == "bool" and len(args) == 1:
                    return f"bool({self._fmt_expr(args[0])})"
                if fname == "_illegal":
                    return "# illegal"

            if func is not None:
                fn_str = self._fmt_expr(func)
                args_str = ", ".join(self._fmt_expr(arg) for arg in args)
                return f"{fn_str}({args_str})"

        # Fallback: try common attributes
        if hasattr(expr, "name"):
            return self._py_name(expr.name)
        if hasattr(expr, "value") and not callable(expr.value):
            return str(expr.value)
        return str(expr)

    def _fmt_condition(self, condition: Any) -> str:
        """Format a condition as a Python boolean expression."""
        if condition is None:
            return "True"
        if isinstance(condition, tuple) and len(condition) == 3:
            op_str, lhs, rhs = condition
            py_op = _TUPLE_OP_PY.get(str(op_str), str(op_str))
            lhs_py = self._fmt_expr(lhs)
            rhs_py = self._fmt_expr(rhs)
            return f"{lhs_py} {py_op} {rhs_py}"
        return self._fmt_expr(condition)

    def _fmt_value(self, value: Any) -> str:
        """Format an FSMAssign RHS as a Python expression."""
        if value is None:
            return "0"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return str(int(value))
        if isinstance(value, str):
            return self._py_name(value)
        if isinstance(value, tuple) and len(value) == 3:
            lhs, op, rhs = value
            lhs_str = self._fmt_expr(lhs)
            rhs_str = self._fmt_expr(rhs)
            t = type(op).__name__
            if t == "AugOp":
                op_str = _AUGOP_PY.get(op.value, str(op.value))
            else:
                op_str = _TUPLE_OP_PY.get(str(op), str(op))
            return f"{lhs_str} {op_str} {rhs_str}"
        return self._fmt_expr(value)

    # ------------------------------------------------------------------ #
    # Operation emitter
    # ------------------------------------------------------------------ #

    def _emit_ops(
        self,
        ops: list,
        indent: int,
        wc_counter: Optional[str] = None,
    ):
        """Emit FSMAssign / FSMCond operations.

        ``wc_counter`` — when set, an empty ``else_ops`` FSMCond whose
        condition references this counter name gets an injected
        ``self._CNT = self._CNT - 1`` else-branch.
        """
        from zuspec.synth.sprtl.fsm_ir import FSMAssign, FSMCond

        sp = "    " * indent

        for op in ops:
            if isinstance(op, FSMAssign):
                tgt = (
                    self._py_name(op.target)
                    if isinstance(op.target, str)
                    else self._fmt_expr(op.target)
                )
                val = self._fmt_value(op.value)
                self._buf.write(f"{sp}{tgt} = {val}\n")

            elif isinstance(op, FSMCond):
                cond_str = self._fmt_condition(op.condition)
                self._buf.write(f"{sp}if {cond_str}:\n")
                if op.then_ops:
                    self._emit_ops(op.then_ops, indent + 1)
                else:
                    self._buf.write(f"{sp}    pass\n")

                # Determine else branch
                if op.else_ops:
                    self._buf.write(f"{sp}else:\n")
                    self._emit_ops(op.else_ops, indent + 1)
                elif wc_counter is not None and self._cond_is_counter_zero(
                    op.condition, wc_counter
                ):
                    # Inject counter decrement for WAIT_CYCLES states
                    self._buf.write(f"{sp}else:\n")
                    self._buf.write(
                        f"{sp}    self._{wc_counter} = self._{wc_counter} - 1\n"
                    )

            else:
                self._buf.write(f"{sp}# <unsupported: {type(op).__name__}>\n")

    def _cond_is_counter_zero(self, condition: Any, counter_name: str) -> bool:
        """Return True if *condition* matches ``(eq, counter_name, 0)``."""
        if not isinstance(condition, tuple) or len(condition) != 3:
            return False
        op, lhs, rhs = condition
        if str(op) != "eq":
            return False
        lhs_name = lhs if isinstance(lhs, str) else getattr(lhs, "name", None)
        rhs_val = rhs if isinstance(rhs, (int, float)) else getattr(rhs, "value", None)
        return lhs_name == counter_name and rhs_val == 0

    def _emit_wait_cycles_state(self, state, counter_name: str, indent: int):
        """Emit a WAIT_CYCLES state as a merged if/else block.

        Merges the FSMCond body (from ``state.operations``) with the
        conditional transition so that both the body ops *and* the
        next-state assignment are inside the same ``if CNT == 0:`` block,
        with ``else: CNT -= 1``.  This avoids the correctness bug that
        arises from decrementing and then re-testing the counter.
        """
        from zuspec.synth.sprtl.fsm_ir import FSMAssign, FSMCond

        sp = "    " * indent

        # Collect the primary FSMCond (if any) and standalone assigns
        primary_cond: Optional[FSMCond] = None
        other_ops: list = []
        for op in state.operations:
            if isinstance(op, FSMCond) and primary_cond is None:
                primary_cond = op
            else:
                other_ops.append(op)

        # Collect transitions that match the counter-zero condition
        counter_zero_trs = [
            t for t in state.transitions
            if self._cond_is_counter_zero(t.condition, counter_name)
        ]
        other_trs = [
            t for t in state.transitions
            if not self._cond_is_counter_zero(t.condition, counter_name)
        ]

        # Emit the main if CNT == 0: block
        self._buf.write(f"{sp}if self._{counter_name} == 0:\n")
        body_written = False

        if primary_cond is not None:
            for op in primary_cond.then_ops:
                if isinstance(op, FSMAssign):
                    tgt = (
                        self._py_name(op.target)
                        if isinstance(op.target, str)
                        else self._fmt_expr(op.target)
                    )
                    val = self._fmt_value(op.value)
                    self._buf.write(f"{sp}    {tgt} = {val}\n")
                    body_written = True
                elif isinstance(op, FSMCond):
                    self._emit_ops([op], indent + 1)
                    body_written = True

        # Next-state assignments that fire when counter == 0
        for tr in counter_zero_trs:
            next_enc = self._state_encoding.get(tr.target_state, tr.target_state)
            wc_info = self._wc_counters.get(tr.target_state)
            self._buf.write(f"{sp}    self._state = {next_enc}\n")
            if wc_info:
                cname, init_val, _ = wc_info
                self._buf.write(f"{sp}    self._{cname} = {init_val}\n")
            body_written = True

        # Any other standalone ops in then-body
        for op in other_ops:
            self._emit_ops([op], indent + 1)
            body_written = True

        if not body_written:
            self._buf.write(f"{sp}    pass\n")

        # else: decrement counter
        self._buf.write(f"{sp}else:\n")
        self._buf.write(f"{sp}    self._{counter_name} = self._{counter_name} - 1\n")

        # Any remaining unconditional transitions (unusual, but handle gracefully)
        if other_trs:
            self._emit_transitions(other_trs, indent)

    # ------------------------------------------------------------------ #
    # Transition emitter
    # ------------------------------------------------------------------ #

    def _emit_transitions(self, transitions: list, indent: int):
        """Emit next-state assignments for a list of FSMTransition objects.

        For transitions targeting a WAIT_CYCLES state, inject the counter
        initialisation alongside the state assignment.

        Conditional transitions are emitted as ``if``/``elif`` chains;
        an unconditional fallback transition becomes the final ``else``.
        """
        sp = "    " * indent

        conditional = [t for t in transitions if t.condition is not None]
        unconditional = [t for t in transitions if t.condition is None]

        for i, tr in enumerate(conditional):
            next_enc = self._state_encoding.get(tr.target_state, tr.target_state)
            wc_info = self._wc_counters.get(tr.target_state)
            cond_str = self._fmt_condition(tr.condition)
            keyword = "elif" if i > 0 else "if"
            self._buf.write(f"{sp}{keyword} {cond_str}:\n")
            self._buf.write(f"{sp}    self._state = {next_enc}\n")
            if wc_info:
                cname, init_val, _ = wc_info
                self._buf.write(f"{sp}    self._{cname} = {init_val}\n")

        if unconditional:
            tr = unconditional[0]
            next_enc = self._state_encoding.get(tr.target_state, tr.target_state)
            wc_info = self._wc_counters.get(tr.target_state)
            if conditional:
                self._buf.write(f"{sp}else:\n")
                self._buf.write(f"{sp}    self._state = {next_enc}\n")
                if wc_info:
                    cname, init_val, _ = wc_info
                    self._buf.write(f"{sp}    self._{cname} = {init_val}\n")
            else:
                self._buf.write(f"{sp}self._state = {next_enc}\n")
                if wc_info:
                    cname, init_val, _ = wc_info
                    self._buf.write(f"{sp}self._{cname} = {init_val}\n")

    # ------------------------------------------------------------------ #
    # Helper: port/register widths
    # ------------------------------------------------------------------ #

    def _port_width(self, port_name: str) -> int:
        for p in (getattr(self._fsm, "ports", None) or []):
            if p.name == port_name:
                return getattr(p, "width", 1)
        return 1

    # ------------------------------------------------------------------ #
    # Main emit entry
    # ------------------------------------------------------------------ #

    def emit(self) -> str:
        fsm = self._fsm
        cf = self._cf

        self._buf.write("# Generated by zuspec-synth — IR round-trip reconstruction\n")
        self._buf.write("import zuspec.dataclasses as zdc\n")
        self._buf.write("\n")

        # Class declaration
        cls_name = (cf.module_name if cf else None) or fsm.name or "Module"
        self._buf.write("@zdc.dataclass\n")
        self._buf.write(f"class {cls_name}(zdc.SyncComponent):\n")

        # reset_domain (propagate from original component)
        rd = (
            getattr(self._component_cls, "reset_domain", None)
            if self._component_cls
            else None
        )
        if rd is not None:
            rd_style = getattr(getattr(rd, "style", None), "value", None)
            if rd_style == "none":
                self._buf.write('    reset_domain = zdc.ResetDomain(style="none")\n')
                self._buf.write("\n")

        # Port declarations
        self._buf.write("    # Ports\n")
        cf_ports_map: Dict[str, Any] = {}
        if cf:
            for p in cf.ports:
                if p.name not in ("clk", "rst_n"):
                    cf_ports_map[p.name] = p

        for pname in self._port_names:
            p = cf_ports_map.get(pname)
            if p is None:
                continue
            w = p.width
            type_ann = f"zdc.bv" if w > 1 else "zdc.bit"
            if p.direction == "output":
                decl = f"zdc.output(width={w})" if w > 1 else "zdc.output()"
            else:
                decl = f"zdc.input(width={w})" if w > 1 else "zdc.input()"
            self._buf.write(f"    {pname}: {type_ann} = {decl}\n")
        self._buf.write("\n")

        # State register
        self._buf.write("    # State register\n")
        state_reg = next(
            (
                r
                for r in (getattr(fsm, "registers", None) or [])
                if r.name == "state"
            ),
            None,
        )
        state_width = getattr(state_reg, "width", self._state_width)
        self._buf.write(f"    _state: zdc.bv = zdc.field(width={state_width})\n")
        self._buf.write("\n")

        # Wait-counter registers
        wait_counters = [
            r
            for r in (getattr(fsm, "registers", None) or [])
            if r.name != "state"
        ]
        if wait_counters:
            self._buf.write("    # Wait-cycle counter registers\n")
            for r in wait_counters:
                w = getattr(r, "width", 1)
                self._buf.write(f"    _{r.name}: zdc.bv = zdc.field(width={w})\n")
            self._buf.write("\n")

        # Auto-inferred registers
        if self._inferred_names:
            self._buf.write("    # Auto-inferred registers\n")
            for name in sorted(self._inferred_names):
                self._buf.write(f"    _{name}: zdc.bv = zdc.field(width=32)\n")
            self._buf.write("\n")

        # @zdc.sync method — state machine body
        self._buf.write("    @zdc.sync\n")
        self._buf.write("    def _fsm(self):\n")

        # Build encoding → state mapping (sorted for deterministic output)
        enc_to_state: Dict[int, Any] = {}
        for state in fsm.states:
            enc = self._state_encoding.get(state.id, state.id)
            enc_to_state[enc] = state

        for branch_idx, (enc_val, state) in enumerate(sorted(enc_to_state.items())):
            keyword = "if" if branch_idx == 0 else "elif"
            self._buf.write(
                f"        {keyword} self._state == {enc_val}:  # {state.name}\n"
            )

            # Determine wait-cycles counter name for this state (if WAIT_CYCLES)
            from zuspec.synth.sprtl.fsm_ir import FSMStateKind

            wc_counter_name: Optional[str] = None
            if state.kind == FSMStateKind.WAIT_CYCLES:
                wc_info = self._wc_counters.get(state.id)
                if wc_info:
                    wc_counter_name = wc_info[0]

            state_body_written = False

            if state.kind == FSMStateKind.WAIT_CYCLES and wc_counter_name:
                # Merge operations + transitions into a single if/else block.
                # The FSMCond in operations has condition ``CNT == 0``.
                # The conditional transition also has ``CNT == 0``.
                # We emit:  if CNT == 0: {ops} {next-state}
                #           else: CNT -= 1
                self._emit_wait_cycles_state(state, wc_counter_name, indent=3)
                state_body_written = True

            else:
                if state.operations:
                    self._emit_ops(state.operations, indent=3, wc_counter=None)
                    state_body_written = True

                if state.transitions:
                    self._emit_transitions(state.transitions, indent=3)
                    state_body_written = True

            if not state_body_written:
                self._buf.write("            pass\n")

        self._buf.write("\n")

        return self._buf.getvalue()
