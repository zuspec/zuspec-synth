"""ExprLowerer — lower Python AST expressions and statements to Verilog 2005.

Used by :class:`~zuspec.synth.passes.pipeline_sv_emit.PipelineSVCodegen` to
emit the body of each stage's ``always @(*)`` combinational block.

Variable naming convention
--------------------------
- A variable *defined* in stage ``S`` (with stage name in lowercase = ``s``) is
  named ``{var}_{s}`` in the generated SV.
- A variable *read* from an earlier stage via a pipeline register (a
  :class:`~zuspec.synth.ir.pipeline_ir.ChannelDecl`) is named ``{ch.name}_q``
  (e.g. ``a_if_to_ex_q``).
- A component port ``self.X`` is just ``X``.

Width resolution
----------------
The lowerer resolves widths from the ``annotation_map`` stored in
:class:`~zuspec.synth.ir.pipeline_ir.PipelineIR`.  ``zdc.u8/u16/u32/u64`` and
the signed equivalents ``i8/i16/i32/i64`` are recognised; everything else
defaults to 32 bits.
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from zuspec.synth.ir.pipeline_ir import ChannelDecl, PipelineIR, RegFileAccess, StageIR

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regfile call detection helpers
# ---------------------------------------------------------------------------

def _is_regfile_call(node: ast.expr, method: str) -> bool:
    """Return True if *node* is ``self.FIELD.METHOD(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
    )


def _regfile_field(node: ast.Call) -> str:
    """Return the field name from a ``self.FIELD.read/write(...)`` call."""
    return node.func.value.attr  # type: ignore[union-attr]


def is_regfile_read_stmt(stmt: ast.stmt) -> bool:
    """Return True if *stmt* is ``var = self.FIELD.read(addr)`` or annotated form."""
    if isinstance(stmt, ast.AnnAssign):
        return stmt.value is not None and _is_regfile_call(stmt.value, "read")
    if isinstance(stmt, ast.Assign) and stmt.targets:
        return _is_regfile_call(stmt.value, "read")
    return False


def is_regfile_write_stmt(stmt: ast.stmt) -> bool:
    """Return True if *stmt* is a bare ``self.FIELD.write(addr, data)`` call."""
    return (
        isinstance(stmt, ast.Expr)
        and _is_regfile_call(stmt.value, "write")
    )

# ---------------------------------------------------------------------------
# Width helpers
# ---------------------------------------------------------------------------

_WIDTH_MAP: Dict[str, int] = {
    "bit": 1,
    "u8": 8,  "u16": 16,  "u32": 32,  "u64": 64,
    "i8": 8,  "i16": 16,  "i32": 32,  "i64": 64,
    "bool_t": 1, "bool": 1,
}


def _get_sv_width(annotation: Optional[ast.expr]) -> int:
    """Return the bit-width for a zdc type annotation AST node."""
    if annotation is None:
        return 32
    if isinstance(annotation, ast.Attribute):
        name = annotation.attr
    elif isinstance(annotation, ast.Name):
        name = annotation.id
    elif isinstance(annotation, ast.Subscript):
        # e.g. zdc.Array[u32, 4] — use element type
        return _get_sv_width(annotation.value)
    else:
        return 32
    return _WIDTH_MAP.get(name, 32)


# ---------------------------------------------------------------------------
# Port scanner — collect self.X reads / writes across all stages
# ---------------------------------------------------------------------------

class _PortScanner(ast.NodeVisitor):
    """Scan stage operations and collect component I/O ports."""

    def __init__(
        self,
        annotation_map: Dict[str, Any],
        port_widths: Dict[str, int],
        regfile_fields: Optional[Set[str]] = None,
    ) -> None:
        self._ann = annotation_map
        self._port_widths = port_widths
        self._regfile_fields: Set[str] = regfile_fields or set()
        self.inputs:  Dict[str, int] = {}   # port_name → width
        self.outputs: Dict[str, int] = {}   # port_name → width
        # Track AST ids that are output targets so we don't double-count
        self._output_nodes: Set[int] = set()

    def _port_width(self, port_name: str) -> int:
        """Return the width for a named port, from class fields or default 32."""
        return self._port_widths.get(port_name, 32)

    def visit_Assign(self, node: ast.Assign) -> None:
        # First pass: mark output targets (self.X = ...)
        for t in node.targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                port = t.attr
                w = self._port_width(port)
                # Refine from rvalue variable annotation if available
                if isinstance(node.value, ast.Name):
                    ann = self._ann.get(node.value.id)
                    if ann is not None:
                        w = _get_sv_width(ann)
                self.outputs.setdefault(port, w)
                self._output_nodes.add(id(t))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # self.X appearing as a read expression → input port
        # Skip self.FIELD references where FIELD is a regfile — those are
        # not ports, they're inlined as mem arrays by the emitter.
        if (
            id(node) not in self._output_nodes
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr not in self._regfile_fields
        ):
            w = self._port_width(node.attr)
            self.inputs.setdefault(node.attr, w)
        self.generic_visit(node)


def collect_ports(
    pip: PipelineIR,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Return ``(inputs, outputs)`` — each a list of ``(port_name, width)``.

    ``IndexedRegFile`` fields are excluded from the port list since they are
    emitted as inline ``reg`` arrays, not as module I/O.
    """
    regfile_fields = {d.field_name for d in getattr(pip, "regfile_decls", [])}
    scanner = _PortScanner(pip.annotation_map, pip.port_widths, regfile_fields)
    for stage in pip.stages:
        for stmt in stage.operations:
            scanner.visit(stmt)
    return list(scanner.inputs.items()), list(scanner.outputs.items())


# ---------------------------------------------------------------------------
# Per-stage expression / statement lowerer
# ---------------------------------------------------------------------------

# Operator tables
_BINOP: Dict[type, str] = {
    ast.Add:      "+",
    ast.Sub:      "-",
    ast.Mult:     "*",
    ast.Div:      "/",
    ast.FloorDiv: "/",
    ast.Mod:      "%",
    ast.BitAnd:   "&",
    ast.BitOr:    "|",
    ast.BitXor:   "^",
    ast.LShift:   "<<",
    ast.RShift:   ">>",
}

_UNOP: Dict[type, str] = {
    ast.USub:   "-",
    ast.UAdd:   "+",
    ast.Invert: "~",
    ast.Not:    "!",
}

_CMP: Dict[type, str] = {
    ast.Eq:    "==",
    ast.NotEq: "!=",
    ast.Lt:    "<",
    ast.LtE:   "<=",
    ast.Gt:    ">",
    ast.GtE:   ">=",
}

_BOOLOP: Dict[type, str] = {
    ast.And: "&&",
    ast.Or:  "||",
}


class ExprLowerer:
    """Lower Python AST expressions and statements to Verilog 2005 strings.

    Parameters
    ----------
    stage:
        The :class:`~zuspec.synth.ir.pipeline_ir.StageIR` being lowered.
    pip:
        The full :class:`~zuspec.synth.ir.pipeline_ir.PipelineIR`; used to
        look up pipeline-register names for variables crossing stage boundaries.
    """

    def __init__(self, stage: StageIR, pip: PipelineIR) -> None:
        self.stage = stage
        self.pip   = pip
        self._stage_lower = stage.name.lower()
        self._ann = pip.annotation_map

        # Map: var_name → SV signal for reading that var in this stage.
        # Populated from stage.inputs (pipeline register reads).
        self._reg_read: Dict[str, str] = {}
        for ch in stage.inputs:
            suffix = f"_{ch.src_stage.lower()}_to_{ch.dst_stage.lower()}"
            var_name = ch.name[: -len(suffix)] if ch.name.endswith(suffix) else ch.name
            self._reg_read[var_name] = f"{ch.name}_q"

        # Track variables defined in *this* stage so they take priority.
        self._defined: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lower_expr(self, node: ast.expr) -> str:
        """Return the SV expression string for *node*."""
        if isinstance(node, ast.BinOp):
            op  = _BINOP.get(type(node.op), "/* op? */")
            lhs = self.lower_expr(node.left)
            rhs = self.lower_expr(node.right)
            return f"({lhs} {op} {rhs})"

        if isinstance(node, ast.UnaryOp):
            op      = _UNOP.get(type(node.op), "/* uop? */")
            operand = self.lower_expr(node.operand)
            return f"({op}{operand})"

        if isinstance(node, ast.BoolOp):
            op    = _BOOLOP.get(type(node.op), "&&")
            parts = [f"({self.lower_expr(v)})" for v in node.values]
            return f" {op} ".join(parts)

        if isinstance(node, ast.Compare):
            lhs   = self.lower_expr(node.left)
            parts = [lhs]
            for cmp_op, cmp_val in zip(node.ops, node.comparators):
                parts.append(_CMP.get(type(cmp_op), "=="))
                parts.append(self.lower_expr(cmp_val))
            return " ".join(parts)

        if isinstance(node, ast.IfExp):
            cond  = self.lower_expr(node.test)
            then  = self.lower_expr(node.body)
            else_ = self.lower_expr(node.orelse)
            return f"({cond} ? {then} : {else_})"

        if isinstance(node, ast.Subscript):
            base = self.lower_expr(node.value)
            sl   = self.lower_expr(node.slice)  # type: ignore[arg-type]
            return f"{base}[{sl}]"

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return node.attr   # component port → bare name
            return f"{self.lower_expr(node.value)}__{node.attr}"

        if isinstance(node, ast.Name):
            return self._resolve_var(node.id)

        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return "1'b1" if v else "1'b0"
            if isinstance(v, int):
                return str(v)
            return f"/* {v!r} */"

        if isinstance(node, ast.Tuple):
            return "{" + ", ".join(self.lower_expr(e) for e in node.elts) + "}"

        return f"/* TODO: {ast.dump(node)[:80]} */"

    def lower_stmts(self, stmts: List[ast.stmt]) -> List[str]:
        """Return a flat list of SV source lines for *stmts*.

        Regfile read/write statements are skipped — handled by dedicated emitters.
        """
        lines: List[str] = []
        for stmt in stmts:
            if is_regfile_read_stmt(stmt) or is_regfile_write_stmt(stmt):
                continue
            lines.extend(self._lower_stmt(stmt))
        return lines

    def collect_signals(self, stmts: List[ast.stmt]) -> List[Tuple[str, int]]:
        """Return ``[(sig_name, width)]`` for all signals defined in *stmts*.

        Regfile read result variables are excluded — they are declared by the
        ``_emit_regfile_read_muxes`` emitter as module-scope ``reg`` signals.

        Does not modify internal state.  Used to pre-declare signals at module
        scope before emitting procedural always blocks.
        """
        result: List[Tuple[str, int]] = []
        for stmt in stmts:
            if is_regfile_read_stmt(stmt) or is_regfile_write_stmt(stmt):
                continue
            for name, width in self._collect_signals_stmt(stmt):
                result.append((name, width))
        return result

    def collect_output_ports(self, stmts: List[ast.stmt]) -> List[Tuple[str, int]]:
        """Return ``[(port_name, width)]`` for all ``self.X = ...`` assignments.

        Used to generate default-zero assignments in the ``else`` branch of the
        valid guard, preventing output-port latches.
        """
        result: List[Tuple[str, int]] = []
        for stmt in stmts:
            if not isinstance(stmt, ast.Assign):
                continue
            for t in stmt.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    w = 32
                    if isinstance(stmt.value, ast.Name):
                        ann = self._ann.get(stmt.value.id)
                        if ann is not None:
                            w = _get_sv_width(ann)
                    result.append((t.attr, w))
        return result

    def lower_stmts_procedural(self, stmts: List[ast.stmt]) -> List[str]:
        """Lower *stmts* to blocking-assignment SV lines (no ``wire``/``assign``).

        Suitable for use inside ``always @(*)`` procedural blocks.  All signal
        declarations are assumed to have been emitted at module scope via
        :meth:`collect_signals`.

        Regfile read/write statements are skipped here — they are emitted by
        the dedicated ``_emit_regfile_*`` methods in :class:`PipelineSVCodegen`.
        """
        lines: List[str] = []
        for stmt in stmts:
            if is_regfile_read_stmt(stmt) or is_regfile_write_stmt(stmt):
                continue  # handled by _emit_regfile_* in PipelineSVCodegen
            lines.extend(self._lower_stmt_proc(stmt))
        return lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_var(self, name: str) -> str:
        """Map a Python variable name to its SV signal name in this stage."""
        # Locally defined variable takes priority
        if name in self._defined:
            return f"{name}_{self._stage_lower}"
        # Variable arriving via pipeline register
        if name in self._reg_read:
            return self._reg_read[name]
        # Unresolved — leave as-is (may be a loop variable, builtin, etc.)
        return name

    def _lower_stmt(self, node: ast.stmt) -> List[str]:
        # ── Annotated assignment: var: type = expr ──────────────────────
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            width    = _get_sv_width(node.annotation)
            sig      = f"{var_name}_{self._stage_lower}"
            self._defined.add(var_name)
            rhs = self.lower_expr(node.value) if node.value else "/* undef */"
            return [
                f"wire [{width - 1}:0] {sig};",
                f"assign {sig} = {rhs};",
            ]

        # ── Plain assignment ─────────────────────────────────────────────
        if isinstance(node, ast.Assign):
            lines: List[str] = []
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    # self.out = val  →  output port assignment
                    rhs = self.lower_expr(node.value)
                    lines.append(f"assign {t.attr} = {rhs};")
                elif isinstance(t, ast.Name):
                    var_name = t.id
                    self._defined.add(var_name)
                    sig = f"{var_name}_{self._stage_lower}"
                    # Infer width from annotation_map
                    ann = self._ann.get(var_name)
                    width = _get_sv_width(ann) if ann is not None else 32
                    rhs = self.lower_expr(node.value)
                    lines += [
                        f"wire [{width - 1}:0] {sig};",
                        f"assign {sig} = {rhs};",
                    ]
                else:
                    lines.append(f"/* TODO: assign target {ast.dump(t)[:60]} */")
            return lines

        # ── Augmented assignment: var op= expr ──────────────────────────
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            op       = _BINOP.get(type(node.op), "+")
            sig      = f"{var_name}_{self._stage_lower}"
            prev     = self._resolve_var(var_name)
            rhs      = self.lower_expr(node.value)
            ann      = self._ann.get(var_name)
            width    = _get_sv_width(ann) if ann is not None else 32
            self._defined.add(var_name)
            return [
                f"wire [{width - 1}:0] {sig};",
                f"assign {sig} = ({prev} {op} {rhs});",
            ]

        # ── If statement ─────────────────────────────────────────────────
        if isinstance(node, ast.If):
            lines = [f"if ({self.lower_expr(node.test)}) begin"]
            for s in node.body:
                for ln in self._lower_stmt(s):
                    lines.append("    " + ln)
            if node.orelse:
                lines.append("end else begin")
                for s in node.orelse:
                    for ln in self._lower_stmt(s):
                        lines.append("    " + ln)
            lines.append("end")
            return lines

        # ── Skip: pass, bare expression (e.g. function calls), return ───
        if isinstance(node, (ast.Pass, ast.Expr, ast.Return)):
            return []

        return [f"/* TODO: {type(node).__name__} */"]

    # ------------------------------------------------------------------
    # Procedural (blocking-assignment) lowering
    # ------------------------------------------------------------------

    def _lower_stmt_proc(self, node: ast.stmt) -> List[str]:
        """Lower *node* to blocking SV assignments (no wire/assign keywords)."""
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            sig      = f"{var_name}_{self._stage_lower}"
            self._defined.add(var_name)
            rhs = self.lower_expr(node.value) if node.value else "0"
            return [f"{sig} = {rhs};"]

        if isinstance(node, ast.Assign):
            lines: List[str] = []
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    rhs = self.lower_expr(node.value)
                    lines.append(f"{t.attr} = {rhs};")
                elif isinstance(t, ast.Name):
                    var_name = t.id
                    self._defined.add(var_name)
                    sig = f"{var_name}_{self._stage_lower}"
                    rhs = self.lower_expr(node.value)
                    lines.append(f"{sig} = {rhs};")
                else:
                    lines.append(f"/* TODO: assign target {ast.dump(t)[:60]} */")
            return lines

        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            op       = _BINOP.get(type(node.op), "+")
            sig      = f"{var_name}_{self._stage_lower}"
            prev     = self._resolve_var(var_name)
            rhs      = self.lower_expr(node.value)
            self._defined.add(var_name)
            return [f"{sig} = ({prev} {op} {rhs});"]

        if isinstance(node, ast.If):
            lines = [f"if ({self.lower_expr(node.test)}) begin"]
            for s in node.body:
                for ln in self._lower_stmt_proc(s):
                    lines.append("    " + ln)
            if node.orelse:
                lines.append("end else begin")
                for s in node.orelse:
                    for ln in self._lower_stmt_proc(s):
                        lines.append("    " + ln)
            lines.append("end")
            return lines

        if isinstance(node, (ast.Pass, ast.Expr, ast.Return)):
            return []

        return [f"/* TODO: {type(node).__name__} */"]

    def _collect_signals_stmt(self, node: ast.stmt) -> List[Tuple[str, int]]:
        """Return ``[(sig_name, width)]`` defined by *node* (no recursion into if bodies)."""
        result: List[Tuple[str, int]] = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            sig      = f"{var_name}_{self._stage_lower}"
            width    = _get_sv_width(node.annotation)
            result.append((sig, width))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id != "self":
                    var_name = t.id
                    sig      = f"{var_name}_{self._stage_lower}"
                    ann      = self._ann.get(var_name)
                    width    = _get_sv_width(ann) if ann else 32
                    result.append((sig, width))
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            sig      = f"{var_name}_{self._stage_lower}"
            ann      = self._ann.get(var_name)
            width    = _get_sv_width(ann) if ann else 32
            result.append((sig, width))
        elif isinstance(node, ast.If):
            for s in node.body + node.orelse:
                result.extend(self._collect_signals_stmt(s))
        return result
