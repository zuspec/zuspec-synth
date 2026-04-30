"""Struct IR rewriter: rewrite flat ``_action_result_*`` registers to typed struct IR.

Given ``flat_to_struct`` from ``struct_annotator.build_struct_metadata``, this
pass mutates ``FSMModule`` in place:

1. Populates ``fsm.user_structs`` and ``fsm.struct_instances``.
2. Removes from ``fsm.registers`` every entry whose name is a flat struct
   prefix (2-part) or a flat struct field (3-part prefix + field suffix).
3. Rewrites every ``FSMAssign.target`` string → ``ExprStructField`` or
   ``ExprStructRef``.
4. Recursively rewrites IR value expressions:
   - ``ExprRefUnresolved/ExprRefLocal(name=flat_prefix)`` → ``ExprStructRef``
   - ``ExprAttribute(base=flat_prefix_ref, attr=field)`` → ``ExprStructField``
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .fsm_ir import (
    FSMModule, FSMAssign, FSMCond, FSMState, FSMStructDef, FSMStructInstance,
    ExprStructField, ExprStructRef,
)


# ---------------------------------------------------------------------------
# Expression rewriting
# ---------------------------------------------------------------------------

def _extract_flat_name(expr: Any) -> Optional[str]:
    """Return the flat string name if expr is an ExprRefUnresolved or ExprRefLocal."""
    t = type(expr).__name__
    if t in ("ExprRefUnresolved", "ExprRefLocal"):
        return getattr(expr, "name", None)
    return None


def _rewrite_expr(expr: Any, flat_to_struct: Dict[str, FSMStructInstance]) -> Any:
    """Recursively rewrite struct-prefix references in an IR expression.

    Returns a (possibly new) expression with ``ExprStructField`` /
    ``ExprStructRef`` substituted where appropriate.
    """
    if expr is None:
        return expr

    # Plain string values — rewrite if they match a struct prefix or field
    if isinstance(expr, str):
        if expr in flat_to_struct:
            return ExprStructRef(instance_name=flat_to_struct[expr].instance_name)
        for prefix, si in sorted(flat_to_struct.items(), key=lambda kv: -len(kv[0])):
            if expr.startswith(prefix + "_"):
                field_name = expr[len(prefix) + 1:]
                return ExprStructField(instance_name=si.instance_name, field_name=field_name)
        return expr

    t = type(expr).__name__

    # ExprAttribute: base.attr
    # Handles both:
    #   ExprAttribute(ExprRefLocal("_action_result_fetch"), "out") → ExprStructRef("fetch_out")
    #   ExprAttribute(ExprRefLocal("_action_result_dec_fetch"), "instr") → ExprStructField("dec_fetch","instr")
    #   ExprAttribute(<rewritten to ExprStructRef("x")>, "field") → ExprStructField("x","field")
    if t == "ExprAttribute":
        value = getattr(expr, "value", None)
        attr = getattr(expr, "attr", "")
        base_name = _extract_flat_name(value)
        if base_name:
            # 2-level: base_name + "_" + attr matches a flat_to_struct key → whole-struct ref
            combined = f"{base_name}_{attr}"
            if combined in flat_to_struct:
                return ExprStructRef(instance_name=flat_to_struct[combined].instance_name)
            # 1-level: base_name directly matches → field ref on that struct
            if base_name in flat_to_struct:
                return ExprStructField(
                    instance_name=flat_to_struct[base_name].instance_name,
                    field_name=attr,
                )
        # Recurse — handles nested ExprAttribute chains
        new_value = _rewrite_expr(value, flat_to_struct)
        if isinstance(new_value, ExprStructRef):
            # Inner node resolved to a struct instance; this attr is a field on it.
            return ExprStructField(instance_name=new_value.instance_name, field_name=attr)
        if new_value is not value:
            try:
                expr.value = new_value
            except (AttributeError, TypeError):
                pass
        return expr

    # Name reference — may be a whole-struct or field reference
    if t in ("ExprRefUnresolved", "ExprRefLocal"):
        name = getattr(expr, "name", "")
        if name in flat_to_struct:
            return ExprStructRef(instance_name=flat_to_struct[name].instance_name)
        # 3-part flat field name: _action_result_dec_fetch_instr → dec_fetch.instr
        for prefix, si in sorted(flat_to_struct.items(), key=lambda kv: -len(kv[0])):
            if name.startswith(prefix + "_"):
                field_name = name[len(prefix) + 1:]
                return ExprStructField(instance_name=si.instance_name, field_name=field_name)
        return expr

    # ExprCompare has a single 'left' AND a list 'comparators'; handle it before the
    # generic dispatch tables to avoid early-return conflicts.
    if t == "ExprCompare":
        left = getattr(expr, "left", None)
        if left is not None:
            new_left = _rewrite_expr(left, flat_to_struct)
            if new_left is not left:
                try:
                    expr.left = new_left
                except (AttributeError, TypeError):
                    pass
        comparators = getattr(expr, "comparators", None)
        if isinstance(comparators, list):
            new_comps = [_rewrite_expr(c, flat_to_struct) for c in comparators]
            try:
                expr.comparators = new_comps
            except (AttributeError, TypeError):
                pass
        return expr

    # Recurse into known compound expression types
    _LIST_ATTRS = {
        "ExprBool": ["values"],
        "ExprConcat": ["parts"],
        "ExprCall": ["args"],
    }
    _SINGLE_ATTRS = {
        "ExprBin": ["lhs", "rhs"],
        "ExprUnary": ["operand"],
        "ExprCond": ["test", "body", "orelse"],
        "ExprSubscript": ["value", "slice"],
        "ExprSlice": ["upper", "lower"],
        "ExprSext": ["value"],
        "ExprZext": ["value"],
        "ExprCbit": ["value"],
        "ExprSigned": ["value"],
    }

    list_attrs = _LIST_ATTRS.get(t)
    if list_attrs:
        for attr in list_attrs:
            lst = getattr(expr, attr, None)
            if isinstance(lst, list):
                new_lst = [_rewrite_expr(v, flat_to_struct) for v in lst]
                try:
                    setattr(expr, attr, new_lst)
                except (AttributeError, TypeError):
                    pass
        return expr

    single_attrs = _SINGLE_ATTRS.get(t)
    if single_attrs:
        for attr in single_attrs:
            child = getattr(expr, attr, None)
            if child is not None:
                new_child = _rewrite_expr(child, flat_to_struct)
                if new_child is not child:
                    try:
                        setattr(expr, attr, new_child)
                    except (AttributeError, TypeError):
                        pass
        return expr

    return expr


# ---------------------------------------------------------------------------
# Target rewriting
# ---------------------------------------------------------------------------

def _rewrite_target(
    target: str,
    flat_to_struct: Dict[str, FSMStructInstance],
) -> Any:
    """Rewrite a plain string assignment target to an IR struct node if applicable.

    3-part flat field: ``_action_result_dec_out_funct3``
      → ``ExprStructField("dec_out", "funct3")``

    2-part flat struct ref: ``_action_result_dec_out``
      → ``ExprStructRef("dec_out")``

    Returns the original string if no match.
    """
    if target in flat_to_struct:
        return ExprStructRef(instance_name=flat_to_struct[target].instance_name)

    # Try 3-part: find the longest flat_prefix that is a prefix of target+"_"
    # (sort by length descending to prefer the most specific match)
    for prefix, si in sorted(flat_to_struct.items(), key=lambda kv: -len(kv[0])):
        sep = prefix + "_"
        if target.startswith(sep):
            field_name = target[len(sep):]
            return ExprStructField(instance_name=si.instance_name, field_name=field_name)

    return target


# ---------------------------------------------------------------------------
# Operation / state rewriting
# ---------------------------------------------------------------------------

def _rewrite_ops(ops: list, flat_to_struct: Dict[str, FSMStructInstance]) -> None:
    """Rewrite all operations in a list in place."""
    for op in ops:
        if isinstance(op, FSMAssign):
            if isinstance(op.target, str):
                op.target = _rewrite_target(op.target, flat_to_struct)
            op.value = _rewrite_expr(op.value, flat_to_struct)
        elif isinstance(op, FSMCond):
            op.condition = _rewrite_expr(op.condition, flat_to_struct)
            _rewrite_ops(op.then_ops, flat_to_struct)
            _rewrite_ops(op.else_ops, flat_to_struct)
        else:
            # Generic: try rewriting any 'value' attribute
            if hasattr(op, "value"):
                try:
                    op.value = _rewrite_expr(op.value, flat_to_struct)
                except (AttributeError, TypeError):
                    pass
            if hasattr(op, "arg_exprs"):
                try:
                    op.arg_exprs = [_rewrite_expr(e, flat_to_struct) for e in op.arg_exprs]
                except (AttributeError, TypeError):
                    pass


# ---------------------------------------------------------------------------
# Register filtering
# ---------------------------------------------------------------------------

def _is_struct_register(name: str, flat_to_struct: Dict[str, FSMStructInstance]) -> bool:
    """Return True if register *name* is a flat struct prefix or flat struct field."""
    if name in flat_to_struct:
        return True
    for prefix in flat_to_struct:
        if name.startswith(prefix + "_"):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_struct_ir(
    fsm: FSMModule,
    struct_defs: List[FSMStructDef],
    flat_to_struct: Dict[str, FSMStructInstance],
) -> None:
    """Mutate *fsm* in place to use struct IR nodes.

    After this call:
    - ``fsm.user_structs`` is populated with unique struct type definitions.
    - ``fsm.struct_instances`` lists all struct register instances.
    - Flat ``_action_result_*`` entries are removed from ``fsm.registers``.
    - All ``FSMAssign.target`` and value expressions are rewritten.
    """
    if not flat_to_struct:
        return

    # 1. Populate struct definitions and instances on the FSM module
    fsm.user_structs = struct_defs

    # Deduplicate instances (same instance_name may appear from multiple walkers)
    seen_instances: Set[str] = set()
    for si in flat_to_struct.values():
        if si.instance_name not in seen_instances:
            fsm.struct_instances.append(si)
            seen_instances.add(si.instance_name)

    # 2. Remove flat registers that are now represented by struct instances
    fsm.registers = [
        r for r in fsm.registers
        if not _is_struct_register(r.name, flat_to_struct)
    ]

    # 3. Rewrite all assignment targets and value expressions in every state
    for state in fsm.states:
        _rewrite_ops(state.operations, flat_to_struct)

    # 4. Rewrite transition conditions
    for state in fsm.states:
        for trans in state.transitions:
            if trans.condition is not None:
                trans.condition = _rewrite_expr(trans.condition, flat_to_struct)
