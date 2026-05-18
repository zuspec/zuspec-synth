"""Tests for ir_to_python.py — Phase 1: Python expression/statement emitter.

Uses the blinky SPL example to drive integration tests.  Unit tests
construct minimal IR nodes directly without DataModelFactory.
"""
from __future__ import annotations

import os
import sys

import pytest

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
_blinky_dir = os.path.join(_this_dir, "..", "..", "..", "design", "spl", "blinky")

for _p in [_synth_src, _dc_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zuspec.synth.sprtl.ir_to_python import ir_expr_to_python, ir_stmts_to_python


# ---------------------------------------------------------------------------
# Minimal stub IR nodes for unit tests
# ---------------------------------------------------------------------------

class _Node:
    """Simple stub IR node."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.__class__.__name__ = kw.pop('_type', self.__class__.__name__)


def _make_node(type_name: str, **kw):
    """Create a stub node with the given type name and attributes."""
    obj = object.__new__(type(_Node_named(type_name), (), {}))
    for k, v in kw.items():
        setattr(obj, k, v)
    return obj


class _Meta(type):
    def __new__(mcs, name, bases, ns):
        return super().__new__(mcs, name, bases, ns)


def _Node_named(type_name: str):
    """Return a class whose __name__ is type_name."""
    return type(type_name, (), {})


def _ref(index: int):
    cls = _Node_named("ExprRefField")
    obj = cls()
    obj.index = index
    return obj


def _const(value):
    cls = _Node_named("ExprConstant")
    obj = cls()
    obj.value = value
    return obj


def _binop(lhs, rhs, op_name: str):
    cls = _Node_named("ExprBin")
    obj = cls()
    obj.lhs = lhs
    obj.rhs = rhs
    op_cls = type(op_name, (), {"name": op_name})
    obj.op = op_cls()
    return obj


def _unary(operand, op_name: str):
    cls = _Node_named("ExprUnary")
    obj = cls()
    obj.operand = operand
    op_cls = type(op_name, (), {"name": op_name})
    obj.op = op_cls()
    return obj


def _subscript(base, index):
    cls = _Node_named("ExprSubscript")
    obj = cls()
    obj.value = base
    obj.slice = _const(index)
    return obj


def _stmt_assign(target, value):
    cls = _Node_named("StmtAssign")
    obj = cls()
    obj.targets = [target]
    obj.value = value
    return obj


def _stmt_augassign(target, value, op_name: str):
    cls = _Node_named("StmtAugAssign")
    obj = cls()
    obj.target = target
    obj.value = value
    op_cls = type(op_name, (), {"name": op_name})
    obj.op = op_cls()
    return obj


def _stmt_if(test, body, orelse=None):
    cls = _Node_named("StmtIf")
    obj = cls()
    obj.test = test
    obj.body = body
    obj.orelse = orelse or []
    return obj


# ---------------------------------------------------------------------------
# Unit tests: expressions
# ---------------------------------------------------------------------------

class TestExprRefField:
    def test_ref_field_with_self_prefix(self):
        expr = _ref(0)
        result = ir_expr_to_python(expr, {0: "counter"})
        assert result == "self.counter"

    def test_ref_field_without_self_prefix(self):
        expr = _ref(0)
        result = ir_expr_to_python(expr, {0: "counter"}, self_prefix=False)
        assert result == "counter"

    def test_ref_field_unknown_index(self):
        expr = _ref(99)
        result = ir_expr_to_python(expr, {})
        assert result == "self._f99"


class TestExprConstant:
    def test_small_int(self):
        result = ir_expr_to_python(_const(1), {})
        assert result == "1"

    def test_zero(self):
        result = ir_expr_to_python(_const(0), {})
        assert result == "0"

    def test_large_int_uses_bv(self):
        # Values > 0x10000 are wrapped in zdc.bv()
        result = ir_expr_to_python(_const(0x400000), {})
        assert "zdc.bv" in result

    def test_bool_true(self):
        result = ir_expr_to_python(_const(True), {})
        assert result == "True"

    def test_bool_false(self):
        result = ir_expr_to_python(_const(False), {})
        assert result == "False"


class TestExprSubscript:
    def test_subscript(self):
        base = _ref(0)
        expr = _subscript(base, 22)
        result = ir_expr_to_python(expr, {0: "counter"})
        assert result == "self.counter[22]"


class TestExprBinOp:
    def test_add(self):
        a = _ref(0)
        b = _ref(1)
        expr = _binop(a, b, "Add")
        result = ir_expr_to_python(expr, {0: "a", 1: "b"})
        assert "self.a" in result and "self.b" in result and "+" in result

    def test_bitand(self):
        a = _ref(0)
        b = _const(0xFF)
        expr = _binop(a, b, "BitAnd")
        result = ir_expr_to_python(expr, {0: "mask"})
        assert "&" in result


class TestExprUnary:
    def test_invert(self):
        expr = _unary(_ref(0), "Invert")
        result = ir_expr_to_python(expr, {0: "BTN_N"})
        assert "~" in result and "BTN_N" in result


# ---------------------------------------------------------------------------
# Unit tests: statements
# ---------------------------------------------------------------------------

class TestStmtAssign:
    def test_assign(self):
        stmt = _stmt_assign(_ref(0), _ref(1))
        lines = ir_stmts_to_python([stmt], {0: "led", 1: "counter"}, indent=0)
        assert len(lines) == 1
        assert lines[0] == "self.led = self.counter"

    def test_assign_with_indent(self):
        stmt = _stmt_assign(_ref(0), _const(1))
        lines = ir_stmts_to_python([stmt], {0: "x"}, indent=8)
        assert lines[0].startswith("        ")


class TestStmtAugAssign:
    def test_aug_assign_add(self):
        stmt = _stmt_augassign(_ref(0), _const(1), "Add")
        lines = ir_stmts_to_python([stmt], {0: "counter"}, indent=0)
        assert len(lines) == 1
        assert lines[0] == "self.counter += 1"


class TestStmtIf:
    def test_if_only(self):
        cond = _ref(0)
        body = [_stmt_assign(_ref(1), _const(0))]
        stmt = _stmt_if(cond, body)
        lines = ir_stmts_to_python([stmt], {0: "cond", 1: "out"}, indent=0)
        assert lines[0].startswith("if ")
        assert any("self.out" in l for l in lines)

    def test_if_else(self):
        cond = _ref(0)
        body = [_stmt_assign(_ref(1), _const(1))]
        orelse = [_stmt_assign(_ref(1), _const(0))]
        stmt = _stmt_if(cond, body, orelse)
        lines = ir_stmts_to_python([stmt], {0: "cond", 1: "out"}, indent=0)
        assert any("else:" in l for l in lines)

    def test_if_elif_chain(self):
        inner_if = _stmt_if(_ref(1), [_stmt_assign(_ref(2), _const(1))])
        outer_if = _stmt_if(_ref(0), [_stmt_assign(_ref(2), _const(0))], [inner_if])
        lines = ir_stmts_to_python([outer_if], {0: "a", 1: "b", 2: "out"}, indent=0)
        assert any("elif" in l for l in lines)


# ---------------------------------------------------------------------------
# Integration tests: blinky SPL
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blinky_fsm():
    """Run blinky through ComponentFieldsPass + ProcessToFSMPass."""
    if _blinky_dir not in sys.path:
        sys.path.insert(0, _blinky_dir)
    import importlib
    blink_mod = importlib.import_module("blink")
    Blink = blink_mod.Blink

    import zuspec.dataclasses as zdc
    from zuspec.dataclasses.data_model_factory import DataModelFactory
    from zuspec.synth.passes.component_fields import ComponentFieldsPass
    from zuspec.synth.passes.process_to_fsm import ProcessToFSMPass
    from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig

    ctx = DataModelFactory().build(Blink)
    ir = SynthIR(component=Blink, model_context=ctx)
    cfg = SynthConfig()
    ir = ComponentFieldsPass(cfg).run(ir)
    ir = ProcessToFSMPass(cfg).run(ir)
    return ir.fsm_modules[0], ir.component_fields


class TestBlinkyRoundtrip:
    def test_blinky_single_state(self, blinky_fsm):
        fsm, _ = blinky_fsm
        assert fsm.single_state is True

    def test_blinky_body_stmts_not_empty(self, blinky_fsm):
        fsm, _ = blinky_fsm
        assert len(fsm.body_stmts) > 0

    def test_blinky_counter_increment(self, blinky_fsm):
        fsm, _ = blinky_fsm
        lines = ir_stmts_to_python(fsm.body_stmts, fsm.body_idx_to_name, indent=8)
        joined = "\n".join(lines)
        assert "self._counter" in joined
        assert "+=" in joined
        assert "zdc.bv(1)" in joined

    def test_blinky_no_await_in_output(self, blinky_fsm):
        fsm, _ = blinky_fsm
        lines = ir_stmts_to_python(fsm.body_stmts, fsm.body_idx_to_name, indent=8)
        assert not any("await" in l for l in lines)
        assert not any("tick()" in l for l in lines)

    def test_blinky_comb_body(self, blinky_fsm):
        import zuspec.dataclasses as zdc
        _, cf = blinky_fsm

        # Re-get the model context
        if _blinky_dir not in sys.path:
            sys.path.insert(0, _blinky_dir)
        import importlib
        blink_mod = importlib.import_module("blink")
        Blink = blink_mod.Blink
        from zuspec.dataclasses.data_model_factory import DataModelFactory
        ctx = DataModelFactory().build(Blink)
        comp_ir = (ctx.type_m.get(getattr(Blink, "__qualname__", None))
                   or ctx.type_m.get(Blink.__name__))
        comb_procs = getattr(comp_ir, "comb_processes", [])

        assert comb_procs, "Expected at least one comb process"
        lines = ir_stmts_to_python(
            getattr(comb_procs[0], "body", []), cf.idx_to_name, indent=8
        )
        joined = "\n".join(lines)
        assert "self.LED_GREEN" in joined
        assert "self._counter[22]" in joined
        assert "self.LED_RED" in joined
        assert "~" in joined or "BTN_N" in joined
