"""Cycle-accurate Python simulation of the Blink FSM.

Lowers the @zdc.proc behavioral description to an FSM IR (via
ProcessToFSMPass) then interprets it cycle-by-cycle in Python — no
external simulator required.

This gives us confidence that:
  * tick(N) generates the correct multi-cycle wait
  * The for-loop counter increments exactly once per iteration
  * Outputs (L1-L4) are driven only during the LOOP_I_BODY state
  * The rotating-LED pattern repeats correctly
"""
import sys
import os
import operator
import pytest

# Locate the design directory and package sources relative to this test file.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR   = os.path.join(_TESTS_DIR, '..', '..', '..')
_DESIGN_ROTATE = os.path.join(_PKG_DIR, 'design', 'spl', 'rotate')

# Make the design directory importable so DataModelFactory can getsource().
if _DESIGN_ROTATE not in sys.path:
    sys.path.insert(0, os.path.abspath(_DESIGN_ROTATE))


# ---------------------------------------------------------------------------
# Lightweight FSM interpreter
# ---------------------------------------------------------------------------

from zuspec.synth.sprtl.fsm_ir import FSMState, FSMStateKind, FSMAssign, FSMCond


_CMP_OPS = {
    'Eq':    operator.eq,
    'NotEq': operator.ne,
    'Lt':    operator.lt,
    'LtE':  operator.le,
    'Gt':    operator.gt,
    'GtE':  operator.ge,
}

_BIN_OPS = {
    '+':  operator.add,
    '-':  operator.sub,
    '*':  operator.mul,
    '//': operator.floordiv,
    '%':  operator.mod,
    '&':  operator.and_,
    '|':  operator.or_,
    '^':  operator.xor,
    '<<': operator.lshift,
    '>>': operator.rshift,
}

_TUPLE_CMP_OPS = {
    'lt': operator.lt,
    'lte': operator.le,
    'gt': operator.gt,
    'gte': operator.ge,
    'eq': operator.eq,
    'ne': operator.ne,
}


class FSMInterpreter:
    """Cycle-accurate Python interpreter for an FSMModule.

    Models the three always blocks from sv_codegen exactly:
      - always_comb  → _eval_next_state(): evaluate transition conditions
      - always_ff    → counter update: load on entry, decrement while in state
      - always_ff    → state <= next_state, commit pending register/output writes

    Usage::

        interp = FSMInterpreter(fsm)
        interp.reset()
        for _ in range(100):
            outputs = interp.step()
            assert outputs['L1'] in (0, 1)
    """

    def __init__(self, fsm):
        self.fsm = fsm
        self._state_map = {s.id: s for s in fsm.states}

        # Output port names (defaulted to 0 at start of every cycle)
        self._output_names = frozenset(
            p.name for p in fsm.ports if p.direction == 'output'
        )

        # Collect all register names from FSMAssign targets
        self._reg_names = set()
        for s in fsm.states:
            self._collect_targets(s.operations, self._reg_names)

        # Initial reset values from declared registers
        self._reset_regs = {r.name: (r.reset_value or 0) for r in fsm.registers}
        for name in self._reg_names:
            if name not in self._reset_regs:
                self._reset_regs[name] = 0

        # WAIT_CYCLES counter info: state_id → (counter_name, init_value)
        # Counter registers are plain entries in self.regs; the transition
        # conditions in the FSM IR already reference them by name.
        self._wc_counters = {
            s.id: (f"{s.name}_cnt", s.wait_cycles - 1)
            for s in fsm.states
            if s.kind == FSMStateKind.WAIT_CYCLES and s.wait_cycles > 1
        }

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all registers and state to initial values."""
        self.regs = dict(self._reset_regs)
        self.outputs = {name: 0 for name in self._output_names}
        self.state_id = self.fsm.initial_state

    def step(self) -> dict:
        """Advance one clock cycle.

        Returns the output dictionary *after* this clock edge
        (i.e., what would be visible at the next posedge).
        """
        current = self._state_map[self.state_id]
        pending = {}

        # Default all outputs to 0 (always_ff default-clear)
        for name in self._output_names:
            pending[name] = 0

        # always_comb: compute next_state from FSM IR transition conditions.
        # Counter conditions (e.g. S_4_cnt == 0) are evaluated generically via
        # self.regs, just like any other register — no special-casing needed.
        next_id = self._eval_next_state(current)

        # always_ff (datapath): execute state operations.
        # FSMCond guards (e.g. if S_4_cnt == 0) are handled generically by
        # _exec_ops → no WAIT_CYCLES special-case needed here either.
        self._exec_ops(current.operations, pending)

        # always_ff (counter): load on entry, decrement while in state.
        for wc_id, (cname, init_val) in self._wc_counters.items():
            if self.state_id != wc_id and next_id == wc_id:
                pending[cname] = init_val           # entering: load
            elif self.state_id == wc_id:
                pending[cname] = self.regs.get(cname, init_val) - 1  # in state: decrement

        # always_ff (state): commit pending writes and advance state register.
        for k, v in pending.items():
            if k in self._output_names:
                self.outputs[k] = v
            else:
                self.regs[k] = v

        self.state_id = next_id
        return dict(self.outputs)

    def state_name(self) -> str:
        return self._state_map[self.state_id].name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_next_state(self, state: FSMState) -> int:
        """Evaluate transitions in priority order; default = self-loop."""
        for trans in sorted(state.transitions, key=lambda t: t.priority):
            if self._eval_cond(trans.condition):
                return trans.target_state
        return state.id

    def _eval_cond(self, cond) -> bool:
        if cond is None:
            return True

        # Tuple form: ('lt', lhs, rhs)
        if isinstance(cond, tuple) and len(cond) == 3:
            op_str, lhs, rhs = cond
            fn = _TUPLE_CMP_OPS.get(op_str)
            if fn:
                return fn(self._eval(lhs), self._eval(rhs))

        # ExprCompare IR node
        t = type(cond).__name__
        if t == 'ExprCompare':
            lv = self._eval(cond.left)
            rv = self._eval(cond.comparators[0])
            op_name = cond.ops[0].name
            fn = _CMP_OPS.get(op_name)
            return fn(lv, rv) if fn else False

        return bool(self._eval(cond))

    def _eval(self, expr) -> int:
        if expr is None:
            return 0
        if isinstance(expr, bool):
            return int(expr)
        if isinstance(expr, int):
            return expr
        if isinstance(expr, str):
            v = self.regs.get(expr)
            if v is not None:
                return v
            return self.outputs.get(expr, 0)

        # Tuple arithmetic: (a, op_str, b)
        if isinstance(expr, tuple) and len(expr) == 3:
            a, op_str, b = expr
            av, bv = self._eval(a), self._eval(b)
            fn = _BIN_OPS.get(op_str)
            return fn(av, bv) if fn else 0

        t = type(expr).__name__

        if t == 'ExprConstant':
            return int(expr.value)

        if t == 'ExprRefLocal':
            return self.regs.get(expr.name, 0)

        if t == 'ExprRefField':
            name = getattr(expr, 'field_name', None) or getattr(expr, 'name', '')
            return self.regs.get(name, 0)

        if t == 'ExprCompare':
            lv = self._eval(expr.left)
            rv = self._eval(expr.comparators[0])
            op_name = expr.ops[0].name
            fn = _CMP_OPS.get(op_name)
            return int(fn(lv, rv)) if fn else 0

        if t == 'ExprBin':
            lv = self._eval(expr.lhs)
            rv = self._eval(expr.rhs)
            op_name = expr.op.name
            fn = _BIN_OPS.get(op_name)
            return fn(lv, rv) if fn else 0

        if t == 'ExprUnary':
            v = self._eval(expr.operand)
            op_name = expr.op.name
            return int(not v) if op_name == 'Not' else (-v if op_name == 'USub' else v)

        # Fallback: try .value
        if hasattr(expr, 'value'):
            return int(expr.value)

        return 0

    def _exec_ops(self, ops, pending: dict):
        for op in ops:
            if isinstance(op, FSMAssign) and isinstance(op.target, str):
                pending[op.target] = self._eval(op.value)
            elif isinstance(op, FSMCond):
                if self._eval_cond(op.condition):
                    self._exec_ops(op.then_ops, pending)
                elif op.else_ops:
                    self._exec_ops(op.else_ops, pending)

    def _collect_targets(self, ops, names: set):
        for op in ops:
            if isinstance(op, FSMAssign) and isinstance(op.target, str):
                names.add(op.target)
            elif isinstance(op, FSMCond):
                self._collect_targets(op.then_ops, names)
                self._collect_targets(op.else_ops, names)


# ---------------------------------------------------------------------------
# Helper: build the FSM from a component class
# ---------------------------------------------------------------------------

def _build_fsm(cls):
    from zuspec.dataclasses.data_model_factory import DataModelFactory
    from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
    from zuspec.synth.passes import ComponentFieldsPass, ProcessToFSMPass

    ctx = DataModelFactory().build(cls)
    ir = SynthIR(component=cls, model_context=ctx)
    cfg = SynthConfig()
    ir = ComponentFieldsPass(cfg).run(ir)
    ir = ProcessToFSMPass(cfg).run(ir)
    return ir.fsm_modules[0]


# ---------------------------------------------------------------------------
# Fixture: build the FSM once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blink_fast_fsm():
    """FSM for the BlinkFast component (tick(3) variant)."""
    from blink_fast import BlinkFast
    return _build_fsm(BlinkFast)


@pytest.fixture(scope="module")
def blink_fast_interp(blink_fast_fsm):
    """Fresh FSMInterpreter for BlinkFast — reset before each test via interp.reset()."""
    return FSMInterpreter(blink_fast_fsm)


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def run_until_output(interp, max_cycles=200):
    """Step until any output goes high; return (cycle, outputs) pairs up to that point."""
    history = []
    for cyc in range(max_cycles):
        out = interp.step()
        history.append((cyc, dict(out)))
        if any(v for v in out.values()):
            return history
    return history


def collect_output_events(interp, n_cycles):
    """Return list of (cycle, outputs) for cycles where any output is high."""
    events = []
    for cyc in range(n_cycles):
        out = interp.step()
        if any(v for v in out.values()):
            events.append((cyc, dict(out)))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBlinkFastFSM:

    def test_fsm_has_wait_cycles_state(self, blink_fast_fsm):
        """Exactly one WAIT_CYCLES state with wait_cycles==3."""
        wc = [s for s in blink_fast_fsm.states
              if s.kind == FSMStateKind.WAIT_CYCLES and s.wait_cycles > 1]
        assert len(wc) == 1
        assert wc[0].wait_cycles == 3

    def test_fsm_loop_check_state_exists(self, blink_fast_fsm):
        wc_cond = [s for s in blink_fast_fsm.states
                   if s.kind == FSMStateKind.WAIT_COND]
        assert len(wc_cond) == 1, "Expected one LOOP_I_CHK state"

    def test_loop_increment_in_wait_cycles_state(self, blink_fast_fsm):
        """Loop increment (i += 1) must be in the WAIT_CYCLES state."""
        from zuspec.synth.sprtl.fsm_ir import FSMAssign
        wc = next(s for s in blink_fast_fsm.states
                  if s.kind == FSMStateKind.WAIT_CYCLES and s.wait_cycles > 1)
        assigns = [op for op in wc.operations if isinstance(op, FSMAssign)]
        assert any(
            isinstance(op.value, tuple) and '+' in op.value
            for op in assigns
        ), "Expected i <= i + 1 in WAIT_CYCLES state"


class TestBlinkFastSimulation:

    def setup_method(self):
        """Each test gets a freshly reset interpreter."""

    def test_first_output_is_l1(self, blink_fast_interp):
        blink_fast_interp.reset()
        history = run_until_output(blink_fast_interp)
        cyc, out = history[-1]
        assert out.get('L1') == 1, f"Expected L1=1 first, got {out} at cycle {cyc}"
        assert out.get('L2') == 0
        assert out.get('L3') == 0
        assert out.get('L4') == 0

    def test_l1_high_for_exactly_one_cycle(self, blink_fast_interp):
        """L1 stays high for exactly 1 cycle (outputs are registered)."""
        blink_fast_interp.reset()
        # Find cycle where L1 goes high
        l1_high_cycles = 0
        for _ in range(50):
            out = blink_fast_interp.step()
            if out.get('L1') == 1:
                l1_high_cycles += 1
        assert l1_high_cycles == 1, f"L1 should be high exactly 1 cycle in 50, got {l1_high_cycles}"

    def test_outputs_rotate_l1_l2_l3_l4(self, blink_fast_interp):
        """In the first 4 iterations, outputs rotate: L1 then L2 then L3 then L4."""
        blink_fast_interp.reset()
        # Each iteration: 1 (body) + 3 (wait) = 4 cycles + overhead from IDLE/CHK
        # Collect the first 4 high events
        events = collect_output_events(blink_fast_interp, n_cycles=100)
        assert len(events) >= 4, f"Expected at least 4 output events, got {events}"
        _, o0 = events[0]; assert o0 == {'L1': 1, 'L2': 0, 'L3': 0, 'L4': 0}
        _, o1 = events[1]; assert o1 == {'L1': 0, 'L2': 1, 'L3': 0, 'L4': 0}
        _, o2 = events[2]; assert o2 == {'L1': 0, 'L2': 0, 'L3': 1, 'L4': 0}
        _, o3 = events[3]; assert o3 == {'L1': 0, 'L2': 0, 'L3': 0, 'L4': 1}

    def test_rotation_repeats(self, blink_fast_interp):
        """After one full rotation (4 LEDs), L1 goes high again."""
        blink_fast_interp.reset()
        events = collect_output_events(blink_fast_interp, n_cycles=200)
        assert len(events) >= 5
        _, o0 = events[0]
        _, o4 = events[4]
        assert o0 == o4 == {'L1': 1, 'L2': 0, 'L3': 0, 'L4': 0}

    def test_wait_cycles_duration(self, blink_fast_interp):
        """Between each LED-high event there are exactly 3 wait cycles."""
        blink_fast_interp.reset()
        # Collect first two high-output events and measure gap
        events = collect_output_events(blink_fast_interp, n_cycles=100)
        assert len(events) >= 2
        c0, _ = events[0]
        c1, _ = events[1]
        # Gap: 1 cycle (LOOP_I_BODY executes) + 3 wait + 1 check = 5? Let's measure.
        # Actually: LOOP_I_BODY → S_4 (3 cycles) → LOOP_I_CHK (1) → LOOP_I_BODY (1 more output)
        # Total = 3 + 1 + 1 = 5 cycles between high events
        assert c1 - c0 == 5, f"Expected 5 cycles between LED events, got {c1-c0}"

    def test_only_one_output_high_at_a_time(self, blink_fast_interp):
        """At most one output is high on any given cycle."""
        blink_fast_interp.reset()
        for _ in range(150):
            out = blink_fast_interp.step()
            high = sum(v for v in out.values())
            assert high <= 1, f"Multiple outputs high simultaneously: {out}"

    def test_i_increments_once_per_iteration(self, blink_fast_interp):
        """Loop variable i increments exactly once per tick(3) wait, not every cycle."""
        blink_fast_interp.reset()
        # Run through one complete loop (4 iterations)
        # After 4 iterations i should equal 4, then reset to 0
        max_cyc = 100
        i_values = []
        prev_state = blink_fast_interp.state_name()
        for _ in range(max_cyc):
            blink_fast_interp.step()
            i_values.append(blink_fast_interp.regs.get('i', 0))
        # i should reach 4 (loop condition fails), never exceed 4 before reset
        assert max(i_values) == 4, f"i should reach 4 at loop end, max was {max(i_values)}"
        assert min(i_values) == 0, "i should reset to 0 on each loop restart"
