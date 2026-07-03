"""Transparent SPL→RTL lowering — language-independent, ``ir.core``-native.

Input: a bound SPL ``DataTypeComponent`` whose ``run`` process is
``forever { <comb> await beat ... }`` with ``put``/``tick`` beats and pin fields
tagged via ``pragmas`` (the SPL IR contract).  Any front end that produces this
shape (fw-hdl SV, zuspec-dataclasses Python, …) can drive this lowering.

Output: an RTL ``DataTypeComponent`` whose behaviour is a single clocked
``sync_processes`` FSM:
  - one state per awaited beat (a ``state`` register; omitted if only one beat);
  - a ``tick(N)`` beat → a down-counter (load N-1, decrement, terminal via the
    underflow/MSB bit — no wide equality comparator);
  - a ``put`` beat → ``pin <= value`` (registered output) advancing one cycle;
  - synchronous reset initialises every register.

Everything is built through :mod:`zuspec.synth.spl.builders` so the be-sv RTL
contract holds (internal regs are plain ``Field``; comparisons are
``ExprCompare``).  The result is emitted by ``zuspec.be.sv`` (the lowering itself
has no back-end dependency).

Transparency contract: **1 awaited beat = 1 state = 1 cycle**; no state merging,
no reordering, no scheduler.
"""
from __future__ import annotations

import dataclasses as dc
from typing import Dict, List, Optional, Tuple

import zuspec.ir.core as ir

from . import builders as b
from .config import SplConfig, SplLowerError


# ---------------------------------------------------------------------------
# beat model
# ---------------------------------------------------------------------------
@dc.dataclass
class PutBeat:
    pin: str            # module pin written, e.g. 'led'
    value: ir.Expr      # SPL-indexed value expression


@dc.dataclass
class TickBeat:
    cycles: int         # N from tick(N)


@dc.dataclass
class WaitBeat:
    cond: ir.Expr       # advance when this condition is true (else hold the state)


def _bottom_ref(e) -> Optional[ir.ExprRefField]:
    """Descend an attribute chain to the underlying ``ExprRefField``."""
    while isinstance(e, ir.ExprAttribute):
        e = e.value
    return e if isinstance(e, ir.ExprRefField) else None


def _is_await(s) -> bool:
    return isinstance(s, ir.StmtExpr) and isinstance(s.expr, ir.ExprAwait)


def _classify_beat(stmt, spl: ir.DataTypeComponent):
    call = stmt.expr.value                 # ExprCall
    method = call.func.attr                # 'put' / 'tick' / 'wait_until'
    if method in ("tick", "cycles"):   # fw-hdl tick(N) / zdc cycles(N)
        if not call.args or not isinstance(call.args[0], ir.ExprConstant):
            raise SplLowerError(f"{method}() requires a constant cycle count")
        return TickBeat(cycles=int(call.args[0].value))
    if method in ("wait_until", "wait"):   # wait until a condition holds
        if not call.args:
            raise SplLowerError(f"{method}() requires a condition argument")
        return WaitBeat(cond=call.args[0])
    # protocol beat (put): resolve the tagged port field
    base = _bottom_ref(call.func.value)
    if base is None:
        raise SplLowerError(f"cannot resolve receiver of beat {method!r}")
    port = spl.fields[base.index]
    protocol = port.pragmas.get("fw_protocol")
    pin = port.pragmas.get("fw_pin")
    if protocol != "put" or pin is None:
        raise SplLowerError(f"unsupported beat {method!r} on {port.name!r}")
    if not call.args:
        raise SplLowerError("put() requires a value argument")
    return PutBeat(pin=pin, value=call.args[0])


def _parse_loop(loop_body, spl) -> Tuple[List, List[Tuple[object, List]]]:
    """Split a loop body into (leading_comb, [(beat, following_comb), ...])."""
    leading: List = []
    beats: List[Tuple[object, List]] = []
    for s in loop_body:
        if _is_await(s):
            beats.append((_classify_beat(s, spl), []))
        elif beats:
            beats[-1][1].append(s)
        else:
            leading.append(s)
    return leading, beats


# ---------------------------------------------------------------------------
# expression / statement rewriting: SPL field index -> RTL field index by name
# ---------------------------------------------------------------------------
def _rewrite_expr(e, spl_fields, name2idx: Dict[str, int]) -> ir.Expr:
    if isinstance(e, ir.ExprRefField):
        name = spl_fields[e.index].name
        return b.sref(name2idx[name])
    if isinstance(e, ir.ExprConstant):
        return ir.ExprConstant(value=e.value)
    if isinstance(e, ir.TypeExprRefSelf):
        return ir.TypeExprRefSelf()
    if isinstance(e, ir.ExprUnary):
        return ir.ExprUnary(op=e.op, operand=_rewrite_expr(e.operand, spl_fields, name2idx))
    if isinstance(e, ir.ExprBin):
        return ir.ExprBin(lhs=_rewrite_expr(e.lhs, spl_fields, name2idx),
                          op=e.op, rhs=_rewrite_expr(e.rhs, spl_fields, name2idx))
    if isinstance(e, ir.ExprCompare):
        return ir.ExprCompare(left=_rewrite_expr(e.left, spl_fields, name2idx),
                             ops=list(e.ops),
                             comparators=[_rewrite_expr(c, spl_fields, name2idx)
                                          for c in e.comparators])
    if isinstance(e, ir.ExprSubscript):
        sl = e.slice
        new_slice = (_rewrite_expr(sl, spl_fields, name2idx)
                     if isinstance(sl, ir.Expr) else sl)
        return ir.ExprSubscript(value=_rewrite_expr(e.value, spl_fields, name2idx),
                               slice=new_slice)
    raise SplLowerError(f"cannot rewrite expression {type(e).__name__}")


def _rewrite_stmt(s, spl_fields, name2idx) -> ir.Stmt:
    if isinstance(s, ir.StmtAssign):
        return ir.StmtAssign(
            targets=[_rewrite_expr(t, spl_fields, name2idx) for t in s.targets],
            value=_rewrite_expr(s.value, spl_fields, name2idx))
    if isinstance(s, ir.StmtIf):
        return ir.StmtIf(test=_rewrite_expr(s.test, spl_fields, name2idx),
                         body=[_rewrite_stmt(x, spl_fields, name2idx) for x in s.body],
                         orelse=[_rewrite_stmt(x, spl_fields, name2idx) for x in s.orelse])
    raise SplLowerError(f"cannot rewrite statement {type(s).__name__}")


def _rw(stmts, spl_fields, name2idx) -> List[ir.Stmt]:
    return [_rewrite_stmt(s, spl_fields, name2idx) for s in stmts]


# ---------------------------------------------------------------------------
# component lowering
# ---------------------------------------------------------------------------
def _role(f) -> Optional[str]:
    return getattr(f, "pragmas", {}).get("fw_role")


def _bits_for(n: int) -> int:
    return max(1, (n - 1).bit_length()) if n > 1 else 1


def _find_run(spl: ir.DataTypeComponent):
    """The coroutine process: a `while True:` of awaited beats (proc or sync)."""
    for bucket in (spl.proc_processes, spl.sync_processes):
        for p in bucket:
            if p.body and isinstance(p.body[0], ir.StmtWhile):
                return p
    return None


def _meta_field(spl, run, key):
    e = getattr(run, "metadata", {}).get(key)
    return spl.fields[e.index] if isinstance(e, ir.ExprRefField) else None


def _clock_reset(spl, run):
    """Clock/reset via the run process's ``metadata`` (the be-sv/zdc convention),
    falling back to fw-hdl's ``Field.pragmas`` role tags."""
    clock = _meta_field(spl, run, "clock") or next(
        (f for f in spl.fields if _role(f) == "clock"), None)
    reset = _meta_field(spl, run, "reset") or next(
        (f for f in spl.fields if _role(f) == "reset"), None)
    return clock, reset


def lower_component(spl: ir.DataTypeComponent,
                    config: Optional[SplConfig] = None) -> ir.DataTypeComponent:
    config = config or SplConfig()
    run = _find_run(spl)
    if run is None:
        raise SplLowerError(f"component {spl.name!r} has no run process (forever loop)")
    clock, reset = _clock_reset(spl, run)
    if clock is None:
        raise SplLowerError(f"component {spl.name!r} has no clock (binding required)")
    # Data pins = module ports (FieldInOut) other than clock/reset.
    pins = [f for f in spl.fields
            if isinstance(f, ir.FieldInOut) and f is not clock and f is not reset]
    # State registers = is_reg fields that aren't ports (process locals like `v`).
    state_regs = [f for f in spl.fields
                  if not isinstance(f, ir.FieldInOut) and getattr(f, "is_reg", False)]
    # Mealy outputs (D1): output pins driven *combinationally* (e.g. a register
    # update's hwif we/next pulse), so `we` is high for exactly one cycle.  These
    # are NOT registered — they go into a comb process, never the sync reset-init.
    mealy_names = {f.name for f in pins
                   if f.is_out and f.pragmas.get("fw_mealy")}
    # Read inputs (fw_read): a register-block readback the FSM reads in a wait
    # beat's reaction (a read-gated update).  The readback is a *registered* field
    # value, so it has not settled the cycle the change pulse arrives — the
    # reaction must fire one cycle later.  A wait whose reaction reads such a pin
    # therefore registers its change pulse into an `armed` flop and gates on that.
    read_names = {f.name for f in pins
                  if not f.is_out and f.pragmas.get("fw_read")}

    loop = run.body[0]

    leading, beats = _parse_loop(loop.body, spl)
    n = len(beats)
    if n == 0:
        raise SplLowerError("run loop has no awaited beats")

    def _spl_reads(nodes) -> bool:
        """True if any node reads an fw_read pin (recursing over Expr/Stmt)."""
        for node in nodes:
            if isinstance(node, ir.ExprRefField) and \
                    spl.fields[node.index].name in read_names:
                return True
            kids = []
            for attr in ("test", "value", "operand", "left", "lhs", "rhs", "base", "slice"):
                c = getattr(node, attr, None)
                if isinstance(c, (ir.Expr, ir.Stmt)):
                    kids.append(c)
            for attr in ("body", "orelse", "targets", "comparators", "args"):
                kids += [c for c in (getattr(node, attr, None) or [])
                         if isinstance(c, (ir.Expr, ir.Stmt))]
            if _spl_reads(kids):
                return True
        return False

    # Which wait beats are read-gated (their reaction samples a settled readback)?
    read_gated = {i for i, (bt, following) in enumerate(beats)
                  if isinstance(bt, WaitBeat) and read_names and _spl_reads(following)}
    if read_gated and n > 1:
        raise SplLowerError(
            "read-gated update inside a multi-beat FSM is not yet supported "
            "(needs the Phase-B nested-state lowering)")
    # A tick(N) with N>1 is a multi-cycle wait → needs a down-counter.  A tick(1)
    # / cycles(1) is a pure 1-cycle boundary (the loop body runs every cycle) and
    # needs no counter — just an unconditional 1-cycle advance.
    needs_counter = any(isinstance(bt, TickBeat) and bt.cycles > 1 for bt, _ in beats)
    max_cycles = max((bt.cycles for bt, _ in beats
                      if isinstance(bt, TickBeat) and bt.cycles > 1), default=2)

    # --- build the RTL field list (indices are fresh) -------------------
    rtl_fields: List[ir.Field] = []
    name2idx: Dict[str, int] = {}

    def add(field: ir.Field) -> None:
        name2idx[field.name] = len(rtl_fields)
        rtl_fields.append(field)

    add(b.in_port(clock.name, 1))
    if reset is not None:
        add(b.in_port(reset.name, 1))
    for p in pins:
        is_out = isinstance(p, ir.FieldInOut) and p.is_out
        add(ir.FieldInOut(name=p.name, datatype=p.datatype, is_out=is_out))
    for r in state_regs:
        add(ir.Field(name=r.name, datatype=r.datatype, is_reg=True))
    need_state = n > 1
    if need_state:
        add(b.reg("state", _bits_for(n)))
    # Down-counter terminal: load (cycles-1), decrement, detect underflow via the
    # borrow/MSB bit -> a single-bit test (no wide equality comparator), so the
    # terminal is off the critical path and costs no comparator LUTs.  One extra
    # bit above the value width holds the underflow.
    count_bits = _bits_for(max_cycles) + 1
    if needs_counter:
        add(b.reg("count", count_bits))
    # One `armed` flop per read-gated wait beat (delays its change pulse a cycle).
    armed_of: Dict[int, str] = {}
    for i in sorted(read_gated):
        name = "armed" if len(read_gated) == 1 else f"armed{i}"
        add(b.reg(name, 1))
        armed_of[i] = name

    def idx(name: str) -> int:
        return name2idx[name]

    # --- per-beat state bodies ------------------------------------------
    def trans_comb(i: int) -> List[ir.Stmt]:
        """Comb statements fired when beat i completes (loop wrap on last)."""
        following = beats[i][1]
        if i == n - 1:
            return _rw(following + leading, spl.fields, name2idx)
        return _rw(following, spl.fields, name2idx)

    def goto(next_i: int) -> List[ir.Stmt]:
        return [b.assign(b.sref(idx("state")), b.const(next_i))] if need_state else []

    def reload_of(j: int) -> List[ir.Stmt]:
        """Preload the down-counter for a multi-cycle tick state j (else no-op)."""
        bt = beats[j][0]
        if isinstance(bt, TickBeat) and bt.cycles > 1:
            return [b.assign(b.sref(idx("count")), b.const(bt.cycles - 1))]
        return []

    msb = count_bits - 1 if needs_counter else 0

    def collect_targets(s, acc) -> None:
        """Names this statement assigns to (recursing into ``if`` branches)."""
        if isinstance(s, ir.StmtAssign):
            for t in s.targets:
                if isinstance(t, ir.ExprRefField):
                    acc.add(rtl_fields[t.index].name)
        elif isinstance(s, ir.StmtIf):
            for sub in list(s.body) + list(s.orelse):
                collect_targets(sub, acc)

    def split_mealy(stmts):
        """Partition rewritten following-comb into (registered, Mealy) statements.

        A statement that writes any Mealy output (possibly inside an ``if`` — a
        read-gated update) goes to the comb process; one writing only registered
        state stays in the sync FSM.  Mixing the two in one statement is rejected."""
        reg_s, mealy_s = [], []
        for s in stmts:
            tgts: set = set()
            collect_targets(s, tgts)
            if tgts & mealy_names:
                if tgts - mealy_names:
                    raise SplLowerError(
                        "a following-comb statement mixes Mealy and registered writes")
                mealy_s.append(s)
            else:
                reg_s.append(s)
        return reg_s, mealy_s

    state_bodies: List[List[ir.Stmt]] = []
    # Mealy segments: (state_index|None, fire_condition|None, [mealy drives]).  The
    # drives fire combinationally when the FSM is in this beat's state AND the beat
    # advances — so the hwif `we` is high exactly the cycle the update happens.
    mealy_segments: List[Tuple[Optional[int], Optional[ir.Expr], List[ir.Stmt]]] = []
    for i, (beat, _following) in enumerate(beats):
        nxt = (i + 1) % n
        reg_following, mealy_following = split_mealy(trans_comb(i))
        fire: Optional[ir.Expr] = None
        if isinstance(beat, PutBeat):
            body = [b.assign(b.sref(idx(beat.pin)),
                             _rewrite_expr(beat.value, spl.fields, name2idx))]
            body += reg_following + goto(nxt) + reload_of(nxt)
        elif isinstance(beat, WaitBeat):
            # FSM wait-state: hold here until the condition is true, then run the
            # following comb and advance. (This is the wait_change -> FSM wait the
            # register watch-set lowers to: the condition is the change pulse.)
            cond = _rewrite_expr(beat.cond, spl.fields, name2idx)
            advance = reg_following + goto(nxt) + reload_of(nxt)
            body = [b.if_(cond, advance, [])] if advance else []
            if i in armed_of:
                # read-gated: register the change pulse so the reaction samples the
                # settled readback one cycle later; the comb fires off `armed`.
                body.append(b.assign(b.sref(idx(armed_of[i])),
                                     _rewrite_expr(beat.cond, spl.fields, name2idx)))
                fire = b.sref(idx(armed_of[i]))
            else:
                fire = _rewrite_expr(beat.cond, spl.fields, name2idx)   # comb gate
        elif isinstance(beat, TickBeat) and beat.cycles > 1:   # multi-cycle: down-count, terminal = MSB
            terminal = ir.ExprSubscript(value=b.sref(idx("count")), slice=b.const(msb))
            done = reg_following + goto(nxt) + reload_of(nxt)
            cont = [b.assign(b.sref(idx("count")),
                             b.binop(ir.BinOp.Sub, b.sref(idx("count")), b.const(1)))]
            body = [b.if_(terminal, done, cont)]
            fire = ir.ExprSubscript(value=b.sref(idx("count")), slice=b.const(msb))
        else:                   # tick(1)/cycles(1): a pure 1-cycle advance
            body = reg_following + goto(nxt) + reload_of(nxt)
        state_bodies.append(body)
        if mealy_following:
            mealy_segments.append((i if need_state else None, fire, mealy_following))

    # --- assemble the FSM chain (if state==0 {..} elif .. else {last}) ---
    if need_state:
        chain: List[ir.Stmt] = state_bodies[-1]
        for i in range(n - 2, -1, -1):
            chain = [b.if_(b.cmp_eq(b.sref(idx("state")), b.const(i)),
                           state_bodies[i], chain)]
    else:
        chain = state_bodies[0]

    # --- synchronous reset: init every register (Mealy outputs excluded) ---
    reset_init: List[ir.Stmt] = []
    for f in rtl_fields:
        if f.name in mealy_names:
            continue                              # comb-driven, never registered
        is_reg = getattr(f, "is_reg", False) or (isinstance(f, ir.FieldInOut) and f.is_out)
        if not is_reg:
            continue
        if f.name == "count" and needs_counter:
            # If the first state is a multi-cycle tick it runs before any
            # preloader, so the counter must reset to that tick's reload value;
            # otherwise the preceding state preloads it, so 0 is fine.
            first = beats[0][0]
            init = (first.cycles - 1) if (isinstance(first, TickBeat)
                                          and first.cycles > 1) else 0
        else:
            init = _reset_value(f, spl)
        reset_init.append(b.assign(b.sref(idx(f.name)), b.const(init)))

    # --- comb process: default each Mealy output low, then drive under its gate -
    comb_processes: List[ir.Function] = []
    if mealy_segments:
        comb_body: List[ir.Stmt] = [
            b.assign(b.sref(idx(name)), b.const(0))
            for name in (f.name for f in rtl_fields if f.name in mealy_names)]
        for state_i, fire, drives in mealy_segments:
            inner = [b.if_(fire, drives, [])] if fire is not None else list(drives)
            if state_i is not None:
                comb_body.append(b.if_(b.cmp_eq(b.sref(idx("state")), b.const(state_i)),
                                       inner, []))
            else:
                comb_body.extend(inner)
        comb_processes.append(b.comb_proc("_comb_" + spl.name, comb_body))

    # A purely combinational engine (no state, no registered outputs — the minimal
    # wait+update) needs no clocked process at all.
    has_registers = bool(reset_init) or need_state or needs_counter
    sync_processes: List[ir.Function] = []
    if has_registers:
        if reset is not None:
            # fw-hdl std reset is active-high (fw_clock_xtor_if: `if (reset)`).
            # be-sv emits a synchronous sensitivity list; an async `posedge reset`
            # is a future be-sv enhancement — sync active-high matches the std
            # testbench, which holds reset across clock edges.
            reset_ref = b.sref(idx(reset.name))
            reset_test = (b.lognot(reset_ref) if config.reset_style.endswith("low")
                          else reset_ref)
            proc_body = [b.if_(reset_test, reset_init, chain)]
        else:
            proc_body = chain
        sync_processes.append(b.sync_proc(
            "_" + spl.name, proc_body,
            clock_index=idx(clock.name),
            reset_index=idx(reset.name) if reset is not None else None))

    return b.component(spl.name, rtl_fields,
                       sync_processes=sync_processes,
                       comb_processes=comb_processes)


def _reset_value(field, spl: ir.DataTypeComponent) -> int:
    """Reset value for a register: ``reset_value`` (zdc) or ``initial_value``
    (fw-hdl), else 0."""
    for f in spl.fields:
        if f.name == field.name:
            rv = getattr(f, "reset_value", None)
            if rv is not None:
                try:
                    return int(rv)
                except (TypeError, ValueError):
                    pass
            iv = getattr(f, "initial_value", None)
            if isinstance(iv, ir.ExprConstant):
                return int(iv.value)
    return 0


def lower_context(spl_ctxt: ir.Context,
                  config: Optional[SplConfig] = None) -> ir.Context:
    """Lower every component in an SPL context to RTL IR (raises ``SplLowerError``)."""
    components = [lower_component(dtype, config)
                  for dtype in spl_ctxt.type_m.values()
                  if isinstance(dtype, ir.DataTypeComponent)]
    return b.context(components)
