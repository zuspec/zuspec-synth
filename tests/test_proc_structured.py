"""zuspec.synth.proc.structured: the structured emission simulates as the flat one.

Each design is lowered and optimized once, written both ways, and the two
modules are simulated side by side with iverilog on the same random inputs
(with occasional resets); every output is compared every cycle."""
import shutil
import subprocess

import pytest

from zuspec.synth.proc import ir as P
from zuspec.synth.proc import term as T
from zuspec.synth.proc.emit import emit_verilog
from zuspec.synth.proc.lower import LowerError, LowerOptions, lower, optimize
from zuspec.synth.proc.cosim import random_tb
from zuspec.synth.proc.structured import emit_structured

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog")


def V(v: P.Var) -> T.Term:
    return T.var(v.name, v.w)


def _design(name, procs, vars_, ports):
    clk = P.Var("clk", 1, "in")
    rst = P.Var("rst", 1, "in")
    return P.Design(name, clk, rst, ports=ports, vars=[clk, rst] + vars_, procs=procs)


def _out(*vs):
    return [P.Port(v.name + "_o", v.w, "out", v) for v in vs]


def counters():
    en = P.Var("en", 1, "in")
    a = P.Var("a", 4, "reg")
    b = P.Var("b", 4, "reg")
    z = P.Var("z", 4, "reg")
    one = T.const(1, 4)
    body = [P.Loop([
        P.Await(None, initial=True, label="w"),
        P.If(V(en), [P.Assign(a, T.add(V(a), one), nb=True),
                     P.Assign(b, T.add(V(b), one), nb=True)], []),
        P.If(T.eq(V(a), V(b)), [], [P.Assign(z, T.concat(V(en), V(en), V(en), V(en)), nb=True)]),
    ])]
    return _design("counters", [P.Proc("p", body)], [en, a, b, z], _out(a, b, z))


def handshake():
    """A request/response process: conditional waits, a loop with a break,
    blocking reads after writes, a private temp with no register."""
    req = P.Var("req", 1, "in")
    din = P.Var("din", 8, "in")
    busy = P.Var("busy", 1, "reg")
    acc = P.Var("acc", 8, "reg")
    cnt = P.Var("cnt", 3, "reg")
    done = P.Var("done", 1, "reg")
    tmp = P.Var("tmp", 8, "reg")          # private, never live across an edge
    body = [P.Loop([
        P.Await(None, initial=True, label="idle"),
        P.Assign(done, T.ZERO),
        P.Await(V(req), label="wait_req"),
        P.Assign(busy, T.ONE),
        P.Assign(cnt, T.const(0, 3)),
        P.Assign(acc, V(din)),
        P.Loop([
            P.Await(None, label="step"),
            P.Assign(tmp, T.add(V(acc), V(din))),
            P.If(T.eq(T.slice_(V(tmp), 0, 1), T.ONE),
                 [P.Assign(acc, T.xor(V(tmp), T.const(0x5A, 8)))],
                 [P.Assign(acc, V(tmp))]),
            P.Assign(cnt, T.add(V(cnt), T.const(1, 3))),
            P.If(T.eq(V(cnt), T.const(5, 3)), [P.Break()], []),
            P.If(T.land(V(req), T.eq(T.slice_(V(din), 7, 1), T.ONE)), [P.Break()], []),
        ]),
        P.Assign(busy, T.ZERO),
        P.Assign(done, T.ONE),
    ])]
    return _design("handshake", [P.Proc("hs", body)], [req, din, busy, acc, cnt, done, tmp],
                   _out(busy, acc, cnt, done))


def nonblocking():
    """Reads after a nonblocking write see the old value."""
    i = P.Var("i", 4, "in")
    a = P.Var("a", 4, "reg")
    b = P.Var("b", 4, "reg")
    body = [P.Loop([
        P.Await(None, initial=True, label="w"),
        P.Assign(a, T.add(V(a), V(i)), nb=True),
        P.Assign(b, V(a), nb=True),
    ])]
    return _design("nonblocking", [P.Proc("p", body)], [i, a, b], _out(a, b))


def two_writers():
    """Two processes write bits of one var; one of them overlaps."""
    s = P.Var("s", 2, "in")
    x = P.Var("x", 8, "reg")
    hi = [P.Loop([
        P.Await(None, initial=True, label="h"),
        P.If(T.slice_(V(s), 0, 1),
             [P.Assign(x, T.concat(T.add(T.slice_(V(x), 4, 4), T.const(1, 4)),
                                   T.slice_(V(x), 0, 4)))], []),
    ])]
    lo = [P.Loop([
        P.Await(None, initial=True, label="l"),
        P.If(T.slice_(V(s), 1, 1),
             [P.Assign(x, T.concat(T.slice_(V(x), 5, 3),
                                   T.sub(T.slice_(V(x), 0, 5), T.const(3, 5))))], []),
    ])]
    return _design("two_writers", [P.Proc("hi", hi, priority=1), P.Proc("lo", lo)], [s, x],
                   _out(x))


def rtl_fsm():
    """An always_ff-style block: one state, a case on a state register (as an
    if chain), a wide counter reloaded with the same value in several arms."""
    go = P.Var("go", 1, "in")
    ld = P.Var("ld", 8, "in")
    st = P.Var("st", 2, "reg")
    cnt = P.Var("cnt", 8, "reg")
    out = P.Var("out", 1, "reg")
    zero = T.eq(V(cnt), T.const(0, 8))

    def is_(k):
        return T.eq(V(st), T.const(k, 2))
    dec = P.Assign(cnt, T.sub(V(cnt), T.const(1, 8)), nb=True)
    body = [P.Loop([
        P.Await(None, initial=True, label="clk"),
        P.If(is_(0), [P.If(V(go), [P.Assign(st, T.const(1, 2), nb=True),
                                   P.Assign(cnt, V(ld), nb=True)], [])],
             [P.If(is_(1), [P.If(zero, [P.Assign(st, T.const(2, 2), nb=True),
                                        P.Assign(cnt, V(ld), nb=True),
                                        P.Assign(out, T.ONE, nb=True)], [dec])],
                   [P.If(is_(2), [P.If(zero, [P.Assign(st, T.const(0, 2), nb=True),
                                              P.Assign(cnt, V(ld), nb=True),
                                              P.Assign(out, T.ZERO, nb=True)], [dec])],
                         [P.Assign(st, T.const(0, 2), nb=True)])])]),
    ])]
    return _design("rtl_fsm", [P.Proc("f", body)], [go, ld, st, cnt, out], _out(st, cnt, out))


DESIGNS = [counters, handshake, nonblocking, two_writers, rtl_fsm]


def _run(tmp_path, design, opts):
    nl = optimize(lower(design(), opts))
    flat = emit_verilog(nl).replace(f"module {nl.name} (", f"module {nl.name}_f (")
    st = emit_structured(nl).replace(f"module {nl.name} (", f"module {nl.name}_s (")
    (tmp_path / "f.v").write_text(flat)
    (tmp_path / "s.v").write_text(st)
    (tmp_path / "tb.v").write_text(random_tb(nl))
    r = subprocess.run(["iverilog", "-o", str(tmp_path / "sim"), str(tmp_path / "tb.v"),
                        str(tmp_path / "f.v"), str(tmp_path / "s.v")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["vvp", "-n", str(tmp_path / "sim")], capture_output=True, text=True)
    assert "PASS" in r.stdout, r.stdout[-2000:]
    return nl, st


@pytest.mark.parametrize("design", DESIGNS, ids=lambda d: d.__name__)
@pytest.mark.parametrize("seqcorr", [False, "all"])
def test_structured_simulates_as_flat(tmp_path, design, seqcorr):
    if seqcorr:
        pytest.importorskip("pysat")
    _run(tmp_path, design, LowerOptions(seqcorr=seqcorr))


def test_removed_registers_become_wires(tmp_path):
    pytest.importorskip("pysat")
    _nl, st = _run(tmp_path, counters, LowerOptions(seqcorr="all"))
    # b is a and z is 0: no flops of their own
    assert "wire [3:0] b;" in st and "wire [3:0] z;" in st
    assert "reg [3:0] a;" in st


def test_structure_follows_the_source(tmp_path):
    _nl, st = _run(tmp_path, handshake, LowerOptions(seqcorr=False))
    assert "case (hs__state)" in st
    assert st.count("always @*") == 1
    # the private temp is block-local: no flop
    assert "tmp <=" not in st


def test_case_and_value_selector(tmp_path):
    _nl, st = _run(tmp_path, rtl_fsm, LowerOptions(seqcorr=False))
    assert "case (st)" in st
    # cnt is loaded with ld in three arms and decremented in two: a selector
    assert "case (cnt__sel)" in st and "cnt__n = ld;" in st


def test_mixed_blocking_nonblocking_is_an_error():
    a = P.Var("a", 4, "reg")
    body = [P.Loop([P.Await(None, initial=True, label="w"),
                    P.Assign(a, T.add(V(a), T.const(1, 4))),
                    P.Assign(a, V(a), nb=True)])]
    nl = optimize(lower(_design("mixed", [P.Proc("p", body)], [a], _out(a)),
                        LowerOptions(seqcorr=False)))
    with pytest.raises(LowerError, match="blocking and nonblocking"):
        emit_structured(nl)
