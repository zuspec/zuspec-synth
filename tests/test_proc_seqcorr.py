"""zuspec.synth.proc: combinational processes and sequential correspondence."""
import random

import pytest

from zuspec.synth.proc import ir as P
from zuspec.synth.proc import term as T
from zuspec.synth.proc.lower import LowerError, LowerOptions, lower, optimize
from zuspec.synth.proc.seqcorr import Aig, _Blaster, correspond, from_netlist

pytest.importorskip("pysat")


def V(v: P.Var) -> T.Term:
    return T.var(v.name, v.w)


# ------------------------------------------------------------ term evaluation
def ev(t: T.Term, env) -> int:
    m = (1 << t.w) - 1
    op = t.op
    if op == "const":
        return t.val
    if op == "var":
        return env[t.name] & m
    a = [ev(x, env) for x in t.args]
    if op == "slice":
        return (a[0] >> t.val) & m
    if op == "concat":
        r = 0
        for x, y in zip(t.args, a):
            r = (r << x.w) | y
        return r
    if op == "not":
        return ~a[0] & m
    if op in ("and", "or", "xor"):
        r = a[0]
        for y in a[1:]:
            r = r & y if op == "and" else r | y if op == "or" else r ^ y
        return r
    if op == "ite":
        return a[1] if a[0] else a[2]
    if op == "add":
        return (a[0] + a[1]) & m
    if op == "sub":
        return (a[0] - a[1]) & m
    if op == "mul":
        return (a[0] * a[1]) & m
    if op == "eq":
        return int(a[0] == a[1])
    if op == "ult":
        return int(a[0] < a[1])
    if op == "slt":
        w = t.args[0].w
        s = lambda x: x - (1 << w) if x >> (w - 1) else x
        return int(s(a[0]) < s(a[1]))
    if op in ("shl", "lshr", "ashr"):
        w, n = t.w, a[1]
        if op == "shl":
            return (a[0] << n) & m if n < w else 0
        if op == "lshr":
            return a[0] >> n if n < w else 0
        sgn = a[0] >> (w - 1)
        if n >= w:
            return m if sgn else 0
        r = a[0] >> n
        return r | (m & ~((1 << (w - n)) - 1)) if sgn else r
    if op == "redor":
        return int(a[0] != 0)
    if op == "redand":
        return int(a[0] == (1 << t.args[0].w) - 1)
    if op == "redxor":
        return bin(a[0]).count("1") & 1
    raise AssertionError(op)


def _mk_terms():
    x, y = T.var("x", 6), T.var("y", 6)
    s = T.var("s", 3)
    c = T.var("c", 1)
    return [T.add(x, y), T.sub(x, y), T.mul(x, y), T.eq(x, y), T.ult(x, y), T.slt(x, y),
            T.shl(x, s), T.lshr(x, s), T.ashr(x, s), T.redor(x), T.redand(y), T.redxor(x),
            T.ite(c, x, y), T.concat(T.slice_(x, 2, 3), T.not_(T.slice_(y, 0, 2))),
            T.xor(x, T.and_(y, T.or_(x, T.const(5, 6))))], ["x", "y", "s", "c"]


def test_bit_blaster_matches_term_evaluation():
    terms, names = _mk_terms()
    widths = {"x": 6, "y": 6, "s": 3, "c": 1}
    g = Aig()
    leaf = {n: [g.pi() for _ in range(w)] for n, w in widths.items()}
    bl = _Blaster(g, leaf, {})
    lits = [bl.bits(t) for t in terms]
    rng = random.Random(3)
    for _ in range(300):
        env = {n: rng.getrandbits(w) for n, w in widths.items()}
        pis = {}
        for n, w in widths.items():
            for b, l in enumerate(leaf[n]):
                pis[l >> 1] = (env[n] >> b) & 1
        v = g.simulate(pis, 1)
        for t, ls in zip(terms, lits):
            got = sum(Aig.lit_val(v, l, 1) << b for b, l in enumerate(ls))
            assert got == ev(t, env), (T.to_str(t), env)


# ------------------------------------------------------------ small designs
def _design(procs, vars_, ports):
    clk = P.Var("clk", 1, "in")
    rst = P.Var("rst", 1, "in")
    return P.Design("t", clk, rst, ports=ports, vars=[clk, rst] + vars_, procs=procs)


def _counters():
    """Two 4-bit counters stepped by the same enable: always equal. A third
    register is the complement of the first; a fourth is always 0 (it
    reloads a value that is 0 whenever it is loaded)."""
    en = P.Var("en", 1, "in")
    a = P.Var("a", 4, "reg")
    b = P.Var("b", 4, "reg")
    na = P.Var("na", 4, "reg", init=0xF)
    z = P.Var("z", 4, "reg")
    oa = P.Var("oa", 4, "net")
    one = T.const(1, 4)
    body = [P.Loop([
        P.Await(None, initial=True, label="w"),
        P.If(V(en), [P.Assign(a, T.add(V(a), one), nb=True),
                     P.Assign(b, T.add(V(b), one), nb=True),
                     P.Assign(na, T.sub(V(na), one), nb=True)], []),
        P.If(T.eq(V(a), V(b)), [], [P.Assign(z, T.concat(V(en), V(en), V(en), V(en)), nb=True)]),
    ])]
    comb = P.Proc("o", [P.Assign(oa, T.xor(T.xor(V(a), V(b)), T.xor(V(na), T.not_(V(z)))))], comb=True)
    return _design([P.Proc("p", body), comb], [en, a, b, na, z, oa],
                   [P.Port("oa", 4, "out", oa)])


def test_flop_correspondence_finds_equal_complement_and_constant():
    nl = lower(_counters(), LowerOptions(reg_corr=False))
    g, _bl = from_netlist(nl)
    res = correspond(g, ["rst"], nodes="flops", cycles=200)
    facts = dict(res.flop_facts())
    for i in range(4):
        assert facts.get(f"b[{i}]") == f"a[{i}]"
        assert facts.get(f"na[{i}]") == f"~a[{i}]"
        assert facts.get(f"z[{i}]") == "0"


def test_seqcorr_option_removes_the_redundant_registers():
    nl = optimize(lower(_counters(), LowerOptions(reg_corr=False, seqcorr="all")))
    assert [r.name for r in nl.regs] == ["a"]
    assert any(line.startswith("seqcorr:") for line in nl.report)


def test_not_equal_after_reset_is_not_merged():
    """Same next-state function, different reset values: not equal."""
    a = P.Var("a", 1, "reg", init=0)
    b = P.Var("b", 1, "reg", init=1)
    i = P.Var("i", 1, "in")
    o = P.Var("o", 1, "net")
    body = [P.Loop([P.Await(None, initial=True, label="w"),
                    P.Assign(a, V(i), nb=True), P.Assign(b, V(i), nb=True)])]
    comb = P.Proc("c", [P.Assign(o, T.xor(V(a), V(b)))], comb=True)
    d = _design([P.Proc("p", body), comb], [i, a, b, o], [P.Port("o", 1, "out", o)])
    nl = lower(d, LowerOptions(reg_corr=False))
    res = correspond(from_netlist(nl)[0], ["rst"], cycles=100)
    assert res.flop_facts() == []


# ------------------------------------------------------------ combinational processes
def test_comb_process_default_then_override():
    s = P.Var("s", 1, "in")
    a = P.Var("a", 4, "in")
    y = P.Var("y", 4, "net")
    d = _design([P.Proc("c", [P.Assign(y, T.const(0, 4)),
                              P.If(V(s), [P.Assign(y, V(a))], [])], comb=True)],
                [s, a, y], [P.Port("y", 4, "out", y)])
    nl = lower(d)
    (n, t), = [(n, t) for n, t in nl.wires if n == "y"]
    assert ev(t, {"s": 1, "a": 9}) == 9 and ev(t, {"s": 0, "a": 9}) == 0


def test_comb_process_latch_is_an_error():
    s = P.Var("s", 1, "in")
    a = P.Var("a", 4, "in")
    y = P.Var("y", 4, "net")
    d = _design([P.Proc("c", [P.If(V(s), [P.Assign(y, V(a))], [])], comb=True)],
                [s, a, y], [P.Port("y", 4, "out", y)])
    with pytest.raises(LowerError, match="latch"):
        lower(d)
