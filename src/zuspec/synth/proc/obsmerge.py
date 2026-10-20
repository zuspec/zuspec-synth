"""Register merging modulo observability (the *tight* view, synthesis-plan §0.8).

seqcorr merges register bits that are equal in *every* reachable state. A
copy is usually equal to its source only while it is *live*: from the cycle
it is loaded to the last cycle that reads it. Outside that window it keeps
an old value nobody reads. Such a copy can still be replaced by its source.

For a register ``r`` and a replacement ``x[b]`` per bit (a bit of another
register, its complement, or a constant) this pass proves invariants

    c == v  ->  r == x

one per *condition* ``c == v`` (a narrow register, often a process state, at
one of its values). Then it checks that replacing every read of ``r`` by
``x`` changes no output and no next value of a register that stays.

How:

1. *Observability* (syntactic): per register bit and reader (an output, or
   another register's next value), a signal that is 1 when the reader reads
   the bit through the selects in the way (an ``ite`` passes its arms' reads
   under its condition, a 1-bit ``and``/``or`` under its other operands).
   Continuous assignments are not readers; what reads them is.
2. *Candidates* (random simulation from reset, as seqcorr's): a bit is live
   in a cycle when an output reads it, or a register reads it and is live in
   the next cycle, or it keeps its value (by construction of its next-state
   term) into a cycle in which it is live.
   Per bit, a replacement equals it in every live cycle (a word is usually a
   copy of one other word: that one is preferred). Every condition that
   occurs in a live cycle, and in no cycle that breaks one of the equalities,
   is a candidate; the conditions together must cover the live cycles. A
   copy written by one process and read by another needs conditions over
   both (the reader's state, the writer's handshake flag): no single control
   register is chosen, induction picks.
3. *Induction* (SAT): the implications hold one cycle after reset, and,
   assumed together with seqcorr's equalities (an inductive invariant of
   their own), they hold in the next cycle. Counterexamples drop
   candidates until a fixpoint.
4. *Substitution check* (SAT, one frame): under the invariant, the outputs and
   the next values of every register that stays are the same with each
   replaced bit read as its replacement. A counterexample un-merges the bits
   it involves.

Why that is enough: let K be the bits that stay. In every reachable state s
of the original machine M the invariant holds, so (4) gives that the merged
machine, in state s restricted to K, has the outputs of M in s and moves to
the next state of M restricted to K, which is reachable again. By induction
over the cycles after reset, M and the merged machine have the same outputs.

The invariant is the certificate: :func:`optimize` returns it (with the merge
list), so a third tool can check it (synthesis-plan §0.8.5).
"""
from __future__ import annotations

import collections
import dataclasses as dc
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import term as T
from .seqcorr import Aig, _Blaster, correspond, from_netlist


# ---------------------------------------------------------------- observability
def _sinks(nl) -> List[T.Term]:
    """What is observed: outputs and the inputs of instances. Continuous
    assignments are not, by themselves: their readers are."""
    out = [t for _, t in nl.outputs]
    nets = {n for n, _ in nl.nets}
    for inst in nl.instances:
        for c in inst.conns:
            if c.term is not None:
                out.append(c.term)
            elif c.var is not None and c.var.name not in nets and c.direction != "out":
                out.append(T.var(c.var.name, c.var.w))
    return out


def _care(nl, g: Aig, bl: _Blaster) -> Dict[Tuple[str, Optional[str]], List[int]]:
    """(register, reader) -> per bit, a literal: the reader (a register's next
    value, or None: an output or instance input) reads that bit in this cycle
    (an over-approximation of observability). A register's own next value is
    not a reader of it."""
    regs = {r.name for r in nl.regs}
    width = {r.name: r.w for r in nl.regs}
    out: Dict[Tuple[str, Optional[str]], List[int]] = {}

    def read(name: str, owner: Optional[str], bits, c: int) -> None:
        ls = out.setdefault((name, owner), [0] * width[name])
        for b in bits:
            ls[b] = g.or_(ls[b], c)
    wires = bl.wires
    # continuous assignments are not observed themselves: what reads them is
    roots: List[Tuple[Optional[str], T.Term]] = [(None, t) for t in _sinks(nl)]
    roots += [(r.name, r.next) for r in nl.regs]

    def kids(t: T.Term):
        if t.op == "var" and t.name in wires:
            return (wires[t.name],)
        return t.args

    for owner, root in roots:
        # topological order of the cone (parents first)
        order: List[T.Term] = []
        seen: Set[T.Term] = set()
        stack = [(root, False)]
        while stack:
            t, done = stack.pop()
            if done:
                order.append(t)
                continue
            if t in seen:
                continue
            seen.add(t)
            stack.append((t, True))
            for a in kids(t):
                if a not in seen:
                    stack.append((a, False))
        order.reverse()
        care: Dict[T.Term, int] = {root: 1}
        for t in order:
            c = care.get(t, 0)
            if c == 0:
                continue
            if t.op == "slice" and t.args[0].op == "var" and t.args[0].name in regs:
                # a read of some bits of a register: only those are observed
                if t.args[0].name != owner:
                    read(t.args[0].name, owner, range(t.val, t.val + t.w), c)
                continue
            if t.op == "var":
                if t.name in wires:
                    w = wires[t.name]
                    care[w] = g.or_(care.get(w, 0), c)
                elif t.name in regs and t.name != owner:
                    read(t.name, owner, range(t.w), c)
                continue
            if t.op == "ite":
                cond, a, b = t.args
                cl = bl.bits(cond)[0]
                care[cond] = g.or_(care.get(cond, 0), c)
                care[a] = g.or_(care.get(a, 0), g.and_(c, cl))
                care[b] = g.or_(care.get(b, 0), g.and_(c, cl ^ 1))
                continue
            if t.op in ("and", "or") and t.w == 1:
                ls = [bl.bits(a)[0] for a in t.args]
                for i, a in enumerate(t.args):
                    o = c
                    for j, l in enumerate(ls):
                        if j != i:
                            o = g.and_(o, l if t.op == "and" else l ^ 1)
                    care[a] = g.or_(care.get(a, 0), o)
                continue
            for a in t.args:
                care[a] = g.or_(care.get(a, 0), c)
    return out


def _hold(nl, g: Aig, bl: _Blaster) -> Dict[str, List[int]]:
    """Register name -> per bit, a literal: its next value is, by construction,
    the bit itself (every select on the way picks the arm that keeps it)."""
    out: Dict[str, List[int]] = {}
    for r in nl.regs:
        memo: Dict[Tuple[T.Term, int], int] = {}

        def h(t: T.Term, lo: int) -> int:
            k = (t, lo)
            if k in memo:
                return memo[k]
            if t.op == "var" and t.name == r.name and lo == 0 and t.w == r.w:
                v = 1
            elif (t.op == "slice" and t.args[0].op == "var" and t.args[0].name == r.name
                  and t.val == lo):
                v = 1
            elif t.op == "ite":
                c = bl.bits(t.args[0])[0]
                v = g.mux(c, h(t.args[2], lo), h(t.args[1], lo))
            elif t.op == "concat":
                v, o = 1, lo
                for a in reversed(t.args):
                    v = g.and_(v, h(a, o))
                    o += a.w
            else:
                v = 0
            memo[k] = v
            return v
        out[r.name] = [h(T.slice_(r.next, b, 1), b) for b in range(r.w)]
    return out


# ---------------------------------------------------------------- SAT frames
class _Cnf:
    """Two time frames of the AIG, encoded on demand. Frame 1's flop outputs
    are frame 0's next-state literals."""

    def __init__(self, g: Aig):
        from pysat.solvers import Solver
        self.g = g
        self.n = 1                       # var 1: true
        self.S = Solver(name="cadical153")
        self.S.add_clause([1])
        self.var = [dict(), dict()]      # frame -> node -> var
        self.flop_next = {f.q: f.next for f in g.flops}
        self._xc: Dict[Tuple[int, int], int] = {}

    def new(self) -> int:
        self.n += 1
        return self.n

    def lit(self, f: int, l: int) -> int:
        v = self.node(f, l >> 1)
        return -v if l & 1 else v

    def node(self, f: int, n0: int) -> int:
        if n0 == 0:
            return -1
        vm = self.var[f]
        if n0 in vm:
            return vm[n0]
        g = self.g
        stack = [n0]
        while stack:
            n = stack[-1]
            if n in vm:
                stack.pop()
                continue
            fi = g.fanin[n]
            if fi is None:
                stack.pop()
                if f == 1 and n in self.flop_next:
                    nl_ = self.flop_next[n]
                    # frame 0's next value, which may still need encoding
                    if (nl_ >> 1) not in self.var[0] and nl_ >> 1:
                        self.node(0, nl_ >> 1)
                    vm[n] = self.lit(0, nl_)
                else:
                    vm[n] = self.new()
                continue
            todo = [x >> 1 for x in fi if (x >> 1) and (x >> 1) not in vm]
            if todo:
                stack.extend(todo)
                continue
            stack.pop()
            a = self._l(f, fi[0])
            b = self._l(f, fi[1])
            z = self.new()
            self.S.add_clause([-z, a])
            self.S.add_clause([-z, b])
            self.S.add_clause([z, -a, -b])
            vm[n] = z
        return vm[n0]

    def _l(self, f: int, l: int) -> int:
        if l >> 1 == 0:
            return -1 if not (l & 1) else 1
        v = self.var[f][l >> 1]
        return -v if l & 1 else v

    def xor(self, a: int, b: int) -> int:
        k = (a, b) if a < b else (b, a)
        z = self._xc.get(k)
        if z is None:
            z = self.new()
            self._xc[k] = z
            S = self.S
            S.add_clause([-z, a, b])
            S.add_clause([-z, -a, -b])
            S.add_clause([z, -a, b])
            S.add_clause([z, a, -b])
        return z

    def and2(self, a: int, b: int) -> int:
        z = self.new()
        self.S.add_clause([-z, a])
        self.S.add_clause([-z, b])
        self.S.add_clause([z, -a, -b])
        return z


# ---------------------------------------------------------------- the pass
@dc.dataclass
class Cand:
    """(ctl == val) -> reg[b] == x for every (b, x) in bits."""
    reg: str
    bits: Tuple[Tuple[int, int, int], ...]   # (bit, flop node, literal of x: 0/1 constant)
    live: int         # literal of ctl == val
    ctl: str
    val: int


@dc.dataclass
class Merge:
    reg: str
    bit: int
    what: str         # replacement, as text
    term: T.Term      # replacement term
    when: List[Tuple[str, int]]   # the (ctl, val) under which it is proved equal

    def invariant(self, w_of) -> T.Term:
        """AND over when of (ctl == val -> reg[bit] == replacement), as a 1-bit
        term (w_of: name -> width)."""
        r = T.slice_(T.var(self.reg, w_of(self.reg)), self.bit, 1)
        e = T.eq(r, self.term)
        out = []
        for cn, v in self.when:
            c = T.var(cn, w_of(cn))
            out.append(T.or_(T.ne(c, T.const(v, c.w)), e))
        return T.and_(*out)


def optimize(nl, *, cycles: int = 1000, width: int = 256, seed: int = 1,
             max_ctl_w: int = 5, base_nodes: str = "all", log=None,
             debug=None, watch=None, constrain=None) -> Tuple[List[str], List[Merge]]:
    """Merge register bits into what they equal while live; in place.
    Returns (report lines, merges).

    ``constrain(inputs, reg, mask)``: an environment rule applied to the
    random stimulus (inputs: name -> lanes, modified in place; reg(name): a
    register's bits as lanes). It only steers which candidates are learned;
    nothing is proved under it. ``debug``/``watch``: diagnostics."""
    say = log or (lambda s: None)
    dbg = debug or (lambda s: None)
    if not nl.reset:
        return ["obsmerge: skipped (no reset)"], []
    g, bl = from_netlist(nl)
    n_logic = len(g.fanin)
    reads = _care(nl, g, bl)
    hold = _hold(nl, g, bl)
    regs = {r.name: r for r in nl.regs}
    flops = g.flops
    fq = [f.q for f in flops]
    qidx = {f.q: i for i, f in enumerate(flops)}
    # bits whose next value is a constant (merged by an earlier round, or
    # never written) take no part: they no longer follow the original machine
    triv = {i for i, f in enumerate(flops) if f.next in (0, 1)}
    by_reg: Dict[str, List[int]] = collections.defaultdict(list)
    for i, f in enumerate(flops):
        by_reg[f.name].append(i)
    part = {n for n, idx in by_reg.items() if any(i in triv for i in idx)}
    # control registers: narrow ones; state registers of processes first
    ctls = [r.name for r in nl.regs if r.w <= max_ctl_w and r.name not in part]
    ctls.sort(key=lambda n: (not n.endswith("__state"), not n.endswith("state"), -regs[n].w, n))

    # the conditions (control register == value), as nodes
    cond: Dict[Tuple[str, int], int] = {}
    for cn in ctls:
        for v in range(1 << len(by_reg[cn])):
            m = 1
            for k, fi in enumerate(by_reg[cn]):
                m = g.and_(m, 2 * fq[fi] ^ (0 if (v >> k) & 1 else 1))
            cond[(cn, v)] = m
    # the netlist's own equalities (seqcorr's inductive invariant), over its
    # logic and the conditions: equalities between conditions are how the
    # states of two processes that run in lock step are related, which the
    # induction below needs
    if base_nodes == "all":
        cn_ = [n for n in range(1, n_logic) if g.is_and(n)]
        cn_ += [l >> 1 for l in cond.values() if l > 1]
    else:
        cn_ = []
    base = correspond(g, [nl.reset], nodes="flops", cycles=cycles, width=width, seed=seed,
                      cand_nodes=cn_)
    say(f"obsmerge: the netlist's own equalities: {sum(len(c) - 1 for c in base.classes)}")

    # ---- 1. simulation from reset (seqcorr's input model)
    mask = (1 << width) - 1
    rng = random.Random(seed)
    rnd = lambda: rng.getrandbits(width)
    inp = [n for _, n in g.inputs]
    rst = [n for nm, n in g.inputs if nm == nl.reset or nm.split("[")[0] == nl.reset]
    grp = width // 4
    gm = [((1 << grp) - 1) << (grp * k) for k in range(4)]

    def flips() -> int:
        f, x = 0, rnd()
        for k in range(4):
            if k:
                x &= rnd() & rnd()
            f |= x & gm[k]
        return f
    # per flop bit i: who reads it (reader register or None), and how
    entries: List[Tuple[int, Optional[str], int]] = []
    fidx = {(f.name, f.bit): i for i, f in enumerate(flops)}
    for (r, o), ls in sorted(reads.items(), key=lambda k: (k[0][0], k[0][1] or "")):
        for b, l in enumerate(ls):
            if l:
                entries.append((fidx[(r, b)], o, l))
    read_lit = [l for _i, _o, l in entries]
    hold_lit = [hold[f.name][f.bit] for f in flops]
    cur = {n: rnd() for n in inp}
    for n in rst:
        cur[n] = mask
    v = g.simulate({**{q: rnd() for q in fq}, **cur}, mask)
    state = [g.lit_val(v, f.next, mask) for f in flops]
    Q: List[List[int]] = []      # per cycle: flop values
    C: List[List[int]] = []      # per cycle: per (register, reader): read
    H: List[List[int]] = []      # per cycle: hold values (reset: no hold)
    for cyc in range(cycles):
        for n in inp:
            cur[n] ^= flips()
        for n in rst:
            x = mask
            for _ in range(8):
                x &= rnd()
            cur[n] = x
        if constrain is not None:
            # an environment rule (simulation only: no proof uses it yet)
            named = {nm: cur[n] for nm, n in g.inputs}
            sv = dict(zip(fq, state))
            constrain(named, lambda r: [sv[fq[i]] for i in by_reg[r]], mask)
            for nm, n in g.inputs:
                cur[n] = named[nm]
        pis = dict(cur)
        for q, s in zip(fq, state):
            pis[q] = s
        v = g.simulate(pis, mask)
        Q.append(state)
        C.append([g.lit_val(v, l, mask) for l in read_lit])
        nr = mask
        for n in rst:
            nr &= ~cur[n]
        H.append([g.lit_val(v, l, mask) & nr for l in hold_lit])
        state = [g.lit_val(v, f.next, mask) for f in flops]
    Q.append(state)

    # ---- 2. liveness and candidates per register. The trace of a signal is
    # one integer: cycle t's lanes at bits [t * width, (t + 1) * width).
    ALL = (1 << (cycles * width)) - 1

    def whole(per_cycle) -> int:
        return int.from_bytes(b"".join(x.to_bytes(width // 8, "little") for x in per_cycle),
                              "little")
    FQ = [whole([Q[t][i] for t in range(cycles)]) for i in range(len(flops))]
    _mt: Dict[Tuple[str, int], int] = {}

    def minterm(cn: str, val: int) -> int:
        """The lane-cycles where control register cn has value val."""
        k = (cn, val)
        if k not in _mt:
            m = ALL
            for b, i in enumerate(by_reg[cn]):
                m &= FQ[i] if (val >> b) & 1 else ~FQ[i]
            _mt[k] = m & ALL
        return _mt[k]
    # Liveness, backward over the trace, for all bits at once: a bit is live
    # at t when an output reads it at t, or a register s reads it at t and s
    # is live at t + 1, or it keeps its value into t + 1 and is live there.
    nf = len(flops)
    lives: List[List[int]] = [[0] * cycles for _ in range(nf)]
    nxt = [0] * nf
    for t in range(cycles - 1, -1, -1):
        wl: Dict[str, int] = {}
        for name, idx in by_reg.items():
            x = 0
            for i in idx:
                x |= nxt[i]
            wl[name] = x
        Ht, Ct = H[t], C[t]
        cur_l = [Ht[i] & nxt[i] for i in range(nf)]
        for e, (i, o, _l) in enumerate(entries):
            cur_l[i] |= Ct[e] & (mask if o is None else wl[o])
        nxt = cur_l
        for i in range(nf):
            lives[i][t] = nxt[i]
    LV = [whole(lives[i]) for i in range(nf)]
    cands: List[Cand] = []
    for name, idx in by_reg.items():
        nlive = [bin(LV[i]).count("1") for i in idx]
        dbg(f"{name}: live in {max(nlive)} lane-cycles (most live bit)")
        # replacements per bit: flop bits (either phase) and constants
        own = set(idx)
        per_bit: Dict[int, List[int]] = {}
        for k, i in enumerate(idx):
            if i in triv or nlive[k] < 64:
                continue
            LIVE = LV[i]
            R = FQ[i] & LIVE
            ok = []
            if not R:
                ok.append(0)
            if R == LIVE:
                ok.append(1)
            for j in range(nf):
                if j in own or j in triv:
                    continue
                d = R ^ (FQ[j] & LIVE)
                if not d:
                    ok.append(2 * fq[j])
                elif d == LIVE:
                    ok.append(2 * fq[j] + 1)
            if ok:
                per_bit[i] = ok
        dbg(f"{name}: live bits {[n_ for n_ in nlive]}; replacements per live bit "
            f"{ {flops[i].bit: len(v) for i, v in per_bit.items()} }")
        if not per_bit:
            continue
        # one replacement per bit (constants first); then every (c == v) that
        # occurs in live cycles and never with a broken equality is a
        # candidate of its own: induction keeps the subset that is inductive
        # together, and the substitution check decides whether what is left
        # covers every cycle in which the register is read
        # a word is usually a copy of one other word: prefer the register most
        # bits can stand for, at the same bit position, uncomplemented
        votes = collections.Counter(flops[qidx[x >> 1]].name for ok in per_bit.values()
                                    for x in ok if x >= 2)
        src = votes.most_common(1)[0][0] if votes else None

        def pick(i: int, ok: List[int]) -> int:
            def key(x):
                if x < 2:
                    return (1, 0)
                f = flops[qidx[x >> 1]]
                return (0 if f.name == src else 2, (f.bit != flops[i].bit) + 2 * (x & 1))
            return min(ok, key=key)
        choice = {i: pick(i, ok) for i, ok in per_bit.items()}
        # per bit: every condition that occurs while the bit is live and never
        # with the equality broken
        nb = 0
        for i in per_bit:
            x = choice[i]
            XV = (ALL if x else 0) if x < 2 else FQ[qidx[x >> 1]] ^ (ALL if x & 1 else 0)
            BAD = FQ[i] ^ XV
            LIVE = LV[i]
            COVER = 0
            mine = []
            for cn in ctls:
                if cn == name:
                    continue
                for v in range(1 << len(by_reg[cn])):
                    mt = minterm(cn, v)
                    if mt & LIVE and not mt & BAD:
                        mine.append((cn, v))
                        COVER |= mt
            if (LIVE & ~COVER) or not mine:
                continue
            nb += 1
            xs = ((flops[i].bit, fq[i], x),)
            for cn, v in mine:
                cands.append(Cand(name, xs, cond[(cn, v)], cn, v))
        dbg(f"{name}: {nb} bits with covering conditions")
    say(f"obsmerge: {len(cands)} candidates from {cycles} simulated cycles")
    if not cands:
        return ["obsmerge: no candidates"], []

    # ---- 3. induction, assuming seqcorr's equalities (an inductive invariant)
    cnf = _Cnf(g)
    S_ = cnf.S
    inv0 = []
    for c in base.classes:
        r, rp = c[0]
        for n, p in c[1:]:
            e = cnf.xor(cnf.node(0, n), cnf.node(0, r))
            inv0.append(-e if p == rp else e)
    rst0 = [cnf.node(0, n) for n in rst]

    _vm: Dict[Tuple[int, int], int] = {}

    def diff(f: int, c: Cand) -> int:
        """Literal: in frame f some bit of c's register differs from its x."""
        k = (f, id(c.bits))
        if k not in _vm:
            ds = []
            for _b, q, x in c.bits:
                xl = cnf.lit(f, x) if x >= 2 else (1 if x else -1)
                ds.append(cnf.xor(cnf.node(f, q), xl))
            z = cnf.new()
            cnf.S.add_clause([-z] + ds)
            for d in ds:
                cnf.S.add_clause([z, -d])
            _vm[k] = z
        return _vm[k]

    def viol(f: int, c: Cand) -> int:
        """Literal: frame f breaks candidate c (L and some bit differs)."""
        return cnf.and2(cnf.lit(f, c.live), diff(f, c))

    def sim_frames(mdl):
        pos = lambda var: 1 if mdl[var - 1] > 0 else 0
        pis0 = {}
        for n in range(1, len(g.fanin)):
            if g.fanin[n] is None:
                vv = cnf.var[0].get(n)
                pis0[n] = pos(vv) if vv is not None else 0
        v0 = g.simulate(pis0, 1)
        pis1 = {n: (pos(cnf.var[1][n]) if n in cnf.var[1] else 0) for n in inp}
        pis1.update({f.q: g.lit_val(v0, f.next, 1) for f in flops})
        return v0, g.simulate(pis1, 1)

    def regval(v, name: str) -> str:
        return hex(sum(((v[fq[i]] & 1) << flops[i].bit) for i in by_reg[name]))

    def holds(v, c: Cand, m: int = 1) -> int:
        """Mask of lanes where candidate c holds in node values v."""
        L = g.lit_val(v, c.live, m)
        d = 0
        for _b, q, x in c.bits:
            xv = (m if x else 0) if x < 2 else g.lit_val(v, x, m)
            d |= v[q] ^ xv
        return ~(L & d) & m

    def fixpoint(induct: bool) -> int:
        nonlocal cands
        it = 0
        while cands:
            it += 1
            assum = list(inv0) if induct else list(rst0)
            if induct:
                for c in cands:
                    assum.append(-viol(0, c))
            act = cnf.new()
            S_.add_clause([-act] + [viol(1, c) for c in cands])
            ok = S_.solve(assumptions=assum + [act])
            mdl = S_.get_model() if ok else None
            S_.add_clause([-act])
            if not ok:
                return it
            v0, v1 = sim_frames(mdl)
            n_ = len(cands)
            if debug:
                killed = [c for c in cands if not holds(v1, c)]
                for c in killed:
                    if watch and c.reg not in watch:
                        continue
                    srcs = sorted({flops[qidx[x >> 1]].name for _b, _q, x in c.bits if x >= 2})
                    show = [c.reg] + srcs + [c.ctl] + [n for n in ctls if n.endswith("state")]
                    dbg(f"killed {c.reg} when {c.ctl}=={c.val} ({'induction' if induct else 'base'})"
                        + "".join(f"\n    {n}: {regval(v0, n)} -> {regval(v1, n)}" for n in show))
            cands = [c for c in cands if holds(v1, c)]
            # forward from the counterexample's next state (see seqcorr)
            st = {f.q: (mask if g.lit_val(v0, f.next, 1) else 0) for f in flops}
            cur2 = {n: rnd() for n in inp}
            for _ in range(20):
                for n in inp:
                    cur2[n] ^= flips()
                for n in rst:
                    cur2[n] = 0
                vv = g.simulate({**st, **cur2}, mask)
                cands = [c for c in cands if holds(vv, c, mask) == mask]
                st = {f.q: g.lit_val(vv, f.next, mask) for f in flops}
            say(f"obsmerge: {'induction' if induct else 'base'} {it}: {n_} -> {len(cands)}")
        return it
    n0 = len(cands)
    it_b = fixpoint(False)
    it_i = fixpoint(True)
    say(f"obsmerge: {len(cands)} of {n0} candidates proved ({it_b} + {it_i} SAT iterations)")
    if not cands:
        return [f"obsmerge: none of {n0} candidates proved"], []
    inv = inv0 + [-viol(0, c) for c in cands]

    # ---- 4. choose a replacement per bit (replacements are bits that stay)
    xof: Dict[int, int] = {}
    for c in cands:
        for _b, q, x in c.bits:
            xof[q] = x
    removed: Dict[int, int] = {}
    pinned: Set[int] = set()
    blocked: Set[int] = set()

    def choose():
        removed.clear()
        pinned.clear()
        # constants first, then bits whose replacement can stand for nothing
        # itself (a source)
        order = sorted(xof, key=lambda q: (xof[q] >= 2, (xof[q] >> 1) in xof, q))
        for q in order:
            if q in pinned or q in blocked:
                continue
            x = xof[q]
            if x >= 2 and (x >> 1) in removed:
                continue
            removed[q] = x
            if x >= 2:
                pinned.add(x >> 1)

    # ---- 5. substitution check
    leaf = bl.leaf
    sinks_t = _sinks(nl)
    while True:
        choose()
        if not removed:
            return ["obsmerge: no merge passed the substitution check"], []
        leaf2 = dict(leaf)
        for name, idx in by_reg.items():
            ls = list(leaf[name])
            ch = False
            for k, i in enumerate(idx):
                x = removed.get(fq[i])
                if x is not None:
                    ls[flops[i].bit] = x
                    ch = True
            if ch:
                leaf2[name] = ls
        bl2 = _Blaster(g, leaf2, dict(nl.wires))
        pairs = []
        for t in sinks_t:
            pairs.extend(zip(bl.bits(t), bl2.bits(t)))
        for r in nl.regs:
            a, b = bl.bits(r.next), bl2.bits(r.next)
            for k, i in enumerate(by_reg[r.name]):
                if fq[i] not in removed:
                    pairs.append((a[flops[i].bit], b[flops[i].bit]))
        diff = [(a, b) for a, b in pairs if a != b]
        if not diff:
            break
        act = cnf.new()
        S_.add_clause([-act] + [cnf.xor(cnf.lit(0, a), cnf.lit(0, b)) for a, b in diff])
        ok = S_.solve(assumptions=inv + [act])
        mdl = S_.get_model() if ok else None
        S_.add_clause([-act])
        if not ok:
            break
        v0, _ = sim_frames(mdl)
        bad_q = [q for q, x in removed.items()
                 if v0[q] != ((x & 1) if x < 2 else g.lit_val(v0, x, 1))]
        if not bad_q:
            raise RuntimeError("obsmerge: substitution check failed with no differing bit")
        blocked.update(bad_q)
        say(f"obsmerge: substitution check: {len(bad_q)} bits kept")

    # ---- 6. apply
    flop_at = {f.q: f for f in flops}

    def x_term(x: int) -> T.Term:
        if x < 2:
            return T.const(x, 1)
        f = flop_at[x >> 1]
        t = T.slice_(T.var(f.name, regs[f.name].w), f.bit, 1)
        return T.not_(t) if x & 1 else t
    def x_str(x: int) -> str:
        if x < 2:
            return str(x)
        f = flop_at[x >> 1]
        return f"{'~' if x & 1 else ''}{f.name}[{f.bit}]"
    when: Dict[Tuple[str, int], List[Tuple[str, int]]] = collections.defaultdict(list)
    for c in cands:
        for b, _q, _x in c.bits:
            when[(c.reg, b)].append((c.ctl, c.val))
    m: Dict[T.Term, T.Term] = {}
    merges: List[Merge] = []
    for name, idx in by_reg.items():
        r = regs[name]
        q = T.var(name, r.w)
        bits = [T.slice_(q, b, 1) for b in range(r.w)]
        ch = False
        for i in idx:
            x = removed.get(fq[i])
            if x is not None:
                xt = x_term(x)
                bits[flops[i].bit] = xt
                ch = True
                merges.append(Merge(name, flops[i].bit, x_str(x), xt, when[(name, flops[i].bit)]))
        if ch:
            m[q] = T.concat(*reversed(bits))
    from .lower import _subst_all
    _subst_all(nl, m)
    # merged bits are read nowhere now: they hold their reset value
    for r in nl.regs:
        idx = by_reg[r.name]
        mb = {flops[i].bit for i in idx if fq[i] in removed}
        if mb:
            r.next = T.concat(*reversed([
                T.const((r.init >> b) & 1, 1) if b in mb else T.slice_(r.next, b, 1)
                for b in range(r.w)]))
    lines = [f"obsmerge: {len(merges)} register bits merged into what they equal while live "
             f"({n0} candidates, {len(cands)} proved by induction, {len(blocked)} kept by the "
             f"substitution check)"]
    by_w: Dict[str, List[str]] = collections.defaultdict(list)
    for mg in merges:
        by_w[mg.reg].append(f"[{mg.bit}]={mg.what}")
    for reg, items in by_w.items():
        conds = collections.defaultdict(set)
        for mg in merges:
            if mg.reg == reg:
                for cn, v in mg.when:
                    conds[cn].add(v)
        conds = {cn: [str(v) for v in sorted(vs)] for cn, vs in conds.items()}
        lines.append(f"obsmerge: {reg} {' '.join(items)} when "
                     + " or ".join(f"{cn} in {{{','.join(vs)}}}" for cn, vs in conds.items()))
    return lines, merges
