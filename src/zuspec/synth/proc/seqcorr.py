"""Sequential redundancy by induction (signal correspondence).

Finds the register bits, and the combinational signals, that are equal to one
another, to one another's complement, or to a constant in every state
reachable from reset. The method is van Eijk's: random simulation from reset
proposes candidate classes; SAT then checks that they hold right after reset
(base) and that, assumed all together in one cycle, they hold in the next
(induction). Each counterexample splits the classes it violates, until a
fixpoint. What is left is an inductive invariant, so every equality in it
holds in every reachable state.

Flop correspondence (``nodes="flops"``) only proposes register bits. Signal
correspondence (``nodes="all"``, the default) also proposes internal signals,
whose equalities strengthen the induction hypothesis.

The netlist is bit-blasted into an and-inverter graph (:func:`from_netlist`);
the result is applied back to the netlist's terms (:func:`apply`): a register
bit equal to a constant or to another register bit is replaced by it, and a
1-bit signal equal to an earlier one is replaced by that one. The existing
passes (sequential constants, cone of influence) then remove what is left
unread.

Needs python-sat (``pysat``) for the SAT checks.
"""
from __future__ import annotations

import collections
import dataclasses as dc
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import term as T


class SeqCorrError(Exception):
    pass


# ---------------------------------------------------------------- the AIG
class Aig:
    """And-inverter graph with structural hashing.

    A literal is ``2 * node + complement``; node 0 is constant false, so
    literal 0 is false and 1 is true. Nodes are created in topological order:
    primary inputs and flop outputs first (as they are met), then ands.
    """

    def __init__(self):
        self.fanin: List[Optional[Tuple[int, int]]] = [None]   # node 0: const
        self._strash: Dict[Tuple[int, int], int] = {}
        self.inputs: List[Tuple[str, int]] = []      # (name, node)
        self.outputs: List[Tuple[str, int]] = []     # (name, lit)
        self.flops: List["Flop"] = []

    # construction
    def pi(self) -> int:
        self.fanin.append(None)
        return 2 * (len(self.fanin) - 1)

    def and_(self, a: int, b: int) -> int:
        if a > b:
            a, b = b, a
        if a == 0:
            return 0
        if a == 1:
            return b
        if a == b:
            return a
        if a ^ 1 == b:
            return 0
        k = (a, b)
        n = self._strash.get(k)
        if n is None:
            self.fanin.append(k)
            n = len(self.fanin) - 1
            self._strash[k] = n
        return 2 * n

    def or_(self, a: int, b: int) -> int:
        return self.and_(a ^ 1, b ^ 1) ^ 1

    def xor(self, a: int, b: int) -> int:
        return self.or_(self.and_(a, b ^ 1), self.and_(a ^ 1, b))

    def mux(self, s: int, a: int, b: int) -> int:
        """s ? b : a"""
        return self.or_(self.and_(s ^ 1, a), self.and_(s, b))

    @property
    def n_ands(self) -> int:
        return sum(1 for f in self.fanin[1:] if f is not None)

    def is_and(self, n: int) -> bool:
        return self.fanin[n] is not None

    # simulation: bit-parallel over the bits of Python ints
    def simulate(self, pis: Dict[int, int], mask: int) -> List[int]:
        """``pis``: node -> value for inputs and flop outputs; returns node values."""
        v = [0] * len(self.fanin)
        for n, x in pis.items():
            v[n] = x
        for n, f in enumerate(self.fanin):
            if f is not None:
                a, b = f
                va = v[a >> 1] ^ (mask if a & 1 else 0)
                vb = v[b >> 1] ^ (mask if b & 1 else 0)
                v[n] = va & vb
        return v

    @staticmethod
    def lit_val(v: List[int], lit: int, mask: int) -> int:
        return v[lit >> 1] ^ (mask if lit & 1 else 0)


@dc.dataclass
class Flop:
    name: str                       # register name
    bit: int
    q: int                          # node of the flop output
    next: int = 0                   # literal of the next-state value (reset included)


# ---------------------------------------------------------------- bit-blasting
class _Blaster:
    """Terms -> AIG literals, least-significant bit first."""

    def __init__(self, g: Aig, leaf: Dict[str, List[int]], wires: Dict[str, T.Term]):
        self.g = g
        self.leaf = leaf            # register / input / net name -> bit literals
        self.wires = wires
        self.memo: Dict[T.Term, List[int]] = {}
        self.term_of: Dict[int, Tuple[T.Term, int]] = {}   # node -> (1-bit term, its lit)

    def bits(self, t0: T.Term) -> List[int]:
        stack = [t0]
        while stack:
            t = stack[-1]
            if t in self.memo:
                stack.pop()
                continue
            deps = list(t.args)
            if t.op == "var" and t.name in self.wires:
                deps = [self.wires[t.name]]
            todo = [a for a in deps if a not in self.memo]
            if todo:
                stack.extend(todo)
                continue
            stack.pop()
            r = self._op(t)
            assert len(r) == t.w, (t.op, t.w, len(r))
            self.memo[t] = r
            if t.w == 1 and (r[0] >> 1) not in self.term_of:
                self.term_of[r[0] >> 1] = (t, r[0])
        return self.memo[t0]

    def _op(self, t: T.Term) -> List[int]:
        g, op, w = self.g, t.op, t.w
        A = [self.memo[a] for a in t.args]
        if op == "const":
            return [(t.val >> i) & 1 for i in range(w)]
        if op == "var":
            if t.name in self.wires:
                return list(self.memo[self.wires[t.name]])
            if t.name not in self.leaf:
                raise SeqCorrError(f"unknown signal {t.name}")
            return list(self.leaf[t.name])
        if op == "slice":
            return A[0][t.val:t.val + w]
        if op == "concat":
            out: List[int] = []
            for a in reversed(A):
                out.extend(a)
            return out
        if op == "not":
            return [x ^ 1 for x in A[0]]
        if op in ("and", "or", "xor"):
            f = {"and": g.and_, "or": g.or_, "xor": g.xor}[op]
            out = list(A[0])
            for a in A[1:]:
                out = [f(x, y) for x, y in zip(out, a)]
            return out
        if op == "ite":
            c = A[0][0]
            return [g.mux(c, y, x) for x, y in zip(A[1], A[2])]
        if op == "add":
            return self._add(A[0], A[1], 0)
        if op == "sub":
            return self._add(A[0], [x ^ 1 for x in A[1]], 1)
        if op == "mul":
            acc = [0] * w
            for i, bi in enumerate(A[1]):
                part = [0] * i + [g.and_(x, bi) for x in A[0][:w - i]]
                acc = self._add(acc, part, 0)
            return acc
        if op == "eq":
            e = 1
            for x, y in zip(A[0], A[1]):
                e = g.and_(e, g.xor(x, y) ^ 1)
            return [e]
        if op in ("ult", "slt"):
            a, b = list(A[0]), list(A[1])
            if op == "slt":
                a[-1] ^= 1
                b[-1] ^= 1
            # a < b  <=>  no carry out of a + ~b + 1
            lt = 0
            for x, y in zip(a, b):
                # lt' = (~x & y) | (~(x ^ y) & lt)
                lt = g.or_(g.and_(x ^ 1, y), g.and_(g.xor(x, y) ^ 1, lt))
            return [lt]
        if op in ("shl", "lshr", "ashr"):
            a, n = list(A[0]), A[1]
            fill = a[-1] if op == "ashr" else 0
            big = 0
            for k, s in enumerate(n):
                sh = 1 << k
                if sh >= w:
                    big = g.or_(big, s)
                    continue
                if op == "shl":
                    moved = [0] * sh + a[:w - sh]
                else:
                    moved = a[sh:] + [fill] * sh
                a = [g.mux(s, x, y) for x, y in zip(a, moved)]
            return [g.mux(big, x, fill) for x in a]
        if op in ("redor", "redand", "redxor"):
            f = {"redor": g.or_, "redand": g.and_, "redxor": g.xor}[op]
            r = A[0][0]
            for x in A[0][1:]:
                r = f(r, x)
            return [r]
        raise SeqCorrError(f"term op {op}")

    def _add(self, a: List[int], b: List[int], c: int) -> List[int]:
        g = self.g
        out = []
        for x, y in zip(a, b):
            s = g.xor(x, y)
            out.append(g.xor(s, c))
            c = g.or_(g.and_(x, y), g.and_(s, c))
        return out


def from_netlist(nl) -> Tuple[Aig, _Blaster]:
    """The AIG of a :class:`~zuspec.synth.proc.lower.Netlist`. Register bits
    come first (so a register bit is only ever merged into another register
    bit or a constant), then inputs and instance-driven nets (free), then the
    logic. The reset is synchronous: next = reset ? init : next."""
    g = Aig()
    leaf: Dict[str, List[int]] = {}
    for r in nl.regs:
        leaf[r.name] = [g.pi() for _ in range(r.w)]
        for b, l in enumerate(leaf[r.name]):
            g.flops.append(Flop(r.name, b, l >> 1))
    for n, w in list(nl.inputs) + list(nl.nets):
        if n == nl.clock or n in leaf:
            continue
        leaf[n] = [g.pi() for _ in range(w)]
        for b, l in enumerate(leaf[n]):
            g.inputs.append((n if w == 1 else f"{n}[{b}]", l >> 1))
    # the clock only reaches instance ports; as a value it is never read
    leaf[nl.clock] = [0]
    bl = _Blaster(g, leaf, dict(nl.wires))
    rst = leaf[nl.reset][0] if nl.reset else 0
    fl = iter(g.flops)
    for r in nl.regs:
        nb = bl.bits(r.next)
        for b in range(r.w):
            f = next(fl)
            f.next = g.mux(rst, nb[b], (r.init >> b) & 1)
    for n, t in nl.outputs:
        for b, l in enumerate(bl.bits(t)):
            g.outputs.append((n if t.w == 1 else f"{n}[{b}]", l))
    for n, t in nl.wires:
        for b, l in enumerate(bl.bits(t)):
            g.outputs.append((f"{n}[{b}]", l))
    for inst in nl.instances:
        for c in inst.conns:
            t = c.term if c.term is not None else (
                T.var(c.var.name, c.var.w) if c.var is not None and c.var.name in leaf else None)
            if t is not None:
                for b, l in enumerate(bl.bits(t)):
                    g.outputs.append((f"{inst.name}.{c.port}[{b}]", l))
    return g, bl


# ---------------------------------------------------------------- correspondence
@dc.dataclass
class Result:
    aig: Aig
    classes: List[List[Tuple[int, int]]]       # [(node, phase)], representative first
    stats: Dict[str, int]

    def rep_map(self) -> Dict[int, int]:
        """node -> literal of its representative (0/1 for a constant)."""
        out = {}
        for c in self.classes:
            r, rp = c[0]
            for n, p in c[1:]:
                out[n] = 2 * r + (p ^ rp)
        return out

    def flop_facts(self) -> List[Tuple[str, str]]:
        """(flop, what it equals): a constant, a flop, or a node."""
        g = self.aig
        fname = {f.q: f"{f.name}[{f.bit}]" for f in g.flops}
        out = []
        triv = {f.q for f in g.flops if f.next in (0, 1)}
        for n, l in sorted(self.rep_map().items()):
            if n not in fname or n in triv:
                continue
            r = l >> 1
            neg = "~" if l & 1 else ""
            if r == 0:
                what = str(l & 1)
            elif r in fname:
                what = neg + fname[r]
            else:
                what = f"{neg}node {r}"
            out.append((fname[n], what))
        return out

    def live_flops(self) -> Set[int]:
        """Flops left in the cone of the outputs once every member is replaced
        by its representative (with constants propagated)."""
        g = self.aig
        rm = self.rep_map()
        new = Aig()
        sub: Dict[int, int] = {0: 0}
        pis: Dict[int, int] = {}
        # representatives are the lowest node of their class, so they are
        # rebuilt before any node that reads a member
        for n in range(1, len(g.fanin)):
            if g.fanin[n] is None:
                pis[n] = sub[n] = new.pi()
            elif n not in rm:
                a, b = g.fanin[n]
                sub[n] = new.and_(_s(sub, rm, a), _s(sub, rm, b))
        fl = {f.q: f for f in g.flops}
        live: Set[int] = set()
        seen: Set[int] = set()
        stack = [_s(sub, rm, l) >> 1 for _, l in g.outputs]
        back = {v >> 1: k for k, v in pis.items()}
        while stack:
            n = stack.pop()
            if n in seen or n == 0:
                continue
            seen.add(n)
            f = new.fanin[n]
            if f is not None:
                stack.extend((f[0] >> 1, f[1] >> 1))
            else:
                o = back[n]
                if o in fl:
                    live.add(o)
                    stack.append(_s(sub, rm, fl[o].next) >> 1)
        return live


def _s(sub, rm, l: int) -> int:
    n = l >> 1
    if n in rm:
        r = rm[n]
        return sub[r >> 1] ^ (r & 1) ^ (l & 1)
    return sub[n] ^ (l & 1)


def correspond(g: Aig, reset: Sequence[str], *, nodes: str = "all", cycles: int = 1000,
               width: int = 256, seed: int = 1, forward: int = 20, log=None,
               cand_nodes: Optional[Sequence[int]] = None) -> Result:
    """Signal (``nodes="all"``) or flop (``"flops"``) correspondence.
    ``cand_nodes``: propose these nodes (and the flops) instead.

    ``reset``: the names of the active-high reset inputs. The reachable
    states are those after a cycle with reset asserted, from any state, with
    any other inputs.
    """
    from pysat.solvers import Solver

    say = log or (lambda s: None)
    mask = (1 << width) - 1
    rng = random.Random(seed)
    rnd = lambda: rng.getrandbits(width)
    inp = [n for _, n in g.inputs]
    rst = [n for nm, n in g.inputs if nm in reset or nm.split("[")[0] in reset]
    if not rst:
        raise SeqCorrError(f"no reset input among {reset}")
    flops = [f.q for f in g.flops]
    if cand_nodes is not None:
        fs = set(flops)
        cand = flops + sorted(n for n in set(cand_nodes) if n not in fs and g.is_and(n))
    else:
        cand = flops + ([n for n in range(1, len(g.fanin)) if g.is_and(n)] if nodes == "all"
                        else [])

    def step(state: Dict[int, int], ins: Dict[int, int], m: int):
        v = g.simulate({**state, **ins}, m)
        return v, {f.q: g.lit_val(v, f.next, m) for f in g.flops}

    # 1. simulation from reset: inputs hold their values for a while (lanes
    #    in four groups flip with probability 1/2, 1/8, 1/32, 1/128 per cycle);
    #    reset is rare
    grp = width // 4
    gm = [((1 << grp) - 1) << (grp * k) for k in range(4)]

    def flips() -> int:
        f, x = 0, rnd()
        for k in range(4):
            if k:
                x &= rnd() & rnd()
            f |= x & gm[k]
        return f
    cur = {n: rnd() for n in inp}
    for n in rst:
        cur[n] = mask
    _, state = step({q: rnd() for q in flops}, cur, mask)
    phase: Dict[int, int] = {}
    sig: Dict[int, int] = {n: 0 for n in cand}
    for cyc in range(cycles):
        for n in inp:
            cur[n] ^= flips()
        for n in rst:
            x = mask
            for _ in range(8):
                x &= rnd()
            cur[n] = x
        v, nxt = step(state, cur, mask)
        for n in cand:
            x = v[n]
            if cyc == 0:
                phase[n] = x & 1
            if phase[n]:
                x ^= mask
            sig[n] = hash((sig[n], x))
        state = nxt
    zero = 0
    for _ in range(cycles):
        zero = hash((zero, 0))
    by: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for n in cand:
        by[sig[n]].append((n, phase[n]))
    classes = []
    for k, mem in by.items():
        if k == zero:
            classes.append([(0, 0)] + mem)
        elif len(mem) > 1:
            classes.append(mem)
    stats = {"flops": len(flops), "ands": g.n_ands}

    def count():
        return sum(len(c) - 1 for c in classes)
    stats["sim_candidates"] = count()
    say(f"seqcorr: {len(flops)} flops, {g.n_ands} ands; {count()} candidates "
        f"from {cycles} simulated cycles")

    # 2. SAT: frame 0 (state free) and frame 1 (state = next of frame 0)
    nv = [0]

    def new() -> int:
        nv[0] += 1
        return nv[0]
    T = new()
    clauses = [[T]]

    def frame(state_lits: Dict[int, int]) -> List[int]:
        var = [0] * len(g.fanin)
        var[0] = -T
        for n in range(1, len(g.fanin)):
            f = g.fanin[n]
            if f is None:
                var[n] = state_lits[n] if n in state_lits else new()
            else:
                a = var[f[0] >> 1] * (-1 if f[0] & 1 else 1)
                b = var[f[1] >> 1] * (-1 if f[1] & 1 else 1)
                z = new()
                var[n] = z
                clauses.extend(([-z, a], [-z, b], [z, -a, -b]))
        return var

    def sl(var, l: int) -> int:
        return var[l >> 1] * (-1 if l & 1 else 1)
    V0 = frame({})
    V1 = frame({f.q: sl(V0, f.next) for f in g.flops})
    S = Solver(name="cadical153", bootstrap_with=clauses)
    xc: Dict[Tuple[int, int], int] = {}

    def xor(a: int, b: int) -> int:
        k = (a, b) if a < b else (b, a)
        z = xc.get(k)
        if z is None:
            z = new()
            xc[k] = z
            S.add_clause([-z, a, b])
            S.add_clause([-z, -a, -b])
            S.add_clause([z, -a, b])
            S.add_clause([z, a, -b])
        return z

    def refine(val, m: int = 1) -> None:
        nonlocal classes
        out = []
        for c in classes:
            sp: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
            for n, p in c:
                sp[val(n) ^ (m if p else 0)].append((n, p))
            for k in sp.values():
                if len(k) > 1:
                    p0 = k[0][1]
                    out.append([(n, p ^ p0) for n, p in k])
        out = [c for c in out if len(c) > 1]
        classes = out

    def fixpoint(induct: bool) -> int:
        it = 0
        while True:
            it += 1
            assum = [] if induct else [V0[n] for n in rst]
            miters = []
            for c in classes:
                r, rp = c[0]
                for n, p in c[1:]:
                    if induct:
                        e = xor(V0[n], V0[r])
                        assum.append(-e if p == rp else e)
                    e = xor(V1[n], V1[r])
                    miters.append(e if p == rp else -e)
            if not miters:
                return it
            act = new()
            S.add_clause([-act] + miters)
            ok = S.solve(assumptions=assum + [act])
            mdl = S.get_model() if ok else None
            S.add_clause([-act])
            if not ok:
                return it
            pos = lambda var: 1 if mdl[var - 1] > 0 else 0
            pis0 = {n: pos(V0[n]) for n in range(1, len(g.fanin)) if g.fanin[n] is None}
            v0 = g.simulate(pis0, 1)
            pis1 = {n: pos(V1[n]) for n in inp}
            pis1.update({f.q: g.lit_val(v0, f.next, 1) for f in g.flops})
            v1 = g.simulate(pis1, 1)
            refine(lambda n: v1[n])
            # The counterexample's frame 0 satisfies every candidate, so also
            # the invariant the fixpoint will end with; that invariant is
            # inductive, so every state reached from here satisfies it too.
            # Refining on those states (all lanes, random held inputs) is
            # sound and loses no true equality.
            st = {f.q: (mask if g.lit_val(v0, f.next, 1) else 0) for f in g.flops}
            cur = {n: rnd() for n in inp}
            for _ in range(forward):
                for n in inp:
                    cur[n] ^= flips()
                for n in rst:
                    cur[n] = 0
                v, st = step(st, cur, mask)
                refine(lambda n: v[n], mask)

    it_b = fixpoint(False)
    stats["after_base"] = count()
    it_i = fixpoint(True)
    stats["proved"] = count()
    stats["iterations"] = it_b + it_i
    res = Result(g, sorted(classes, key=lambda c: c[0][0]), stats)
    fset = set(flops)
    # a bit whose next value is a constant is a constant already: not counted
    triv = {f.q for f in g.flops if f.next in (0, 1)}
    stats["flops"] = len(flops) - len(triv)
    stats["flops_merged"] = sum(1 for c in classes for n, _ in c[1:] if n in fset and n not in triv)
    stats["flops_live"] = len(res.live_flops() - triv)
    say(f"seqcorr: base {it_b} and induction {it_i} SAT iterations; {count()} nodes "
        f"proved redundant, {stats['flops_merged']} of them flops; "
        f"{stats['flops_live']} of {len(flops)} flops left after merging")
    return res


# ---------------------------------------------------------------- applying
def apply(nl, res: Result, bl: _Blaster) -> List[str]:
    """Rewrite the netlist's terms with the proven equalities.

    A register bit equal to a constant or to another register bit, and a
    1-bit term equal to an earlier signal that some 1-bit term computes, are
    replaced by that. Representatives are always earlier nodes, so no
    replacement can reach back to what it replaces. Returns report lines."""
    rm = res.rep_map()
    term_of = bl.term_of
    regs = {r.name: r for r in nl.regs}
    flop_at = {f.q: f for f in res.aig.flops}

    def rep_term(lit: int) -> Optional[T.Term]:
        """A term with the value of literal *lit*'s node after merging, xor
        the literal's own complement."""
        n = lit >> 1
        if n not in rm:
            return None
        r = rm[n]
        val_ph = (r & 1) ^ (lit & 1)          # value(lit) = value(node r) ^ val_ph
        rn = r >> 1
        if rn == 0:
            return T.const(val_ph, 1)
        if rn in flop_at:
            f = flop_at[rn]
            x = T.slice_(T.var(f.name, regs[f.name].w), f.bit, 1)
            return T.not_(x) if val_ph else x
        hit = term_of.get(rn)
        if hit is None:
            return None
        rt, rl = hit
        return T.not_(rt) if val_ph ^ (rl & 1) else rt

    memo: Dict[T.Term, T.Term] = {}

    def sub(t0: T.Term) -> T.Term:
        stack = [t0]
        while stack:
            t = stack[-1]
            if t in memo:
                stack.pop()
                continue
            if t.w == 1 and t in bl.memo:
                r = rep_term(bl.memo[t][0])
                if r is not None and r is not t:
                    # the representative is rewritten too (it is built from
                    # earlier nodes only, so this cannot come back to t)
                    if r not in memo:
                        stack.append(r)
                        continue
                    memo[t] = memo[r]
                    stack.pop()
                    continue
            if t.op == "var" and t.name in regs and t in bl.memo:
                bits = [rep_term(l) for l in bl.memo[t]]
                if any(b is not None for b in bits):
                    memo[t] = T.concat(*reversed([
                        b if b is not None else T.slice_(t, i, 1) for i, b in enumerate(bits)]))
                else:
                    memo[t] = t
                stack.pop()
                continue
            todo = [a for a in t.args if a not in memo]
            if todo:
                stack.extend(todo)
                continue
            stack.pop()
            args = [memo[a] for a in t.args]
            memo[t] = T.rebuild(t, args) if any(x is not y for x, y in zip(args, t.args)) else t
        return memo[t0]

    for r in nl.regs:
        r.next = sub(r.next)
    # an output that is a merged register reads its representative
    nl.outputs = [(n, sub(t)) for n, t in nl.outputs]
    nl.wires = [(n, sub(t)) for n, t in nl.wires]
    for inst in nl.instances:
        for c in inst.conns:
            if c.term is not None:
                c.term = sub(c.term)
    # the value of every register var, read here or not (code in the
    # source's structure may read registers these terms no longer do)
    for r in nl.regs:
        q = T.var(r.name, r.w)
        if q not in memo:
            sub(q)
    nl.record_subst({q: memo[q] for q in (T.var(r.name, r.w) for r in nl.regs)
                     if q in memo and memo[q] is not q})
    # merged register bits are read nowhere now: they hold their reset value
    merged: Dict[str, Set[int]] = {}
    for f in res.aig.flops:
        if f.q in rm:
            merged.setdefault(f.name, set()).add(f.bit)
    for r in nl.regs:
        bs = merged.get(r.name)
        if bs:
            r.next = T.concat(*reversed([
                T.const((r.init >> b) & 1, 1) if b in bs else T.slice_(r.next, b, 1)
                for b in range(r.w)]))
    st = res.stats
    lines = [f"seqcorr: {st['proved']} signals proved equal to earlier ones or constant "
             f"({st['sim_candidates']} candidates from simulation, {st['iterations']} SAT "
             f"iterations); {st['flops_merged']} register bits merged, "
             f"{st['flops'] - st['flops_live']} register bits removable"]
    facts = res.flop_facts()
    if facts:
        by: Dict[str, List[str]] = {}
        for f, what in facts:
            by.setdefault(what if what in ("0", "1") else "other", []).append(
                f if what in ("0", "1") else f"{f} = {what}")
        for k, v in by.items():
            lines.append(f"seqcorr: {'constant ' + k if k != 'other' else 'equal'}: " + ", ".join(v))
    return lines


def optimize(nl, *, nodes: str = "all", cycles: int = 1000, seed: int = 1, log=None) -> List[str]:
    """Signal correspondence over the whole netlist, applied in place."""
    if not nl.reset:
        return ["seqcorr: skipped (no reset)"]
    g, bl = from_netlist(nl)
    res = correspond(g, [nl.reset], nodes=nodes, cycles=cycles, seed=seed, log=log)
    return apply(nl, res, bl)
