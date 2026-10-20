"""Lower clocked processes to a register-transfer netlist.

Each process becomes a state machine whose states are its awaits. For every
state, one clock cycle is executed symbolically: from the await, through the
code, to the awaits the cycle ends at. Branches, a ``wait`` whose condition
already holds, and loop exits cost no cycle; only clock edges do. The result
per state is, for every var, its value at the end of the cycle, and the next
state, as bit-vector terms.

Writes are then resolved per bit. A bit a process writes back unchanged
simplifies to the register bit itself, so it is not a write: a struct
read-modify-write only writes the fields it changed. Bits that several
processes write go to the highest-priority writer that writes them.

Semantics are those of :mod:`zuspec.synth.proc.ir`. Two choices are made here:

  - a ``wait`` reached a second time in one cycle ends the cycle (it is checked
    again in the next one), so a loop through a wait never becomes a
    combinational loop;
  - a cycle that loops back without passing any await is an error (a loop that
    never waits for the clock).
"""
from __future__ import annotations

import dataclasses as dc
from typing import Dict, List, Optional, Set, Tuple

from . import ir as P
from . import term as T


class LowerError(Exception):
    pass


# ======================================================================= CFG
class Node:
    __slots__ = ("id", "kind", "var", "value", "nb", "cond", "next", "alt", "await_",
                 "label")

    def __init__(self, nid: int, kind: str):
        self.id = nid
        self.kind = kind          # 'assign' | 'branch' | 'await' | 'end' | 'nop'
        self.var: Optional[P.Var] = None
        self.value: Optional[T.Term] = None
        self.nb = False
        self.cond: Optional[T.Term] = None
        self.next: Optional[Node] = None
        self.alt: Optional[Node] = None   # branch: taken when cond is false
        self.await_: Optional[P.Await] = None
        self.label = ""

    def __repr__(self) -> str:
        return f"<{self.kind}{self.id}>"


class CFG:
    def __init__(self, proc: P.Proc):
        self.nodes: List[Node] = []
        self.awaits: List[Node] = []
        self.initial: Optional[Node] = None
        self.end = self._new("end")
        entry = self._build(proc.body, self.end, None, None)
        # the process body runs once; when it falls off the end it stays done
        self.entry = entry
        if self.initial is None:
            raise LowerError(f"process {proc.name} has no initial await")

    def _new(self, kind: str) -> Node:
        n = Node(len(self.nodes), kind)
        self.nodes.append(n)
        return n

    def _build(self, stmts: List[P.Stmt], succ: Node, brk: Optional[Node],
               cont: Optional[Node]) -> Node:
        cur = succ
        for s in reversed(stmts):
            cur = self._stmt(s, cur, brk, cont)
        return cur

    def _stmt(self, s, succ: Node, brk, cont) -> Node:
        if isinstance(s, P.Assign):
            n = self._new("assign")
            n.var, n.value, n.nb, n.next = s.var, s.value, s.nb, succ
            return n
        if isinstance(s, P.If):
            n = self._new("branch")
            n.cond = s.cond
            n.next = self._build(s.then, succ, brk, cont)
            n.alt = self._build(s.orelse, succ, brk, cont)
            return n
        if isinstance(s, P.Await):
            n = self._new("await")
            n.await_ = s
            n.cond = s.cond
            n.next = succ
            n.label = s.label
            self.awaits.append(n)
            if s.initial:
                if self.initial is not None:
                    raise LowerError("two initial awaits")
                self.initial = n
            return n
        if isinstance(s, P.Loop):
            head = self._new("nop")          # loop head (the pre-test)
            exit_ = succ
            if s.post is not None:
                # do { body } while (post): body first, then test
                test = self._new("branch")
                test.cond = s.post
                test.next = head
                test.alt = exit_
                body = self._build(s.body, test, exit_, test)
                head.next = body
                if s.pre is not None:
                    pre = self._new("branch")
                    pre.cond = s.pre
                    pre.next = body
                    pre.alt = exit_
                    head.next = pre
                return head
            body = self._build(s.body, head, exit_, head)
            if s.pre is not None:
                pre = self._new("branch")
                pre.cond = s.pre
                pre.next = body
                pre.alt = exit_
                head.next = pre
            else:
                head.next = body
            return head
        if isinstance(s, P.Break):
            if brk is None:
                raise LowerError("break outside a loop")
            n = self._new("nop")
            n.next = brk
            return n
        if isinstance(s, P.Continue):
            if cont is None:
                raise LowerError("continue outside a loop")
            n = self._new("nop")
            n.next = cont
            return n
        raise LowerError(f"unknown statement {s!r}")


# ================================================================ symbolic
@dc.dataclass
class Env:
    """Values at a program point within one cycle."""
    cur: Dict[P.Var, T.Term]                       # blocking-visible value
    nbw: Dict[P.Var, Tuple[T.Term, T.Term]]        # var -> (written, value)


@dc.dataclass
class Leaf:
    cond: T.Term
    next: Node                                     # the await the cycle ends at
    env: Env


class CycleExec:
    """Symbolic execution of one cycle from one state."""

    def __init__(self, cfg: CFG, qref: Dict[P.Var, T.Term], proc_name: str,
                 dead: Optional[Dict[P.Var, T.Term]] = None):
        self.cfg = cfg
        self.q = qref                              # var -> its register/net term
        self.proc_name = proc_name
        # private vars with no value across a clock edge: read before written
        # in a cycle they are don't-care (0), so they never need a register
        self.dead = dead or {}

    def rd(self, v: P.Var) -> T.Term:
        return self.dead.get(v, self.q[v])

    def run(self, start: Node) -> List[Leaf]:
        return self.run_multi([(start, T.ONE)])

    def run_multi(self, starts: List[Tuple[Node, T.Term]]) -> List[Leaf]:
        """One cycle from several states at once, each guarded by its state
        condition. Paths from different states that reach the same program
        point merge there, so code after a join (the rest of a subroutine
        called from several states, a common tail) is evaluated once over
        muxed values instead of once per state."""
        env0 = Env(cur={}, nbw={})
        leaves: List[Leaf] = []
        pending: Dict[tuple, List[Tuple[T.Term, Env]]] = {}
        order: List[tuple] = []
        graph_nodes: Dict[tuple, Node] = {}
        first = []
        for start, at in starts:
            if start.kind != "await":
                raise LowerError("a state must be an await")
            if start.cond is None:
                first.append((start.next, frozenset(), at))
            else:
                c = self._eval(start.cond, env0)
                leaves.append(Leaf(T.and_(at, T.not_(c)), start, env0))
                first.append((start.next, frozenset([start.id]), T.and_(at, c)))
        keys = []
        for node, passed, c in first:
            key = self._key(node, passed)
            keys.append((key, node))
            self._add(pending, order, graph_nodes, key, node, c, env0)
        # topological processing of the (node, passed) DAG
        for key in self._toposort(keys):
            contribs = pending.get(key)
            if not contribs:
                continue
            node = graph_nodes[key]
            c, env = self._merge(contribs)
            if c.is_const and not c.val:
                continue
            self._step(node, key[1], c, env, leaves, pending, order, graph_nodes)
        return leaves

    def _key(self, node: Node, passed: frozenset) -> tuple:
        """The DAG key of a program point. Of the waits already passed this
        cycle only those that can still be reached matter (reaching one again
        ends the cycle); dropping the others lets paths from different states
        merge."""
        if passed:
            passed = passed & self._reach()[node.id]
        return (node.id, passed)

    def _reach(self) -> Dict[int, frozenset]:
        """Per node: the conditional waits reachable from it within a cycle
        (not through a clock edge, which always ends the cycle)."""
        r = self.cfg.__dict__.get("_cond_reach")
        if r is not None:
            return r
        nodes = self.cfg.nodes
        reach: Dict[int, Set[int]] = {n.id: set() for n in nodes}
        changed = True
        while changed:
            changed = False
            for n in reversed(nodes):
                if n.kind == "end" or (n.kind == "await" and n.cond is None):
                    continue
                cur = reach[n.id]
                new = set(cur)
                if n.kind == "await":
                    new.add(n.id)
                for x in (n.next, n.alt):
                    if x is not None:
                        new |= reach[x.id]
                if len(new) != len(cur):
                    reach[n.id] = new
                    changed = True
        r = {k: frozenset(v) for k, v in reach.items()}
        self.cfg.__dict__["_cond_reach"] = r
        return r

    # -- DAG order -------------------------------------------------------
    def _succ(self, node: Node, passed: frozenset):
        """Successor (node, passed) keys within the cycle."""
        k = node.kind
        if k == "assign" or k == "nop":
            return [self._key(node.next, passed)]
        if k == "branch":
            return [self._key(node.next, passed), self._key(node.alt, passed)]
        if k == "await":
            if node.cond is None or node.id in passed:
                return []
            return [self._key(node.next, passed | {node.id})]
        return []

    def _toposort(self, first: List[Tuple[tuple, Node]]) -> List[tuple]:
        order: List[tuple] = []
        state: Dict[tuple, int] = {}       # 1 = on stack, 2 = done
        byid = {n.id: n for n in self.cfg.nodes}

        def visit(key):
            st = state.get(key)
            if st == 2:
                return
            if st == 1:
                node = byid[key[0]]
                raise LowerError(f"process {self.proc_name}: a loop that never waits "
                                 f"for the clock (through node {node.id} {node.label})")
            state[key] = 1
            for kk in self._succ(byid[key[0]], key[1]):
                visit(kk)
            state[key] = 2
            order.append(key)

        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 100000))
        try:
            for key, _node in first:
                visit(key)
        finally:
            sys.setrecursionlimit(old)
        order.reverse()
        return order

    def _add(self, pending, order, graph_nodes, key, node, c, env) -> None:
        if c.is_const and not c.val:
            return
        graph_nodes[key] = node
        pending.setdefault(key, []).append((c, env))

    # -- one node --------------------------------------------------------
    def _step(self, node: Node, passed, c, env: Env, leaves, pending, order,
              graph_nodes) -> None:
        k = node.kind

        def go(nx: Node, cond: T.Term, e: Env, p=passed):
            self._add(pending, order, graph_nodes, self._key(nx, p), nx, cond, e)
        if k == "nop":
            go(node.next, c, env)
        elif k == "assign":
            val = self._eval(node.value, env)
            if node.nb:
                nbw = dict(env.nbw)
                nbw[node.var] = (T.ONE, val)
                env2 = Env(cur=env.cur, nbw=nbw)
            else:
                cur = dict(env.cur)
                cur[node.var] = val
                env2 = Env(cur=cur, nbw=env.nbw)
            go(node.next, c, env2)
        elif k == "branch":
            b = T.bool_(self._eval(node.cond, env))
            go(node.next, T.and_(c, b), env)
            go(node.alt, T.and_(c, T.not_(b)), env)
        elif k == "await":
            if node.cond is None or node.id in passed:
                leaves.append(Leaf(c, node, env))
                return
            w = T.bool_(self._eval(node.cond, env))
            leaves.append(Leaf(T.and_(c, T.not_(w)), node, env))
            go(node.next, T.and_(c, w), env, passed | {node.id})
        elif k == "end":
            leaves.append(Leaf(c, node, env))
        else:
            raise LowerError(f"bad node {k}")

    # -- values ----------------------------------------------------------
    def _eval(self, t: T.Term, env: Env) -> T.Term:
        """A term over vars, with each var replaced by its current value."""
        m: Dict[T.Term, T.Term] = {}
        for v, d0 in self.dead.items():
            m[self.q[v]] = d0
        for v, val in env.cur.items():
            m[self.q[v]] = val
        if not m:
            return t
        return T.substitute(t, m)

    def _merge(self, contribs: List[Tuple[T.Term, Env]]) -> Tuple[T.Term, Env]:
        if len(contribs) == 1:
            return contribs[0]
        conds = [c for c, _ in contribs]
        cond = T.or_(*conds)
        keys_cur: Set[P.Var] = set()
        keys_nb: Set[P.Var] = set()
        for _, e in contribs:
            keys_cur.update(e.cur)
            keys_nb.update(e.nbw)
        cur: Dict[P.Var, T.Term] = {}
        for v in keys_cur:
            vals = [(c, e.cur.get(v, self.rd(v))) for c, e in contribs]
            cur[v] = _select(vals)
        nbw: Dict[P.Var, Tuple[T.Term, T.Term]] = {}
        for v in keys_nb:
            ws = [(c, e.nbw.get(v, (T.ZERO, self.rd(v)))) for c, e in contribs]
            we = _select([(c, w[0]) for c, w in ws])
            val = _select([(c, w[1]) for c, w in ws])
            nbw[v] = (we, val)
        return cond, Env(cur=cur, nbw=nbw)


def _select(vals: List[Tuple[T.Term, T.Term]]) -> T.Term:
    """ite chain over disjoint conditions; the last value is the default.

    The select is only used where one of the conditions holds, so conjuncts
    common to all of them (the condition of the code around the branch that
    split them) are dropped from each."""
    if _DROP_COMMON[0] and len(vals) > 1:
        live = [c for c, _ in vals if not (c.is_const and not c.val)]
        if len(live) > 1:
            common = frozenset.intersection(*[T.conjuncts(c) for c in live])
            if common:
                vals = [(c if c.is_const else T.and_(*(T.conjuncts(c) - common)), v)
                        for c, v in vals]
    if _GROUP[0]:
        return _select_grouped(vals)
    out = vals[-1][1]
    for c, v in reversed(vals[:-1]):
        out = T.ite(c, v, out)
    return out


_GROUP = [True]
_DROP_COMMON = [True]


def _select_grouped(vals: List[Tuple[T.Term, T.Term]]) -> T.Term:
    """The same select with the arms grouped by value: one OR of conditions
    per distinct value, and the value with the most arms (usually "keeps its
    value") is the default, so it needs no condition at all. Correct because
    the conditions are disjoint and the select is only used where one of them
    holds."""
    conds: Dict[T.Term, List[T.Term]] = {}
    for c, v in vals:
        if c.is_const and not c.val:
            continue
        conds.setdefault(v, []).append(c)
    if not conds:
        return vals[-1][1]
    if len(conds) == 1:
        return next(iter(conds))
    groups = list(conds.items())
    # default: the group with the most arms; ties go to the later one (the
    # old default)
    di = max(range(len(groups)), key=lambda i: (len(groups[i][1]), i))
    dv = groups[di][0]
    out = dv
    if _GROUP[0] == "or":
        for i in reversed(range(len(groups))):
            if i != di:
                out = T.ite(T.or_(*groups[i][1]), groups[i][0], out)
        return out
    # the other arms stay as they were, in order (no OR of conditions)
    for c, v in reversed(vals):
        if v is not dv and not (c.is_const and not c.val):
            out = T.ite(c, v, out)
    return out


# ================================================================ netlist
@dc.dataclass
class Reg:
    name: str
    w: int
    init: int
    next: T.Term
    origin: str = ""


@dc.dataclass
class Netlist:
    name: str
    inputs: List[Tuple[str, int]]
    outputs: List[Tuple[str, T.Term]]
    regs: List[Reg]
    wires: List[Tuple[str, T.Term]]
    instances: List[P.Instance]
    clock: str
    reset: Optional[str]
    report: List[str] = dc.field(default_factory=list)
    nets: List[Tuple[str, int]] = dc.field(default_factory=list)   # driven by instances


@dc.dataclass
class LowerOptions:
    encoding: str = "binary"         # 'binary' | 'onehot' | 'factored'
    # execute all states of a process in one pass, merging paths from
    # different states where they join (see CycleExec.run_multi)
    share_joins: bool = True
    # selects grouped by value (see _select_grouped)
    group_selects: object = True     # False | True (drop default arms) | "or"
    # selects drop conjuncts common to all their conditions (see _select)
    drop_common: bool = False
    # 1-bit ORs of conjunctions simplified where paths rejoin (term.or_)
    simplify_or: bool = False
    # netlist optimizer: register correspondence (see _reg_corr)
    reg_corr: bool = True
    # netlist optimizer: sequential redundancy by SAT induction over the
    # whole netlist (seqcorr.optimize): False | "flops" | "all"
    seqcorr: object = "all"


def lower(d: P.Design, opts: Optional[LowerOptions] = None) -> Netlist:
    opts = opts or LowerOptions()
    _GROUP[0] = opts.group_selects
    _DROP_COMMON[0] = opts.drop_common
    T.SIMPLIFY_OR[0] = opts.simplify_or
    _OPT["reg_corr"] = opts.reg_corr
    _OPT["seqcorr"] = opts.seqcorr
    qref: Dict[P.Var, T.Term] = {v: T.var(v.name, v.w) for v in d.vars}
    report: List[str] = []
    regs: List[Reg] = []
    # per var: [(proc priority, proc index, next term)]
    writers: Dict[P.Var, List[Tuple[int, int, T.Term]]] = {}
    comb_wires = _lower_comb(d, qref, report)
    seq = [p for p in d.procs if not p.comb]
    if len(seq) != len(d.procs):
        d = dc.replace(d, procs=seq)
    cfgs = [CFG(proc) for proc in d.procs]
    # what combinational processes write or read is observed by them
    comb_seen = set(comb_wires)
    for t in comb_wires.values():
        comb_seen |= {x.name for x in T.leaves(t)}
    dead = _dead_private(d, cfgs, qref, comb_seen)
    if dead:
        report.append(f"no register for {len(dead)} process-private vars "
                      f"({sum(v.w for v in dead)} bits) not live across a clock edge")
    for pi, proc in enumerate(d.procs):
        cfg = cfgs[pi]
        states = _reachable_states(cfg, qref, proc.name, dead)
        n = len(states)
        idx = {s.id: i for i, s in enumerate(states)}
        sname = f"{proc.name}__state"
        if opts.encoding == "onehot" and n > 1:
            sw = n
            sq = T.var(sname, sw)

            def at(i, sq=sq):
                return T.slice_(sq, i, 1)

            def code(i, sw=sw):
                return T.const(1 << i, sw)
            sinit = 1 << idx[cfg.initial.id]
        elif opts.encoding == "factored" and n > 1:
            # copies of one source wait (a subroutine inlined at several call
            # sites) share a position code; a site field tells them apart:
            # code = {position, site}, so logic that is the same for every
            # copy decodes the position alone
            groups: Dict[str, List[int]] = {}
            for i, s in enumerate(states):
                # the label's source part: '<call path>@<source location>'
                src = s.label.rpartition("@")[2] if s.label else ""
                groups.setdefault(src or f"#{s.id}", []).append(i)
            gl = list(groups.values())
            pw = max(1, (len(gl) - 1).bit_length())
            kw = max(0, (max(len(g) for g in gl) - 1).bit_length())
            sw = pw + kw
            codes = [0] * n
            for p_, g in enumerate(gl):
                for k, i in enumerate(g):
                    codes[i] = (p_ << kw) | k
            sq = T.var(sname, sw)

            def at(i, sq=sq, sw=sw, codes=codes):
                return T.eq(sq, T.const(codes[i], sw))

            def code(i, sw=sw, codes=codes):
                return T.const(codes[i], sw)
            sinit = codes[idx[cfg.initial.id]]
        else:
            sw = max(1, (n - 1).bit_length())
            sq = T.var(sname, sw)

            def at(i, sq=sq, sw=sw):
                return T.eq(sq, T.const(i, sw))

            def code(i, sw=sw):
                return T.const(i, sw)
            sinit = idx[cfg.initial.id]
        written: Set[P.Var] = set()

        def finals(leaves: List[Leaf], stay: T.Term):
            vs: Set[P.Var] = set()
            for lf in leaves:
                vs.update(lf.env.cur)
                vs.update(lf.env.nbw)
            vs -= set(dead)
            final: Dict[P.Var, T.Term] = {}
            for v in vs:
                vals = []
                for lf in leaves:
                    b = lf.env.cur.get(v, qref[v])
                    if v in lf.env.nbw:
                        we, nv = lf.env.nbw[v]
                        b = T.ite(we, nv, b)
                    vals.append((lf.cond, b))
                final[v] = _select(vals)
            nxt = _select([(lf.cond, code(idx[lf.next.id]) if lf.next.kind == "await"
                            else stay) for lf in leaves])
            return vs, final, nxt
        if opts.share_joins:
            leaves = CycleExec(cfg, qref, proc.name, dead).run_multi(
                [(s, at(i) if n > 1 else T.ONE) for i, s in enumerate(states)])
            vs, final, nstate = finals(leaves, sq)
            written |= vs
            if n > 1:
                regs.append(Reg(sname, sw, sinit, nstate,
                                origin=f"state of {proc.name} ({n} states)"))
            for v in vs:
                if final[v] is not qref[v]:
                    writers.setdefault(v, []).append((proc.priority, pi, final[v]))
            report.append(f"{proc.name}: {n} states")
            continue
        per_state: List[Tuple[T.Term, Dict[P.Var, T.Term], T.Term]] = []
        for i, s in enumerate(states):
            leaves = CycleExec(cfg, qref, proc.name, dead).run(s)
            vs, final, nxt = finals(leaves, code(i))
            written |= vs
            per_state.append((at(i), final, nxt))
        # combine states: the last state's logic is the default
        if n > 1:
            nstate = _select([(m, nx) for (m, _f, nx) in per_state])
            regs.append(Reg(sname, sw, sinit, nstate, origin=f"state of {proc.name} ({n} states)"))
        for v in written:
            vals = [(m, f.get(v, qref[v])) for (m, f, _nx) in per_state]
            nv = _select(vals) if n > 1 else vals[0][1]
            if nv is qref[v]:
                continue
            writers.setdefault(v, []).append((proc.priority, pi, nv))
        report.append(f"{proc.name}: {n} states")
    # resolve writes per bit
    for v in d.vars:
        ws = writers.get(v)
        if not ws:
            continue
        if v.kind != "reg":
            raise LowerError(f"a process writes {v.name}, which is a {v.kind}")
        ws.sort(key=lambda x: (-x[0], x[1]))
        if len(ws) == 1:
            regs.append(Reg(v.name, v.w, v.init, ws[0][2], origin=v.origin))
            continue
        bits = []
        contested = []
        for b in range(v.w):
            qb = T.slice_(qref[v], b, 1)
            cand = []
            for prio, pi, nv in ws:
                nb_ = T.slice_(nv, b, 1)
                we = _we_of(nb_, qb)
                if not (we.is_const and not we.val):
                    cand.append((we, nb_, pi))
            if len(cand) > 1:
                contested.append((b, [d.procs[pi].name for _, _, pi in cand]))
            out = qb
            for we, nb_, _pi in reversed(cand):
                out = T.ite(we, nb_, out)
            bits.append(out)
        nv = T.concat(*reversed(bits))
        if contested:
            groups: Dict[tuple, List[int]] = {}
            for b, names in contested:
                groups.setdefault(tuple(names), []).append(b)
            for names, bl in groups.items():
                report.append(f"warning: {v.name} bits {_ranges(bl)} written by "
                              f"{', '.join(names)}; priority goes to {names[0]}")
        regs.append(Reg(v.name, v.w, v.init, nv, origin=v.origin))
    # state no process writes keeps its reset value
    driven = {r.name for r in regs}
    consts = [(v.name, T.const(v.init, v.w)) for v in d.vars
              if v.kind == "reg" and v.name not in driven]
    inputs = [(d.clock.name, 1)] + ([(d.reset.name, 1)] if d.reset else [])
    for v in d.vars:
        if v.kind in ("in",) and v is not d.clock and v is not d.reset:
            inputs.append((v.name, v.w))
    outputs = [(p.name, qref[p.var]) for p in d.ports if p.direction == "out"]
    for v in d.vars:
        if v.name in comb_wires and (v.name in driven or v.kind == "in"):
            raise LowerError(f"{v.name}: written by a combinational process and "
                             f"{'a clocked process' if v.name in driven else 'is an input'}")
    consts = [(n, t) for n, t in consts if n not in comb_wires]
    wires = [(a.var.name, a.value) for a in d.assigns] + list(comb_wires.items()) + consts
    assigned = {n for n, _ in wires}
    nets = [(v.name, v.w) for v in d.vars
            if v.kind == "net" and v.name not in assigned]
    return Netlist(name=d.name, inputs=inputs, outputs=outputs, regs=regs, wires=wires,
                   instances=list(d.instances), clock=d.clock.name,
                   reset=d.reset.name if d.reset else None, report=report, nets=nets)


def _lower_comb(d: P.Design, qref, report: List[str]) -> Dict[str, T.Term]:
    """Combinational processes -> {net name: value}. The body runs once,
    symbolically, from no state; each var it writes must be written on every
    path (a var kept on some path is a latch) and not read before it is
    written (a combinational loop)."""
    out: Dict[str, T.Term] = {}
    for proc in d.procs:
        if not proc.comb:
            continue
        body = [P.Await(None, initial=True, label=f"{proc.name}@comb")] + list(proc.body)
        cfg = CFG(P.Proc(proc.name, body, origin=proc.origin))
        if len(cfg.awaits) != 1:
            raise LowerError(f"combinational process {proc.name} waits")
        leaves = CycleExec(cfg, qref, proc.name).run(cfg.initial)
        leaves = [lf for lf in leaves if not (lf.cond.is_const and not lf.cond.val)]
        vs: Set[P.Var] = set()
        for lf in leaves:
            if lf.env.nbw:
                raise LowerError(f"combinational process {proc.name}: nonblocking write to "
                                 f"{', '.join(sorted(v.name for v in lf.env.nbw))}")
            vs.update(lf.env.cur)
        for v in sorted(vs, key=lambda v: v.name):
            missing = [lf for lf in leaves if v not in lf.env.cur]
            if missing:
                raise LowerError(f"combinational process {proc.name}: {v.name} is not written "
                                 "on every path (it would be a latch)")
            val = _select([(lf.cond, lf.env.cur[v]) for lf in leaves])
            if any(x.name == v.name for x in T.leaves(val)):
                raise LowerError(f"combinational process {proc.name}: {v.name} keeps its "
                                 "value on some path (a latch) or is read before it is "
                                 "written (a combinational loop)")
            if v.name in out:
                raise LowerError(f"{v.name}: written by two combinational processes")
            out[v.name] = val
    if out:
        report.append(f"combinational processes: {sum(1 for p in d.procs if p.comb)}, "
                      f"{len(out)} nets")
    return out


def _node_uses(n: Node, byname: Dict[str, P.Var]) -> Set[P.Var]:
    ts = []
    if n.kind == "assign":
        ts.append(n.value)
    if n.kind in ("branch", "await") and n.cond is not None:
        ts.append(n.cond)
    out: Set[P.Var] = set()
    for t in ts:
        for leaf in T.leaves(t):
            v = byname.get(leaf.name)
            if v is not None:
                out.add(v)
    return out


def _dead_private(d: P.Design, cfgs: List[CFG], qref,
                  observed_names: Set[str] = frozenset()) -> Dict[P.Var, T.Term]:
    """Process-private vars whose value never has to survive a clock edge.

    A var is private when one process alone reads and writes it and nothing
    else observes it (not a port, not read by an instance or a continuous
    assignment). Backward liveness over the process's CFG: if it is live at no
    await, every read of it in a cycle follows a write in the same cycle (or
    is of an undefined value), so it needs no register."""
    byname = {v.name: v for v in d.vars}
    observed: Set[P.Var] = {v for v in d.vars if v.name in observed_names}
    for p in d.ports:
        observed.add(p.var)
    for a in d.assigns:
        observed |= {byname[x.name] for x in T.leaves(a.value) if x.name in byname}
    for inst in d.instances:
        for c in inst.conns:
            if c.var is not None:
                observed.add(c.var)
            if c.term is not None:
                observed |= {byname[x.name] for x in T.leaves(c.term) if x.name in byname}
    refs: Dict[P.Var, Set[int]] = {}
    per_proc_uses: List[Dict[int, Set[P.Var]]] = []
    for pi, cfg in enumerate(cfgs):
        uses: Dict[int, Set[P.Var]] = {}
        for n in cfg.nodes:
            u = _node_uses(n, byname)
            uses[n.id] = u
            for v in u:
                refs.setdefault(v, set()).add(pi)
            if n.kind == "assign":
                refs.setdefault(n.var, set()).add(pi)
        per_proc_uses.append(uses)
    dead: Dict[P.Var, T.Term] = {}
    for pi, cfg in enumerate(cfgs):
        private = {v for v, ps in refs.items()
                   if ps == {pi} and v.kind == "reg" and v not in observed}
        if not private:
            continue
        uses = per_proc_uses[pi]
        live_in: Dict[int, Set[P.Var]] = {n.id: set() for n in cfg.nodes}
        changed = True
        while changed:
            changed = False
            for n in reversed(cfg.nodes):
                succ = [x for x in (n.next, n.alt) if x is not None]
                out: Set[P.Var] = set()
                for x in succ:
                    out |= live_in[x.id]
                if n.kind == "assign":
                    li = (out - {n.var}) | (uses[n.id] & private)
                else:
                    li = out | (uses[n.id] & private)
                if li != live_in[n.id]:
                    live_in[n.id] = li
                    changed = True
        live_at_awaits: Set[P.Var] = set()
        for n in cfg.awaits:
            live_at_awaits |= live_in[n.next.id] | (uses[n.id] & private)
        for v in private - live_at_awaits:
            dead[v] = T.const(0, v.w)
    return dead


def _we_of(t: T.Term, qb: T.Term) -> T.Term:
    """Write enable of one bit: where the next-value term is not the register
    bit itself."""
    if t is qb:
        return T.ZERO
    if t.op == "ite":
        return T.ite(t.args[0], _we_of(t.args[1], qb), _we_of(t.args[2], qb))
    return T.ONE


def _reachable_states(cfg: CFG, qref, name: str, dead=None) -> List[Node]:
    """Awaits reachable from the initial one, in discovery order."""
    seen: Dict[int, Node] = {}
    order: List[Node] = []
    work = [cfg.initial]
    while work:
        s = work.pop(0)
        if s.id in seen:
            continue
        seen[s.id] = s
        order.append(s)
        for lf in CycleExec(cfg, qref, name, dead).run(s):
            if lf.cond.is_const and not lf.cond.val:
                continue
            nx = lf.next
            if nx.kind == "end":
                raise LowerError(f"process {name} can finish; a process must run forever")
            if nx.id not in seen:
                work.append(nx)
    return order


def _ranges(bits: List[int]) -> str:
    bits = sorted(bits)
    out = []
    lo = prev = bits[0]
    for b in bits[1:]:
        if b == prev + 1:
            prev = b
            continue
        out.append(f"{prev}:{lo}" if prev != lo else f"{lo}")
        lo = prev = b
    out.append(f"{prev}:{lo}" if prev != lo else f"{lo}")
    return ",".join(out)


# ============================================================ netlist passes
def optimize(nl: Netlist) -> Netlist:
    """Sequential constants and bit-level cone of influence, to a fixpoint.

    - A register bit whose next value is, on every path, either itself or its
      reset value never changes: it is a constant.
    - A register bit nothing observable depends on (outputs, instance inputs,
      continuous assignments, through any chain of registers) is dead.
    Both become constants at their reset value, which removes the flop.
    - Register bits that are always equal (register correspondence) are
      replaced by one of them; the others are then dead."""
    before = sum(r.w for r in nl.regs)

    def fixpoint():
        for _ in range(64):
            changed = _seq_const(nl)
            if _OPT["reg_corr"]:
                changed |= _reg_corr(nl)
            changed |= _coi(nl)
            if not changed:
                break
    fixpoint()
    if _OPT["seqcorr"]:
        from . import seqcorr
        nl.regs = [r for r in nl.regs if not _is_const_reg(r)]
        nodes = "flops" if _OPT["seqcorr"] == "flops" else "all"
        nl.report.extend(seqcorr.optimize(nl, nodes=nodes))
        fixpoint()
    nl.regs = [r for r in nl.regs if not _is_const_reg(r)]
    _check_defined(nl)
    after_bits = sum(r.w for r in nl.regs)
    nl.report.append(f"optimize: {before} register bits -> {after_bits}")
    return nl


_OPT = {"reg_corr": True, "seqcorr": False}


def _check_defined(nl: Netlist) -> None:
    """Every signal a term reads is a register, an input, a wire or a net
    (a pass that removes a register must leave no reference to it)."""
    known = ({r.name for r in nl.regs} | {n for n, _ in nl.inputs} | {n for n, _ in nl.wires}
             | {n for n, _ in nl.nets})
    terms = [r.next for r in nl.regs] + [t for _, t in nl.wires] + [t for _, t in nl.outputs]
    terms += [c.term for i in nl.instances for c in i.conns if c.term is not None]
    seen: Set[T.Term] = set()
    for t in terms:
        for x in T.leaves(t):
            if x not in seen:
                seen.add(x)
                if x.name not in known:
                    raise LowerError(f"internal: {x.name} is read but no longer defined")


def _reg_corr(nl: Netlist) -> bool:
    """Register correspondence by partition refinement (van Eijk): start from
    classes of register bits with the same reset value; substitute each class
    by its representative in every next-state term and split classes whose
    members' next-state bits differ; repeat until stable. What is left is an
    inductive invariant (equal after reset, and equal next whenever equal
    now), so each bit can be replaced by its representative. Next-state terms
    are compared structurally (hash-consed), which is sound and cheap."""
    regs = [r for r in nl.regs if not _is_const_reg(r)]
    outs = {t.name for _, t in nl.outputs if t.op == "var"}
    # representatives: registers that are outputs first (they stay anyway)
    regs.sort(key=lambda r: r.name not in outs)
    # bits already constant (next is the reset value) take no part
    bits = []
    for r in regs:
        for b in range(r.w):
            nb = T.slice_(r.next, b, 1)
            if not (nb.is_const and nb.val == (r.init >> b) & 1):
                bits.append((r, b))
    cls: Dict[Tuple[str, int], int] = {(r.name, b): (r.init >> b) & 1 for r, b in bits}
    n_cls = len(set(cls.values()))
    while True:
        rep: Dict[int, Tuple[Reg, int]] = {}
        for r, b in bits:
            rep.setdefault(cls[(r.name, b)], (r, b))
        m = _corr_map(regs, cls, rep)
        memo: Dict[T.Term, T.Term] = {}
        ids: Dict[tuple, int] = {}
        new: Dict[Tuple[str, int], int] = {}
        for r in regs:
            sn = T.substitute(r.next, m, memo) if m else r.next
            for b in range(r.w):
                if (r.name, b) in cls:
                    key = (cls[(r.name, b)], T.slice_(sn, b, 1))
                    new[(r.name, b)] = ids.setdefault(key, len(ids))
        if len(ids) == n_cls:
            break
        cls, n_cls = new, len(ids)
    # merged bits of registers that are not outputs
    gone = [(r, b) for r, b in bits if rep[cls[(r.name, b)]] != (r, b) and r.name not in outs]
    if not gone or not m:
        return False
    _subst_all(nl, m)
    # what was merged is referenced nowhere now: tie it to its reset value
    by = {}
    for r, b in gone:
        by.setdefault(r.name, set()).add(b)
    for r in regs:
        bs = by.get(r.name)
        if bs:
            parts = [T.const((r.init >> b) & 1, 1) if b in bs else T.slice_(r.next, b, 1)
                     for b in range(r.w)]
            r.next = T.concat(*reversed(parts))
    pairs: Dict[Tuple[str, str], List[int]] = {}
    for r, b in gone:
        rr, _rb = rep[cls[(r.name, b)]]
        pairs.setdefault((r.name, rr.name), []).append(b)
    nl.report.append(f"register correspondence: {len(gone)} register bits equal to others: "
                     + ", ".join(f"{a}[{_ranges(bs)}] = {b_}" for (a, b_), bs in pairs.items()))
    return True


def _corr_map(regs, cls, rep) -> Dict[T.Term, T.Term]:
    m: Dict[T.Term, T.Term] = {}
    for r in regs:
        q = T.var(r.name, r.w)
        parts = []
        moved = False
        for b in range(r.w):
            k = cls.get((r.name, b))
            rr, rb = rep[k] if k is not None else (r, b)
            if rr is not r or rb != b:
                moved = True
            parts.append(T.slice_(T.var(rr.name, rr.w), rb, 1))
        if moved:
            m[q] = T.concat(*reversed(parts))
    return m


def _is_const_reg(r: Reg) -> bool:
    return r.next.is_const and r.next.val == r.init


def _subst_all(nl: Netlist, m: Dict[T.Term, T.Term]) -> None:
    memo: Dict[T.Term, T.Term] = {}
    for r in nl.regs:
        r.next = T.substitute(r.next, m, memo)
    # an output that is a register stays the register (it keeps its constant
    # bits itself); anything else is rewritten
    nl.outputs = [(n, t if t.op == "var" else T.substitute(t, m, memo)) for n, t in nl.outputs]
    nl.wires = [(n, T.substitute(t, m, memo)) for n, t in nl.wires]
    for inst in nl.instances:
        for c in inst.conns:
            if c.term is not None:
                c.term = T.substitute(c.term, m, memo)


def _seq_const(nl: Netlist) -> bool:
    m: Dict[T.Term, T.Term] = {}
    for r in nl.regs:
        if _is_const_reg(r):
            continue
        q = T.var(r.name, r.w)
        bits = []
        any_const = False
        for i in range(r.w):
            qb = T.slice_(q, i, 1)
            ib = (r.init >> i) & 1
            nb = T.slice_(r.next, i, 1)
            if _only_self_or(nb, qb, ib):
                bits.append(T.const(ib, 1))
                any_const = True
            else:
                bits.append(qb)
        if any_const:
            m[q] = T.concat(*reversed(bits))
    if not m:
        return False
    _subst_all(nl, m)
    for r in nl.regs:
        if T.var(r.name, r.w) in m:
            val = m[T.var(r.name, r.w)]
            # constant bits of the register keep their reset value
            r.next = T.concat(*[
                T.const((r.init >> i) & 1, 1) if T.slice_(val, i, 1).is_const
                else T.slice_(r.next, i, 1) for i in reversed(range(r.w))])
    return True


def _only_self_or(t: T.Term, qb: T.Term, ib: int) -> bool:
    """Every leaf of the if-then-else tree *t* is the register bit or the
    constant *ib*."""
    stack = [t]
    while stack:
        x = stack.pop()
        if x is qb:
            continue
        if x.is_const:
            if x.val != ib:
                return False
            continue
        if x.op == "ite":
            stack.append(x.args[1])
            stack.append(x.args[2])
            continue
        if x.op in ("and", "or") and x.w == 1:
            # 1-bit and/or of the bit with conditions: ib=0 & and-form keeps 0, etc.
            if x.op == "and" and ib == 0 and any(a is qb for a in x.args):
                continue
            if x.op == "or" and ib == 1 and any(a is qb for a in x.args):
                continue
        return False
    return True


def _coi(nl: Netlist) -> bool:
    regs = {r.name: r for r in nl.regs if not _is_const_reg(r)}
    need: Dict[str, int] = {}          # reg name -> needed bit mask
    work: List[Tuple[str, int]] = []

    def mark_term(t: T.Term, bitmask: Optional[int] = None) -> None:
        """Mark the register bits *t* (or the given bits of it) depends on."""
        stack = [(t, (1 << t.w) - 1 if bitmask is None else bitmask)]
        seen = set()
        while stack:
            x, bm = stack.pop()
            if bm == 0 or (x, bm) in seen:
                continue
            seen.add((x, bm))
            op = x.op
            if op == "const":
                continue
            if op == "var":
                if x.name in regs:
                    old = need.get(x.name, 0)
                    new = old | bm
                    if new != old:
                        need[x.name] = new
                        work.append((x.name, new & ~old))
                continue
            if op == "slice":
                stack.append((x.args[0], bm << x.val))
            elif op == "concat":
                pos = x.w
                for a in x.args:
                    pos -= a.w
                    stack.append((a, (bm >> pos) & ((1 << a.w) - 1)))
            elif op in ("and", "or", "xor", "not"):
                for a in x.args:
                    stack.append((a, bm))
            elif op == "ite":
                stack.append((x.args[0], 1))
                stack.append((x.args[1], bm))
                stack.append((x.args[2], bm))
            elif op in ("add", "sub"):
                hi = bm.bit_length()
                low = (1 << hi) - 1
                for a in x.args:
                    stack.append((a, low))
            else:
                for a in x.args:
                    stack.append((a, (1 << a.w) - 1))

    for _n, t in nl.outputs:
        mark_term(t)
    for _n, t in nl.wires:
        mark_term(t)
    for inst in nl.instances:
        for c in inst.conns:
            if c.var is not None and c.var.name in regs:
                mark_term(T.var(c.var.name, c.var.w))
            if c.term is not None:
                mark_term(c.term)
    while work:
        name, bm = work.pop()
        r = regs[name]
        mark_term(r.next, bm)
    changed = False
    for name, r in regs.items():
        nb = need.get(name, 0)
        full = (1 << r.w) - 1
        if nb == full:
            continue
        # dead bits hold their reset value
        parts = []
        for i in reversed(range(r.w)):
            if (nb >> i) & 1:
                parts.append(T.slice_(r.next, i, 1))
            else:
                parts.append(T.const((r.init >> i) & 1, 1))
        nn = T.concat(*parts)
        if nn is not r.next:
            r.next = nn
            changed = True
    return changed
