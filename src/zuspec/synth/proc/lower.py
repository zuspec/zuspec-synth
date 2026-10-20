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
        env0 = Env(cur={}, nbw={})
        leaves: List[Leaf] = []
        # pending contributions per (node, passed): list of (cond, env)
        pending: Dict[tuple, List[Tuple[T.Term, Env]]] = {}
        order: List[tuple] = []

        if start.kind != "await":
            raise LowerError("a state must be an await")
        if start.cond is None:
            first = [((start.next.id, frozenset()), start.next, T.ONE, env0)]
        else:
            c = self._eval(start.cond, env0)
            leaves.append(Leaf(T.not_(c), start, env0))
            first = [((start.next.id, frozenset([start.id])), start.next, c, env0)]
        # topological processing of the (node, passed) DAG
        graph_nodes: Dict[tuple, Node] = {}
        for key, node, c, env in first:
            self._add(pending, order, graph_nodes, key, node, c, env)
        topo = self._toposort(first, start)
        for key in topo:
            contribs = pending.get(key)
            if not contribs:
                continue
            node = graph_nodes[key]
            c, env = self._merge(contribs)
            if c.is_const and not c.val:
                continue
            passed = key[1]
            self._step(node, passed, c, env, start, leaves, pending, order, graph_nodes)
        return leaves

    # -- DAG order -------------------------------------------------------
    def _succ(self, node: Node, passed: frozenset, start: Node):
        """Successor (node, passed) keys within the cycle."""
        k = node.kind
        if k == "assign" or k == "nop":
            return [(node.next, passed)]
        if k == "branch":
            return [(node.next, passed), (node.alt, passed)]
        if k == "await":
            if node.cond is None or node is start or node.id in passed:
                return []
            return [(node.next, passed | {node.id})]
        return []

    def _toposort(self, first, start: Node) -> List[tuple]:
        order: List[tuple] = []
        state: Dict[tuple, int] = {}       # 1 = on stack, 2 = done
        nodes: Dict[tuple, Node] = {}

        def visit(node: Node, passed: frozenset):
            key = (node.id, passed)
            st = state.get(key)
            if st == 2:
                return
            if st == 1:
                raise LowerError(f"process {self.proc_name}: a loop that never waits "
                                 f"for the clock (through node {node.id} {node.label})")
            state[key] = 1
            nodes[key] = node
            for nn, pp in self._succ(node, passed, start):
                visit(nn, pp)
            state[key] = 2
            order.append(key)

        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 100000))
        try:
            for (key, node, _c, _e) in first:
                visit(node, key[1])
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
    def _step(self, node: Node, passed, c, env: Env, start, leaves, pending, order,
              graph_nodes) -> None:
        k = node.kind
        if k == "nop":
            self._add(pending, order, graph_nodes, (node.next.id, passed), node.next, c, env)
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
            self._add(pending, order, graph_nodes, (node.next.id, passed), node.next, c, env2)
        elif k == "branch":
            b = T.bool_(self._eval(node.cond, env))
            self._add(pending, order, graph_nodes, (node.next.id, passed), node.next,
                      T.and_(c, b), env)
            self._add(pending, order, graph_nodes, (node.alt.id, passed), node.alt,
                      T.and_(c, T.not_(b)), env)
        elif k == "await":
            if node.cond is None or node is start or node.id in passed:
                leaves.append(Leaf(c, node, env))
                return
            w = T.bool_(self._eval(node.cond, env))
            leaves.append(Leaf(T.and_(c, T.not_(w)), node, env))
            nxt = (node.next.id, passed | {node.id})
            self._add(pending, order, graph_nodes, nxt, node.next, T.and_(c, w), env)
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
    """ite chain over disjoint conditions; the last value is the default."""
    out = vals[-1][1]
    for c, v in reversed(vals[:-1]):
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
    encoding: str = "binary"         # 'binary' | 'onehot'


def lower(d: P.Design, opts: Optional[LowerOptions] = None) -> Netlist:
    opts = opts or LowerOptions()
    qref: Dict[P.Var, T.Term] = {v: T.var(v.name, v.w) for v in d.vars}
    report: List[str] = []
    regs: List[Reg] = []
    # per var: [(proc priority, proc index, next term)]
    writers: Dict[P.Var, List[Tuple[int, int, T.Term]]] = {}
    cfgs = [CFG(proc) for proc in d.procs]
    dead = _dead_private(d, cfgs, qref)
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
        else:
            sw = max(1, (n - 1).bit_length())
            sq = T.var(sname, sw)

            def at(i, sq=sq, sw=sw):
                return T.eq(sq, T.const(i, sw))

            def code(i, sw=sw):
                return T.const(i, sw)
            sinit = idx[cfg.initial.id]
        per_state: List[Tuple[T.Term, Dict[P.Var, T.Term], T.Term]] = []
        written: Set[P.Var] = set()
        for i, s in enumerate(states):
            leaves = CycleExec(cfg, qref, proc.name, dead).run(s)
            final: Dict[P.Var, T.Term] = {}
            vs: Set[P.Var] = set()
            for lf in leaves:
                vs.update(lf.env.cur)
                vs.update(lf.env.nbw)
            vs -= set(dead)
            written |= vs
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
                            else code(i)) for lf in leaves])
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
    wires = [(a.var.name, a.value) for a in d.assigns] + consts
    assigned = {n for n, _ in wires}
    nets = [(v.name, v.w) for v in d.vars
            if v.kind == "net" and v.name not in assigned]
    return Netlist(name=d.name, inputs=inputs, outputs=outputs, regs=regs, wires=wires,
                   instances=list(d.instances), clock=d.clock.name,
                   reset=d.reset.name if d.reset else None, report=report, nets=nets)


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


def _dead_private(d: P.Design, cfgs: List[CFG], qref) -> Dict[P.Var, T.Term]:
    """Process-private vars whose value never has to survive a clock edge.

    A var is private when one process alone reads and writes it and nothing
    else observes it (not a port, not read by an instance or a continuous
    assignment). Backward liveness over the process's CFG: if it is live at no
    await, every read of it in a cycle follows a write in the same cycle (or
    is of an undefined value), so it needs no register."""
    byname = {v.name: v for v in d.vars}
    observed: Set[P.Var] = set()
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
    Both become constants at their reset value, which removes the flop."""
    before = sum(r.w for r in nl.regs)
    for _ in range(64):
        changed = _seq_const(nl)
        changed |= _coi(nl)
        if not changed:
            break
    nl.regs = [r for r in nl.regs if not _is_const_reg(r)]
    after_bits = sum(r.w for r in nl.regs)
    nl.report.append(f"optimize: {before} register bits -> {after_bits}")
    return nl


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
