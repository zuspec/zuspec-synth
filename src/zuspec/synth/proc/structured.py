"""Write a lowered netlist with each clocked process in its source's structure.

The flat emitter (:mod:`zuspec.synth.proc.emit`) writes one next-state term
per register. This one writes each clocked process as two blocks:

  - ``always @*``: the next state and the next value of every var the process
    writes, as the process's own code: a ``case`` on the state, and in each
    state the code of one cycle (from its await to the awaits the cycle ends
    at) as nested ``if``/``else``;
  - ``always @(posedge clock)``: the registers, reset or loaded from those
    next values.

The cycle code of a state is a DAG (``CycleExec``'s program points: a CFG node
and the conditional waits already passed). Each branch is written with its
arms running to the branch's immediate post-dominator, and the code after
that join is written once. Code reached from two arms without being their
join is written in both (counted in the report line).

Every var a process writes has a temp (``<var>__n``), assigned the var's
register value first: reads of the var after a blocking write read the temp,
which is the IR's blocking semantics. A var with only nonblocking writes is
read from its register. Process-private vars with no register are block-local
(default 0, as in the flat lowering).

The netlist optimizer's results come in as register substitutions
(:meth:`Netlist.register_values`): a register it removed is declared a
``wire`` with its replacement, and only the bits it kept are flops. Bits that
several processes write take their next value from the flat netlist's per-bit
resolution. Combinational processes, continuous assigns, outputs and
instances are written as in the flat emitter.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Set, Tuple

from . import ir as P
from . import term as T
from .emit import _Emitter, _id, _rng
from .lower import CycleExec, LowerError, Netlist, Node, ProcInfo

_EXIT = "exit"


class _Region:
    """The cycle code of one state, as a DAG with a sink."""

    def __init__(self, ex: CycleExec, start: Node):
        self.ex = ex
        self.byid = {n.id: n for n in ex.cfg.nodes}
        # key -> (kind, payload, successors)
        #   ('assign', Node, [k]) | ('branch', cond, [t, f])
        #   | ('exit', await Node or None (the end: stay), [])
        self.g: Dict[tuple, tuple] = {}
        if start.cond is None:
            self.root = self._key(start.next, frozenset())
        else:
            # a wait whose condition holds continues in the same cycle
            self.root = ("wait", start.id)
            self.g[self.root] = ("branch", start.cond,
                                 [self._key(start.next, frozenset([start.id])),
                                  self._exit(start)])
        self._build()
        self._ipdom()

    def _exit(self, node: Optional[Node]) -> tuple:
        k = (_EXIT, node.id if node is not None else "end")
        self.g.setdefault(k, ("exit", node, []))
        return k

    def _key(self, node: Node, passed: frozenset) -> tuple:
        while node.kind == "nop":          # loop heads, break, continue
            node = node.next
        if node.kind == "end":
            return self._exit(None)
        if node.kind == "await" and (node.cond is None or node.id in passed):
            return self._exit(node)        # the cycle ends here
        return self.ex._key(node, passed)

    def _build(self) -> None:
        todo = [self.root]
        done: Set[tuple] = set()
        while todo:
            k = todo.pop()
            if k in done:
                continue
            done.add(k)
            e = self.g.get(k)
            if e is None:
                node, passed = self.byid[k[0]], k[1]
                if node.kind == "assign":
                    e = ("assign", node, [self._key(node.next, passed)])
                elif node.kind == "branch":
                    e = ("branch", node.cond, [self._key(node.next, passed),
                                               self._key(node.alt, passed)])
                elif node.kind == "await":       # a conditional wait not yet passed
                    e = ("branch", node.cond,
                         [self._key(node.next, passed | {node.id}), self._exit(node)])
                else:
                    raise LowerError(f"bad node {node.kind}")
                self.g[k] = e
            todo.extend(x for x in e[2] if x not in done)

    def _ipdom(self) -> None:
        order: List[tuple] = []
        seen: Set[tuple] = set()
        stack = [(self.root, False)]
        while stack:
            k, done = stack.pop()
            if done:
                order.append(k)
                continue
            if k in seen:
                continue
            seen.add(k)
            stack.append((k, True))
            for s in self.g[k][2]:
                if s not in seen:
                    stack.append((s, False))
        # order: postorder (successors first); number = position, sink = -1
        num = {k: i for i, k in enumerate(order)}
        sink = None
        self.ipdom: Dict[tuple, Optional[tuple]] = {}
        for k in order:
            succ = self.g[k][2]
            if not succ:
                self.ipdom[k] = sink
                continue
            cur = succ[0]
            for s in succ[1:]:
                a, b = cur, s
                while a != b:
                    na = -1 if a is None else num[a]
                    nb = -1 if b is None else num[b]
                    if na > nb:
                        a = self.ipdom[a]
                    else:
                        b = self.ipdom[b]
                cur = a
            self.ipdom[k] = cur
        self.order = order


class _ProcWriter:
    """The always blocks of one clocked process."""

    def __init__(self, sw: "_Structured", pi: ProcInfo):
        self.sw = sw
        self.pi = pi
        self.name = pi.proc.name
        self.lines: List[str] = []
        self.temps: List[Tuple[str, int]] = []     # expression temps
        self.n_temp = 0
        self.emitted = 0
        self.ex = CycleExec(pi.cfg, pi.qref, self.name, pi.dead)
        self._byid = {n.id: n for n in pi.cfg.nodes}
        # vars this process writes, by kind of write
        blocking: Set[P.Var] = set()
        nonblocking: Set[P.Var] = set()
        for n in pi.cfg.nodes:
            if n.kind == "assign" and n.var not in pi.dead:
                (nonblocking if n.nb else blocking).add(n.var)
        mixed = blocking & nonblocking
        if mixed:
            raise LowerError(f"process {self.name}: {sorted(v.name for v in mixed)} have "
                             f"blocking and nonblocking writes (not supported by "
                             f"structured emission)")
        self.written = sorted(blocking | nonblocking, key=lambda v: v.name)
        self.blocking = blocking
        self.temp_of: Dict[str, str] = {}
        for v in self.written:
            if v.kind != "reg":
                raise LowerError(f"process {self.name} writes {v.name}, which is a {v.kind}")
            own = sw.nl.writers.get(v.name, [])
            base = v.name if own == [self.name] or not own else f"{v.name}__{self.name}"
            self.temp_of[v.name] = sw.alloc(base + "__n", v.w)
        self.dead = sorted(pi.dead, key=lambda v: v.name)
        for v in self.dead:
            sw.declare_reg(v.name, v.w)
        if pi.state_var:
            self.state_n = sw.alloc(pi.state_var + "__n", pi.state_w)
        self._encode_values()

    def _encode_values(self) -> None:
        """Value selectors. A var written only by nonblocking writes is not
        read back in the block, so its assignments can be moved: when it is
        loaded with a few distinct values at more places than there are
        values, each place sets a narrow selector instead, and one ``case`` on
        the selector after the process's code picks the value. The decision
        tree then runs on the selector, not on the full width (a wide counter
        reloaded in many states would otherwise get one wide mux per level).
        Only values that read no temp of the block qualify (they mean the
        same after the code as where they were written)."""
        self.sel_of: Dict[str, Tuple[str, Dict[T.Term, int]]] = {}
        # what reads a temp of this block (not a register) can change
        mine = {v.name for v in self.blocking} | {v.name for v in self.dead}
        sites: Dict[P.Var, List[T.Term]] = {}
        for n in self.pi.cfg.nodes:
            if n.kind == "assign" and n.nb and n.var not in self.pi.dead:
                sites.setdefault(n.var, []).append(n.value)
        for v, vals in sites.items():
            distinct = list(dict.fromkeys(vals))
            k = len(distinct)
            sw_ = k.bit_length()
            if len(vals) <= k or sw_ >= v.w:
                continue
            if any(x.name in mine for t in distinct for x in T.leaves(t)):
                continue
            sel = self.sw.alloc(v.name + "__sel", sw_)
            self.sel_of[v.name] = (sel, {t: i + 1 for i, t in enumerate(distinct)})

    # -- names ------------------------------------------------------------
    def rd(self, name: str) -> str:
        t = self.temp_of.get(name)
        if t is not None and self.sw.byname[name] in self.blocking:
            return t
        return _id(name)

    def state_const(self, i: int) -> str:
        return self.sw.state_names[(self.name, i)]

    # -- expressions -----------------------------------------------------
    def expr(self, t: T.Term, pre: List[str], ind: str) -> str:
        counts: Dict[T.Term, int] = {}
        stack = [t]
        while stack:
            x = stack.pop()
            c = counts.get(x, 0)
            counts[x] = c + 1
            if c == 0:
                stack.extend(x.args)
        memo: Dict[T.Term, str] = {}
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 100000))
        try:
            return self._ex(t, counts, memo, pre, ind)
        finally:
            sys.setrecursionlimit(old)

    def _temp(self, t: T.Term, text: str, pre: List[str], ind: str) -> str:
        self.n_temp += 1
        nm = self.sw.alloc(f"{self.name}__t{self.n_temp}", t.w)
        self.temps.append((nm, t.w))
        pre.append(f"{ind}{nm} = {text};")
        return nm

    def _ex(self, t: T.Term, counts, memo, pre, ind) -> str:
        r = memo.get(t)
        if r is not None:
            return r
        op = t.op
        if op == "const":
            return f"{t.w}'h{t.val:x}"
        if op == "var":
            return self.rd(t.name)
        a0 = t.args[0] if t.args else None
        if op == "slice" and a0.op in ("var", "const"):
            if a0.op == "const":
                return f"{t.w}'h{(a0.val >> t.val) & ((1 << t.w) - 1):x}"
            b = self.rd(a0.name)
            if a0.w == 1:
                return b
            if t.w == 1:
                return f"{b}[{t.val}]"
            return f"{b}[{t.val + t.w - 1}:{t.val}]"
        a = [self._ex(x, counts, memo, pre, ind) for x in t.args]
        if op == "slice":
            base = a[0]
            if not _is_name(base):
                base = self._temp(a0, base, pre, ind)
            s = base if a0.w == 1 else (f"{base}[{t.val}]" if t.w == 1
                                        else f"{base}[{t.val + t.w - 1}:{t.val}]")
        elif op == "concat":
            s = "{" + ", ".join(a) + "}"
        elif op == "not":
            s = f"~{a[0]}"
        elif op in ("and", "or", "xor"):
            s = "(" + f" {dict(zip(('and', 'or', 'xor'), '&|^'))[op]} ".join(a) + ")"
        elif op in ("add", "sub", "mul"):
            s = f"({a[0]} {dict(add='+', sub='-', mul='*')[op]} {a[1]})"
        elif op == "eq":
            s = f"({a[0]} == {a[1]})"
        elif op == "ult":
            s = f"({a[0]} < {a[1]})"
        elif op == "slt":
            s = f"($signed({a[0]}) < $signed({a[1]}))"
        elif op == "ite":
            s = f"({a[0]} ? {a[1]} : {a[2]})"
        elif op == "shl":
            s = f"({a[0]} << {a[1]})"
        elif op == "lshr":
            s = f"({a[0]} >> {a[1]})"
        elif op == "ashr":
            # $unsigned(): its argument is self-determined, so the shift
            # stays arithmetic in any context
            s = f"$unsigned($signed({a[0]}) >>> {a[1]})"
        elif op == "redor":
            s = f"(|{a[0]})"
        elif op == "redand":
            s = f"(&{a[0]})"
        elif op == "redxor":
            s = f"(^{a[0]})"
        else:
            raise LowerError(f"structured emission: operator {op}")
        if counts.get(t, 0) > 1 and op not in ("not",):
            s = self._temp(t, s, pre, ind)
        memo[t] = s
        return s

    # -- statements ------------------------------------------------------
    def byid_loc(self, k):
        """The CFG node of a region key (None for a wait at the state's start
        or an exit)."""
        if isinstance(k[0], int):
            return self._byid.get(k[0])
        if k[0] == "wait":
            return self._byid.get(k[1])
        return None

    def region(self, i: int, start: Node, ind: str) -> List[str]:
        rg = _Region(self.ex, start)
        self._liveness(rg)
        items: list = []
        self._walk(rg, rg.root, None, i, items)
        out: List[str] = []
        _print(items, ind, out)
        return out

    def _liveness(self, rg: _Region) -> None:
        """Which private vars are read later in the cycle, per region node. A
        write to a private var nothing reads afterwards is not written (no
        private var survives the cycle, so none is live at an exit)."""
        local = {v.name for v in self.dead}
        uses: Dict[T.Term, Set[str]] = {}

        def used(t: T.Term) -> Set[str]:
            u = uses.get(t)
            if u is None:
                u = {x.name for x in T.leaves(t) if x.name in local}
                uses[t] = u
            return u
        live: Dict[tuple, frozenset] = {}
        for k in rg.order:                  # successors first
            kind, pay, succ = rg.g[k]
            if kind == "exit":
                live[k] = frozenset()
            elif kind == "assign":
                out = live[succ[0]]
                if pay.var.name in local:
                    live[k] = (out - {pay.var.name}) | used(pay.value) \
                        if pay.var.name in out else out
                else:
                    live[k] = out | used(pay.value)
            else:
                live[k] = live[succ[0]] | live[succ[1]] | used(pay)
        self.live = live

    def _walk(self, rg: _Region, k, stop, i: int, out: list) -> None:
        """The region from *k* to *stop* as a statement tree: ('s', text) |
        ('if', cond text, then items, else items, (selector text, width,
        value) when the condition compares a selector with a constant)."""
        while k is not None and k != stop:
            kind, pay, succ = rg.g[k]
            self.emitted += 1
            if kind == "exit":
                if pay is not None and self.pi.state_var:
                    j = next((jj for jj, s in enumerate(self.pi.states) if s is pay), None)
                    if j is None:
                        raise LowerError(f"process {self.name}: exit to an unreachable await")
                    if j != i:
                        out.append(("s", f"{self.state_n} = {self.state_const(j)};", ""))
                return
            if kind == "assign":
                node = pay
                if node.var in self.pi.dead and node.var.name not in self.live[succ[0]]:
                    k = succ[0]                 # a dead store
                    continue
                enc = self.sel_of.get(node.var.name)
                if enc is None:
                    pre: List[str] = []
                    val = self.expr(node.value, pre, "")
                    out.extend(("s", x, node.loc) for x in pre)
                if enc is not None:
                    sel, codes = enc
                    out.append(("s", f"{sel} = {codes[node.value]};", node.loc))
                    k = succ[0]
                    continue
                lhs = (_id(node.var.name) if node.var in self.pi.dead
                       else self.temp_of[node.var.name])
                out.append(("s", f"{lhs} = {val};", node.loc))
                k = succ[0]
                continue
            # branch
            j = rg.ipdom[k]
            pre = []
            cond = T.bool_(pay)
            c = self.expr(cond, pre, "")
            loc = getattr(self.byid_loc(k), "loc", "")
            out.extend(("s", x, loc) for x in pre)
            sel = None
            m = _eq_const(cond)
            if m is not None:
                p2: List[str] = []
                st = self.expr(m[0], p2, "")
                if not p2:
                    sel = (st, m[0].w, m[1])
            then: list = []
            els: list = []
            self._walk(rg, succ[0], j, i, then, )
            self._walk(rg, succ[1], j, i, els)
            if then or els:
                out.append(("if", c, then, els, sel, loc))
            k = j

    def comb_block(self) -> List[str]:
        pi = self.pi
        body: List[str] = []
        if pi.state_var:
            body.append(f"        case ({_id(pi.state_var)})")
            n = len(pi.states)
            for i, s in enumerate(pi.states):
                lbl = _await_note(s)
                item = "default" if i == n - 1 else self.state_const(i)
                if pi.onehot and i != n - 1:
                    item = f"{_id(pi.state_var)}[{i}]"
                code = self.region(i, s, " " * 16)
                note = f"  // {self.state_const(i)}" if i == n - 1 else ""
                body.append(f"            {item}: begin{note}  // {lbl}")
                body.extend(code)
                body.append("            end")
            body.append("        endcase")
            if pi.onehot:
                body[0] = "        case (1'b1)"
        else:
            body.extend(self.region(0, pi.states[0], " " * 8))
        for name, (sel, codes) in self.sel_of.items():
            w = self.sw.byname[name].w
            pre: List[str] = []
            arms = [(c, self.expr(t, pre, "")) for t, c in codes.items()]
            body.extend(f"        {x}" for x in pre)
            body.append(f"        case ({sel})  // the value {name} was given")
            for c, val in arms:
                body.append(f"            {self.sw.width_of(sel)}'d{c}: {self.temp_of[name]} = {val};")
            body.append("            default: ;")
            body.append("        endcase")
        out = [f"    // process {self.name}" + (f" ({pi.proc.origin})" if pi.proc.origin else ""),
               "    always @* begin"]
        if pi.state_var:
            out.append(f"        {self.state_n} = {_id(pi.state_var)};")
        for v in self.written:
            out.append(f"        {self.temp_of[v.name]} = {_id(v.name)};")
        for v in self.dead:
            out.append(f"        {_id(v.name)} = {v.w}'h0;")
        for nm, w in self.temps:
            out.append(f"        {nm} = {w}'h0;")
        for sel, _codes in self.sel_of.values():
            out.append(f"        {sel} = {self.sw.width_of(sel)}'d0;")
        out.extend(body)
        out.append("    end")
        return out


def _await_note(s: Node) -> str:
    """Where a state's await is: its source line, and the call path it was
    reached through."""
    path = s.label.rpartition("@")[0] if "@" in s.label else s.label
    parts = [x for x in (s.loc, path) if x]
    return ", ".join(parts) if parts else f"await {s.id}"


def _eq_const(t: T.Term):
    """(selector, value) when *t* compares a term with a constant."""
    if t.op == "eq":
        a, b = t.args
        if b.op == "const" and a.op != "const":
            return a, b.val
        if a.op == "const" and b.op != "const":
            return b, a.val
    return None


def _print(items: list, ind: str, out: List[str], last: Optional[list] = None) -> None:
    """Print a statement tree. A line whose source line differs from the
    previous one's gets a ``// file:line`` comment."""
    last = last if last is not None else [""]

    def src(loc: str) -> str:
        if not loc or loc == last[0]:
            return ""
        last[0] = loc
        return f"  // {loc}"
    for it in items:
        if it[0] == "s":
            out.append(ind + it[1] + src(it[2]))
            continue
        _, c, then, els, sel, loc = it
        # a chain of compares of one selector with constants: a case
        arms = []
        cur = it
        while cur is not None and cur[4] is not None and cur[2] \
                and (not arms or cur[4][0] == arms[0][0][0]):
            arms.append((cur[4], cur[2]))
            nxt = cur[3]
            if len(nxt) == 1 and nxt[0][0] == "if":
                cur = nxt[0]
            else:
                cur = None
                rest = nxt
                break
        else:
            rest = [cur] if cur is not None else []
        if len(arms) >= 3:
            st, w, _v = arms[0][0]
            out.append(f"{ind}case ({st}){src(loc)}")
            seen = set()
            for (_s, _w, v), body in arms:
                if v in seen:
                    continue                    # the first arm wins, as in the if chain
                seen.add(v)
                out.append(f"{ind}    {w}'h{v:x}: begin")
                _print(body, ind + "        ", out, last)
                out.append(f"{ind}    end")
            if rest:
                out.append(f"{ind}    default: begin")
                _print(rest, ind + "        ", out, last)
                out.append(f"{ind}    end")
            out.append(f"{ind}endcase")
            continue
        if not then:
            out.append(f"{ind}if (!{_paren(c)}) begin{src(loc)}")
            _print(els, ind + "    ", out, last)
            out.append(f"{ind}end")
            continue
        out.append(f"{ind}if ({_strip(c)}) begin{src(loc)}")
        _print(then, ind + "    ", out, last)
        while len(els) == 1 and els[0][0] == "if" and els[0][2]:
            e = els[0]
            out.append(f"{ind}end else if ({_strip(e[1])}) begin{src(e[5])}")
            _print(e[2], ind + "    ", out, last)
            els = e[3]
        if els:
            out.append(f"{ind}end else begin")
            _print(els, ind + "    ", out, last)
        out.append(f"{ind}end")


def _is_name(s: str) -> bool:
    return all(c.isalnum() or c == "_" for c in s) and not s[0].isdigit()


def _strip(c: str) -> str:
    return c[1:-1] if c.startswith("(") and c.endswith(")") and _balanced(c[1:-1]) else c


def _paren(c: str) -> str:
    return c if _is_name(c) or (c.startswith("(") and _balanced(c[1:-1])) else f"({c})"


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
            if d < 0:
                return False
    return d == 0


class _Structured:
    def __init__(self, nl: Netlist):
        self.nl = nl
        self.used: Set[str] = set()
        self.decls: List[str] = []
        self.byname: Dict[str, P.Var] = {}
        for pi in nl.procs:
            for v in pi.qref:
                self.byname[v.name] = v
        self.used |= {_id(n) for n in self.byname}
        self.used |= {_id(n) for n, _ in nl.inputs} | {_id(n) for n, _ in nl.outputs}
        self.used |= {_id(n) for n, _ in nl.wires} | {_id(n) for n, _ in nl.nets}
        self.used |= set(nl.reg_shapes)
        self.state_names: Dict[Tuple[str, int], str] = {}
        self.widths: Dict[str, int] = {}

    def alloc(self, base: str, w: int, declare: bool = True) -> str:
        nm = _id(base)
        k = 1
        while nm in self.used:
            k += 1
            nm = _id(f"{base}{k}")
        self.used.add(nm)
        if declare:
            self.declare_reg(nm, w)
        return nm

    def width_of(self, nm: str) -> int:
        return self.widths[nm]

    def declare_reg(self, nm: str, w: int) -> None:
        self.widths[_id(nm)] = w
        self.decls.append(f"    reg {_rng(w)}{_id(nm)};")


def emit_structured(nl: Netlist) -> str:
    if not nl.procs and nl.regs:
        raise LowerError("structured emission needs the lowering's process info")
    s = _Structured(nl)
    e = _Emitter(nl)
    vals = nl.register_values()
    # state constants
    params: List[str] = []
    for pi in nl.procs:
        if not pi.state_var:
            continue
        for i, st in enumerate(pi.states):
            nm = _id(f"{pi.proc.name}__S{i}")
            s.used.add(nm)
            s.state_names[(pi.proc.name, i)] = nm
            params.append(f"    localparam {_rng(pi.state_w)}{nm} = {pi.state_w}'h{pi.codes[i]:x};"
                          f"    // {_await_note(st)}")
    writers = [_ProcWriter(s, pi) for pi in nl.procs]
    by_proc = {w.name: w for w in writers}
    blocks: List[str] = []
    for w in writers:
        blocks.extend(w.comb_block())
    # registers
    clk = _id(nl.clock)
    rst = _id(nl.reset) if nl.reset else None
    live = {r.name: r for r in nl.regs}
    reg_decls: List[str] = []
    wire_decls: List[str] = []
    ffs: Dict[str, List[Tuple[str, str, str]]] = {}   # block -> (lhs, init, next)
    inits: List[str] = []
    dup = []
    state_of = {pi.state_var: pi for pi in nl.procs if pi.state_var}
    for name, (w, init) in nl.reg_shapes.items():
        val = vals.get(name)
        q = T.var(name, w)
        if val is None:
            ff, flop = _id(name), list(range(w))
        else:
            # flop bits are the var's own bits in its value
            flop = [b for b in range(w) if T.slice_(val, b, 1) is T.slice_(q, b, 1)]
            ff = s.alloc(name + "__ff", w, declare=False) if flop else None
            if ff:
                val = T.substitute(val, {q: T.var(ff, w)})
            wire_decls.append((name, w, val))
        if ff is None:
            continue
        wr = nl.writers.get(name, [])
        if name in state_of:
            pi = state_of[name]
            blk = pi.proc.name
            tmp = by_proc[blk].state_n
        elif len(wr) == 1 and wr[0] in by_proc and name in by_proc[wr[0]].temp_of:
            blk = wr[0]
            tmp = by_proc[blk].temp_of[name]
        else:
            # several writers: the flat netlist's per-bit resolution
            r = live.get(name)
            nxt_t = r.next if r is not None else nl.next_by_source(name)
            if nxt_t is None:
                raise LowerError(f"internal: {name} has flops but no next value")
            blk, tmp = "(several writers)", None
            dup.append(name)
        reg_decls.append(f"    reg {_rng(w)}{ff};")
        for hi, lo in _ranges_of(flop):
            bw = hi - lo + 1
            full = bw == w
            sl = "" if full else (f"[{hi}]" if hi == lo else f"[{hi}:{lo}]")
            iv = f"{bw}'h{(init >> lo) & ((1 << bw) - 1):x}"
            nx = (tmp + sl) if tmp is not None else e.ref(T.slice_(nxt_t, lo, bw))
            ffs.setdefault(blk, []).append((f"{ff}{sl}", iv, nx))
            inits.append(f"    initial {ff}{sl} = {iv};")
    out: List[str] = []
    ports = [_id(n) for n, _ in nl.inputs] + [_id(n) for n, _ in nl.outputs]
    out.append(f"// Generated by zuspec.synth.proc (structured) from the process lowering of {nl.name}.")
    for r in nl.report:
        out.append(f"//   {r}")
    out.append(f"module {_id(nl.name)} (")
    out.append(",\n".join(f"    {p}" for p in ports))
    out.append(");")
    for n, w in nl.inputs:
        out.append(f"    input  wire {_rng(w)}{_id(n)};")
    for n, t in nl.outputs:
        out.append(f"    output wire {_rng(t.w)}{_id(n)};")
    out.extend(params)
    out.extend(reg_decls)
    out.extend(s.decls)
    ports_set = {n for n, _ in nl.inputs} | {n for n, _ in nl.outputs}
    # private vars with no register are the processes' own (the flat
    # netlist ties them to a constant, unread)
    local = {v.name for pi in nl.procs for v in pi.dead}
    nl_wires = [(n, t) for n, t in nl.wires if n not in local]
    for n, w, _v in wire_decls:
        out.append(f"    wire {_rng(w)}{_id(n)};")
    for n, t in nl_wires:
        if n in ports_set:
            continue
        out.append(f"    wire {_rng(t.w)}{_id(n)};")
    for n, w in nl.nets:
        if n in ports_set:
            continue
        out.append(f"    wire {_rng(w)}{_id(n)};")
    body: List[str] = []
    for n, w, v in wire_decls:
        body.append(f"    assign {_id(n)} = {e.ref(v)};")
    for n, t in nl_wires:
        body.append(f"    assign {_id(n)} = {e.ref(t)};")
    for n, t in nl.outputs:
        if t.op == "var" and _id(t.name) == _id(n):
            continue
        body.append(f"    assign {_id(n)} = {e.ref(t)};")
    body.extend(blocks)
    for blk, items in ffs.items():
        body.append(f"    always @(posedge {clk}) begin  // registers of {blk}")
        if rst:
            body.append(f"        if ({rst}) begin")
            body.extend(f"            {lhs} <= {iv};" for lhs, iv, _n in items)
            body.append("        end else begin")
            body.extend(f"            {lhs} <= {nx};" for lhs, _iv, nx in items)
            body.append("        end")
        else:
            body.extend(f"        {lhs} <= {nx};" for lhs, _iv, nx in items)
        body.append("    end")
    if rst and inits:
        # as the flat emitter: the initial value is for simulation before the
        # first reset only
        body.append("`ifndef SYNTHESIS")
        body.extend(inits)
        body.append("`endif")
    else:
        body.extend(inits)
    for inst in nl.instances:
        params_ = ""
        if inst.params:
            params_ = " #(" + ", ".join(f".{k}({v})" for k, v in inst.params.items()) + ")"
        conns = []
        for c in inst.conns:
            if c.var is not None:
                conns.append(f".{c.port}({_id(c.var.name)})")
            elif c.term is not None:
                conns.append(f".{c.port}({e.ref(c.term)})")
            else:
                conns.append(f".{c.port}()")
        body.append(f"    {inst.module}{params_} {_id(inst.name)} (" + ", ".join(conns) + ");")
    dupl = sum(w.emitted for w in writers)
    out.append(f"    // structured: {len(writers)} processes, {dupl} region nodes written, "
               f"{len(dup)} registers with several writers")
    out.extend(e.lines)
    out.extend(body)
    out.append("endmodule")
    return "\n".join(out) + "\n"


def _ranges_of(bits: List[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for b in sorted(bits):
        if out and out[-1][0] == b - 1:
            out[-1] = (b, out[-1][1])
        else:
            out.append((b, b))
    return list(reversed(out))
