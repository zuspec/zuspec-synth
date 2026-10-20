"""Process IR: clocked processes over bit-vector state, before FSM lowering.

A :class:`Design` is what a front end hands the lowering: a single-clock
netlist with

  - ``vars``: every signal the processes touch. ``reg`` is state a process
    writes (the lowering makes it a register, reset to ``init``); ``in`` is a
    module input; ``net`` is driven by something else in the netlist (a leaf
    instance output, a continuous assignment);
  - ``procs``: clocked processes, each a structured statement body that awaits
    clock edges and conditions, and combinational processes (``comb``: no
    awaits; what they write are nets);
  - ``assigns``: continuous assignments (``net = term``);
  - ``instances``: leaf modules kept as they are, connected to vars.

Semantics (the contract the lowering implements):

  - A process runs from its ``initial`` await. Code between awaits runs in one
    clock cycle. ``Await(None)`` waits for the next clock edge;
    ``Await(cond)`` continues in the same cycle if ``cond`` holds, otherwise it
    waits, checking ``cond`` every cycle.
  - Reading a var gives the process's current view: its own blocking writes in
    this cycle are visible at once; nonblocking writes and other processes'
    writes are visible from the next cycle.
  - A bit that several processes write in the same cycle goes to the process
    with the highest ``priority`` (ties: the lower index in ``procs``).
  - Reset (``Design.reset``, active high, synchronous) puts every process in
    its initial await and every reg at ``init``.
"""
from __future__ import annotations

import dataclasses as dc
from typing import Dict, List, Optional, Union

from .term import Term


@dc.dataclass(eq=False)
class Var:
    name: str
    w: int
    kind: str = "reg"            # 'reg' | 'in' | 'out' | 'net'
    init: int = 0
    origin: str = ""             # where it came from, for reports

    def __repr__(self) -> str:
        return f"Var({self.name}:{self.w})"


@dc.dataclass(eq=False)
class Assign:
    var: Var
    value: Term                  # full width of var
    nb: bool = False             # nonblocking
    loc: str = ""                # source 'file:line', for the output's comments


@dc.dataclass(eq=False)
class If:
    cond: Term
    then: List["Stmt"] = dc.field(default_factory=list)
    orelse: List["Stmt"] = dc.field(default_factory=list)
    loc: str = ""


@dc.dataclass(eq=False)
class Loop:
    """``while (pre) { body }`` / ``do { body } while (post)`` / forever."""
    body: List["Stmt"] = dc.field(default_factory=list)
    pre: Optional[Term] = None
    post: Optional[Term] = None


@dc.dataclass(eq=False)
class Break:
    pass


@dc.dataclass(eq=False)
class Continue:
    pass


@dc.dataclass(eq=False)
class Await:
    cond: Optional[Term] = None  # None: the next clock edge
    initial: bool = False
    label: str = ""
    loc: str = ""


Stmt = Union[Assign, If, Loop, Break, Continue, Await]


@dc.dataclass(eq=False)
class Proc:
    name: str
    body: List[Stmt]
    priority: int = 0
    origin: str = ""
    # combinational: no awaits; the body runs whenever what it reads changes,
    # and every var it writes is a net with the body's final value. Each such
    # var must be written on every path before it is read (else: a latch)
    comb: bool = False


@dc.dataclass(eq=False)
class Conn:
    port: str
    var: Optional[Var] = None    # connected var (whole)
    term: Optional[Term] = None  # or an input expression
    direction: str = "in"


@dc.dataclass(eq=False)
class Instance:
    module: str
    name: str
    params: Dict[str, object] = dc.field(default_factory=dict)
    conns: List[Conn] = dc.field(default_factory=list)


@dc.dataclass(eq=False)
class Port:
    name: str
    w: int
    direction: str               # 'in' | 'out'
    var: Var


@dc.dataclass(eq=False)
class Design:
    name: str
    clock: Var
    reset: Optional[Var]
    ports: List[Port] = dc.field(default_factory=list)
    vars: List[Var] = dc.field(default_factory=list)
    procs: List[Proc] = dc.field(default_factory=list)
    assigns: List[Assign] = dc.field(default_factory=list)
    instances: List[Instance] = dc.field(default_factory=list)

    def var(self, name: str) -> Var:
        for v in self.vars:
            if v.name == name:
                return v
        raise KeyError(name)


def dump(d: Design) -> str:
    """A readable listing of a design (for debugging and golden tests)."""
    out: List[str] = [f"design {d.name}  clock={d.clock.name} reset={d.reset.name if d.reset else None}"]
    for v in d.vars:
        out.append(f"  {v.kind:4} {v.name}[{v.w}] = {v.init:#x}    // {v.origin}")
    for a in d.assigns:
        out.append(f"  assign {a.var.name} = {a.value!r}")
    for i in d.instances:
        cs = ", ".join(f".{c.port}({c.var.name if c.var else c.term!r})" for c in i.conns)
        out.append(f"  {i.module} {i.name}({cs})")
    for p in d.procs:
        out.append(f"  proc {p.name} (priority {p.priority})")
        _dump_stmts(p.body, out, 4)
    return "\n".join(out)


def _dump_stmts(stmts: List[Stmt], out: List[str], ind: int) -> None:
    sp = " " * ind
    for s in stmts:
        if isinstance(s, Assign):
            out.append(f"{sp}{s.var.name} {'<=' if s.nb else '='} {s.value!r}")
        elif isinstance(s, If):
            out.append(f"{sp}if {s.cond!r}")
            _dump_stmts(s.then, out, ind + 2)
            if s.orelse:
                out.append(f"{sp}else")
                _dump_stmts(s.orelse, out, ind + 2)
        elif isinstance(s, Loop):
            out.append(f"{sp}loop pre={s.pre!r} post={s.post!r}")
            _dump_stmts(s.body, out, ind + 2)
        elif isinstance(s, Await):
            what = "edge" if s.cond is None else f"until {s.cond!r}"
            out.append(f"{sp}await {what}{' INITIAL' if s.initial else ''}  {s.label}")
        elif isinstance(s, Break):
            out.append(f"{sp}break")
        elif isinstance(s, Continue):
            out.append(f"{sp}continue")
