"""An equivalence miter of two netlists with an invariant certificate
(synthesis-plan §0.8.5).

:func:`miter` puts netlist *a* (the debug view) and netlist *b* (the tight
view, *a* after obsmerge) side by side on shared inputs, as one netlist whose
outputs are 1-bit checks:

- ``eq_<port>``: the two views drive the output the same;
- ``kept_<reg>``: the register bits *b* kept equal *a*'s;
- ``inv_<k>``: obsmerge's invariants on *a* (``L -> r[b] == x``).

:func:`sby` writes a wrapper that assumes reset in the first cycle and
asserts every check after it, and a SymbiYosys file for k-induction. The
outputs are what must hold; the other checks are the certificate, the
strengthening that makes the property inductive.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import ir as P
from . import term as T
from .lower import Netlist


def _renamer(nl: Netlist, pre: str):
    local = {r.name for r in nl.regs} | {n for n, _ in nl.wires} | {n for n, _ in nl.nets}
    memo: Dict[T.Term, T.Term] = {}
    m: Dict[T.Term, T.Term] = {}

    def name(n: str) -> str:
        return pre + n if n in local else n

    def term(t: T.Term) -> T.Term:
        for x in T.leaves(t):
            if x.name in local and x not in m:
                m[x] = T.var(pre + x.name, x.w)
        return T.substitute(t, m, memo)
    return name, term


def miter(a: Netlist, b: Netlist, invariants: List[Tuple[str, T.Term]] = (),
          name: str = "miter") -> Netlist:
    """*invariants*: (label, 1-bit term over *a*'s signals)."""
    from .lower import Reg
    an, at = _renamer(a, "a__")
    bn, bt = _renamer(b, "b__")
    regs, wires, nets, insts = [], [], [], []
    for nl, nm, tm in ((a, an, at), (b, bn, bt)):
        regs += [Reg(nm(r.name), r.w, r.init, tm(r.next), r.origin) for r in nl.regs]
        wires += [(nm(n), tm(t)) for n, t in nl.wires]
        nets += [(nm(n), w) for n, w in nl.nets]
        for inst in nl.instances:
            conns = []
            for c in inst.conns:
                v = P.Var(nm(c.var.name), c.var.w, c.var.kind) if c.var is not None else None
                conns.append(P.Conn(c.port, v, tm(c.term) if c.term is not None else None,
                                    c.direction))
            insts.append(P.Instance(inst.module, nm(inst.name), dict(inst.params), conns))
    outs: List[Tuple[str, T.Term]] = []
    bo = dict(b.outputs)
    for n, t in a.outputs:
        outs.append((f"eq_{n}", T.eq(at(t), bt(bo[n]))))
    bregs = {r.name: r for r in b.regs}
    for r in a.regs:
        rb = bregs.get(r.name)
        if rb is None:
            continue
        bits = [k for k in range(r.w) if not T.slice_(rb.next, k, 1).is_const]
        if not bits:
            continue
        qa, qb = T.var("a__" + r.name, r.w), T.var("b__" + r.name, r.w)
        outs.append((f"kept_{r.name}", T.and_(*[
            T.eq(T.slice_(qa, k, 1), T.slice_(qb, k, 1)) for k in bits])))
    for k, (lbl, t) in enumerate(invariants):
        outs.append((f"inv_{k}", at(t)))
    return Netlist(name=name, inputs=list(a.inputs), outputs=outs, regs=regs, wires=wires,
                   instances=insts, clock=a.clock, reset=a.reset, nets=nets,
                   reg_shapes={r.name: (r.w, r.init) for r in regs})


def sby(m: Netlist, verilog: List[str], depth: int = 20, top: str = "miter_top",
        engine: str = "smtbmc boolector", mode: str = "prove") -> Tuple[str, str]:
    """(wrapper Verilog, .sby text). *verilog*: the files to read (the miter
    netlist and its leaf modules), relative to the sby file."""
    ins = [(n, w) for n, w in m.inputs if n != m.clock]
    lines = [f"module {top} (input {m.clock}" + "".join(
        f", input [{w - 1}:0] {n}" if w > 1 else f", input {n}" for n, w in ins) + ");"]
    for n, t in m.outputs:
        lines.append(f"  wire {n};")
    conns = [f".{m.clock}({m.clock})"] + [f".{n}({n})" for n, _ in ins] + \
        [f".{n}({n})" for n, _ in m.outputs]
    lines.append(f"  {m.name} u ({', '.join(conns)});")
    lines.append("  reg init = 1'b1;")
    lines.append(f"  always @(posedge {m.clock}) init <= 1'b0;")
    lines.append("  always @* begin")
    if m.reset:
        lines.append(f"    if (init) assume({m.reset});")
    lines.append("    if (!init) begin")
    for n, _ in m.outputs:
        lines.append(f"      {n}_ok: assert({n});")
    lines.append("    end")
    lines.append("  end")
    lines.append("endmodule")
    files = list(verilog) + [f"{top}.sv"]
    sby_text = "\n".join([
        "[options]", f"mode {mode}", f"depth {depth}", "multiclock off", "",
        "[engines]", engine, "",
        "[script]",
        *[f"read_verilog -formal {f}" for f in files],
        f"prep -flatten -top {top}", "",
        "[files]", *files, ""])
    return "\n".join(lines) + "\n", sby_text
