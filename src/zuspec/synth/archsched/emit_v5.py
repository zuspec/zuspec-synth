"""V5 (synthesizable SV) folded datapath from an architecture schedule.

Input: a ``sched.Schedule`` of a specialized XLS IR function (node -> step,
S-box node -> pool instance). Output: one Verilog module with a step counter,
exactly the bound S-box instances, value registers and per-step input muxes.

Construction (``fold``):

* **Step classes (rerolling).** Each step's nodes (IR order) get a structural
  signature: op, type, attributes, and operand references that are either
  local (an earlier node of the same step), a table or array-index literal
  (inlined) or a *leaf* numbered by first use (a register, a port, a value
  precomputed from the key, or another literal). Steps with equal
  signatures form a class and share one copy of the logic; the class's leaf
  inputs are muxed by the step counter. In AES this rerolls the rounds
  (A1: steps 1..9 are one class, step 10 the last round).
* **S-box binding.** S-box nodes are taken out of the class logic: each
  bound instance (``pool[inst]`` from the schedule; a class uses its first
  member's binding for all members) is one ``<top>__sbox`` instance whose
  input is muxed by step over the classes that use it.
* **Register allocation.** A value defined in step ``d`` and read in a later
  step ``u`` lives in a register over steps ``d+1..u``. Registers are shared
  by value lifetime (left-edge, per width), preferring the register that the
  same class output used before, so loop-carried values of rerolled rounds
  land in the same register and their muxes collapse.
* **Key values not in any step** (``key_expansion: precomputed``) are a
  combinational ``<top>__keyexp`` module from a key register loaded with the
  block; its S-boxes are extra ``<top>__sbox`` instances (reported).
* **Handshake** (XLS function-pipeline ports): ``input_valid`` starts an op
  when idle (step 0 = the cycle with ``input_valid``; its values are
  registered at that edge); ``output_valid`` is high during step
  ``latency-1``, with ``out`` combinational from that step. Input changes
  while busy are ignored.

``simulate`` runs the same structure step by step in Python (used by the
unit tests against ``xls_ir.evaluate``); ``emit_sv`` writes the module.
"""
from __future__ import annotations

import dataclasses as dc
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import xls_ir
from .sched import Schedule

# ---------------------------------------------------------------- types


def _dims(t: str) -> Tuple[int, List[int]]:
    """``bits[8][4][4]`` -> (8, [4, 4]) (outer dimension last)."""
    m = re.fullmatch(r"bits\[(\d+)\]((?:\[\d+\])*)", t.replace(" ", ""))
    if not m:
        raise ValueError(f"emit_v5: unsupported type {t!r}")
    return int(m.group(1)), [int(d) for d in re.findall(r"\[(\d+)\]", m.group(2))]


def flat_width(t: str) -> int:
    w, ds = _dims(t)
    for d in ds:
        w *= d
    return w


def _index_slice(t: str, idx: Sequence[int]) -> Tuple[int, int]:
    """(offset, width) of ``array_index(x, idx)`` in the flattened value
    (element 0 at the LSB, as XLS flattens arrays)."""
    w, ds = _dims(t)
    off = 0
    for i in idx:
        d = ds.pop()
        ew = flat_width(f"bits[{w}]" + "".join(f"[{x}]" for x in ds))
        off += min(i, d - 1) * ew
    return off, flat_width(f"bits[{w}]" + "".join(f"[{x}]" for x in ds))


def flatten(v, t: str) -> int:
    """Nested-list value of type ``t`` -> int (element 0 at the LSB)."""
    w, ds = _dims(t)
    if not ds:
        return int(v)
    sub = f"bits[{w}]" + "".join(f"[{x}]" for x in ds[:-1])
    ew = flat_width(sub)
    r = 0
    for i, e in enumerate(v):
        r |= flatten(e, sub) << (i * ew)
    return r


def _mask(w: int) -> int:
    return (1 << w) - 1


# ---------------------------------------------------------------- structure

#: a source of a value at some step: ("reg", r) | ("port", param) |
#: ("pre", node) | ("lit", node) | ("cls", class, pos) | ("sbox", inst)
Src = Tuple


@dc.dataclass
class SClass:
    idx: int
    members: List[int]                   # steps
    rep: List[str]                       # representative node names (step members[0])
    nodes: Dict[int, List[str]]          # step -> node names (aligned with rep)
    leaves: Dict[int, List[str]]         # step -> leaf node names (aligned)
    exports: List[int]                   # positions read after their step
    sbox_inst: Dict[int, Tuple[str, int]]  # position -> (pool, inst)


@dc.dataclass
class Folded:
    s: Schedule
    latency: int
    classes: List[SClass]
    step_class: Dict[int, int]
    regs: List[int]                      # register widths
    reg_of: Dict[str, int]               # value -> register
    reg_writes: Dict[int, Dict[int, Src]]  # reg -> step -> source
    leaf_src: Dict[int, Dict[int, List[Src]]]  # class -> step -> source per leaf
    sbox_in: Dict[Tuple[str, int], Dict[int, Tuple[int, int]]]  # inst -> step -> (class, pos)
    pre_nodes: List[str]                 # precomputed (key) nodes, IR order
    pre_sboxes: List[str]
    out_src: Src
    lifetimes: Dict[str, Tuple[int, int]]

    @property
    def sbox_insts(self) -> List[Tuple[str, int]]:
        return sorted(self.sbox_in)

    @property
    def reg_bits(self) -> int:
        return sum(self.regs)

    def summary(self) -> dict:
        return {
            "latency": self.latency, "classes": len(self.classes),
            "class_members": [len(c.members) for c in self.classes],
            "logic_nodes_unshared": sum(len(c.rep) * len(c.members) for c in self.classes),
            "logic_nodes_emitted": sum(len(c.rep) for c in self.classes),
            "sbox_instances": len(self.sbox_in), "key_sbox_instances": len(self.pre_sboxes),
            "registers": len(self.regs), "register_bits": self.reg_bits,
            "values_registered": len(self.reg_of),
            "value_bits_registered": sum(flat_width(self.s.fn.by_name()[v].type)
                                         for v in self.reg_of),
            "leaf_muxes": sum(1 for c in self.classes for k in range(len(next(iter(
                self.leaf_src[c.idx].values()))))
                if len({tuple(self.leaf_src[c.idx][st][k]) for st in c.members}) > 1)
            if self.classes else 0,
        }


def _local_ops(n: xls_ir.Node) -> List[Tuple[str, str]]:
    """(role, operand): role 'idx' for array indices, else 'op'."""
    out = [("op", o) for o in n.operands]
    out += [("idx", o) for o in n.attrs.get("indices", [])]
    out += [("op", o) for o in n.attrs.get("cases", [])]
    if isinstance(n.attrs.get("default"), str):
        out.append(("op", n.attrs["default"]))
    return out


_SKIP_ATTRS = {"indices", "cases", "default", "assumed_in_bounds", "value"}


def _bind_by_level(classes: List["SClass"], nodes, s: Schedule) -> None:
    """Rebind S-box positions so that chained levels use disjoint instances.

    With several S-box levels per step (partial unroll), the schedule's
    binding may put level 1 of one step and level 2 of another on the same
    instance; through the step muxes that is a structural combinational
    loop (instance i -> logic -> instance j -> logic -> instance i). Here
    level ``l`` of every class uses instance block ``l`` of its pool (in the
    order of the schedule's instances). Kept as scheduled when the blocks
    would not fit the pool, or with one level per step.
    """
    lvl: Dict[Tuple[int, int], int] = {}
    need: Dict[Tuple[str, int], int] = {}
    for c in classes:
        loc = {x: j for j, x in enumerate(c.rep)}
        lv: Dict[int, int] = {}
        for j, x in enumerate(c.rep):
            l = max((lv[loc[o]] for o in xls_ir.all_operands(nodes[x]) if o in loc), default=0)
            lv[j] = l + (1 if j in c.sbox_inst else 0)
        cnt: Dict[Tuple[str, int], int] = {}
        for j, inst in c.sbox_inst.items():
            lvl[(c.idx, j)] = lv[j]
            key = (inst[0], lv[j])
            cnt[key] = cnt.get(key, 0) + 1
        for key, n in cnt.items():
            need[key] = max(need.get(key, 0), n)
    if max((l for _, l in need), default=1) <= 1:
        return
    for pool, size in s.pools.items():
        levels = sorted(l for p, l in need if p == pool)
        if sum(need[(pool, l)] for l in levels) > size:
            return
    base: Dict[Tuple[str, int], int] = {}
    for pool in s.pools:
        b = 0
        for l in sorted(l for p, l in need if p == pool):
            base[(pool, l)] = b
            b += need[(pool, l)]
    for c in classes:
        by: Dict[Tuple[str, int], List[int]] = {}
        for j, inst in c.sbox_inst.items():
            by.setdefault((inst[0], lvl[(c.idx, j)]), []).append(j)
        new = {}
        for key, js in by.items():
            for r, j in enumerate(sorted(js, key=lambda j: c.sbox_inst[j][1])):
                new[j] = (key[0], base[key] + r)
        c.sbox_inst = new


def fold(s: Schedule) -> Folded:
    f = s.fn
    nodes = f.by_name()
    pl = s.placed
    L = s.latency
    order = {n.name: i for i, n in enumerate(f.nodes)}
    consumers: Dict[str, List[str]] = {n.name: [] for n in f.nodes}
    for n in f.nodes:
        for o in set(xls_ir.all_operands(n)):
            consumers[o].append(n.name)

    def is_table(x: str) -> bool:
        return nodes[x].op == "literal" and isinstance(nodes[x].value, list)

    per_step: Dict[int, List[str]] = {}
    pre: List[str] = []
    for n in f.nodes:
        st = pl[n.name].step
        if n.op == "literal":
            continue
        if n.op == "param":
            continue
        if st < 0:
            pre.append(n.name)
        else:
            per_step.setdefault(st, []).append(n.name)

    # Signatures.
    sigs: Dict[int, tuple] = {}
    leaves: Dict[int, List[str]] = {}
    for st in range(L):
        ns = per_step.get(st, [])
        loc = {x: i for i, x in enumerate(ns)}
        lv: Dict[str, int] = {}
        sig = []
        for x in ns:
            n = nodes[x]
            refs = []
            for role, o in _local_ops(n):
                if o in loc:
                    refs.append(("n", loc[o]))
                elif is_table(o) or (nodes[o].op == "literal" and role == "idx"):
                    refs.append(("k", o if is_table(o) else nodes[o].value))
                else:
                    if o not in lv:
                        lv[o] = len(lv)
                    refs.append(("l", lv[o]))
            attrs = tuple(sorted((k, str(v)) for k, v in n.attrs.items() if k not in _SKIP_ATTRS))
            sig.append((n.op, n.type, n.kind, attrs, tuple(refs)))
        sigs[st] = tuple(sig)
        leaves[st] = list(lv)

    groups: Dict[tuple, List[int]] = {}
    for st in range(L):
        groups.setdefault(sigs[st], []).append(st)
    classes: List[SClass] = []
    step_class: Dict[int, int] = {}
    for sig, members in sorted(groups.items(), key=lambda kv: kv[1][0]):
        k = len(classes)
        rep = per_step.get(members[0], [])
        c = SClass(k, members, rep, {m: per_step.get(m, []) for m in members},
                   {m: leaves[m] for m in members}, [], {})
        exp = set()
        for m in members:
            for j, x in enumerate(c.nodes[m]):
                if x == f.ret or any(pl[u].step > m for u in consumers[x]):
                    exp.add(j)
        c.exports = sorted(exp)
        for j, x in enumerate(rep):
            if nodes[x].kind == "sbox" or pl[x].pool:
                c.sbox_inst[j] = (pl[x].pool or "sbox", pl[x].inst)
        classes.append(c)
        for m in members:
            step_class[m] = k
    _bind_by_level(classes, nodes, s)
    for c in classes:
        assert len(set(c.sbox_inst.values())) == len(c.sbox_inst), \
            f"class {c.idx}: two S-box lookups on one instance"

    # Values needing a register: defined at step d >= 0, read at a later step.
    life: Dict[str, Tuple[int, int]] = {}
    key_of: Dict[str, tuple] = {}
    for n in f.nodes:
        st = pl[n.name].step
        if st < 0 or n.op == "literal":
            continue
        last = max((pl[u].step for u in consumers[n.name]), default=st)
        if n.name == f.ret:
            last = max(last, L - 1)
        if last > st:
            life[n.name] = (st + 1, last)
            if n.op == "param":
                key_of[n.name] = ("param", n.name)
            else:
                c = classes[step_class[st]]
                key_of[n.name] = (c.idx, c.nodes[st].index(n.name))
    regs: List[int] = []
    reg_free_at: List[int] = []          # first step the register is free again
    reg_of: Dict[str, int] = {}
    pref: Dict[tuple, int] = {}
    for v in sorted(life, key=lambda v: (life[v][0], order[v])):
        a, b = life[v]
        w = flat_width(nodes[v].type)
        r = pref.get(key_of[v])
        if r is None or regs[r] != w or reg_free_at[r] > a:
            cand = [i for i in range(len(regs)) if regs[i] == w and reg_free_at[i] <= a]
            # Prefer a register no class output has claimed yet.
            claimed = set(pref.values())
            free = [i for i in cand if i not in claimed] or cand
            r = free[0] if free else None
        if r is None:
            r = len(regs)
            regs.append(w)
            reg_free_at.append(0)
        reg_free_at[r] = b + 1
        reg_of[v] = r
        pref.setdefault(key_of[v], r)

    reg_writes: Dict[int, Dict[int, Src]] = {}
    for v, r in reg_of.items():
        d = life[v][0] - 1
        if nodes[v].op == "param":
            src = ("port", v)
        else:
            c = classes[step_class[d]]
            src = ("cls", c.idx, c.nodes[d].index(v))
        assert d not in reg_writes.setdefault(r, {}), f"register {r} written twice in step {d}"
        reg_writes[r][d] = src

    def src_at(x: str, st: int) -> Src:
        n = nodes[x]
        if n.op == "literal":
            return ("lit", x)
        if n.op == "param":
            return ("port", x) if st == 0 else ("reg", reg_of[x])
        if pl[x].step < 0:
            return ("pre", x)
        if pl[x].step == st:
            c = classes[step_class[st]]
            return ("cls", c.idx, c.nodes[st].index(x))
        return ("reg", reg_of[x])

    leaf_src = {c.idx: {m: [src_at(x, m) for x in c.leaves[m]] for m in c.members}
                for c in classes}
    sbox_in: Dict[Tuple[str, int], Dict[int, Tuple[int, int]]] = {}
    for c in classes:
        for j, inst in c.sbox_inst.items():
            for m in c.members:
                sbox_in.setdefault(inst, {})[m] = (c.idx, j)
    pre_sb = [x for x in pre if nodes[x].kind == "sbox"]
    return Folded(s, L, classes, step_class, regs, reg_of, reg_writes, leaf_src, sbox_in,
                  pre, pre_sb, src_at(f.ret, L - 1), life)


# ---------------------------------------------------------------- semantics

def _eval_op(n: xls_ir.Node, vals: List[int], nodes: Dict[str, xls_ir.Node]) -> int:
    op = n.op
    w = flat_width(n.type)
    ops = xls_ir.all_operands(n)
    if op == "bit_slice":
        return (vals[0] >> n.attrs["start"]) & _mask(n.attrs["width"])
    if op in ("concat", "array"):
        r = 0
        seq = list(zip(n.operands, vals))
        if op == "array":
            seq = seq[::-1]
        for o, v in seq:
            r = (r << flat_width(nodes[o].type)) | v
        return r
    if op in ("xor", "and", "or", "nor", "nand"):
        r = vals[0]
        for x in vals[1:]:
            r = (r ^ x) if op == "xor" else (r & x) if op in ("and", "nand") else (r | x)
        return (~r & _mask(w)) if op in ("nor", "nand") else r
    if op == "not":
        return ~vals[0] & _mask(w)
    if op in ("identity", "zero_ext"):
        return vals[0]
    if op == "sign_ext":
        iw = flat_width(nodes[ops[0]].type)
        x = vals[0]
        return x | (_mask(w) & ~_mask(iw)) if (x >> (iw - 1)) & 1 else x
    if op == "eq":
        return int(vals[0] == vals[1])
    if op == "ne":
        return int(vals[0] != vals[1])
    if op == "array_index":
        src = nodes[ops[0]]
        if src.op == "literal" and isinstance(src.value, list) and not _dims(src.type)[1][:-1]:
            return src.value[min(vals[1], len(src.value) - 1)]
        idx = vals[1:]
        off, ew = _index_slice(src.type, idx)
        return (vals[0] >> off) & _mask(ew)
    if op == "sel":
        cases = vals[1:1 + len(n.attrs["cases"])]
        return cases[vals[0]] if vals[0] < len(cases) else vals[-1]
    raise ValueError(f"emit_v5: unsupported op {op} ({n.name})")


def _lit(n: xls_ir.Node) -> int:
    return flatten(n.value, n.type)


def simulate(fd: Folded, inputs: Dict[str, int]) -> int:
    """Run the folded structure (registers, classes, S-box muxes) for one op;
    ``inputs``: flattened param values. Returns the flattened result."""
    f = fd.s.fn
    nodes = f.by_name()
    regs = [0] * len(fd.regs)
    pre: Dict[str, int] = {}

    def lit_or_port(src: Src) -> Optional[int]:
        if src[0] == "lit":
            return _lit(nodes[src[1]])
        if src[0] == "port":
            return inputs[src[1]]
        if src[0] == "pre":
            return pre[src[1]]
        return None

    for x in fd.pre_nodes:        # key expansion, from the (registered) key
        n = nodes[x]
        vals = []
        for o in xls_ir.all_operands(n):
            on = nodes[o]
            vals.append(_lit(on) if on.op == "literal" and not isinstance(on.value, list)
                        else inputs[o] if on.op == "param" else pre.get(o, 0))
        pre[x] = _eval_op(n, vals, nodes)
    out = None
    for st in range(fd.latency):
        c = fd.classes[fd.step_class[st]]
        leafv = {}
        for x, src in zip(c.leaves[st], fd.leaf_src[c.idx][st]):
            v = lit_or_port(src)
            leafv[x] = regs[src[1]] if src[0] == "reg" else v
        # Evaluate with the representative's structure, step's own names.
        cur: Dict[str, int] = {}
        names = c.nodes[st]
        for j, x in enumerate(names):
            n = nodes[x]
            vals = []
            for o in xls_ir.all_operands(n):
                on = nodes[o]
                if o in cur:
                    vals.append(cur[o])
                elif on.op == "literal" and isinstance(on.value, list):
                    vals.append(0)        # table operand: handled by _eval_op
                elif on.op == "literal" and o not in leafv:
                    vals.append(_lit(on))
                else:
                    vals.append(leafv[o])
            cur[x] = _eval_op(n, vals, nodes)
        new = list(regs)
        for r, w in fd.reg_writes.items():
            if st in w:
                src = w[st]
                new[r] = inputs[src[1]] if src[0] == "port" else cur[c.nodes[st][src[2]]]
        if st == fd.latency - 1:
            o = fd.out_src
            out = cur[c.nodes[st][o[2]]] if o[0] == "cls" else regs[o[1]] if o[0] == "reg" \
                else lit_or_port(o)
        regs = new
    return out


# ---------------------------------------------------------------- SV


def _sbox_table(f: xls_ir.Function) -> List[int]:
    nodes = f.by_name()
    for n in f.nodes:
        if n.kind == "sbox":
            return nodes[n.operands[0]].value
    raise ValueError("emit_v5: no S-box lookup in the function")


def _expr(n: xls_ir.Node, ref, nodes) -> str:
    """SV expression of ``n``; ``ref(o)`` gives an operand's SV name."""
    op = n.op
    ops = xls_ir.all_operands(n)
    w = flat_width(n.type)
    if op == "bit_slice":
        return f"{ref(ops[0])}[{n.attrs['start'] + n.attrs['width'] - 1}:{n.attrs['start']}]"
    if op == "concat":
        return "{" + ", ".join(ref(o) for o in n.operands) + "}"
    if op == "array":
        return "{" + ", ".join(ref(o) for o in n.operands[::-1]) + "}"
    if op in ("xor", "and", "or"):
        sym = {"xor": "^", "and": "&", "or": "|"}[op]
        return f" {sym} ".join(ref(o) for o in n.operands)
    if op in ("nor", "nand"):
        sym = {"nor": "|", "nand": "&"}[op]
        return "~(" + f" {sym} ".join(ref(o) for o in n.operands) + ")"
    if op == "not":
        return f"~{ref(ops[0])}"
    if op in ("identity",):
        return ref(ops[0])
    if op == "zero_ext":
        iw = flat_width(nodes[ops[0]].type)
        return f"{{{w - iw}'d0, {ref(ops[0])}}}" if w > iw else ref(ops[0])
    if op == "sign_ext":
        iw = flat_width(nodes[ops[0]].type)
        if w == iw:
            return ref(ops[0])
        msb = ref(ops[0]) if iw == 1 else f"{ref(ops[0])}[{iw - 1}]"
        return f"{{{{{w - iw}{{{msb}}}}}, {ref(ops[0])}}}"
    if op == "eq":
        return f"{ref(ops[0])} == {ref(ops[1])}"
    if op == "ne":
        return f"{ref(ops[0])} != {ref(ops[1])}"
    if op == "array_index":
        idx = []
        for i in n.attrs["indices"]:
            if nodes[i].op != "literal":
                raise ValueError(f"emit_v5: variable index {n.name} into a non-table array")
            idx.append(nodes[i].value)
        off, ew = _index_slice(nodes[ops[0]].type, idx)
        return f"{ref(ops[0])}[{off + ew - 1}:{off}]"
    raise ValueError(f"emit_v5: unsupported op {op} ({n.name})")


def _case(lines: List[str], sel: str, target: str, by_step: Dict[int, str], w: int,
          ind: str = "    "):
    """``target`` = by_step[cnt] (combinational); default = the most common."""
    groups: Dict[str, List[int]] = {}
    for st, e in sorted(by_step.items()):
        groups.setdefault(e, []).append(st)
    if len(groups) == 1:
        lines.append(f"{ind}assign {target} = {next(iter(groups))};")
        return
    dflt = max(groups, key=lambda e: (len(groups[e]), -min(groups[e])))
    lines.append(f"{ind}always @* begin")
    lines.append(f"{ind}    case ({sel})")
    for e, sts in groups.items():
        if e == dflt:
            continue
        lines.append(f"{ind}    {', '.join(str(x) for x in sts)}: {target} = {e};")
    lines.append(f"{ind}    default: {target} = {dflt};")
    lines.append(f"{ind}    endcase")
    lines.append(f"{ind}end")


def emit_sv(fd: Folded, top: str, orig: Optional[xls_ir.Function] = None,
            task: str = "zuspec.synth.ArchLower") -> str:
    s = fd.s
    f = s.fn
    nodes = f.by_name()
    sp = s.spec
    L = fd.latency
    cw = max(1, (L - 1).bit_length())
    params = (orig or f).params
    pw = {p.name: flat_width(p.type) for p in params}
    ow = flat_width(nodes[f.ret].type)
    sm = fd.summary()
    lines = [
        f"// Generated by {task} ({sp.arch}: {sp.fold}"
        + (f" x{sp.unroll}" if sp.unroll else "")
        + f", {sp.sbox_count} {sp.sbox_kind} S-box(es), key expansion {sp.key_expansion}). "
        "Do not edit.",
        f"// {f.sv_path}: latency {L} clk (step 0 = load), II {s.ii}.",
        f"// {len(fd.classes)} step classes for {L} steps (logic nodes {sm['logic_nodes_emitted']}"
        f" emitted / {sm['logic_nodes_unshared']} unfolded); {len(fd.sbox_in)} S-box instance(s)"
        f" in the datapath, {len(fd.pre_sboxes)} in key expansion;"
        f" {len(fd.regs)} registers ({fd.reg_bits} bits) for {sm['values_registered']} values"
        f" ({sm['value_bits_registered']} bits).",
    ]
    for c in fd.classes:
        lines.append(f"//   class {c.idx}: steps {_ranges(c.members)}: {len(c.rep)} nodes, "
                     f"{len(c.sbox_inst)} S-box, {len(c.exports)} exports")
    tab = _sbox_table(f)
    lines += [
        f"module {top}__sbox(input wire [7:0] a, output reg [7:0] y);",
        "    always @* begin",
        "        case (a)",
    ]
    lines += [f"        8'h{i:02x}: y = 8'h{v:02x};" for i, v in enumerate(tab)]
    lines += ["        endcase", "    end", "endmodule", ""]

    def lit_sv(x: str) -> str:
        n = nodes[x]
        return f"{flat_width(n.type)}'h{_lit(n):x}"

    # Key expansion module (precomputed values).
    pre_used = sorted({src[1] for c in fd.classes for m in c.members
                       for src in fd.leaf_src[c.idx][m] if src[0] == "pre"}, key=fd.pre_nodes.index) \
        if fd.pre_nodes else []
    if fd.pre_nodes:
        kparams = [p for p in f.params if any(p.name in xls_ir.all_operands(nodes[x])
                                              for x in fd.pre_nodes)]
        lines.append(f"// Key values precomputed for the op ({len(fd.pre_nodes)} nodes, "
                     f"{len(fd.pre_sboxes)} S-box lookups), combinational from the key register.")
        hdr = [f"input wire [{pw[p.name] - 1}:0] {p.name}" for p in kparams]
        hdr += [f"output wire [{flat_width(nodes[x].type) - 1}:0] o_{x.replace('.', '_')}"
                for x in pre_used]
        lines.append(f"module {top}__keyexp(" + ", ".join(hdr) + ");")

        def kref(o: str) -> str:
            if nodes[o].op == "literal":
                return lit_sv(o)
            return o if nodes[o].op == "param" else "k_" + o.replace(".", "_")

        for x in fd.pre_nodes:
            n = nodes[x]
            w = flat_width(n.type)
            if n.kind == "sbox":
                lines.append(f"    wire [7:0] k_{x.replace('.', '_')};")
                lines.append(f"    {top}__sbox u_{x.replace('.', '_')}(.a({kref(n.attrs['indices'][0])}), "
                             f".y(k_{x.replace('.', '_')}));")
            else:
                lines.append(f"    wire [{w - 1}:0] k_{x.replace('.', '_')} = {_expr(n, kref, nodes)};")
        for x in pre_used:
            lines.append(f"    assign o_{x.replace('.', '_')} = k_{x.replace('.', '_')};")
        lines += ["endmodule", ""]

    # Top.
    ports = ["input wire clk", "input wire rst", "input wire input_valid"]
    ports += [f"input wire [{pw[p.name] - 1}:0] {p.name}" for p in params]
    ports += ["output wire output_valid", f"output wire [{ow - 1}:0] out"]
    lines.append(f"module {top}(")
    lines.append("    " + ",\n    ".join(ports))
    lines.append(");")
    lines += [
        f"    reg busy;",
        f"    reg [{cw - 1}:0] cnt;",
        f"    wire start = !busy && input_valid;",
        f"    wire en = busy || start;",
        f"    always @(posedge clk) begin",
        f"        if (rst) begin",
        f"            busy <= 1'b0;",
        f"            cnt <= {cw}'d0;",
        f"        end else if (start) begin",
        f"            busy <= 1'b1;",
        f"            cnt <= {cw}'d1;",
        f"        end else if (busy) begin",
        f"            if (cnt == {cw}'d{L - 1}) begin",
        f"                busy <= 1'b0;",
        f"                cnt <= {cw}'d0;",
        f"            end else",
        f"                cnt <= cnt + {cw}'d1;",
        f"        end",
        f"    end",
        f"    assign output_valid = busy && cnt == {cw}'d{L - 1};",
    ]
    for r, w in enumerate(fd.regs):
        lines.append(f"    reg [{w - 1}:0] r{r};")
    if fd.pre_nodes:
        kparams = [p for p in f.params if any(p.name in xls_ir.all_operands(nodes[x])
                                              for x in fd.pre_nodes)]
        conn = []
        for p in kparams:
            lines.append(f"    reg [{pw[p.name] - 1}:0] {p.name}_q;   // key held for the op")
            lines.append(f"    always @(posedge clk) if (start) {p.name}_q <= {p.name};")
            conn.append(f".{p.name}({p.name}_q)")
        for x in pre_used:
            lines.append(f"    wire [{flat_width(nodes[x].type) - 1}:0] pk_{x.replace('.', '_')};")
            conn.append(f".o_{x.replace('.', '_')}(pk_{x.replace('.', '_')})")
        lines.append(f"    {top}__keyexp u_keyexp(" + ", ".join(conn) + ");")

    def src_sv(src: Src) -> str:
        if src[0] == "reg":
            return f"r{src[1]}"
        if src[0] == "port":
            return src[1]
        if src[0] == "pre":
            return "pk_" + src[1].replace(".", "_")
        if src[0] == "lit":
            return lit_sv(src[1])
        if src[0] == "cls":
            return f"c{src[1]}_n{src[2]}"
        raise ValueError(src)

    # S-box instances.
    sb_name = {inst: f"sb{i}" for i, inst in enumerate(fd.sbox_insts)}
    for inst, nm in sb_name.items():
        lines.append(f"    reg [7:0] {nm}_a;  // {inst[0]}[{inst[1]}]")
        lines.append(f"    wire [7:0] {nm}_y;")
        lines.append(f"    {top}__sbox u_{nm}(.a({nm}_a), .y({nm}_y));")

    # Classes.
    for c in fd.classes:
        rep = c.members[0]
        loc = {x: j for j, x in enumerate(c.rep)}
        leaf_ix = {x: k for k, x in enumerate(c.leaves[rep])}
        lines.append(f"    // class {c.idx}: steps {_ranges(c.members)}")
        nl = len(c.leaves[rep])
        for k in range(nl):
            w = flat_width(nodes[c.leaves[rep][k]].type)
            by = {m: src_sv(fd.leaf_src[c.idx][m][k]) for m in c.members}
            if len(set(by.values())) == 1:
                lines.append(f"    wire [{w - 1}:0] c{c.idx}_l{k} = {next(iter(by.values()))};")
            else:
                lines.append(f"    reg [{w - 1}:0] c{c.idx}_l{k};")
                _case(lines, "cnt", f"c{c.idx}_l{k}", by, w)

        def ref(o: str, c=c, loc=loc, leaf_ix=leaf_ix) -> str:
            if o in loc:
                return f"c{c.idx}_n{loc[o]}"
            if o in leaf_ix:
                return f"c{c.idx}_l{leaf_ix[o]}"
            return lit_sv(o)

        for j, x in enumerate(c.rep):
            n = nodes[x]
            w = flat_width(n.type)
            if j in c.sbox_inst:
                lines.append(f"    wire [7:0] c{c.idx}_n{j} = {sb_name[c.sbox_inst[j]]}_y;")
            else:
                lines.append(f"    wire [{w - 1}:0] c{c.idx}_n{j} = {_expr(n, ref, nodes)};")
        # S-box index operands, used by the S-box input muxes.
        for j in c.sbox_inst:
            n = nodes[c.rep[j]]
            lines.append(f"    wire [7:0] c{c.idx}_a{j} = {ref(n.attrs['indices'][0])};")

    for inst, nm in sb_name.items():
        by = {m: f"c{ci}_a{j}" for m, (ci, j) in fd.sbox_in[inst].items()}
        if len(set(by.values())) == 1:
            lines.append(f"    always @* {nm}_a = {next(iter(by.values()))};")
        else:
            _case(lines, "cnt", f"{nm}_a", by, 8)

    # Register writes.
    for r, ws in sorted(fd.reg_writes.items()):
        steps = sorted(ws)
        lines.append(f"    always @(posedge clk) if (en) begin")
        lines.append(f"        case (cnt)")
        for st in steps:
            lines.append(f"        {st}: r{r} <= {src_sv(ws[st])};")
        lines.append(f"        default: ;")
        lines.append(f"        endcase")
        lines.append(f"    end")

    lines.append(f"    assign out = {src_sv(fd.out_src)};")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def summary_extra(fd: Folded) -> dict:
    return {"v5": fd.summary()}


def _ranges(xs: Sequence[int]) -> str:
    out, i = [], 0
    xs = list(xs)
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}" if i == j else f"{xs[i]}..{xs[j]}")
        i = j + 1
    return ",".join(out)
