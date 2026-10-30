"""A small reader for XLS text IR (``*.opt.ir``) as a dataflow graph.

Only what the architecture scheduler needs: the top function's nodes, their
operands and attributes, in the file's (topological) order. ``specialize``
binds parameters to constants and folds what that makes constant (selects
on a constant selector, and/or/xor with identity or absorbing constants,
indexing through ``array`` nodes), then removes nodes the return value no
longer needs. ``classify`` marks each node with a resource kind: S-box
lookups (indexing a 256-entry byte table that is a non-linear permutation),
constant GF multiplies (indexing a linear byte table), logic, wiring.
"""
from __future__ import annotations

import dataclasses as dc
import re
from typing import Dict, List, Optional


@dc.dataclass
class Node:
    name: str
    type: str
    op: str
    operands: List[str]
    attrs: Dict[str, object]
    is_ret: bool = False
    #: constant value once known (int, or nested lists for arrays)
    value: object = None
    kind: str = ""            # sbox | gfmul | logic | mux | wire | literal | param

    @property
    def width(self) -> int:
        m = re.fullmatch(r"bits\[(\d+)\]", self.type.strip())
        return int(m.group(1)) if m else 0

    @property
    def is_const(self) -> bool:
        return self.value is not None


@dc.dataclass
class Function:
    name: str
    params: List[Node]
    nodes: List[Node]          # params first, then body in topological order
    ret: str

    def by_name(self) -> Dict[str, Node]:
        return {n.name: n for n in self.nodes}

    @property
    def sv_path(self) -> str:
        """``aes_pkg__encrypt`` -> ``aes_pkg::encrypt``."""
        return self.name.replace("__", "::", 1)


def _split_top(s: str) -> List[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _parse_value(s: str):
    s = s.strip()
    if s.startswith("["):
        return [_parse_value(x) for x in _split_top(s[1:-1])]
    if ":" in s:                     # bits[8]:5 form
        s = s.split(":", 1)[1]
    return int(s, 0)


_NODE = re.compile(r"^\s*(ret\s+)?([A-Za-z_][\w.]*)\s*:\s*(.*?)\s*=\s*([a-z_]+)\((.*)\)\s*$")
_FN = re.compile(r"^\s*(top\s+)?fn\s+(\w+)\((.*)\)\s*->\s*(.*?)\s*\{\s*$")


def parse(text: str, fn: Optional[str] = None) -> Function:
    """The top function (or ``fn``) of an XLS IR package."""
    funcs: Dict[str, Function] = {}
    top = None
    cur: Optional[Function] = None
    for line in text.splitlines():
        m = _FN.match(line)
        if m:
            params = []
            for p in _split_top(m.group(3)):
                pm = re.match(r"(\w+)\s*:\s*(.*?)\s+id=\d+", p)
                params.append(Node(pm.group(1), pm.group(2), "param", [], {}, kind="param"))
            cur = Function(m.group(2), params, list(params), "")
            funcs[cur.name] = cur
            if m.group(1):
                top = cur.name
            continue
        if cur is None:
            continue
        if line.strip() == "}":
            cur = None
            continue
        m = _NODE.match(line)
        if not m:
            continue
        name, typ, op, args = m.group(2), m.group(3), m.group(4), m.group(5)
        operands, attrs = [], {}
        for a in _split_top(args):
            km = re.match(r"^(\w+)=(.*)$", a, re.S)
            if km:
                k, v = km.group(1), km.group(2).strip()
                if k in ("id", "pos"):
                    continue
                if k in ("indices", "cases"):
                    attrs[k] = _split_top(v[1:-1])
                elif k == "value":
                    attrs[k] = _parse_value(v)
                elif re.fullmatch(r"-?\d+", v):
                    attrs[k] = int(v)
                else:
                    attrs[k] = v
            else:
                operands.append(a)
        n = Node(name, typ, op, operands, attrs, is_ret=bool(m.group(1)))
        if op == "literal":
            n.value = attrs["value"]
        cur.nodes.append(n)
        if n.is_ret:
            cur.ret = name
    name = fn or top or next(iter(funcs))
    return funcs[name]


def all_operands(n: Node) -> List[str]:
    """Data operands: positional, then indices/cases/default."""
    out = list(n.operands)
    out += list(n.attrs.get("indices", []))
    out += list(n.attrs.get("cases", []))
    if isinstance(n.attrs.get("default"), str):
        out.append(n.attrs["default"])
    return out


def _mask(w: int) -> int:
    return (1 << w) - 1


def _eval(n: Node, nodes: Dict[str, Node]):
    """Constant value of ``n`` if all its operands are constant, else None."""
    ops = [nodes[o] for o in all_operands(n)]
    if n.op in ("param",) or any(not o.is_const for o in ops):
        return None
    pos = [nodes[o].value for o in n.operands]
    w = n.width
    op = n.op
    if op == "bit_slice":
        return (pos[0] >> n.attrs["start"]) & _mask(n.attrs["width"])
    if op == "concat":
        r = 0
        for o in n.operands:
            r = (r << nodes[o].width) | nodes[o].value
        return r
    if op in ("xor", "and", "or", "nor", "nand"):
        r = pos[0]
        for x in pos[1:]:
            r = (r ^ x) if op == "xor" else (r & x) if op in ("and", "nand") else (r | x)
        if op in ("nor", "nand"):
            r = ~r & _mask(w)
        return r
    if op == "not":
        return ~pos[0] & _mask(w)
    if op == "sign_ext":
        iw = nodes[n.operands[0]].width
        x = pos[0]
        if x >> (iw - 1) & 1:
            x |= _mask(w) & ~_mask(iw)
        return x
    if op == "zero_ext":
        return pos[0]
    if op == "eq":
        return int(pos[0] == pos[1])
    if op == "ne":
        return int(pos[0] != pos[1])
    if op == "sel":
        cases = [nodes[c].value for c in n.attrs["cases"]]
        s = pos[0]
        if s < len(cases):
            return cases[s]
        return nodes[n.attrs["default"]].value
    if op == "array":
        return list(pos)
    if op == "array_index":
        r = pos[0]
        for i in n.attrs["indices"]:
            r = r[min(nodes[i].value, len(r) - 1)]
        return r
    return None


def specialize(f: Function, consts: Dict[str, int]) -> Function:
    """Bind parameters to constants; fold; remove dead nodes."""
    nodes = f.by_name()
    alias: Dict[str, str] = {}

    def res(x: str) -> str:
        while x in alias:
            x = alias[x]
        return x

    for p, v in consts.items():
        if p not in nodes:
            raise KeyError(f"specialize: {f.name} has no parameter {p}")
        nodes[p].value = v
    for n in f.nodes:
        if n.op in ("param", "literal"):
            continue
        n.operands = [res(o) for o in n.operands]
        for k in ("indices", "cases"):
            if k in n.attrs:
                n.attrs[k] = [res(o) for o in n.attrs[k]]
        if isinstance(n.attrs.get("default"), str):
            n.attrs["default"] = res(n.attrs["default"])
        val = _eval(n, nodes)
        if val is not None:
            n.value = val
            continue
        if n.op == "sel" and nodes[n.operands[0]].is_const:
            s = nodes[n.operands[0]].value
            cs = n.attrs["cases"]
            alias[n.name] = cs[s] if s < len(cs) else n.attrs["default"]
            continue
        if n.op in ("and", "or", "xor"):
            w = n.width
            ident = _mask(w) if n.op == "and" else 0
            absorb = 0 if n.op == "and" else _mask(w) if n.op == "or" else None
            if absorb is not None and any(nodes[o].is_const and nodes[o].value == absorb
                                          for o in n.operands):
                n.value = absorb
                continue
            keep = [o for o in n.operands if not (nodes[o].is_const and nodes[o].value == ident)]
            if len(keep) == 1:
                alias[n.name] = keep[0]
                continue
            if keep:
                n.operands = keep
        if n.op == "array_index":
            src = nodes[n.operands[0]]
            idx = n.attrs["indices"]
            if src.op == "array" and idx and nodes[idx[0]].is_const and len(idx) == 1:
                alias[n.name] = src.operands[nodes[idx[0]].value]
                continue
    # Constants become literals; drop what the return value does not need.
    ret = res(f.ret)
    for n in f.nodes:
        if n.is_const and n.op not in ("param", "literal"):
            n.op, n.operands, n.attrs = "literal", [], {"value": n.value}
    live, work = set(), [ret]
    while work:
        x = work.pop()
        if x in live:
            continue
        live.add(x)
        work += all_operands(nodes[x])
    body = [n for n in f.nodes if n.name in live and n.op != "param"]
    params = [p for p in f.params if p.name not in consts]
    for n in f.nodes:
        n.is_ret = (n.name == ret)
    return Function(f.name, params, params + body, ret)


def _is_byte_table(n: Node) -> bool:
    return (n.op == "literal" and isinstance(n.value, list) and len(n.value) == 256
            and all(isinstance(x, int) for x in n.value))


def table_kind(t: List[int]) -> str:
    """``sbox`` for a non-linear byte permutation, ``gfmul`` for a linear
    table (T[a^b] == T[a]^T[b]: a constant GF(2^8) multiply), else ``lut``."""
    linear = all(t[a ^ b] == t[a] ^ t[b] for a in (1, 2, 4, 8, 16, 32, 64, 128)
                 for b in range(256))
    if linear:
        return "gfmul"
    if sorted(t) == list(range(256)):
        return "sbox"
    return "lut"


_WIRE = {"bit_slice", "concat", "array", "sign_ext", "zero_ext", "tuple", "tuple_index",
         "identity"}
_LOGIC = {"xor", "and", "or", "nor", "nand", "not", "eq", "ne"}


def classify(f: Function) -> Dict[str, int]:
    """Set ``kind`` on every node; returns counts per kind."""
    nodes = f.by_name()
    tables: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for n in f.nodes:
        if n.op == "param":
            n.kind = "param"
        elif n.op == "literal":
            n.kind = "literal"
            if _is_byte_table(n):
                tables[n.name] = table_kind(n.value)
        elif n.op == "array_index":
            src = n.operands[0]
            if src in tables:
                n.kind = tables[src]
            elif all(nodes[i].is_const for i in n.attrs.get("indices", [])):
                n.kind = "wire"
            else:
                n.kind = "mux"
        elif n.op in _WIRE:
            n.kind = "wire"
        elif n.op in _LOGIC:
            n.kind = "logic"
        elif n.op in ("sel", "one_hot_sel", "priority_sel"):
            n.kind = "mux"
        else:
            n.kind = "logic"
        counts[n.kind] = counts.get(n.kind, 0) + 1
    return counts


def evaluate(f: Function, inputs: Dict[str, object]):
    """Interpret ``f`` on constant inputs (for checking ``specialize``)."""
    nodes = {n.name: dc.replace(n, value=n.value) for n in f.nodes}
    for k, v in inputs.items():
        nodes[k].value = v
    for n in f.nodes:
        m = nodes[n.name]
        if m.op in ("param", "literal"):
            continue
        m.value = _eval(m, nodes)
        if m.value is None:
            raise ValueError(f"evaluate: cannot evaluate {n.name} ({n.op})")
    return nodes[f.ret].value
