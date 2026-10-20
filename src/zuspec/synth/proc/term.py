"""Bit-vector terms: the expression language of the process lowering.

A :class:`Term` is an immutable, hash-consed bit-vector expression with an exact
width. Building a term simplifies it on the way (constant folding, identities,
slices pushed through concatenations and if-then-else), so two terms that are
structurally equal are the same object, and a register bit that a process
writes back unchanged simplifies to the register bit itself. That last property
is what lets the lowering find, per bit, which process really writes what.

Operators (``op``):

  const  var  slice(lo)  concat  not  and  or  xor  add  sub  mul
  eq  ult  slt  ite  shl  lshr  ashr  redor  redand  redxor

``concat`` takes its arguments most-significant first. Comparisons give 1 bit.
``var`` names a signal of the netlist: a register's current value, an input,
or a net.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


class Term:
    __slots__ = ("op", "w", "args", "val", "name", "_hash", "__weakref__")

    def __init__(self, op: str, w: int, args: Tuple["Term", ...], val: int, name: str):
        self.op = op
        self.w = w
        self.args = args
        self.val = val
        self.name = name
        self._hash = hash((op, w, args, val, name))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return self is other

    def __repr__(self) -> str:
        return to_str(self)

    @property
    def is_const(self) -> bool:
        return self.op == "const"


_TABLE: Dict[tuple, Term] = {}


def _mk(op: str, w: int, args: Tuple[Term, ...] = (), val: int = 0, name: str = "") -> Term:
    key = (op, w, args, val, name)
    t = _TABLE.get(key)
    if t is None:
        t = Term(op, w, args, val, name)
        _TABLE[key] = t
    return t


def _m(w: int) -> int:
    return (1 << w) - 1


# ---------------------------------------------------------------- leaves
def const(v: int, w: int) -> Term:
    assert w > 0, "zero-width constant"
    return _mk("const", w, (), v & _m(w))


def var(name: str, w: int) -> Term:
    assert w > 0, f"zero-width var {name}"
    return _mk("var", w, (), 0, name)


ONE = const(1, 1)
ZERO = const(0, 1)


def ones(w: int) -> Term:
    return const(_m(w), w)


# ---------------------------------------------------------- structure
def slice_(t: Term, lo: int, w: int) -> Term:
    assert 0 <= lo and lo + w <= t.w and w > 0, f"bad slice [{lo}+:{w}] of {t.w}"
    if lo == 0 and w == t.w:
        return t
    if t.op == "const":
        return const(t.val >> lo, w)
    if t.op == "slice":
        return slice_(t.args[0], t.val + lo, w)
    if t.op == "concat":
        # find the parts covering [lo, lo+w)
        parts = []
        pos = t.w
        for a in t.args:                      # MSB first
            pos -= a.w
            a_lo, a_hi = pos, pos + a.w       # bit range [a_lo, a_hi)
            s_lo, s_hi = max(lo, a_lo), min(lo + w, a_hi)
            if s_lo < s_hi:
                parts.append(slice_(a, s_lo - a_lo, s_hi - s_lo))
        return concat(*parts)
    if t.op == "ite":
        return ite(t.args[0], slice_(t.args[1], lo, w), slice_(t.args[2], lo, w))
    if t.op in ("and", "or", "xor"):
        return _bitwise(t.op, [slice_(a, lo, w) for a in t.args])
    if t.op == "not":
        return not_(slice_(t.args[0], lo, w))
    if t.op in ("add", "sub", "mul") and lo == 0:
        return _arith(t.op, slice_(t.args[0], 0, w), slice_(t.args[1], 0, w))
    return _mk("slice", w, (t,), lo)


def concat(*parts: Term) -> Term:
    flat: List[Term] = []
    for p in parts:
        if p.op == "concat":
            flat.extend(p.args)
        else:
            flat.append(p)
    # merge neighbours: constants, and adjacent slices of the same term
    out: List[Term] = []
    for p in flat:
        if out:
            q = out[-1]
            if q.op == "const" and p.op == "const":
                out[-1] = const((q.val << p.w) | p.val, q.w + p.w)
                continue
            qb, qlo = _as_slice(q)
            pb, plo = _as_slice(p)
            if qb is not None and qb is pb and plo + p.w == qlo:
                out[-1] = slice_(qb, plo, q.w + p.w)
                continue
        out.append(p)
    if len(out) == 1:
        return out[0]
    return _mk("concat", sum(p.w for p in out), tuple(out))


def _as_slice(t: Term):
    if t.op == "slice":
        return t.args[0], t.val
    if t.op == "var":
        return t, 0
    return None, 0


def zext(t: Term, w: int) -> Term:
    if w == t.w:
        return t
    if w < t.w:
        return slice_(t, 0, w)
    return concat(const(0, w - t.w), t)


def sext(t: Term, w: int) -> Term:
    if w <= t.w:
        return slice_(t, 0, w)
    sign = slice_(t, t.w - 1, 1)
    return concat(*([sign] * (w - t.w)), t)


def resize(t: Term, w: int, signed: bool = False) -> Term:
    return sext(t, w) if signed else zext(t, w)


# ---------------------------------------------------------- logic
def not_(t: Term) -> Term:
    if t.op == "const":
        return const(~t.val, t.w)
    if t.op == "not":
        return t.args[0]
    if t.op == "concat":
        return concat(*[not_(a) for a in t.args])
    return _mk("not", t.w, (t,))


def _bitwise(op: str, args: List[Term]) -> Term:
    w = args[0].w
    assert all(a.w == w for a in args), f"{op} width mismatch {[a.w for a in args]}"
    flat: List[Term] = []
    for a in args:
        if a.op == op:
            flat.extend(a.args)
        else:
            flat.append(a)
    cval = {"and": _m(w), "or": 0, "xor": 0}[op]
    rest: List[Term] = []
    for a in flat:
        if a.op == "const":
            cval = (cval & a.val) if op == "and" else (cval | a.val) if op == "or" else (cval ^ a.val)
        else:
            rest.append(a)
    if op == "and" and cval == 0:
        return const(0, w)
    if op == "or" and cval == _m(w):
        return ones(w)
    if op in ("and", "or"):
        # idempotence, and x op ~x
        uniq: List[Term] = []
        seen = set()
        for a in rest:
            if a in seen:
                continue
            seen.add(a)
            uniq.append(a)
        for a in uniq:
            if a.op == "not" and a.args[0] in seen and w == 1:
                return ZERO if op == "and" else ONE
        rest = uniq
        if op == "and" and cval != _m(w):
            rest.append(const(cval, w))
        if op == "or" and cval != 0:
            rest.append(const(cval, w))
    else:  # xor: pairs cancel
        cnt: Dict[Term, int] = {}
        order: List[Term] = []
        for a in rest:
            if a not in cnt:
                order.append(a)
            cnt[a] = cnt.get(a, 0) + 1
        rest = [a for a in order if cnt[a] % 2]
        if cval:
            if not rest:
                return const(cval, w)
            if cval == _m(w) and len(rest) == 1:
                return not_(rest[0])
            rest.append(const(cval, w))
    if not rest:
        return const(cval, w)
    if len(rest) == 1:
        return rest[0]
    rest.sort(key=lambda a: a._hash)
    return _mk(op, w, tuple(rest))


def and_(*args: Term) -> Term:
    return _bitwise("and", list(args))


def or_(*args: Term) -> Term:
    return _bitwise("or", list(args))


def xor(*args: Term) -> Term:
    return _bitwise("xor", list(args))


def ite(c: Term, a: Term, b: Term) -> Term:
    assert c.w == 1, "ite condition must be 1 bit"
    assert a.w == b.w, f"ite width mismatch {a.w} {b.w}"
    if c.op == "const":
        return a if c.val else b
    if a is b:
        return a
    if c.op == "not":
        return ite(c.args[0], b, a)
    if a.w == 1:
        if a is ONE and b is ZERO:
            return c
        if a is ZERO and b is ONE:
            return not_(c)
        if a is ONE:
            return or_(c, b)
        if b is ZERO:
            return and_(c, a)
        if a is ZERO:
            return and_(not_(c), b)
        if b is ONE:
            return or_(not_(c), a)
    # ite(c, ite(c, x, y), z) = ite(c, x, z); ite(c, x, ite(c, y, z)) = ite(c, x, z)
    if a.op == "ite" and a.args[0] is c:
        a = a.args[1]
    if b.op == "ite" and b.args[0] is c:
        b = b.args[2]
    if a is b:
        return a
    # push through concatenations so per-bit identity survives
    if a.op == "concat" or b.op == "concat":
        cuts = sorted(set(_cuts(a)) | set(_cuts(b)))
        if len(cuts) > 2:
            parts = []
            for hi, lo in zip(reversed(cuts[1:]), reversed(cuts[:-1])):
                parts.append(ite(c, slice_(a, lo, hi - lo), slice_(b, lo, hi - lo)))
            return concat(*parts)
    return _mk("ite", a.w, (c, a, b))


def _cuts(t: Term) -> List[int]:
    if t.op != "concat":
        return [0, t.w]
    out = [0]
    pos = 0
    for a in reversed(t.args):
        pos += a.w
        out.append(pos)
    return out


def bool_(t: Term) -> Term:
    """1 if t != 0."""
    if t.w == 1:
        return t
    return redor(t)


def lnot(t: Term) -> Term:
    return not_(bool_(t))


def land(*ts: Term) -> Term:
    return and_(*[bool_(t) for t in ts])


def lor(*ts: Term) -> Term:
    return or_(*[bool_(t) for t in ts])


def redor(t: Term) -> Term:
    if t.w == 1:
        return t
    if t.op == "const":
        return const(int(t.val != 0), 1)
    if t.op == "concat":
        return or_(*[redor(a) for a in t.args])
    return _mk("redor", 1, (t,))


def redand(t: Term) -> Term:
    if t.w == 1:
        return t
    if t.op == "const":
        return const(int(t.val == _m(t.w)), 1)
    if t.op == "concat":
        return and_(*[redand(a) for a in t.args])
    return _mk("redand", 1, (t,))


def redxor(t: Term) -> Term:
    if t.w == 1:
        return t
    if t.op == "const":
        return const(bin(t.val).count("1") & 1, 1)
    if t.op == "concat":
        return xor(*[redxor(a) for a in t.args])
    return _mk("redxor", 1, (t,))


# ---------------------------------------------------------- comparison
def eq(a: Term, b: Term) -> Term:
    assert a.w == b.w, f"eq width mismatch {a.w} {b.w}"
    if a is b:
        return ONE
    if a.op == "const" and b.op == "const":
        return const(int(a.val == b.val), 1)
    if a.op == "const":
        a, b = b, a
    if a.w == 1 and b.op == "const":
        return a if b.val else not_(a)
    if b.op == "const" and a.op == "concat":
        parts = []
        pos = a.w
        for p in a.args:
            pos -= p.w
            parts.append(eq(p, const(b.val >> pos, p.w)))
        return and_(*parts)
    if b.op == "const" and a.op == "ite":
        # eq(ite(c, x, y), k) with constant arms folds
        x, y = a.args[1], a.args[2]
        if x.op == "const" and y.op == "const":
            return ite(a.args[0], const(int(x.val == b.val), 1), const(int(y.val == b.val), 1))
    return _mk("eq", 1, tuple(sorted((a, b), key=lambda t: t._hash)))


def ne(a: Term, b: Term) -> Term:
    return not_(eq(a, b))


def ult(a: Term, b: Term) -> Term:
    assert a.w == b.w
    if a.op == "const" and b.op == "const":
        return const(int(a.val < b.val), 1)
    if a is b:
        return ZERO
    if b.op == "const" and b.val == 0:
        return ZERO
    return _mk("ult", 1, (a, b))


def ule(a: Term, b: Term) -> Term:
    return not_(ult(b, a))


def slt(a: Term, b: Term) -> Term:
    assert a.w == b.w
    if a.op == "const" and b.op == "const":
        return const(int(_sv(a) < _sv(b)), 1)
    if a is b:
        return ZERO
    return _mk("slt", 1, (a, b))


def sle(a: Term, b: Term) -> Term:
    return not_(slt(b, a))


def _sv(t: Term) -> int:
    v = t.val
    return v - (1 << t.w) if (v >> (t.w - 1)) & 1 else v


# ---------------------------------------------------------- arithmetic
def _arith(op: str, a: Term, b: Term) -> Term:
    assert a.w == b.w, f"{op} width mismatch {a.w} {b.w}"
    w = a.w
    if a.op == "const" and b.op == "const":
        v = {"add": a.val + b.val, "sub": a.val - b.val, "mul": a.val * b.val}[op]
        return const(v, w)
    if op == "add":
        if a.op == "const" and a.val == 0:
            return b
        if b.op == "const" and b.val == 0:
            return a
        if b.op == "const" and a.op != "const":
            pass
        args = tuple(sorted((a, b), key=lambda t: t._hash))
        return _mk("add", w, args)
    if op == "sub":
        if b.op == "const" and b.val == 0:
            return a
        if a is b:
            return const(0, w)
        return _mk("sub", w, (a, b))
    if op == "mul":
        for x, y in ((a, b), (b, a)):
            if x.op == "const":
                if x.val == 0:
                    return const(0, w)
                if x.val == 1:
                    return y
        return _mk("mul", w, tuple(sorted((a, b), key=lambda t: t._hash)))
    raise ValueError(op)


def add(a: Term, b: Term) -> Term:
    return _arith("add", a, b)


def sub(a: Term, b: Term) -> Term:
    return _arith("sub", a, b)


def mul(a: Term, b: Term) -> Term:
    return _arith("mul", a, b)


def neg(a: Term) -> Term:
    return sub(const(0, a.w), a)


def _shift(op: str, a: Term, n: Term) -> Term:
    w = a.w
    if n.op == "const":
        k = n.val
        if k == 0:
            return a
        if k >= w:
            if op == "ashr":
                return sext(slice_(a, w - 1, 1), w)
            return const(0, w)
        if op == "shl":
            return concat(slice_(a, 0, w - k), const(0, k))
        if op == "lshr":
            return concat(const(0, k), slice_(a, k, w - k))
        return sext(slice_(a, k, w - k), w)
    if a.op == "const" and n.op == "const":
        pass
    return _mk(op, w, (a, n))


def shl(a: Term, n: Term) -> Term:
    return _shift("shl", a, n)


def lshr(a: Term, n: Term) -> Term:
    return _shift("lshr", a, n)


def ashr(a: Term, n: Term) -> Term:
    return _shift("ashr", a, n)


# ---------------------------------------------------------- utilities
def substitute(t: Term, m: Dict[Term, Term], memo: Optional[Dict[Term, Term]] = None) -> Term:
    """Replace leaves (and any subterm) per *m*, rebuilding with simplification."""
    if memo is None:
        memo = {}
    r = memo.get(t)
    if r is not None:
        return r
    if t in m:
        r = m[t]
    elif not t.args:
        r = t
    else:
        args = [substitute(a, m, memo) for a in t.args]
        r = rebuild(t, args) if any(x is not y for x, y in zip(args, t.args)) else t
    memo[t] = r
    return r


def rebuild(t: Term, args: List[Term]) -> Term:
    op = t.op
    if op == "slice":
        return slice_(args[0], t.val, t.w)
    if op == "concat":
        return concat(*args)
    if op == "not":
        return not_(args[0])
    if op in ("and", "or", "xor"):
        return _bitwise(op, args)
    if op in ("add", "sub", "mul"):
        return _arith(op, args[0], args[1])
    if op == "eq":
        return eq(args[0], args[1])
    if op == "ult":
        return ult(args[0], args[1])
    if op == "slt":
        return slt(args[0], args[1])
    if op == "ite":
        return ite(args[0], args[1], args[2])
    if op in ("shl", "lshr", "ashr"):
        return _shift(op, args[0], args[1])
    if op == "redor":
        return redor(args[0])
    if op == "redand":
        return redand(args[0])
    if op == "redxor":
        return redxor(args[0])
    raise ValueError(op)


def leaves(t: Term, acc=None) -> set:
    if acc is None:
        acc = set()
    seen = set()
    stack = [t]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        if x.op == "var":
            acc.add(x)
        stack.extend(x.args)
    return acc


def to_str(t: Term, depth: int = 6) -> str:
    if t.op == "const":
        return f"{t.w}'h{t.val:x}"
    if t.op == "var":
        return t.name
    if depth == 0:
        return "…"
    a = [to_str(x, depth - 1) for x in t.args]
    if t.op == "slice":
        return f"{a[0]}[{t.val + t.w - 1}:{t.val}]"
    if t.op == "concat":
        return "{" + ", ".join(a) + "}"
    if t.op == "ite":
        return f"({a[0]} ? {a[1]} : {a[2]})"
    if t.op == "not":
        return f"~{a[0]}"
    sym = {"and": "&", "or": "|", "xor": "^", "add": "+", "sub": "-", "mul": "*",
           "eq": "==", "ult": "<", "slt": "<s", "shl": "<<", "lshr": ">>", "ashr": ">>>"}
    if t.op in sym:
        return "(" + f" {sym[t.op]} ".join(a) + ")"
    return f"{t.op}({', '.join(a)})"
