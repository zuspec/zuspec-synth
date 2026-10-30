"""Resource-constrained list scheduling with binding and loop folding.

Input: an XLS IR function (``xls_ir``), classified, and an ``ArchSpec``.

* Step 0 is the load step: ``data_inputs`` (the block) are registered at
  its end, so nodes that depend on them start at step 1. Other inputs (the
  key, held in a CSR register) are available from step 0.
* Resources: S-box lookups use a pool of ``sbox_count`` instances per step
  (``key_sbox_count`` more for the key schedule with
  ``key_expansion: separate``). Other nodes (XOR, GF constant multiplies,
  wiring) are free; the delay model only reports them.
* Loop folding: the fold sets how many S-box levels one step may chain
  (``ArchSpec.levels_per_step``): one round per clock for iterated, column
  and byte folds, ``unroll`` rounds for a partial unroll. The unrolled IR's
  rounds then fall onto steps, and the S-box pool is reused across them;
  ``rounds`` in the result reports which steps each round (S-box level of
  the data path) took.
* List scheduling: per step, nodes whose operands are done are ready; free
  nodes are placed at once (ASAP, chained), S-box nodes by priority (longest
  remaining S-box chain to the result, then IR order) while instances and
  the chaining budget allow. Repeats until the step is full.
* Binding: an S-box node takes the instance of the nearest S-box ancestor
  if free in its step (keeps a byte lane on one instance), else the lowest
  free instance.
* II: blocks do not overlap in a folded datapath (the state register and the
  S-boxes are reused every step), so II = latency unless the fold is a full
  unroll, where it is the resource-constrained minimum (ResMII) over the
  modulo reservation table. ResMII is reported for every arch.
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from . import xls_ir
from .spec import ArchSpec


@dc.dataclass
class Placed:
    step: int                 # -1: precomputed (key load) or constant
    pool: str = ""            # "sbox" / "key_sbox" for resource nodes
    inst: int = -1


@dc.dataclass
class Schedule:
    fn: xls_ir.Function
    spec: ArchSpec
    placed: Dict[str, Placed]
    latency: int              # clocks from op start to result (steps 0..last)
    ii: int
    res_mii: int
    pools: Dict[str, int]     # pool -> instances available
    used: Dict[str, int]      # pool -> instances actually bound
    usage: Dict[int, Dict[str, int]]          # step -> pool -> lookups
    rounds: List[Tuple[int, int, int]]        # (round, first step, last step)
    step_delay_ps: Dict[int, int]
    sbox_ops: int             # S-box lookups scheduled in the op
    precomputed_sbox_ops: int
    # Set by ``archsched.run``: what the schedule was made from (to_json's guard).
    ir_sha256: str = ""
    consts: Dict[str, int] = dc.field(default_factory=dict)
    data_inputs: List[str] = dc.field(default_factory=list)

    @property
    def sbox_count(self) -> int:
        return sum(self.used.values())

    @property
    def steps_per_round(self) -> Optional[int]:
        spans = {b - a + 1 for _, a, b in self.rounds}
        return spans.pop() if len(spans) == 1 else None


def _data_dependent(f: xls_ir.Function, data_inputs: Sequence[str]) -> Dict[str, bool]:
    dep: Dict[str, bool] = {}
    for n in f.nodes:
        if n.op == "param":
            dep[n.name] = n.name in data_inputs
        else:
            dep[n.name] = any(dep.get(o, False) for o in xls_ir.all_operands(n))
    return dep


def schedule(f: xls_ir.Function, spec: ArchSpec,
             data_inputs: Sequence[str] = ("block_",)) -> Schedule:
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    nodes = f.by_name()
    if not any(n.kind for n in f.nodes):
        xls_ir.classify(f)
    dep = _data_dependent(f, data_inputs)
    ops = {n.name: [o for o in xls_ir.all_operands(n)] for n in f.nodes}
    succs: Dict[str, List[str]] = {n.name: [] for n in f.nodes}
    for n, os_ in ops.items():
        for o in set(os_):
            succs[o].append(n)

    def is_sbox(x: str) -> bool:
        return nodes[x].kind == "sbox"

    pools = {"sbox": spec.sbox_count}
    if spec.key_expansion == "separate":
        pools["key_sbox"] = spec.key_sboxes

    def pool_of(x: str) -> str:
        if spec.key_expansion == "separate" and not dep[x]:
            return "key_sbox"
        return "sbox"

    # Priority: S-box levels on the longest path from the node to the result.
    height: Dict[str, int] = {}
    for n in reversed(f.nodes):
        h = max((height[s] for s in succs[n.name]), default=0)
        height[n.name] = h + (1 if n.kind == "sbox" else 0)
    order = {n.name: i for i, n in enumerate(f.nodes)}

    placed: Dict[str, Placed] = {}
    lvl: Dict[str, int] = {}         # S-box levels chained within its step
    precomputed_sbox = 0
    todo = set()
    for n in f.nodes:
        if n.op == "literal":
            placed[n.name] = Placed(-1)
        elif spec.key_expansion == "precomputed" and not dep[n.name]:
            placed[n.name] = Placed(-1)
            precomputed_sbox += int(n.kind == "sbox")
        elif n.op == "param" and not dep[n.name]:
            placed[n.name] = Placed(0)
            lvl[n.name] = 0
        else:
            todo.add(n.name)
    data_params = [p.name for p in f.params if dep[p.name]]
    for p in data_params:
        todo.discard(p)
    remaining = {x: sum(1 for o in set(ops[x]) if o not in placed) for x in todo}
    ready = {x for x in todo if remaining[x] == 0}
    L = spec.levels_per_step
    usage: Dict[int, Dict[str, int]] = {}
    sbox_anc: Dict[str, Optional[str]] = {}

    def place(x: str, step: int, pool: str = "", inst: int = -1):
        placed[x] = Placed(step, pool, inst)
        todo.discard(x)
        ready.discard(x)
        for s in succs[x]:
            if s in remaining:
                remaining[s] -= 1
                if remaining[s] == 0:
                    ready.add(s)

    def chain(x: str, step: int) -> int:
        return (1 if is_sbox(x) else 0) + max(
            (lvl[o] for o in ops[x] if placed[o].step == step and o in lvl), default=0)

    def nearest_sbox(x: str) -> Optional[str]:
        if x in sbox_anc:
            return sbox_anc[x]
        r = None
        for o in ops[x]:
            if nodes[o].kind == "sbox":
                r = o
                break
        if r is None:
            for o in ops[x]:
                r = nearest_sbox(o)
                if r is not None:
                    break
        sbox_anc[x] = r
        return r

    step = 0
    guard = 0
    while todo:
        if step == 1:
            for p in data_params:
                # The load register: data inputs are used from step 1.
                lvl[p] = 0
                place(p, 0)
        free_inst = {p: set(range(c)) for p, c in pools.items()}
        progress = True
        while progress:
            progress = False
            # Free nodes, ASAP and chained (iterate: placing one may ready more).
            while True:
                fr = [x for x in ready if not is_sbox(x)]
                if not fr:
                    break
                for x in sorted(fr, key=order.get):
                    lvl[x] = chain(x, step)
                    place(x, step)
                    progress = True
            cand = [x for x in ready if is_sbox(x)]
            cand.sort(key=lambda x: (-height[x], order[x]))
            for x in cand:
                p = pool_of(x)
                if not free_inst[p] or chain(x, step) > L:
                    continue
                anc = nearest_sbox(x)
                pref = placed[anc].inst if anc is not None and anc in placed else -1
                inst = pref if pref in free_inst[p] else min(free_inst[p])
                free_inst[p].discard(inst)
                lvl[x] = chain(x, step)
                place(x, step, p, inst)
                usage.setdefault(step, {}).setdefault(p, 0)
                usage[step][p] += 1
                progress = True
        step += 1
        guard += 1
        if guard > 100000:
            raise RuntimeError("schedule: no progress")
    last = max(pl.step for pl in placed.values())
    latency = last + 1

    used = {p: len({pl.inst for pl in placed.values() if pl.pool == p}) for p in pools}

    # ResMII: smallest II whose modulo reservation table fits the pools.
    res_mii = latency
    for ii in range(1, latency + 1):
        ok = True
        for p, c in pools.items():
            for r in range(ii):
                if sum(usage.get(s, {}).get(p, 0) for s in range(r, latency, ii)) > c:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            res_mii = ii
            break
    ii = res_mii if (spec.fold == "unrolled" and spec.levels_per_step == math.inf) else latency

    # Rounds: S-box level along the data path.
    depth: Dict[str, int] = {}
    for n in f.nodes:
        d = max((depth.get(o, 0) for o in ops[n.name] if dep.get(o, False)), default=0)
        depth[n.name] = d + (1 if (n.kind == "sbox" and dep[n.name]) else 0)
    by_round: Dict[int, List[int]] = {}
    for n in f.nodes:
        if n.kind == "sbox" and dep[n.name]:
            by_round.setdefault(depth[n.name], []).append(placed[n.name].step)
    rounds = [(r, min(s), max(s)) for r, s in sorted(by_round.items())]

    # Per-step combinational delay (longest chain of nodes within a step).
    arr: Dict[str, int] = {}
    step_delay: Dict[int, int] = {}
    for n in f.nodes:
        pl = placed[n.name]
        if pl.step < 0 or n.op == "param":
            continue
        d = spec.delay_ps(n.kind)
        a = max((arr.get(o, 0) for o in ops[n.name] if placed[o].step == pl.step
                 and nodes[o].op != "param"), default=0) + d
        arr[n.name] = a
        step_delay[pl.step] = max(step_delay.get(pl.step, 0), a)

    sbox_ops = sum(1 for n in f.nodes if n.kind == "sbox" and placed[n.name].step >= 0)
    return Schedule(f, spec, placed, latency, ii, res_mii, pools, used, usage, rounds,
                    step_delay, sbox_ops, precomputed_sbox)


def report(s: Schedule, full: bool = True) -> str:
    sp = s.spec
    worst = max(s.step_delay_ps.values(), default=0)
    lines = [
        f"# zuspec.synth.ArchSched: {s.fn.sv_path} on {sp.arch} ({sp.path})",
        f"fold:            {sp.fold}" + (f" x{sp.unroll}" if sp.unroll else "")
        + f" (S-box levels per step: {sp.levels_per_step})",
        f"key_expansion:   {sp.key_expansion}",
        f"latency:         {s.latency} clk (steps 0..{s.latency - 1}; step 0 = load)",
        f"II:              {s.ii} clk (ResMII {s.res_mii})",
        f"S-box count:     {s.sbox_count} ({sp.sbox_kind}; "
        + ", ".join(f"{p} {s.used[p]}/{c}" for p, c in s.pools.items()) + ")",
        f"S-box lookups:   {s.sbox_ops} in the op, {s.precomputed_sbox_ops} at key load",
        f"rounds:          {len(s.rounds)} data-path S-box levels"
        + (f", {s.steps_per_round} step(s) each" if s.steps_per_round else ""),
        f"delay model:     {sp.target}: worst step ~{worst} ps (~{1e6 / worst:.0f} MHz)"
        if worst else f"delay model:     {sp.target}",
    ]
    if sp.latency is not None and sp.latency != s.latency:
        lines.append(f"note:            spec latency {sp.latency}, scheduled {s.latency}")
    if sp.ii is not None and sp.ii < s.ii:
        lines.append(f"note:            spec II {sp.ii} not met (scheduled {s.ii})")
    lines.append("")
    lines.append("## rounds (round: steps)")
    for r, a, b in s.rounds:
        lines.append(f"  round {r}: steps {a}..{b}")
    lines.append("")
    lines.append("## S-box lookups per step")
    for st in range(s.latency):
        u = s.usage.get(st, {})
        lines.append(f"  step {st}: " + ", ".join(f"{p} {u.get(p, 0)}" for p in s.pools)
                     + f"; ~{s.step_delay_ps.get(st, 0)} ps")
    if full:
        lines.append("")
        lines.append("## node -> step / resource (step -1: constant or at key load)")
        for n in s.fn.nodes:
            pl = s.placed[n.name]
            res = f" {pl.pool}[{pl.inst}]" if pl.pool else ""
            lines.append(f"  {n.name} {n.kind} {pl.step}{res}")
    return "\n".join(lines) + "\n"


def summary(s: Schedule) -> dict:
    return {"function": s.fn.sv_path, "arch": s.spec.arch, "latency": s.latency,
            "ii": s.ii, "res_mii": s.res_mii, "sbox_count": s.sbox_count,
            "sbox_pools": dict(s.used), "sbox_ops": s.sbox_ops,
            "precomputed_sbox_ops": s.precomputed_sbox_ops,
            "key_expansion": s.spec.key_expansion, "rounds": len(s.rounds),
            "steps_per_round": s.steps_per_round,
            "worst_step_ps": max(s.step_delay_ps.values(), default=0)}


# ---- Serialisation (ArchSched -> ArchLower re-entry) ----------------------

FORMAT = "zuspec.synth.archSchedule"
VERSION = 1


def ir_hash(ir_text: str) -> str:
    return hashlib.sha256(ir_text.encode()).hexdigest()


def _spec_dict(sp: ArchSpec) -> dict:
    return {f.name: getattr(sp, f.name) for f in dc.fields(ArchSpec)}


def inputs_hash(sp: ArchSpec, consts: Dict[str, int], data_inputs: Sequence[str]) -> str:
    """Hash of everything besides the IR that the schedule depends on: the
    spec (minus its file path), the specialization constants and the data
    inputs."""
    d = _spec_dict(sp)
    d.pop("path", None)
    blob = json.dumps({"spec": d, "specialize": {k: int(v) for k, v in consts.items()},
                       "data_inputs": list(data_inputs)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def to_json(s: Schedule) -> str:
    """The full schedule (everything the report, emit_v3 and emit_v5 read),
    deterministic (sorted keys; ordered maps as lists of pairs in the
    schedule's own order). ``fn`` is not stored: ``from_json`` rebuilds it
    from the IR text, which must hash to ``ir_sha256``."""
    if not s.ir_sha256:
        raise ValueError("to_json: schedule has no IR hash (make it with archsched.run)")
    d = {
        "format": FORMAT, "version": VERSION,
        "function": s.fn.sv_path,
        "ir_sha256": s.ir_sha256,
        "inputs_sha256": inputs_hash(s.spec, s.consts, s.data_inputs),
        "spec": _spec_dict(s.spec),
        "specialize": dict(s.consts),
        "data_inputs": list(s.data_inputs),
        "latency": s.latency, "ii": s.ii, "res_mii": s.res_mii,
        "pools": [[p, c] for p, c in s.pools.items()],
        "used": [[p, c] for p, c in s.used.items()],
        "usage": [[st, p, c] for st, u in s.usage.items() for p, c in u.items()],
        "rounds": [list(r) for r in s.rounds],
        "step_delay_ps": [[st, d] for st, d in s.step_delay_ps.items()],
        "sbox_ops": s.sbox_ops, "precomputed_sbox_ops": s.precomputed_sbox_ops,
        "placed": [[x, pl.step, pl.pool, pl.inst] for x, pl in s.placed.items()],
    }
    return json.dumps(d, indent=1, sort_keys=True) + "\n"


def from_json(text: str, ir_text: str, sp: Optional[ArchSpec] = None,
              consts: Optional[Dict[str, int]] = None,
              data_inputs: Optional[Sequence[str]] = None):
    """(original fn, Schedule) from ``to_json`` output and the IR it was
    made from. Raises ValueError if the IR text, or the spec / constants /
    data inputs given, are not the ones the schedule was made from."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    d = json.loads(text)
    if d.get("format") != FORMAT or d.get("version") != VERSION:
        raise ValueError(f"from_json: not a {FORMAT} v{VERSION} file "
                         f"(format={d.get('format')!r}, version={d.get('version')!r})")
    if ir_hash(ir_text) != d["ir_sha256"]:
        raise ValueError(f"from_json: schedule of {d['function']} was made from a "
                         f"different IR (sha256 {d['ir_sha256'][:12]}..., "
                         f"this IR {ir_hash(ir_text)[:12]}...)")
    if sp is None:
        sp = ArchSpec(**d["spec"])
    consts = dict(d["specialize"]) if consts is None else {k: int(v) for k, v in consts.items()}
    data_inputs = list(d["data_inputs"]) if data_inputs is None else list(data_inputs)
    if inputs_hash(sp, consts, data_inputs) != d["inputs_sha256"]:
        raise ValueError(f"from_json: schedule of {d['function']} was made for a different "
                         f"arch spec / specialize / data_inputs")
    orig = xls_ir.parse(ir_text)
    f = xls_ir.specialize(xls_ir.parse(ir_text), consts)
    xls_ir.classify(f)
    if f.sv_path != d["function"]:
        raise ValueError(f"from_json: IR function {f.sv_path} is not {d['function']}")
    placed = {x: Placed(st, pool, inst) for x, st, pool, inst in d["placed"]}
    names = {n.name for n in f.nodes}
    if set(placed) != names:
        raise ValueError("from_json: placed nodes do not match the IR's nodes")
    usage: Dict[int, Dict[str, int]] = {}
    for st, p, c in d["usage"]:
        usage.setdefault(st, {})[p] = c
    s = Schedule(f, sp, placed, d["latency"], d["ii"], d["res_mii"],
                 {p: c for p, c in d["pools"]}, {p: c for p, c in d["used"]}, usage,
                 [tuple(r) for r in d["rounds"]],
                 {st: v for st, v in d["step_delay_ps"]},
                 d["sbox_ops"], d["precomputed_sbox_ops"],
                 d["ir_sha256"], consts, data_inputs)
    return orig, s
