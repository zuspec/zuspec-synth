"""dv-flow task ``zuspec.synth.ArchSched``."""
from __future__ import annotations

import json
import os
from typing import List

import yaml
from dv_flow.mgr import FileSet, TaskDataResult
from dv_flow.mgr.task_data import SeverityE, TaskMarker

from . import emit_v3, run, sched, spec as spec_mod


def _error(msg: str) -> TaskDataResult:
    return TaskDataResult(status=1, changed=False, output=[],
                          markers=[TaskMarker(msg=msg, severity=SeverityE.Error)])


def _files(input, filetypes) -> List[str]:
    return [os.path.join(fs.basedir, f) for fs in getattr(input, "inputs", []) or []
            if getattr(fs, "filetype", None) in filetypes for f in fs.files]


async def ArchSched(ctxt, input) -> TaskDataResult:
    p = input.params
    # The arch spec: the yaml file whose `arch` matches (or the only one).
    specs = []
    for f in _files(input, ("yaml", "archSpec")):
        with open(f) as fp:
            d = yaml.safe_load(fp)
        if isinstance(d, dict) and "arch" in d:
            specs.append((f, d))
    if p.arch:
        specs = [(f, d) for f, d in specs if str(d["arch"]).lower() == p.arch.lower()]
    if len(specs) != 1:
        return _error(f"ArchSched: need exactly one arch spec (arch={p.arch!r}), "
                      f"found {[f for f, _ in specs]}")
    try:
        arch = spec_mod.from_dict(specs[0][1], specs[0][0])
    except ValueError as e:
        return _error(str(e))

    irs = {}
    for f in _files(input, ("xlsOptIR", "xlsIR")):
        mod = os.path.basename(f).split(".")[0]
        if f.endswith(".opt.ir") or mod not in irs:
            irs[mod] = f
    funcs = list(p.functions or []) or [arch.algorithm]
    mods = [fn.replace("::", "__") for fn in funcs]
    missing = [m for m in mods if m not in irs]
    if missing:
        return _error(f"ArchSched: no XLS IR for {missing} (have {sorted(irs)})")

    consts = {k: int(v) for k, v in dict(p.specialize or {}).items()}
    rundir = input.rundir
    os.makedirs(rundir, exist_ok=True)
    items, reports, summ = [], [], []
    for m in mods:
        with open(irs[m]) as fp:
            text = fp.read()
        try:
            orig, s = run(text, arch, consts, list(p.data_inputs))
        except (KeyError, ValueError, RuntimeError) as e:
            return _error(f"ArchSched: {m}: {e}")
        rep = f"{m}.{arch.arch}.sched.txt"
        with open(os.path.join(rundir, rep), "w") as fp:
            fp.write(sched.report(s))
        reports.append(rep)
        items.append((orig, s, rep))
        summ.append(sched.summary(s))
    with open(os.path.join(rundir, "schedule.json"), "w") as fp:
        json.dump(summ, fp, indent=2)
    reports.append("schedule.json")
    pkg = p.pkg_name or f"{arch.arch.lower()}_stage_tlm_pkg"
    clk_if = p.clk_if or f"{pkg}_clk_if"
    with open(os.path.join(rundir, f"{pkg}.sv"), "w") as fp:
        fp.write(emit_v3.stage_tlm_sv(items, pkg, clk_if, list(p.imports or []),
                                      dict(p.arg_types or {}), consts))
    return TaskDataResult(status=0, changed=True, output=[
        FileSet(src=input.name, filetype="systemVerilogSource", basedir=rundir,
                files=[f"{pkg}.sv"]),
        FileSet(src=input.name, filetype="text", basedir=rundir, files=reports)])
