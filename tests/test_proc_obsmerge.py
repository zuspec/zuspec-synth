"""zuspec.synth.proc.obsmerge: register bits merged into what they equal while
live. Each design is lowered twice (with and without the pass) and the two
netlists are simulated side by side with iverilog on the same random inputs."""
import shutil
import subprocess

import pytest

from zuspec.synth.proc import ir as P
from zuspec.synth.proc import term as T
from zuspec.synth.proc.cosim import random_tb
from zuspec.synth.proc.emit import emit_verilog
from zuspec.synth.proc.lower import LowerOptions, lower, optimize

from test_proc_structured import DESIGNS, V, _design, _out

pytestmark = [pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog"),
              pytest.mark.filterwarnings("ignore")]
pytest.importorskip("pysat")


def copy():
    """A command register loaded while idle, copied at the start of a
    three-cycle operation and read at its end: the copy equals the source
    whenever it is read."""
    ld = P.Var("ld", 1, "in")
    go = P.Var("go", 1, "in")
    din = P.Var("din", 8, "in")
    src = P.Var("src", 8, "reg")
    cp = P.Var("cp", 8, "reg")
    out = P.Var("out", 8, "reg")
    busy = P.Var("busy", 1, "reg")
    body = [P.Loop([
        P.Await(None, initial=True, label="idle"),
        P.If(V(ld), [P.Assign(src, V(din))], []),
        P.If(V(go), [
            P.Assign(cp, V(src)),
            P.Assign(busy, T.ONE),
            P.Await(None, label="s1"),
            P.Await(None, label="s2"),
            P.Assign(out, T.xor(V(cp), T.const(0x3C, 8))),
            P.Assign(busy, T.ZERO),
        ], []),
    ])]
    return _design("copy", [P.Proc("p", body)], [ld, go, din, src, cp, out, busy],
                   _out(out, busy))


def relay():
    """Two processes with a request/response channel: a caller latches an
    operand into a request register, the callee copies it into a work
    register, and the result comes back through a response register."""
    go = P.Var("go", 1, "in")
    din = P.Var("din", 4, "in")
    arg = P.Var("arg", 4, "reg")
    req = P.Var("req", 4, "reg")
    req_v = P.Var("req_v", 1, "reg")
    work = P.Var("work", 4, "reg")
    rsp = P.Var("rsp", 4, "reg")
    rsp_v = P.Var("rsp_v", 1, "reg")
    res = P.Var("res", 4, "reg")
    caller = [P.Loop([
        P.Await(None, initial=True, label="c_idle"),
        P.If(V(go), [
            P.Assign(arg, V(din)),
            P.Assign(req, V(arg)),
            P.Assign(req_v, T.ONE),
            P.Await(V(rsp_v), label="c_wait"),
            P.Assign(req_v, T.ZERO),
            P.Assign(res, V(rsp)),
        ], []),
    ])]
    callee = [P.Loop([
        P.Await(None, initial=True, label="e_idle"),
        P.Assign(rsp_v, T.ZERO),
        P.If(V(req_v), [
            P.Assign(work, V(req)),
            P.Await(None, label="e_1"),
            P.Assign(work, T.add(V(work), T.const(3, 4))),
            P.Await(None, label="e_2"),
            P.Assign(rsp, V(work)),
            P.Assign(rsp_v, T.ONE),
            P.Await(None, label="e_3"),
        ], []),
    ])]
    return _design("relay", [P.Proc("caller", caller), P.Proc("callee", callee)],
                   [go, din, arg, req, req_v, work, rsp, rsp_v, res], _out(res, req_v))


def _cosim(tmp_path, design):
    ref = optimize(lower(design(), LowerOptions()))
    nl = optimize(lower(design(), LowerOptions(obsmerge=True)))
    a = emit_verilog(ref).replace(f"module {ref.name} (", f"module {ref.name}_f (")
    b = emit_verilog(nl).replace(f"module {nl.name} (", f"module {nl.name}_s (")
    (tmp_path / "f.v").write_text(a)
    (tmp_path / "s.v").write_text(b)
    (tmp_path / "tb.v").write_text(random_tb(ref, cycles=2000))
    r = subprocess.run(["iverilog", "-o", str(tmp_path / "sim"), str(tmp_path / "tb.v"),
                        str(tmp_path / "f.v"), str(tmp_path / "s.v")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["vvp", "-n", str(tmp_path / "sim")], capture_output=True, text=True)
    assert "PASS" in r.stdout, r.stdout[-2000:]
    return ref, nl


@pytest.mark.parametrize("design", DESIGNS + [copy, relay], ids=lambda d: d.__name__)
def test_merged_simulates_as_unmerged(tmp_path, design):
    _cosim(tmp_path, design)


def _bits(nl):
    return sum(r.w for r in nl.regs)


def test_copy_merged(tmp_path):
    ref, nl = _cosim(tmp_path, copy)
    assert {(m.reg, m.what) for m in nl.merges} >= {("cp", f"src[{b}]") for b in range(8)}, \
        nl.report
    assert "cp" not in {r.name for r in nl.regs}
    assert _bits(nl) == _bits(ref) - 8


def test_relay_merged(tmp_path):
    ref, nl = _cosim(tmp_path, relay)
    assert _bits(nl) < _bits(ref), "\n".join(nl.report)


@pytest.mark.skipif(shutil.which("sby") is None, reason="needs SymbiYosys")
@pytest.mark.parametrize("design", [copy, relay, DESIGNS[1]], ids=lambda d: d.__name__)
def test_certificate_proves(tmp_path, design):
    """The debug and tight netlists are equivalent by k-induction, with
    obsmerge's invariants as the strengthening."""
    assert "DONE (PASS" in _prove(tmp_path, design)


@pytest.mark.skipif(shutil.which("sby") is None, reason="needs SymbiYosys")
def test_certificate_catches_a_wrong_merge(tmp_path):
    """cp merged into the complement of src: the proof fails."""
    def wrong(nl):
        m = {T.var("cp", 8): T.not_(T.var("src", 8))}
        nl.regs = [r for r in nl.regs if r.name != "cp"]
        for r in nl.regs:
            r.next = T.substitute(r.next, m)
    assert "DONE (FAIL" in _prove(tmp_path, copy, obsmerge=False, mutate=wrong)


def _prove(tmp_path, design, obsmerge=True, mutate=None):
    from zuspec.synth.proc.miter import miter, sby
    ref = optimize(lower(design(), LowerOptions()))
    nl = optimize(lower(design(), LowerOptions(obsmerge=obsmerge)))
    if mutate:
        mutate(nl)
    w = {r.name: r.w for r in ref.regs}
    m = miter(ref, nl, [(f"{g.reg}[{g.bit}]", g.invariant(w.__getitem__)) for g in nl.merges])
    (tmp_path / "miter.v").write_text(emit_verilog(m))
    top, text = sby(m, ["miter.v"], depth=8)
    (tmp_path / "miter_top.sv").write_text(top)
    (tmp_path / "eq.sby").write_text(text)
    r = subprocess.run(["sby", "-f", "eq.sby"], cwd=tmp_path, capture_output=True, text=True)
    return r.stdout
