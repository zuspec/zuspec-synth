"""zuspec.synth.archsched.sched.to_json / from_json: the full schedule
round-trips (V3 and V5 from it are identical to the direct path)."""
import json

import pytest

from zuspec.synth.archsched import emit_v3, emit_v5, load, run, sched

from test_archsched import _aes_ir, _spec

ARCHS = [
    ("A1", dict(fold="iterated", sbox_count=16), "precomputed"),
    ("A2", dict(fold="column", sbox_count=4), "shared"),
    ("A4", dict(fold="unrolled", unroll=2, sbox_count=32), "precomputed"),
]
CONSTS = {"key_width": 0}


def _v3(orig, s):
    return emit_v3.stage_tlm_sv([(orig, s, "t.txt")], "p_pkg", "p_clk_if", ["aes_pkg"],
                                {"key_width": "aes_pkg::KeyWidth"}, CONSTS)


def _v5(orig, s):
    return emit_v5.emit_sv(emit_v5.fold(s), "aes_pkg__encrypt", orig)


@pytest.fixture(scope="module")
def ir():
    return _aes_ir()


@pytest.mark.parametrize("name,arch,kexp", ARCHS, ids=[a[0] for a in ARCHS])
def test_roundtrip(ir, name, arch, kexp):
    sp = _spec(key_expansion=kexp, **arch)
    orig, s = run(ir, sp, CONSTS)
    js = sched.to_json(s)
    d = json.loads(js)
    assert (d["format"], d["version"]) == (sched.FORMAT, sched.VERSION)
    assert len(d["placed"]) == len(s.fn.nodes)

    orig2, s2 = load(js, ir, sp, CONSTS, ["block_"])
    assert sched.to_json(s2) == js                      # idempotent
    assert sched.report(s2) == sched.report(s)
    assert _v3(orig2, s2) == _v3(orig, s)
    assert _v5(orig2, s2) == _v5(orig, s)

    # Without the spec / params: taken from the json.
    orig3, s3 = load(js, ir)
    assert _v5(orig3, s3) == _v5(orig, s)


def test_guard(ir):
    sp = _spec(key_expansion="precomputed", fold="iterated", sbox_count=16)
    _, s = run(ir, sp, CONSTS)
    js = sched.to_json(s)
    with pytest.raises(ValueError, match="different IR"):
        load(js, ir + "\n")
    with pytest.raises(ValueError, match="different arch spec"):
        load(js, ir, _spec(key_expansion="precomputed", fold="iterated", sbox_count=8))
    with pytest.raises(ValueError, match="different arch spec"):
        load(js, ir, sp, {"key_width": 1})
    with pytest.raises(ValueError, match="different arch spec"):
        load(js, ir, sp, CONSTS, ["block_", "key"])
    # The spec's file path is not part of the guard.
    sp2 = _spec(key_expansion="precomputed", fold="iterated", sbox_count=16)
    sp2.path = "/elsewhere/a1.yaml"
    load(js, ir, sp2, CONSTS)
    d = json.loads(js)
    d["version"] = 99
    with pytest.raises(ValueError, match="not a"):
        load(json.dumps(d), ir)
