"""zuspec.synth.archsched.emit_v5: folded V5 datapath from an arch schedule."""
import re

import pytest

from zuspec.synth.archsched import emit_v5, run, xls_ir

from test_archsched import _aes_ir, _spec, toy_ir

KEY = list(range(16)) + [0] * 16
PT = bytes.fromhex("00112233445566778899aabbccddeeff")
BLK = [list(PT[4 * i:4 * i + 4]) for i in range(4)]
CT = "69c4e0d86a7b0430d8cdb78070b4c55a"  # FIPS-197 C.1


def _unflat(v: int, n: int = 16) -> str:
    return bytes((v >> (8 * i)) & 0xFF for i in range(n)).hex()


def test_flatten_and_index():
    t = "bits[8][4][4]"
    v = emit_v5.flatten(BLK, t)
    assert _unflat(v) == PT.hex()                      # element 0 at the LSB
    assert emit_v5._index_slice(t, [2]) == (64, 32)
    assert emit_v5._index_slice(t, [2, 3]) == (88, 8)


def test_toy_fold_simulate():
    sp = _spec(sbox_count=2)                  # 4 lanes on 2 S-boxes: 2 steps per round
    _, s = run(toy_ir(lanes=4, rounds=4), sp, {}, ["x"])
    fd = emit_v5.fold(s)
    assert len(fd.sbox_in) == 2
    assert len(fd.classes) < s.latency        # rounds rerolled
    x = [3, 1, 4, 1]
    exp = emit_v5.flatten(xls_ir.evaluate(s.fn, {"k": 7, "x": x}), "bits[8][4]")
    assert emit_v5.simulate(fd, {"k": 7, "x": emit_v5.flatten(x, "bits[8][4]")}) == exp
    sv = emit_v5.emit_sv(fd, "toy__f")
    assert len(re.findall(r"toy__f__sbox u_sb\d+\(", sv)) == 2


def _no_overlap(fd):
    """Values sharing a register have disjoint lifetimes."""
    by = {}
    for v, r in fd.reg_of.items():
        by.setdefault(r, []).append(fd.lifetimes[v])
    for iv in by.values():
        iv.sort()
        for (a0, b0), (a1, b1) in zip(iv, iv[1:]):
            assert b0 < a1


@pytest.mark.parametrize("arch,kexp,lat,sboxes,classes", [
    (dict(fold="iterated", sbox_count=16), "precomputed", 11, 16, 4),            # A1
    (dict(fold="column", sbox_count=4), "shared", 50, 4, 16),                    # A2
    (dict(fold="unrolled", unroll=2, sbox_count=32), "precomputed", 6, 32, 4),   # A4
])
def test_aes_v5(arch, kexp, lat, sboxes, classes):
    orig, s = run(_aes_ir(), _spec(key_expansion=kexp, **arch), {"key_width": 0})
    fd = emit_v5.fold(s)
    assert (fd.latency, len(fd.sbox_in), len(fd.classes)) == (lat, sboxes, classes)
    assert fd.reg_bits < sum(emit_v5.flat_width(s.fn.by_name()[v].type) for v in fd.reg_of)
    _no_overlap(fd)
    ins = {"key": emit_v5.flatten(KEY, "bits[8][32]"),
           "block_": emit_v5.flatten(BLK, "bits[8][4][4]")}
    assert _unflat(emit_v5.simulate(fd, ins)) == CT
    sv = emit_v5.emit_sv(fd, "aes_pkg__encrypt", orig)
    assert len(re.findall(r"aes_pkg__encrypt__sbox u_sb\d+\(", sv)) == sboxes
    assert "input wire [255:0] key" in sv and "output wire [127:0] out" in sv
    assert f"assign output_valid = busy && cnt == 4'd{lat - 1};" in sv \
        or f"cnt == 6'd{lat - 1}" in sv or f"cnt == 3'd{lat - 1}" in sv
    if kexp == "precomputed":
        assert "module aes_pkg__encrypt__keyexp(" in sv
        assert len(fd.pre_sboxes) == 40


def test_levels_bound_disjoint():
    """Partial unroll: each chained S-box level has its own instances (no
    structural combinational loop through the step muxes)."""
    _, s = run(_aes_ir(), _spec(key_expansion="precomputed", fold="unrolled", unroll=2,
                                sbox_count=32), {"key_width": 0})
    fd = emit_v5.fold(s)
    nodes = s.fn.by_name()
    level_of = {}
    for c in fd.classes:
        loc = {x: j for j, x in enumerate(c.rep)}
        lv = {}
        for j, x in enumerate(c.rep):
            l = max((lv[loc[o]] for o in xls_ir.all_operands(nodes[x]) if o in loc), default=0)
            lv[j] = l + (1 if j in c.sbox_inst else 0)
            if j in c.sbox_inst:
                assert level_of.setdefault(c.sbox_inst[j], lv[j]) == lv[j]
    assert sorted(set(level_of.values())) == [1, 2]
