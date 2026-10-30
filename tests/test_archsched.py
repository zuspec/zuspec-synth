"""zuspec.synth.archsched: XLS IR reader, list scheduler, V3 emitter."""
import gzip
import os

import pytest

from zuspec.synth.archsched import emit_v3, run, sched, spec, xls_ir

DATA = os.path.join(os.path.dirname(__file__), "data")


def _aes_sbox():
    def mul(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            a = ((a << 1) ^ (0x11B if a & 0x80 else 0)) & 0xFF
            b >>= 1
        return r
    inv = [0] + [next(b for b in range(1, 256) if mul(a, b) == 1) for a in range(1, 256)]
    out = []
    for a in range(256):
        x = inv[a]
        s = x
        for i in range(1, 5):
            s ^= ((x << i) | (x >> (8 - i))) & 0xFF
        out.append(s ^ 0x63)
    return out


def toy_ir(lanes=4, rounds=3):
    """`rounds` of: s_i = S[x_i]; x_i = s_i ^ s_{i+1} ^ k (k held, x per op)."""
    sb = ", ".join(str(v) for v in _aes_sbox())
    lines = ["package toy__f", "",
             f"top fn toy__f(k: bits[8] id=1, x: bits[8][{lanes}] id=2) -> bits[8][{lanes}] {{",
             f"  S: bits[8][256] = literal(value=[{sb}], id=3)"]
    nid = 10
    cur = []
    for i in range(lanes):
        lines.append(f"  l{i}: bits[2] = literal(value={i}, id={nid})"); nid += 1
        lines.append(f"  x0_{i}: bits[8] = array_index(x, indices=[l{i}], id={nid})"); nid += 1
        cur.append(f"x0_{i}")
    for r in range(1, rounds + 1):
        s = []
        for i in range(lanes):
            lines.append(f"  s{r}_{i}: bits[8] = array_index(S, indices=[{cur[i]}], id={nid})")
            nid += 1
            s.append(f"s{r}_{i}")
        nxt = []
        for i in range(lanes):
            lines.append(f"  x{r}_{i}: bits[8] = xor({s[i]}, {s[(i + 1) % lanes]}, k, id={nid})")
            nid += 1
            nxt.append(f"x{r}_{i}")
        cur = nxt
    lines.append(f"  ret out: bits[8][{lanes}] = array({', '.join(cur)}, id={nid})")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _spec(**kw):
    d = dict(arch="T", algorithm="toy::f", fold="iterated", sbox_count=4, sbox_kind="lut")
    d.update(kw)
    return spec.from_dict(d)


def _sched(sp, **kw):
    _, s = run(toy_ir(**kw), sp, {}, ["x"])
    return s


def test_table_kind():
    sbox = _aes_sbox()
    assert xls_ir.table_kind(sbox) == "sbox"
    xt = [((a << 1) ^ (0x11B if a & 0x80 else 0)) & 0xFF for a in range(256)]
    assert xls_ir.table_kind(xt) == "gfmul"


def test_iterated_full_width():
    s = _sched(_spec())
    assert (s.latency, s.ii, s.sbox_count) == (4, 4, 4)     # load + 3 rounds
    assert s.steps_per_round == 1


def test_resource_limit_folds_rounds():
    s = _sched(_spec(sbox_count=2, fold="column"))
    assert s.latency == 1 + 3 * 2
    assert s.sbox_count == 2
    assert all(u["sbox"] <= 2 for u in s.usage.values())
    assert s.steps_per_round == 2


def test_chaining_and_binding():
    s = _sched(_spec(sbox_count=4))
    nodes = s.fn.by_name()
    for n in s.fn.nodes:
        if n.kind != "sbox":
            continue
        pl = s.placed[n.name]
        assert 0 <= pl.inst < 4 and pl.pool == "sbox"
        # One S-box level per step: no S-box ancestor in the same step.
        work = list(xls_ir.all_operands(n))
        seen = set()
        while work:
            o = work.pop()
            if o in seen or s.placed[o].step != pl.step:
                continue
            seen.add(o)
            assert nodes[o].kind != "sbox"
            work += xls_ir.all_operands(nodes[o])
    # Each step binds distinct instances.
    by_step = {}
    for n, pl in s.placed.items():
        if pl.pool:
            assert (pl.step, pl.inst) not in by_step
            by_step[(pl.step, pl.inst)] = n


def test_unrolled_pipelines():
    s = _sched(_spec(fold="unrolled", sbox_count=12))
    assert s.latency == 2 and s.ii == 1
    s = _sched(_spec(fold="unrolled", unroll=2, sbox_count=8))
    assert s.latency == 1 + 2


def test_bad_spec():
    with pytest.raises(ValueError):
        _spec(fold="sideways")
    with pytest.raises(ValueError):
        _spec(colour="blue")


def _aes_ir():
    with gzip.open(os.path.join(DATA, "aes_pkg__encrypt.opt.ir.gz"), "rt") as fp:
        return fp.read()


def test_aes_specialize_preserves_function():
    text = _aes_ir()
    key = list(range(16)) + [0] * 16
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    blk = [list(pt[4 * i:4 * i + 4]) for i in range(4)]
    f = xls_ir.specialize(xls_ir.parse(text), {"key_width": 0})
    counts = xls_ir.classify(f)
    assert counts["sbox"] == 200              # 10 x 16 data + 10 x 4 key
    ct = xls_ir.evaluate(f, {"key": key, "block_": blk})
    assert bytes(sum(ct, [])).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"  # FIPS-197 C.1


@pytest.mark.parametrize("arch,kexp,lat,sboxes", [
    (dict(fold="iterated", sbox_count=16), "precomputed", 11, 16),   # A1
    (dict(fold="iterated", sbox_count=16), "shared", 14, 16),
    (dict(fold="iterated", sbox_count=16), "separate", 11, 20),
    (dict(fold="column", sbox_count=4), "shared", 50, 4),            # A2
    (dict(fold="unrolled", unroll=2, sbox_count=32), "precomputed", 6, 32),  # A4
])
def test_aes_arch(arch, kexp, lat, sboxes):
    sp = _spec(key_expansion=kexp, **arch)
    orig, s = run(_aes_ir(), sp, {"key_width": 0})
    assert (s.latency, s.sbox_count) == (lat, sboxes)
    assert len(s.rounds) == 10
    rep = sched.report(s)
    assert f"latency:         {lat} clk" in rep
    sv = emit_v3.stage_tlm_sv([(orig, s, "t.txt")], "p_pkg", "p_clk_if", ["aes_pkg"],
                              {"key_width": "aes_pkg::KeyWidth"}, {"key_width": 0})
    assert f"static const int unsigned LATENCY  = {lat};" in sv
    assert ("task op(input bit[7:0] key[0:31], input aes_pkg::KeyWidth key_width, "
            "input bit[7:0] block_[0:3][0:3], output bit[7:0] result[0:3][0:3]);") in sv
    assert "result = aes_pkg::encrypt(key, key_width, block_);" in sv
