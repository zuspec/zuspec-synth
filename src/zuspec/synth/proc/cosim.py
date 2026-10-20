"""Co-simulation of two emissions of one lowered netlist (flat and structured).

:func:`random_tb` writes a testbench that drives both modules (renamed
``<name>_f`` and ``<name>_s``) with the same random inputs and occasional
resets, and compares every output every cycle; it prints ``PASS`` or the
mismatches and ``FAIL <n>``."""
from __future__ import annotations

import random

from .lower import Netlist


def random_tb(nl: Netlist, cycles: int = 400, seed: int = 1) -> str:
    ins = [(n, w) for n, w in nl.inputs if n not in (nl.clock, nl.reset)]
    outs = [(n, t.w) for n, t in nl.outputs]
    rng = random.Random(seed)
    lines = ["`timescale 1ns/1ns", "module tb;", "  reg clk = 0, rst = 1;"]
    for n, w in ins:
        lines.append(f"  reg [{w - 1}:0] {n} = 0;")
    for tag in ("f", "s"):
        for n, w in outs:
            lines.append(f"  wire [{w - 1}:0] {tag}_{n};")
        conns = [f".{nl.clock}(clk)"] + ([f".{nl.reset}(rst)"] if nl.reset else [])
        conns += [f".{n}({n})" for n, _ in ins] + [f".{n}({tag}_{n})" for n, _ in outs]
        lines.append(f"  {nl.name}_{tag} u_{tag} ({', '.join(conns)});")
    lines.append("  integer errs = 0;")
    lines.append("  always #5 clk = ~clk;")
    lines.append("  initial begin")
    for c in range(cycles):
        r = 1 if c < 2 or rng.random() < 0.02 else 0
        lines.append(f"    rst = {r};" + "".join(f" {n} = {rng.getrandbits(w)};" for n, w in ins))
        lines.append("    @(posedge clk); #1;")
        for n, _ in outs:
            lines.append(f"    if (f_{n} !== s_{n}) begin errs = errs + 1; "
                         f"$display(\"cycle {c}: {n} flat=%h structured=%h\", f_{n}, s_{n}); end")
    lines.append("    if (errs == 0) $display(\"PASS\"); else $display(\"FAIL %0d\", errs);")
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"
