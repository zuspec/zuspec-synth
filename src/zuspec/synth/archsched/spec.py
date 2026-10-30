"""Architecture specs (fw-aes ``src/arch/a<N>.yaml``, docs/design/arch-specs.md).

Fields read: ``arch``, ``algorithm``, ``fold``, ``unroll``, ``sbox_count``,
``sbox_kind``, ``ii``, ``latency`` (targets, checked not imposed),
``target`` (delay model), plus two extensions this scheduler needs:

``key_expansion``
    How the block-independent part of the function (the key schedule: every
    node that does not depend on a ``data_inputs`` parameter) is built.
    ``shared`` (default, as arch-specs.md says): scheduled in the op, on the
    same S-boxes as the data path. ``separate``: scheduled in the op on its
    own ``key_sbox_count`` S-boxes. ``precomputed``: done when the key is
    loaded (``set_key``; the key is a CSR-held input, contract C3), so it
    costs the op no clocks and no S-boxes; the round keys are stored.
``key_sbox_count``
    S-boxes for ``key_expansion: separate`` (default 4 per round per clock).
"""
from __future__ import annotations

import dataclasses as dc
import math
from typing import Dict, Optional

import yaml

FOLDS = ("unrolled", "iterated", "column", "byte")
KEY_EXPANSIONS = ("shared", "separate", "precomputed")

#: Combinational delay (ps) per node kind, per target. Rough ECP5 (-6/-8)
#: figures: LUT4 + routing ~ 400 ps per level; a 256x8 table is ~3 levels
#: (LUT4 + PFU muxes), a Canright S-box ~6 XOR/AND levels. Guesses, used
#: only for the reported per-step delay / fmax estimate.
DELAY_PS: Dict[str, Dict[str, int]] = {
    "ecp5": {"sbox:lut": 1300, "sbox:canright": 2400, "sbox:canright_masked": 2400,
             "gfmul": 800, "logic": 400, "mux": 500, "wire": 0},
    "ice40": {"sbox:lut": 2000, "sbox:canright": 3600, "sbox:canright_masked": 3600,
              "gfmul": 1200, "logic": 600, "mux": 700, "wire": 0},
}


@dc.dataclass
class ArchSpec:
    arch: str
    algorithm: str
    fold: str = "iterated"
    unroll: Optional[int] = None
    sbox_count: int = 16
    sbox_kind: str = "lut"
    ii: Optional[int] = None
    latency: Optional[int] = None
    target: str = "ecp5"
    key_expansion: str = "shared"
    key_sbox_count: Optional[int] = None
    pipeline_stages: Optional[int] = None
    mask_order: Optional[int] = None
    rand_bits_per_sbox: Optional[int] = None
    path: str = ""

    @property
    def levels_per_step(self) -> float:
        """S-box levels a clock step may chain (loop folding): one round per
        clock for iterated/column/byte folds, ``unroll`` for a partial unroll,
        unbounded for a full unroll."""
        if self.fold == "unrolled":
            return float(self.unroll) if self.unroll else math.inf
        return 1.0

    @property
    def key_sboxes(self) -> int:
        if self.key_sbox_count is not None:
            return self.key_sbox_count
        return 4 * (self.unroll or 1)

    def delay_ps(self, kind: str) -> int:
        tbl = DELAY_PS.get(self.target, DELAY_PS["ecp5"])
        if kind == "sbox":
            return tbl.get(f"sbox:{self.sbox_kind}", tbl["sbox:lut"])
        return tbl.get(kind, 0)


def load(path: str) -> ArchSpec:
    with open(path) as fp:
        d = yaml.safe_load(fp)
    return from_dict(d, path)


def from_dict(d: dict, path: str = "") -> ArchSpec:
    known = {f.name for f in dc.fields(ArchSpec)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{path}: unknown arch-spec field(s) {sorted(unknown)}")
    s = ArchSpec(**{k: v for k, v in d.items()}, path=path)
    if s.fold not in FOLDS:
        raise ValueError(f"{path}: fold must be one of {FOLDS}, not {s.fold!r}")
    if s.key_expansion not in KEY_EXPANSIONS:
        raise ValueError(f"{path}: key_expansion must be one of {KEY_EXPANSIONS}")
    return s
