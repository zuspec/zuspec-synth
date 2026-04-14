#****************************************************************************
# Copyright 2019-2026 Matthew Ballance and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#****************************************************************************
"""IndexedPool hazard analysis and SystemVerilog scoreboard generation.

``IndexedPool`` models per-index lock/share reservations used by the MLS
scheduler to detect data hazards between pipelined actions.  The two hazard
kinds are:

**RAW** (Read-After-Write)
    An in-flight ``lock(idx)`` from a producer and a subsequent
    ``share(idx)`` from a consumer that could alias (same ``idx``).
    The pipeline must stall the consumer until the producer commits.

**WAW** (Write-After-Write)
    Two concurrent ``lock(idx)`` claims that could alias.
    The second write must be serialised after the first.

This module contains:

``IndexedPoolHazardPair``
    Describes one RAW or WAW hazard pair between a lock port and a
    share/lock port.

``IndexedPoolHazardAnalyzer``
    Determines which comparators are needed from the ``IndexedPoolDeclIR``
    and concurrent port counts.

``IndexedPoolSVGenerator``
    Generates a synthesisable Verilog 2005 scoreboard module that:

    * Maintains a one-hot bitmap over all ``depth`` indices.
    * **Set** path: instruction dispatched from RegRead → mark ``rd`` in-flight.
    * **Clear** path: WriteBack commits → unmark ``rd``.
    * **Query** paths (one per share port): combinatorial check whether a
      consumer's ``rs`` index is currently locked.
    * **Hazard** output: OR of all active query hits → stall signal for
      RegRead.

    When ``set_we`` and ``clear_we`` alias to the same index in the same
    cycle, ``set`` wins (the instruction that just entered EX keeps the
    reservation alive).

    ``noop_idx`` (e.g. x0 in RISC-V) is excluded from all comparisons and
    bitmap storage; the synthesiser will prune any logic reaching it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

from ..elab.elab_ir import IndexedPoolDeclIR


def _pfx(prefix: str, name: str) -> str:
    """Return '{prefix}_{name}' when prefix is non-empty, else '{name}'."""
    return f'{prefix}_{name}' if prefix else name


# ---------------------------------------------------------------------------
# Hazard data structures
# ---------------------------------------------------------------------------

@dataclass
class IndexedPoolHazardPair:
    """One hazard between two concurrent IndexedPool accesses.

    Attributes
    ----------
    kind:
        ``'RAW'`` — ``lock`` producer vs ``share`` consumer: stall consumer.
        ``'WAW'`` — two concurrent ``lock`` claims: serialise.
    lock_port:
        Index of the lock-port (0-based) that produces the value.
    share_port:
        Index of the share-port that reads the value (RAW only; ``-1`` for WAW).
    """
    kind:       str   # 'RAW' | 'WAW'
    lock_port:  int
    share_port: int   # -1 for WAW


# ---------------------------------------------------------------------------
# Hazard analyser
# ---------------------------------------------------------------------------

class IndexedPoolHazardAnalyzer:
    """Determines which hazard-detection comparators are needed.

    Rules
    -----
    * ``(lock, share)`` pair → RAW hazard → stall comparator.
    * ``(lock, lock)`` pair → WAW hazard → stall comparator.
    * ``(share, share)`` pair → no hazard.
    * Any pair involving ``noop_idx`` is structural no-op and is omitted.

    Parameters
    ----------
    n_lock_ports:
        Number of concurrent ``lock`` claims (typically 1 for in-order).
    n_share_ports:
        Number of concurrent ``share`` claims (typically 2: rs1 + rs2).
    """

    def analyze(
            self,
            decl: IndexedPoolDeclIR,
            n_lock_ports:  int = 1,
            n_share_ports: int = 2,
            proven_distinct: FrozenSet[Tuple[int, int]] = frozenset(),
    ) -> List[IndexedPoolHazardPair]:
        """Return all hazard pairs that require hardware comparators."""
        pairs: List[IndexedPoolHazardPair] = []

        # RAW: each (share_port, lock_port) combination
        for lp in range(n_lock_ports):
            for sp in range(n_share_ports):
                if (lp, sp) in proven_distinct:
                    continue
                pairs.append(IndexedPoolHazardPair(kind='RAW', lock_port=lp, share_port=sp))

        # WAW: each pair of lock ports
        for lp_a in range(n_lock_ports):
            for lp_b in range(lp_a + 1, n_lock_ports):
                pairs.append(IndexedPoolHazardPair(kind='WAW', lock_port=lp_a, share_port=-1))

        return pairs


# ---------------------------------------------------------------------------
# Verilog generator
# ---------------------------------------------------------------------------

class IndexedPoolSVGenerator:
    """Generates a synthesisable Verilog 2005 scoreboard module.

    Port naming convention
    ----------------------
    Set (dispatch, one per lock port)::

        input  wire              set{N}_we
        input  wire [IDX_W-1:0] set{N}_idx

    Clear (commit, one per lock port)::

        input  wire              clear{N}_we
        input  wire [IDX_W-1:0] clear{N}_idx

    Query (hazard check, one per share port)::

        input  wire [IDX_W-1:0] query{N}_idx

    Output::

        output wire              hazard   // OR of all active query hits
    """

    def generate(
            self,
            decl: IndexedPoolDeclIR,
            hazards: Optional[List[IndexedPoolHazardPair]] = None,
            n_lock_ports:  int = 1,
            n_share_ports: int = 2,
            module_prefix: str = "",
    ) -> str:
        """Return the complete Verilog 2005 scoreboard module source."""
        if hazards is None:
            hazards = []

        aw      = decl.idx_width
        depth   = decl.depth
        noop    = decl.noop_idx
        name    = _pfx(module_prefix, f'{decl.field_name}_scoreboard')

        noop_sv = f"{aw}'d{noop}" if noop is not None else None

        lines: List[str] = []
        lines.append(f'// {"=" * 60}')
        lines.append(f'// {name} — pipeline scoreboard for {decl.field_name}')
        lines.append(f'//')
        lines.append(f'// Tracks in-flight rd indices to detect RAW / WAW hazards.')
        lines.append(f'// set_we/set_idx  : instruction dispatched (RegRead → Execute)')
        lines.append(f'// clear_we/clear_idx : instruction committed (WriteBack)')
        lines.append(f'// query{{N}}_idx   : rs1/rs2 of instruction at RegRead')
        lines.append(f'// hazard          : assert to stall RegRead stage')
        if noop is not None:
            lines.append(f'// noop_idx={noop}     : excluded (x0 in RISC-V)')
        lines.append(f'// {"=" * 60}')
        lines.append(f'module {name} #(')
        lines.append(f'    parameter DEPTH     = {depth},')
        lines.append(f'    parameter IDX_WIDTH = {aw}')
        if noop is not None:
            lines.append(f'    ,parameter NOOP_IDX  = {noop}')
        lines.append(f') (')
        lines.append(f'    input  wire                   clk,')
        lines.append(f'    input  wire                   rst_n,')

        for lp in range(n_lock_ports):
            sfx = '' if n_lock_ports == 1 else str(lp)
            lines.append(f'    // Dispatch: instruction entering execute — lock its rd')
            lines.append(f'    input  wire                   set{sfx}_we,')
            lines.append(f'    input  wire [IDX_WIDTH-1:0]   set{sfx}_idx,')

        for lp in range(n_lock_ports):
            sfx = '' if n_lock_ports == 1 else str(lp)
            lines.append(f'    // Commit: writeback completing — unlock rd')
            lines.append(f'    input  wire                   clear{sfx}_we,')
            lines.append(f'    input  wire [IDX_WIDTH-1:0]   clear{sfx}_idx,')

        for sp in range(n_share_ports):
            lines.append(f'    // Hazard query for share port {sp} (rs{sp+1})')
            comma = ',' if sp < n_share_ports - 1 else ''
            lines.append(f'    input  wire [IDX_WIDTH-1:0]   query{sp}_idx,')

        lines.append(f'    output wire                   hazard')
        lines.append(f');')
        lines.append(f'    reg [DEPTH-1:0] scoreboard;')
        lines.append(f'')

        # RAW hazard comparators (combinatorial)
        raw_hazards = [h for h in hazards if h.kind == 'RAW']
        if raw_hazards:
            lines.append(f'    // RAW hazard comparators')
            lines.append(f'    // idx is a rand variable — MLS conservatively generates a')
            lines.append(f'    // comparator for every concurrent (lock_port, share_port) pair.')
            for h in raw_hazards:
                sp = h.share_port
                lp = h.lock_port
                lp_sfx = '' if n_lock_ports == 1 else str(lp)
                wire_name = f'raw_lp{lp_sfx}_sp{sp}'
                noop_guard = f'(query{sp}_idx != NOOP_IDX) && ' if noop is not None else ''
                lines.append(
                    f'    wire {wire_name} = {noop_guard}scoreboard[query{sp}_idx];'
                )
            lines.append(f'')
            or_terms = ' | '.join(
                f'raw_lp{"" if n_lock_ports == 1 else h.lock_port}_sp{h.share_port}'
                for h in raw_hazards
            )
            lines.append(f'    assign hazard = {or_terms};')
        else:
            lines.append(f"    assign hazard = 1'b0;  // no hazard pairs")
        lines.append(f'')

        # Scoreboard update logic
        lines.append(f'    // Scoreboard update — synchronous, set wins over clear on conflict.')
        lines.append(f'    always @(posedge clk or negedge rst_n) begin')
        lines.append(f"        if (!rst_n) begin")
        lines.append(f"            scoreboard <= {{DEPTH{{1'b0}}}};")
        lines.append(f"        end else begin")

        for lp in range(n_lock_ports):
            sfx = '' if n_lock_ports == 1 else str(lp)
            noop_guard = f' && clear{sfx}_idx != NOOP_IDX' if noop is not None else ''
            lines.append(f'            if (clear{sfx}_we{noop_guard})')
            lines.append(f"                scoreboard[clear{sfx}_idx] <= 1'b0;")
            noop_guard2 = f' && set{sfx}_idx != NOOP_IDX' if noop is not None else ''
            lines.append(f'            // set after clear: same-cycle conflict → set wins')
            lines.append(f'            if (set{sfx}_we{noop_guard2})')
            lines.append(f"                scoreboard[set{sfx}_idx] <= 1'b1;")

        lines.append(f'        end')
        lines.append(f'    end')
        lines.append(f'endmodule')

        return '\n'.join(lines)
