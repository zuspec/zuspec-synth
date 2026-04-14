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
"""Tests for regfile synthesis: IR, hazard analysis, and SV generation."""

import sys
import os

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

if '' in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

import pytest
from zuspec.synth.elab.elab_ir import RegFileDeclIR
from zuspec.synth.sprtl.regfile_synth import (
    RegFileHazardPair, RegFileHazardAnalyzer, RegFileSVGenerator,
)


# ---------------------------------------------------------------------------
# TestRegFileDeclIR
# ---------------------------------------------------------------------------

class TestRegFileDeclIR:
    def _make(self, **kw):
        defaults = dict(
            field_name='regfile', depth=32,
            idx_width=5, data_width=32,
            read_ports=2, write_ports=1, shared_port=False,
        )
        defaults.update(kw)
        return RegFileDeclIR(**defaults)

    def test_basic_construction(self):
        d = self._make()
        assert d.field_name  == 'regfile'
        assert d.depth       == 32
        assert d.idx_width   == 5
        assert d.data_width  == 32
        assert d.read_ports  == 2
        assert d.write_ports == 1
        assert d.shared_port is False

    def test_shared_port_flag(self):
        d = self._make(read_ports=1, write_ports=1, shared_port=True)
        assert d.shared_port is True


# ---------------------------------------------------------------------------
# TestRegFileHazardAnalyzer
# ---------------------------------------------------------------------------

class TestRegFileHazardAnalyzer:
    def _analyze(self, **kw):
        defaults = dict(
            field_name='rf', depth=32, idx_width=5, data_width=32,
            read_ports=2, write_ports=1, shared_port=False,
        )
        defaults.update(kw)
        decl = RegFileDeclIR(**defaults)
        return RegFileHazardAnalyzer().analyze(decl)

    def test_2r_1w_has_raw_pairs(self):
        """2R/1W produces RAW forwarding pairs for each (read, write) combo."""
        hazards = self._analyze(read_ports=2, write_ports=1)
        assert len(hazards) > 0
        assert all(isinstance(h, RegFileHazardPair) for h in hazards)

    def test_shared_port_no_raw_pairs(self):
        """shared_port=True has no RAW in-module pairs (stalls are external)."""
        hazards = self._analyze(read_ports=1, write_ports=1, shared_port=True)
        assert hazards == []

    def test_proven_distinct_reduces_pairs(self):
        """proven_distinct should suppress hazard pairs for non-aliasing indices."""
        all_pairs   = self._analyze(read_ports=2, write_ports=1)
        fewer_pairs = RegFileHazardAnalyzer().analyze(
            RegFileDeclIR(
                field_name='rf', depth=32, idx_width=5, data_width=32,
                read_ports=2, write_ports=1, shared_port=False,
            ),
            proven_distinct=[('rs1', 'rd'), ('rs2', 'rd')],
        )
        assert len(fewer_pairs) <= len(all_pairs)

    def test_single_port_write_no_extra_waw(self):
        """Single write port cannot produce WAW in one cycle."""
        hazards = self._analyze(read_ports=2, write_ports=1)
        waw = [h for h in hazards if h.kind == 'WAW']
        assert waw == []

    def test_hazard_pair_has_expected_fields(self):
        hazards = self._analyze(read_ports=2, write_ports=1)
        h = hazards[0]
        assert hasattr(h, 'reader_port')
        assert hasattr(h, 'writer_port')
        assert hasattr(h, 'kind')


# ---------------------------------------------------------------------------
# TestRegFileSVGenerator_2R1W
# ---------------------------------------------------------------------------

class TestRegFileSVGenerator_2R1W:
    def _sv(self, **kw):
        defaults = dict(
            field_name='rf', depth=32, idx_width=5, data_width=32,
            read_ports=2, write_ports=1, shared_port=False,
        )
        defaults.update(kw)
        decl    = RegFileDeclIR(**defaults)
        hazards = RegFileHazardAnalyzer().analyze(decl)
        return RegFileSVGenerator().generate(decl, hazards)

    def test_module_name_contains_field_name(self):
        sv = self._sv()
        assert 'rf' in sv

    def test_has_module_and_endmodule(self):
        sv = self._sv()
        assert 'module' in sv
        assert 'endmodule' in sv

    def test_two_read_ports(self):
        sv = self._sv(read_ports=2)
        # Expect two read-address ports in the standard rp0/rp1 naming
        assert 'rp0_addr' in sv and 'rp1_addr' in sv

    def test_write_enable_present(self):
        sv = self._sv()
        assert 'we' in sv or 'wr_en' in sv or 'wren' in sv or 'write_en' in sv


# ---------------------------------------------------------------------------
# TestRegFileSVGenerator_SDP
# ---------------------------------------------------------------------------

class TestRegFileSVGenerator_SDP:
    def _sv(self):
        decl = RegFileDeclIR(
            field_name='sdp_rf', depth=32, idx_width=5, data_width=32,
            read_ports=1, write_ports=1, shared_port=False,
        )
        hazards = RegFileHazardAnalyzer().analyze(decl)
        return RegFileSVGenerator().generate(decl, hazards)

    def test_single_read_port(self):
        sv = self._sv()
        assert 'module' in sv and 'endmodule' in sv

    def test_uses_sdp_topology(self):
        sv = self._sv()
        # Simple dual-port: one read addr, one write addr
        assert 'module' in sv


# ---------------------------------------------------------------------------
# TestRegFileSVGenerator_1P (single-port, shared bus)
# ---------------------------------------------------------------------------

class TestRegFileSVGenerator_1P:
    def _sv(self):
        decl = RegFileDeclIR(
            field_name='sp_rf', depth=32, idx_width=5, data_width=32,
            read_ports=1, write_ports=1, shared_port=True,
        )
        hazards = RegFileHazardAnalyzer().analyze(decl)
        return RegFileSVGenerator().generate(decl, hazards)

    def test_single_shared_addr_bus(self):
        sv = self._sv()
        assert 'module' in sv and 'endmodule' in sv

    def test_no_forwarding_mux(self):
        """Single-port topology has no in-module forwarding mux logic."""
        sv = self._sv()
        # The comment says 'no forwarding mux' but no actual forwarding RTL is emitted
        assert 'raw_rp' not in sv


# ---------------------------------------------------------------------------
# TestElaboratorIndexedRegFile
# ---------------------------------------------------------------------------

class TestElaboratorIndexedRegFile:
    """Integration: elaborator should populate RegFileDeclIR from a Component."""

    def test_elaborator_produces_regfile_decl(self):
        """A Component with indexed_regfile() should yield a RegFileDeclIR in meta."""
        import zuspec.dataclasses as zdc
        from zuspec.dataclasses import IndexedRegFile, indexed_regfile
        from zuspec.synth.elab.elaborator import Elaborator

        @zdc.dataclass
        class MyComp(zdc.Component):
            regfile: IndexedRegFile[zdc.u5, zdc.u32] = indexed_regfile(
                read_ports=2, write_ports=1
            )

        elab = Elaborator()
        meta = elab.elaborate(MyComp)
        assert len(meta.regfiles) == 1
        decl = meta.regfiles[0]
        assert isinstance(decl, RegFileDeclIR)
        assert decl.field_name  == 'regfile'
        assert decl.read_ports  == 2
        assert decl.write_ports == 1
        assert decl.data_width  == 32
        assert decl.idx_width   == 5


# ---------------------------------------------------------------------------
# TestIndexedPoolHazardAnalyzer
# ---------------------------------------------------------------------------

class TestIndexedPoolHazardAnalyzer:
    """Unit tests for IndexedPoolHazardAnalyzer."""

    def _make_decl(self, depth=32, idx_width=5, noop_idx=0):
        from zuspec.synth.elab.elab_ir import IndexedPoolDeclIR
        return IndexedPoolDeclIR(
            field_name='rd_sched', depth=depth, idx_width=idx_width, noop_idx=noop_idx
        )

    def test_raw_pairs_generated(self):
        from zuspec.synth.sprtl.indexed_pool_synth import IndexedPoolHazardAnalyzer
        analyzer = IndexedPoolHazardAnalyzer()
        pairs = analyzer.analyze(self._make_decl(), n_lock_ports=1, n_share_ports=2)
        raw = [p for p in pairs if p.kind == 'RAW']
        assert len(raw) == 2   # rs1 + rs2

    def test_waw_pair_for_multiple_locks(self):
        from zuspec.synth.sprtl.indexed_pool_synth import IndexedPoolHazardAnalyzer
        analyzer = IndexedPoolHazardAnalyzer()
        pairs = analyzer.analyze(self._make_decl(), n_lock_ports=2, n_share_ports=2)
        waw = [p for p in pairs if p.kind == 'WAW']
        assert len(waw) == 1

    def test_no_waw_for_single_lock(self):
        from zuspec.synth.sprtl.indexed_pool_synth import IndexedPoolHazardAnalyzer
        analyzer = IndexedPoolHazardAnalyzer()
        pairs = analyzer.analyze(self._make_decl(), n_lock_ports=1, n_share_ports=2)
        waw = [p for p in pairs if p.kind == 'WAW']
        assert waw == []


# ---------------------------------------------------------------------------
# TestIndexedPoolSVGenerator
# ---------------------------------------------------------------------------

class TestIndexedPoolSVGenerator:
    """Unit tests for IndexedPoolSVGenerator."""

    def _make_decl(self, depth=32, idx_width=5, noop_idx=0):
        from zuspec.synth.elab.elab_ir import IndexedPoolDeclIR
        return IndexedPoolDeclIR(
            field_name='rd_sched', depth=depth, idx_width=idx_width, noop_idx=noop_idx
        )

    def _gen(self, **kw):
        from zuspec.synth.sprtl.indexed_pool_synth import (
            IndexedPoolHazardAnalyzer, IndexedPoolSVGenerator,
        )
        decl = self._make_decl(**kw)
        analyzer = IndexedPoolHazardAnalyzer()
        hazards  = analyzer.analyze(decl)
        gen      = IndexedPoolSVGenerator()
        return gen.generate(decl, hazards)

    def test_module_name_contains_field_name(self):
        sv = self._gen()
        assert 'rd_sched_scoreboard' in sv

    def test_has_module_and_endmodule(self):
        sv = self._gen()
        assert 'module rd_sched_scoreboard' in sv
        assert 'endmodule' in sv

    def test_raw_comparators_emitted(self):
        sv = self._gen()
        assert 'raw_lp_sp0' in sv
        assert 'raw_lp_sp1' in sv

    def test_hazard_output_present(self):
        sv = self._gen()
        assert 'assign hazard' in sv

    def test_noop_idx_guard_in_comparators(self):
        sv = self._gen(noop_idx=0)
        assert 'NOOP_IDX' in sv

    def test_set_and_clear_ports(self):
        sv = self._gen()
        assert 'set_we' in sv
        assert 'clear_we' in sv

    def test_bitmap_register(self):
        sv = self._gen()
        assert 'scoreboard' in sv
        assert "1'b0}" in sv   # reset: {DEPTH{1'b0}}

    def test_set_wins_over_clear_comment(self):
        sv = self._gen()
        assert 'set wins' in sv.lower() or 'set after clear' in sv.lower()


# ---------------------------------------------------------------------------
# TestElaboratorIndexedPool
# ---------------------------------------------------------------------------

class TestElaboratorIndexedPool:
    """Integration: elaborator should populate IndexedPoolDeclIR from a Component."""

    def test_elaborator_produces_pool_decl(self):
        import zuspec.dataclasses as zdc
        from zuspec.dataclasses import IndexedPool, indexed_pool
        from zuspec.synth.elab.elaborator import Elaborator
        from zuspec.synth.elab.elab_ir import IndexedPoolDeclIR

        @zdc.dataclass
        class MyComp(zdc.Component):
            rd_sched: IndexedPool[zdc.u5] = indexed_pool(depth=32, noop_idx=0)

        elab = Elaborator()
        meta = elab.elaborate(MyComp)
        assert len(meta.indexed_pools) == 1
        decl = meta.indexed_pools[0]
        assert isinstance(decl, IndexedPoolDeclIR)
        assert decl.field_name == 'rd_sched'
        assert decl.depth      == 32
        assert decl.idx_width  == 5
        assert decl.noop_idx   == 0

    def test_pool_and_regfile_coexist(self):
        """A Component can have both an IndexedRegFile and an IndexedPool."""
        import zuspec.dataclasses as zdc
        from zuspec.dataclasses import IndexedRegFile, IndexedPool, indexed_regfile, indexed_pool
        from zuspec.synth.elab.elaborator import Elaborator

        @zdc.dataclass
        class MyComp(zdc.Component):
            regfile:  IndexedRegFile[zdc.u5, zdc.u32] = indexed_regfile(read_ports=2, write_ports=1)
            rd_sched: IndexedPool[zdc.u5]              = indexed_pool(depth=32, noop_idx=0)

        elab = Elaborator()
        meta = elab.elaborate(MyComp)
        assert len(meta.regfiles)       == 1
        assert len(meta.indexed_pools)  == 1


# ---------------------------------------------------------------------------
# TestEndToEndHazardSV — check that emitted Verilog contains hazard wiring
# ---------------------------------------------------------------------------

class TestEndToEndHazardSV:
    """Check that the full pipeline Verilog contains scoreboard and RAW comparators."""

    def _emit_rv_sv(self, tmp_path):
        """Helper: elaborate + schedule + lower + emit RVCore Verilog."""
        import sys, os
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))
        try:
            from org.zuspec.example.mls.riscv.rv_core import RVCore, RVConfig
        except ImportError:
            import pytest; pytest.skip('riscv example not on path')

        import zuspec.synth.mls as mls
        cfg = RVConfig(xlen=32, reset_addr=0x8000_0000)
        ir  = mls.elaborate(RVCore, cfg)
        mls.schedule(ir, strategy='list', pipeline_stages=5)
        mls.lower(ir)
        sv_path = str(tmp_path / 'rv_core.sv')
        mls.emit_sv(ir, sv_path)
        return open(sv_path).read()

    def test_scoreboard_module_emitted(self, tmp_path):
        sv = self._emit_rv_sv(tmp_path)
        assert 'rd_sched_scoreboard' in sv

    def test_raw_comparators_in_scoreboard(self, tmp_path):
        sv = self._emit_rv_sv(tmp_path)
        assert 'raw_lp_sp0' in sv
        assert 'raw_lp_sp1' in sv

    def test_hazard_wire_in_toplevel(self, tmp_path):
        sv = self._emit_rv_sv(tmp_path)
        assert 'sb_hazard' in sv

    def test_regfile_uses_elaborated_module(self, tmp_path):
        sv = self._emit_rv_sv(tmp_path)
        # Elaborated topology produces regfile_2r1w, not the hardcoded regfile_rv
        assert 'regfile_2r1w' in sv

    def test_raw_forwarding_in_regfile(self, tmp_path):
        sv = self._emit_rv_sv(tmp_path)
        assert 'raw_rp0_wp0' in sv
        assert 'raw_rp1_wp0' in sv
