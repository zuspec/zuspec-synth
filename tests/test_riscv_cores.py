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
"""Example RISC-V pipeline cores synthesised with the @zdc.pipeline constructs.

These tests serve a dual purpose:
  1. Realistic integration examples for the pipeline synthesis infrastructure.
  2. Regression tests that the full pass chain produces coherent Verilog 2005
     for common micro-architecture patterns.

Two micro-architecture variants are covered:

  ``_RiscV3Stage``
    A compressed 3-stage pipeline (ID → EX → WB) — suitable for
    low-frequency or deeply embedded applications.  Both source registers
    are read in ID; the ALU operates in EX; the result is written back to
    the register file in WB with full WB→ID forwarding on both read ports.

  ``_RiscV5Stage``
    The classic 5-stage in-order pipeline (IF → ID → EX → MEM → WB).
    Instruction decode inputs are captured in IF; register file reads happen
    in ID; the ALU runs in EX; a simplified MEM stage passes the result
    through (no external memory model is needed for synthesis); WB writes
    back to the register file.  WB→ID forwarding is provided for both read
    ports.  The extra IF→ID pipeline stage means signals travel two
    register-file boundary crossings before reaching WB, which exercises
    deeper pipeline-register chains.

Both models use a 32-entry × 32-bit ``IndexedRegFile``, matching the
RV32I integer register file.
"""

import pytest
import sys
import os

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
if "" in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

import zuspec.dataclasses as zdc
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.passes import (
    PipelineAnnotationPass,
    HazardAnalysisPass,
    ForwardingGenPass,
    StallGenPass,
    SVEmitPass,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_chain(component_cls, *, forward_default: bool = True) -> SynthIR:
    """Run the complete Approach-C pass chain and return the final SynthIR."""
    cfg = SynthConfig(forward_default=forward_default)
    ir = SynthIR()
    ir.component = component_cls
    for pass_cls in [
        PipelineAnnotationPass,
        HazardAnalysisPass,
        ForwardingGenPass,
        StallGenPass,
    ]:
        ir = pass_cls(cfg).run(ir)
    ir = SVEmitPass(cfg).run(ir)
    return ir


# ---------------------------------------------------------------------------
# 3-stage RISC-V core: ID → EX → WB
# ---------------------------------------------------------------------------

class _RiscV3Stage:
    """Minimal 3-stage RV32I-style integer pipeline.

    Stage layout
    ------------
    ID  — Capture instruction fields; read both source registers from
          the 32-entry regfile.
    EX  — Compute ``result = rdata1 + imm`` (I-type ALU operation).
    WB  — Write ``result`` back to ``rd`` in the regfile; drive ``out``.

    Forwarding
    ----------
    WB → ID bypass on ``rdata1`` and ``rdata2`` eliminates the single-cycle
    RAW hazard caused by reading a register that was just written in WB.
    """

    # Instruction decode fields (inputs to the pipeline)
    rs1:  zdc.u5
    rs2:  zdc.u5
    rd:   zdc.u5
    imm:  zdc.u32

    # Result output
    out:  zdc.u32

    # 32 × 32-bit integer register file (RV32I x0–x31)
    regfile: zdc.IndexedRegFile[zdc.u32, 32]

    @zdc.pipeline(
        clock="clk",
        reset="rst_n",
        stages=["ID", "EX", "WB"],
        forward=[
            zdc.forward(signal="regfile.rdata1", from_stage="WB", to_stage="ID"),
            zdc.forward(signal="regfile.rdata2", from_stage="WB", to_stage="ID"),
        ],
    )
    def execute(self):
        # ── ID stage ──────────────────────────────────────────────────────
        ID = zdc.stage()
        rs1:   zdc.u5  = self.rs1
        rs2:   zdc.u5  = self.rs2
        rd:    zdc.u5  = self.rd
        rdata1: zdc.u32 = self.regfile.read(rs1)
        rdata2: zdc.u32 = self.regfile.read(rs2)

        # ── EX stage ──────────────────────────────────────────────────────
        EX = zdc.stage()
        imm:    zdc.u32 = self.imm
        result: zdc.u32 = rdata1 + imm   # simplified: always I-type (rs1 + imm)

        # ── WB stage ──────────────────────────────────────────────────────
        WB = zdc.stage()
        self.regfile.write(rd, result)
        self.out = result


# ---------------------------------------------------------------------------
# 5-stage RISC-V core: IF → ID → EX → MEM → WB
# ---------------------------------------------------------------------------

class _RiscV5Stage:
    """Classic 5-stage in-order RV32I-style integer pipeline.

    Stage layout
    ------------
    IF  — Capture instruction-decode fields (``rs1``, ``rs2``, ``rd``,
          ``imm``).  In a real processor this stage would fetch from an
          instruction memory; here the decoded fields arrive as module
          inputs to keep the model self-contained.
    ID  — Read both source operands from the 32-entry register file.
    EX  — Compute ``result = rdata1 + imm`` (I-type ALU operation).
    MEM — Data-memory access stage.  Simplified here: passes ``result``
          through as ``wb_data`` with no external memory interface, so the
          module remains synthesisable without a memory model.
    WB  — Write ``wb_data`` back to ``rd`` in the regfile; drive ``out``.

    Forwarding
    ----------
    WB → ID bypass on ``rdata1`` and ``rdata2`` covers the longest
    RAW path (4 cycles between read and write).  In a real design the
    EX/MEM → EX path would also be needed, but those paths require
    mux-select logic based on ``rd`` matching which is modelled here by
    the regfile forwarding mechanism.
    """

    # Instruction decode inputs
    rs1:  zdc.u5
    rs2:  zdc.u5
    rd:   zdc.u5
    imm:  zdc.u32

    # Result output
    out:  zdc.u32

    # 32 × 32-bit integer register file
    regfile: zdc.IndexedRegFile[zdc.u32, 32]

    @zdc.pipeline(
        clock="clk",
        reset="rst_n",
        stages=["IF", "ID", "EX", "MEM", "WB"],
        forward=[
            zdc.forward(signal="regfile.rdata1", from_stage="WB", to_stage="ID"),
            zdc.forward(signal="regfile.rdata2", from_stage="WB", to_stage="ID"),
        ],
    )
    def execute(self):
        # ── IF stage ──────────────────────────────────────────────────────
        IF = zdc.stage()
        rs1: zdc.u5  = self.rs1
        rs2: zdc.u5  = self.rs2
        rd:  zdc.u5  = self.rd

        # ── ID stage ──────────────────────────────────────────────────────
        ID = zdc.stage()
        rdata1: zdc.u32 = self.regfile.read(rs1)
        rdata2: zdc.u32 = self.regfile.read(rs2)

        # ── EX stage ──────────────────────────────────────────────────────
        EX = zdc.stage()
        imm:    zdc.u32 = self.imm
        result: zdc.u32 = rdata1 + imm

        # ── MEM stage ─────────────────────────────────────────────────────
        MEM = zdc.stage()
        wb_data: zdc.u32 = result   # pass-through; extend here for load/store

        # ── WB stage ──────────────────────────────────────────────────────
        WB = zdc.stage()
        self.regfile.write(rd, wb_data)
        self.out = wb_data


# ===========================================================================
# Tests — 3-stage core
# ===========================================================================

class TestRiscV3Stage:
    """Synthesis correctness tests for the 3-stage RV32I pipeline."""

    @pytest.fixture(scope="class")
    def ir(self):
        return _run_chain(_RiscV3Stage)

    @pytest.fixture(scope="class")
    def sv(self, ir):
        return ir.lowered_sv.get("sv/pipeline/top", "")

    # -- Pass-chain sanity ------------------------------------------------

    def test_no_error(self, ir):
        """Pass chain completes without raising."""
        assert ir.pipeline_ir is not None

    def test_stage_count(self, ir):
        """Pipeline has exactly 3 stages: ID, EX, WB."""
        assert len(ir.pipeline_ir.stages) == 3
        names = [s.name for s in ir.pipeline_ir.stages]
        assert names == ["ID", "EX", "WB"]

    def test_two_regfile_reads_detected(self, ir):
        """Both rdata1 and rdata2 register file reads are detected."""
        reads = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "read"]
        assert len(reads) == 2
        result_vars = {a.result_var for a in reads}
        assert "rdata1" in result_vars
        assert "rdata2" in result_vars

    def test_regfile_write_detected(self, ir):
        """Exactly one register file write is detected in WB."""
        writes = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "write"]
        assert len(writes) == 1
        assert writes[0].stage == "WB"

    def test_one_regfile_decl(self, ir):
        """A single RegFileDeclInfo is built (one field, two read ports)."""
        decls = ir.pipeline_ir.regfile_decls
        assert len(decls) == 1
        d = decls[0]
        assert d.field_name == "regfile"
        assert d.depth == 32
        assert d.data_width == 32

    def test_regfile_hazards_detected(self, ir):
        """RAW hazards between WB-write and ID-reads are detected."""
        assert len(ir.pipeline_ir.regfile_hazards) >= 1

    def test_forwarding_resolved(self, ir):
        """All regfile hazards are resolved (by forwarding)."""
        for h in ir.pipeline_ir.regfile_hazards:
            assert h.resolved_by is not None

    # -- Structural SV checks ---------------------------------------------

    def test_sv_emitted(self, sv):
        """SV output is non-empty."""
        assert len(sv) > 100

    def test_sv_module_name(self, sv):
        """Module is named after the component class."""
        assert "module _RiscV3Stage" in sv

    def test_sv_endmodule(self, sv):
        """SV ends with endmodule."""
        assert "endmodule" in sv

    def test_sv_regfile_array(self, sv):
        """A 32×32-bit register file array is declared."""
        assert "regfile_mem" in sv
        assert "[31:0]" in sv
        assert "[0:31]" in sv

    def test_sv_two_read_muxes(self, sv):
        """Both rdata1 and rdata2 read-bypass muxes are present."""
        assert "rdata1" in sv
        assert "rdata2" in sv

    def test_sv_clocked_write(self, sv):
        """Clocked write port is emitted."""
        assert "regfile_mem[" in sv
        assert "<=" in sv

    def test_sv_valid_chain(self, sv):
        """ID, EX, WB valid registers are present."""
        assert "id_valid_q" in sv
        assert "ex_valid_q" in sv
        assert "wb_valid_q" in sv

    def test_sv_pipeline_registers(self, sv):
        """Inter-stage pipeline registers connect ID→EX and EX→WB."""
        # rdata1 crosses from ID to EX
        assert "id_to_ex" in sv
        # result crosses from EX to WB
        assert "ex_to_wb" in sv

    def test_sv_no_regfile_port(self, sv):
        """'regfile' does not appear as a module port."""
        # Ports are in the module header before the first ');\n'
        header = sv.split(");")[0]
        # The word 'regfile' should not appear in the port list
        assert "regfile" not in header

    def test_sv_clock_reset_ports(self, sv):
        """Clock and reset ports are present."""
        assert "input  wire clk" in sv
        assert "input  wire rst_n" in sv

    def test_sv_instruction_input_ports(self, sv):
        """rs1, rs2, rd, imm appear as input ports."""
        assert "input  wire [4:0] rs1" in sv
        assert "input  wire [4:0] rs2" in sv
        assert "input  wire [4:0] rd" in sv
        assert "input  wire [31:0] imm" in sv

    def test_sv_output_port(self, sv):
        """out appears as an output port."""
        assert "output wire [31:0] out" in sv or "output reg  [31:0] out" in sv


# ===========================================================================
# Tests — 5-stage core
# ===========================================================================

class TestRiscV5Stage:
    """Synthesis correctness tests for the 5-stage RV32I pipeline."""

    @pytest.fixture(scope="class")
    def ir(self):
        return _run_chain(_RiscV5Stage)

    @pytest.fixture(scope="class")
    def sv(self, ir):
        return ir.lowered_sv.get("sv/pipeline/top", "")

    # -- Pass-chain sanity ------------------------------------------------

    def test_no_error(self, ir):
        """Pass chain completes without raising."""
        assert ir.pipeline_ir is not None

    def test_stage_count(self, ir):
        """Pipeline has exactly 5 stages: IF, ID, EX, MEM, WB."""
        assert len(ir.pipeline_ir.stages) == 5
        names = [s.name for s in ir.pipeline_ir.stages]
        assert names == ["IF", "ID", "EX", "MEM", "WB"]

    def test_two_regfile_reads_in_id(self, ir):
        """Both register file reads are in the ID stage."""
        reads = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "read"]
        assert len(reads) == 2
        for r in reads:
            assert r.stage == "ID"

    def test_regfile_write_in_wb(self, ir):
        """Register file write is in WB."""
        writes = [a for a in ir.pipeline_ir.regfile_accesses if a.kind == "write"]
        assert len(writes) == 1
        assert writes[0].stage == "WB"

    def test_one_regfile_decl(self, ir):
        """Single RegFileDeclInfo for the 32-entry × 32-bit regfile."""
        decls = ir.pipeline_ir.regfile_decls
        assert len(decls) == 1
        assert decls[0].depth == 32
        assert decls[0].data_width == 32

    def test_regfile_hazards_resolved(self, ir):
        """All regfile hazards are resolved."""
        for h in ir.pipeline_ir.regfile_hazards:
            assert h.resolved_by is not None

    # -- Structural SV checks ---------------------------------------------

    def test_sv_emitted(self, sv):
        assert len(sv) > 100

    def test_sv_module_name(self, sv):
        assert "module _RiscV5Stage" in sv

    def test_sv_endmodule(self, sv):
        assert "endmodule" in sv

    def test_sv_regfile_array(self, sv):
        """32-entry × 32-bit array."""
        assert "regfile_mem" in sv
        assert "[31:0]" in sv
        assert "[0:31]" in sv

    def test_sv_two_read_muxes(self, sv):
        assert "rdata1" in sv
        assert "rdata2" in sv

    def test_sv_valid_chain_all_stages(self, sv):
        """Valid registers for all 5 stages."""
        for stage in ("if", "id", "ex", "mem", "wb"):
            assert f"{stage}_valid_q" in sv

    def test_sv_deep_pipeline_registers(self, sv):
        """Signals cross more stage boundaries (IF→ID, ID→EX, EX→MEM, MEM→WB)."""
        # rs1 must cross IF→ID boundary
        assert "if_to_id" in sv
        # rdata1 crosses ID→EX
        assert "id_to_ex" in sv
        # result crosses EX→MEM
        assert "ex_to_mem" in sv
        # wb_data crosses MEM→WB
        assert "mem_to_wb" in sv

    def test_sv_no_regfile_port(self, sv):
        """regfile field is not a module port."""
        header = sv.split(");")[0]
        assert "regfile" not in header

    def test_sv_clock_reset_ports(self, sv):
        assert "input  wire clk" in sv
        assert "input  wire rst_n" in sv

    def test_sv_instruction_input_ports(self, sv):
        assert "input  wire [4:0] rs1" in sv
        assert "input  wire [4:0] rs2" in sv
        assert "input  wire [4:0] rd" in sv
        assert "input  wire [31:0] imm" in sv

    def test_sv_output_port(self, sv):
        assert "output wire [31:0] out" in sv or "output reg  [31:0] out" in sv

    def test_sv_mem_stage_passthrough(self, sv):
        """wb_data signal (MEM pass-through) appears in the SV."""
        assert "wb_data" in sv
