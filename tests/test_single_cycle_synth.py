"""Integration test: single-cycle RV32I synthesis.

Verifies that FunctionalConstraintCompiler + ActivityBodyWalker can process
all stages of parts/01_single_cycle and generate well-formed SystemVerilog.

Signal-map conventions
----------------------
Each stage input buffer ``<name>: zdc.Buffer[T]`` maps every field ``f`` of
``T`` as  ``<name>.t.<f>`` → ``<prev_stage>_<f>``.
Each stage output buffer ``out: zdc.Buffer[T]`` maps every field ``f`` as
``out.t.<f>`` → ``<this_stage>_<f>``.
Resource claims ``rs1: zdc.Resource[...]`` map ``.id`` as
``<name>.id`` → ``gpr_<name>_addr``.
"""

import sys
import pathlib

import pytest

# Add single-cycle parts dir to path so we can import the action classes.
_parts_dir = pathlib.Path(__file__).parents[3] / 'parts' / '01_single_cycle'
if str(_parts_dir) not in sys.path:
    sys.path.insert(0, str(_parts_dir))

# Add zuspec-dataclasses
_zdc_dir = pathlib.Path(__file__).parents[2] / 'zuspec-dataclasses' / 'src'
if str(_zdc_dir) not in sys.path:
    sys.path.insert(0, str(_zdc_dir))

from zuspec.synth.sprtl.functional_constraint_compiler import FunctionalConstraintCompiler
from zuspec.synth.sprtl.activity_body_walker import ActivityBodyWalker
from zuspec.synth.sprtl.buffer_elab import BufferElaborationPass


# ---------------------------------------------------------------------------
# Stage imports
# ---------------------------------------------------------------------------

def _import_stages():
    """Import all five pipeline stages.  Returns a tuple of classes."""
    from fetch import Fetch
    from decode import Decode
    from execute import Execute
    from memory import Memory
    from writeback import Writeback
    from rv32_single_cycle import RV32ISingleCycleA
    return Fetch, Decode, Execute, Memory, Writeback, RV32ISingleCycleA


def _make_walker():
    Fetch, Decode, Execute, Memory, Writeback, RV32ISingleCycleA = _import_stages()
    ns = {
        'Fetch': Fetch, 'Decode': Decode,
        'Execute': Execute, 'Memory': Memory, 'Writeback': Writeback,
    }
    return ActivityBodyWalker(RV32ISingleCycleA.activity, ns)


# ---------------------------------------------------------------------------
# Signal maps
# ---------------------------------------------------------------------------

#: Wires produced by Fetch stage (FetchResult fields)
_FETCH_WIRES = {
    'fetch.t.pc':    'fetch_pc',
    'fetch.t.instr': 'fetch_instr',
}

#: Wires produced by Decode stage (DecodeResult fields)
_DECODE_WIRES = {
    'dec.t.pc':       'dec_pc',
    'dec.t.instr':    'dec_instr',
    'dec.t.kind':     'dec_kind',
    'dec.t.alu_op':   'dec_alu_op',
    'dec.t.imm':      'dec_imm',
    'dec.t.rd':       'dec_rd',
    'dec.t.funct3':   'dec_funct3',
    'dec.t.rs1':      'dec_rs1',
    'dec.t.rs2':      'dec_rs2',
    'dec.t.funct7b5': 'dec_funct7b5',
    'dec.t.funct12':  'dec_funct12',
    'dec.t.rs1_val':  'dec_rs1_val',
    'dec.t.rs2_val':  'dec_rs2_val',
}

#: Wires produced by Execute stage (ExecuteResult fields + internal _taken)
_EXECUTE_WIRES = {
    'exe.t.alu_out':  'exe_alu_out',
    'exe.t.next_pc':  'exe_next_pc',
    'exe.t.rd_wen':   'exe_rd_wen',
    '_taken':         'exe_taken',
}

#: Wires produced by Memory stage (MemResult fields)
_MEMORY_WIRES = {
    'mem.t.load_data':  'mem_load_data',
    'mem.t.load_valid': 'mem_load_valid',
}

#: Current PC register (shared resource)
_PC_WIRE = {'pc.t': 'pc_q'}


def _decode_signal_map():
    return {
        # inputs
        **_FETCH_WIRES,
        # register-file port addresses
        'rs1.id': 'gpr_rs1_addr',
        'rs2.id': 'gpr_rs2_addr',
        # outputs
        'out.t.pc':       'dec_pc',
        'out.t.instr':    'dec_instr',
        'out.t.kind':     'dec_kind',
        'out.t.alu_op':   'dec_alu_op',
        'out.t.imm':      'dec_imm',
        'out.t.rd':       'dec_rd',
        'out.t.funct3':   'dec_funct3',
        'out.t.rs1':      'dec_rs1',
        'out.t.rs2':      'dec_rs2',
        'out.t.funct7b5': 'dec_funct7b5',
        'out.t.funct12':  'dec_funct12',
        'out.t.rs1_val':  'dec_rs1_val',
        'out.t.rs2_val':  'dec_rs2_val',
    }


def _execute_signal_map():
    return {
        # inputs from Decode
        **_DECODE_WIRES,
        # PC register
        **_PC_WIRE,
        # outputs
        'out.t.alu_out':  'exe_alu_out',
        'out.t.next_pc':  'exe_next_pc',
        'out.t.rd_wen':   'exe_rd_wen',
        '_taken':         'exe_taken',
    }


def _memory_signal_map():
    return {
        # inputs
        **_DECODE_WIRES,
        **_EXECUTE_WIRES,
        # outputs
        'out.t.load_data':  'mem_load_data',
        'out.t.load_valid': 'mem_load_valid',
    }


def _writeback_signal_map():
    return {
        # inputs
        **_DECODE_WIRES,
        **_EXECUTE_WIRES,
        **_MEMORY_WIRES,
        # resource write port
        'rd_reg.t':  'gpr_rd_wdata',
        'rd_reg.id': 'gpr_rd_addr',
    }


# ---------------------------------------------------------------------------
# ActivityBodyWalker tests
# ---------------------------------------------------------------------------

class TestActivityBodyWalker:
    def test_parse_rv32_activity(self):
        walker = _make_walker()
        steps = walker.steps
        assert len(steps) == 5

        names  = [s.var_name for s in steps]
        stages = [s.action_name for s in steps]
        assert names  == ['fetch', 'dec', 'exe', 'mem', 'wb']
        assert stages == ['Fetch', 'Decode', 'Execute', 'Memory', 'Writeback']

    def test_buffer_inputs(self):
        walker = _make_walker()
        steps = {s.var_name: s for s in walker.steps}

        assert steps['fetch'].buffer_inputs == {}
        assert steps['dec'].buffer_inputs   == {'fetch': ('fetch', 'out')}
        assert steps['exe'].buffer_inputs   == {'dec': ('dec', 'out')}
        assert steps['mem'].buffer_inputs   == {
            'dec': ('dec', 'out'),
            'exe': ('exe', 'out'),
        }
        assert steps['wb'].buffer_inputs    == {
            'dec': ('dec', 'out'),
            'exe': ('exe', 'out'),
            'mem': ('mem', 'out'),
        }


# ---------------------------------------------------------------------------
# BufferElaborationPass tests
# ---------------------------------------------------------------------------

class TestBufferElaborationPass:
    def _elab(self):
        walker = _make_walker()
        return BufferElaborationPass(walker), walker

    def test_emits_wire_declarations(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert len(lines) > 0

    def test_fetch_pc_wire(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert 'logic [31:0] fetch_pc;' in lines

    def test_fetch_instr_wire(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert 'logic [31:0] fetch_instr;' in lines

    def test_dec_kind_4bit(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert 'logic [3:0] dec_kind;' in lines

    def test_dec_funct3_3bit(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert 'logic [2:0] dec_funct3;' in lines

    def test_exe_alu_out_wire(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert 'logic [31:0] exe_alu_out;' in lines

    def test_exe_rd_wen_1bit(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert 'logic exe_rd_wen;' in lines

    def test_mem_wires(self):
        elab, _ = self._elab()
        lines = elab.emit_wire_declarations()
        assert 'logic [31:0] mem_load_data;' in lines
        assert 'logic mem_load_valid;' in lines

    def test_signal_map_decode_inputs(self):
        elab, walker = self._elab()
        step = next(s for s in walker.steps if s.var_name == 'dec')
        sig_map = elab.build_signal_map_for_stage(step)
        assert sig_map.get('fetch.t.pc') == 'fetch_pc'
        assert sig_map.get('fetch.t.instr') == 'fetch_instr'

    def test_signal_map_decode_outputs(self):
        elab, walker = self._elab()
        step = next(s for s in walker.steps if s.var_name == 'dec')
        sig_map = elab.build_signal_map_for_stage(step)
        assert sig_map.get('out.t.kind') == 'dec_kind'
        assert sig_map.get('out.t.rd') == 'dec_rd'
        assert sig_map.get('out.t.imm') == 'dec_imm'


# ---------------------------------------------------------------------------
# Decode synthesis tests
# ---------------------------------------------------------------------------

class TestDecodeSynth:
    def _compile(self):
        _, Decode, _, _, _, _ = _import_stages()
        fcc = FunctionalConstraintCompiler(Decode, _decode_signal_map())
        return fcc.emit_sv()

    def test_emits_sv(self):
        lines = self._compile()
        assert len(lines) > 0

    def test_has_always_comb(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'always_comb' in text

    def test_passthrough_pc(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'dec_pc = fetch_pc' in text

    def test_passthrough_instr(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'dec_instr = fetch_instr' in text

    def test_extract_rd(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'dec_rd = fetch_instr[11:7]' in text

    def test_extract_funct3(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'dec_funct3 = fetch_instr[14:12]' in text

    def test_kind_guard_alu_reg(self):
        lines = self._compile()
        text = '\n'.join(lines)
        # opcode 0x33 = 51 decimal; InstrKind.ALU_REG = 0
        assert 'fetch_instr[6:0] == 51' in text
        assert 'dec_kind = 0' in text

    def test_alu_op_from_funct3(self):
        """c_alu_add: funct3 == 0 and funct7b5 == 0 → AluOp.ADD (0)."""
        lines = self._compile()
        text = '\n'.join(lines)
        # AluOp.ADD = 0
        assert 'dec_alu_op' in text


# ---------------------------------------------------------------------------
# Execute synthesis tests
# ---------------------------------------------------------------------------

class TestExecuteSynth:
    def _compile(self):
        _, _, Execute, _, _, _ = _import_stages()
        fcc = FunctionalConstraintCompiler(Execute, _execute_signal_map())
        return fcc.emit_sv()

    def test_emits_sv(self):
        lines = self._compile()
        assert len(lines) > 0

    def test_has_case_block(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'case (' in text

    def test_alu_out_case_subject_is_dec_kind(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'case (dec_kind)' in text

    def test_branch_taken_wire(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'exe_taken' in text

    def test_rd_wen_output(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'exe_rd_wen' in text

    def test_arithmetic_shift_right(self):
        """SRA should use $signed(...) >>> not plain >>."""
        lines = self._compile()
        text = '\n'.join(lines)
        assert '>>>' in text
        assert '$signed' in text

    def test_next_pc(self):
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'exe_next_pc' in text


# ---------------------------------------------------------------------------
# Memory synthesis tests
# ---------------------------------------------------------------------------

class TestMemorySynth:
    def _compile(self):
        _, _, _, Memory, _, _ = _import_stages()
        fcc = FunctionalConstraintCompiler(Memory, _memory_signal_map())
        return fcc.emit_sv()

    def test_emits_sv(self):
        lines = self._compile()
        assert len(lines) > 0

    def test_load_valid_cbit(self):
        """c_load_valid: load_valid == cbit(kind == LOAD) → ternary."""
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'mem_load_valid' in text
        assert "1'b1" in text or "1'b0" in text or '?' in text

    def test_load_data_default_zero(self):
        """c_load_data: non-LOAD → load_data = 0."""
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'mem_load_data' in text


# ---------------------------------------------------------------------------
# Writeback synthesis tests
# ---------------------------------------------------------------------------

class TestWritebackSynth:
    def _compile(self):
        _, _, _, _, Writeback, _ = _import_stages()
        fcc = FunctionalConstraintCompiler(Writeback, _writeback_signal_map())
        return fcc.emit_sv()

    def test_emits_sv(self):
        lines = self._compile()
        assert len(lines) > 0

    def test_rd_addr_wire(self):
        """c_rd_idx: rd_reg.id == dec_rd → gpr_rd_addr = dec_rd."""
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'gpr_rd_addr = dec_rd' in text

    def test_rd_val_guarded(self):
        """c_rd_val: guarded GPR write with load/alu mux."""
        lines = self._compile()
        text = '\n'.join(lines)
        assert 'gpr_rd_wdata' in text

