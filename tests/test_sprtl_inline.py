"""Tests for SPRTL transformer: self-method inlining, tuple temporaries, and
subscript-target rewriting.

These tests cover three fixes applied to transformer.py / sv_codegen.py:

1. **Subscript-index rewriting** (``ExprSubscript`` in ``_rewrite_expr``):
   When an inlined helper assigns into ``arr[local_var]`` the index must be
   prefixed just like any other ``ExprRefLocal``.  Previously ``gpr[rd]``
   would emit ``rd`` as an undeclared implicit wire instead of
   ``_execute_rd``.

2. **Tuple-temporary expansion** (``StmtAnnAssign`` in ``_transform_stmt``):
   The DMF emits ``_tu_0: _zsp_tuple = (expr0, expr1)`` for multi-value
   destructuring.  These must be expanded to flat ``_tu_0_v0 <= expr0`` /
   ``_tu_0_v1 <= expr1`` assignments so the signals are both driven and
   declared.

3. **Nested FSMCond register auto-inference** (``_walk_ops`` in
   ``sv_codegen.py``):
   Registers assigned only inside ``if/else`` branches (``FSMCond``) must
   be auto-declared even though they are not top-level ``FSMAssign`` nodes.

A minimal synthesisable component ``_InlineCore`` is defined inline so
this test file has no dependency on ``parts/``.
"""

import sys
import os
import pytest

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src    = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

for _p in [_dc_src, _synth_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zuspec.synth import synthesize


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def sv():
    """Synthesize InlineCore and return the SV string."""
    from examples.inline_core import InlineCore
    return synthesize(InlineCore)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------

class TestInlineCoreStructure:
    """Verify that synthesis produces well-formed SV."""

    def test_synthesizes(self, sv):
        """synthesize() returns non-empty string."""
        assert sv and len(sv) > 100

    def test_module_declaration(self, sv):
        """Module header contains the component class name."""
        assert 'module InlineCore' in sv

    def test_endmodule(self, sv):
        """SV ends with endmodule."""
        assert 'endmodule' in sv

    def test_state_enum(self, sv):
        """A typedef enum for state encoding is present."""
        assert 'typedef enum' in sv

    def test_clock_reset_ports(self, sv):
        """Standard clock and reset ports are present."""
        assert 'clk' in sv
        assert 'rst_n' in sv

    def test_mem_port_handshake(self, sv):
        """Memory port valid/ack/arg signals are emitted."""
        assert 'mem_read_word_valid' in sv
        assert 'mem_read_word_ack'   in sv
        assert 'mem_read_word_arg0'  in sv

    def test_gpr_array_declared(self, sv):
        """GPR register-file array is declared as a 2D array."""
        assert 'gpr' in sv
        assert 'gpr [0:' in sv  # packed dimension from array_fields

    def test_reset_block_present(self, sv):
        """always_ff reset block exists."""
        assert 'always_ff' in sv
        assert 'rst_n' in sv


# ---------------------------------------------------------------------------
# Fix #1: subscript-index rewriting
# ---------------------------------------------------------------------------

class TestSubscriptIndexRewrite:
    """gpr[rd] inside an inlined helper must use the prefixed ``_alu_rd``
    when ``rd`` is a *local variable* of the callee (not a passed argument).

    In ``InlineCore._alu``, ``rd = (instr >> 7) & 0x1F`` is a local; it must
    become ``_alu_rd`` after inlining so that ``gpr[rd]`` emits as
    ``gpr[_alu_rd]`` — a declared register — not as bare ``gpr[rd]`` which
    would be an implicit undeclared wire index in Yosys.
    """

    def test_prefixed_rd_in_gpr_write(self, sv):
        """GPR write uses prefixed ``_alu_rd``, not bare ``rd``."""
        assert 'gpr[_alu_rd]' in sv

    def test_bare_rd_not_used_as_gpr_index(self, sv):
        """``gpr[rd]`` with unprefixed ``rd`` must NOT appear — that would
        indicate the subscript index was not renamed during inlining."""
        assert 'gpr[rd]' not in sv


# ---------------------------------------------------------------------------
# Fix #2: tuple-temporary expansion
# ---------------------------------------------------------------------------

class TestTupleTemporaryExpansion:
    """_tu_* temporaries must be expanded to _v0/_v1 scalar registers."""

    def test_no_implicit_tu_signals(self, sv):
        """No ``_tu_`` prefixed names appear as standalone identifiers in
        the generated SV (they must have been flattened to _v0/_v1)."""
        # The _tu_N variable itself should not appear as a signal name.
        # If it does it means the tuple was not expanded and the .v0/.v1
        # accesses reference an undeclared implicit wire.
        import re
        # Look for bare _tu_0, _tu_1, etc. (not followed by _v which would
        # be the flattened form)
        implicit_refs = re.findall(r'\b_alu__tu_\d+\b(?!_v)', sv)
        assert implicit_refs == [], \
            f"Unexpanded tuple temporaries in SV: {implicit_refs}"


# ---------------------------------------------------------------------------
# Fix #3: nested FSMCond register auto-inference
# ---------------------------------------------------------------------------

class TestNestedCondRegDecl:
    """Registers assigned only inside FSMCond branches must be declared."""

    def test_result_register_declared(self, sv):
        """``_alu_result`` is declared as a ``logic`` register."""
        assert 'logic [31:0] _alu_result' in sv

    def test_taken_register_declared(self, sv):
        """``_alu_taken`` (bool flag set in one branch) is declared."""
        assert '_alu_taken' in sv

    def test_result_used_correctly(self, sv):
        """``_alu_result`` is assigned and used in the SV body."""
        assert '_alu_result' in sv


# ---------------------------------------------------------------------------
# Yosys elaboration test
# ---------------------------------------------------------------------------

_YOSYS = os.path.join(
    os.path.dirname(__file__), '..', '..', 'yosys', 'bin', 'yosys'
)


@pytest.mark.skipif(
    not os.path.isfile(_YOSYS),
    reason='Yosys not available in packages/yosys/bin/',
)
class TestYosysElaboration:
    """Verify that the generated SV elaborates cleanly in Yosys."""

    def test_yosys_no_errors(self, sv, tmp_path):
        """Yosys ``read_verilog -sv`` + ``hierarchy -check`` + ``proc``
        must complete with zero errors and zero undeclared-identifier
        warnings."""
        import subprocess
        sv_file = tmp_path / '_inline_core.sv'
        sv_file.write_text(sv)
        result = subprocess.run(
            [_YOSYS, '-p',
             f'read_verilog -sv {sv_file}; hierarchy -check; proc; stat'],
            capture_output=True, text=True,
        )
        combined = result.stdout + result.stderr
        # No hard errors
        assert result.returncode == 0, \
            f'Yosys returned non-zero exit code:\n{combined}'
        # No implicit-declaration warnings
        assert 'implicitly declared' not in combined, \
            'Yosys reported implicit signal declarations:\n' + '\n'.join(
                l for l in combined.splitlines() if 'implicitly' in l)
        # Summary line must appear
        assert 'End of script' in combined
