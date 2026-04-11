"""Generate ``.sby`` configuration files for Symbiyosys formal verification.

Each generated ``.sby`` file drives ``sby`` against a single Verilog file
that has formal properties embedded via ``ifdef FORMAL`` blocks (produced by
:class:`~zuspec.synth.verify.verilog_props.VerilogPropertyWrapper`).
"""
from __future__ import annotations

from ..ir.pipeline_ir import PipelineIR


def generate_sby(
    pip: PipelineIR,
    formal_dut_path: str,
    *,
    mode: str = "prove",
    depth: int = 20,
    engine: str = "smtbmc boolector",
) -> str:
    """Return ``.sby`` file contents for formal property checking of *pip*.

    The generated script uses ``sby`` in ``prove`` mode (k-induction) by default.
    Set *mode* to ``"bmc"`` for bounded model checking only.

    A single Verilog file (*formal_dut_path*) is expected — the DUT with
    ``ifdef FORMAL`` blocks embedded by
    :class:`~zuspec.synth.verify.verilog_props.VerilogPropertyWrapper`.

    :param pip: Lowered :class:`~zuspec.synth.ir.pipeline_ir.PipelineIR`.
    :param formal_dut_path: Path to the formal-annotated DUT Verilog file.
    :param mode: sby verification mode: ``"prove"`` (k-induction) or ``"bmc"``.
    :param depth: Verification depth (BMC steps or k for induction).
    :param engine: sby engine line, e.g. ``"smtbmc boolector"`` or ``"abc pdr"``.
    :returns: Complete ``.sby`` file contents as a string.
    """
    lines = [
        "[options]",
        f"mode {mode}",
        f"depth {depth}",
        "",
        "[engines]",
        engine,
        "",
        "[script]",
        f"read -formal {formal_dut_path}",
        f"prep -top {pip.module_name}",
        "",
        "[files]",
        formal_dut_path,
    ]
    return "\n".join(lines) + "\n"
