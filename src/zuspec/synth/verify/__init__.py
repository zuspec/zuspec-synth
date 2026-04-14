# zuspec.synth.verify — Verification passes
from .deadlock import check_deadlock_freedom, DeadlockDiagnostic
from .structural import (
    run_all_checks,
    check_channel_width_consistency,
    check_forwarding_completeness,
    check_stall_cond_nontrivial,
    check_regfile_addr_width,
    StructuralError,
)
from .verilog_props import VerilogPropertyWrapper
from .sby_gen import generate_sby

__all__ = [
    "check_deadlock_freedom",
    "DeadlockDiagnostic",
    "run_all_checks",
    "check_channel_width_consistency",
    "check_forwarding_completeness",
    "check_stall_cond_nontrivial",
    "check_regfile_addr_width",
    "StructuralError",
    "VerilogPropertyWrapper",
    "generate_sby",
]
