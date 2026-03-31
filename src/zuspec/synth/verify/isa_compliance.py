"""
Structural ISA compliance check (Phase 3 stub).

Rather than symbolically executing all instruction encodings, this pass
performs a *structural* check: it verifies that the execution-unit instances
present in ``ComponentSynthMeta`` cover all ISA extensions declared in the
config.

Mapping:
  ALUUnit     → base integer instructions (always required)
  MulDivUnit  → M extension (integer multiply/divide)
  FPUnit      → F/D extension (single/double-precision FP)
  LoadStoreUnit → load/store instructions (always required)

If the config claims an extension (e.g. ``has_M=True``) but the corresponding
execution unit is absent from the instances list, the group is added to
``uncovered`` and the result is ``"PARTIAL"``.  If all required groups are
covered the result is ``"PASS"``.

Full symbolic ISA compliance verification (exhaustive encoding coverage) is
deferred to a future Phase 4 item.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from ..elab.elab_ir import ComponentSynthMeta

# Maps (config_attr, required_value) -> (unit_type_name, isa_group_label)
_EXTENSION_CHECKS: List[Tuple[str, Any, str, str]] = [
    ("has_M", True,  "MulDivUnit",   "RV_M_mul_div"),
    ("has_F", True,  "FPUnit",       "RV_F_float"),
    # Base integer + load/store are always expected (no config gate)
]

_ALWAYS_REQUIRED: List[Tuple[str, str]] = [
    ("ALUUnit",       "RV_I_alu"),
    ("LoadStoreUnit", "RV_I_load_store"),
]


def check_isa_compliance(
    meta: ComponentSynthMeta,
    config: Any,
) -> Tuple[str, List[str]]:
    """Return (result, uncovered_groups).

    ``result`` is ``"PASS"`` when all expected groups are covered, or
    ``"PARTIAL"`` when one or more groups are missing.
    """
    instance_type_names = {inst.comp_type.__name__ for inst in meta.instances}
    uncovered: List[str] = []

    # Always-required units
    for unit_name, group_label in _ALWAYS_REQUIRED:
        if unit_name not in instance_type_names:
            uncovered.append(group_label)

    # Extension-gated units
    for attr, expected_val, unit_name, group_label in _EXTENSION_CHECKS:
        if getattr(config, attr, None) == expected_val:
            if unit_name not in instance_type_names:
                uncovered.append(group_label)

    result = "PASS" if not uncovered else "PARTIAL"
    return result, uncovered
