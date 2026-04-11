"""
Structural consistency checks on a lowered PipelineIR.

These are fast (no solver, no subprocess) and run as part of the normal pytest
suite.  They catch bugs that would otherwise produce silently wrong Verilog.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List

from ..ir.pipeline_ir import PipelineIR, RegFileHazard, ForwardingDecl
from ..passes.expr_lowerer import _get_sv_width


@dataclass
class StructuralError:
    """A structural consistency error found in a :class:`PipelineIR`.

    :param code: Short error code, e.g. ``"WIDTH_MISMATCH"``.
    :param message: Human-readable description.
    :param context: Stage/channel/field name for easy filtering (may be empty).
    """
    code: str
    message: str
    context: str = ""


def check_channel_width_consistency(pip: PipelineIR) -> List[StructuralError]:
    """Verify that every ``ChannelDecl.width`` matches the type annotation width.

    For each channel whose variable name is present in
    ``pip.annotation_map``, the stored width is compared against the
    width derived from the annotation via
    :func:`~zuspec.synth.passes.expr_lowerer._get_sv_width`.  A mismatch
    produces a :class:`StructuralError` with code ``"WIDTH_MISMATCH"``.

    :param pip: Lowered :class:`PipelineIR` to inspect.
    :returns: List of :class:`StructuralError` (empty when all widths agree).
    """
    errors: List[StructuralError] = []
    for ch in pip.channels:
        # Variable name = stem before first underscore in channel name
        var_name = ch.name.split("_")[0]
        if var_name not in pip.annotation_map:
            continue
        expected_width = _get_sv_width(pip.annotation_map[var_name])
        if ch.width != expected_width:
            errors.append(StructuralError(
                code="WIDTH_MISMATCH",
                message=(
                    f"Channel '{ch.name}' has width {ch.width} but annotation "
                    f"for '{var_name}' implies {expected_width}"
                ),
                context=ch.name,
            ))
    return errors


def check_forwarding_completeness(pip: PipelineIR) -> List[StructuralError]:
    """Verify that every regfile hazard resolved by forwarding has a matching :class:`ForwardingDecl`.

    Builds a set of ``(from_stage, to_stage, signal)`` tuples from
    ``pip.forwarding`` where ``suppressed == False``.  For each
    :class:`RegFileHazard` with ``resolved_by == "forward"`` checks that
    the corresponding declaration exists.

    :param pip: Lowered :class:`PipelineIR` to inspect.
    :returns: List of :class:`StructuralError` with code ``"FWD_MISSING"``.
    """
    errors: List[StructuralError] = []
    fwd_set = {
        (fwd.from_stage, fwd.to_stage, fwd.signal)
        for fwd in pip.forwarding
        if not fwd.suppressed
    }
    for hz in pip.regfile_hazards:
        if hz.resolved_by != "forward":
            continue
        key = (hz.write_stage, hz.read_stage, hz.write_data_var)
        if key not in fwd_set:
            errors.append(StructuralError(
                code="FWD_MISSING",
                message=(
                    f"RegFileHazard '{hz.field_name}' "
                    f"({hz.write_stage}→{hz.read_stage}) resolved_by='forward' "
                    f"but no ForwardingDecl found for signal '{hz.write_data_var}'"
                ),
                context=hz.field_name,
            ))
    return errors


def check_stall_cond_nontrivial(pip: PipelineIR) -> List[StructuralError]:
    """Verify that no ``stall_cond`` is syntactically equivalent to ``False`` / ``0``.

    A stall condition of ``False`` means the stage can never stall, which is
    almost certainly a bug in the pipeline body.

    :param pip: Lowered :class:`PipelineIR` to inspect.
    :returns: List of :class:`StructuralError` with code ``"DEAD_STALL"``.
    """
    errors: List[StructuralError] = []
    for stage in pip.stages:
        if stage.stall_cond is None:
            continue
        if (
            isinstance(stage.stall_cond, ast.Constant)
            and stage.stall_cond.value in (False, 0)
        ):
            errors.append(StructuralError(
                code="DEAD_STALL",
                message=(
                    f"Stage '{stage.name}' has a stall_cond that is "
                    f"syntactically False/0 — stage can never stall"
                ),
                context=stage.name,
            ))
    return errors


def check_regfile_addr_width(pip: PipelineIR) -> List[StructuralError]:
    """Verify that write and read address variables have matching widths in each hazard.

    When both ``write_addr_var`` and ``read_addr_var`` are present in
    ``pip.annotation_map`` and their widths differ, the hardware comparator
    will silently zero-extend or truncate, producing wrong hazard detection.

    :param pip: Lowered :class:`PipelineIR` to inspect.
    :returns: List of :class:`StructuralError` with code ``"ADDR_WIDTH_MISMATCH"``.
    """
    errors: List[StructuralError] = []
    for hz in pip.regfile_hazards:
        if hz.write_addr_var == hz.read_addr_var:
            continue
        wr_ann = pip.annotation_map.get(hz.write_addr_var)
        rd_ann = pip.annotation_map.get(hz.read_addr_var)
        if wr_ann is None or rd_ann is None:
            continue
        wr_w = _get_sv_width(wr_ann)
        rd_w = _get_sv_width(rd_ann)
        if wr_w != rd_w:
            errors.append(StructuralError(
                code="ADDR_WIDTH_MISMATCH",
                message=(
                    f"RegFileHazard '{hz.field_name}': write addr '{hz.write_addr_var}' "
                    f"width={wr_w} != read addr '{hz.read_addr_var}' width={rd_w}"
                ),
                context=hz.field_name,
            ))
    return errors


def run_all_checks(pip: PipelineIR) -> List[StructuralError]:
    """Run every structural check and return the combined error list.

    Executes :func:`check_channel_width_consistency`,
    :func:`check_forwarding_completeness`,
    :func:`check_stall_cond_nontrivial`, and
    :func:`check_regfile_addr_width` in order.

    :param pip: Lowered :class:`PipelineIR` to inspect.
    :returns: Combined list of all :class:`StructuralError` instances found.
              An empty list means the IR is structurally consistent.
    """
    errors: List[StructuralError] = []
    errors.extend(check_channel_width_consistency(pip))
    errors.extend(check_forwarding_completeness(pip))
    errors.extend(check_stall_cond_nontrivial(pip))
    errors.extend(check_regfile_addr_width(pip))
    return errors
