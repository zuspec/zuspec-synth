"""
Type lowering: zuspec types → SystemVerilog logic[N-1:0].
"""
from __future__ import annotations
from typing import Any, Optional
import typing


def _get_annotated_width(hint) -> Optional[int]:
    """Extract bit width from Annotated[int, U(N)] / Annotated[int, S(N)] hints."""
    if hint is None:
        return None
    origin = typing.get_origin(hint)
    if origin is typing.Annotated:
        args = typing.get_args(hint)
        for meta in args[1:]:
            if hasattr(meta, 'width'):
                return meta.width
    return None


def lower_type(hint, default_width: int = 32) -> int:
    """Return the bit width for a given type hint.

    Rules:
      - Annotated[int, U(N)] / Annotated[int, S(N)] → N
      - bool / bit                                    → 1
      - int (bare)                                    → default_width
      - None                                          → default_width
    """
    if hint is None:
        return default_width
    if hint is bool:
        return 1
    w = _get_annotated_width(hint)
    if w is not None:
        return w
    if hint is int:
        return default_width
    return default_width


def sv_type_str(width: int) -> str:
    """Format 'logic' or 'logic [N-1:0]'."""
    if width <= 1:
        return 'logic'
    return f'logic [{width - 1}:0]'
