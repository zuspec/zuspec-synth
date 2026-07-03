"""Configuration + error type for the transparent SPL→RTL lowering."""
from __future__ import annotations

import dataclasses as dc


@dc.dataclass
class SplConfig:
    """Options for the transparent SPL→RTL lowering.

    Attributes
    ----------
    reset_style:
        One of ``async_high`` (default — active-high, the std convention),
        ``sync_high``, ``sync_low``, ``async_low``.  Only the polarity is used
        today (be-sv emits a synchronous sensitivity list).
    """

    reset_style: str = "async_high"


class SplLowerError(Exception):
    """Raised when an SPL component cannot be lowered (unsupported/ill-formed)."""
