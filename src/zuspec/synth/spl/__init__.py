"""zuspec.synth.spl — transparent, language-independent SPL→RTL lowering.

The shared "no-magic" lowering: takes a bound SPL ``zuspec.ir.core``
``DataTypeComponent`` (awaited-beat coroutine + tagged pins) and produces an
RTL ``DataTypeComponent`` (clocked ``sync_processes`` FSM) that ``zuspec.be.sv``
netlists.  Any front end producing the SPL IR contract (fw-hdl SV,
zuspec-dataclasses Python, …) drives the same code.

1 awaited beat = 1 state = 1 cycle; no scheduler, no inference.
"""
from .config import SplConfig, SplLowerError
from .lower import lower_component, lower_context

__all__ = ["SplConfig", "SplLowerError", "lower_component", "lower_context"]
