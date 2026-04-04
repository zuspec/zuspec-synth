"""SynthPass — specialises the generic Pass contract to SynthIR."""
from __future__ import annotations

from zuspec.dataclasses.transform.pass_ import Pass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR


class SynthPass(Pass):
    """A ``Pass`` specialised to operate on :py:class:`SynthIR`.

    Every synthesis pass receives a :py:class:`SynthConfig` at construction so
    it has access to ISA-level settings (XLEN, reset address, etc.) without
    relying on global state.

    Args:
        config: Synthesis configuration for this pass.
    """

    def __init__(self, config: SynthConfig) -> None:
        self._config = config

    @property
    def config(self) -> SynthConfig:
        """The synthesis configuration supplied at construction."""
        return self._config
