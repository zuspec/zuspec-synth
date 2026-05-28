"""SynthPass — specialises the generic Pass contract to SynthIR."""
from __future__ import annotations

from typing import Optional

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

    @property
    def output_layer(self) -> "Optional[IRLayer]":  # noqa: F821
        """IR layer produced by this pass, or ``None`` if it does not transition a layer.

        Override in subclasses that cross a named IR boundary.  The
        :class:`~zuspec.dataclasses.transform.pass_manager.LayeredPassManager`
        reads this property after each pass and runs the appropriate
        :class:`~zuspec.synth.verify.layer_verifiers.IRLayerVerifier`.
        """
        return None

    @staticmethod
    def propagate_loc(src: "BaseP", dst: "BaseP") -> None:  # noqa: F821
        """Copy ``src.loc`` to ``dst`` when ``src`` has a non-None location.

        A thin wrapper around :meth:`~zuspec.ir.core.base.Base.copy_loc` that
        is available as a static method on every ``SynthPass`` subclass for
        convenience::

            self.propagate_loc(source_node, new_node)
        """
        from zuspec.ir.core.base import BaseP as _BaseP
        if hasattr(dst, "copy_loc"):
            dst.copy_loc(src)  # type: ignore[union-attr]
