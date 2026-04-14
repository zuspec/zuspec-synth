"""AsyncPipelineElaboratePass — elaborate a component and extract its async pipeline IR.

This pass:
1. Builds the model context using ``DataModelFactory``.
2. Finds the ``@zdc.pipeline`` async method on the component class.
3. Runs ``AsyncPipelineFrontendPass`` to extract ``IrPipeline`` from the method AST.
4. Stores the result on ``ir.async_pipeline_ir``.
"""
from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import Optional

from zuspec.dataclasses.ir.pipeline_async import IrPipeline
from zuspec.dataclasses.ir.pipeline_async_pass import AsyncPipelineFrontendPass

from .synth_pass import SynthPass
from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR

_log = logging.getLogger(__name__)


class AsyncPipelineElaboratePass(SynthPass):
    """Elaborate component and extract async pipeline IR."""

    @property
    def name(self) -> str:
        return "AsyncPipelineElaboratePass"

    def run(self, ir: SynthIR) -> SynthIR:
        """Build model context and extract IrPipeline from the component's async pipeline method."""
        comp_cls = ir.component

        # Build model context (equivalent to ElaboratePass for unit tests)
        if comp_cls is not None and ir.model_context is None:
            try:
                from zuspec.dataclasses.data_model_factory import DataModelFactory
                ir.model_context = DataModelFactory().build(comp_cls)
                _log.debug("[AsyncPipelineElaboratePass] built model context for %s", comp_cls)
            except Exception as exc:
                _log.warning("[AsyncPipelineElaboratePass] DataModelFactory failed (%s)", exc)

        # Find and parse the @zdc.pipeline async method
        ip = self._extract_ir_pipeline(comp_cls)
        if ip is None:
            _log.warning(
                "AsyncPipelineElaboratePass: no @zdc.pipeline async method found on %s",
                comp_cls,
            )
        ir.async_pipeline_ir = ip
        return ir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_ir_pipeline(self, comp_cls) -> Optional[IrPipeline]:
        """Locate the @zdc.pipeline async method and run the frontend pass on it."""
        if comp_cls is None:
            return None

        # Find an async method with @zdc.pipeline (or @pipeline) decorator
        for name in dir(comp_cls):
            try:
                method = getattr(comp_cls, name)
            except Exception:
                continue
            if not callable(method):
                continue
            if not inspect.iscoroutinefunction(method):
                continue
            ip = self._try_parse_method(method)
            if ip is not None:
                return ip
        return None

    def _try_parse_method(self, method) -> Optional[IrPipeline]:
        """Parse one async method and return IrPipeline if it has a @zdc.pipeline decorator."""
        try:
            src = textwrap.dedent(inspect.getsource(method))
        except (OSError, TypeError):
            return None
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None
        fp = AsyncPipelineFrontendPass()
        fp.visit(tree)
        return fp.result

