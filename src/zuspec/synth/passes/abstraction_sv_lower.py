"""AbstractionSVLowerPass — interface-dispatch SV lowering pass.

Provides two entry points:

``run_abstraction_sv_lower(component_ir)``
    Lightweight function that rewrites proc bodies and stores SV module/instance
    text.  Runs *before* ``SynthIR`` is constructed (used by the synthesis
    pipeline in ``__init__.py``).

``AbstractionSVEmitPass``
    ``SynthPass`` subclass for use inside a ``ProtocolSynthPipeline``.  Reads
    ``ir.model_context``, iterates ``AbstractionFieldIR`` fields, rewrites proc
    bodies, stores SV text in ``ir.lowered_sv``, and populates ``ir.queue_nodes``
    for Queue fields so that ``ProtocolSVEmitPass`` can emit FIFO modules with the
    correct ``module_prefix``.
"""
from __future__ import annotations

import logging
from typing import Any

from zuspec.ir.core.abstraction_field_ir import AbstractionFieldIR

_log = logging.getLogger(__name__)


def run_abstraction_sv_lower(component_ir, parent_prefix: str = "") -> None:
    """Lower all ``AbstractionFieldIR`` fields in *component_ir* in place.

    Parameters
    ----------
    component_ir:
        ``DataTypeComponent`` IR node produced by ``DataModelFactory``.
        Its ``fields`` list is mutated in place (proc bodies rewritten).
    parent_prefix:
        Optional string prefix used when building hierarchical SV identifiers,
        e.g. ``"u_top."``.  Passed through to ``sv_instance_text()``.

    Returns
    -------
    None
        The pass mutates *component_ir* in place and returns nothing.
    """
    try:
        from zuspec.ir.core.registry import global_registry
        registry = global_registry()
    except ImportError:
        _log.debug("AbstractionSVLowerPass: zuspec-ir-core not available; skipping")
        return

    abstraction_fields = [
        (idx, f)
        for idx, f in enumerate(component_ir.fields)
        if isinstance(f, AbstractionFieldIR)
    ]

    if not abstraction_fields:
        return

    for _idx, field_ir in abstraction_fields:
        model = registry.get_sv_model(field_ir.spec_type_name)
        if model is None:
            _log.warning(
                "AbstractionSVLowerPass: no SV model registered for %r; skipping",
                field_ir.spec_type_name,
            )
            continue

        # Rewrite proc bodies
        for proc in getattr(component_ir, "proc_processes", []):
            try:
                new_body = model.rewrite_proc_stmts(proc.body, field_ir)
                proc.body.clear()
                proc.body.extend(new_body)
            except NotImplementedError:
                raise  # propagate synthesis-blocking errors
            except Exception as exc:
                _log.warning(
                    "AbstractionSVLowerPass: rewrite_proc_stmts failed for %r: %s",
                    field_ir.field_name, exc,
                )

        # Collect any standalone module/instance text — store on component_ir if
        # it has a lowered_sv dict-like attribute (only present on SynthIR, not
        # DataTypeComponent).  Store as list[str] for compatibility with assembler.
        lowered_sv = getattr(component_ir, "lowered_sv", None)
        if lowered_sv is not None:
            mod_text = model.sv_module_text(field_ir)
            if mod_text:
                lowered_sv[f"sv/module/{field_ir.field_name}"] = [mod_text]

            inst_text = model.sv_instance_text(field_ir, parent_prefix)
            if inst_text:
                lowered_sv[f"sv/instance/{field_ir.field_name}"] = [inst_text]

        _log.debug(
            "AbstractionSVLowerPass: lowered abstraction field %r (%s)",
            field_ir.field_name, field_ir.spec_type_name,
        )


# ---------------------------------------------------------------------------
# AbstractionSVEmitPass — SynthPass wrapper for ProtocolSynthPipeline
# ---------------------------------------------------------------------------

class AbstractionSVEmitPass:
    """Emit SV for all ``AbstractionFieldIR`` fields in the top-level component.

    Designed for use inside ``ProtocolSynthPipeline`` as a replacement for
    ``QueueLowerPass``.  For each ``AbstractionFieldIR`` in the component:

    * Rewrites proc bodies via ``model.rewrite_proc_stmts()``.
    * Stores ``[sv_module_text()]`` in ``ir.lowered_sv["sv/module/<field>"]``.
    * Stores ``[sv_instance_text()]`` in ``ir.lowered_sv["sv/instance/<field>"]``.
    * For Queue fields: also appends a ``QueueIR`` to ``ir.queue_nodes`` so that
      ``ProtocolSVEmitPass`` can emit the FIFO module with the correct
      ``module_prefix``.
    """

    def __init__(self, config=None) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "abstraction_sv_emit"

    def run(self, ir):
        ctxt = ir.model_context
        if ctxt is None:
            return ir

        comp_cls = ir.component
        if comp_cls is None:
            return ir

        # Find the DataTypeComponent for the top-level component class.
        from zuspec.dataclasses import ir as zdc_ir
        comp_dtype = None
        for dtype in ctxt.type_m.values():
            if isinstance(dtype, zdc_ir.DataTypeComponent) and getattr(dtype, "py_type", None) is comp_cls:
                comp_dtype = dtype
                break
        if comp_dtype is None:
            comp_names = {getattr(comp_cls, "__name__", None), getattr(comp_cls, "__qualname__", None)}
            for name, dtype in ctxt.type_m.items():
                if name in comp_names and isinstance(dtype, zdc_ir.DataTypeComponent):
                    comp_dtype = dtype
                    break

        if comp_dtype is None:
            return ir

        try:
            from zuspec.ir.core.registry import global_registry
            from zuspec.dataclasses.queue_type import Queue as _Queue
            from zuspec.synth.ir.protocol_ir import QueueIR
            registry = global_registry()
        except ImportError:
            return ir

        for field in getattr(comp_dtype, "fields", []):
            if not isinstance(field, AbstractionFieldIR):
                continue

            model = registry.get_sv_model(field.spec_type_name)
            if model is None:
                _log.warning(
                    "AbstractionSVEmitPass: no SV model for %r; skipping",
                    field.spec_type_name,
                )
                continue

            # Rewrite proc bodies
            for proc in getattr(comp_dtype, "proc_processes", []):
                try:
                    new_body = model.rewrite_proc_stmts(proc.body, field)
                    proc.body.clear()
                    proc.body.extend(new_body)
                except NotImplementedError:
                    raise
                except Exception as exc:
                    _log.warning(
                        "AbstractionSVEmitPass: rewrite_proc_stmts failed for %r: %s",
                        field.field_name, exc,
                    )

            is_queue = getattr(field, "py_cls", None) is _Queue

            if is_queue:
                # Queue SV emission is handled by ProtocolSVEmitPass._emit_queues()
                # via queue_nodes.  Don't also store in lowered_sv or we get duplicates.
                n = field.ir_node
                node = QueueIR(name=n["name"], elem_width=n["elem_width"], depth=n["depth"])
                ir.queue_nodes.append(node)
                _log.debug(
                    "AbstractionSVEmitPass: %s → QueueIR(w=%d, d=%d)",
                    field.field_name, n["elem_width"], n["depth"],
                )
            else:
                # Non-Queue abstraction fields: store SV module/instance text
                mod_text = model.sv_module_text(field)
                if mod_text:
                    ir.lowered_sv[f"sv/module/{field.field_name}"] = [mod_text]

                inst_text = model.sv_instance_text(field, "")
                if inst_text:
                    ir.lowered_sv[f"sv/instance/{field.field_name}"] = [inst_text]

            _log.debug(
                "AbstractionSVEmitPass: emitted SV for %r (%s)",
                field.field_name, field.spec_type_name,
            )

        return ir
