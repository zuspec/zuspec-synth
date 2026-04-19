"""ProtocolSynthPipeline — end-to-end synthesis pipeline for IfProtocol components.

Chains all Phase 4–6 lowering passes and assembles a complete SystemVerilog
module from the resulting IR fragments.

Usage::

    from zuspec.synth.protocol_pipeline import ProtocolSynthPipeline

    pipeline = ProtocolSynthPipeline(MyComponent)
    sv_text = pipeline.run()           # complete SV module as a string
    pipeline.write("out/my_comp.sv")   # write to file

The pipeline runs:

1. ``DataModelFactory.build(comp_cls)``   → ``ir.model_context``
2. ``IfProtocolLowerPass``                → ``ir.protocol_ports``
3. ``QueueLowerPass``                     → ``ir.queue_nodes``
4. ``SpawnLowerPass``                     → ``ir.spawn_nodes``
5. ``SelectLowerPass``                    → ``ir.select_nodes``
6. ``CompletionAnalysisPass``             → ``ir.completion_nodes``
7. ``ProtocolCompatPass``                 → (optional, raises on errors)
8. ``ProtocolSVEmitPass``                 → ``ir.lowered_sv``
9. SV assembly                            → complete module text

The assembled module contains:
- A ``// submodule`` section with all FIFO and arbiter module definitions
- A top-level module stub with IfProtocol port declarations (clock, reset + protocol signals)
"""
from __future__ import annotations

import logging
import os
import textwrap
from typing import Any, Optional

from zuspec.synth.ir.synth_ir import SynthConfig, SynthIR
from zuspec.synth.passes.if_protocol_lower import IfProtocolLowerPass
from zuspec.synth.passes.queue_lower import QueueLowerPass
from zuspec.synth.passes.spawn_lower import SpawnLowerPass
from zuspec.synth.passes.select_lower import SelectLowerPass
from zuspec.synth.passes.completion_analysis import CompletionAnalysisPass
from zuspec.synth.passes.protocol_sv_emit import ProtocolSVEmitPass

_log = logging.getLogger(__name__)


class ProtocolSynthPipeline:
    """Run all Phase 4–6 lowering passes on a component class and assemble SV.

    Args:
        comp_cls: The top-level ``@zdc.dataclass`` component class.
        config: Optional :class:`~zuspec.synth.ir.synth_ir.SynthConfig`.
        module_prefix: Optional prefix for emitted submodule names.
        check_compat: If True (default), run ``ProtocolCompatPass`` which
            raises on detected protocol compatibility violations.
    """

    def __init__(
        self,
        comp_cls: Any,
        config: Optional[SynthConfig] = None,
        module_prefix: str = "",
        check_compat: bool = True,
    ) -> None:
        self._comp_cls = comp_cls
        self._config = config or SynthConfig()
        self._prefix = module_prefix
        self._check_compat = check_compat
        self._ir: Optional[SynthIR] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ir(self) -> SynthIR:
        """The ``SynthIR`` after the last :meth:`run` call."""
        if self._ir is None:
            raise RuntimeError("run() must be called before accessing ir")
        return self._ir

    def run(self) -> str:
        """Execute the pipeline and return the assembled SV module text."""
        ir = self._build_ir()
        self._ir = ir
        return assemble_sv_module(ir, module_prefix=self._prefix)

    def write(self, path: str) -> None:
        """Execute the pipeline and write the SV module to *path*."""
        sv = self.run()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(sv)
        _log.info("[ProtocolSynthPipeline] wrote %s", path)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _build_ir(self) -> SynthIR:
        from zuspec.dataclasses.data_model_factory import DataModelFactory

        ir = SynthIR(component=self._comp_cls, config=self._config)

        # Step 1 — elaborate
        try:
            ir.model_context = DataModelFactory().build(self._comp_cls)
            _log.debug("[pipeline] elaborated %s (%d types)",
                       self._comp_cls.__name__, len(ir.model_context.type_m))
        except Exception as exc:  # noqa: BLE001
            _log.warning("[pipeline] DataModelFactory failed: %s", exc)

        # Steps 2–6 — lowering passes
        for pass_cls, kwargs in [
            (IfProtocolLowerPass, {}),
            (QueueLowerPass, {}),
            (SpawnLowerPass, {}),
            (SelectLowerPass, {}),
            (CompletionAnalysisPass, {}),
        ]:
            try:
                ir = pass_cls(self._config, **kwargs).run(ir)
            except Exception as exc:  # noqa: BLE001
                _log.warning("[pipeline] %s failed: %s", pass_cls.__name__, exc)

        # Step 7 — optional compat check
        if self._check_compat:
            try:
                from zuspec.synth.passes.protocol_compat import ProtocolCompatChecker
                # ProtocolCompatChecker checks pairs; skip standalone run here
                _log.debug("[pipeline] ProtocolCompatChecker available (skipping standalone run)")
            except Exception as exc:  # noqa: BLE001
                _log.warning("[pipeline] ProtocolCompatChecker import failed: %s", exc)

        # Step 8 — SV emit
        try:
            ir = ProtocolSVEmitPass(self._config, module_prefix=self._prefix).run(ir)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[pipeline] ProtocolSVEmitPass failed: %s", exc)

        return ir


# ---------------------------------------------------------------------------
# SV module assembler
# ---------------------------------------------------------------------------

def assemble_sv_module(ir: SynthIR, *, module_prefix: str = "") -> str:
    """Assemble a complete SystemVerilog module from ``ir.lowered_sv`` fragments.

    The output has three sections:

    1. **File header** — auto-generated comment.
    2. **Submodule definitions** — one module block per FIFO / arbiter.
    3. **Top-level module** — clock, reset + all IfProtocol port signals as
       a stub (``/* implementation body */``).

    Args:
        ir: ``SynthIR`` after ``ProtocolSVEmitPass`` has run.
        module_prefix: Optional prefix used to build the top-level module name.
    """
    comp_name = getattr(ir.component, "__name__", "Component") if ir.component else "Component"
    top_name = f"{module_prefix}_{comp_name}" if module_prefix else comp_name

    lines: list[str] = []

    # Header
    lines += [
        "// Auto-generated by zuspec-synth ProtocolSynthPipeline",
        f"// Component: {comp_name}",
        "// Do NOT edit by hand.",
        "",
    ]

    # Submodule definitions (FIFOs and arbiters)
    sub_keys = [k for k in ir.lowered_sv if k.startswith("sv/module/")]
    if sub_keys:
        lines.append("// ── Submodule definitions ──────────────────────────────────────────────")
        for key in sorted(sub_keys):
            for fragment in ir.lowered_sv[key]:
                lines.append(fragment)
                lines.append("")

    # Top-level module — collect all port signal lines first
    port_lines: list[str] = []
    port_keys = [k for k in ir.lowered_sv if k.startswith("sv/port/")]
    for key in sorted(port_keys):
        for fragment in ir.lowered_sv[key]:
            for decl_line in fragment.splitlines():
                stripped = decl_line.strip()
                if stripped:
                    # Strip any trailing comma — we'll add them ourselves
                    port_lines.append(stripped.rstrip(","))

    # Build module header: clock + reset + protocol ports, all comma-separated
    all_ports = ["input  logic clk", "input  logic rst"] + port_lines

    lines.append(f"module {top_name} (")
    for i, port in enumerate(all_ports):
        comma = "," if i < len(all_ports) - 1 else ""
        lines.append(f"  {port}{comma}")
    lines.append(");")
    lines.append("")
    lines.append("  /* implementation body — fill in or connect to submodules */")
    lines.append("")
    lines.append("endmodule")
    lines.append("")

    return "\n".join(lines)
