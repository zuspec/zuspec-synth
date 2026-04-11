"""AutoThreadPass — insert auto-threading channels for variables that skip stages.

After :class:`~zuspec.synth.passes.pipeline_frontend.PipelineFrontendPass` has
built the initial ``PipelineIR``, some variables are produced at stage k and
consumed at stage m where ``m > k + 1``.  These variables must be carried
through the intermediate stages ``k+1 … m-1`` via pipeline registers (even
though those stages do not use the value).

This pass inserts the necessary threading
:class:`~zuspec.synth.ir.pipeline_ir.ChannelDecl` entries and updates
``StageIR.inputs`` / ``StageIR.outputs`` for the intermediate stages.

Algorithm
---------
For each variable ``v`` with a direct channel from stage ``k`` to stage ``m``:

1. If ``m == k + 1``, no threading needed (already adjacent).
2. For each intermediate stage ``j ∈ [k+1, m-1]``:
   a. Create channel ``{v}_thru_{stages[j].name.lower()}`` from
      ``stages[j-1]`` to ``stages[j+1]``.
   b. Append this channel to ``stages[j].inputs`` and ``stages[j].outputs``.
   c. Add all threading channels to ``pip.channels``.

Threading channels have the same ``width`` as the original channel.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

from .synth_pass import SynthPass
from zuspec.synth.ir.pipeline_ir import ChannelDecl, PipelineIR, StageIR
from zuspec.synth.ir.synth_ir import SynthIR

_log = logging.getLogger(__name__)


class AutoThreadPass(SynthPass):
    """Insert threading pipeline registers for variables skipping stages.

    This pass is a no-op when ``ir.pipeline_ir`` is ``None``.
    """

    @property
    def name(self) -> str:
        return "auto_thread"

    def run(self, ir: SynthIR) -> SynthIR:
        """Insert threading channels into ``ir.pipeline_ir``.

        :param ir: Synthesis IR with ``ir.pipeline_ir`` set by
                   :class:`PipelineFrontendPass`.
        :type ir: SynthIR
        :return: Updated IR with threading channels added.
        :rtype: SynthIR
        """
        if ir.pipeline_ir is None:
            _log.debug("[AutoThreadPass] no pipeline_ir — skipping")
            return ir

        pip = ir.pipeline_ir
        stage_idx: Dict[str, int] = {s.name: s.index for s in pip.stages}

        new_channels: List[ChannelDecl] = []

        # Process each existing channel; skip adjacent ones
        for ch in list(pip.channels):
            src_idx = stage_idx.get(ch.src_stage)
            dst_idx = stage_idx.get(ch.dst_stage)
            if src_idx is None or dst_idx is None:
                continue
            gap = dst_idx - src_idx
            if gap <= 1:
                continue  # adjacent — no threading needed

            # Extract the variable name: ch.name ends with _{src}_{to}_{dst}
            # strip suffix to get base variable name
            suffix = f"_{ch.src_stage.lower()}_to_{ch.dst_stage.lower()}"
            vname = ch.name[: -len(suffix)] if ch.name.endswith(suffix) else ch.name

            _log.debug("[AutoThreadPass] threading '%s' from %s→%s through %d stage(s)",
                       vname, ch.src_stage, ch.dst_stage, gap - 1)

            prev_stage = pip.stages[src_idx]
            for j in range(src_idx + 1, dst_idx):
                thru_stage = pip.stages[j]
                next_stage = pip.stages[j + 1] if j + 1 < len(pip.stages) else None

                thru_name = f"{vname}_thru_{thru_stage.name.lower()}"
                thru_ch = ChannelDecl(
                    name=thru_name,
                    width=ch.width,
                    depth=1,
                    src_stage=prev_stage.name,
                    dst_stage=thru_stage.name if next_stage is None else next_stage.name,
                )

                # The threading register sits between prev_stage and thru_stage:
                # prev_stage outputs the threading value, thru_stage inputs it.
                in_ch = ChannelDecl(
                    name=thru_name,
                    width=ch.width,
                    depth=1,
                    src_stage=prev_stage.name,
                    dst_stage=thru_stage.name,
                )
                out_ch = ChannelDecl(
                    name=thru_name,
                    width=ch.width,
                    depth=1,
                    src_stage=thru_stage.name,
                    dst_stage=ch.dst_stage,
                )

                thru_stage.inputs.append(in_ch)
                thru_stage.outputs.append(out_ch)

                new_channels.append(in_ch)
                prev_stage = thru_stage

        pip.channels.extend(new_channels)

        if new_channels:
            _log.info("[AutoThreadPass] inserted %d threading channel(s) in %s",
                      len(new_channels), pip.module_name)

        return ir
