#****************************************************************************
# Copyright 2019-2026 Matthew Ballance and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#****************************************************************************
"""Tests for PipelineIR data model."""

import pytest
import sys
import os

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, "..", "src")
_dc_src = os.path.join(_this_dir, "..", "..", "zuspec-dataclasses", "src")
if "" in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

from zuspec.synth.ir.pipeline_ir import (
    ChannelDecl,
    ForwardingDecl,
    HazardPair,
    PipelineIR,
    StageIR,
)


class TestChannelDecl:
    def test_defaults(self):
        ch = ChannelDecl(name="x", width=32, depth=1, src_stage="IF", dst_stage="EX")
        assert ch.name == "x"
        assert ch.width == 32
        assert ch.src_stage == "IF"
        assert ch.dst_stage == "EX"

    def test_depth_field(self):
        ch = ChannelDecl(name="y", width=32, depth=1, src_stage="EX", dst_stage="WB")
        assert ch.depth == 1


class TestForwardingDecl:
    def test_enabled_forward(self):
        f = ForwardingDecl(signal="res", from_stage="EX", to_stage="ID")
        assert not f.suppressed  # default False = forward enabled

    def test_suppressed_forward(self):
        f = ForwardingDecl(signal="flags", from_stage="WB", to_stage="EX",
                           suppressed=True)
        assert f.suppressed


class TestHazardPair:
    def test_raw(self):
        h = HazardPair(kind="RAW", signal="a", producer_stage="EX",
                       consumer_stage="ID")
        assert h.kind == "RAW"

    def test_all_kinds(self):
        for kind in ("RAW", "WAW", "WAR"):
            h = HazardPair(kind=kind, signal="x", producer_stage="EX",
                           consumer_stage="ID")
            assert h.kind == kind


class TestStageIR:
    def test_defaults(self):
        s = StageIR(name="IF", index=0, inputs=[], outputs=[], ports=[])
        assert s.name == "IF"
        assert s.index == 0
        assert s.operations == []
        assert s.cycle_lo == 0
        assert s.cycle_hi == 0

    def test_operations_list(self):
        s = StageIR(name="EX", index=1, inputs=[], outputs=[], ports=[])
        s.operations.append("some_stmt")
        assert len(s.operations) == 1


class TestPipelineIR:
    def test_minimal(self):
        pip = PipelineIR(
            module_name="alu",
            stages=[],
            channels=[],
            meta=None,
            pipeline_stages=0,
        )
        assert pip.module_name == "alu"
        assert pip.stages == []
        assert pip.channels == []
        assert pip.forwarding == []
        assert pip.hazards == []

    def test_add_stages(self):
        stages = [StageIR(name=n, index=i, inputs=[], outputs=[], ports=[])
                  for i, n in enumerate(("IF", "ID", "EX", "MEM", "WB"))]
        pip = PipelineIR(
            module_name="cpu", stages=stages, channels=[], meta=None,
            pipeline_stages=5,
        )
        assert len(pip.stages) == 5
        assert pip.stages[2].name == "EX"

    def test_approach_default(self):
        pip = PipelineIR(module_name="p", stages=[], channels=[], meta=None,
                         pipeline_stages=0)
        assert pip.approach in ("A", "C", "sdc", "")

    def test_initiation_interval_default(self):
        pip = PipelineIR(module_name="p", stages=[], channels=[], meta=None,
                         pipeline_stages=0)
        assert pip.initiation_interval == 1
