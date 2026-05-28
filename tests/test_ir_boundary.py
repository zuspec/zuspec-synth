"""Tests for IRBoundarySerializer / IRBoundaryDeserializer (R3)."""
import pytest
import yaml

from zuspec.synth.ir.boundary import IRBoundarySerializer, IRBoundaryDeserializer
from zuspec.synth.ir.synth_ir import SynthIR, SynthConfig
from zuspec.synth.ir.layers import IRLayer


class TestIRBoundarySerializer:
    def _make_ir(self, **kwargs) -> SynthIR:
        return SynthIR(**kwargs)

    def test_serializes_to_yaml_with_headers(self):
        ir = self._make_ir(sv_path="/tmp/out.sv")
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.STRUCTURAL)
        data = yaml.safe_load(text)
        assert "_schema_version" in data
        assert data.get("_layer") == "STRUCTURAL"

    def test_excluded_fields_not_in_yaml(self):
        class FakeComp:
            pass

        ir = self._make_ir(component=FakeComp)
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.SCHEDULED)
        data = yaml.safe_load(text)
        # Live Python class references must be excluded.
        assert "component" not in data

    def test_lowered_sv_serialized(self):
        ir = self._make_ir(lowered_sv={"sv/pipeline/top": "module Top();\nendmodule"})
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.STRUCTURAL)
        data = yaml.safe_load(text)
        assert "lowered_sv" in data
        assert "sv/pipeline/top" in data["lowered_sv"]

    def test_layer_activity_in_output(self):
        ir = self._make_ir()
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.ACTIVITY)
        data = yaml.safe_load(text)
        assert data["_layer"] == "ACTIVITY"


class TestIRBoundaryDeserializer:
    def test_deserializes_to_synth_ir(self):
        ir = SynthIR(lowered_sv={"sv/pipeline/top": "module Top();\nendmodule"})
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.STRUCTURAL)

        deser = IRBoundaryDeserializer()
        loaded, layer = deser.deserialize_synth_ir(text)
        assert isinstance(loaded, SynthIR)
        assert layer == IRLayer.STRUCTURAL

    def test_lowered_sv_preserved_after_roundtrip(self):
        ir = SynthIR(lowered_sv={"sv/pipeline/top": "module Foo();"})
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.STRUCTURAL)

        deser = IRBoundaryDeserializer()
        loaded, _ = deser.deserialize_synth_ir(text)
        assert loaded.lowered_sv.get("sv/pipeline/top") == "module Foo();"

    def test_sv_path_preserved_after_roundtrip(self):
        ir = SynthIR(sv_path="/some/path/out.sv")
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.STRUCTURAL)

        deser = IRBoundaryDeserializer()
        loaded, _ = deser.deserialize_synth_ir(text)
        assert loaded.sv_path == "/some/path/out.sv"

    def test_layer_scheduled_roundtrip(self):
        ir = SynthIR()
        ser = IRBoundarySerializer()
        text = ser.serialize_synth_ir(ir, IRLayer.SCHEDULED)
        deser = IRBoundaryDeserializer()
        _, layer = deser.deserialize_synth_ir(text)
        assert layer == IRLayer.SCHEDULED
