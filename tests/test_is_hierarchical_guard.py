"""Tests for the _is_hierarchical() AbstractionFieldIR guard (Phase 1)."""

import pytest
from zuspec.ir.core.abstraction_field_ir import AbstractionFieldIR


def _make_abstraction_field(**overrides):
    defaults = dict(
        spec_type_name="Counter",
        field_name="cnt",
        field_index=0,
        py_cls=object,
        inst_kwargs={},
        ir_node=None,
    )
    defaults.update(overrides)
    return AbstractionFieldIR(**defaults)


def test_abstraction_field_ir_not_hierarchical():
    """_is_hierarchical must return False when a field is an AbstractionFieldIR."""
    from zuspec.synth import _is_hierarchical

    abstraction_field = _make_abstraction_field()

    # Build a minimal fake component_ir whose only field is an AbstractionFieldIR.
    class FakeComponentIR:
        fields = [abstraction_field]

    assert _is_hierarchical(FakeComponentIR(), ctx=None) is False


def test_plain_field_still_hierarchical_when_appropriate():
    """A genuine DataTypeComponent sub-field should still trigger the hierarchical path."""
    from zuspec.synth import _is_hierarchical

    # Construct a Field-like object that looks like a DataTypeComponent reference.
    class FakeDataTypeRef:
        ref_name = "SubComp"

    FakeDataTypeRef.__name__ = "DataTypeRef"

    class FakePlainField:
        datatype = FakeDataTypeRef()

    # type().__name__ trick: _is_hierarchical checks type(f).__name__ == "Field"
    FakePlainField.__name__ = "Field"

    # Build a ctx.type_m that maps "SubComp" → a DataTypeComponent-like object
    class FakeDataTypeComponent:
        pass

    FakeDataTypeComponent.__name__ = "DataTypeComponent"

    class FakeCtx:
        type_m = {"SubComp": FakeDataTypeComponent()}

    class FakeComponentIR:
        fields = [FakePlainField()]

    # The guard should NOT skip FakePlainField (it has no is_abstraction_field attr)
    # and should proceed to the DataTypeRef check.
    result = _is_hierarchical(FakeComponentIR(), ctx=FakeCtx())
    assert result is True
