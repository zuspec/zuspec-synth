"""Tests for LoweringRegistry (Phase 1).

Per the plan these tests live in the zuspec-synth test suite so that
zuspec-synth's own tests can also import and exercise the registry.
"""

import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest
from zuspec.ir.core.registry import LoweringRegistry, global_registry, _load_plugins
from zuspec.ir.core.interfaces import (
    ElaboratableInterface,
    SVEmittableInterface,
    SVAEmittableInterface,
    CSimEmittableInterface,
)


# ---------------------------------------------------------------------------
# Helpers: minimal concrete classes that satisfy each interface
# ---------------------------------------------------------------------------

class _FullModel:
    """Inline model: implements all four interfaces."""

    @classmethod
    def elaborate_field(cls, field_name, field_index, inst_kwargs, element_type=None):
        return None

    @classmethod
    def sv_module_text(cls, field_ir):
        return ""

    @classmethod
    def sv_instance_text(cls, field_ir, parent_prefix):
        return ""

    @classmethod
    def rewrite_proc_stmts(cls, stmts, field_ir):
        return stmts

    @classmethod
    def sva_assert_properties(cls, field_ir):
        return []

    @classmethod
    def sva_assume_properties(cls, field_ir):
        return []

    @classmethod
    def bmc_depth(cls, field_ir):
        return 0

    @classmethod
    def cutpoint_signals(cls, field_ir):
        return []


class _SVOnlyModel:
    @classmethod
    def sv_module_text(cls, field_ir):
        return "module foo(); endmodule"

    @classmethod
    def sv_instance_text(cls, field_ir, parent_prefix):
        return ""

    @classmethod
    def rewrite_proc_stmts(cls, stmts, field_ir):
        return stmts


class _FVOnlyModel:
    @classmethod
    def sva_assert_properties(cls, field_ir):
        return ["assert property (...)"]

    @classmethod
    def sva_assume_properties(cls, field_ir):
        return []

    @classmethod
    def bmc_depth(cls, field_ir):
        return 4

    @classmethod
    def cutpoint_signals(cls, field_ir):
        return []


class MyClass:
    """Dummy class used as the key for external-model tests."""


# ---------------------------------------------------------------------------
# register() — inline model
# ---------------------------------------------------------------------------

def test_register_inline_model_sv():
    reg = LoweringRegistry()
    reg.register(_FullModel)
    assert reg.get_sv_model(_FullModel) is _FullModel


def test_register_inline_model_fv():
    reg = LoweringRegistry()
    reg.register(_FullModel)
    assert reg.get_fv_model(_FullModel) is _FullModel


def test_register_inline_model_elab():
    reg = LoweringRegistry()
    reg.register(_FullModel)
    assert reg.get_elab_model(_FullModel) is not None


# ---------------------------------------------------------------------------
# attach_external_model()
# ---------------------------------------------------------------------------

def test_attach_external_model_sv():
    reg = LoweringRegistry()
    reg.attach_external_model(MyClass, "sv", _SVOnlyModel)
    assert reg.get_sv_model(MyClass) is _SVOnlyModel


def test_attach_external_model_fv():
    reg = LoweringRegistry()
    reg.attach_external_model(MyClass, "fv", _FVOnlyModel)
    assert reg.get_fv_model(MyClass) is _FVOnlyModel


def test_attach_external_model_csim():
    reg = LoweringRegistry()
    reg.attach_external_model(MyClass, "csim", _SVOnlyModel)
    assert reg.get_csim_model(MyClass) is _SVOnlyModel


def test_attach_external_model_elab():
    reg = LoweringRegistry()
    reg.attach_external_model(MyClass, "elab", _FullModel)
    assert reg.get_elab_model(MyClass) is _FullModel


# ---------------------------------------------------------------------------
# Lookup by name string
# ---------------------------------------------------------------------------

def test_lookup_by_name_string():
    reg = LoweringRegistry()
    reg.register(_FullModel)
    assert reg.get_sv_model("_FullModel") is _FullModel


def test_lookup_by_name_string_external():
    reg = LoweringRegistry()
    reg.attach_external_model(MyClass, "sv", _SVOnlyModel)
    assert reg.get_sv_model("MyClass") is _SVOnlyModel


# ---------------------------------------------------------------------------
# Type-keyed cache
# ---------------------------------------------------------------------------

def test_lookup_by_type_uses_type_cache():
    """A second call should hit the type-keyed cache, not the string dict."""
    reg = LoweringRegistry()
    reg.register(_FullModel)

    # Prime the cache
    first = reg.get_sv_model(_FullModel)
    assert first is _FullModel

    # The type cache should be populated now
    assert _FullModel in reg._sv_cache
    assert reg._sv_cache[_FullModel] is _FullModel

    # Second call must return the same object
    second = reg.get_sv_model(_FullModel)
    assert second is _FullModel


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_target_raises():
    reg = LoweringRegistry()
    with pytest.raises(ValueError, match="Unknown target"):
        reg.attach_external_model(MyClass, "bad_target", _SVOnlyModel)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_global_registry_is_singleton():
    r1 = global_registry()
    r2 = global_registry()
    assert r1 is r2


# ---------------------------------------------------------------------------
# Re-registration overwrites
# ---------------------------------------------------------------------------

def test_second_register_overwrites_first():
    reg = LoweringRegistry()

    class Model1:
        @classmethod
        def sv_module_text(cls, field_ir): return "v1"
        @classmethod
        def sv_instance_text(cls, field_ir, pp): return ""
        @classmethod
        def rewrite_proc_stmts(cls, stmts, field_ir): return stmts

    class Model2:
        @classmethod
        def sv_module_text(cls, field_ir): return "v2"
        @classmethod
        def sv_instance_text(cls, field_ir, pp): return ""
        @classmethod
        def rewrite_proc_stmts(cls, stmts, field_ir): return stmts

    reg.attach_external_model(MyClass, "sv", Model1)
    assert reg.get_sv_model(MyClass) is Model1

    reg.attach_external_model(MyClass, "sv", Model2)
    assert reg.get_sv_model(MyClass) is Model2


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------

def test_plugin_loading_calls_register_fn():
    """_load_plugins() should call the register function from each entry point."""
    mock_register = MagicMock()
    mock_ep = MagicMock()
    mock_ep.load.return_value = mock_register

    reg = LoweringRegistry()

    with patch.object(importlib.metadata, "entry_points", return_value=[mock_ep]):
        _load_plugins(reg)

    mock_register.assert_called_once_with(reg)
