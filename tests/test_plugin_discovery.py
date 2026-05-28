"""Tests for plugin discovery via importlib.metadata entry-points.

Covers:
- ``_load_plugins()`` invokes discovered entry-point callables with the registry
- ``LoweringRegistry.attach_external_model()`` + ``get_sv_model()`` round-trip
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call


class TestPluginLoading:
    def test_plugin_loading_calls_register_fn(self):
        """_load_plugins() loads each entry-point and calls it with the registry."""
        from zuspec.ir.core.registry import _load_plugins, LoweringRegistry

        registry = LoweringRegistry()
        hook = MagicMock()

        fake_ep = MagicMock()
        fake_ep.load.return_value = hook

        with patch("importlib.metadata.entry_points", return_value=[fake_ep]) as mock_eps:
            _load_plugins(registry)

        mock_eps.assert_called_once_with(group="zuspec.lowering")
        fake_ep.load.assert_called_once()
        hook.assert_called_once_with(registry)

    def test_plugin_exception_does_not_propagate(self):
        """A failing plugin must not break the synthesis run."""
        from zuspec.ir.core.registry import _load_plugins, LoweringRegistry

        registry = LoweringRegistry()

        bad_ep = MagicMock()
        bad_ep.load.return_value = MagicMock(side_effect=RuntimeError("boom"))

        with patch("importlib.metadata.entry_points", return_value=[bad_ep]):
            _load_plugins(registry)  # must not raise

    def test_multiple_plugins_all_invoked(self):
        """All entry-points in the group are called even if one is first."""
        from zuspec.ir.core.registry import _load_plugins, LoweringRegistry

        registry = LoweringRegistry()
        hook_a = MagicMock()
        hook_b = MagicMock()

        ep_a = MagicMock()
        ep_a.load.return_value = hook_a
        ep_b = MagicMock()
        ep_b.load.return_value = hook_b

        with patch("importlib.metadata.entry_points", return_value=[ep_a, ep_b]):
            _load_plugins(registry)

        hook_a.assert_called_once_with(registry)
        hook_b.assert_called_once_with(registry)


class TestAttachExternalModel:
    def test_round_trip_sv(self):
        """attach_external_model('sv') is visible via get_sv_model()."""
        from zuspec.ir.core.registry import LoweringRegistry

        registry = LoweringRegistry()

        class _FakeCls:
            pass

        fake_model = MagicMock()
        fake_model.spec_type_name = "_FakeCls"

        registry.attach_external_model(_FakeCls, "sv", fake_model)
        result = registry.get_sv_model("_FakeCls")
        assert result is fake_model

    def test_invalid_target_raises(self):
        """attach_external_model raises ValueError for unknown target."""
        from zuspec.ir.core.registry import LoweringRegistry
        import pytest

        registry = LoweringRegistry()

        class _FakeCls2:
            pass

        with pytest.raises(ValueError, match="Unknown target"):
            registry.attach_external_model(_FakeCls2, "bogus_target", MagicMock())

    def test_overwrite_model(self):
        """A second attach_external_model call for the same class overwrites the model."""
        from zuspec.ir.core.registry import LoweringRegistry

        registry = LoweringRegistry()

        class _FakeCls3:
            pass

        model_v1 = MagicMock()
        model_v2 = MagicMock()
        registry.attach_external_model(_FakeCls3, "sv", model_v1)
        registry.attach_external_model(_FakeCls3, "sv", model_v2)
        assert registry.get_sv_model("_FakeCls3") is model_v2
