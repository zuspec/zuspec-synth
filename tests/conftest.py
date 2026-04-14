"""Pytest configuration for zuspec-synth tests.

Adds the tests directory to sys.path so that local helper modules
(e.g. test_components.py) can be imported, regardless of pytest
import-mode (including importlib mode used at the repo root).
"""
import sys
import os

import pytest

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)


# ------------------------------------------------------------------ #
# Formal verification marker                                           #
# ------------------------------------------------------------------ #

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--formal",
        action="store_true",
        default=False,
        help="Include @pytest.mark.formal tests (requires sby + Yosys).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "formal: mark a test as requiring sby / Yosys (skip unless --formal is passed).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list,
) -> None:
    if not config.getoption("--formal"):
        skip = pytest.mark.skip(reason="pass --formal to run Tier-3 formal tests")
        for item in items:
            if "formal" in item.keywords:
                item.add_marker(skip)
