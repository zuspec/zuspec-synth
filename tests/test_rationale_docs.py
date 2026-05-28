"""Verify that all four rationale docs exist and are non-trivially sized (R4)."""
import pytest
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]  # …/zuspec-pss-examples-2
_RATIONALE_DIR = _REPO_ROOT / "docs" / "rationale"

_EXPECTED_DOCS = [
    "activity-ir.md",
    "pipeline-ir.md",
    "structural-rtl-ir.md",
    "pass-pipeline.md",
]

_MIN_BYTES = 500


@pytest.mark.parametrize("filename", _EXPECTED_DOCS)
def test_rationale_doc_exists_and_has_content(filename):
    path = _RATIONALE_DIR / filename
    assert path.exists(), f"Rationale doc missing: {path}"
    size = path.stat().st_size
    assert size >= _MIN_BYTES, (
        f"Rationale doc {filename!r} is too small ({size} bytes, expected >= {_MIN_BYTES}). "
        "Did the doc get truncated?"
    )


def test_rationale_directory_exists():
    assert _RATIONALE_DIR.is_dir(), f"docs/rationale/ directory missing: {_RATIONALE_DIR}"
