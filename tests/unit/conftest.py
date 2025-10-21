"""Unit tests configuration.

Created by @pytholic on 2025.10.20
"""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to unit test fixtures directory."""
    return Path(__file__).parent / "fixtures"
