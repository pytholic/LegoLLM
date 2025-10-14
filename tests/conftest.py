"""Configuration for pytest."""

from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def simple_training_text() -> str:
    """Simple text for basic training tests."""
    return "Hello world! This is a test. Hello everyone."


@pytest.fixture
def multilingual_text() -> str:
    """Multilingual text for testing Unicode handling with emoji."""
    return "Hello world! 안녕하세요 세계! Bonjour le monde! こんにちは世界! 😊 🚀"
