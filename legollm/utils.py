"""Utility functions for the project.

Created by @pytholic on 2025.10.19
"""

from pathlib import Path


def read_text_file(filepath: str | Path) -> str:
    """Read a text file and return the content as a string."""
    filepath = Path(filepath)
    with open(filepath, encoding="utf-8") as file:
        return file.read()
