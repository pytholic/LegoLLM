"""Utility functions for data processing.

Created by @pytholic on 2025.10.19
"""


def read_text_file(filepath: str) -> str:
    """Read a text file and return the content as a string."""
    with open(filepath, encoding="utf-8") as file:
        return file.read()
