"""LegoLLM - A modular LLM framework."""

from .config import settings
from .exceptions import ProjectBaseError
from .logging import logger

# Public API exports
__all__ = [
    "ProjectBaseError",
    "logger",
    "settings",
]
