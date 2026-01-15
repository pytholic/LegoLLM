"""Self-contained model architectures.

Each file in this module contains a complete architecture implementation
including model classes, configs, and weight loading utilities.

This follows the "one file per architecture" pattern for easy extensibility.
"""

from legollm.architectures.gpt2 import (
    GPT2,
    GPT2_CONFIG_124M,
    GPT2_CONFIG_355M,
    GPT2_CONFIG_774M,
    GPT2_CONFIG_1558M,
    GPT2ConfigDataclass,
)

__all__ = [
    "GPT2",
    "GPT2_CONFIG_124M",
    "GPT2_CONFIG_355M",
    "GPT2_CONFIG_774M",
    "GPT2_CONFIG_1558M",
    "GPT2ConfigDataclass",
]
