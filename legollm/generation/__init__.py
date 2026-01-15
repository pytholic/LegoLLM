"""Text generation utilities.

Available:
- generate_text: Core generation function with sampling strategies
- generate_and_decode: High-level generate + decode function
- SamplingStrategy: Enum for greedy/stochastic sampling
- apply_top_k_filtering: Top-k filtering
- apply_top_p_filtering: Nucleus (top-p) sampling
"""

from legollm.generation.generate import (
    SamplingStrategy,
    apply_top_k_filtering,
    apply_top_p_filtering,
    generate_and_decode,
    generate_text,
)

__all__ = [
    "SamplingStrategy",
    "apply_top_k_filtering",
    "apply_top_p_filtering",
    "generate_and_decode",
    "generate_text",
]
