"""Feed-forward network implementations.

Available:
- MLP: Standard GELU-activated MLP

Todo:
- SwiGLU: SwiGLU activation (for Llama/Qwen)
- MoE: Mixture of Experts
"""

from legollm.components.feedforward.mlp import MLP

__all__ = [
    "MLP",
]
