"""Low-Rank Adaptation (LoRA) - placeholder stub.

TODO: Implement LoRA.

Reference: https://arxiv.org/abs/2106.09685

LoRA is a parameter-efficient fine-tuning technique that adds
low-rank decomposition matrices to existing weights, allowing
fine-tuning with a fraction of the parameters.

Key concepts:
- Original weight W stays frozen
- Add low-rank matrices: W' = W + B*A where B is (d, r) and A is (r, k)
- Only train A and B (much fewer parameters)
- r (rank) is typically 4-64

Typical target modules for LLMs:
- Query and Value projections (q_proj, v_proj)
- Sometimes Key and Output projections too
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    TODO: Implement LoRA linear layer.

    This wraps an existing Linear layer and adds low-rank adaptation.
    The original weights are frozen, only the LoRA matrices are trained.

    Args:
        base_layer: The original nn.Linear layer to adapt
        r: Rank of the low-rank decomposition
        alpha: Scaling factor (typically alpha/r is used as scaling)
        dropout: Dropout probability for LoRA

    Forward computation:
        output = base_layer(x) + (dropout(x) @ A.T @ B.T) * (alpha / r)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LoRALinear."""
        super().__init__()
        raise NotImplementedError(
            "LoRALinear not yet implemented. See PEFT library or LoRA paper for reference."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        raise NotImplementedError

    def merge(self) -> None:
        """Merge LoRA weights into base layer for inference."""
        raise NotImplementedError


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA to specified modules in a model.

    TODO: Implement LoRA application.

    Args:
        model: The model to adapt
        target_modules: List of module names to apply LoRA to
                       (e.g., ["q_proj", "v_proj"])
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout

    Returns:
        Model with LoRA layers applied

    Example:
        >>> model = GPT2(GPT2_CONFIG_124M)
        >>> model = apply_lora(model, ["q_proj", "v_proj"], r=8)
        >>> # Only LoRA parameters are trainable
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    """
    raise NotImplementedError(
        "apply_lora not yet implemented. See PEFT library for reference implementation."
    )


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get only the LoRA parameters from a model.

    Useful for creating optimizer with only LoRA params.
    """
    raise NotImplementedError


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base model for inference.

    After merging, the model has the same architecture as the original
    but with updated weights. This removes the LoRA overhead during inference.
    """
    raise NotImplementedError
