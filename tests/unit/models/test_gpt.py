"""Unit tests for the GPT architecture."""

from __future__ import annotations

import pytest
import torch

from legollm.models.architectures.gpt import GPT, GPT2Config125M
from legollm.utils import count_model_params


class TestGPTArchitecture:
    """Tests covering core GPT wiring and configuration."""

    def test_forward_output_shape(self) -> None:
        """Model produces logits with expected dimensionality."""
        config = GPT2Config125M()
        model = GPT(config)
        model.eval()

        batch_size, seq_len = 2, 16
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(token_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_output_head_weight_tying(self) -> None:
        """Output projection reuses token embedding weights (GPT-2 weight tying)."""
        model = GPT(GPT2Config125M())
        embedding_weight = model.transformer_embeddings.token_embedding.embedding.weight
        head_weight = model.out_head.weight

        assert embedding_weight.data_ptr() == head_weight.data_ptr()
        assert model.out_head.bias is None

    def test_parameter_count_matches_gpt2_reference(self) -> None:
        """Total trainable parameters align with the GPT-2 125M specification."""
        model = GPT(GPT2Config125M())
        total_params_millions = count_model_params(model)

        assert total_params_millions == pytest.approx(124.439808, rel=1e-5)
