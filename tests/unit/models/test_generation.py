"""Tests for text generation utilities.

Created by @pytholic on 2025.12.15
"""

import pytest
import torch

from legollm.models.architectures.gpt import GPT, GPT2Config125M
from legollm.models.generate import (
    SamplingStrategy,
    apply_top_k_filtering,
    apply_top_p_filtering,
    generate_text,
)


class TestTopKFiltering:
    """Tests for top-k filtering."""

    def test_top_k_keeps_correct_number_of_tokens(self) -> None:
        """Top-k filtering keeps exactly k tokens."""
        logits = torch.randn(2, 100)  # (batch=2, vocab=100)
        k = 10

        filtered = apply_top_k_filtering(logits, k)

        # Count non-inf values per batch
        for i in range(2):
            non_inf_count = (filtered[i] != float("-inf")).sum()
            assert non_inf_count == k

    def test_top_k_keeps_highest_probability_tokens(self) -> None:
        """Top-k keeps the k highest values."""
        logits = torch.tensor([[5.0, 1.0, 3.0, 2.0, 4.0]])  # (1, 5)
        k = 3

        filtered = apply_top_k_filtering(logits, k)

        # Should keep: 5.0, 4.0, 3.0
        assert filtered[0, 0] == 5.0  # kept
        assert filtered[0, 1] == float("-inf")  # removed
        assert filtered[0, 2] == 3.0  # kept
        assert filtered[0, 3] == float("-inf")  # removed
        assert filtered[0, 4] == 4.0  # kept

    def test_top_k_handles_k_larger_than_vocab(self) -> None:
        """Top-k doesn't crash when k > vocab_size."""
        logits = torch.randn(1, 10)
        k = 100  # Larger than vocab

        filtered = apply_top_k_filtering(logits, k)

        # Should keep all tokens
        assert (filtered != float("-inf")).all()


class TestTopPFiltering:
    """Tests for nucleus (top-p) filtering."""

    def test_top_p_keeps_minimum_one_token(self) -> None:
        """Nucleus sampling always keeps at least one token."""
        logits = torch.randn(2, 100)
        p = 0.01  # Very restrictive

        filtered = apply_top_p_filtering(logits, p)

        # Should keep at least 1 token per batch
        for i in range(2):
            non_inf_count = (filtered[i] != float("-inf")).sum()
            assert non_inf_count >= 1

    def test_top_p_keeps_correct_number_of_tokens(self) -> None:
        """Top-p filtering keeps tokens whose cumulative probability is within the threshold."""
        # Sort logits in descending order, so cumulative_probs increases
        # Logits: [8.0, 5.0, 3.0, 2.0, 1.0]
        # Probs:  [0.943, 0.047, 0.006, 0.002, 0.001]
        # Cumsum: [0.943, 0.990, 0.996, 0.998, 0.999]
        logits = torch.tensor([[8.0, 5.0, 3.0, 2.0, 1.0]])
        p = 0.95  # Threshold

        filtered = apply_top_p_filtering(logits, p)

        # With p=0.95, should keep 8.0 and 5.0 (cumulative 0.990), remove rest
        assert filtered[0, 0] == 8.0
        assert filtered[0, 1] == 5.0
        assert filtered[0, 2] == float("-inf")
        assert filtered[0, 3] == float("-inf")
        assert filtered[0, 4] == float("-inf")


class TestGenerateText:
    """Tests for text generation function."""

    @pytest.fixture
    def model(self) -> GPT:
        """Create a small test model."""
        config = GPT2Config125M()
        config.num_layers = 2
        config.num_heads = 4
        config.embed_dim = 128
        config.context_length = 64
        return GPT(config)

    def test_output_shape_correct(self, model: GPT) -> None:
        """Generated output has correct shape."""
        batch_size, seq_len = 2, 10
        max_new_tokens = 5

        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = generate_text(
            model, token_ids, max_new_tokens=max_new_tokens, strategy=SamplingStrategy.GREEDY
        )

        expected_len = seq_len + max_new_tokens
        assert output.shape == (batch_size, expected_len)

    def test_greedy_is_deterministic(self, model: GPT) -> None:
        """Greedy sampling produces same output every time."""
        token_ids = torch.randint(0, 1000, (1, 10))

        output1 = generate_text(
            model, token_ids.clone(), max_new_tokens=5, strategy=SamplingStrategy.GREEDY
        )
        output2 = generate_text(
            model, token_ids.clone(), max_new_tokens=5, strategy=SamplingStrategy.GREEDY
        )

        assert torch.equal(output1, output2)

    def test_temperature_affects_diversity(self, model: GPT) -> None:
        """Higher temperature increases output diversity."""
        token_ids = torch.randint(0, 1000, (1, 10))

        # Low temperature (more focused)
        torch.manual_seed(42)
        low_temp_outputs = [
            generate_text(
                model,
                token_ids.clone(),
                max_new_tokens=10,
                temperature=0.1,
                strategy=SamplingStrategy.STOCHASTIC,
            )
            for _ in range(10)
        ]

        # High temperature (more diverse)
        torch.manual_seed(42)
        high_temp_outputs = [
            generate_text(
                model,
                token_ids.clone(),
                max_new_tokens=10,
                temperature=2.0,
                strategy=SamplingStrategy.STOCHASTIC,
            )
            for _ in range(10)
        ]

        # High temp should have more unique outputs
        low_unique = len({tuple(out.flatten().tolist()) for out in low_temp_outputs})
        high_unique = len({tuple(out.flatten().tolist()) for out in high_temp_outputs})

        assert high_unique >= low_unique

    def test_handles_long_sequences(self, model: GPT) -> None:
        """Generation handles sequences longer than context window."""
        context_length = model.config.context_length
        long_seq_len = context_length + 50

        token_ids = torch.randint(0, 1000, (1, long_seq_len))

        # Should not crash
        output = generate_text(model, token_ids, max_new_tokens=5, strategy=SamplingStrategy.GREEDY)

        assert output.shape[1] == long_seq_len + 5

    def test_filters_combine_correctly(self, model: GPT) -> None:
        """Top-k and top-p can be used together."""
        token_ids = torch.randint(0, 1000, (1, 10))

        # Should not crash when both are specified
        output = generate_text(
            model,
            token_ids,
            max_new_tokens=5,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            strategy=SamplingStrategy.STOCHASTIC,
        )

        assert output.shape[1] == 15  # 10 + 5


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_new_tokens(self) -> None:
        """Handles max_new_tokens=0."""
        config = GPT2Config125M()
        config.num_layers = 1
        model = GPT(config)

        token_ids = torch.randint(0, 1000, (1, 10))

        output = generate_text(model, token_ids, max_new_tokens=0, strategy=SamplingStrategy.GREEDY)

        # Should return input unchanged
        assert torch.equal(output, token_ids)

    def test_single_token_input(self) -> None:
        """Handles single token input."""
        config = GPT2Config125M()
        config.num_layers = 1
        model = GPT(config)

        token_ids = torch.randint(0, 1000, (1, 1))

        output = generate_text(model, token_ids, max_new_tokens=5, strategy=SamplingStrategy.GREEDY)

        assert output.shape[1] == 6  # 1 + 5
