"""Tests for legollm.generation.generate.

Covers: top-k/top-p filtering, pre-allocated buffer generation,
streaming, context-window sliding, and helper functions.
"""

import pytest
import torch

from legollm.architectures.gpt2 import GPT2, GPT2_CONFIG_124M
from legollm.generation.generate import (
    SamplingStrategy,
    _get_context_length,
    _sample_next_token,
    apply_top_k_filtering,
    apply_top_p_filtering,
    generate_text,
)


@pytest.fixture
def small_model():
    """GPT2 with minimal layers for fast tests."""
    cfg = {**GPT2_CONFIG_124M, "n_layers": 2, "n_heads": 4, "n_embd": 128}
    return GPT2(cfg)


# ---------------------------------------------------------------------------
# Top-k filtering
# ---------------------------------------------------------------------------
class TestTopKFiltering:
    def test_keeps_correct_number_of_tokens(self):
        logits = torch.randn(2, 100)
        filtered = apply_top_k_filtering(logits, 10)
        for i in range(2):
            assert (filtered[i] != float("-inf")).sum() == 10

    def test_keeps_highest_values(self):
        logits = torch.tensor([[5.0, 1.0, 3.0, 2.0, 4.0]])
        filtered = apply_top_k_filtering(logits, 3)
        assert filtered[0, 0] == 5.0
        assert filtered[0, 1] == float("-inf")
        assert filtered[0, 2] == 3.0
        assert filtered[0, 3] == float("-inf")
        assert filtered[0, 4] == 4.0

    def test_k_larger_than_vocab_keeps_all(self):
        logits = torch.randn(1, 10)
        filtered = apply_top_k_filtering(logits, 100)
        assert (filtered != float("-inf")).all()


# ---------------------------------------------------------------------------
# Top-p filtering
# ---------------------------------------------------------------------------
class TestTopPFiltering:
    def test_keeps_minimum_one_token(self):
        logits = torch.randn(2, 100)
        filtered = apply_top_p_filtering(logits, 0.01)
        for i in range(2):
            assert (filtered[i] != float("-inf")).sum() >= 1

    def test_keeps_correct_tokens(self):
        # Probs ≈ [0.943, 0.047, 0.006, 0.002, 0.001]
        logits = torch.tensor([[8.0, 5.0, 3.0, 2.0, 1.0]])
        filtered = apply_top_p_filtering(logits, 0.95)
        assert filtered[0, 0] == 8.0
        assert filtered[0, 1] == 5.0
        assert filtered[0, 2] == float("-inf")
        assert filtered[0, 3] == float("-inf")
        assert filtered[0, 4] == float("-inf")


# ---------------------------------------------------------------------------
# generate_text — pre-allocated buffer
# ---------------------------------------------------------------------------
class TestGenerateText:
    def test_output_shape(self, small_model):
        token_ids = torch.randint(0, 100, (1, 8))
        out = generate_text(small_model, token_ids, max_new_tokens=5)
        assert out.shape == (1, 13)

    def test_prompt_preserved_in_output(self, small_model):
        token_ids = torch.randint(0, 100, (1, 6))
        out = generate_text(small_model, token_ids, max_new_tokens=3)
        assert torch.equal(out[:, :6], token_ids)

    def test_zero_new_tokens_returns_input(self, small_model):
        token_ids = torch.randint(0, 100, (1, 5))
        out = generate_text(small_model, token_ids, max_new_tokens=0)
        assert torch.equal(out, token_ids)

    def test_greedy_is_deterministic(self, small_model):
        token_ids = torch.randint(0, 100, (1, 6))
        out1 = generate_text(small_model, token_ids.clone(), max_new_tokens=5)
        out2 = generate_text(small_model, token_ids.clone(), max_new_tokens=5)
        assert torch.equal(out1, out2)

    def test_long_input_past_context_length(self, small_model):
        ctx_len = small_model.config["context_length"]
        token_ids = torch.randint(0, 100, (1, ctx_len + 20))
        out = generate_text(small_model, token_ids, max_new_tokens=5)
        assert out.shape[1] == ctx_len + 20 + 5

    def test_filters_combine(self, small_model):
        token_ids = torch.randint(0, 100, (1, 10))
        out = generate_text(
            small_model,
            token_ids,
            max_new_tokens=5,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            strategy=SamplingStrategy.STOCHASTIC,
        )
        assert out.shape[1] == 15


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
class TestStreamGenerate:
    def test_yields_correct_count(self, small_model):
        token_ids = torch.randint(0, 100, (1, 6))
        tokens = list(generate_text(small_model, token_ids, max_new_tokens=5, stream=True))
        assert len(tokens) == 5

    def test_yields_ints(self, small_model):
        token_ids = torch.randint(0, 100, (1, 6))
        for tok in generate_text(small_model, token_ids, max_new_tokens=3, stream=True):
            assert isinstance(tok, int)

    def test_stream_matches_non_stream_greedy(self, small_model):
        token_ids = torch.randint(0, 100, (1, 6))
        full = generate_text(
            small_model, token_ids.clone(), max_new_tokens=5, strategy=SamplingStrategy.GREEDY
        )
        streamed = list(
            generate_text(
                small_model,
                token_ids.clone(),
                max_new_tokens=5,
                strategy=SamplingStrategy.GREEDY,
                stream=True,
            )
        )
        new_tokens_from_full = full[0, token_ids.size(1) :].tolist()
        assert new_tokens_from_full == streamed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestSampleNextToken:
    def test_greedy_picks_argmax(self):
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        tok = _sample_next_token(logits, SamplingStrategy.GREEDY, 1.0, None, None)
        assert tok.item() == 1

    def test_invalid_strategy_raises(self):
        logits = torch.tensor([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Invalid sampling strategy"):
            _sample_next_token(logits, "invalid", 1.0, None, None)


class TestGetContextLength:
    def test_reads_from_dict_config(self, small_model):
        result = _get_context_length(small_model, fallback=999)
        assert result == small_model.config["context_length"]

    def test_fallback_when_no_config(self):
        bare = torch.nn.Linear(10, 10)
        assert _get_context_length(bare, fallback=42) == 42
