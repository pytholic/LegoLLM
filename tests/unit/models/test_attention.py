"""Unit tests for the attention modules.

Created by @pytholic on 2025.11.02
"""
# pyright: reportMissingModuleSource=false

from __future__ import annotations

import math

import pytest
import torch

from legollm.models.attention import (  # type: ignore[import]
    CausalAttention,
    MultiHeadCausalAttention,
)


def _set_identity_projections(module: CausalAttention) -> None:
    """Configure projections to behave like identity mappings.

    Ensures the attention computations are deterministic and easy to verify.
    """
    with torch.no_grad():
        proj_weight = torch.zeros_like(module.q_proj.weight)
        shared_dim = min(module.d_in, module.head_dim)
        proj_weight[:shared_dim, :shared_dim] = torch.eye(shared_dim)
        module.q_proj.weight.copy_(proj_weight)
        module.k_proj.weight.copy_(proj_weight)
        module.v_proj.weight.copy_(proj_weight)
        if module.q_proj.bias is not None:
            module.q_proj.bias.zero_()
        if module.k_proj.bias is not None:
            module.k_proj.bias.zero_()
        if module.v_proj.bias is not None:
            module.v_proj.bias.zero_()

        out_weight = torch.zeros_like(module.out_proj.weight)
        out_dim = min(module.d_in, module.head_dim)
        out_weight[:out_dim, :out_dim] = torch.eye(out_dim)
        module.out_proj.weight.copy_(out_weight)
        if module.out_proj.bias is not None:
            module.out_proj.bias.zero_()


def _set_identity_projections_multihead(module: MultiHeadCausalAttention) -> None:
    """Configure multi-head projections with deterministic identity-like weights."""
    with torch.no_grad():
        weight = torch.zeros_like(module.q_proj.weight)
        dim = min(module.d_in, module.d_out)
        weight[:dim, :dim] = torch.eye(dim)
        module.q_proj.weight.copy_(weight)
        module.k_proj.weight.copy_(weight)
        module.v_proj.weight.copy_(weight)
        if module.q_proj.bias is not None:
            module.q_proj.bias.zero_()
        if module.k_proj.bias is not None:
            module.k_proj.bias.zero_()
        if module.v_proj.bias is not None:
            module.v_proj.bias.zero_()

        module.out_proj.weight.copy_(torch.eye(module.d_out))
        if module.out_proj.bias is not None:
            module.out_proj.bias.zero_()


class TestCausalAttention:
    """Test the CausalAttention module."""

    def test_output_shape(self) -> None:
        """Test the output shape of the Attention module."""
        d_in = head_dim = 4
        context_length = seq_len = 4

        attention = CausalAttention(
            d_in=d_in,
            head_dim=head_dim,
            context_length=context_length,
            dropout=0.0,
            bias=False,
            causal=True,
        )
        attention.eval()
        _set_identity_projections(attention)

        x = torch.arange(seq_len * d_in, dtype=torch.float32).view(1, seq_len, d_in)
        output = attention(x)
        assert output.shape == (1, seq_len, head_dim)

    def test_attention_causal_mask_blocks_future_positions(self) -> None:
        """Ensure causal masking prevents attending to future tokens."""
        d_in = head_dim = 4
        context_length = seq_len = 4

        attention = CausalAttention(
            d_in=d_in,
            head_dim=head_dim,
            context_length=context_length,
            dropout=0.0,
            bias=False,
            causal=True,
        )
        attention.eval()
        _set_identity_projections(attention)

        x = torch.arange(seq_len * d_in, dtype=torch.float32).view(1, seq_len, d_in)

        output = attention(x)

        q = attention.q_proj(x)
        k = attention.k_proj(x)
        v = attention.v_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.shape[-1])
        mask = attention.causal_mask[:seq_len, :seq_len]  # type: ignore[attr-defined]
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(masked_scores, dim=-1)
        expected = attention.out_proj(weights @ v)

        assert torch.allclose(output, expected, atol=1e-6)

        # mask for the future positions (for comparison)
        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        masked_weights = weights[0].masked_select(future_mask)
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))

    def test_attention_without_causal_mask_allows_future_positions(self) -> None:
        """Without causal masking, tokens may attend to future positions."""
        d_in = head_dim = 4
        context_length = seq_len = 4

        attention = CausalAttention(
            d_in=d_in,
            head_dim=head_dim,
            context_length=context_length,
            dropout=0.0,
            bias=False,
            causal=False,
        )
        attention.eval()
        _set_identity_projections(attention)

        x = torch.arange(seq_len * d_in, dtype=torch.float32).view(1, seq_len, d_in)

        q = attention.q_proj(x)
        k = attention.k_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.shape[-1])
        weights = torch.softmax(scores, dim=-1)

        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        masked_weights = weights[0].masked_select(future_mask)
        assert torch.any(masked_weights > 0)


class TestMultiHeadCausalAttention:
    """Test the MultiHeadCausalAttention module."""

    @pytest.mark.parametrize(
        "d_in,d_out,num_heads,batch_size,seq_len",
        [
            pytest.param(4, 4, 1, 2, 3, id="single_head"),
            pytest.param(4, 8, 2, 1, 4, id="two_heads"),
            pytest.param(8, 8, 4, 2, 5, id="four_heads"),
        ],
    )
    def test_output_shape_and_head_dim(
        self, d_in: int, d_out: int, num_heads: int, batch_size: int, seq_len: int
    ) -> None:
        """Multi-head attention preserves expected output shape and head dimension."""
        attention = MultiHeadCausalAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=seq_len,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            causal=True,
        )

        x = torch.randn(batch_size, seq_len, d_in)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, d_out)
        assert attention.head_dim == d_out // num_heads

    def test_q_projection_splits_into_heads(self) -> None:
        """Q projection reshapes cleanly into the expected number of heads."""
        batch_size, seq_len = 2, 4
        d_in = d_out = 6
        num_heads = 3
        attention = MultiHeadCausalAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=seq_len,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            causal=True,
        )

        x = torch.randn(batch_size, seq_len, d_in)
        q = attention.q_proj(x)
        q_heads = q.view(batch_size, seq_len, num_heads, attention.head_dim)

        assert q_heads.shape == (batch_size, seq_len, num_heads, attention.head_dim)

    def test_causal_mask_blocks_future_positions(self) -> None:
        """Causal masking prevents attention to future positions across heads."""
        d_in = d_out = 4
        seq_len = 4
        attention = MultiHeadCausalAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=seq_len,
            num_heads=2,
            dropout=0.0,
            bias=False,
            causal=True,
        )
        attention.eval()
        _set_identity_projections_multihead(attention)

        x = torch.arange(seq_len * d_in, dtype=torch.float32).view(1, seq_len, d_in)
        output = attention(x)

        q = attention.q_proj(x)
        k = attention.k_proj(x)
        v = attention.v_proj(x)

        q_heads = q.view(1, seq_len, attention.num_heads, attention.head_dim).transpose(1, 2)
        k_heads = k.view(1, seq_len, attention.num_heads, attention.head_dim).transpose(1, 2)
        v_heads = v.view(1, seq_len, attention.num_heads, attention.head_dim).transpose(1, 2)

        scores = q_heads @ k_heads.transpose(-2, -1) / attention.head_dim**0.5
        mask = attention.causal_mask[:seq_len, :seq_len]
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(masked_scores, dim=-1)
        expected = weights @ v_heads
        expected = expected.transpose(1, 2).contiguous().view(1, seq_len, attention.d_out)
        expected = attention.out_proj(expected)

        assert torch.allclose(output, expected, atol=1e-6)

        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        masked_weights = weights[0, :, future_mask]
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))

    def test_non_causal_mode_allows_future_positions(self) -> None:
        """Without causal masking, heads may attend to future tokens."""
        d_in = d_out = 4
        seq_len = 4
        attention = MultiHeadCausalAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=seq_len,
            num_heads=2,
            dropout=0.0,
            bias=False,
            causal=False,
        )
        attention.eval()
        _set_identity_projections_multihead(attention)

        x = torch.arange(seq_len * d_in, dtype=torch.float32).view(1, seq_len, d_in)

        q = attention.q_proj(x)
        k = attention.k_proj(x)

        q_heads = q.view(1, seq_len, attention.num_heads, attention.head_dim).transpose(1, 2)
        k_heads = k.view(1, seq_len, attention.num_heads, attention.head_dim).transpose(1, 2)

        scores = q_heads @ k_heads.transpose(-2, -1) / attention.head_dim**0.5
        weights = torch.softmax(scores, dim=-1)
        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        masked_weights = weights[0, :, future_mask]

        assert torch.any(masked_weights > 0)
