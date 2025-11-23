"""Tests for dtype alignment across model components.

Created by @pytholic on 2025.11.23
"""

import pytest
import torch

from legollm.models.architectures.gpt import GPT, GPT2Config125M


class TestDtypeAlignment:
    """Test dtype alignment across all model components."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float16, id="float16"),
            pytest.param(torch.bfloat16, id="bfloat16"),
        ],
    )
    def test_all_parameters_match_config_dtype(self, dtype: torch.dtype) -> None:
        """Test that all model parameters match the configured dtype."""
        config = GPT2Config125M()
        config.dtype = dtype
        model = GPT(config)

        for name, param in model.named_parameters():
            assert param.dtype == dtype, (
                f"Parameter {name} has dtype {param.dtype}, expected {dtype}"
            )

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float16, id="float16"),
            pytest.param(torch.bfloat16, id="bfloat16"),
        ],
    )
    def test_all_buffers_match_config_dtype(self, dtype: torch.dtype) -> None:
        """Test that all model buffers (including causal masks) match the configured dtype."""
        config = GPT2Config125M()
        config.dtype = dtype
        model = GPT(config)

        for name, buffer in model.named_buffers():
            assert buffer.dtype == dtype, (
                f"Buffer {name} has dtype {buffer.dtype}, expected {dtype}"
            )

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float16, id="float16"),
            pytest.param(torch.bfloat16, id="bfloat16"),
        ],
    )
    def test_forward_pass_output_dtype(self, dtype: torch.dtype) -> None:
        """Test that forward pass output matches the configured dtype."""
        config = GPT2Config125M()
        config.dtype = dtype
        model = GPT(config)

        batch_size = 2
        seq_len = 10
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(token_ids)

        assert output.dtype == dtype, f"Output has dtype {output.dtype}, expected {dtype}"
        assert output.shape == (
            batch_size,
            seq_len,
            config.vocab_size,
        ), f"Output shape {output.shape} does not match expected"

    def test_model_dtype_property(self) -> None:
        """Test that the model.dtype property returns the correct dtype."""
        config = GPT2Config125M()
        config.dtype = torch.float16
        model = GPT(config)

        assert model.dtype == torch.float16, "model.dtype property does not match config"

    def test_dtype_consistency_across_components(self) -> None:
        """Test that all major components have consistent dtypes."""
        config = GPT2Config125M()
        config.dtype = torch.float16
        model = GPT(config)

        # Check embeddings
        assert all(p.dtype == torch.float16 for p in model.transformer_embeddings.parameters()), (
            "Embeddings have inconsistent dtype"
        )

        # Check transformer blocks
        for i, block in enumerate(model.trt_blocks):
            assert all(p.dtype == torch.float16 for p in block.parameters()), (
                f"Transformer block {i} has inconsistent dtype"
            )

        # Check final layer norm
        assert all(p.dtype == torch.float16 for p in model.ln_f.parameters()), (
            "Final layer norm has inconsistent dtype"
        )

        # Check output head (shares weights with embeddings)
        assert model.out_head.weight.dtype == torch.float16, "Output head has inconsistent dtype"

    def test_causal_mask_dtype_in_attention(self) -> None:
        """Test that causal masks in attention modules match model dtype."""
        config = GPT2Config125M()
        config.dtype = torch.float16
        model = GPT(config)

        # Check causal masks in all transformer blocks
        for i, block in enumerate(model.trt_blocks):
            causal_mask = block.attn.causal_mask
            assert causal_mask.dtype == torch.float16, (
                f"Causal mask in block {i} has dtype {causal_mask.dtype}, expected float16"
            )

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float16, id="float16"),
        ],
    )
    def test_forward_pass_with_different_sequence_lengths(self, dtype: torch.dtype) -> None:
        """Test that dtype is maintained across different sequence lengths."""
        config = GPT2Config125M()
        config.dtype = dtype
        model = GPT(config)

        for seq_len in [5, 10, 50, 100]:
            token_ids = torch.randint(0, config.vocab_size, (1, seq_len))
            output = model(token_ids)
            assert output.dtype == dtype, (
                f"Output dtype mismatch for seq_len={seq_len}: got {output.dtype}, expected {dtype}"
            )

    def test_weight_tying_preserves_dtype(self) -> None:
        """Test that weight tying between embeddings and output head preserves dtype."""
        config = GPT2Config125M()
        config.dtype = torch.float16
        model = GPT(config)

        # Check that weights are tied
        assert (
            model.out_head.weight is model.transformer_embeddings.token_embedding.embedding.weight
        ), "Weights are not tied"

        # Check that both have the same dtype
        assert model.out_head.weight.dtype == torch.float16, (
            "Output head weight has incorrect dtype"
        )
        assert (
            model.transformer_embeddings.token_embedding.embedding.weight.dtype == torch.float16
        ), "Token embedding weight has incorrect dtype"
