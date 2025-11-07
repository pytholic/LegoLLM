"""Test the embedding classes.

Created by @pytholic on 2025.10.28
"""

import pytest
import torch

from legollm.core.exceptions import EmbeddingsError
from legollm.models.positional_embedding import PositionalEmbedding
from legollm.models.token_embedding import TokenEmbedding
from legollm.models.transformer_embeddings import TransformerEmbeddings


class TestTokenEmbedding:
    """Test the TokenEmbedding class."""

    # Shape tests
    @pytest.mark.parametrize(
        "batch_size, seq_len, vocab_size, embed_dim",
        [
            pytest.param(2, 5, 100, 10, id="batch_size_2_seq_len_5_vocab_size_100_embed_dim_10"),
            pytest.param(3, 4, 100, 8, id="batch_size_3_seq_len_4_vocab_size_100_embed_dim_8"),
        ],
    )
    def test_output_shape(self, batch_size: int, seq_len: int, vocab_size: int, embed_dim: int):
        """Test the output shape of the TokenEmbedding model.

        Method:
            - Pass integer tokens to the model
            - Check if the output shape is (batch_size, sequence_length, embedding_dim)
        """
        token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = token_embedding(token_ids)
        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_edge_case_shapes(self):
        """Test batch_size=1, seq_len=1."""
        token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        token_ids = torch.randint(0, 100, (1, 1))
        output = token_embedding(token_ids)
        assert output.shape == (1, 1, 10)

    # Functional tests
    @pytest.mark.parametrize("token_ids", [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    def test_same_token_same_embedding(self, token_ids: tuple[int, int, int]):
        """Test that the same token has the same embedding."""
        token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        token_ids = torch.tensor([token_ids])  # pyright: ignore[reportAssignmentType]
        output = token_embedding(token_ids)
        assert (
            output[0, 0][0] == output[0, 1][0] == output[0, 2][0]
        )  # [0] because it contains another grad item in the tensor

    @pytest.mark.parametrize("token_ids", [(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    def test_different_tokens_different_embeddings(self, token_ids: tuple[int, int, int]):
        """Test that different tokens have different embeddings."""
        token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        token_ids = torch.tensor([token_ids])  # pyright: ignore[reportAssignmentType]
        output = token_embedding(token_ids)
        assert output[0, 0][0] != output[0, 1][0] != output[0, 2][0]

    def test_embedding_consistency_across_positions(self):
        """Test that the embeddings are consistent across positions.
        Same token in different positions produces same embedding vector.
        """
        token_ids_seqs: list[tuple[int, int, int]] = [(1, 0, 2), (0, 1, 2), (2, 1, 0)]
        output_embeddings: list[torch.Tensor] = []
        token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        for token_ids in token_ids_seqs:
            token_ids = torch.tensor([token_ids])  # pyright: ignore[reportAssignmentType]
            output_embeddings.append(token_embedding(token_ids))

        assert (
            output_embeddings[0][0, 1][0]
            == output_embeddings[1][0, 0][0]
            == output_embeddings[2][0, 2][0]
        )
        assert (
            output_embeddings[0][0, 0][0]
            == output_embeddings[1][0, 1][0]
            == output_embeddings[2][0, 1][0]
        )
        assert (
            output_embeddings[0][0, 2][0]
            == output_embeddings[1][0, 2][0]
            == output_embeddings[2][0, 0][0]
        )

    # Parameter tests
    def test_weight_matrix_shape(self):
        """Test that the embedding weight matrix shape is (vocab_size, embed_dim)."""
        vocab_size = 100
        embed_dim = 10
        token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        assert token_embedding.embedding.weight.shape == (vocab_size, embed_dim)

    def test_parameter_count(self):
        """Test that the number of parameters is (vocab_size * embed_dim)."""
        vocab_size = 100
        embed_dim = 10
        token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        assert sum(p.numel() for p in token_embedding.parameters()) == vocab_size * embed_dim

    # Edge cases
    def test_first_token(self):
        """Token ID 0 works correctly."""
        token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        token_ids = torch.tensor([[0]])
        output = token_embedding(token_ids)
        assert output.shape == (1, 1, 10)
        assert output[0, 0, 0] == token_embedding.embedding.weight[0, 0]

    def test_last_token(self):
        """Token ID vocab_size-1 works correctly."""
        token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        token_ids = torch.tensor([[99]])
        output = token_embedding(token_ids)
        assert output.shape == (1, 1, 10)
        assert output[0, 0, 0] == token_embedding.embedding.weight[99, 0]

    def test_out_of_bounds_raises_error(self):
        """Token ID >= vocab_size raises IndexError."""
        token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        token_ids = torch.tensor([[100]])
        with pytest.raises(IndexError):
            token_embedding(token_ids)

    # Device tests
    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu", id="cpu"),
            pytest.param(
                "mps",
                id="mps",
                marks=pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="MPS not available"
                ),
            ),
            pytest.param(
                "cuda",
                id="cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_device(self, device: str):
        """Embedding works on CPU (default)."""
        emb = TokenEmbedding(vocab_size=100, embed_dim=10).to(device)
        token_ids = torch.tensor([[1, 2, 3]]).to(device)
        output = emb(token_ids)
        assert output.device.type == device


class TestPositionalEmbedding:
    """Test the PositionalEmbedding class."""

    def test_output_shape(self):
        """Test the output shape of the PositionalEmbedding model."""
        positional_embedding = PositionalEmbedding(context_length=100, embed_dim=10)
        output = positional_embedding(seq_len=10)
        assert output.shape == (10, 10)

    def test_max_seq_len_boundary(self):
        """Test that the positional embeddings are correct at the boundary of the sequence length."""
        positional_embedding = PositionalEmbedding(context_length=100, embed_dim=10)
        output = positional_embedding(seq_len=100)
        assert output.shape == (100, 10)
        assert output[0, 0] == positional_embedding.pos_embedding.weight[0, 0]
        assert output[99, 0] == positional_embedding.pos_embedding.weight[99, 0]

    def test_position_embeddings_are_learnable(self):
        """Test that the positional embeddings are learnable."""
        positional_embedding = PositionalEmbedding(context_length=100, embed_dim=10)
        assert positional_embedding.pos_embedding.weight.requires_grad


class TestTransformerEmbeddings:
    """Test the TransformerEmbeddings class."""

    def test_output_shape(self):
        """Test the output shape of the TransformerEmbeddings model."""
        seq_len = 3
        transformer_embeddings = TransformerEmbeddings(
            vocab_size=100, embed_dim=10, context_length=100
        )
        token_ids = torch.arange(seq_len).unsqueeze(0)
        output = transformer_embeddings(token_ids=token_ids, seq_len=seq_len)
        assert output.shape == (1, seq_len, 10)

    def test_combines_token_and_positional_embeddings(self):
        """Test that the TransformerEmbeddings model combines the token and positional embeddings correctly."""
        seq_len = 3

        transformer_embeddings = TransformerEmbeddings(
            vocab_size=100, embed_dim=10, context_length=100
        )
        transformer_embeddings.eval()  # disable dropout

        token_ids = torch.arange(seq_len).unsqueeze(0)

        # Get transformer embeddings
        output = transformer_embeddings(token_ids=token_ids, seq_len=seq_len)

        # Get separate token and positional embeddings
        token_embeddings = transformer_embeddings.token_embedding(token_ids)
        positional_embeddings = transformer_embeddings.positional_embedding(seq_len)

        # Assert that the combined embeddings are equal to the transformer embeddings
        assert torch.allclose(output, token_embeddings + positional_embeddings)

    def test_dropout_is_applied(self):
        pass

    def test_sequence_length_flexibility(self):
        """Test that the TransformerEmbeddings model can handle sequences of different lengths till context length."""
        context_length = 100
        seq_lens = [3, 5, 7, context_length]
        for seq_len in seq_lens:
            transformer_embeddings = TransformerEmbeddings(
                vocab_size=100, embed_dim=10, context_length=context_length
            )
            token_ids = torch.arange(seq_len).unsqueeze(0)
            output = transformer_embeddings(token_ids=token_ids, seq_len=seq_len)
            assert output.shape == (1, seq_len, 10)

    def test_sequence_length_greater_than_context_length_raises_error(self):
        """Test that the TransformerEmbeddings model raises an error if the sequence length is greater than context length."""
        context_length = 100
        seq_len = 101
        transformer_embeddings = TransformerEmbeddings(
            vocab_size=100, embed_dim=10, context_length=context_length
        )
        token_ids = torch.arange(seq_len).unsqueeze(0)
        with pytest.raises(EmbeddingsError):
            transformer_embeddings(token_ids=token_ids, seq_len=seq_len)
