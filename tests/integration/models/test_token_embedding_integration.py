"""Integration tests for the TokenEmbedding model.

Created by @pytholic on 2025.10.28
"""

import json
from pathlib import Path

import numpy as np
import torch

from legollm.data.dataloader import DataLoader, DataLoaderConfig
from legollm.models.token_embedding import TokenEmbedding


class TestTokenEmbeddingIntegration:
    """Integration tests for the TokenEmbedding model.

    Methodology:
        - Create a dataloader from the tokenizer
        - Use the TokenEmbedding model to embed the data
        - Check if the embedded data is the same as the original data
    """

    def test_token_embedding_with_dataloader(self, tmp_path: Path):
        """Test TokenEmbedding processes batches from DataLoader correctly."""
        # Create a simpel dataset with 1000 tokens
        vocab_size = 50
        num_tokens = 500
        token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)

        # Save as .bin file
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        token_ids.numpy().astype("uint8").tofile(dataset_dir / "train.bin")

        # Create metadata file
        with open(dataset_dir / "meta.json", "w") as f:
            json.dump({"vocab_size": vocab_size}, f)

        # Create a DataLoader
        config = DataLoaderConfig(
            dataset_path=dataset_dir,
            block_size=8,
            batch_size=4,
            device="cpu",
            token_buffer_size=1000,
            dtype=np.uint8,
            split="train",
        )
        dataloader = DataLoader(config)

        # Create TokenEmbedding
        embed_dim = 16
        token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)

        # Get a batch and embed it
        x, y = next(iter(dataloader))
        embedded_x = token_embedding(x)

        # Verify shapes
        assert x.shape == (4, 8), f"Expected (4, 8), got {x.shape}"
        assert embedded_x.shape == (4, 8, 16), f"Expected (4, 8, 16), got {embedded_x.shape}"

        # Verify embeddings are different for different tokens
        # (unless by chance they're the same token)
        if x[0, 0] != x[0, 1]:
            assert not torch.allclose(embedded_x[0, 0][0], embedded_x[0, 1][0])

    def test_embedding_batch_consistency(self, tmp_path: Path):
        """Test that the embeddings are consistent across multiple batches."""
        num_batches = 10
        vocab_size = 10
        batch_size = 1

        # Create repeating pattern
        # We need to handle block_size+1 for proper alignment of the token sequences
        token_ids = np.array([], dtype=np.uint8)
        for _ in range(num_batches):
            token_ids = np.append(token_ids, np.arange(vocab_size, dtype=np.uint8))
            token_ids = np.append(token_ids, np.uint8(0))

        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()
        token_ids.tofile(dataset_dir / "train.bin")
        with open(dataset_dir / "meta.json", "w") as f:
            json.dump({"vocab_size": vocab_size}, f)

        config = DataLoaderConfig(
            dataset_path=dataset_dir,
            block_size=vocab_size,  # block_size=10
            batch_size=batch_size,  # batch_size=1
            device="cpu",
            split="train",
        )
        dataloader = DataLoader(config)

        # Create TokenEmbedding
        embed_dim = 16
        token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)

        # Get two batches
        x1, _ = dataloader.get_batch()  # Shape: (1, 10) = [[0,1,2,3,4,5,6,7,8,9]]
        x2, _ = dataloader.get_batch()  # Shape: (1, 10) = [[0,1,2,3,4,5,6,7,8,9]]

        emb1 = token_embedding(x1)  # Shape: (1, 10, 16)
        emb2 = token_embedding(x2)  # Shape: (1, 10, 16)

        # Assert shapes
        assert emb1.shape == (batch_size, vocab_size, embed_dim)
        assert emb2.shape == (batch_size, vocab_size, embed_dim)

        # Assert embeddings are identical
        assert torch.allclose(emb1, emb2), (
            "Same token sequences should produce identical embeddings"
        )
