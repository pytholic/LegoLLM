"""Unit tests for the Trainer class."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from legollm.data.dataloader import DataLoader, DataLoaderConfig
from legollm.training import Trainer, TrainerConfig


class SimpleModel(nn.Module):
    """Minimal model for testing trainer mechanics."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.ln(x)
        return self.fc(x)


@pytest.fixture
def temp_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset for testing."""
    dataset_path = tmp_path / "test_dataset"
    dataset_path.mkdir()

    vocab_size = 100
    num_tokens = 1000

    # Create random token data
    rng = np.random.default_rng(42)
    train_data = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    val_data = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)

    train_data.tofile(dataset_path / "train.bin")
    val_data.tofile(dataset_path / "val.bin")

    # Create metadata (must be named meta.json per load_dataset_metadata)
    metadata = {"vocab_size": vocab_size, "tokenizer": "test"}
    import json

    with open(dataset_path / "meta.json", "w") as f:
        json.dump(metadata, f)

    return dataset_path


@pytest.fixture
def trainer_with_model(temp_dataset: Path, tmp_path: Path) -> Trainer:
    """Create a trainer with a simple model for testing."""
    config = TrainerConfig(
        max_iters=100,
        eval_interval=50,
        log_interval=10,
        checkpoint_interval=50,
        checkpoint_dir=tmp_path / "checkpoints",
        learning_rate=1e-3,
        warmup_iters=10,
        lr_decay_iters=100,
        min_lr=1e-4,
        device="cpu",
        compile=False,
        eval_iters=5,
    )

    model = SimpleModel(vocab_size=100, hidden_size=32)

    loader_config = DataLoaderConfig(
        dataset_path=temp_dataset,
        block_size=16,
        batch_size=4,
        device="cpu",
        split="train",
    )
    train_loader = DataLoader(loader_config)

    val_loader_config = DataLoaderConfig(
        dataset_path=temp_dataset,
        block_size=16,
        batch_size=4,
        device="cpu",
        split="val",
    )
    val_loader = DataLoader(val_loader_config)

    return Trainer(config, model, train_loader, val_loader)


class TestLearningRateSchedule:
    """Tests for cosine decay learning rate schedule with warmup."""

    def test_warmup_starts_near_zero(self) -> None:
        """LR at iteration 0 should be close to zero (linear warmup)."""
        config = TrainerConfig(
            learning_rate=6e-4,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=6e-5,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )
        # Access the class method directly by creating minimal trainer
        model = SimpleModel()
        trainer = _create_minimal_trainer(config, model)

        lr_at_0 = trainer._get_lr(0)
        expected = config.learning_rate * 1 / (config.warmup_iters + 1)
        assert lr_at_0 == pytest.approx(expected, rel=1e-5)

    def test_warmup_reaches_max_lr(self) -> None:
        """LR at end of warmup should reach max learning rate."""
        config = TrainerConfig(
            learning_rate=6e-4,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=6e-5,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )
        model = SimpleModel()
        trainer = _create_minimal_trainer(config, model)

        # At warmup_iters - 1 (last warmup step)
        lr_at_warmup_end = trainer._get_lr(config.warmup_iters - 1)
        expected = config.learning_rate * config.warmup_iters / (config.warmup_iters + 1)
        assert lr_at_warmup_end == pytest.approx(expected, rel=1e-3)

    def test_cosine_decay_midpoint(self) -> None:
        """LR at midpoint of decay should be average of max and min."""
        config = TrainerConfig(
            learning_rate=6e-4,
            warmup_iters=0,
            lr_decay_iters=1000,
            min_lr=6e-5,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )
        model = SimpleModel()
        trainer = _create_minimal_trainer(config, model)

        midpoint = config.lr_decay_iters // 2
        lr_at_midpoint = trainer._get_lr(midpoint)
        expected_midpoint = (config.learning_rate + config.min_lr) / 2
        assert lr_at_midpoint == pytest.approx(expected_midpoint, rel=1e-3)

    def test_lr_reaches_min_after_decay(self) -> None:
        """LR after decay period should be at min_lr."""
        config = TrainerConfig(
            learning_rate=6e-4,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=6e-5,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )
        model = SimpleModel()
        trainer = _create_minimal_trainer(config, model)

        lr_after_decay = trainer._get_lr(config.lr_decay_iters + 100)
        assert lr_after_decay == config.min_lr


class TestOptimizerConfiguration:
    """Tests for optimizer parameter grouping."""

    def test_weight_decay_applied_to_weights_not_biases(self, trainer_with_model: Trainer) -> None:
        """Weight decay should be applied to weights but not biases/layernorms."""
        optimizer = trainer_with_model.optimizer

        # Should have 2 param groups: decay and no_decay
        assert len(optimizer.param_groups) == 2

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == trainer_with_model.config.weight_decay
        assert no_decay_group["weight_decay"] == 0.0

    def test_all_parameters_in_optimizer(self, trainer_with_model: Trainer) -> None:
        """All model parameters should be in the optimizer."""
        model_params = set(trainer_with_model.model.parameters())
        optimizer_params = set()
        for group in trainer_with_model.optimizer.param_groups:
            optimizer_params.update(group["params"])

        assert model_params == optimizer_params


class TestCheckpointing:
    """Tests for checkpoint save and load functionality."""

    def test_save_and_load_checkpoint(self, trainer_with_model: Trainer, tmp_path: Path) -> None:
        """Checkpoint should preserve model state, optimizer state, and training progress."""
        trainer = trainer_with_model

        # Simulate some training progress
        trainer.iter_num = 42
        trainer.best_val_loss = 1.5

        # Modify model weights to verify they're restored
        original_weight = trainer.model.fc.weight.clone()

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint("test_checkpoint.pt")
        assert checkpoint_path.exists()

        # Modify trainer state
        trainer.iter_num = 0
        trainer.best_val_loss = float("inf")
        trainer.model.fc.weight.data.fill_(0.0)

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        # Verify state restored
        assert trainer.iter_num == 42
        assert trainer.best_val_loss == 1.5
        assert torch.allclose(trainer.model.fc.weight, original_weight)

    def test_checkpoint_contains_required_keys(self, trainer_with_model: Trainer) -> None:
        """Checkpoint should contain all necessary keys for resuming."""
        checkpoint_path = trainer_with_model.save_checkpoint("test.pt")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "iter_num",
            "best_val_loss",
            "config",
        }
        assert required_keys <= set(checkpoint.keys())


def _create_minimal_trainer(config: TrainerConfig, model: nn.Module) -> Trainer:
    """Create a trainer with mock dataloaders for testing LR schedule."""

    # Create a minimal mock dataloader
    class MockDataLoader:
        def __iter__(self):
            while True:
                yield torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long)

        def __len__(self):
            return 1

    mock_loader = MockDataLoader()
    return Trainer(config, model, mock_loader, mock_loader)
