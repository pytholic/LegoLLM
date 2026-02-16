"""Train a language model.

Usage:
    uv run python scripts/train.py --config configs/train/the_verdict.yaml
    uv run python scripts/train.py --config configs/train/the_verdict.yaml --max-iters 500 --lr 1e-3

Created by @pytholic on 2025.12.15
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from legollm.architectures.gpt2 import GPT2, GPT2_CONFIG_124M, GPT2_CONFIG_355M
from legollm.data.dataloader import DataLoader, DataLoaderConfig
from legollm.logging import logger
from legollm.training import Trainer, TrainerConfig
from legollm.utils import count_model_params

MODEL_CONFIGS = {
    "gpt2-124m": GPT2_CONFIG_124M,
    "gpt2-355m": GPT2_CONFIG_355M,
}


def load_config(yaml_path: str) -> TrainerConfig:
    """Load training configuration from a YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # Support both flat and nested (config: ...) YAML layouts
    config = raw["config"] if "config" in raw else raw

    if "dataset_path" in config:
        config["dataset_path"] = Path(config["dataset_path"])
    if "checkpoint_dir" in config:
        config["checkpoint_dir"] = Path(config["checkpoint_dir"])
    if config.get("resume_from"):
        config["resume_from"] = Path(config["resume_from"])

    return TrainerConfig(**config)


def create_model(model_type: str) -> nn.Module:
    """Create model based on model type string."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return GPT2(MODEL_CONFIGS[model_type])


def main() -> None:
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--dataset", type=str, help="Dataset path (overrides config).")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model type (overrides config).",
    )
    parser.add_argument("--max-iters", type=int, help="Max training iterations (overrides config).")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config).")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config).")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    # Load config from YAML
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # CLI overrides
    if args.dataset:
        config.dataset_path = Path(args.dataset)
    if args.model:
        config.model_type = args.model
    if args.max_iters:
        config.max_iters = args.max_iters
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.resume:
        config.resume_from = Path(args.resume)

    # Log configuration
    logger.info("Training configuration:")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  Model: {config.model_type}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Max iterations: {config.max_iters}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Block size: {config.block_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        DataLoaderConfig(
            dataset_path=config.dataset_path,
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=config.device,
            split="train",
        )
    )
    val_loader = DataLoader(
        DataLoaderConfig(
            dataset_path=config.dataset_path,
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=config.device,
            split="val",
        )
    )
    logger.info(f"  Train batches per epoch: {len(train_loader)}")
    logger.info(f"  Val batches per epoch: {len(val_loader)}")

    # Create model
    logger.info(f"Creating {config.model_type} model...")
    model = create_model(config.model_type)
    model = model.to(config.device)
    logger.info(f"  Model parameters: {count_model_params(model):.2f}M")

    if config.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)  # type: ignore[assignment]

    # Train
    trainer = Trainer(
        config=config, model=model, train_dataloader=train_loader, val_dataloader=val_loader
    )

    if config.resume_from:
        logger.info(f"Resuming from checkpoint: {config.resume_from}")
        trainer.load_checkpoint(config.resume_from)

    logger.info("Starting training...")
    history = trainer.train()

    # Plot losses
    trainer.plot_losses(history)

    # Log final stats
    logger.info("Training complete!")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        logger.info(f"  Best val loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
