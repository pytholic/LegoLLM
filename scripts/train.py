"""Train a language model.

Usage:
    python scripts/train.py --config configs/train_gpt2.yaml
    python scripts/train.py --dataset data/processed/the_verdict --model gpt2-124m

Created by @pytholic on 2025.12.15
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from legollm.architectures.gpt2 import GPT2, GPT2_CONFIG_124M, GPT2_CONFIG_355M
from legollm.data.dataloader import DataLoader, DataLoaderConfig
from legollm.logging import logger
from legollm.training import Trainer, TrainerConfig
from legollm.utils import count_model_params

# Available model configurations
MODEL_CONFIGS = {
    "gpt2-124m": GPT2_CONFIG_124M,
    "gpt2-355m": GPT2_CONFIG_355M,
}


@dataclass
class TrainScriptConfig:
    """Configuration for the training script.

    Defaults are optimized for small datasets (the_verdict, tiny_shakespeare).
    For large-scale training (OpenWebText), use a YAML config with nanoGPT values:
        - max_iters: 600000
        - block_size: 1024
        - batch_size: 12
        - warmup_iters: 2000
        - lr_decay_iters: 600000

    Reference: https://github.com/karpathy/nanoGPT
    """

    # Data
    dataset_path: Path = Path("data/processed/the_verdict")
    block_size: int = 256  # small dataset context length
    batch_size: int = 8  # small batch for limited data

    # Model
    model_type: str = "gpt2-124m"

    # Training - small dataset defaults
    max_iters: int = 5000  # enough for tiny datasets to converge
    eval_interval: int = 500
    log_interval: int = 10
    checkpoint_interval: int = 1000
    checkpoint_dir: Path = Path("checkpoints")

    # Optimizer (AdamW) - same as nanoGPT
    learning_rate: float = 6e-4  # max learning rate
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

    # LR schedule - scaled for small datasets
    decay_lr: bool = True
    warmup_iters: int = 100  # ~2% of max_iters
    lr_decay_iters: int = 5000  # match max_iters
    min_lr: float = 6e-5  # ~= learning_rate/10 per Chinchilla

    # Other
    grad_clip: float = 1.0
    compile: bool = False  # disable for faster startup on small runs
    eval_iters: int = 10  # fewer eval batches for small val sets
    device: str = field(
        default_factory=lambda: (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    # Resume training
    resume_from: Path | None = None


def load_config(yaml_path: str) -> TrainScriptConfig:
    """Load training configuration from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Convert path strings to Path objects
    if "dataset_path" in data:
        data["dataset_path"] = Path(data["dataset_path"])
    if "checkpoint_dir" in data:
        data["checkpoint_dir"] = Path(data["checkpoint_dir"])
    if data.get("resume_from"):
        data["resume_from"] = Path(data["resume_from"])

    return TrainScriptConfig(**data)


def create_model(config: TrainScriptConfig) -> GPT2:
    """Create model based on configuration."""
    if config.model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model type: {config.model_type}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    model_config = MODEL_CONFIGS[config.model_type]
    model = GPT2(model_config)
    return model


def main() -> None:
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to processed dataset directory (overrides config).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model type to train (overrides config).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        help="Maximum training iterations (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from.",
    )
    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = TrainScriptConfig()
        logger.info("Using default configuration")

    # Override with CLI arguments
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
    train_loader_config = DataLoaderConfig(
        dataset_path=config.dataset_path,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=config.device,
        split="train",
    )
    val_loader_config = DataLoaderConfig(
        dataset_path=config.dataset_path,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=config.device,
        split="val",
    )
    train_loader = DataLoader(train_loader_config)
    val_loader = DataLoader(val_loader_config)
    logger.info(f"  Train batches per epoch: {len(train_loader)}")
    logger.info(f"  Val batches per epoch: {len(val_loader)}")

    # Create model
    logger.info(f"Creating {config.model_type} model...")
    model = create_model(config)
    model = model.to(config.device)
    num_params = count_model_params(model)
    logger.info(f"  Model parameters: {num_params:.2f}M")

    # Optionally compile model
    if config.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Create trainer config
    trainer_config = TrainerConfig(
        max_iters=config.max_iters,
        eval_interval=config.eval_interval,
        log_interval=config.log_interval,
        checkpoint_interval=config.checkpoint_interval,
        checkpoint_dir=config.checkpoint_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        decay_lr=config.decay_lr,
        warmup_iters=config.warmup_iters,
        lr_decay_iters=config.lr_decay_iters,
        min_lr=config.min_lr,
        grad_clip=config.grad_clip,
        device=config.device,
        compile=config.compile,
        eval_iters=config.eval_iters,
    )

    # Create trainer
    trainer = Trainer(
        config=trainer_config,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    # Resume from checkpoint if specified
    if config.resume_from:
        logger.info(f"Resuming from checkpoint: {config.resume_from}")
        trainer.load_checkpoint(config.resume_from)

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Log final stats
    logger.info("Training complete!")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        logger.info(f"  Best val loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
