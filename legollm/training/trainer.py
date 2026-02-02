"""Training utilities for language models.

Created by @pytholic on 2025.12.15
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from legollm.data.dataloader import DataLoader
from legollm.logging import logger


@dataclass
class TrainerConfig:
    """Configuration for model training.

    Defaults are optimized for training GPT-2 (124M) on OpenWebText,
    following nanoGPT reference implementation.

    Reference: https://github.com/karpathy/nanoGPT
    """

    # Training
    max_iters: int = 600000
    eval_interval: int = 2000
    log_interval: int = 1

    # Checkpointing
    checkpoint_interval: int = 1000
    checkpoint_dir: Path = Path("checkpoints")

    # Optimizer (AdamW)
    learning_rate: float = 6e-4  # max learning rate
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Learning rate schedule (cosine decay with warmup)
    decay_lr: bool = True
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # ~= learning_rate/10 per Chinchilla

    # Gradient clipping
    grad_clip: float = 1.0

    # System
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    compile: bool = True  # use PyTorch 2.0 to compile the model

    # Evaluation
    eval_iters: int = 200  # Number of batches for validation

    def __post_init__(self) -> None:
        """Set derived values."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class Trainer:
    """Trainer for language models."""

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        """Initialize the Trainer."""
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Init training state
        self.iter_num = 0
        self.best_val_loss = float("inf")

        self.optimizer = self._configure_optimizer()
        self._update_learning_rate()

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer with weight decay."""
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No weight decay for biases and layer norms
                if "bias" in name or "ln" in name or "norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
        return optimizer

    def _update_learning_rate(self) -> None:
        """Update learning rate for current iteration."""
        lr = self._get_lr(self.iter_num) if self.config.decay_lr else self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, iter_num: int) -> float:
        """Calculate learning rate for current step."""
        if iter_num < self.config.warmup_iters:
            return self.config.learning_rate * (iter_num + 1) / (self.config.warmup_iters + 1)
        if iter_num > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (iter_num - self.config.warmup_iters) / (
            self.config.lr_decay_iters - self.config.warmup_iters
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Execute a single training step.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            y: Target tensor of shape (batch_size, seq_len)

        Returns:
            Loss value for this step
        """
        self.model.train()

        # Forward pass
        logits = self.model(x)  # (B, T, vocab_size)

        # Compute cross-entropy loss
        # Reshape for cross_entropy: (B*T, vocab_size) and (B*T,)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        losses = []

        val_iter = iter(self.val_dataloader)
        for _ in range(self.config.eval_iters):
            x, y = next(val_iter)
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())

        return sum(losses) / len(losses)

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> Path:
        """Save training checkpoint.

        Args:
            filename: Name of checkpoint file

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        path = self.config.checkpoint_dir / filename
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iter_num = checkpoint["iter_num"]
        self.best_val_loss = checkpoint["best_val_loss"]

    def train(self) -> dict[str, list[float]]:
        """Run the training loop.

        Returns:
            Dictionary with training history (losses, etc.)
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        train_iter = iter(self.train_dataloader)

        logger.info(f"Starting training for {self.config.max_iters} iterations")

        for self.iter_num in range(self.config.max_iters):
            # Update learning rate
            self._update_learning_rate()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Get batch and train
            x, y = next(train_iter)
            loss = self._train_step(x, y)

            # Logging
            if self.iter_num % self.config.log_interval == 0:
                logger.info(f"iter {self.iter_num}: loss={loss:.4f}, lr={current_lr:.2e}")
                history["train_loss"].append(loss)
                history["lr"].append(current_lr)

            # Evaluation
            if self.iter_num % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                logger.info(f"iter {self.iter_num}: val_loss={val_loss:.4f}")
                history["val_loss"].append(val_loss)

                # Save the best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                    logger.info(f"New best val_loss: {val_loss:.4f}")

            # Periodic checkpointing
            if self.iter_num % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.iter_num}.pt")
                logger.info(
                    f"Saved checkpoint to {self.config.checkpoint_dir}/checkpoint_{self.iter_num}.pt"
                )

        # Final checkpoint
        self.save_checkpoint("final.pt")
        logger.info(f"Saved final checkpoint to {self.config.checkpoint_dir}/final.pt")

        logger.info(f"Training completed. Best val_loss: {self.best_val_loss:.4f}")
        return history
