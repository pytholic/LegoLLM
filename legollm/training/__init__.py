"""Training module for LegoLLM.

Provides:
- Trainer: Main training loop with evaluation and checkpointing
- TrainerConfig: Configuration dataclass for training hyperparameters

Created by @pytholic on 2026.02.02
"""

from legollm.training.trainer import Trainer, TrainerConfig

__all__ = ["Trainer", "TrainerConfig"]
