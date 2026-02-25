"""SFT training utilities.

Created by @pytholic on 2026.02.25
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from legollm.core.interfaces import Tokenizer
from legollm.generation import SamplingStrategy, generate_and_decode
from legollm.logging import logger
from legollm.post_training.sft.instruction_dataset import format_input


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: Tokenizer,
    num_epochs: int = 5,
    eval_freq: int = 5,
    eval_iter: int = 5,
    grad_clip: float = 1.0,
    checkpoint_interval: int = 1,
    model_name: str = "gpt2",
    grad_accumulation_steps: int = 1,
) -> dict[str, list[float]]:
    """Train the model via supervised fine-tuning.

    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        tokenizer: The tokenizer to use.
        num_epochs: The number of epochs to train for.
        eval_freq: Evaluate every N steps within an epoch.
        eval_iter: Number of val batches to use per evaluation.
        grad_clip: The gradient clip value.
        checkpoint_interval: The interval (in epochs) to save checkpoints.
        model_name: Model variant name for best checkpoint filename.
            Available: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl".
        grad_accumulation_steps: The number of steps to accumulate gradients before updating the model.

    Returns:
        Dictionary with "train_loss", "val_loss", and "steps" lists.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    best_val_loss = float("inf")
    val_loss = float("inf")
    global_step = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "steps": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        for step, (inputs, targets) in enumerate(train_loader):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            loss.backward()

            if (step + 1) % grad_accumulation_steps == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % eval_freq == 0:
                    val_loss = evaluate(model, val_loader, eval_iter)
                    logger.info(
                        f"Epoch {epoch}, Step {global_step}: "
                        f"train_loss={loss:.4f}, val_loss={val_loss:.4f}"
                    )
                    history["train_loss"].append(loss.item())
                    history["val_loss"].append(val_loss)
                    history["steps"].append(global_step)
                    model.train()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, epoch, val_loss, f"{model_name}-sft.pt")
                        logger.info(f"New best val_loss: {val_loss:.4f}")

        # Periodic checkpointing
        if epoch % checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(model, optimizer, epoch, val_loss)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Generate sample text after each epoch
        sample = val_loader.dataset.data[0]
        prompt = format_input(sample)
        response_text = "\n\n### Response:\n"
        full_prompt = prompt + response_text
        generated_text = generate_and_decode(
            prompt=full_prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=256,
            temperature=0.7,
            top_k=20,
            top_p=0.99,
            strategy=SamplingStrategy.STOCHASTIC,
            eos_token_id=tokenizer.eot_token,
            stream=False,
        )
        generated_response = str(generated_text).strip()
        print(f"Instruction: {sample['instruction']}")
        print(f"Expected:    {sample['output']}")
        print(f"Generated:   {generated_response}")
        print("=" * 60)

    return history


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    filename: str | None = None,
) -> Path:
    """Save the model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer.
        epoch: The epoch number.
        val_loss: The validation loss.
        filename: Optional filename. Defaults to checkpoint_{epoch}.pt.
    """
    checkpoint_dir = Path("checkpoints/sft")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if filename is None:
        filename = f"checkpoint_{epoch}.pt"
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    num_batches: int | None = None,
) -> float:
    """Evaluate the model.

    Args:
        model: The model to evaluate.
        val_loader: The validation data loader.
        num_batches: Number of batches to evaluate on. None = all batches.

    Returns:
        The average validation loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if num_batches is not None and i >= num_batches:
                break
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            losses.append(loss.item())
    val_loss = sum(losses) / len(losses)
    return val_loss


def plot_losses(
    steps: list[float], train_losses: list[float], val_losses: list[float], save_path: str | Path
) -> None:
    """Plot the training and validation losses.

    Args:
        steps: The global step numbers for each logged point.
        train_losses: The training losses.
        val_losses: The validation losses.
        save_path: The path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_losses, color="#1f77b4", label="Train")
    ax.plot(steps, val_losses, color="#ff7f0e", label="Val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
