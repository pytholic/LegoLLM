"""Instruction fine-tuning script for GPT-2.

Usage:
    # Baseline
    uv run python scripts/post_training/sft.py

    # Reflection (reflection-improved train, same locked test/val)
    uv run python scripts/post_training/sft.py \
        --train data/finetuning/instruction_dataset_reflection_instruction.jsonl \
        --checkpoint-suffix sft-reflection \
        --responses-out data/finetuning/instruction_dataset_test_responses_reflection.json
"""

import argparse
import json
from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

from legollm.architectures.gpt2 import GPT2, MODEL_CONFIGS, GPT2Variant, load_gpt2_weights
from legollm.generation import SamplingStrategy, generate_and_decode
from legollm.logging import logger, progress_bar
from legollm.post_training.sft import (
    InstructionDataset,
    custom_collate_fn,
    format_input,
    plot_losses,
    train,
)
from legollm.utils import get_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Instruction fine-tuning for GPT-2")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-medium",
        choices=[v.value for v in GPT2Variant],
        help="GPT-2 variant (default: gpt2-medium)",
    )
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length for collate truncation (default: 1024)",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="data/finetuning/instruction_dataset_train.json",
        help="Training split file (.json or .jsonl, default: data/finetuning/instruction_dataset_train.json)",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="data/finetuning/instruction_dataset_val.json",
        help="Validation split file (.json or .jsonl, default: data/finetuning/instruction_dataset_val.json)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/finetuning/instruction_dataset_test.json",
        help="Test split file (.json or .jsonl, default: data/finetuning/instruction_dataset_test.json)",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="sft",
        help="Suffix for checkpoint filename: checkpoints/sft/{model}-{suffix}.pt (default: sft)",
    )
    parser.add_argument(
        "--responses-out",
        type=str,
        default="data/finetuning/instruction_dataset_test_responses.json",
        help="Output path for test responses JSON (default: data/finetuning/instruction_dataset_test_responses.json)",
    )
    return parser.parse_args()


def _load_split(path: str) -> list[dict[str, str]]:
    """Load a JSON or JSONL split file."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def main() -> None:
    """Run instruction fine-tuning."""
    args = parse_args()

    train_data = _load_split(args.train)
    val_data = _load_split(args.val)
    test_data = _load_split(args.test)
    logger.info(
        f"Loaded splits — train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}"
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=args.max_length,
        pad_token_id=tokenizer.eot_token,
    )

    torch.manual_seed(42)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Load pretrained model
    variant = GPT2Variant(args.model)
    model = GPT2(MODEL_CONFIGS[variant])
    load_gpt2_weights(model, variant=variant)
    model = model.to(device)
    logger.info("Model loaded and ready for training")

    # Train
    history = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        num_epochs=args.num_epochs,
        grad_clip=1.0,
        checkpoint_interval=1,
        model_name=variant.value,
        grad_accumulation_steps=args.grad_accumulation_steps,
        checkpoint_suffix=args.checkpoint_suffix,
    )

    plot_losses(
        history["steps"],
        history["train_loss"],
        history["val_loss"],
        f"checkpoints/sft/loss_{args.checkpoint_suffix}.png",
    )

    # Load best checkpoint for evaluation
    best_checkpoint_path = f"checkpoints/sft/{variant}-{args.checkpoint_suffix}.pt"
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded best checkpoint from {best_checkpoint_path}")

    # Preview a few responses
    for sample in test_data[:3]:
        input_prompt = format_input(sample)
        full_prompt = input_prompt + "\n\n### Response:\n"
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
        print(input_prompt)
        print(f"\nCorrect response:\n>> {sample['output']}")
        print(f"\nModel response:\n>> {generated_text.strip()}")
        print("-" * 50)

    # Extract all test responses for evaluation
    with progress_bar("Extracting test_data responses", total=len(test_data)) as (progress, task):
        for i, sample in enumerate(test_data):
            full_prompt = format_input(sample) + "\n\n### Response:\n"
            generated_text = generate_and_decode(
                prompt=full_prompt,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=256,
                temperature=0.5,
                top_k=20,
                top_p=0.99,
                strategy=SamplingStrategy.STOCHASTIC,
                eos_token_id=tokenizer.eot_token,
                stream=False,
            )
            test_data[i]["generated_response"] = str(generated_text).strip()
            progress.update(task, advance=1)

    with open(args.responses_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Test responses saved to {args.responses_out}")


if __name__ == "__main__":
    main()
