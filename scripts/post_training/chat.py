"""Interactive chat with a fine-tuned GPT-2 model.

Usage:
    uv run python scripts/post_training/chat.py
    uv run python scripts/post_training/chat.py --checkpoint checkpoints/sft/gpt2-medium-sft.pt
"""

import argparse
from pathlib import Path

import tiktoken
import torch

from legollm.architectures.gpt2 import GPT2, MODEL_CONFIGS, GPT2Variant
from legollm.generation import SamplingStrategy, generate_and_decode
from legollm.logging import logger
from legollm.post_training.sft import format_input
from legollm.utils import get_device, load_pretrained_state_dict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactive chat with fine-tuned GPT-2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sft/gpt2-medium-sft.pt",
        help="Path to fine-tuned checkpoint (default: checkpoints/sft/gpt2-medium-sft.pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-medium",
        choices=[v.value for v in GPT2Variant],
        help="GPT-2 variant (default: gpt2-medium)",
    )
    return parser.parse_args()


def chat_loop(model: torch.nn.Module, tokenizer: tiktoken.Encoding) -> None:
    """Interactive chat loop."""
    print("\nChat started!")
    print("Type your instruction or 'quit' to exit.\n")

    while True:
        try:
            instruction = input("User: ").strip()
            if instruction.lower() in ("exit", "quit", "bye"):
                print("Goodbye!")
                break
            if not instruction:
                continue

            sample = {"instruction": instruction, "input": ""}
            prompt = format_input(sample) + "\n\n### Response:\n"

            token_stream = generate_and_decode(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=1024,
                temperature=0.5,
                top_k=50,
                top_p=0.99,
                strategy=SamplingStrategy.STOCHASTIC,
                eos_token_id=tokenizer.eot_token,
                stream=True,
            )

            for token in token_stream:
                print(token, end="", flush=True)
            print("\n")

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break


def main() -> None:
    """Load model and start chat."""
    args = parse_args()
    device = get_device()

    variant = GPT2Variant(args.model)
    model = GPT2(MODEL_CONFIGS[variant])
    state_dict = load_pretrained_state_dict(Path(args.checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    tokenizer = tiktoken.get_encoding("gpt2")

    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
