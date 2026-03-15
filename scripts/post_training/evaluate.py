"""LLM-judge evaluation script for finetuned model responses.

Usage:
    uv run python scripts/post_training/evaluate.py
    uv run python scripts/post_training/evaluate.py --file data/finetuning/instruction_dataset_test_responses_reflection.json --model deepseek-r1:14b
"""

import argparse
import json

from legollm.post_training.providers.base import LLMOptions
from legollm.post_training.providers.ollama_provider import BackendType, create_ollama_backend
from legollm.post_training.sft.evaluate import generate_model_scores


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-judge evaluation for finetuned model responses"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="data/finetuning/instruction_dataset_test_responses_reflection.json",
        help="Path to the test responses JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-r1:14b",
        help="Ollama judge model (default: deepseek-r1:14b)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="http",
        choices=[b.value for b in BackendType],
        help="Ollama backend (default: http)",
    )
    return parser.parse_args()


def main() -> None:
    """Run LLM-judge evaluation."""
    args = parse_args()

    with open(args.file, encoding="utf-8") as f:
        test_data = json.load(f)

    options = LLMOptions(
        temperature=0.7,
        num_ctx=8192,
        top_k=40,
        top_p=0.9,
        num_predict=8192,
    )

    provider = create_ollama_backend(BackendType(args.backend))
    scores = generate_model_scores(test_data, provider=provider, model=args.model, options=options)

    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    if scores:
        print(f"Average score: {sum(scores) / len(scores):.2f}\n")


if __name__ == "__main__":
    main()
