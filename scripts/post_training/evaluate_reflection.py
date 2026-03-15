r"""Compare baseline vs reflection model on a shared, fixed reference.

Baseline responses are scored against the original reference answers.
Reflection responses are also scored against the original reference answers (not the
reflection-improved ones), so both models are judged on the same bar.

Usage:
    uv run python scripts/post_training/evaluate_reflection.py
    uv run python scripts/post_training/evaluate_reflection.py \\
        --baseline data/finetuning/instruction_dataset_test_responses.json \\
        --reflection data/finetuning/instruction_dataset_test_responses_reflection.json \\
        --judge-model deepseek-r1:14b
"""

import argparse
import json

from legollm.post_training.providers.base import LLMOptions
from legollm.post_training.providers.ollama_provider import BackendType, create_ollama_backend
from legollm.post_training.sft.evaluate import generate_model_scores


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare baseline vs reflection model scores")
    parser.add_argument(
        "--baseline",
        type=str,
        default="data/finetuning/instruction_dataset_test_responses.json",
        help="Baseline model response file (source of original reference answers)",
    )
    parser.add_argument(
        "--reflection",
        type=str,
        default="data/finetuning/instruction_dataset_test_responses_reflection.json",
        help="Reflection model response file",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="deepseek-r1:14b",
        help="Ollama judge model (default: deepseek-r1:14b)",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        default="http",
        choices=[b.value for b in BackendType],
        help="Ollama backend (default: http)",
    )
    return parser.parse_args()


def main() -> None:
    """Run LLM-judge evaluation."""
    args = parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(args.reflection, encoding="utf-8") as f:
        reflection_data = json.load(f)

    if len(baseline_data) != len(reflection_data):
        raise ValueError(
            f"Sample count mismatch: baseline={len(baseline_data)}, reflection={len(reflection_data)}. "
            "Both files must come from the same test split."
        )

    # Pair reflection model's generated_response with baseline's original instruction/output
    # so both models are scored against the same reference answers.
    reflection_paired = [
        {**baseline_data[i], "generated_response": reflection_data[i]["generated_response"]}
        for i in range(len(baseline_data))
    ]

    options = LLMOptions(
        temperature=0.7,
        num_ctx=8192,
        top_k=40,
        top_p=0.9,
        num_predict=8192,
    )
    provider = create_ollama_backend(BackendType(args.judge_backend))

    baseline_scores = generate_model_scores(
        baseline_data, provider=provider, model=args.judge_model, options=options
    )
    reflection_scores = generate_model_scores(
        reflection_paired, provider=provider, model=args.judge_model, options=options
    )

    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    reflection_avg = sum(reflection_scores) / len(reflection_scores) if reflection_scores else 0.0
    delta = reflection_avg - baseline_avg

    print("\n" + "=" * 50)
    print(f"  Baseline   ({len(baseline_scores)} samples): {baseline_avg:.2f}")
    print(f"  Reflection ({len(reflection_scores)} samples): {reflection_avg:.2f}")
    print(f"  Delta: {delta:+.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
