"""Generate text from a trained model checkpoint.

Usage:
    uv run python scripts/generate.py --model-path checkpoints/best.pt --prompt "Every effort" --max-new-tokens 50
    uv run python scripts/generate.py --model-path checkpoints/best.pt --prompt "Every" --strategy greedy
"""

import argparse

import tiktoken

from legollm.architectures.gpt2 import GPT2, GPT2_CONFIG_124M, GPT2Variant, load_gpt2_weights
from legollm.generation import SamplingStrategy, generate_and_decode
from legollm.logging import logger


def main() -> None:
    """Generate text from a trained model."""
    parser = argparse.ArgumentParser(description="Generate text from a trained model checkpoint.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=50, help="Max new tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling (0=greedy)."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering (None=disabled).")
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p filtering (None=disabled)."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "stochastic"],
        default="stochastic",
        help="Sampling strategy.",
    )
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu/mps/cuda).")
    parser.add_argument("--stream", action="store_true", help="Stream the generated text.")
    args = parser.parse_args()

    # Load tokenizer
    ## Tiktoken tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")

    ## Custom RegexBPE tokenizer
    # tokenizer = RegexBPETokenizer()
    # # Try to load The Verdict tokenizer; fall back to untrained if not found
    # try:
    #     from legollm.config import TOKENIZERS_DIR

    #     tokenizer.load(str(TOKENIZERS_DIR / "the_verdict_regex_bpe.json"))
    #     logger.info("Loaded The Verdict tokenizer")
    # except FileNotFoundError:
    #     logger.warning("The Verdict tokenizer not found; using untrained tokenizer")

    # Load model
    logger.info("Loading model...")
    model = GPT2(GPT2_CONFIG_124M)
    model = model.to(args.device)

    # # Load checkpoint
    # logger.info(f"Loading checkpoint from {args.model_path}...")
    # checkpoint = load_pretrained_state_dict(
    #     args.model_path, map_location=args.device, weights_only=False
    # )
    # model.load_state_dict(checkpoint)
    # model.eval()
    # logger.info("Model loaded and ready for generation")

    # Load weights from HuggingFace
    logger.info("Loading weights from HuggingFace...")
    load_gpt2_weights(model, variant=GPT2Variant.GPT2)
    model.eval()
    logger.info("Model loaded and ready for generation")

    strategy = SamplingStrategy.GREEDY if args.strategy == "greedy" else SamplingStrategy.STOCHASTIC

    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    # Stream decoded tokens as they are generated
    for token_text in generate_and_decode(
        prompt=args.prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        strategy=strategy,
        stream=args.stream,
    ):
        print(token_text, end="", flush=True)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
