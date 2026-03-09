"""Reflection: a tool for LLMs to reflect on their own behavior and improve.

Usage examples:

uv run python scripts/post_training/reflection.py --provider openai --model gpt-5-nano --mode response

uv run python scripts/post_training/reflection.py --provider ollama --model qwen3.5:9b --mode instruction --ollama-backend http

# 8 GPUs, 1 worker each (default)
make ollama-serve NUM_GPUS=8
uv run python scripts/post_training/reflection.py --num-gpus 8 --workers-per-gpu 1

# 1 GPU, 4 concurrent workers
make ollama-serve NUM_GPUS=1 WORKERS_PER_GPU=4
uv run python scripts/post_training/reflection.py --num-gpus 1 --workers-per-gpu 4

# 4 GPUs, 2 workers each = 8 total concurrent requests
make ollama-serve NUM_GPUS=4 WORKERS_PER_GPU=2
uv run python scripts/post_training/reflection.py --num-gpus 4 --workers-per-gpu 2

Note:
- The instruction mode rewrites both the instruction and the answer.
- The response mode only rewrites the answer.
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum

from legollm.logging import logger, progress_bar
from legollm.post_training.providers.base import LLMOptions, LLMProvider, Message, ProviderType
from legollm.post_training.providers.ollama_provider import BackendType, create_ollama_backend
from legollm.post_training.providers.openai_provider import OpenAIProvider
from legollm.post_training.sft.instruction_dataset import (
    download_and_load_instruction_dataset,
)


class ReflectionMode(StrEnum):
    """The mode of reflection."""

    RESPONSE = "response"
    INSTRUCTION = "instruction"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reflection")
    parser.add_argument(
        "--mode",
        type=ReflectionMode,
        default=ReflectionMode.RESPONSE,
        choices=[v.value for v in ReflectionMode],
        help="The mode of reflection (default: response, choices: response, instruction).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:9b",
        help="The model to use for reflection (default: qwen3.5:9b).",
    )
    parser.add_argument(
        "--provider",
        type=ProviderType,
        default=ProviderType.OLLAMA,
        choices=[v.value for v in ProviderType],
        help="LLM provider: 'ollama' or 'openai' (default: ollama).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI provider (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL override for OpenAI-compatible APIs (vLLM, Together, etc.).",
    )
    parser.add_argument(
        "--ollama-backend",
        type=str,
        default="http",
        choices=["http", "sdk"],
        help="Ollama transport: 'http' (raw requests) or 'sdk' (ollama-python library).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="localhost",
        help="Ollama host (default: 'localhost').",
    )
    parser.add_argument(
        "--ollama-port",
        type=int,
        default=11434,
        help="Ollama port (default: 11434).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of Ollama instances (one per GPU, default: 1).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent requests per Ollama instance (default: 1). "
        "Set OLLAMA_NUM_PARALLEL to the same value when starting ollama serve.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help="Base port for parallel Ollama instances. Instance i uses base-port+i. "
        "Defaults to --ollama-port when --num-gpus=1.",
    )
    return parser.parse_args()


_RESPONSE_EXAMPLE = """Here is an example of the expected output format:

[Instruction]
Convert 45 kilometers to meters.

[The Start of Answer]
45 kilometers is equal to 45,000 meters.
[The End of Answer]

1. Why is this answer not good? Analyse based on Helpfulness, Relevance, Accuracy, and Level of Details.

- **Helpfulness:** The answer is helpful in that it provides the correct conversion.
- **Relevance:** The answer is relevant to the instruction, as it directly addresses the conversion requested.
- **Accuracy:** The answer is accurate; 45 kilometers does indeed equal 45,000 meters.
- **Level of Details:** The answer lacks detail. It does not explain the conversion process or provide any context, which could be beneficial for someone unfamiliar with metric conversions.

2. Based on the reason you provided, generate a better answer:

[Better Answer] To convert 45 kilometers to meters, we use the fact that 1 kilometer equals 1,000 meters. Therefore, 45 kilometers x 1,000 = 45,000 meters. This conversion is based on the metric system, where the prefix "kilo-" always means 1,000 of the base unit. [End]"""


def _build_sample_block(instruction: str, instruction_input: str, output: str) -> str:
    """Build the [Instruction] / [Input] / [Answer] block for a sample."""
    parts = [f"[Instruction]\n{instruction}"]
    if instruction_input:
        parts.append(f"[The Start of Input]\n{instruction_input}\n[The End of Input]")
    parts.append(f"[The Start of Answer]\n{output}\n[The End of Answer]")
    return "\n\n".join(parts)


def build_response_reflection_prompt(
    instruction: str, instruction_input: str, output: str
) -> tuple[str, str]:
    """Build a prompt that reflects on the response quality."""
    system_prompt = "You are a helpful, precise but picky assistant for checking the quality of the answer to a given instruction."

    task = """1. Why is this answer not good? Analyse based on Helpfulness, Relevance, Accuracy, and Level of Details.
2. Based on the reason you provided, generate a better answer, new and complete, as detailed as possible.

Format your better answer exactly as: [Better Answer] your answer [End]

IMPORTANT:
- Do NOT repeat your answer or continue writing after [End]."""

    sample = _build_sample_block(instruction, instruction_input, output)

    prompt = f"""{task}

{_RESPONSE_EXAMPLE}

Now do the same for the following:

{sample}"""

    return system_prompt, prompt.strip()


def build_instruction_reflection_prompt(
    instruction: str, instruction_input: str, output: str
) -> tuple[str, str]:
    """Build a prompt that reflects on the instruction quality."""
    system_prompt = "You are a helpful, precise but picky assistant for checking the quality of a given instruction."

    sample = _build_sample_block(instruction, instruction_input, output)

    task = """
We would like you to evaluate and improve the quality of the instruction.

1. Evaluate the instruction on:
   - **Complexity:** Is the topic trivial or does it require deeper knowledge?
   - **Clarity:** Is the instruction clear and unambiguous?
   - **Detail requirement:** Does it ask for enough detail to produce a thorough answer?
   - **Reasoning:** Does it require problem-solving or just recall?

2. Explain why the current instruction could lead to a shallow or suboptimal answer.

3. Generate a new, improved instruction that:
   - Is more complex and thought-provoking
   - Is self-contained (answerable without external context)
   - Would lead to a more detailed, higher-quality answer

4. Provide a detailed response to your new instruction.
    """

    prompt = f"""{task}

Format your output exactly as:
[New Instruction] your improved instruction [End]
[New Answer] your detailed answer to the new instruction [End]

IMPORTANT:
- Do NOT repeat or continue writing after the final [End].

{sample}"""

    return system_prompt, prompt.strip()


def extract_instruction_and_output(text: str) -> tuple[str, str] | tuple[None, None]:
    """Extract [New Instruction] and [New Answer] blocks from reflected text."""
    # Find the [New Instruction] and [New Answer] blocks
    instruction_match = re.search(r"\[New Instruction\](.*?)\[End\]", text, re.DOTALL)
    output_match = re.search(r"\[New Answer\](.*?)\[End\]", text, re.DOTALL)
    if instruction_match and output_match:
        return instruction_match.group(1).strip(), output_match.group(1).strip()
    return None, None


def _create_provider(args: argparse.Namespace) -> LLMProvider:
    """Instantiate the LLM provider from CLI args."""
    if args.provider == ProviderType.OPENAI:
        return OpenAIProvider(api_key=args.api_key)
    elif args.provider == ProviderType.OLLAMA:
        return create_ollama_backend(
            args.ollama_backend or BackendType.HTTP,
            host=args.ollama_host,
            port=args.ollama_port,
        )
    else:
        raise ValueError(f"Invalid provider: {args.provider}")


def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Set up LLM options
    if args.provider == ProviderType.OPENAI:
        opts = LLMOptions(
            num_ctx=8192,
            num_predict=8192,
        )
    elif args.provider == ProviderType.OLLAMA:
        opts = LLMOptions(
            temperature=0.3,  # lower than generation — we want accurate rewrites, not creative
            num_ctx=8192,
            num_predict=8192,
            top_k=40,
            top_p=0.9,
        )
    else:
        raise ValueError(f"Invalid provider: {args.provider}")

    # Create provider(s)
    # For Ollama: spin up one provider per GPU instance (round-robin across ports).
    # For OpenAI: single provider, no parallelism needed.
    if args.provider == ProviderType.OLLAMA and args.num_gpus > 1:
        base_port = args.base_port or args.ollama_port
        providers = [
            create_ollama_backend(
                args.ollama_backend or BackendType.HTTP,
                host=args.ollama_host,
                port=base_port + i,
            )
            for i in range(args.num_gpus)
        ]
    else:
        providers = [_create_provider(args)]

    # Load instruction dataset
    logger.info("Loading instruction dataset...")
    file_path = "data/finetuning/instruction_dataset.json"
    url = "https://raw.githubusercontent.com/pytholic/LegoLLM/refs/heads/main/data/finetuning/instruction_dataset.json"
    data = download_and_load_instruction_dataset(file_path, url)
    logger.info(f"Loaded {len(data)} samples")

    logger.info(f"Running {args.model} (mode={args.mode}) with options: {opts.to_dict()}")

    # Pick prompt builder based on mode
    if args.mode == ReflectionMode.INSTRUCTION:
        build_prompt = build_instruction_reflection_prompt
    else:
        build_prompt = build_response_reflection_prompt

    output_path = f"data/finetuning/instruction_dataset_reflection_{args.mode.value}.jsonl"
    write_lock = threading.Lock()

    def process_sample(sample: dict[str, str], provider: LLMProvider) -> None:
        """Process a sample using the given provider."""
        instr = sample["instruction"]
        inp = sample["input"]
        out = sample["output"]

        system_prompt, prompt = build_prompt(instr, inp, out)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt),
        ]

        response = provider.chat(
            messages, model=args.model, options=opts, output_format=None, stream=False
        )
        instruction, output = extract_instruction_and_output(response)
        if not (instruction and output):
            logger.error(f"Failed to extract instruction and output from response: {response}")

        with write_lock:
            with open(output_path, "a") as f:
                entry = {"instruction": instruction, "input": inp, "output": output}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Reflection loop: total_workers = num_gpus * workers_per_gpu
    # Each provider (GPU instance) can handle workers_per_gpu concurrent requests.
    num_workers = len(providers) * args.workers_per_gpu
    with progress_bar("Reflecting", total=len(data)) as (progress, task):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_sample, sample, providers[i % len(providers)]): sample
                for i, sample in enumerate(data)
            }
            for future in as_completed(futures):
                future.result()  # re-raises any exception from the worker
                progress.update(task, advance=1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    main()
