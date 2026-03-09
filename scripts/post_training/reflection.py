"""Reflection: a tool for LLMs to reflect on their own behavior and improve.

Usage examples:

uv run python scripts/post_training/reflection.py --provider openai --model gpt-5-nano --mode response

uv run python scripts/post_training/reflection.py --provider ollama --model qwen3.5:9b --mode instruction --ollama-backend http

# 8 instances across 8 GPUs
make ollama-serve NUM_INSTANCES=8
uv run python scripts/post_training/reflection.py --num-instances 8

# 4 instances on the same GPU (for models that don't support OLLAMA_NUM_PARALLEL)
make ollama-serve NUM_INSTANCES=4 GPU_IDS=0,0,0,0
uv run python scripts/post_training/reflection.py --num-instances 4

# 4 instances across 4 GPUs, 2 workers each = 8 total concurrent requests
make ollama-serve NUM_INSTANCES=4 WORKERS_PER_GPU=2
uv run python scripts/post_training/reflection.py --num-instances 4 --workers-per-instance 2

# For failed cases, use the input file
uv run python scripts/post_training/reflection.py --provider ollama --model deepseek-r1:14b --mode instruction --ollama-backend http --num-instances 3 --base-port 11434 --input data/finetuning/instruction_dataset_reflection_instruction_failed.jsonl

Steps:

Step 1: Start instances on the same GPU
make ollama-serve NUM_INSTANCES=3 GPU_IDS=0,0,0 WORKERS_PER_GPU=1 OLLAMA_BIN=/usr/local/bin/ollama BASE_PORT=11434 MODEL=deepseek-r1:14b

Step 2: Pull the model on all instances
make ollama-pull NUM_INSTANCES=3 OLLAMA_BIN=/usr/local/bin/ollama BASE_PORT=11434 MODEL=deepseek-r1:14b

Step 3: Run the reflection script on instances
uv run python scripts/post_training/reflection.py --provider ollama --model deepseek-r1:14b --mode instruction --ollama-backend http --num-instances 3 --base-port 11434

Step 4: Stop all ollama instances
pkill ollama

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
        "--num-instances",
        type=int,
        default=1,
        help="Number of Ollama instances (can share a GPU, default: 1).",
    )
    parser.add_argument(
        "--workers-per-instance",
        type=int,
        default=1,
        help="Concurrent requests per Ollama instance (default: 1). "
        "Set OLLAMA_NUM_PARALLEL to the same value when starting ollama serve.",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        default=False,
        help="Disable thinking mode (required for models that don't support it, e.g. Gemma3).",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help="Base port for parallel Ollama instances. Instance i uses base-port+i. "
        "Defaults to --ollama-port when --num-instances=1.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON or JSONL file to use as input instead of the default dataset.",
    )
    return parser.parse_args()


_INSTRUCTION_EXAMPLE = """Here is an example of the expected output format:

[Instruction]
What is the capital of France?

[The Start of Answer]
Paris.
[The End of Answer]

1. Evaluation:
- **Complexity:** Very low — this is a simple factual recall question.
- **Clarity:** Clear and unambiguous.
- **Detail requirement:** None — it only asks for a single fact.
- **Reasoning:** Pure recall, no problem-solving.

2. The current instruction leads to a one-word answer with no depth. It does not ask the learner to think critically or demonstrate understanding.

3-4. Improved instruction and answer:

[New Instruction] Explain Paris's historical, political, and cultural significance as the capital of France, including how its role has evolved from the medieval period to the present day. [End]
[New Answer] Paris has served as the capital of France since the late 10th century when Hugh Capet, the first Capetian king, established it as his seat of power. Historically, the city grew around the Île de la Cité on the Seine, benefiting from its strategic position for trade and defense. During the medieval period, Paris became a center of learning with the founding of the University of Paris (c. 1150), one of Europe's earliest universities. Politically, Paris has been the epicenter of major events including the French Revolution (1789), which transformed France from an absolute monarchy into a republic. The city housed the National Assembly and became synonymous with democratic ideals. Culturally, Paris earned its reputation as the "City of Light" during the Enlightenment and later as a hub for art movements including Impressionism and Existentialism. Today, Paris serves as France's administrative capital, hosting the Élysée Palace (presidency), the National Assembly, and the Senate, while remaining a global center for fashion, gastronomy, and diplomacy. [End]"""

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

{_INSTRUCTION_EXAMPLE}

Now do the same for the following. You MUST use the exact tags [New Instruction], [New Answer], and [End] as shown in the example above. Do NOT wrap these tags in markdown bold (**) or any other formatting.

Format your output exactly as:
[New Instruction] your improved instruction [End]
[New Answer] your detailed answer to the new instruction [End]

IMPORTANT:
- Use the EXACT tags: [New Instruction], [New Answer], and [End] — no markdown, no bold, no extra brackets.
- Do NOT repeat or continue writing after the final [End].

{sample}"""

    return system_prompt, prompt.strip()


def extract_instruction_and_output(text: str) -> tuple[str, str] | tuple[None, None]:
    """Extract [New Instruction] and [New Answer] blocks from reflected text."""
    # Strip markdown bold/formatting around tags: **[New Instruction]** -> [New Instruction]
    cleaned = re.sub(r"\*{1,2}\[", "[", text)
    cleaned = re.sub(r"\]\*{1,2}", "]", cleaned)

    instruction_match = re.search(r"\[New Instruction\]\s*(.*?)\s*\[End\]", cleaned, re.DOTALL)
    output_match = re.search(r"\[New Answer\]\s*(.*?)\s*\[End\]", cleaned, re.DOTALL)
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
    if args.provider == ProviderType.OLLAMA and args.num_instances > 1:
        base_port = args.base_port or args.ollama_port
        providers = [
            create_ollama_backend(
                args.ollama_backend or BackendType.HTTP,
                host=args.ollama_host,
                port=base_port + i,
            )
            for i in range(args.num_instances)
        ]
    else:
        providers = [_create_provider(args)]

    # Load instruction dataset
    logger.info("Loading instruction dataset...")
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            if args.input.endswith(".jsonl"):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
    else:
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
    failed_path = f"data/finetuning/instruction_dataset_reflection_{args.mode.value}_failed.jsonl"
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

        from legollm.post_training.providers.ollama_provider import HttpBackend, SdkBackend

        if isinstance(provider, HttpBackend | SdkBackend):
            response = provider.chat(
                messages,
                model=args.model,
                options=opts,
                output_format=None,
                stream=False,
                think=not args.no_think,
            )
        else:
            response = provider.chat(
                messages,
                model=args.model,
                options=opts,
                output_format=None,
                stream=False,
            )
        instruction, output = extract_instruction_and_output(response)

        with write_lock:
            if instruction and output:
                with open(output_path, "a") as f:
                    entry = {"instruction": instruction, "input": inp, "output": output}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                logger.error("Failed to extract instruction and output from response")
                with open(failed_path, "a") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Reflection loop: total_workers = num_instances * workers_per_instance
    num_workers = len(providers) * args.workers_per_instance
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
