"""Evaluate the finetuned model on the test data using LLM judge.

Created by @pytholic on 2026.02.26
"""

import json

from legollm.post_training.ollama import LLMOptions, Message, generate_chat_response

file_path = "data/finetuning/instruction_dataset_test_responses.json"

with open(file_path, encoding="utf-8") as f:
    data = json.load(f)


def format_input(sample: dict[str, str]) -> str:
    """Format an instruction sample into Alpaca-style prompt (without response)."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{sample['instruction']}"
    )

    input_text = f"\n\n### Input:\n{sample['input']}" if sample["input"] else ""

    return instruction_text + input_text


response_options = LLMOptions(
    temperature=0.7,
    num_ctx=4096,
    top_k=40,
    top_p=0.9,
    num_predict=512,
)
res = generate_chat_response(
    [Message(role="user", content="What do whale eat?")],
    options=response_options,
    stream=True,
)
