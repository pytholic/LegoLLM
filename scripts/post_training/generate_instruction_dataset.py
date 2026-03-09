"""Generate instruction dataset using LLM.

Make sure ollama is installed and running:
```bash
ollama run llama3.1:8b
```

Generates two types of entries (mirroring Alpaca-style datasets):
  - Mode A (no input): self-contained factual/knowledge questions
  - Mode B (with input): transformation tasks applied to specific content

Created by @pytholic on 2026.02.19
"""

import json
import random
from pathlib import Path

from legollm.logging import progress_bar
from legollm.post_training.providers.base import LLMOptions, Message
from legollm.post_training.providers.ollama_provider import HttpBackend

# ---------------------------------------------------------------------------
# Task type definitions
# ---------------------------------------------------------------------------

# Mode A: self-contained, no input field needed
MODE_A_TASK_TYPES = {
    "factual_lookup": "a short factual question about science, history, or geography (e.g. 'What is the capital of Japan?')",
    "definition": "ask to define a specific term or concept (e.g. 'Define the term ecosystem.')",
    "list_generation": "ask to list or name N specific examples of something (e.g. 'Name three types of biomes.')",
    "calculation": "a concrete math or unit conversion problem with specific numbers (e.g. 'Convert 45 kilometers to meters.')",
    "formula_lookup": "ask for a specific scientific or mathematical formula (e.g. 'What is the formula for the area of a triangle?')",
    "verb_form": "ask for a specific grammatical form of a word (e.g. 'What is the past tense of draw?')",
    "word_relationship": "ask for a synonym, antonym, or rhyme for a specific word (e.g. 'Provide a synonym for bright.')",
}

# Mode B: instruction operates on a provided input text
MODE_B_TASK_TYPES = {
    "grammar_correction": "grammar or spelling correction of a sentence",
    "voice_conversion": "converting a sentence between active and passive voice",
    "tense_conversion": "rewriting a sentence in a different verb tense",
    "translation": "translating a short phrase into another language",
    "sentence_rewriting": "rewriting a sentence using a figurative device such as a metaphor, simile, or idiom",
    "sentiment_analysis": "sentiment analysis of a short text (positive, negative, or neutral)",
    "sorting": "sorting a short list of items alphabetically or numerically",
    "redundancy_removal": "removing redundant or unnecessary words from a sentence",
    "formality_adjustment": "rewriting a sentence to be more formal or more informal",
    "text_classification": "classifying a word or short phrase into a given set of categories",
}

# Rough 70/30 split matching the sample dataset distribution
MODE_WEIGHTS = {"A": 0.7, "B": 0.3}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

MODE_A_PROMPT = """\
Generate a single instruction of type: {task_description}.

Rules:
- Output ONLY the instruction itself — no preamble, no quotes, no explanation.
- Keep it concise (one sentence, under 20 words).
- Make it specific and concrete, not generic.
"""

MODE_B_PROMPT = """\
Generate a transformation task of type: {task_description}.

Provide:
  "instruction": a short imperative instruction (e.g. "Edit the following sentence for grammar.")
  "input": the specific text the instruction should be applied to (a sentence, list, or phrase)

Rules:
- The instruction must NOT contain the input text — keep them separate.
- Keep both fields concise.
"""

RESPONSE_PROMPT = """\
{instruction}

{input_section}Answer directly and concisely."""


def build_mode_a_prompt(task_type: str) -> str:
    """Build a prompt for mode A tasks.

    Mode A refers to self-contained, no input field needed.
    """
    return MODE_A_PROMPT.format(task_description=MODE_A_TASK_TYPES[task_type])


def build_mode_b_prompt(task_type: str) -> str:
    """Build a prompt for mode B tasks.

    Mode B refers to instruction operates on a provided input text.
    """
    return MODE_B_PROMPT.format(task_description=MODE_B_TASK_TYPES[task_type])


MODE_B_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "input": {"type": "string"},
    },
    "required": ["instruction", "input"],
}


def build_response_prompt(instruction: str, input_text: str) -> str:
    """Build a response prompt for mode A and B tasks."""
    input_section = f"{input_text}\n\n" if input_text else ""
    return RESPONSE_PROMPT.format(instruction=instruction, input_section=input_section)


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    provider = HttpBackend()

    instruction_options = LLMOptions(
        temperature=1.0,
        num_ctx=2048,
        top_k=40,
        top_p=0.95,
        num_predict=128,
    )

    response_options = LLMOptions(
        temperature=0.7,
        num_ctx=4096,
        top_k=40,
        top_p=0.9,
        num_predict=512,
    )

    dataset_size = 5
    dataset: list[dict[str, str]] = []

    with progress_bar("Generating instructions", total=dataset_size) as (progress, task):
        for _ in range(dataset_size):
            mode = random.choices(["A", "B"], weights=[MODE_WEIGHTS["A"], MODE_WEIGHTS["B"]])[0]

            if mode == "A":
                task_type = random.choice(list(MODE_A_TASK_TYPES))
                prompt = build_mode_a_prompt(task_type)

                instruction = provider.chat(
                    [Message(role="user", content=prompt)],
                    model="llama3.1:8b",
                    options=instruction_options,
                    output_format=None,
                    stream=False,
                ).strip()

                if not instruction:
                    continue

                output = provider.chat(
                    [Message(role="user", content=build_response_prompt(instruction, ""))],
                    model="llama3.1:8b",
                    options=response_options,
                    output_format=None,
                    stream=False,
                ).strip()

                dataset.append({"instruction": instruction, "input": "", "output": output})

                progress.update(task, advance=1)

            else:  # Mode B
                task_type = random.choice(list(MODE_B_TASK_TYPES))
                prompt = build_mode_b_prompt(task_type)

                raw = provider.chat(
                    [Message(role="user", content=prompt)],
                    model="llama3.1:8b",
                    options=instruction_options,
                    output_format=MODE_B_SCHEMA,
                    stream=False,
                ).strip()

                parsed: dict[str, str] = json.loads(raw)
                instruction = parsed["instruction"]
                input_text = parsed["input"]

                if not instruction or not input_text:
                    continue

                output = provider.chat(
                    [Message(role="user", content=build_response_prompt(instruction, input_text))],
                    model="llama3.1:8b",
                    options=response_options,
                    output_format=None,
                    stream=False,
                ).strip()

                dataset.append({"instruction": instruction, "input": input_text, "output": output})

                progress.update(task, advance=1)

    output_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "finetuning"
        / "instruction_dataset_test.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset)} examples to {output_path}")
