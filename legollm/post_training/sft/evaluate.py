"""Evaluate the finetuned model on the test data using LLM judge.

Created by @pytholic on 2026.02.26
"""

import json
import re

from legollm.logging import progress_bar
from legollm.post_training.ollama import LLMOptions, Message, generate_chat_response

file_path = "data/finetuning/instruction_dataset_test_responses.json"

with open(file_path, encoding="utf-8") as f:
    test_data = json.load(f)


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
    num_ctx=2048,
    top_k=40,
    top_p=0.9,
    num_predict=128,
)

EVAL_PROMPT = """You are a fair judge. Evaluate the model response against the reference answer using the rubric below.

Rubric:
1: Fails to address instructions; irrelevant or incorrect.
2: Partially addresses instructions; significant inaccuracies or irrelevant details.
3: Follows instructions with minor inaccuracies or unnecessary details.
4: Clear, accurate, and relevant with only minor issues.
5: Fully adheres to instructions; concise, accurate, and complete.

Instruction: {instruction}
Reference Answer: {reference}
Model Response: {answer}

Provide a one-sentence evaluation and an integer score (1-5).
Feedback: (one-sentence rationale)
Score: (integer 1-5)"""


def generate_model_scores(json_data: list[dict[str, str]], model: str = "llama3.1:8b") -> list[int]:
    """Generate scores for the model."""
    scores: list[int] = []
    with progress_bar("Generating scores", total=len(json_data)) as (progress, task):
        for i, sample in enumerate(json_data):
            prompt = EVAL_PROMPT.format(
                instruction=(
                    f"Given the input `{format_input(sample)}` "
                    f"and correct output `{sample['output']}`, "
                    f"score the model response `{sample['generated_response']}` on a scale from 1 to 5."
                ),
                reference=sample["output"],
                answer=sample["generated_response"],
            )
            response = generate_chat_response(
                [Message(role="user", content=prompt)],
                model=model,
                options=response_options,
                stream=False,
            )
            try:
                score_match = re.search(r"Score: (\d+)", response)
                if score_match:
                    scores.append(int(score_match.group(1)))
            except ValueError:
                print(f"Could not find score in response: {response}")
                scores.append(0)
            progress.update(task, completed=i + 1)

        return scores


scores = generate_model_scores(test_data)
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores) / len(scores):.2f}\n")
