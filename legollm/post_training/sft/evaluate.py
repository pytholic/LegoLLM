"""LLM-judge evaluation for finetuned model responses."""

import re

from legollm.logging import logger, progress_bar
from legollm.post_training.providers.base import LLMOptions, Message
from legollm.post_training.providers.ollama_provider import HttpBackend, SdkBackend
from legollm.post_training.sft.instruction_dataset import format_input

EVAL_PROMPT = """
You are a fair judge. Evaluate the model response against the reference answer using the rubric below.

Rubric:
1: Fails to address instructions; irrelevant or incorrect.
2: Partially addresses instructions; significant inaccuracies or irrelevant details.
3: Follows instructions with minor inaccuracies or unnecessary details.
4: Clear, accurate, and relevant with only minor issues.
5: Fully adheres to instructions; concise, accurate, and complete.

Instruction: {instruction}
Reference Answer: {reference}
Model Response: {answer}

Output format: Provide a one-sentence evaluation and an integer score (1-5).
Feedback: (one-sentence rationale)
Score: (integer 1-5)"""


def generate_model_scores(
    json_data: list[dict[str, str]],
    provider: HttpBackend | SdkBackend,
    model: str,
    options: LLMOptions,
) -> list[int]:
    """Score model responses with an LLM judge. Returns a list of integer scores (1-5)."""
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
            response = provider.chat(
                [Message(role="user", content=prompt)],
                model=model,
                options=options,
                output_format=None,
                stream=False,
            )
            logger.info(f"Response: {response}")
            score_match = re.search(r"Score: (\d+)", response)
            if score_match:
                scores.append(int(score_match.group(1)))
            else:
                logger.warning(f"Could not find score in response: {response}")
                scores.append(0)
            progress.update(task, completed=i + 1)
    return scores
