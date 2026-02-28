# Supervised Fine-Tuning (SFT)

Instruction fine-tuning GPT-2 on Alpaca-style data, with LLM-judge evaluation.

## Data

~1,100 instruction-response pairs generated via `llama3.1:8b` (Ollama). Two modes:

- **Mode A** (70%): Self-contained questions (factual, definitions, calculations)
- **Mode B** (30%): Transformation tasks with input text (grammar correction, translation, etc.)

Format: `{"instruction": "...", "input": "...", "output": "..."}`

Generation script: `scripts/post_training/generate_instruction_dataset.py`

## Pipeline

```
1. Generate data        → scripts/post_training/generate_instruction_dataset.py
2. Train (SFT)          → scripts/post_training/sft.py
3. Extract responses    → (included in sft.py, saves test split responses)
4. Evaluate (LLM judge) → legollm/post_training/sft/evaluate.py
5. Chat                 → scripts/post_training/chat.py
```

## Code

| File                     | Role                                                    |
| ------------------------ | ------------------------------------------------------- |
| `instruction_dataset.py` | Dataset class, Alpaca prompt formatting, collate fn     |
| `trainer.py`             | Training loop, validation, checkpointing, loss plotting |
| `evaluate.py`            | LLM-judge scoring (1-5 rubric) via Ollama               |

Shared Ollama client: `legollm/post_training/ollama.py`

## Usage

```bash
# Generate instruction data (requires ollama running)
uv run python scripts/post_training/generate_instruction_dataset.py

# Fine-tune
uv run python scripts/post_training/sft.py --model gpt2-medium --num-epochs 5 --batch-size 2 --grad-accumulation-steps 4

# Evaluate
uv run python legollm/post_training/sft/evaluate.py

# Interactive chat
uv run python scripts/post_training/chat.py
```

## Results

| Metric          | Value                     |
| --------------- | ------------------------- |
| Model           | GPT-2 Medium (355M)       |
| Dataset         | ~1,100 Alpaca-style pairs |
| LLM Judge Score | 2.19 / 5.0                |

Known limitation: responses tend to be short due to concise training data ("Answer directly and concisely" in generation prompts).

## Future

- Self-refine / reflection pipeline for higher quality training data
- DPO preference tuning
