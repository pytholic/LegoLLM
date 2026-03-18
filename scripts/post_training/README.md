# Post-Training Scripts

Scripts for building and improving instruction datasets, and evaluating fine-tuned models.

## Pipeline Overview

```
generate_instruction_dataset.py  →  reflection.py  →  sft.py  →  evaluate.py
      (generate data)                 (improve data)    (train)     (evaluate)
```

______________________________________________________________________

## 1. Generate Instruction Dataset

Generates an instruction dataset from raw data using an LLM.

```shell
uv run python scripts/post_training/generate_instruction_dataset.py
```

Output: `data/finetuning/instruction_dataset.json`

______________________________________________________________________

## 2. Reflection (Data Improvement)

Rewrites dataset entries using an LLM judge to improve quality. Two modes:

- **`response`** — rewrites only the answer to be more helpful and detailed
- **`instruction`** — rewrites both the instruction and the answer to be more complex

### Single instance (local)

```shell
# OpenAI
uv run python scripts/post_training/reflection.py \
    --provider openai --model gpt-4o-mini --mode response

# Ollama
uv run python scripts/post_training/reflection.py \
    --provider ollama --model qwen3.5:9b --mode instruction
```

### Multi-instance (parallel, multi-GPU)

Spin up one Ollama instance per GPU, then run reflection across all of them:

```shell
# Start all instances and pull the model in one step, then run reflection
make ollama-start reflect NUM_INSTANCES=4 GPU_IDS=0,1,2,3 MODEL=deepseek-r1:14b

# Stop all instances when done
make ollama-stop
```

Or run the steps individually if you need more control:

```shell
make ollama-serve NUM_INSTANCES=4 GPU_IDS=0,1,2,3 MODEL=deepseek-r1:14b
make ollama-pull  NUM_INSTANCES=4 MODEL=deepseek-r1:14b
uv run python scripts/post_training/reflection.py \
    --provider ollama --model deepseek-r1:14b --mode instruction \
    --num-instances 4 --base-port 11434
make ollama-stop
```

### Retrying failed samples

Failed samples are saved to `data/finetuning/instruction_dataset_reflection_{mode}_failed.jsonl`. Retry with:

```shell
uv run python scripts/post_training/reflection.py \
    --input data/finetuning/instruction_dataset_reflection_instruction_failed.jsonl
```

### Installing Ollama (Linux, no root)

```shell
curl -LO https://github.com/ollama/ollama/releases/download/v0.17.7/ollama-linux-amd64.tar.zst
mkdir ollama && tar --use-compress-program=unzstd -xvf ollama-linux-amd64.tar.zst -C ollama/
./ollama/bin/ollama serve   # terminal 1
./ollama/bin/ollama pull qwen3.5:9b  # terminal 2
```

______________________________________________________________________

## 3. Fine-tune (SFT)

Fine-tunes GPT-2 on the instruction dataset.

```shell
uv run python scripts/post_training/sft.py
uv run python scripts/post_training/sft.py \
    --model gpt2-medium --num-epochs 5 --batch-size 4 \
    --max-length 1024 --grad-accumulation-steps 4
```

Output: checkpoint at `checkpoints/sft/` and test responses at `data/finetuning/instruction_dataset_test_responses_reflection.json`.

______________________________________________________________________

## 4. Evaluate

Scores fine-tuned model responses using an LLM judge (1–5 rubric).

```shell
uv run python scripts/post_training/evaluate.py
uv run python scripts/post_training/evaluate.py \
    --file data/finetuning/instruction_dataset_test_responses_reflection.json \
    --model deepseek-r1:14b
```
