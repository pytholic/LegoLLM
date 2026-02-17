# 🧱 LegoLLM: Modular Language Model Development from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Building a complete LLM development framework from first principles - one modular "Lego piece" at a time. From tokenization to alignment, every component is implemented from scratch for deep understanding and maximum flexibility.

> **Current Stage:** Phase 2 complete — GPT-2 pretrained, next up: Instruction Fine-tuning (SFT)

______________________________________________________________________

## 📑 Table of Contents

- [Why LegoLLM?](#why-legollm)
- [Current Components](#current-components)
  - [Tokenization](#tokenization)
  - [GPT-2 Architecture](#gpt-2-architecture)
  - [Data Pipeline](#data-pipeline)
  - [Training](#training)
  - [Generation](#generation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)
- [License](#license)

______________________________________________________________________

## Why LegoLLM?

Modern LLMs are complex systems, but they're built from understandable components. LegoLLM breaks down this complexity by implementing each piece from scratch:

```
Raw Text → Tokenization → Embeddings → Attention → Transformers → Training
         → Optimization → Alignment → Production Framework
```

**Design Philosophy:**

- **Modular**: Swap GPT-2 attention for LLaMA's GQA? Just change one component
- **Educational**: Understand *why* each piece exists, not just *how* it works
- **Production-Ready**: Clean, typed, tested code suitable for research and deployment
- **Progressive**: Master foundations before advancing to cutting-edge techniques

______________________________________________________________________

## Current Components

### Tokenization

> **Why it matters:** Tokenization is the bridge between human language and neural networks. Poor tokenization = poor model performance, regardless of architecture quality.

#### The Problem

Language models can't process raw text. They need:

1. **Fixed vocabulary** - Can't have infinite possible inputs
2. **Numerical representation** - Neural networks work with numbers
3. **Efficient encoding** - Balance between vocabulary size and sequence length

#### What's Implemented

**Two tokenizers, each teaching different concepts:**

##### 1. **NaiveBPETokenizer** - Core BPE Algorithm

Educational implementation showing the fundamental Byte-Pair Encoding algorithm:

- ✅ UTF-8 byte-level encoding (universal, handles all languages)
- ✅ Iterative pair merging for compression
- ✅ Vocabulary building and management
- ✅ Save/load with Base64 encoding

**Use case:** Understanding how BPE works at its core.

##### 2. **RegexBPETokenizer** - Production-Style BPE

GPT-2/GPT-4 style tokenizer with important optimizations:

- ✅ Regex-based text splitting (prevents cross-category merges)
- ✅ Special token support (`<|endoftext|>`, `<|im_start|>`, etc.)
- ✅ Configurable split patterns (GPT-2, GPT-4 patterns included)
- ✅ Proper handling of contractions and punctuation

**Use case:** Training actual language models, matching production behavior.

#### Key Differences

| Feature                      | NaiveBPE                    | RegexBPE               |
| ---------------------------- | --------------------------- | ---------------------- |
| **Merges across categories** | Yes (can merge "dog" + ".") | No (regex prevents it) |
| **Special tokens**           | ❌                          | ✅                     |
| **Vocab efficiency**         | Lower                       | Higher                 |
| **Training speed**           | Faster                      | Slower                 |
| **Production use**           | ❌ Educational only         | ✅ Production-ready    |

### GPT-2 Architecture

Self-contained implementation in `legollm/architectures/gpt2.py` with all components inline:

- ✅ Custom LayerNorm (learnable scale + shift)
- ✅ Multi-Head Causal Attention (separate Q/K/V projections)
- ✅ GELU-activated Feed-Forward Network (MLP)
- ✅ Pre-norm Transformer Block with residual connections
- ✅ All 4 model sizes: 124M, 355M, 774M, 1558M
- ✅ **Pretrained weight loading** from HuggingFace via safetensors
  - Conv1D → Linear transpose (4 matrices per block)
  - Fused QKV split into separate q_proj, k_proj, v_proj
  - Weight tying (tok_emb ↔ out_head)

### Data Pipeline

- ✅ Data preparation: raw text → tokenize → train/val `.bin` files + `meta.json`
- ✅ Memory-efficient DataLoader using numpy memmap + circular buffer
- ✅ Configurable batch size, sequence length, device placement

### Training

- ✅ Trainer with cosine LR schedule (linear warmup → cosine decay → min_lr floor)
- ✅ AdamW with separate param groups (weight decay on 2D params only, not biases/layernorms)
- ✅ Checkpointing (model + optimizer state + iteration + best val loss)

### Generation

- ✅ Greedy and stochastic sampling (top-k, top-p, temperature)
- ✅ Streaming generation via Python generators (token-by-token output)
- ✅ Pre-allocated buffer approach (no torch.cat memory leak)
- ✅ Context window sliding for sequences beyond model's context length

______________________________________________________________________

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

### Setup

```bash
git clone https://github.com/pytholic/LegoLLM.git
cd LegoLLM
uv sync
```

______________________________________________________________________

## Quick Start

### Load Pretrained GPT-2 and Generate

```python
import torch
import tiktoken
from legollm.architectures.gpt2 import GPT2, GPT2Variant, MODEL_CONFIGS, load_gpt2_weights

# Load pretrained GPT-2 124M
model = GPT2(MODEL_CONFIGS[GPT2Variant.GPT2])
load_gpt2_weights(model, GPT2Variant.GPT2)
model.eval()

# Generate text
enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor(enc.encode("Every effort moves you")).unsqueeze(0)
with torch.no_grad():
    for _ in range(30):
        logits = model(ids[:, -1024:])
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
print(enc.decode(ids[0].tolist()))
```

### Command-Line Tools

LegoLLM provides convenient CLI commands (registered in `pyproject.toml`):

```bash
# Download datasets
uv run data-download tiny_shakespeare ./data/raw/tiny_shakespeare/

# Prepare data for training (tokenize + split)
uv run data-prepare --config configs/dataset_config.yaml --verbose

# Get dataset statistics
uv run data-summary ./data/raw/tiny_shakespeare/tiny_shakespeare.txt
```

### Train on Custom Data

```bash
# Train GPT-2
uv run python scripts/train.py --dataset data/processed/the_verdict --model gpt2-124m

# Generate from trained model
uv run python scripts/generate.py \
    --checkpoint checkpoints/latest.pt \
    --prompt "The verdict was" \
    --max-new-tokens 100 \
    --strategy stochastic --temperature 0.8
```

### Training a Tokenizer

```python
from legollm.core.tokenization import RegexBPETokenizer
from legollm.core.utils import read_file

# Load your training corpus
text = read_file("data/corpus.txt")

# Initialize and train
tokenizer = RegexBPETokenizer()
tokenizer.train(
    text=text,
    vocab_size=5000,  # Target vocabulary size
    verbose=True
)

# Save for later use
tokenizer.save("models/my_tokenizer.json")
```

### Using a Trained Tokenizer

```python
from legollm.core.tokenization import RegexBPETokenizer

# Load trained tokenizer
tokenizer = RegexBPETokenizer()
tokenizer.load("models/my_tokenizer.json")

# Encode text to token IDs
text = "Hello, world! This is a test."
token_ids = tokenizer.encode(text)
print(f"Encoded: {token_ids}")

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
assert decoded == text  # Perfect round-trip
```

### Working with Special Tokens

```python
# Register special tokens
tokenizer.register_special_tokens({
    "<|endoftext|>": 50256,
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
})

# Encode with special tokens
text = "Hello<|endoftext|>world"
ids = tokenizer.encode(text)
# Output: [72, 101, 108, 108, 111, 50256, 119, 111, 114, 108, 100]

# Special tokens preserved during decoding
decoded = tokenizer.decode(ids)
# Output: "Hello<|endoftext|>world"
```

### Comparing Tokenizers

```python
from legollm.core.tokenization import NaiveBPETokenizer, RegexBPETokenizer

text = "Hello world! How are you?"

# Train both tokenizers
naive = NaiveBPETokenizer()
naive.train(text, vocab_size=300)

regex = RegexBPETokenizer()
regex.train(text, vocab_size=300)

# Compare encodings
print(f"Naive: {naive.encode(text)}")
print(f"Regex: {regex.encode(text)}")

# Regex typically produces more efficient tokenization
# due to preventing cross-category merges
```

______________________________________________________________________

## Project Structure

```
legollm/
├── architectures/           # Self-contained model files
│   └── gpt2.py             #   GPT-2 (all sizes) + pretrained weight loading
├── components/              # Reusable building blocks
│   ├── attention.py         #   Multi-head attention
│   ├── embeddings.py        #   Token + positional embeddings
│   ├── feedforward.py       #   MLP / feed-forward networks
│   ├── normalization.py     #   Layer normalization
│   └── blocks.py            #   Transformer blocks
├── core/
│   ├── interfaces.py        #   Protocol contracts (Tokenizer, DocumentSplitter)
│   ├── exceptions.py        #   Exception hierarchy
│   └── tokenization/        #   BPE tokenizers (RegexBPE, NaiveBPE, Simple)
├── data/dataloader.py       #   Memory-efficient DataLoader (memmap + circular buffer)
├── training/trainer.py      #   Trainer (cosine LR, AdamW, checkpointing)
├── generation/              #   Text generation (greedy, top-k, top-p, streaming)
├── peft/lora.py             #   LoRA (stub — upcoming)
├── finetuning/              #   Instruction fine-tuning (upcoming)
├── config.py                #   Path constants
├── utils.py                 #   Helpers (dtype, metadata, param counting)
└── logging.py               #   Rich-based logging

scripts/
├── train.py                 # Training entry point (CLI + optional YAML config)
├── generate.py              # Text generation with streaming
├── prepare.py               # Data preparation: raw text → tokenized .bin files
└── download_data.py         # Dataset downloading

tests/
└── unit/                    # 218 tests (tokenization, data, training, generation)
```

______________________________________________________________________

## Development Roadmap

### ✅ Phase 1: Core Foundation — Complete

- [x] Tokenization (NaiveBPE, RegexBPE, Simple)
- [x] Token + positional embeddings
- [x] Multi-head causal attention
- [x] GPT-2 architecture (124M–1558M)
- [x] DataLoader (memmap + circular buffer)
- [x] Trainer (cosine LR, AdamW, checkpointing)
- [x] Generation (greedy, top-k, top-p, streaming)

### ✅ Phase 2: Validate & Load Pretrained — Complete

- [x] Small-scale pretraining on The Verdict
- [x] Load pretrained GPT-2 weights from HuggingFace (safetensors)
- [x] Weight mapping: Conv1D transpose, fused QKV split, key renaming

### 🚧 Phase 3: Instruction Fine-tuning — In Progress

- [ ] Instruction dataset + Alpaca-style prompt formatting
- [ ] Custom collate with dynamic padding + loss masking (-100)
- [ ] SFT training script (full fine-tuning on GPT-2 124M)
- [ ] LoRA implementation + comparison with full fine-tuning

### 📋 Phase 4: Modern Architecture (Llama3)

- [ ] RoPE (Rotary Position Embeddings)
- [ ] RMSNorm + SwiGLU activation
- [ ] Grouped Query Attention (GQA) + KV Cache
- [ ] Assemble Llama3 + load pretrained weights

### 📋 Phase 5: Alignment

- [ ] DPO (Direct Preference Optimization)
- [ ] PPO / RLHF (optional)
- [ ] Mixture of Experts (MoE)

______________________________________________________________________

## Development

```bash
# Run all tests
uv run pytest -x --tb=no -rs

# Run unit tests only
uv run pytest -x --tb=no -rs tests/unit

# Run a specific test file
uv run pytest tests/unit/generation/test_generate.py -v

# Lint + format
uv run ruff check .
uv run ruff format .
```

______________________________________________________________________

## Contributing

This is a personal learning project, but insights and suggestions are welcome! If you spot an issue or have an improvement idea:

1. Open an issue describing the problem/suggestion
2. For code contributions, ensure tests pass and code is formatted
3. Keep the educational focus - clarity over cleverness

______________________________________________________________________

## References & Resources

### Papers

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)

### Courses

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/)
- [Stanford CS224N: Natural Language Processing](http://web.stanford.edu/class/cs224n/)

### Books

- Build a Large Language Model (From Scratch) by Sebastian Raschka

### Implementations

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka
- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [minbpe](https://github.com/karpathy/minbpe) by Andrej Karpathy
- [tiktoken](https://github.com/openai/tiktoken) by OpenAI

______________________________________________________________________

## Acknowledgments

Built with inspiration from the excellent educational content by:

- Andrej Karpathy ([Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html))
- Sebastian Raschka ([LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch))
- Stanford CS336 course materials

______________________________________________________________________

**Next Up:** Instruction Fine-tuning (SFT) on pretrained GPT-2
