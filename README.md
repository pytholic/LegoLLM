# 🧱 LegoLLM: Modular Language Model Development from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Building a complete LLM development framework from first principles - one modular "Lego piece" at a time. From tokenization to alignment, every component is implemented from scratch for deep understanding and maximum flexibility.

> **Current Stage:** Phase 1 - Core Foundation (Tokenization Complete)

______________________________________________________________________

## 📑 Table of Contents

- [Why LegoLLM?](#why-legollm)
- [Current Components](#current-components)
  - [Tokenization](#tokenization)
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

______________________________________________________________________

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/LegoLLM.git
cd LegoLLM

# Setup environment (uses uv)
make setup

# Or manually with pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

______________________________________________________________________

## Quick Start

### Command-Line Tools

LegoLLM provides convenient command-line tools for data management:

```shell
# Download datasets
download-data tiny_shakespeare ./data/raw/tiny_shakespeare/

# Prepare data for training (tokenize + split)
prepare --config configs/datasets/tiny_shakespeare.yaml

# Get dataset statistics
dataset-summary ./data/raw/tiny_shakespeare/tiny_shakespeare.txt
```

See the [`scripts/` directory](scripts/README.md) for detailed documentation.

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
LegoLLM/
├── legollm/                  # Main package
│   ├── core/                 # Foundation components
│   │   ├── tokenization/    # ← Current: Complete ✅
│   │   │   ├── bpe/                # BPE tokenizers
│   │   │   │   ├── base_bpe.py          # Base BPE class
│   │   │   │   ├── naive_bpe_tokenizer.py    # Educational BPE
│   │   │   │   └── regex_bpe_tokenizer.py    # Production BPE
│   │   │   └── simple/             # Simple tokenizers
│   │   │       ├── simple_tokenizer.py       # Basic word tokenizer
│   │   │       └── vocabulary.py             # Vocab management
│   │   ├── interfaces.py    # Protocol definitions
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── utils.py         # Utility functions
│   │
│   ├── logging.py           # Logging utilities
│   ├── attention/            # Coming: Attention mechanisms
│   ├── layers/              # Coming: Transformer layers
│   ├── models/              # Coming: Complete models
│   └── training/            # Coming: Training infrastructure
│
├── scripts/                 # Command-line tools
│   ├── prepare.py          # Data preparation pipeline
│   ├── download_data.py    # Dataset downloader
│   └── dataset_summary.py  # Dataset statistics
│
├── configs/                 # Configuration files
│   └── datasets/           # Dataset configs (YAML)
│
├── tests/                   # Test suite
│   ├── unit/
│   └── integration/
│
├── data/                    # Training data & models
│   ├── raw/                # Raw datasets
│   ├── processed/          # Tokenized train/val splits
│   └── tokenizers/         # Trained tokenizer models
│
├── experiments/             # Experimental notebooks
└── docs/                    # Documentation (future)
```

______________________________________________________________________

## Development Roadmap

### ✅ Phase 1: Core Foundation

**Status:** In Progress

#### Tokenization

- [x] Simple whitespace tokenizer
- [x] Vocabulary builder
- [x] Naive BPE implementation
- [x] Regex-based BPE (GPT-2/GPT-4 style)
- [x] Special token support
- [x] Save/load functionality

#### Data Processing

- [x] Prepare data
- [ ] Dataset and sliding window
- [ ] Dataloader for efficiency

#### Embeddings

- [ ] Token embeddings
- [ ] Positional embeddings (learned)
- [ ] Positional embeddings (sinusoidal)

#### Attention Mechanism

- [ ] Scaled dot-product attention
- [ ] Multi-head attention
- [ ] Causal masking

#### Transformer Block

- [ ] Layer normalization
- [ ] Feed-forward networks
- [ ] Residual connections
- [ ] Complete transformer block

### 📋 Phase 2: Modern Architecture

**Status:** Planned

- [ ] RoPE (Rotary Position Embeddings)
- [ ] RMSNorm & modern activations (SiLU, GELU)
- [ ] Grouped Query Attention (GQA)
- [ ] KV caching & generation optimization
- [ ] Flash Attention integration

### 📋 Phase 3: Advanced Optimization

**Status:** Planned

- [ ] Mixed precision training (AMP)
- [ ] Gradient checkpointing
- [ ] LoRA (Low-Rank Adaptation)
- [ ] QLoRA (Quantized LoRA)
- [ ] 4-bit quantization (NF4)
- [ ] Mixture of Experts (MoE)

### 📋 Phase 4: Alignment & Training

**Status:** Planned

- [ ] Training pipeline & data loaders
- [ ] Reward models & human preferences
- [ ] PPO (Proximal Policy Optimization)
- [ ] RLHF (Reinforcement Learning from Human Feedback)
- [ ] DPO (Direct Preference Optimization)
- [ ] Evaluation metrics & benchmarks

______________________________________________________________________

## Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/unit/test_tokenization.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
make type-check

# Run all checks
make check
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

- [minbpe](https://github.com/karpathy/minbpe) by Andrej Karpathy
- [tiktoken](https://github.com/openai/tiktoken) by OpenAI

______________________________________________________________________

## Acknowledgments

Built with inspiration from the excellent educational content by:

- Andrej Karpathy ([Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html))
- Sebastian Raschka ([LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch))
- Stanford CS336 course materials

______________________________________________________________________

**Next Up:** Token & Positional Embeddings
