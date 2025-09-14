# 🧱 Lego-LLM: Modular Language Model Development from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

In this comprehensive project, I'm building a complete LLM development framework from first principles, designed as modular "Lego pieces" that can be combined, swapped, and extended. The journey starts with fundamental text processing and systematically builds up to recent alignment techniques , modern architectures, and optimizations.


**Technical Progression:**

```docker

Raw Text → Tokenization → Attention → Transformers → Modern Optimizations 
(RoPE, GQA, Flash Attention) → Parameter-Efficient Methods (LoRA, QLoRA) → 
Scaling (MoE) → Alignment (RLHF, DPO) → Production Framework
```

The modular design enables rapid experimentation: want to try "GPT-2 with RoPE and sliding window attention"? Just swap the components. Need "LLaMA with MoE and QLoRA fine-tuning"? The framework supports it seamlessly.

## ey Features
- **Modular Architecture**: Each component (attention, normalization, embedding) is a swappable piece
- **Multiple Model Support**: Seamlessly switch between GPT-2, LLaMA, Gemma, Qwen architectures
- **Complete Pipeline**: From raw text tokenization to RLHF-aligned instruction-following models
- **Production Quality**: Clean, typed, tested code suitable for research and deployment
- **Educational Depth**: Implement every technique from scratch to build deep understanding
-   **Package Management**: Uses [`uv`](https://github.com/astral-sh/uv) for fast dependency management.
-   **Code Quality**: Integrated with [`Ruff`](https://github.com/astral-sh/ruff) for linting and formatting.
-   **Testing**: `pytest` setup for unit and integration tests.
-   **Type Hinting**: Enforced with `pyright` for robust code.
-   **Documentation**: [`mkdocs`](https://www.mkdocs.org/) with Material theme ready to go.
-   **Simple Setup**: Easy Makefile commands for environment setup and project management.

## Getting Started

1.  **Clone this template with your desired project name.**

    ```bash
    git clone <ADD REPO>
    ```

2.  **Set up your environment.**

    ```bash
    make setup
    ```

    This will:
    - Create a virtual environment using `uv`
    - Install all dependencies (dev and docs)
    - Activate the virtual environment

    *I tend to use `uv` for almost all my projects now as it is faster and simpler to use.*

3.  **Customize your project.**

    - Update `pyproject.toml` with your project name, version, and build paths
    - Rename/organize files in the `src/` directory as needed
    - Create subfolders like `core/` or rename modules to fit your project structure

## Project Structure

*Add later*

## Usage

*Add later*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
