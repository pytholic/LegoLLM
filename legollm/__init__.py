"""LegoLLM - Build LLMs from scratch with modular components.

Main modules:
- architectures: Self-contained model implementations (GPT2, Llama3, Qwen3)
- components: Reusable building blocks (attention, embeddings, normalization)
- generation: Text generation utilities
- optimization: Performance optimizations (KV cache, etc.)
- peft: Parameter-efficient fine-tuning (LoRA, etc.)
- finetuning: Fine-tuning utilities
- training: Training utilities
- data: Data loading utilities
- core: Tokenization and core interfaces
"""

__all__ = [
    "architectures",
    "components",
    "core",
    "data",
    "finetuning",
    "generation",
    "models",
    "optimization",
    "peft",
    "training",
]
