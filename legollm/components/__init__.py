"""Reusable building blocks for transformer models.

This module provides modular components that can be composed to build
custom architectures or used by the self-contained architecture files.

Submodules:
- attention: Multi-head attention variants (MHA, GQA, Flash)
- embeddings: Token and positional embeddings (absolute, RoPE)
- normalization: Layer normalization variants (LayerNorm, RMSNorm)
- feedforward: Feed-forward network variants (MLP, SwiGLU, MoE)
- blocks: Transformer block implementations
"""
