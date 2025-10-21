# Tokenization in Language Models

**Goal**: Understand and implement a base tokenizer

Language models process text by converting it into sequences of integer token IDs. Each token ID serves as an index to retrieve the corresponding embedding vector from an embedding table (lookup table). This makes tokenization a critical preprocessing step in the NLP pipeline.

## Purpose of Tokenization

Tokenization breaks down raw text into discrete units (tokens) that the model can process. These tokens can represent:

- Individual characters
- Subword units (e.g., "playing" → "play" + "##ing")
- Complete words
- Special symbols

## Core Components of a Tokenizer

Tokenizer acts as a conversion layer between raw test and token sequence. A typical tokenizer implementation provides three essential methods:

### `encode(text: str) -> List[int]`

- Converts input text into a sequence of token IDs
- This is the primary method used during model inference and training

### `decode(token_ids: List[int]) -> str`

- Converts a sequence of token IDs back into human-readable text
- Essential for interpreting model outputs

### `tokenize(text: str) -> List[str]` (optional but common)

- Splits text into individual token strings without converting to IDs
- Useful for vocabulary construction and debugging
- Often used during the initial vocabulary building phase

### `train(self, texts: str | list[str], vocab_size: int, **kwargs: Any) -> None`

- Train the tokenizer like BPE, WordPiece, SentencePiece on corpus to learn merges/vocabulary

> **ℹ️ Note**
>
> Keep in mind that the Tokenizer is a separate entity from the LLM, and it warrants its own separate training data. This data may or may not be same as the data used for LLM training.
