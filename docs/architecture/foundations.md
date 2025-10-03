# Tokenization Foundation

**Date**: 2025.09.14 ~
**Goal**: Build a production-ready tokenizer from scratch to understand the foundation of LLM text processing

## Tokenization in Language Models

Language models process text by converting it into sequences of integer token IDs. Each token ID serves as an index to retrieve the corresponding embedding vector from an embedding table (lookup table). This makes tokenization a critical preprocessing step in the NLP pipeline.

### Purpose of Tokenization

Tokenization breaks down raw text into discrete units (tokens) that the model can process. These tokens can represent:

- Individual characters
- Subword units (e.g., "playing" → "play" + "##ing")
- Complete words
- Special symbols

## 🎯 Quick Start

```python
from legollm.core.tokenization import SimpleTokenizer

# Load vocabulary and tokenize
tokenizer = SimpleTokenizer.from_file("data/vocab.json")
token_ids = tokenizer.encode("I'd rather like to tell you")
# Output: [42, 15, 89, 12, ...]

# Convert back to text
text = tokenizer.decode(token_ids)
# Output: "I'd rather like to tell you"
```

______________________________________________________________________

## 📊 Dataset Selection

### The Verdict by Shakespeare (20KB)

- **Source**: [the-verdict.txt](https://github.com/pytholic/LegoLLM/blob/main/data/the-verdict.txt)
- **Size**: 20,479 characters
- **Why chosen**: Small enough for fast iteration, complex enough for real-world patterns (contractions, punctuation, varied vocabulary)

______________________________________________________________________

## 🔧 Core Components

### 1. VocabularyBuilder

**Purpose**: Construct a word-to-ID mapping from raw tokens

**Key Method**:

```python
def build_from_tokens(self, tokens: list[str]) -> dict[str, int]:
    """
    Build vocabulary from deduplicated tokens.

    Why: We need a fixed vocabulary before we can encode text.
    This solves the "chicken-and-egg" problem: tokenizer needs vocab,
    but vocab needs tokenization.
    """
```

**Design Decision**: Separate `tokenize()` from `encode()` to break the circular dependency.

______________________________________________________________________

### 2. Tokenizer Interface

All tokenizers implement three core methods:

#### `tokenize(text: str) -> list[str]`

- **Purpose**: Split text into token strings
- **Use case**: Vocabulary building, debugging
- **Example**: `"I'd go"` → `["I", "'", "d", "go"]`

#### `encode(text: str) -> list[int]`

- **Purpose**: Convert text to token IDs for model input
- **Use case**: Training, inference
- **Example**: `"I'd go"` → `[42, 15, 89, 12]`

#### `decode(token_ids: list[int]) -> str`

- **Purpose**: Convert token IDs back to human-readable text
- **Use case**: Interpreting model outputs
- **Example**: `[42, 15, 89, 12]` → `"I'd go"`

______________________________________________________________________

## WhitespaceTokenizer (Baseline Implementation)

**Algorithm**:

1. Separate punctuation from words
2. Split on whitespace
3. Remove empty strings

**Limitations**:

- ❌ Large vocabulary (every word is unique)
- ❌ Can't handle unknown words
- ❌ No subword tokenization

**Why implemented**: Establishes baseline before moving to BPE/WordPiece

```python
# Example usage
tokenizer = WhitespaceTokenizer()
tokens = tokenizer.tokenize("I'd rather go.")
# Output: ["I", "'", "d", "rather", "go", "."]
```

______________________________________________________________________

## SimpleTokenizer (Vocabulary-Based)

**Enhancement over WhitespaceTokenizer**:

- ✅ Uses pre-built vocabulary
- ✅ Handles unknown tokens via `<|UNK|>`
- ✅ Supports special tokens (`<|endoftext|>`)

**Special Tokens**:

- `<|UNK|>`: Handles out-of-vocabulary words
- `<|endoftext|>`: Marks document boundaries (critical for training)

```python
# Building vocabulary
vocab_builder = VocabularyBuilder()
tokens = tokenizer.tokenize(raw_text)
vocab = vocab_builder.build_from_tokens(tokens)

# Saving for reuse
vocabulary_manager.save(vocab, "data/vocab.json")
```

______________________________________________________________________

## 💡 Key Design Patterns

### Pattern 1: Separation of Concerns

```python
VocabularyBuilder  # Builds vocab
VocabularyManager  # Saves/loads vocab
Tokenizer          # Uses vocab for encoding
```

**Why**: Each class has a single responsibility, making testing and extension easier.

### Pattern 2: Special Token Handling

```python
vocab = {
    "<|UNK|>": 0,      # Unknown words
    "<|endoftext|>": 1, # Document separator
    ...
}
```

**Why**: Special tokens are essential for:

- Handling out-of-vocabulary words gracefully
- Marking sequence boundaries during training
- Future extensibility (padding, masking, etc.)

______________________________________________________________________

## 🧪 Example Workflow

### Step 1: Build Vocabulary

```python
from legollm.tokenization import VocabularyBuilder, WhitespaceTokenizer

raw_text = read_file("data/the-verdict.txt")
tokenizer = WhitespaceTokenizer()
tokens = tokenizer.tokenize(raw_text)

vocab_builder = VocabularyBuilder()
vocab = vocab_builder.build_from_tokens(tokens)
print(f"Vocabulary size: {len(vocab)}")
# Output: Vocabulary size: 1,234
```

### Step 2: Encode Text

```python
from legollm.tokenization import SimpleTokenizer

tokenizer = SimpleTokenizer(vocab)
text = "'I'd rather like to tell you--because I've always suspected you.'"
ids = tokenizer.encode(text)
print(ids)
# Output: [42, 15, 89, 12, 56, ...]
```

### Step 3: Decode IDs

```python
decoded_text = tokenizer.decode(ids)
print(decoded_text)
# Output: "'I'd rather like to tell you--because I've always suspected you.'"
```

______________________________________________________________________

## 🚧 Known Limitations & Next Steps

### Current Limitations

1. **Vocabulary explosion**: Every unique word gets an ID (inefficient)
2. **No subword handling**: Can't decompose unknown words
3. **Poor generalization**: Struggles with morphological variants ("play" vs "playing")

### Week 2 Preview: BPE Tokenization

- ✅ Subword tokenization (handle "playing" → "play" + "##ing")
- ✅ Smaller vocabulary with better coverage
- ✅ Algorithm: Byte-Pair Encoding (used in GPT models)

______________________________________________________________________

## 📚 References

- [Hugging Face Tokenizers Guide](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Let's build a GPT Tokenizer by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Sebastian Raschka's LLM Course - Chapter 2](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11)

______________________________________________________________________

## 🎓 Learning Outcomes

After completing Week 1, you should be able to:

- ✅ Explain why tokenization is necessary for LLMs
- ✅ Implement a basic whitespace tokenizer from scratch
- ✅ Build and manage vocabularies programmatically
- ✅ Handle special tokens (`<|UNK|>`, `<|endoftext|>`)
- ✅ Understand the limitations that motivate BPE/WordPiece algorithms
