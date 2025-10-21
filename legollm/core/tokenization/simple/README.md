# Simple Whitespace Tokenizer

## Implementation

- **Preprocessing**

  - Separate punctuation
  - Strip empty strings
  - Vocabulary builder
    - Build vocabulary from token after removing duplicates
      - `build_from_tokens(self, tokens: list[str]) -> dict[str, int]`

- **Problems**

  - Circular vocab issue i.e. we assume we have vocab for tokenizer but we need tokenizer for vocab
    - Solution: Add a `tokenize` method since we first need to build the vocabulary to use it.

## Example

### Basic Usage

```python
from legollm.core.tokenization import SimpleTokenizer
from legollm.core.utils import read_file

corpus_text = read_file("data/the-verdict.txt")

tokenizer = SimpleTokenizer.from_corpus(corpus_text)
print(f"\n✓ Created tokenizer with {len(tokenizer.vocab)} tokens (including special tokens)")

# Test text
test_text = "'I'd rather like to tell you--because I've always suspected you of loathing my work.'"
print(f"\n📝 Input text:\n{test_text}")

# Encode
ids = tokenizer.encode(test_text)
print(f"\n🔢 Encoded IDs:\n{ids}")

# Decode
decoded_text = tokenizer.decode(ids)
print(f"\n📝 Decoded text:\n{decoded_text}")
```

### Build Vocabulary

```python
def main() -> None:
    """Main function."""
    raw_text = read_file("data/the-verdict.txt")
    vocabulary_builder = VocabularyBuilder()
    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.tokenize(raw_text)
    vocab = vocabulary_builder.build_from_tokens(tokens)
    print(len(vocab))

if __name__ == "__main__":
    main()
```

### Encode and Decode

```python
def main() -> None:
    """Main function."""
    vocabulary_manager = VocabularyManager()
    vocab = vocabulary_manager.load("data/vocab.json")
    tokenizer = SimpleTokenizer(vocab)
    ids = tokenizer.encode(
        "'I'd rather like to tell you--because I've always suspected you of loathing my work.'"
    )
    print(ids)
    text = tokenizer.decode(ids)
    print(text)

if __name__ == "__main__":
    main()
```

## Adding Special Context Tokens

- Add special tokens like `<|UNK|>` and `<|endoftext|>`
