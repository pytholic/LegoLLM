# Week 1: Tokenization Foundation

**Date**: 2025.09.14 ~
**Goal**: Implement simple tokenizer and BPE tokenization from scratch

## Datasets Explored

- The Verdict by Shakespeare (20KB)
  - https://github.com/pytholic/LegoLLM/blob/main/data/the-verdict.txt
  - Chose because: Small, simple, good for testing
  - Total length: 20479 characters (including whitespaces and punctuation I think)

## Text Preprocessing

- Added a vocabulary builder
  - Build vocabulary after removing duplicates

## **Tokenizer Implementation**

- Will include `encode` and `decode` methods
- Also includes `tokenize` method for building initial vocabulary

### **Initial approach**: Simple whitespace tokenizer

- Separate punctuation
- Strip empty strings
- **Problems**
  - Circular vocab issue i.e. we assume we have vocab for tokenizer but we need tokenizer for vocab
    - Solution: Add a `tokenize` method since we first need to build the vocabulary to use it.

**Example**

Build vocabulary:

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

Encode and decode:

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
