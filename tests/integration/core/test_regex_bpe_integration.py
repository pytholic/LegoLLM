"""Integration tests for RegexBPETokenizer.

Tests end-to-end workflows and real-world usage scenarios.

Created by @pytholic on 2025.10.13
"""

from pathlib import Path

import pytest

from legollm.core.tokenization.bpe.regex_bpe_tokenizer import RegexBPETokenizer


@pytest.fixture
def sample_corpus() -> str:
    """Sample corpus for training."""
    return """
    The quick brown fox jumps over the lazy dog.
    Hello world! This is a test of the tokenizer.
    Machine learning is fascinating and powerful.
    Natural language processing enables many applications.
    """


@pytest.fixture
def verdict_corpus() -> str:
    """Load the-verdict.txt corpus for realistic testing."""
    corpus_path = Path(__file__).parent.parent.parent.parent / "data" / "the-verdict.txt"
    if corpus_path.exists():
        return corpus_path.read_text(encoding="utf-8")
    # Fallback if file doesn't exist
    return "Sample text for testing purposes."


class TestRegexBPETokenizerEndToEnd:
    """Test complete end-to-end workflows."""

    def test_train_encode_decode_workflow(self, sample_corpus: str):
        """Test complete workflow: train -> encode -> decode."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(sample_corpus, vocab_size=300)

        # Test on training data
        encoded = tokenizer.encode(sample_corpus)
        decoded = tokenizer.decode(encoded)

        assert decoded == sample_corpus
        assert len(encoded) <= len(sample_corpus.encode("utf-8"))  # Should not expand

    def test_train_save_load_encode_workflow(self, sample_corpus: str, tmp_path: Path):
        """Test workflow: train -> save -> load -> encode."""
        # Train and save
        tokenizer = RegexBPETokenizer()
        tokenizer.train(sample_corpus, vocab_size=300)
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(save_path))

        # Load in new instance
        new_tokenizer = RegexBPETokenizer()
        new_tokenizer.load(str(save_path))

        # Test encoding produces same results
        simple_training_text = "Hello world test"
        original_encoded = tokenizer.encode(simple_training_text)
        loaded_encoded = new_tokenizer.encode(simple_training_text)

        assert original_encoded == loaded_encoded

    def test_multilingual_workflow(self):
        """Test workflow with multilingual text."""
        multilingual_text = """
        English: Hello world!
        Korean: 안녕하세요 세계!
        Japanese: こんにちは世界!
        French: Bonjour le monde!
        Spanish: ¡Hola mundo!
        """

        tokenizer = RegexBPETokenizer()
        tokenizer.train(multilingual_text, vocab_size=350)

        # Test each language
        simple_training_texts = [
            "Hello world!",
            "안녕하세요",
            "こんにちは",
            "Bonjour",
            "¡Hola!",
        ]

        for text in simple_training_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text

    def test_special_tokens_workflow(self, sample_corpus: str):
        """Test workflow with special tokens."""
        special_tokens = {
            "<|endoftext|>": 100257,
            "<|startoftext|>": 100258,
        }

        tokenizer = RegexBPETokenizer()
        tokenizer.train(sample_corpus, vocab_size=300, special_tokens=special_tokens)

        # Test with special tokens
        simple_training_text = "<|startoftext|>Hello world<|endoftext|>"
        encoded = tokenizer.encode(simple_training_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == simple_training_text
        assert 100257 in encoded
        assert 100258 in encoded


class TestRegexBPETokenizerRealWorldScenarios:
    """Test real-world usage scenarios."""

    # combine all below into one test with pytets.parametrize and pytest.param
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param(
                """
def hello_world():
    print("Hello, world!")
    return 42
""",
                id="code",
            ),
            pytest.param(
                """
# Heading 1
## Heading 2

This is a **bold** text and this is *italic*.

- List item 1
- List item 2
```python
code block
```
""",
                id="markdown",
            ),
            pytest.param(
                """
{
    "name": "test",
    "value": 123,
    "nested": {
        "key": "value"
    }
}
""",
                id="json",
            ),
            pytest.param(
                """
User: What is 2 + 2?
Assistant: The answer is 4.

Here's a Python example:
```python
result = 2 + 2
print(f"Result: {result}")
```

The calculation is straightforward!
""",
                id="mixed_content",
            ),
        ],
    )
    def test_various_text_tokenization(self, text: str):
        """Test tokenizing code snippets."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(text * 5, vocab_size=300)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    @pytest.mark.parametrize(
        "text,vocab_size",
        [
            pytest.param("Short text.", 270, id="small_vocab"),
            pytest.param("Medium length text with more words.", 300, id="medium_vocab"),
            pytest.param("Longer text " * 50, 400, id="large_vocab"),
        ],
    )
    def test_different_vocab_sizes(self, text: str, vocab_size: int):
        """Test training with different vocabulary sizes."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(text, vocab_size=vocab_size)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text
        assert len(tokenizer.vocab) <= vocab_size


class TestRegexBPETokenizerCompressionEfficiency:
    """Test compression efficiency of the tokenizer."""

    def test_compression_ratio(self, sample_corpus: str):
        """Test that BPE doesn't expand text."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(sample_corpus, vocab_size=300)

        encoded = tokenizer.encode(sample_corpus)
        original_bytes = len(sample_corpus.encode("utf-8"))
        compressed_tokens = len(encoded)

        # Should not expand (may not compress on small text)
        assert compressed_tokens <= original_bytes

    def test_repeated_text_compression(self):
        """Test compression on highly repetitive text."""
        repeated_text = "Hello world! " * 100
        tokenizer = RegexBPETokenizer()
        tokenizer.train(repeated_text, vocab_size=300)

        encoded = tokenizer.encode(repeated_text)
        original_bytes = len(repeated_text.encode("utf-8"))
        compressed_tokens = len(encoded)

        # Should not expand on repetitive text
        assert compressed_tokens <= original_bytes

    @pytest.mark.parametrize(
        "vocab_size",
        [
            pytest.param(270, id="small_vocab"),
            pytest.param(300, id="medium_vocab"),
            pytest.param(400, id="large_vocab"),
        ],
    )
    def test_vocab_size_affects_compression(self, sample_corpus: str, vocab_size: int):
        """Test that tokenizer works with different vocabulary sizes."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(sample_corpus, vocab_size=vocab_size)

        encoded = tokenizer.encode(sample_corpus)
        original_bytes = len(sample_corpus.encode("utf-8"))
        compression_ratio = len(encoded) / original_bytes

        # Should not expand text
        assert compression_ratio <= 1.0


class TestRegexBPETokenizerPatternComparison:
    """Test different regex patterns and their effects."""

    def test_gpt2_vs_gpt4_pattern(self, simple_training_text: str):
        """Test GPT-2 vs GPT-4 pattern behavior."""
        # Train with GPT-2 pattern
        tokenizer_gpt2 = RegexBPETokenizer(pattern=RegexBPETokenizer.GPT2_SPLIT_PATTERN)
        tokenizer_gpt2.train(simple_training_text * 10, vocab_size=300)

        # Train with GPT-4 pattern
        tokenizer_gpt4 = RegexBPETokenizer(pattern=RegexBPETokenizer.GPT4_SPLIT_PATTERN)
        tokenizer_gpt4.train(simple_training_text * 10, vocab_size=300)

        # Both should successfully encode/decode
        encoded_gpt2 = tokenizer_gpt2.encode(simple_training_text)
        decoded_gpt2 = tokenizer_gpt2.decode(encoded_gpt2)

        encoded_gpt4 = tokenizer_gpt4.encode(simple_training_text)
        decoded_gpt4 = tokenizer_gpt4.decode(encoded_gpt4)

        assert decoded_gpt2 == simple_training_text
        assert decoded_gpt4 == simple_training_text

    def test_custom_pattern(self, simple_training_text: str):
        """Test custom regex pattern."""
        # Simple word-based pattern
        custom_pattern = r"\w+|\s+|[^\w\s]+"
        tokenizer = RegexBPETokenizer(pattern=custom_pattern)
        tokenizer.train(simple_training_text * 10, vocab_size=300)

        encoded = tokenizer.encode(simple_training_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == simple_training_text


class TestRegexBPETokenizerLargeCorpus:
    """Test with larger, realistic corpus."""

    def test_verdict_corpus_training(self, verdict_corpus: str):
        """Test training on the-verdict.txt corpus."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(verdict_corpus, vocab_size=350)

        assert tokenizer.is_trained
        assert len(tokenizer.vocab) <= 350

        # Test on a sample from the corpus
        sample = verdict_corpus[:500]
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)

        assert decoded == sample

    def test_verdict_corpus_compression(self, verdict_corpus: str):
        """Test that tokenizer doesn't expand text on realistic corpus."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(verdict_corpus, vocab_size=350)

        # Test compression on full corpus
        encoded = tokenizer.encode(verdict_corpus)
        original_bytes = len(verdict_corpus.encode("utf-8"))
        compressed_tokens = len(encoded)

        compression_ratio = compressed_tokens / original_bytes
        # Should not expand text
        assert compression_ratio <= 1.0

    def test_verdict_corpus_persistence(self, verdict_corpus: str, tmp_path: Path):
        """Test save/load with realistic corpus."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(verdict_corpus, vocab_size=350)

        # Save
        save_path = tmp_path / "verdict_tokenizer.json"
        tokenizer.save(str(save_path))

        # Load and verify
        new_tokenizer = RegexBPETokenizer()
        new_tokenizer.load(str(save_path))

        # Test on sample
        sample = "I had always thought Jack Gisburn rather a cheap genius"
        original_encoded = tokenizer.encode(sample)
        loaded_encoded = new_tokenizer.encode(sample)

        assert original_encoded == loaded_encoded


class TestRegexBPETokenizerConsistency:
    """Test consistency and determinism."""

    def test_training_determinism(self, sample_corpus: str):
        """Test that training produces deterministic results."""
        tokenizer1 = RegexBPETokenizer()
        tokenizer1.train(sample_corpus, vocab_size=300)

        tokenizer2 = RegexBPETokenizer()
        tokenizer2.train(sample_corpus, vocab_size=300)

        # Should produce identical vocabularies and merges
        assert tokenizer1.vocab == tokenizer2.vocab
        assert tokenizer1.merges == tokenizer2.merges

    def test_encoding_determinism(self, sample_corpus: str):
        """Test that encoding is deterministic."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(sample_corpus, vocab_size=300)

        simple_training_text = "Hello world test"
        encoded1 = tokenizer.encode(simple_training_text)
        encoded2 = tokenizer.encode(simple_training_text)

        assert encoded1 == encoded2

    def test_multiple_encode_decode_cycles(self, sample_corpus: str):
        """Test multiple encode/decode cycles preserve text."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(sample_corpus, vocab_size=300)

        text = "Hello world! This is a test."

        # Multiple cycles
        for _ in range(5):
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text
            text = decoded  # Use decoded as input for next cycle


class TestRegexBPETokenizerEdgeCasesIntegration:
    """Integration tests for edge cases."""

    def test_empty_corpus_handling(self):
        """Test behavior with minimal corpus."""
        minimal_corpus = "aa"
        tokenizer = RegexBPETokenizer()
        tokenizer.train(minimal_corpus, vocab_size=257)

        encoded = tokenizer.encode("aa")
        decoded = tokenizer.decode(encoded)

        assert decoded == "aa"

    def test_unicode_edge_cases_integration(self):
        """Test various Unicode scenarios."""
        unicode_text = """
        Emoji: 🚀 🌍 ❤️
        Math: ∑ ∫ √ π
        Symbols: © ® ™
        CJK: 中文 日本語 한국어
        """

        tokenizer = RegexBPETokenizer()
        tokenizer.train(unicode_text * 5, vocab_size=400)

        encoded = tokenizer.encode(unicode_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == unicode_text

    def test_special_tokens_in_real_text(self):
        """Test special tokens mixed with realistic text."""
        corpus = "Hello world! This is a test."
        special_tokens = {"<|endoftext|>": 100257}

        tokenizer = RegexBPETokenizer()
        tokenizer.train(corpus, vocab_size=300, special_tokens=special_tokens)

        # Simulate conversation with special tokens
        conversation = """
Hello, how are you?<|endoftext|>
I'm doing well, thank you!<|endoftext|>
That's great to hear.<|endoftext|>
"""

        encoded = tokenizer.encode(conversation)
        decoded = tokenizer.decode(encoded)

        assert decoded == conversation
        assert encoded.count(100257) == 3  # Three special tokens

    @pytest.mark.parametrize(
        "test_input",
        [
            pytest.param("", id="empty_string"),
            pytest.param(" ", id="single_space"),
            pytest.param("\n", id="single_newline"),
            pytest.param("a", id="single_char"),
            pytest.param("   \n\n   ", id="only_whitespace"),
        ],
    )
    def test_minimal_inputs(self, test_input: str):
        """Test minimal and edge case inputs."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train("Hello world test", vocab_size=280)

        encoded = tokenizer.encode(test_input)
        if test_input:  # Non-empty
            decoded = tokenizer.decode(encoded)
            assert decoded == test_input
        else:  # Empty
            assert encoded == []
