"""Integration tests for tokenization components.

Tests the interaction between VocabularyBuilder and SimpleTokenizer
to ensure they work together correctly.

Created by @pytholic on 2025.09.21
"""

import pytest

from legollm.core.exceptions import TokenizerError
from legollm.core.tokenization.tokenizer import SimpleTokenizer
from legollm.core.tokenization.vocabulary import UNK_TOKEN, VocabularyBuilder


class TestTokenizationIntegration:
    """Integration tests for tokenization workflow."""

    def test_vocabulary_builder_with_tokenizer_basic_workflow(self):
        """Test the complete workflow: tokenize -> build vocab -> encode/decode."""
        # Sample text for training
        training_text = "Hello world! This is a test. How are you doing today?"

        # Step 1: Tokenize the training text
        tokenizer = SimpleTokenizer()
        training_tokens = tokenizer.tokenize(training_text)

        # Step 2: Build vocabulary from tokens
        vocab_builder = VocabularyBuilder()
        vocab = vocab_builder.build_from_tokens(training_tokens)

        # Step 3: Create tokenizer with vocabulary
        trained_tokenizer = SimpleTokenizer(vocab)

        # Step 4: Test encoding and decoding
        encoded = trained_tokenizer.encode(training_text)
        decoded = trained_tokenizer.decode(encoded)

        # Verify the roundtrip works
        assert decoded == training_text
        assert isinstance(encoded, list)
        assert all(isinstance(token_id, int) for token_id in encoded)

    def test_unknown_token_handling_integration(self):
        """Test that unknown tokens are properly handled in the full workflow."""
        # Training text (limited vocabulary)
        training_text = "Hello world test"

        # Test text with unknown tokens
        test_text = "Hello world! This is a new test with unknown words."
        expected_decoded = "Hello world <|UNK|> <|UNK|> <|UNK|> <|UNK|> <|UNK|> test <|UNK|> <|UNK|> <|UNK|> <|UNK|>"

        # Build vocabulary from limited training text
        tokenizer = SimpleTokenizer()
        training_tokens = tokenizer.tokenize(training_text)

        vocab_builder = VocabularyBuilder()
        vocab = vocab_builder.build_from_tokens(training_tokens)

        # Create tokenizer with limited vocabulary
        trained_tokenizer = SimpleTokenizer(vocab)

        # Test encoding and decoding with unknown tokens
        encoded = trained_tokenizer.encode(test_text)
        decoded = trained_tokenizer.decode(encoded)

        # Verify unknown tokens are handled correctly
        assert decoded == expected_decoded
        assert UNK_TOKEN in vocab  # UNK token should be in vocabulary

        # Verify that unknown tokens get the UNK token ID
        unk_id = vocab[UNK_TOKEN]
        assert unk_id in encoded  # Should contain unknown token IDs

    def test_vocabulary_consistency_with_tokenizer(self):
        """Test that vocabulary IDs are consistent between encoding and decoding."""
        training_text = "The quick brown fox jumps over the lazy dog."

        # Build vocabulary
        tokenizer = SimpleTokenizer()
        training_tokens = tokenizer.tokenize(training_text)

        vocab_builder = VocabularyBuilder()
        vocab = vocab_builder.build_from_tokens(training_tokens)

        trained_tokenizer = SimpleTokenizer(vocab)

        # Test that each token maps consistently
        for token in training_tokens:
            if token in vocab:  # Skip if token was deduplicated
                token_id = vocab[token]
                # Encode single token
                encoded = trained_tokenizer.encode(token)
                assert token_id in encoded

                # Decode single token ID
                decoded = trained_tokenizer.decode([token_id])
                assert token in decoded

    @pytest.mark.parametrize(
        "edge_case_vocab, should_raise",
        [
            (["hello"], False),
            ([], True),
        ],
    )
    def test_empty_and_edge_cases(self, edge_case_vocab: list[str], should_raise: bool):
        """Test edge cases in the integration workflow."""
        vocab_builder = VocabularyBuilder()

        # Test with empty tokens list should raise error
        if should_raise:
            with pytest.raises(
                TokenizerError, match="Cannot build vocabulary from empty tokens list"
            ):
                vocab_builder.build_from_tokens(edge_case_vocab)

        else:
            # Test with single token
            single_token_vocab = vocab_builder.build_from_tokens(edge_case_vocab)
            tokenizer = SimpleTokenizer(single_token_vocab)

            # Should work with known token
            encoded = tokenizer.encode(edge_case_vocab[0])
            decoded = tokenizer.decode(encoded)
            assert decoded == edge_case_vocab[0]

            # Should handle unknown token
            encoded_unknown = tokenizer.encode("unknown")
            decoded_unknown = tokenizer.decode(encoded_unknown)
            assert UNK_TOKEN in decoded_unknown
