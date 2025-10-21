# Byte Pair Encoding

**Goal**: Understand and implement BPE algorithm from scratch

## Overview

- The BPE algorithm was originally described in 1994: "[A New Algorithm for Data Compression](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)" by Philip Gage.
- First introduced in NMT in the paper "[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)" in 2015.
- The original BPE tokenizer that OpenAI implemented for training the original GPT models can be found [here](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- There's also an implementation called [minBPE](https://github.com/karpathy/minbpe) with training support, which is maybe more performant
- Many projects nowadays use OpenAI's open-source [tiktoken library](https://github.com/openai/tiktoken) due to its computational performance
- BPE has been adapted by many recent language models like BART, RoBERTa, GPT-2 to GPT-4, Llama 3, etc.
- BPE tokenizer works with words or subwords instead of each character

## Bytes and UTF-8

Python strings are immutable sequences of Unicode code points.

### Why Bytes?

Using simple characters (Unicode) has following issues:

1. **Unbounded vocabulary** - Can't enumerate all possible characters
2. **Language bias** - What characters to include?
3. **New characters** - Emoji/symbols added to Unicode constantly

BPE resolves these issues by relying on UTF-8 standard which offers several benefits:

- **Universal base** - 256 bytes represent ALL possible text
- **No unknown tokens** - Can always fall back to bytes
- **Language agnostic** - Chinese, Arabic, Emoji all work
- **Efficient** - Common patterns get merged into single tokens
- **Stable** - UTF-8 standard won't change

> **💡 Note**
>
> UTF-16 and 32 might result in waste of bytes (0 bytes). UTF-8 seems like a nice choice.

Example vocab size:

- **Base**: 256 (bytes 0-255) - handles universality
- **Merges**: ~50,000 learned tokens - handles efficiency
- **Total**: ~50,256 tokens

```
Character-Level Tokenization:
┌─────────────────────────────┐
│ Need vocab entry for EVERY  │
│ possible character ever     │
│ Unicode: 149,186 characters!│
└─────────────────────────────┘
        ↓
    ❌ Impossible

Byte-Level BPE:
┌─────────────────────────────┐
│ Base: 256 bytes (universal) │
│ + Learned: ~50k merges      │
│ = ~50k total vocab          │
└─────────────────────────────┘
        ↓
    ✅ Practical!
```

Convert text to bytes using `bytes`, `bytearray` or `.encode("utf-8")`. We can call `list()` method to get raw bytes.

```python
text = "This is some text"
byte_arr = bytearray(text, "utf-8")
print(byte_arr)  # bytearray(b'This is some text')
```

We can call `list()` method to get raw byte IDs.

### The Need for BPE Merges

- At this point, we have integer IDs and this would be a perfectly valid way to convert text into a token ID representation that we need for the embedding layer of an LLM.
- However, the downside of this approach is that it is creating one ID for each character (that's a lot of IDs for a short text!)
  - This renders us unable to process long sequence lengths since the attention layer in transformer has limited context window.
  - Therefore, we need a way to compress this information. The denser the information, the better!
- The solution is to compress this information using BPE merges.

## BPE Algorithm

### Core Steps

**Step 1: Identify frequent pairs**

- Iteratively find a pair of token IDs that occur most frequently

**Step 2: Replace the pair**

- Replace that pair with a new placeholder ID (one not already in use, e.g., if we start with 0…255, the first placeholder would be 256)
- Add this new ID to the vocabulary lookup table
- The size of the lookup table is a hyperparameter, also called "vocabulary size" (for GPT-2, that's 50,257)

**Step 3: Stopping condition**

- Keep repeating steps 1 and 2, continually merging the most frequent pairs
- Stop when no further compression is possible (e.g., no pair occurs more than once)

*The more steps we take, the larger will be our vocabulary and shorter will be our sequence. We have to find the sweet spot which is a hyperparameter.*

**Decompression (decoding)**

- To restore the original text, reverse the process by substituting each ID with its corresponding pair, using the lookup table

### Example 1: "aaabdaaabac"

#### Compression (Encoding)

Input text: `"aaabdaaabac"`

**Initial State**

```
Text:  a  a  a  b  d  a  a  a  b  a  c
IDs:  [0, 0, 0, 1, 3, 0, 0, 0, 1, 0, 2]

Vocab: {0:'a', 1:'b', 2:'c', 3:'d'}
```

**Step 1: Merge most frequent pair**

Find: `(a, a)` appears 4 times → Merge into `aa`

```
Text:  aa  a  b  d  aa  a  b  a  c
IDs:  [4,  0, 1, 3,  4, 0, 1, 0, 2]

Vocab: {0:'a', 1:'b', 2:'c', 3:'d', 4:'aa'}
```

**Step 2: Merge next frequent pair**

Find: `(aa, a)` appears 2 times → Merge into `aaa`

```
Text:  aaa  b  d  aaa  b  a  c
IDs:  [ 5,  1, 3,  5,  1, 0, 2]

Vocab: {0:'a', 1:'b', 2:'c', 3:'d', 4:'aa', 5:'aaa'}
```

**Step 3: Merge next frequent pair**

Find: `(aaa, b)` appears 2 times → Merge into `aaab`

```
Text:  aaab  d  aaab  a  c
IDs:  [ 6,   3,  6,   0, 2]

Vocab: {0:'a', 1:'b', 2:'c', 3:'d', 4:'aa', 5:'aaa', 6:'aaab'}
```

**Final Compressed Result**

```
Original: 11 characters → "aaabdaaabac"
Encoded:   5 tokens    → [6, 3, 6, 0, 2]
```

> **💡 Key Insight**
>
> Previously merged tokens like 'aa' and 'aaa' themselves become candidates for further merging, creating a hierarchical build-up of common patterns.

#### Decompression (Decoding)

**Given:** `[6, 3, 6, 0, 2]`

**Step 1: Look up each ID in vocabulary**

```
[6,     3,   6,     0,   2]
 ↓      ↓    ↓      ↓    ↓
'aaab' 'd' 'aaab' 'a'  'c'
```

**Step 2: Expand merged tokens (reverse order)**

```
Iteration 1: 'aaab' → 'aaa' + 'b'
Result: ['aaa', 'b', 'd', 'aaa', 'b', 'a', 'c']

Iteration 2: 'aaa' → 'aa' + 'a'
Result: ['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']

Iteration 3: 'aa' → 'a' + 'a'
Result: ['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']
```

**Step 3: Join all tokens**

```
Output: "aaabdaaabac" ✅ (matches original!)
```

### Example 2: "low lower lowest flow flower"

#### Compression (Encoding)

**Initial State**

```
Text: "low lower lowest flow flower"

Vocabulary (assume 256 base tokens: 0-255 for all possible bytes):
0-255: All byte values (includes ASCII characters)

Example mappings:
32: ' ' (space)
101: 'e'
102: 'f'
108: 'l'
111: 'o'
119: 'w'
...
```

**Iteration 1**

1. Identify frequent pairs: `"lo"` appears 3 times (in "low", "lower", "lowest")
2. Replace and record:
   - Replace `"lo"` with new token ID `256`
   - New text: `<256>w <256>wer <256>west f<256>w f<256>wer`
3. Updated vocabulary:

```
0-255: All byte values
256: "lo"
```

**Iteration 2**

1. Identify frequent pairs: `<256>w` appears 4 times (in all words)
2. Replace and record:
   - Replace `<256>w` with new token ID `257`
   - New text: `<257> <257>er <257>est f<257> f<257>er`
3. Updated vocabulary:

```
0-255: All byte values
256: "lo"
257: "<256>w"  (which is "low")
```

**Iteration 3**

1. Identify frequent pairs: `<257>e` appears 2 times (in "lower" and "lowest")
2. Replace and record:
   - Replace `<257>e` with new token ID `258`
   - New text: `<257> <258>r <258>st f<257> f<258>r`
3. Updated vocabulary:

```
0-255: All byte values
256: "lo"
257: "<256>w"  (which is "low")
258: "<257>e"  (which is "lowe")
```

**Final Compressed Result**

```
Original: "low lower lowest flow flower" (29 characters)
Encoded:  [257, 32, 258, 114, 32, 258, 115, 116, 32, 102, 257, 32, 102, 258, 114]
          (low  _   lowe r    _   lowe s    t    _   f    low  _   f    lowe r)
Compression: 29 chars → 15 tokens
```

#### Decompression (Decoding)

**Given encoded sequence:** `[257, 32, 258, 114, 32, 258, 115, 116, 32, 102, 257, 32, 102, 258, 114]`

**Step 1: Look up each ID in vocabulary**

```
Token ID 257 → "<256>w" → expand to "low"
Token ID 32  → ' '
Token ID 258 → "<257>e" → expand to "lowe"
Token ID 114 → 'r'
...
```

**Step 2: Expand merged tokens**

Expand 257:

```
257 → "<256>w"
    → expand 256: "lo" + "w"
    → Result: "low"
```

Expand 258:

```
258 → "<257>e"
    → expand 257: "low" + "e"
    → Result: "lowe"
```

**Step 3: Reconstruct final text**

```
[257,  32,  258,   114, 32,  258,   115, 116, 32,  102, 257,  32,  102, 258,   114]
 ↓     ↓    ↓      ↓    ↓    ↓      ↓    ↓    ↓    ↓    ↓     ↓    ↓    ↓      ↓
"low"+ ' '+"lowe"+'r'+ ' '+"lowe"+'s'+ 't'+ ' '+ 'f'+"low"+ ' '+ 'f'+"lowe"+'r'

Result: "low lower lowest flow flower" ✅
```

### Key Takeaways

1. **Start with 256 base byte tokens**: IDs 0-255 cover all possible byte values
2. **New merges get IDs 256+**: Each frequent pair gets a unique new token ID
3. **Hierarchical merges**: Token 258 references 257, which references 256
4. **Decompression expands recursively**: Follow the merge chain backwards
5. **Efficient compression**: Common subwords become single tokens

## GPT-2's Regex-based BPE (2019)

### The Problem with Naive Byte-Level BPE

**Standard BPE:** Merges anything frequent → wastes vocab on "dog.", "dog!", "dog?" (BPE merges punctuation with words).

Excerpt from GPT-2 paper:

> However, directly applying BPE to the byte sequence results in suboptimal merges due to BPE using a greedy frequency based heuristic for building the token vocabulary. We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog? . This results in a sub-optimal allocation of limited vocabulary slots and model capacity. To avoid this, we prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

```python
# Training text contains:
"I like dogs. I see dogs! Where are dogs?"

# Naive BPE would create separate tokens for:
merges = {
    ('d','o'): 256,
    ('do','g'): 257,
    ('dog','.'): 258,  # ← Wastes a token slot
    ('dog','!'): 259,  # ← Wastes a token slot
    ('dog','?'): 260,  # ← Wastes a token slot
}

# Result: "dog.", "dog!", "dog?" are all different tokens!
# But they're the SAME word with different punctuation.
```

**Problem:**

- Vocabulary fills up with punctuation variants
- "dog", "dog.", "dog!", "dog?" each take separate slots
- Inefficient use of limited vocabulary (GPT-2 has 50,257 tokens max)

### GPT-2's Solution: Don't Merge Across Character Categories

- ✅ Merge within categories (letters, numbers, etc.)
- ✅ Exception: Allow space merges for compression
- ❌ Block cross-category (no "dog." tokens)
- **Result:** More efficient vocabulary, better model capacity usage

#### Character Categories

```python
# Letters: a-z, A-Z
# Numbers: 0-9
# Punctuation: . ! ? , ; :
# Whitespace: space, tab, newline
```

#### Rule: Don't merge different categories

```python
# ✅ ALLOWED: Same category merges
('d','o') → 256      # letter + letter
('2','0') → 257      # number + number
('!','!') → 258      # punctuation + punctuation

# ❌ BLOCKED: Cross-category merges
('dog','.') → ✗      # letter + punctuation (blocked!)
('dog','!') → ✗      # letter + punctuation (blocked!)
('dog','?') → ✗      # letter + punctuation (blocked!)
```

**Result:** One token for "dog", separate tokens for punctuation

```python
# Encoding:
"dogs." → [token_dog, token_s, token_period]
"dogs!" → [token_dog, token_s, token_exclamation]
"dogs?" → [token_dog, token_s, token_question]

# Now "dog" is shared across all contexts!
```

> **ℹ️ Note**
>
> Spaces were exempted from this rule because:
>
> - Spaces always precede words (predictable pattern)
> - Doesn't create fragmentation like punctuation
> - By allowing `(space, letter)` merges, these become single high-value tokens, giving **massive compression**.

The authors restricted certain merges using special regex patterns:

- Regex pattern used can be found [here](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L53).
- It works by splitting text, and then performing merges in those splits independently, followed by concatenation in the end

## Implementation

### Implementation Strategy

Keep in mind that a lot of complex unicode characters can take multiple bytes (up to 4). Therefore the length of text might be different from the number of encoded token ids.

#### Core Helper Functions

**1. Compute Pair Frequencies**

Compute frequency for each consecutive pair in the text.

```python
def _compute_pair_freq(token_ids: list[int]) -> list[tuple[tuple[int, int], int]] | None:
    """Compute frequency of each consecutive pair in the text.

    Args:
        token_ids: List of token IDs.

    Returns:
        List of (pair, frequency) tuples sorted by frequency, or None if no pairs.
    """
    pair_counts = Counter(pairwise(token_ids))
    if not pair_counts:
        return None
    return pair_counts.most_common()
```

**2. Find Most Frequent Pair**

```python
def _find_most_freq_pair(
    pair_freq: list[tuple[tuple[int, int], int]] | None,
) -> tuple[tuple[int, int] | None, int | None]:
    """Find the most frequent pair from pair frequencies.

    Args:
        pair_freq: List of (pair, frequency) tuples.

    Returns:
        Tuple of (most_frequent_pair, occurrences) or (None, None) if empty.
    """
    if not pair_freq:
        return None, None
    pair_id, occurrences = pair_freq[0]
    return pair_id, occurrences
```

**3. Merge Pair**

Swap the most frequent pair with new token ID. Careful not to run out of bounds in the very last position.

```python
def _merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Merge all occurrences of a pair with a new token ID.

    Args:
        token_ids: List of token IDs.
        pair: Pair of token IDs to merge.
        new_id: New token ID to replace the merged pair.

    Returns:
        List of token IDs with the pair replaced by new_id.
    """
    dq = deque(token_ids)
    replaced_token_ids: list[int] = []

    while dq:
        current = dq.popleft()
        if dq and (current, dq[0]) == pair:
            replaced_token_ids.append(new_id)
            dq.popleft()
        else:
            replaced_token_ids.append(current)

    return replaced_token_ids
```

Test it:

```python
test_token_ids = [5, 6, 6, 7, 8, 9, 9, 10]
merged_token_ids = _merge_pair(test_token_ids, (6, 7), 100)
print(merged_token_ids)  # [5, 6, 100, 8, 9, 9, 10]
```

### Naive Implementation

#### Architecture

```python
class NaiveBPETokenizer:
    """Byte-Pair Encoding tokenizer - requires training.

    This is a naive implementation for educational purposes.
    It demonstrates the core BPE algorithm without optimizations.
    """

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self._is_trained = False
```

#### Training

The `train` method builds the vocabulary by:

1. Converting text to UTF-8 bytes
2. Iteratively finding and merging the most frequent pairs
3. Stopping when vocab_size is reached

```python
def train(
    self,
    text: str,
    vocab_size: int,
    *,
    special_tokens: dict[str, int] | None = None,
    verbose: bool = False,
) -> None:
    """Train the tokenizer on corpus to learn merges/vocabulary.

    Args:
        text: Training corpus.
        vocab_size: Target vocabulary size after training.
        special_tokens: Additional special tokens to include.
        verbose: Whether to print verbose output.
    """
    num_merges = vocab_size - 256  # 256 base bytes
    token_ids = list(text.encode("utf-8"))
    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for i in range(num_merges):
        pair_freq = self._compute_pair_freq(token_ids)
        pair_id, occurrences = self._find_most_freq_pair(pair_freq)

        if pair_id is None:
            break

        idx = 256 + i
        merges[pair_id] = idx
        token_ids = self._merge_pair(token_ids, pair_id, idx)
        vocab[idx] = vocab[pair_id[0]] + vocab[pair_id[1]]

        if verbose:
            print(f"merge {i + 1}/{num_merges}: {pair_id} -> {idx}")

    self.merges = merges
    self.vocab = vocab
    self._is_trained = True
```

#### Encoding

The `encode` method applies learned merges to new text:

1. Convert text to UTF-8 byte IDs
2. Repeatedly find the earliest-learned pair that still exists
3. Merge that pair and continue until no more merges possible

```python
def encode(self, text: str) -> list[int]:
    """Encode text into token IDs using learned merges.

    Apply merges in the order they were learned (by merge index).

    Args:
        text: The input text to encode.

    Returns:
        A list of integer token IDs.
    """
    if not self._is_trained:
        raise TokenizerError("Tokenizer must be trained before encoding.")

    if not text:
        return []

    token_ids = list(text.encode("utf-8"))

    while len(token_ids) >= 2:
        pair_freq = self._compute_pair_freq(token_ids)

        # Find the pair with the lowest merge index (earliest learned)
        pair_id = min(pair_freq, key=lambda p: self.merges.get(p, float("inf")))

        # If this pair was not learned during training, stop
        if pair_id not in self.merges:
            break

        idx = self.merges[pair_id]
        token_ids = self._merge_pair(token_ids, pair_id, idx)

    return token_ids
```

#### Decoding

The `decode` method converts token IDs back to text:

1. Look up each token ID in vocabulary
2. Join all bytes together
3. Decode to UTF-8 string

```python
def decode(self, token_ids: list[int]) -> str:
    """Decode token IDs back to text.

    Args:
        token_ids: A list of integer token IDs.

    Returns:
        The decoded text string.
    """
    if not self._is_trained:
        raise TokenizerError("Tokenizer must be trained before decoding.")

    if not token_ids:
        return ""

    token_bytes = b"".join(self.vocab[idx] for idx in token_ids)
    text = token_bytes.decode("utf-8", errors="replace")
    return text
```

Note: We use `errors="replace"` to:

- Prevent crashes from invalid UTF-8
- BPE might merge bytes that form invalid UTF-8
- Shows where problems occur (� is visible)
- Standard practice in production tokenizers (GPT-2, LLaMA use this)

#### Usage Example

```python
from legollm.core.tokenization import NaiveBPETokenizer
from legollm.core.utils import read_file

# Read training text
text = read_file("data/blog.txt")

# Train tokenizer
tokenizer = NaiveBPETokenizer()
tokenizer.train(text, vocab_size=276, verbose=True)

# Save tokenizer
tokenizer.save("data/bpe_tokenizer.json")
tokenizer.save_readable("data/bpe_tokenizer_readable.txt")

# Load tokenizer
tokenizer.load("data/bpe_tokenizer.json")

# Test encoding/decoding
test_text = "The Tokenizer is a necessary and pervasive component."
encoded = tokenizer.encode(test_text)
print(f"Encoded: {encoded}")

decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
print(f"Match: {decoded == test_text}")
```

#### Saving and Loading

**Using Base64 Encoding**

Note how we use `base64` encoding to save and load our vocabulary and merges. BPE merges bytes without caring about UTF-8 boundaries, creating invalid sequences. Base64 stores raw bytes safely without needing UTF-8 validation.

One example is the continuation bytes as mentioned in this Wikipedia [article](https://arc.net/l/quote/nzrhclcj).

```python
# Your BPE vocab has ALL 256 bytes individually
self.vocab = {idx: bytes([idx]) for idx in range(256)}

# When you try to decode byte 128:
self.vocab[128] = b'\x80'

# For saving to JSON:
self.vocab[128].decode('utf-8')
# ❌ Error: 'invalid start byte'
# Because 0x80 (10000000) is a continuation byte without a start byte!
```

We could also use `token.decode('utf-8', errors='backslashreplace')`. However it has some downsides:

- **Complex loading logic**: Need multi-step encoding dance to restore bytes
- **Error-prone**: Easy to mess up the encode/decode chain
- **Mixed representations**: Some tokens are readable strings, others are escape sequences

*Most tokenizer libraries use `base64`, so it seems safe to stick with it.*

**Human Readable Format**

The `save_readable` method saves vocab and merges in a human readable format:

- To handle invalid UTF-8 characters, we use `errors="replace"` in the decode method
- To avoid having a lot of � symbols in the output file, we further convert such tokens to hex
- We also want to avoid printing control characters as they can distort the output
- We can use `unicodedata.category` or [unicode regular expressions](https://www.regular-expressions.info/unicode.html)

### Regex-based Implementation

#### Architecture

```python
class RegexBPETokenizer(BaseBPETokenizer):
    """Regex-based BPE tokenizer matching GPT-2/GPT-4 behavior.

    Key differences from NaiveBPETokenizer:
    1. Splits text into chunks using regex before applying BPE
    2. Supports special tokens with configurable handling
    3. Prevents merging across character categories
    """

    # GPT-2 regex pattern
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # GPT-4 regex pattern (more sophisticated)
    GPT4_SPLIT_PATTERN = (
        r"'(?i:[sdmt]|ll|ve|re)|"  # Contractions
        r"[^\r\n\p{L}\p{N}]?+\p{L}+|"  # Letters with optional prefix
        r"\p{N}{1,3}|"  # Numbers (1-3 digits)
        r" ?[^\s\p{L}\p{N}]++[\r\n]*|"  # Non-alphanumeric with optional space
        r"\s*[\r\n]|"  # Newlines with optional whitespace
        r"\s+(?!\S)|"  # Whitespace not followed by non-whitespace
        r"\s+"  # Remaining whitespace
    )

    def __init__(self, pattern: str | None = None) -> None:
        """Initialize with optional regex pattern (defaults to GPT-4)."""
        self.pattern: str = pattern or self.GPT4_SPLIT_PATTERN
        self.compiled_pattern: rex.Pattern = rex.compile(self.pattern)
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        super().__init__()
```

#### Understanding the Regex Pattern

The GPT-4 split pattern prevents cross-category merges by pre-splitting the text:

```python
GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|"      # Contractions: I'm, you'll, we've
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|"  # Letters with optional punctuation prefix
    r"\p{N}{1,3}|"                 # Numbers (1-3 digits for better compression)
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|" # Punctuation with optional space
    r"\s*[\r\n]|"                  # Newlines with optional whitespace
    r"\s+(?!\S)|"                  # Whitespace not followed by non-whitespace
    r"\s+"                          # Remaining whitespace
)
```

**Example splitting:**

```python
text = "Hello world! 123 test."
# After regex split:
chunks = ["Hello", " world", "!", " ", "123", " test", "."]

# BPE merges applied within each chunk independently
# This prevents "world!" or "test." from becoming single tokens
```

#### Training with Regex

The training process differs from naive BPE:

```python
def train(
    self,
    text: str,
    vocab_size: int,
    *,
    special_tokens: dict[str, int] | None = None,
    verbose: bool = False,
) -> None:
    """Train the tokenizer on corpus to learn merges/vocabulary.

    Steps:
    1. Split text into chunks using regex pattern
    2. Convert each chunk to byte IDs
    3. Compute pair frequencies across ALL chunks
    4. Merge most frequent pair in ALL chunks
    5. Repeat until vocab_size reached
    """
    num_merges = vocab_size - 256

    # Step 1: Split text into chunks using regex
    text_chunks = self.compiled_pattern.findall(text)

    # Step 2: Convert each chunk to token IDs
    token_ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

    # Step 3: Initialize vocab with base bytes
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = {}

    # Step 4: Iteratively merge most common pairs
    for i in range(num_merges):
        # Compute frequencies across ALL chunks
        pair_freq = self._compute_pair_freq_chunks(token_ids)
        pair_id, occurrences = self._find_most_freq_pair(pair_freq)

        if pair_id is None:
            break

        idx = 256 + i

        # Merge this pair in ALL chunks
        token_ids = [self._merge_pair(chunk_ids, pair_id, idx)
                     for chunk_ids in token_ids]

        # Record merge and update vocab
        merges[pair_id] = idx
        vocab[idx] = vocab[pair_id[0]] + vocab[pair_id[1]]

        if verbose:
            print(f"merge {i + 1}/{num_merges}: {pair_id} -> {idx}")

    self.vocab = vocab
    self.merges = merges
    self.register_special_tokens(special_tokens)
    self._is_trained = True
```

**Key difference:** Pair frequencies are computed across all chunks, not the entire text as a single sequence.

```python
def _compute_pair_freq_chunks(
    self, token_ids: list[list[int]]
) -> list[tuple[tuple[int, int], int]] | None:
    """Compute pair frequencies across all chunks.

    Args:
        token_ids: List of token ID lists (one per chunk)

    Returns:
        List of (pair, frequency) tuples sorted by frequency.
    """
    all_freqs = Counter()

    # Accumulate pair frequencies from all chunks
    for chunk_ids in token_ids:
        pair_freq = self._compute_pair_freq(chunk_ids)
        if pair_freq:
            all_freqs.update(dict(pair_freq))

    if not all_freqs:
        return None

    return all_freqs.most_common()
```

#### Encoding with Regex and Special Tokens

Encoding is more complex because it handles:

1. Special tokens (e.g., `<|endoftext|>`)
2. Regex chunking
3. BPE merges within chunks

```python
def encode(self, text: str) -> list[int]:
    """Encode text into token IDs using learned merges and special tokens.

    Process:
    1. Split text on special tokens (e.g., <|endoftext|>)
    2. For non-special text, apply regex chunking and BPE merges
    3. Return combined token IDs including special token IDs

    Args:
        text: The input text to encode.

    Returns:
        A list of integer token IDs.
    """
    if not self._is_trained:
        raise TokenizerError("Tokenizer must be trained before encoding.")

    if not text:
        return []

    token_ids: list[int] = []

    # If no special tokens, encode normally
    if not self.special_tokens:
        text_chunks = self.compiled_pattern.findall(text)
        for chunk in text_chunks:
            token_ids.extend(self._encode_chunk(chunk))
        return token_ids

    # Build regex pattern to split on special tokens
    special_pattern = "|".join(rex.escape(token) for token in self.special_tokens)
    split_pattern = f"({special_pattern})"

    # Split text on special tokens, keeping the special tokens
    splits = rex.split(split_pattern, text)

    # Process each part
    for split in splits:
        if not split:  # Skip empty strings
            continue

        if split in self.special_tokens:
            # This is a special token, add its ID directly
            token_ids.append(self.special_tokens[split])
        else:
            # This is regular text, apply regex chunking and BPE
            text_chunks = self.compiled_pattern.findall(split)
            for chunk in text_chunks:
                token_ids.extend(self._encode_chunk(chunk))

    return token_ids
```

**Example with special tokens:**

```python
text = "Hello<|endoftext|>world"

# After splitting on special tokens:
splits = ["Hello", "<|endoftext|>", "world"]

# Processing:
# "Hello" → regex chunks → BPE encoding → [token_ids]
# "<|endoftext|>" → special token ID → [100257]
# "world" → regex chunks → BPE encoding → [token_ids]

# Final: [72, 101, 108, 108, 111, 100257, 119, 111, 114, 108, 100]
```

#### Encoding Chunks

```python
def _encode_chunk(self, text_chunk: str) -> list[int]:
    """Encode a single text chunk with BPE merges.

    Args:
        text_chunk: A single text chunk (from regex splitting) to encode.

    Returns:
        List of token IDs for the chunk.
    """
    # Convert chunk to byte IDs
    token_ids = list(text_chunk.encode("utf-8"))

    # Apply BPE merges in learned order
    while len(token_ids) >= 2:
        pair_freq = self._compute_pair_freq(token_ids)

        # Find the pair with the lowest merge index (earliest learned)
        pair_id = min(pair_freq, key=lambda p: self.merges.get(p, float("inf")))

        # If this pair was not learned during training, stop
        if pair_id not in self.merges:
            break

        # Get the new token ID for this merge and apply it
        idx = self.merges[pair_id]
        token_ids = self._merge_pair(token_ids, pair_id, idx)

    return token_ids
```

#### Decoding with Special Tokens

```python
def decode(self, token_ids: list[int]) -> str:
    """Decode token IDs back to text.

    Handles both normal tokens and special tokens.

    Args:
        token_ids: A list of integer token IDs.

    Returns:
        The decoded text string.
    """
    if not self._is_trained:
        raise TokenizerError("Tokenizer must be trained before decoding.")

    if not token_ids:
        return ""

    # Convert each token ID to bytes
    split_bytes: list[bytes] = []
    for token_id in token_ids:
        split_bytes.append(self._token_id_to_bytes(token_id))

    # Join all bytes and decode to string
    text = b"".join(split_bytes).decode("utf-8", errors="replace")
    return text

def _token_id_to_bytes(self, token_id: int) -> bytes:
    """Convert a single token ID to its byte representation.

    Handles both normal tokens and special tokens.

    Args:
        token_id: Token ID to convert.

    Returns:
        Byte representation of the token.
    """
    if token_id in self.vocab:
        return self.vocab[token_id]
    elif token_id in self.inverse_special_tokens:
        return self.inverse_special_tokens[token_id].encode("utf-8")
    else:
        raise TokenizerError(f"Unknown token ID: {token_id}")
```

#### Special Token Management

```python
def register_special_tokens(self, special_tokens: dict[str, int] | None = None) -> None:
    """Register special tokens (e.g., <|endoftext|>).

    Args:
        special_tokens: Dictionary of special tokens and their indices.

    Example:
        >>> tokenizer.register_special_tokens({"<|endoftext|>": 100257})
    """
    self.special_tokens = special_tokens or {"<|endoftext|>": 100257}
    self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
```

#### Usage Example

```python
from legollm.core.tokenization import RegexBPETokenizer
from legollm.core.utils import read_file

# Read training text
text = read_file("data/blog.txt")

# Train tokenizer with GPT-4 pattern
tokenizer = RegexBPETokenizer(pattern=RegexBPETokenizer.GPT4_SPLIT_PATTERN)
tokenizer.train(text, vocab_size=276, verbose=True)

# Register special tokens
tokenizer.register_special_tokens({"<|endoftext|>": 100257})

# Save tokenizer
tokenizer.save("data/regex_bpe_tokenizer.json")
tokenizer.save_readable("data/regex_bpe_tokenizer_readable.txt")

# Test encoding/decoding
test_text = "Hello<|endoftext|>world!"
encoded = tokenizer.encode(test_text)
print(f"Encoded: {encoded}")

decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
print(f"Match: {decoded == test_text}")
```

#### End-to-end comparison with tiktoken

```python
"""Main entry point for the application."""

import logging

import tiktoken

from legollm.logging import logger
from legollm.core.tokenization import RegexBPETokenizer
from legollm.core.utils import read_file

logger.setLevel(logging.DEBUG)

def main() -> None:
    """Main function."""
    text = read_file("data/blog.txt")
    tokenizer = RegexBPETokenizer()

    tokenizer.train(text, vocab_size=276, verbose=True, special_tokens={"<|endoftext|>": 100257})
    tokenizer.save("data/bpe_tokenizer.json")
    tokenizer.save_readable("data/bpe_tokenizer_readable.txt")
    tokenizer.load("data/bpe_tokenizer.json")
    text = "안녕하세요 오늘은 날씨가 좋네안 😀 (Hello, today is a good day 😀)"
    logger.debug(tokenizer.decode(tokenizer.encode(text)) == text)
    tiktoken_tokenizer = tiktoken_encode_decode(text)
    logger.debug(tiktoken_tokenizer)
    logger.debug(tokenizer.decode(tokenizer.encode(text)))

def tiktoken_encode_decode(text: str) -> None:
    """Encode and decode text using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text, allowed_special={"<|endoftext|>"}))

if __name__ == "__main__":
    main()
```

#### Comparison: Naive vs Regex BPE

| Feature               | Naive BPE        | Regex BPE            |
| --------------------- | ---------------- | -------------------- |
| Text Splitting        | No pre-splitting | Regex-based chunking |
| Cross-category Merges | Allowed          | Prevented by regex   |
| Special Tokens        | Not supported    | Fully supported      |
| Vocabulary Efficiency | Lower            | Higher               |
| Training Speed        | Faster           | Slower               |
| Production Use        | Educational only | Production-ready     |

## References

- Andrej Karpathy Video: https://www.youtube.com/watch?v=zduSFxRajkE
- Sebastian Raschka Notebook: https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
- HuggingFace BPE: https://huggingface.co/learn/llm-course/en/chapter6/5
- Regex Guide: https://www.regular-expressions.info/quickstart.html
