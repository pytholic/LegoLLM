r"""Chat format templates - placeholder stub.

TODO: Implement chat formatting.

Reference: Sebastian's LLMs-from-scratch llama3.py and qwen3.py

Chat formats structure conversations with special tokens to indicate
roles (system, user, assistant) and turn boundaries.

Different models use different formats:
- Llama 3: <|start_header_id|>role<|end_header_id|>...<|eot_id|>
- Qwen 3: <|im_start|>role\n...<|im_end|>
- ChatML: <|im_start|>role\n...<|im_end|> (similar to Qwen)
"""


class ChatFormat:
    """Base class for chat format templates.

    TODO: Implement chat formatting.

    Subclasses should implement formatting for specific models.
    """

    def __init__(self, tokenizer: object, system_prompt: str | None = None) -> None:
        """Initialize ChatFormat."""
        raise NotImplementedError

    def format_messages(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a list of messages into a prompt string.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            add_generation_prompt: Whether to add assistant prompt for generation

        Returns:
            Formatted prompt string
        """
        raise NotImplementedError

    def encode(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Format and tokenize messages.

        Returns:
            Token IDs
        """
        raise NotImplementedError


class Llama3ChatFormat(ChatFormat):
    """Chat format for Llama 3 models.

    TODO: Implement Llama 3 chat format.

    Format:
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>

    See Sebastian's llama3.py ChatFormat for reference.
    """

    pass


class QwenChatFormat(ChatFormat):
    """Chat format for Qwen models.

    TODO: Implement Qwen chat format.

    Format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant

    See Sebastian's qwen3.py Qwen3Tokenizer for reference.
    """

    pass
