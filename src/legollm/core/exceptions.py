"""Custom exceptions for the project."""


class ProjectBaseError(Exception):
    """Base exception for the project."""

    pass


class TokenizerError(ProjectBaseError):
    """Exception for tokenizer errors."""

    pass
