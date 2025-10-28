"""Custom exceptions for the project."""


class ProjectBaseError(Exception):
    """Base exception for the project."""

    pass


class TokenizerError(ProjectBaseError):
    """Exception for tokenizer errors."""

    pass


class DataLoaderError(ProjectBaseError):
    """Exception for dataloader errors."""

    pass
