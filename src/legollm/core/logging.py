"""Logger configuration for the project."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Configure the logger
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("rich")


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a rich Progress bar with standard configuration.

    Args:
        description: Description to display in the progress bar.

    Returns:
        A configured Progress instance.

    Example:
        >>> progress = create_progress_bar("Training")
        >>> task = progress.add_task(f"[green]{description}", total=100)
        >>> for i in range(100):
        >>> # do work
        >>>     progress.update(task, advance=1)
    """
    # I want to change bar color from pink to light_salmon3
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(finished_style="green"),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


@contextmanager
def progress_bar(
    description: str = "Processing", total: int = 100
) -> Generator[tuple[Progress, int], Any, None]:
    """Context manager for progress bar - simpler API for common use cases.

    Args:
        description: Description to display in the progress bar.
        total: Total number of iterations.

    Yields:
        A task ID to use with progress.update(task_id, advance=1).

    Example:
        >>> with progress_bar("Training BPE tokenizer", total=256) as (progress, task):
        >>>     for i in range(256):
        >>> # do work
        >>>         progress.update(task, advance=1)
    """
    progress = create_progress_bar(description)
    with progress:
        task = progress.add_task(f"[green]{description}", total=total)
        yield progress, task
