"""Logger configuration for the project."""
import logging

from rich.logging import RichHandler

# Configure the logger
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("rich")
