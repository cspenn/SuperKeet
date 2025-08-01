# start src/utils/logger.py
"""Logging utilities for SuperKeet."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config.config_loader import config


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emoji indicators to log messages."""

    EMOJI_MAP = {
        logging.DEBUG: "🐛",
        logging.INFO: "🟢",
        logging.WARNING: "🟡",
        logging.ERROR: "🛑",
        logging.CRITICAL: "🛑",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with emoji indicator.

        Args:
            record: Log record to format.

        Returns:
            Formatted log message with emoji.
        """
        emoji = self.EMOJI_MAP.get(record.levelno, "")
        record.emoji = emoji

        # Add emoji to format string
        original_format = self._style._fmt
        self._style._fmt = f"{emoji} {original_format}"
        result = super().format(record)
        self._style._fmt = original_format

        return result


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with console and file handlers.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    # Get configuration
    log_level = config.get("logging.level", "INFO")
    log_format = config.get(
        "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_dir = Path(config.get("logging.directory", "logs"))

    # Set level
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with emoji
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(EmojiFormatter(log_format))
    logger.addHandler(console_handler)

    # File handler - use daily log files
    log_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"superkeet-{date_str}.log"

    file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    return logger


# Create module logger
logger = setup_logger(__name__)

# end src/utils/logger.py
