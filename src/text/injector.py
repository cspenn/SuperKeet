# start src/text/injector.py
"""Text injection functionality for SuperKeet."""

import time

import pyperclip
from pynput.keyboard import Controller, Key

from src.config.config_loader import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextInjector:
    """Injects text into the active application."""

    def __init__(self) -> None:
        """Initialize the text injector."""
        self.method = config.get("text.method", "clipboard")
        self.delay_ms = config.get("text.delay_ms", 100)
        self.auto_paste = config.get("text.auto_paste", True)
        self.keyboard = Controller()

        logger.info(
            f"TextInjector initialized with method: {self.method}, auto_paste: {self.auto_paste}"
        )

    def inject(self, text: str) -> bool:
        """Inject text into the active application.

        Args:
            text: Text to inject.

        Returns:
            True if successful, False otherwise.
        """
        if not text:
            logger.warning("Empty text provided for injection")
            return False

        try:
            if self.method == "clipboard":
                return self._inject_via_clipboard(text)
            elif self.method == "accessibility":
                # Future enhancement: implement direct accessibility API injection
                logger.warning(
                    "Accessibility method not yet implemented, falling back to clipboard"
                )
                return self._inject_via_clipboard(text)
            else:
                logger.error(f"Unknown injection method: {self.method}")
                return False

        except Exception as e:
            logger.error(f"Text injection failed: {e}")
            return False

    def _inject_via_clipboard(self, text: str) -> bool:
        """Inject text using clipboard and paste command.

        Args:
            text: Text to inject.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Save current clipboard content
            original_clipboard = None
            try:
                original_clipboard = pyperclip.paste()
            except Exception:
                logger.debug("Could not save original clipboard content")

            # Copy text to clipboard
            pyperclip.copy(text)

            if self.auto_paste:
                # Small delay to ensure clipboard is updated
                time.sleep(self.delay_ms / 1000.0)

                # Simulate Cmd+V
                with self.keyboard.pressed(Key.cmd):
                    self.keyboard.press("v")
                    self.keyboard.release("v")

                # Another small delay
                time.sleep(self.delay_ms / 1000.0)

                logger.debug(f"Auto-pasted {len(text)} characters via clipboard")
            else:
                logger.debug(
                    f"Copied {len(text)} characters to clipboard (auto-paste disabled)"
                )

            # Optionally restore original clipboard
            # (commented out as it might interfere with paste operation)
            # if original_clipboard is not None:
            #     pyperclip.copy(original_clipboard)

            return True

        except Exception as e:
            logger.error(f"Clipboard injection failed: {e}")
            return False


# end src/text/injector.py
