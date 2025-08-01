# start src/hotkey/listener.py
"""Global hotkey listener for SuperKeet."""

import threading
from typing import Optional, Set

from pynput import keyboard
from PySide6.QtCore import QObject, Signal

from src.config.config_loader import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HotkeySignals(QObject):
    """Qt signals for hotkey events."""

    hotkey_pressed = Signal()
    hotkey_released = Signal()


class HotkeyListener:
    """Listens for global hotkey combinations."""

    def __init__(self) -> None:
        """Initialize the hotkey listener."""
        self.signals = HotkeySignals()
        self.hotkey_combo = config.get("hotkey.combination", ["cmd", "shift", "space"])

        # Convert string keys to pynput keys
        self.required_keys: Set[keyboard.Key | keyboard.KeyCode] = set()
        for key in self.hotkey_combo:
            if key == "cmd":
                self.required_keys.add(keyboard.Key.cmd)
            elif key == "ctrl":
                self.required_keys.add(keyboard.Key.ctrl)
            elif key == "alt":
                self.required_keys.add(keyboard.Key.alt)
            elif key == "shift":
                self.required_keys.add(keyboard.Key.shift)
            elif key == "space":
                self.required_keys.add(keyboard.Key.space)
            else:
                # Single character key
                self.required_keys.add(keyboard.KeyCode.from_char(key))

        self.pressed_keys: Set[keyboard.Key | keyboard.KeyCode] = set()
        self.hotkey_active = False
        self.listener: Optional[keyboard.Listener] = None
        self.thread: Optional[threading.Thread] = None

        logger.info(f"HotkeyListener initialized with combo: {self.hotkey_combo}")

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Handle key press event.

        Args:
            key: The pressed key.
        """
        self.pressed_keys.add(key)

        # Check if all required keys are pressed
        if self.required_keys.issubset(self.pressed_keys) and not self.hotkey_active:
            self.hotkey_active = True
            logger.debug("Hotkey combination activated")
            self.signals.hotkey_pressed.emit()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Handle key release event.

        Args:
            key: The released key.
        """
        self.pressed_keys.discard(key)

        # Check if hotkey is no longer active
        if self.hotkey_active and not self.required_keys.issubset(self.pressed_keys):
            self.hotkey_active = False
            logger.debug("Hotkey combination deactivated")
            self.signals.hotkey_released.emit()

    def start(self) -> None:
        """Start listening for hotkeys in a background thread."""
        if self.listener is not None:
            logger.warning("Listener already running")
            return

        def run_listener():
            """Run the keyboard listener."""
            with keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            ) as listener:
                self.listener = listener
                logger.info("Hotkey listener started")
                listener.join()

        self.thread = threading.Thread(target=run_listener, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
            logger.info("Hotkey listener stopped")

        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None


# end src/hotkey/listener.py
