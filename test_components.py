# start test_components.py
"""Test individual components of SuperKeet."""

import sys
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_imports():
    """Test all required imports."""
    logger.info("Testing imports...")

    try:
        import numpy

        logger.info("✓ numpy imported successfully")
    except ImportError as e:
        logger.error(f"✗ numpy import failed: {e}")

    try:
        import sounddevice

        logger.info("✓ sounddevice imported successfully")
    except ImportError as e:
        logger.error(f"✗ sounddevice import failed: {e}")

    try:
        import pyperclip

        logger.info("✓ pyperclip imported successfully")
    except ImportError as e:
        logger.error(f"✗ pyperclip import failed: {e}")

    try:
        from pynput import keyboard

        logger.info("✓ pynput imported successfully")
    except ImportError as e:
        logger.error(f"✗ pynput import failed: {e}")

    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        logger.info("✓ PySide6 imported successfully")
    except ImportError as e:
        logger.error(f"✗ PySide6 import failed: {e}")

    try:
        from parakeet_mlx import from_pretrained

        logger.info("✓ parakeet_mlx imported successfully")
    except ImportError as e:
        logger.error(f"✗ parakeet_mlx import failed: {e}")


def test_config():
    """Test configuration loading."""
    logger.info("\nTesting configuration...")

    try:
        from src.config.config_loader import config

        logger.info("✓ Config loaded successfully")
        logger.info(f"  App name: {config.get('app.name')}")
        logger.info(f"  ASR model: {config.get('asr.model_id')}")
        logger.info(f"  Hotkey: {config.get('hotkey.combination')}")
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")


def test_audio_devices():
    """Test audio device detection."""
    logger.info("\nTesting audio devices...")

    try:
        from src.audio.recorder import AudioRecorder

        recorder = AudioRecorder()
        devices = recorder.get_devices()
        logger.info(f"✓ Found {len(devices)} audio input devices:")
        for device in devices[:3]:  # Show first 3 devices
            logger.info(f"  - {device['name']} (index: {device['index']})")
    except Exception as e:
        logger.error(f"✗ Audio device detection failed: {e}")


def main():
    """Run all component tests."""
    logger.info("=== SuperKeet Component Test ===\n")

    test_imports()
    test_config()
    test_audio_devices()

    logger.info("\n=== Test Complete ===")
    logger.info("If all tests passed, you can run: python -m src.main")


if __name__ == "__main__":
    main()

# end test_components.py
