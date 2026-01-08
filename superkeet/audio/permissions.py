# start src/audio/permissions.py
"""Permission validation for audio recording.

This module handles microphone permission validation and PortAudio
reinitialization for the AudioRecorder class.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from superkeet.utils.logger import setup_logger

if TYPE_CHECKING:
    from superkeet.audio.recorder import AudioRecorder

logger = setup_logger(__name__)


def validate_microphone_permissions(recorder: "AudioRecorder") -> bool:
    """Validate microphone permissions with macOS-specific handling.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if permissions are granted and accessible, False otherwise.
    """
    try:
        import platform
        import subprocess

        # Only do macOS-specific checks on macOS
        if platform.system() != "Darwin":
            logger.debug("Non-macOS platform, skipping specific permission checks")
            # Do a basic test by trying to create a test stream
            return test_basic_microphone_access(recorder)

        logger.info("Checking microphone permissions on macOS...")

        # Method 1: Try to access microphone directly
        if not test_basic_microphone_access(recorder):
            logger.warning("Basic microphone access test failed")
            return False

        # Method 2: Check system permission status using tccutil (if available)
        try:
            # Use full path for security (Bandit B607)
            tccutil_path = "/usr/bin/tccutil"
            if Path(tccutil_path).exists():
                result = subprocess.run(
                    [tccutil_path, "list", "Microphone"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            else:
                # Fallback if standard path doesn't exist
                result = subprocess.run(
                    ["tccutil", "list", "Microphone"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

            if result.returncode == 0:
                logger.debug("System microphone permissions checked via tccutil")
            else:
                logger.debug("tccutil not available or failed, using basic test")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("tccutil command not available, using basic test")

        # Method 3: Test with device-appropriate configurations
        # Get device info to determine native channel count
        if recorder.device is not None:
            try:
                device_info = sd.query_devices(recorder.device)
                test_channels = min(2, int(device_info["max_input_channels"]))
                logger.debug(
                    f"Device {recorder.device} supports "
                    f"{device_info['max_input_channels']} channels, "
                    f"testing with {test_channels}"
                )
            except Exception:
                test_channels = 1  # Fallback to mono
        else:
            test_channels = 1  # Default device usually supports mono

        test_configs = [
            {"samplerate": 16000, "channels": test_channels, "blocksize": 1024},
            {"samplerate": 44100, "channels": test_channels, "blocksize": 512},
            {"samplerate": 48000, "channels": test_channels, "blocksize": 2048},
        ]

        for config in test_configs:
            try:
                test_stream = sd.InputStream(
                    channels=config["channels"],
                    samplerate=config["samplerate"],
                    blocksize=config["blocksize"],
                    dtype=np.float32,
                    device=recorder.device,
                )
                test_stream.close()
                logger.info(f"Microphone permission validated with config: {config}")
                return True
            except Exception as e:
                logger.debug(f"Permission test failed with config {config}: {e}")
                continue

        logger.error("All microphone permission tests failed")
        return False

    except Exception as e:
        logger.error(f"Microphone permission validation failed: {e}")
        return False


def test_basic_microphone_access(recorder: "AudioRecorder") -> bool:
    """Test basic microphone access with a quick stream creation.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if basic access works, False otherwise.
    """
    try:
        # Try to create a minimal test stream
        test_stream = sd.InputStream(
            channels=1,
            samplerate=16000,
            blocksize=512,
            dtype=np.float32,
            device=recorder.device,
        )
        test_stream.close()
        logger.debug("Basic microphone access test passed")
        return True
    except Exception as e:
        logger.debug(f"Basic microphone access test failed: {e}")
        return False


def try_portaudio_reinit(recorder: "AudioRecorder") -> bool:
    """Attempt to reinitialize PortAudio.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if reinitialization succeeded, False otherwise.
    """
    logger.info("Attempting PortAudio reinitialization...")

    try:
        # Close any existing stream
        if recorder.stream is not None:
            try:
                recorder.stream.close()
            except Exception as e:
                logger.debug(f"Error closing existing stream: {e}")
            recorder.stream = None

        # Terminate and reinitialize PortAudio
        import sounddevice as sd

        sd._terminate()
        sd._initialize()

        logger.info("PortAudio reinitialized successfully")
        return True

    except Exception as e:
        logger.error(f"PortAudio reinitialization failed: {e}")
        return False


def force_portaudio_reinit() -> bool:
    """Force a complete PortAudio reinitialization.

    Returns:
        True if reinitialization succeeded, False otherwise.
    """
    import time

    logger.info("Forcing complete PortAudio reinitialization...")

    try:
        # Terminate PortAudio
        sd._terminate()

        # Brief pause to allow system audio resources to reset
        time.sleep(0.5)

        # Reinitialize
        sd._initialize()

        # Verify by querying devices
        devices = sd.query_devices()
        logger.info(f"PortAudio reinitialized, found {len(devices)} devices")
        return True

    except Exception as e:
        logger.error(f"Force PortAudio reinitialization failed: {e}")
        return False


def verify_coreaudio_daemon_health() -> bool:
    """Verify Core Audio daemon health on macOS.

    Returns:
        True if daemon appears healthy, False otherwise.
    """
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return True  # Non-macOS, skip this check

    try:
        # Check if coreaudiod is running
        result = subprocess.run(
            ["/bin/ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if "coreaudiod" in result.stdout:
            logger.debug("Core Audio daemon (coreaudiod) is running")
            return True
        else:
            logger.warning("Core Audio daemon (coreaudiod) not found in process list")
            return False

    except Exception as e:
        logger.debug(f"Could not check Core Audio daemon status: {e}")
        return True  # Assume healthy if we can't check


# end src/audio/permissions.py
