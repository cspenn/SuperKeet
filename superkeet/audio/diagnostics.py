# start src/audio/diagnostics.py
"""Diagnostics for audio recording.

This module handles logging, system audio diagnostics, and troubleshooting
suggestions for the AudioRecorder class.
"""

import platform
import subprocess
from typing import TYPE_CHECKING, Optional

import sounddevice as sd

from superkeet.utils.logger import setup_logger

if TYPE_CHECKING:
    from superkeet.audio.recorder import AudioRecorder

logger = setup_logger(__name__)


def log_detailed_error_info(
    error: Exception, device: Optional[int], sample_rate: int, channels: int
) -> None:
    """Log detailed error information for debugging.

    Args:
        error: The exception that occurred.
        device: Device index or None for default.
        sample_rate: The sample rate being used.
        channels: Number of channels being used.
    """
    logger.error("=" * 60)
    logger.error("AUDIO ERROR DETAILS")
    logger.error("=" * 60)
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {error}")
    logger.error(f"Device: {device}")
    logger.error(f"Sample rate: {sample_rate}Hz")
    logger.error(f"Channels: {channels}")
    logger.error(f"Platform: {platform.system()} {platform.release()}")

    # Log device info if available
    try:
        if device is not None:
            device_info = sd.query_devices(device)
            logger.error(f"Device name: {device_info['name']}")
            max_ch = device_info["max_input_channels"]
            logger.error(f"Device max input channels: {max_ch}")
            logger.error(f"Device default rate: {device_info['default_samplerate']}")
    except Exception as e:
        logger.error(f"Could not get device info: {e}")

    logger.error("=" * 60)


def suggest_audio_fixes(error: Exception) -> None:
    """Suggest fixes based on the error type.

    Args:
        error: The exception that occurred.
    """
    error_msg = str(error).lower()

    logger.info("=" * 60)
    logger.info("SUGGESTED FIXES")
    logger.info("=" * 60)

    if "-9986" in str(error) or "internal" in error_msg:
        logger.info("PortAudio internal error detected:")
        logger.info("  1. Try restarting your audio interface")
        logger.info("  2. Check System Preferences > Security > Privacy > Microphone")
        logger.info("  3. Restart the Core Audio daemon:")
        logger.info("     sudo killall coreaudiod")
        logger.info("  4. Disconnect and reconnect your audio device")

    elif "permission" in error_msg or "-9995" in str(error):
        logger.info("Permission error detected:")
        logger.info("  1. Grant microphone access in System Preferences")
        logger.info("  2. Add this application to allowed apps")
        logger.info("  3. Restart the application after granting permissions")

    elif "sample rate" in error_msg or "samplerate" in error_msg:
        logger.info("Sample rate error detected:")
        logger.info("  1. Try a different sample rate (16000, 44100, or 48000)")
        logger.info("  2. Check your audio device's supported sample rates")
        logger.info("  3. Update your audio drivers")

    elif "device" in error_msg or "not found" in error_msg:
        logger.info("Device error detected:")
        logger.info("  1. Check if your microphone is connected")
        logger.info("  2. Select a different input device")
        logger.info("  3. Refresh the device list in settings")

    else:
        logger.info("General troubleshooting steps:")
        logger.info("  1. Restart the application")
        logger.info("  2. Check your audio device connection")
        logger.info("  3. Verify microphone permissions")
        logger.info("  4. Try a different audio device")

    logger.info("=" * 60)


def suggest_coreaudio_daemon_restart() -> None:
    """Suggest how to restart the Core Audio daemon."""
    logger.info("=" * 60)
    logger.info("CORE AUDIO DAEMON RESTART REQUIRED")
    logger.info("=" * 60)
    logger.info("The Core Audio daemon may need to be restarted.")
    logger.info("Run the following command in Terminal:")
    logger.info("  sudo killall coreaudiod")
    logger.info("")
    logger.info("This will temporarily disconnect all audio and")
    logger.info("the daemon will automatically restart.")
    logger.info("=" * 60)


def suggest_rodecast_pro_fixes() -> None:
    """Suggest fixes specific to RODECaster Pro devices."""
    logger.info("=" * 60)
    logger.info("RODECASTER PRO TROUBLESHOOTING")
    logger.info("=" * 60)
    logger.info("1. Ensure RODECaster is set as the system input device")
    logger.info("2. Check RODE Central app for firmware updates")
    logger.info("3. Try a different USB port (preferably USB 3.0)")
    logger.info("4. Disconnect other USB audio devices")
    logger.info("5. Set sample rate to 48000Hz in settings")
    logger.info("6. Restart the RODECaster Pro device")
    logger.info("=" * 60)


def log_system_audio_diagnostics() -> None:
    """Log comprehensive system audio diagnostics."""
    logger.info("=" * 60)
    logger.info("SYSTEM AUDIO DIAGNOSTICS")
    logger.info("=" * 60)

    system = platform.system()
    logger.info(f"Operating System: {system} {platform.release()}")
    logger.info(f"Python version: {platform.python_version()}")

    if system == "Darwin":
        log_macos_audio_diagnostics()
    elif system == "Linux":
        log_linux_audio_diagnostics()
    elif system == "Windows":
        log_windows_audio_diagnostics()

    # Log PortAudio info
    try:
        logger.info("PortAudio Information:")
        hostapis = sd.query_hostapis()
        for i, api in enumerate(hostapis):
            logger.info(f"  Host API {i}: {api['name']}")

        devices = sd.query_devices()
        input_count = sum(1 for d in devices if d["max_input_channels"] > 0)
        logger.info(f"  Input devices found: {input_count}")

    except Exception as e:
        logger.error(f"Could not query PortAudio info: {e}")

    logger.info("=" * 60)


def log_macos_audio_diagnostics() -> None:
    """Log macOS-specific audio diagnostics."""
    logger.info("macOS Audio Diagnostics:")

    # Check coreaudiod status
    log_coreaudio_processes()

    # Log system audio devices
    log_system_audio_devices()


def log_coreaudio_processes() -> None:
    """Log Core Audio related processes."""
    try:
        result = subprocess.run(
            ["/bin/ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        audio_processes = []
        for line in result.stdout.split("\n"):
            if any(
                proc in line.lower()
                for proc in ["coreaudio", "audiod", "soundflower", "blackhole"]
            ):
                audio_processes.append(line)

        if audio_processes:
            logger.info("  Audio-related processes:")
            for proc in audio_processes[:5]:  # Limit output
                logger.info(f"    {proc[:80]}...")
        else:
            logger.info("  No audio-related processes found")

    except Exception as e:
        logger.debug(f"Could not check processes: {e}")


def log_system_audio_devices() -> None:
    """Log system-level audio device information."""
    try:
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            lines = result.stdout.split("\n")[:20]  # Limit output
            logger.info("  System audio devices:")
            for line in lines:
                if line.strip():
                    logger.info(f"    {line}")

    except Exception as e:
        logger.debug(f"Could not get system audio devices: {e}")


def log_linux_audio_diagnostics() -> None:
    """Log Linux-specific audio diagnostics."""
    logger.info("Linux Audio Diagnostics:")

    try:
        # Check PulseAudio/PipeWire
        result = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            logger.info("  PulseAudio/PipeWire info:")
            for line in result.stdout.split("\n")[:10]:
                if line.strip():
                    logger.info(f"    {line}")

    except FileNotFoundError:
        logger.info("  pactl not found (PulseAudio may not be installed)")
    except Exception as e:
        logger.debug(f"Could not get Linux audio info: {e}")


def log_windows_audio_diagnostics() -> None:
    """Log Windows-specific audio diagnostics."""
    logger.info("Windows Audio Diagnostics:")
    logger.info("  (Basic diagnostics only on Windows)")

    try:
        # Check Windows Audio Service
        logger.info("  Checking Windows Audio Service...")

    except Exception as e:
        logger.debug(f"Could not check Windows audio: {e}")


def debug_audio_setup(recorder: "AudioRecorder") -> None:
    """Debug and log the current audio setup.

    Args:
        recorder: The AudioRecorder instance.
    """
    logger.info("=" * 60)
    logger.info("AUDIO SETUP DEBUG")
    logger.info("=" * 60)
    logger.info(f"Configured sample rate: {recorder.configured_sample_rate}Hz")
    logger.info(f"Active sample rate: {recorder.sample_rate}Hz")
    logger.info(f"Channels: {recorder.channels}")
    logger.info(f"Chunk size: {recorder.chunk_size}")
    logger.info(f"Device: {recorder.device}")
    logger.info(f"Gain: {recorder.gain}")
    logger.info(f"Max duration: {recorder.max_recording_duration}s")
    logger.info(f"Buffer limit: {recorder.buffer_size_limit}MB")

    # Log device info
    try:
        if recorder.device is not None:
            device_info = sd.query_devices(recorder.device)
            logger.info(f"Device name: {device_info['name']}")
            logger.info(f"Device max channels: {device_info['max_input_channels']}")
            logger.info(f"Device default rate: {device_info['default_samplerate']}")
        else:
            default_device = sd.default.device[0]
            if default_device is not None:
                device_info = sd.query_devices(default_device)
                logger.info(f"Default device: {device_info['name']}")

    except Exception as e:
        logger.warning(f"Could not get device info: {e}")

    # Log available input devices
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        logger.info(f"Available input devices: {len(input_devices)}")
        for i, d in enumerate(input_devices[:5]):  # Show up to 5
            logger.info(f"  {i}: {d['name']} ({d['max_input_channels']}ch)")

    except Exception as e:
        logger.warning(f"Could not list devices: {e}")

    logger.info("=" * 60)


# end src/audio/diagnostics.py
