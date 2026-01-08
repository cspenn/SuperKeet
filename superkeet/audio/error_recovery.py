# start src/audio/error_recovery.py
"""Error recovery for audio recording.

This module handles error recovery strategies, fallback mechanisms,
and AUHAL recovery for the AudioRecorder class.
"""

import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import sounddevice as sd

from superkeet.audio.device_manager import (
    device_supports_stereo,
    is_rodecast_device,
    test_device_system_accessibility,
    try_alternative_input_devices,
)
from superkeet.audio.diagnostics import (
    log_detailed_error_info,
    suggest_audio_fixes,
    suggest_coreaudio_daemon_restart,
    suggest_rodecast_pro_fixes,
)
from superkeet.audio.permissions import (
    force_portaudio_reinit,
    try_portaudio_reinit,
    verify_coreaudio_daemon_health,
)
from superkeet.utils.logger import setup_logger

if TYPE_CHECKING:
    from superkeet.audio.recorder import AudioRecorder

logger = setup_logger(__name__)


def handle_start_error(recorder: "AudioRecorder", error: Exception) -> bool:
    """Handle errors during stream start.

    Args:
        recorder: The AudioRecorder instance.
        error: The exception that occurred.

    Returns:
        True if recovery succeeded and recording started.
    """
    logger.error(f"Stream start failed: {error}")

    # Log detailed error info
    log_detailed_error_info(
        error, recorder.device, recorder.sample_rate, recorder.channels
    )

    # Try enhanced recovery
    if attempt_enhanced_recovery(recorder, error):
        return True

    # Suggest fixes
    suggest_audio_fixes(error)
    return False


def attempt_enhanced_recovery(recorder: "AudioRecorder", error: Exception) -> bool:
    """Attempt enhanced error recovery strategies.

    Args:
        recorder: The AudioRecorder instance.
        error: The exception that occurred.

    Returns:
        True if recovery succeeded.
    """
    error_str = str(error)
    logger.info("Attempting enhanced recovery...")

    # Strategy 1: Handle specific PortAudio errors
    if "-9986" in error_str:
        logger.info("Detected PortAudio -9986 error, attempting specialized recovery")
        if handle_portaudio_9986_error(recorder, recorder.device):
            return True

    # Strategy 2: RODECaster Pro specific recovery
    if is_rodecast_device(recorder.device):
        logger.info("Detected RODECaster Pro, attempting device-specific recovery")
        if handle_rodecast_pro_errors(recorder, error):
            return True

    # Strategy 3: Try PortAudio reinitialization
    if try_portaudio_reinit(recorder):
        logger.info("PortAudio reinitialized, retrying stream creation")
        if try_restart_recording(recorder):
            return True

    # Strategy 4: Try fallback device
    if try_fallback_device(recorder):
        return True

    # Strategy 5: Try fallback sample rates
    if try_fallback_sample_rates(recorder):
        return True

    # Strategy 6: Try alternative configurations
    if try_alternative_configs(recorder):
        return True

    logger.error("All recovery strategies failed")
    return False


def cleanup_failed_start(recorder: "AudioRecorder") -> None:
    """Clean up after a failed start attempt.

    Args:
        recorder: The AudioRecorder instance.
    """
    recorder.recording = False

    if recorder.stream is not None:
        try:
            recorder.stream.close()
        except Exception as e:
            logger.debug(f"Error closing stream during cleanup: {e}")
        recorder.stream = None

    # Clear any partial data
    recorder.audio_data.clear()
    recorder._total_audio_size = 0


def try_fallback_device(recorder: "AudioRecorder") -> bool:
    """Try using a fallback device.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if fallback device works.
    """
    logger.info("Trying fallback device...")

    original_device = recorder.device

    # Try the system default device
    if original_device is not None:
        recorder.device = None
        if try_restart_recording(recorder):
            logger.info("Fallback to default device succeeded")
            return True

    # Try other available devices
    if try_alternative_input_devices(recorder):
        if try_restart_recording(recorder):
            logger.info("Fallback to alternative device succeeded")
            return True

    # Restore original device
    recorder.device = original_device
    return False


def try_fallback_sample_rates(recorder: "AudioRecorder") -> bool:
    """Try fallback sample rates.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if a fallback rate works.
    """
    logger.info("Trying fallback sample rates...")

    original_rate = recorder.sample_rate
    fallback_rates = [16000, 44100, 48000, 22050, 32000]

    for rate in fallback_rates:
        if rate == original_rate:
            continue

        logger.debug(f"Trying sample rate: {rate}Hz")
        recorder.sample_rate = rate

        if try_restart_recording(recorder):
            logger.info(f"Fallback to {rate}Hz succeeded")
            return True

    # Restore original rate
    recorder.sample_rate = original_rate
    return False


def try_alternative_configs(recorder: "AudioRecorder") -> bool:
    """Try alternative configurations.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if an alternative config works.
    """
    logger.info("Trying alternative configurations...")

    original_channels = recorder.channels
    original_chunk_size = recorder.chunk_size

    # Try mono if currently stereo
    if recorder.channels > 1:
        logger.debug("Trying mono configuration")
        recorder.channels = 1
        if try_restart_recording(recorder):
            logger.info("Fallback to mono succeeded")
            return True

    # Try different chunk sizes
    chunk_sizes = [512, 2048, 4096]
    for chunk in chunk_sizes:
        if chunk == original_chunk_size:
            continue

        logger.debug(f"Trying chunk size: {chunk}")
        recorder.chunk_size = chunk
        if try_restart_recording(recorder):
            logger.info(f"Fallback to chunk size {chunk} succeeded")
            return True

    # Restore original settings
    recorder.channels = original_channels
    recorder.chunk_size = original_chunk_size
    return False


def handle_portaudio_9986_error(
    recorder: "AudioRecorder", device_index: Optional[int]
) -> bool:
    """Handle specific PortAudio -9986 Internal Error scenarios.

    Args:
        recorder: The AudioRecorder instance.
        device_index: The device that failed, or None for default.

    Returns:
        True if recovery succeeded, False otherwise.
    """
    logger.error("HANDLING PortAudio -9986 Internal Error")
    logger.error("This typically indicates Core Audio/PortAudio interface issues")

    # Step 1: Check Core Audio daemon health
    if not verify_coreaudio_daemon_health():
        logger.error("Core Audio daemon appears unhealthy")
        suggest_coreaudio_daemon_restart()
        return False

    # Step 2: Test device accessibility at system level
    if device_index is not None:
        if not test_device_system_accessibility(device_index):
            logger.error(f"Device {device_index} not accessible at system level")
            # Try alternative devices
            return try_alternative_input_devices(recorder)

    # Step 3: AUHAL-specific recovery
    return attempt_auhal_recovery(recorder, device_index)


def attempt_auhal_recovery(
    recorder: "AudioRecorder", device_index: Optional[int]
) -> bool:
    """Attempt AUHAL-specific recovery.

    Args:
        recorder: The AudioRecorder instance.
        device_index: The device index.

    Returns:
        True if recovery succeeded.
    """
    logger.info("Attempting AUHAL recovery...")

    # Force PortAudio reinit
    if not force_portaudio_reinit():
        logger.error("AUHAL recovery: PortAudio reinit failed")
        return False

    # Brief pause for system to settle
    time.sleep(0.5)

    # Try with conservative settings
    conservative_configs = [
        {"samplerate": 48000, "channels": 1, "blocksize": 2048},
        {"samplerate": 44100, "channels": 1, "blocksize": 1024},
        {"samplerate": 16000, "channels": 1, "blocksize": 512},
    ]

    for i, cfg in enumerate(conservative_configs):
        logger.debug(f"AUHAL recovery: trying config {i}: {cfg}")
        try:
            # Test stream creation
            test_stream = sd.InputStream(
                device=device_index,
                samplerate=cfg["samplerate"],
                channels=cfg["channels"],
                blocksize=cfg["blocksize"],
                dtype=np.float32,
            )
            test_stream.close()

            # If test passed, update recorder config
            recorder.sample_rate = cfg["samplerate"]
            recorder.channels = cfg["channels"]
            recorder.chunk_size = cfg["blocksize"]

            logger.info(f"AUHAL recovery successful with config {i}")
            return True

        except Exception as e:
            logger.debug(f"AUHAL recovery config {i} failed: {e}")

    logger.error("AUHAL recovery: all configs failed")
    return False


def handle_rodecast_pro_errors(recorder: "AudioRecorder", error: Exception) -> bool:
    """Handle RODECaster Pro specific errors.

    Args:
        recorder: The AudioRecorder instance.
        error: The exception that occurred.

    Returns:
        True if recovery succeeded.
    """
    if not is_rodecast_device(recorder.device):
        return False

    logger.info("Handling RODECaster Pro specific error...")

    # RODECaster Pro prefers 48000Hz
    original_rate = recorder.sample_rate
    recorder.sample_rate = 48000

    if try_restart_recording(recorder):
        logger.info("RODECaster Pro: 48kHz config succeeded")
        return True

    # Try stereo if supported
    if device_supports_stereo(recorder.device):
        recorder.channels = 2
        if try_restart_recording(recorder):
            logger.info("RODECaster Pro: stereo config succeeded")
            return True

    # Restore and suggest manual fixes
    recorder.sample_rate = original_rate
    suggest_rodecast_pro_fixes()
    return False


def test_enhanced_error_recovery(recorder: "AudioRecorder") -> dict[str, Any]:
    """Test the enhanced error recovery system.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        Dictionary with test results in expected format.
    """
    logger.info("Testing enhanced error recovery system...")

    results: dict[str, Any] = {}

    # Test permission validation
    try:
        from superkeet.audio.permissions import validate_microphone_permissions

        if validate_microphone_permissions(recorder):
            results["permission_validation"] = {"status": "✅ PASSED"}
        else:
            results["permission_validation"] = {"status": "⚠️ FAILED"}
    except Exception as e:
        results["permission_validation"] = {"status": "⚠️ FAILED", "error": str(e)}

    # Test basic access
    try:
        from superkeet.audio.permissions import test_basic_microphone_access

        if test_basic_microphone_access(recorder):
            results["basic_access"] = {"status": "✅ PASSED"}
        else:
            results["basic_access"] = {"status": "⚠️ FAILED"}
    except Exception as e:
        results["basic_access"] = {"status": "⚠️ FAILED", "error": str(e)}

    # Test alternative backends (PortAudio reinit)
    try:
        if try_portaudio_reinit(recorder):
            results["alternative_backends"] = {"status": "✅ TESTED"}
        else:
            results["alternative_backends"] = {"status": "⚠️ FAILED"}
    except Exception as e:
        results["alternative_backends"] = {"status": "⚠️ FAILED", "error": str(e)}

    # Test device enumeration
    try:
        devices = recorder.get_devices()
        if len(devices) > 0:
            results["device_enumeration"] = {
                "status": "✅ PASSED",
                "device_count": len(devices),
            }
        else:
            results["device_enumeration"] = {"status": "⚠️ FAILED"}
    except Exception as e:
        results["device_enumeration"] = {"status": "⚠️ FAILED", "error": str(e)}

    # Test sample rate compatibility
    try:
        test_rates = [16000, 44100, 48000]
        compatible_rates = []
        for rate in test_rates:
            try:
                test_stream = sd.InputStream(
                    channels=1,
                    samplerate=rate,
                    blocksize=1024,
                    dtype=np.float32,
                    device=recorder.device,
                )
                test_stream.close()
                compatible_rates.append(rate)
            except Exception:
                pass
        if compatible_rates:
            results["sample_rate_compatibility"] = {
                "status": "✅ PASSED",
                "compatible_rates": compatible_rates,
            }
        else:
            results["sample_rate_compatibility"] = {"status": "⚠️ FAILED"}
    except Exception as e:
        results["sample_rate_compatibility"] = {"status": "⚠️ FAILED", "error": str(e)}

    # Test diagnostics
    try:
        from superkeet.audio.diagnostics import debug_audio_setup

        debug_audio_setup(recorder)
        results["diagnostics"] = {"status": "✅ PASSED"}
    except Exception as e:
        results["diagnostics"] = {"status": "⚠️ FAILED", "error": str(e)}

    logger.info(f"Recovery test results: {list(results.keys())}")
    return results


def try_restart_recording(recorder: "AudioRecorder") -> bool:
    """Try to restart recording with current settings.

    This is a helper that attempts to create and start a stream
    with the recorder's current configuration.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if recording started successfully.
    """
    try:
        # Close any existing stream
        if recorder.stream is not None:
            try:
                recorder.stream.close()
            except Exception:
                pass
            recorder.stream = None

        # Try to create new stream
        recorder.stream = sd.InputStream(
            channels=recorder.channels,
            samplerate=recorder.sample_rate,
            blocksize=recorder.chunk_size,
            dtype=np.float32,
            device=recorder.device,
            callback=recorder._audio_callback,
        )
        recorder.stream.start()
        recorder.recording = True
        recorder.recording_start_time = time.time()

        logger.debug("Recording restarted successfully")
        return True

    except Exception as e:
        logger.debug(f"Restart recording failed: {e}")
        cleanup_failed_start(recorder)
        return False


# end src/audio/error_recovery.py
