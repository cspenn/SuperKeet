# start src/audio/stream_manager.py
"""Stream management for audio recording.

This module handles stream creation, sample rate optimization, and
PortAudio capability detection for the AudioRecorder class.
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import sounddevice as sd

from superkeet.utils.logger import setup_logger

if TYPE_CHECKING:
    from superkeet.audio.recorder import AudioRecorder

logger = setup_logger(__name__)


def select_optimal_sample_rate(
    recorder: "AudioRecorder", device: Optional[int] = None
) -> int:
    """Select the optimal sample rate for the device.

    Args:
        recorder: The AudioRecorder instance.
        device: Device index or None for default.

    Returns:
        The optimal sample rate.
    """
    target_device = device if device is not None else recorder.device

    # Priority rates to test (in order of preference)
    priority_rates = [16000, 44100, 48000, 22050, 32000, 96000]

    # Get device's default sample rate
    try:
        if target_device is not None:
            device_info = sd.query_devices(target_device)
        else:
            device_info = sd.query_devices(sd.default.device[0])

        default_rate = int(device_info.get("default_samplerate", 44100))
        if default_rate not in priority_rates:
            priority_rates.insert(0, default_rate)

    except Exception as e:
        logger.debug(f"Could not get device default rate: {e}")

    # Test each rate
    for rate in priority_rates:
        if test_device_sample_rate(recorder, rate, target_device):
            logger.debug(f"Selected optimal sample rate: {rate}Hz")
            return rate

    # Fallback to configured rate
    logger.warning("No optimal rate found, using configured rate")
    return recorder.configured_sample_rate


def test_device_sample_rate(
    recorder: "AudioRecorder", sample_rate: int, device: Optional[int] = None
) -> bool:
    """Test if a sample rate works with the device.

    Args:
        recorder: The AudioRecorder instance.
        sample_rate: The sample rate to test.
        device: Device index or None for default.

    Returns:
        True if the sample rate works.
    """
    target_device = device if device is not None else recorder.device

    try:
        test_stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=1024,
            dtype=np.float32,
            device=target_device,
        )
        test_stream.close()
        return True

    except Exception as e:
        logger.debug(f"Sample rate {sample_rate}Hz test failed: {e}")
        return False


def test_device_sample_rate_for_device(
    device: int, sample_rate: int, channels: int = 1
) -> bool:
    """Test if a sample rate works with a specific device.

    Args:
        device: Device index.
        sample_rate: The sample rate to test.
        channels: Number of channels to test.

    Returns:
        True if the sample rate works.
    """
    try:
        test_stream = sd.InputStream(
            channels=channels,
            samplerate=sample_rate,
            blocksize=1024,
            dtype=np.float32,
            device=device,
        )
        test_stream.close()
        return True

    except Exception:
        return False


def detect_portaudio_capabilities(recorder: "AudioRecorder") -> dict[str, Any]:
    """Detect PortAudio capabilities for the current device.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        Dictionary of detected capabilities.
    """
    capabilities: dict[str, Any] = {
        "max_channels": 0,
        "supported_rates": [],
        "default_rate": 0,
        "latencies": {},
        "host_api": None,
        "never_drop_input": False,  # We intentionally don't use this flag
    }

    try:
        device = recorder.device
        if device is None:
            device = sd.default.device[0]

        device_info = sd.query_devices(device)
        capabilities["max_channels"] = device_info["max_input_channels"]
        capabilities["default_rate"] = int(device_info.get("default_samplerate", 44100))

        # Get latency info
        capabilities["latencies"] = {
            "low": device_info.get("default_low_input_latency", 0),
            "high": device_info.get("default_high_input_latency", 0),
        }

        # Get host API info
        host_api_index = device_info.get("hostapi", 0)
        try:
            host_api = sd.query_hostapis(host_api_index)
            capabilities["host_api"] = host_api.get("name", "Unknown")
        except Exception:
            pass

        # Test supported sample rates
        test_rates = [8000, 16000, 22050, 32000, 44100, 48000, 96000]
        for rate in test_rates:
            if test_device_sample_rate(recorder, rate, device):
                capabilities["supported_rates"].append(rate)

    except Exception as e:
        logger.debug(f"Capability detection failed: {e}")

    return capabilities


def get_portaudio_capabilities(recorder: "AudioRecorder") -> dict[str, Any]:
    """Get cached or detect PortAudio capabilities.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        Dictionary of capabilities.
    """
    # Check for cached capabilities
    if hasattr(recorder, "_portaudio_capabilities"):
        return recorder._portaudio_capabilities

    # Detect and cache
    capabilities = detect_portaudio_capabilities(recorder)
    recorder._portaudio_capabilities = capabilities
    return capabilities


def create_stream_with_compatible_params(
    recorder: "AudioRecorder",
    callback: Any,
    device: Optional[int] = None,
) -> Optional[sd.InputStream]:
    """Create a stream with compatible parameters.

    Args:
        recorder: The AudioRecorder instance.
        callback: The audio callback function.
        device: Device index or None for default.

    Returns:
        Created InputStream or None on failure.
    """
    target_device = device if device is not None else recorder.device

    # Get capabilities
    caps = get_portaudio_capabilities(recorder)

    # Determine channels
    max_channels = caps.get("max_channels", 2)
    channels = min(recorder.channels, max_channels)

    # Determine sample rate
    if recorder.sample_rate in caps.get("supported_rates", []):
        sample_rate = recorder.sample_rate
    elif caps.get("supported_rates"):
        # Pick closest supported rate
        sample_rate = min(
            caps["supported_rates"], key=lambda r: abs(r - recorder.sample_rate)
        )
    else:
        sample_rate = caps.get("default_rate", recorder.sample_rate)

    # Try to create stream
    try:
        stream = sd.InputStream(
            channels=channels,
            samplerate=sample_rate,
            blocksize=recorder.chunk_size,
            dtype=np.float32,
            device=target_device,
            callback=callback,
        )
        logger.debug(
            f"Created stream: {channels}ch @ {sample_rate}Hz, "
            f"blocksize={recorder.chunk_size}"
        )
        return stream

    except Exception as e:
        logger.error(f"Stream creation failed: {e}")
        return None


def report_optimization_benefits(recorder: "AudioRecorder") -> None:
    """Report the benefits of sample rate optimization.

    Args:
        recorder: The AudioRecorder instance.
    """
    configured = recorder.configured_sample_rate
    actual = recorder.sample_rate
    native = recorder.parakeet_native_rate

    if actual == native:
        logger.info(
            f"Sample rate optimized: Using native {native}Hz (no resampling needed)"
        )
    elif actual != configured:
        logger.info(
            f"Sample rate adjusted: {configured}Hz -> {actual}Hz "
            f"(resampling to {native}Hz for ASR)"
        )
    else:
        logger.debug(
            f"Using configured sample rate: {actual}Hz (will resample to {native}Hz)"
        )


def validate_sample_rate_compatibility(recorder: "AudioRecorder") -> bool:
    """Validate that the sample rate is compatible with the device.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if sample rate is compatible, False otherwise.
    """
    if test_device_sample_rate(recorder, recorder.sample_rate):
        logger.debug(f"Sample rate {recorder.sample_rate}Hz is compatible")
        return True

    # Try to find a compatible rate
    optimal_rate = select_optimal_sample_rate(recorder)
    if optimal_rate != recorder.sample_rate:
        logger.warning(
            f"Sample rate {recorder.sample_rate}Hz not compatible, "
            f"switching to {optimal_rate}Hz"
        )
        recorder.sample_rate = optimal_rate
        return True

    logger.error("No compatible sample rate found")
    return False


def test_portaudio_capability_detection(recorder: "AudioRecorder") -> dict[str, Any]:
    """Test PortAudio capability detection and return results.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        Dictionary with test results.
    """
    results: dict[str, Any] = {}

    # Test capability detection
    try:
        caps = detect_portaudio_capabilities(recorder)
        results["capability_detection"] = {
            "status": "✅ Capability detection passed",
            "max_channels": caps.get("max_channels", 0),
            "default_rate": caps.get("default_rate", 0),
            "supported_rates": caps.get("supported_rates", []),
            "host_api": caps.get("host_api", "Unknown"),
        }
        logger.info(f"Capability detection: {caps}")
    except Exception as e:
        results["capability_detection"] = {"status": f"FAILED: {e}"}

    # Test compatible stream creation
    try:
        stream = create_stream_with_compatible_params(
            recorder, lambda *args: None, recorder.device
        )
        if stream is not None:
            stream.close()
            results["compatible_stream_creation"] = {
                "status": "✅ Stream creation passed"
            }
        else:
            results["compatible_stream_creation"] = {"status": "FAILED: No stream"}
    except Exception as e:
        results["compatible_stream_creation"] = {"status": f"FAILED: {e}"}

    return results


# end src/audio/stream_manager.py
