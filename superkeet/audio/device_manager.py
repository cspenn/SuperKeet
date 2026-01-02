# start src/audio/device_manager.py
"""Device management for audio recording.

This module handles device enumeration, validation, and configuration
for the AudioRecorder class.
"""

from typing import TYPE_CHECKING, Any, Optional

import sounddevice as sd

from superkeet.utils.logger import setup_logger

if TYPE_CHECKING:
    from superkeet.audio.recorder import AudioRecorder

logger = setup_logger(__name__)


def validate_and_fix_device_configuration(recorder: "AudioRecorder") -> None:
    """Validate configured device and fix if necessary.

    Args:
        recorder: The AudioRecorder instance.
    """
    if recorder.device is None:
        logger.debug("No device configured, using system default")
        return

    try:
        device_info = sd.query_devices(recorder.device)

        # Check if device supports input
        if device_info["max_input_channels"] < 1:
            logger.warning(
                f"Configured device {recorder.device} has no input channels, "
                "falling back to default"
            )
            recorder.device = None
            return

        logger.debug(
            f"Device {recorder.device} validated: "
            f"{device_info['name']}, "
            f"{device_info['max_input_channels']} input channels"
        )

    except Exception as e:
        logger.warning(f"Device {recorder.device} validation failed: {e}")
        recorder.device = None


def get_device_recommendation(recorder: "AudioRecorder") -> Optional[int]:
    """Get a recommended input device.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        Device index of recommended device, or None for system default.
    """
    try:
        devices = sd.query_devices()

        # Priority 1: Current configured device if valid
        if recorder.device is not None:
            try:
                device_info = sd.query_devices(recorder.device)
                if device_info["max_input_channels"] > 0:
                    return recorder.device
            except Exception:
                pass

        # Priority 2: System default input device
        default_input = sd.default.device[0]
        if default_input is not None and default_input >= 0:
            try:
                device_info = sd.query_devices(default_input)
                if device_info["max_input_channels"] > 0:
                    logger.debug(f"Using default input device: {device_info['name']}")
                    return default_input
            except Exception:
                pass

        # Priority 3: First available input device
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                logger.debug(f"Using first available input device: {device['name']}")
                return i

        logger.warning("No input devices found")
        return None

    except Exception as e:
        logger.error(f"Device recommendation failed: {e}")
        return None


def device_supports_stereo(device: Optional[int]) -> bool:
    """Check if a device supports stereo input.

    Args:
        device: Device index or None for default.

    Returns:
        True if device supports stereo (2+ channels).
    """
    try:
        if device is None:
            device_info = sd.query_devices(sd.default.device[0])
        else:
            device_info = sd.query_devices(device)

        return device_info["max_input_channels"] >= 2

    except Exception as e:
        logger.debug(f"Could not check stereo support: {e}")
        return False


def test_device_system_accessibility(device_index: int) -> bool:
    """Test if a device is accessible at the system level.

    Args:
        device_index: The device index to test.

    Returns:
        True if device is accessible, False otherwise.
    """
    try:
        device_info = sd.query_devices(device_index)

        # Check if device has input channels
        if device_info["max_input_channels"] < 1:
            return False

        # Try to query the device specifically
        logger.debug(
            f"Device {device_index} ({device_info['name']}) "
            f"appears accessible with {device_info['max_input_channels']} channels"
        )
        return True

    except Exception as e:
        logger.debug(f"Device {device_index} not accessible: {e}")
        return False


def try_alternative_input_devices(recorder: "AudioRecorder") -> bool:
    """Try alternative input devices when current device fails.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if an alternative device was found and works.
    """
    logger.info("Searching for alternative input devices...")

    try:
        devices = sd.query_devices()
        current_device = recorder.device

        for i, device in enumerate(devices):
            # Skip current failed device
            if i == current_device:
                continue

            # Skip output-only devices
            if device["max_input_channels"] < 1:
                continue

            logger.debug(f"Trying alternative device {i}: {device['name']}")

            # Test the device
            if test_device_system_accessibility(i):
                logger.info(f"Found working alternative device: {device['name']}")
                recorder.device = i
                return True

        logger.warning("No alternative input devices available")
        return False

    except Exception as e:
        logger.error(f"Alternative device search failed: {e}")
        return False


def fallback_device_enumeration() -> list[dict[str, Any]]:
    """Enumerate devices using fallback method.

    Returns:
        List of device dictionaries.
    """
    try:
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "default_samplerate": device["default_samplerate"],
                    }
                )

        return input_devices

    except Exception as e:
        logger.error(f"Fallback device enumeration failed: {e}")
        return []


def is_rodecast_device(device: Optional[int]) -> bool:
    """Check if the device is a RODECaster Pro or similar.

    Args:
        device: Device index or None for default.

    Returns:
        True if device appears to be a RODECaster.
    """
    try:
        if device is None:
            return False

        device_info = sd.query_devices(device)
        device_name = device_info.get("name", "").lower()

        rodecast_indicators = ["rodecaster", "rode", "rodecast"]
        for indicator in rodecast_indicators:
            if indicator in device_name:
                return True

        return False

    except Exception:
        return False


def safe_query_hostapis() -> list[dict[str, Any]]:
    """Safely query host APIs, ensuring device_count is always present.

    Returns:
        List of host API information, or empty list on failure.
    """
    try:
        apis = list(sd.query_hostapis())
        # Ensure device_count is present in all APIs
        for api in apis:
            if "device_count" not in api:
                # Count devices for this host API
                api["device_count"] = 0
                try:
                    devices = sd.query_devices()
                    host_api_index = apis.index(api)
                    for d in devices:
                        if d.get("hostapi") == host_api_index:
                            api["device_count"] += 1
                except Exception:
                    pass
        return apis
    except Exception as e:
        logger.debug(f"Host API query failed: {e}")
        return []


# end src/audio/device_manager.py
