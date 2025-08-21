"""Centralized device validation and management service."""

from typing import Any, Dict, List, Optional, Tuple

import sounddevice as sd

from src.utils.exceptions import AudioDeviceError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DeviceService:
    """Centralized service for audio device validation and management."""

    def __init__(self):
        """Initialize device service."""
        self._cached_devices = None
        self._cache_timestamp = 0

    def validate_device_for_input(self, device_index: int) -> Tuple[bool, str]:
        """Validate that a device supports audio input.

        Args:
            device_index: Device index to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            device_info = sd.query_devices(device_index)

            if device_info["max_input_channels"] == 0:
                error_msg = f"Device '{device_info['name']}' is an output-only device"
                logger.warning(f"🟡 {error_msg}")
                return False, error_msg

            logger.info(
                f"✅ Validated device {device_index}: {device_info['name']} "
                f"({device_info['max_input_channels']} input channels)"
            )
            return True, ""

        except Exception as e:
            error_msg = f"Device {device_index} is not available: {e}"
            logger.error(f"🛑 {error_msg}")
            return False, error_msg

    def get_device_info(self, device_index: Optional[int] = None) -> Dict[str, Any]:
        """Get device information.

        Args:
            device_index: Device index (None for default input device)

        Returns:
            Device information dictionary

        Raises:
            AudioDeviceError: If device is not available
        """
        try:
            if device_index is None:
                return sd.query_devices(kind="input")
            else:
                return sd.query_devices(device_index)
        except Exception as e:
            raise AudioDeviceError(f"Failed to get device info: {e}")

    def get_device_recommendation(self, device_info: Dict[str, Any]) -> str:
        """Get recommendation string for a device.

        Args:
            device_info: Device information dictionary

        Returns:
            Recommendation string with emoji indicator
        """
        device_rate = int(device_info["default_samplerate"])

        # Test if device supports 16kHz
        supports_16k = self._test_device_sample_rate(device_info["index"], 16000)

        if supports_16k:
            return "🟢 OPTIMAL (supports 16kHz)"
        elif device_rate >= 44100:  # Professional rates
            return "🟡 GOOD (professional quality)"
        elif device_rate >= 22050:  # Standard rates
            return "🟡 FAIR (standard quality)"
        else:
            return "🟡 POOR (low quality)"

    def _test_device_sample_rate(self, device_index: int, test_rate: int) -> bool:
        """Test if a device supports a specific sample rate.

        Args:
            device_index: Device index to test
            test_rate: Sample rate to test

        Returns:
            True if device supports the sample rate
        """
        try:
            sd.check_input_settings(
                device=device_index, channels=1, samplerate=test_rate
            )
            return True
        except Exception:
            return False

    def get_available_input_devices(self) -> List[Dict[str, Any]]:
        """Get list of available input devices with recommendations.

        Returns:
            List of device dictionaries with additional metadata
        """
        try:
            devices = sd.query_devices()
            input_devices = []

            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    device_dict = dict(device)
                    device_dict["index"] = i
                    device_dict["recommendation"] = self.get_device_recommendation(
                        device_dict
                    )
                    device_dict["is_default"] = i == sd.default.device[0]
                    input_devices.append(device_dict)

            return input_devices

        except Exception as e:
            logger.error(f"🛑 Failed to get input devices: {e}")
            return []

    def get_optimal_sample_rate_for_device(
        self, device_index: Optional[int] = None
    ) -> int:
        """Get optimal sample rate for a device.

        Args:
            device_index: Device index (None for default)

        Returns:
            Optimal sample rate
        """
        try:
            device_info = self.get_device_info(device_index)
            device_native_rate = int(device_info["default_samplerate"])

            # Test if device supports 16kHz (Parakeet native)
            if self._test_device_sample_rate(
                device_index or device_info["index"], 16000
            ):
                logger.info(
                    "🟢 OPTIMAL: Device supports 16kHz natively - no ASR resampling needed!"
                )
                return 16000

            # Use device native rate if it's reasonable
            if device_native_rate >= 16000:
                logger.info(
                    f"🟡 GOOD: Using device native {device_native_rate}Hz → "
                    "single downsample to 16kHz for ASR"
                )
                return device_native_rate

            # Use a reasonable minimum
            logger.warning(
                f"🟡 SUBOPTIMAL: Device only supports {device_native_rate}Hz - "
                "requires upsampling for ASR"
            )
            return max(device_native_rate, 22050)

        except Exception as e:
            logger.error(f"Sample rate optimization failed: {e}")
            logger.info("🟡 Falling back to 16000Hz")
            return 16000
