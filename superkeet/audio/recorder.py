# start src/audio/recorder.py
"""Audio recording functionality for SuperKeet.

This is the main AudioRecorder class that delegates to specialized modules:
- permissions: Microphone permission validation
- device_manager: Device enumeration and validation
- stream_manager: Stream creation and sample rate optimization
- memory_manager: Buffer management and memory monitoring
- diagnostics: Logging and troubleshooting
- error_recovery: Error handling and recovery strategies
"""

import queue
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal

from superkeet.audio import device_manager as dm
from superkeet.audio import diagnostics as diag
from superkeet.audio import error_recovery as recovery
from superkeet.audio import memory_manager as mem
from superkeet.audio import permissions as perm
from superkeet.audio import stream_manager as stream
from superkeet.config.config_loader import config
from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioRecorder(QObject):
    """Manages audio recording from the default microphone.

    This class provides a high-level interface for audio recording,
    delegating specialized functionality to helper modules.
    """

    # Signal emitted when audio chunk is ready for visualization
    audio_chunk_ready = Signal(np.ndarray)

    def __init__(self) -> None:
        """Initialize the audio recorder."""
        super().__init__()
        # Get initial configuration
        self.configured_sample_rate = config.get("audio.sample_rate", 16000)
        self.channels = config.get("audio.channels", 1)
        self.chunk_size = config.get("audio.chunk_size", 1024)
        self.device = config.get("audio.device", None)
        self.gain = config.get("audio.gain", 1.0)

        # Memory management settings for audio buffers
        self.max_recording_duration = config.get(
            "audio.max_recording_duration", 300
        )  # 5 minutes max
        self.buffer_size_limit = config.get(
            "audio.buffer_size_limit", 100
        )  # 100 MB limit
        self.enable_buffer_monitoring = config.get(
            "audio.enable_buffer_monitoring", True
        )

        # Validate configured device supports input and fallback if needed
        self._validate_and_fix_device_configuration()

        # Parakeet ASR native sample rate - our optimization target
        self.parakeet_native_rate = 16000

        # Determine optimal sample rate based on device capabilities
        self.sample_rate = self._select_optimal_sample_rate()

        # Track if we had to override user configuration
        self.rate_auto_optimized = self.sample_rate != self.configured_sample_rate

        # Audio storage with memory management
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self.recording = False
        self.stream: Optional[sd.InputStream] = None
        self.audio_data: list[np.ndarray] = []
        self.recording_start_time = 0.0

        # Memory monitoring
        self._total_audio_size = 0
        self._last_memory_check = 0.0

        # Debug settings
        self.debug_save_audio = config.get("debug.save_audio_files", False)
        self.debug_audio_dir = config.get("debug.audio_debug_dir", "debug_audio")

        # Initialize debug directory if needed
        if self.debug_save_audio:
            Path(self.debug_audio_dir).mkdir(exist_ok=True)

        # Log initialization with optimization info
        if self.rate_auto_optimized:
            logger.info(
                f"AudioRecorder optimized: {self.configured_sample_rate}Hz -> "
                f"{self.sample_rate}Hz (device-optimal)"
            )
        else:
            logger.info(
                f"AudioRecorder initialized: {self.sample_rate}Hz, {self.channels}ch"
            )

        # Log memory management settings
        logger.info(
            f"Buffer limits: {self.max_recording_duration}s duration, "
            f"{self.buffer_size_limit}MB size"
        )

        # Report optimization benefits
        self._report_optimization_benefits()

        # Debug audio setup on initialization
        self.debug_audio_setup()

        # Validate sample rate compatibility
        self._validate_sample_rate_compatibility()

    # =========================================================================
    # Permission Validation (delegates to permissions module)
    # =========================================================================

    def _validate_microphone_permissions(self) -> bool:
        """Validate microphone permissions."""
        return perm.validate_microphone_permissions(self)

    def _test_basic_microphone_access(self) -> bool:
        """Test basic microphone access."""
        return perm.test_basic_microphone_access(self)

    def _try_portaudio_reinit(self) -> bool:
        """Attempt to reinitialize PortAudio."""
        return perm.try_portaudio_reinit(self)

    def _force_portaudio_reinit(self) -> bool:
        """Force a complete PortAudio reinitialization."""
        return perm.force_portaudio_reinit()

    def _execute_reinit_strategies(self) -> bool:
        """Execute multiple reinitialization strategies."""
        return perm.execute_reinit_strategies(self)

    def _try_single_reinit_strategy(self, strategy: Any) -> bool:
        """Try a single reinitialization strategy."""
        return perm.try_single_reinit_strategy(strategy)

    def _verify_coreaudio_daemon_health(self) -> bool:
        """Verify Core Audio daemon health on macOS."""
        return perm.verify_coreaudio_daemon_health()

    # =========================================================================
    # Device Management (delegates to device_manager module)
    # =========================================================================

    def _validate_and_fix_device_configuration(self) -> None:
        """Validate configured device and fix if necessary."""
        dm.validate_and_fix_device_configuration(self)

    def _get_device_recommendation(self) -> Optional[int]:
        """Get a recommended input device."""
        return dm.get_device_recommendation(self)

    def _device_supports_stereo(self, device: Optional[int] = None) -> bool:
        """Check if device supports stereo input."""
        target_device = device if device is not None else self.device
        return dm.device_supports_stereo(target_device)

    def _test_device_system_accessibility(self, device_index: int) -> bool:
        """Test if a device is accessible at the system level."""
        return dm.test_device_system_accessibility(device_index)

    def _try_alternative_input_devices(self) -> bool:
        """Try alternative input devices when current device fails."""
        return dm.try_alternative_input_devices(self)

    def _fallback_device_enumeration(self) -> list[dict[str, Any]]:
        """Enumerate devices using fallback method."""
        return dm.fallback_device_enumeration()

    def _is_rodecast_device(self, device: Optional[int] = None) -> bool:
        """Check if the device is a RODECaster Pro or similar."""
        target_device = device if device is not None else self.device
        return dm.is_rodecast_device(target_device)

    def _safe_query_hostapis(self) -> list[dict[str, Any]]:
        """Safely query host APIs."""
        return dm.safe_query_hostapis()

    # =========================================================================
    # Stream Management (delegates to stream_manager module)
    # =========================================================================

    def _select_optimal_sample_rate(self, device: Optional[int] = None) -> int:
        """Select the optimal sample rate for the device."""
        return stream.select_optimal_sample_rate(self, device)

    def _test_device_sample_rate(
        self, sample_rate: int, device: Optional[int] = None
    ) -> bool:
        """Test if a sample rate works with the device."""
        return stream.test_device_sample_rate(self, sample_rate, device)

    def _test_device_sample_rate_for_device(
        self, device: int, sample_rate: int, channels: int = 1
    ) -> bool:
        """Test if a sample rate works with a specific device."""
        return stream.test_device_sample_rate_for_device(device, sample_rate, channels)

    def _detect_portaudio_capabilities(self) -> dict[str, Any]:
        """Detect PortAudio capabilities for the current device."""
        return stream.detect_portaudio_capabilities(self)

    def get_portaudio_capabilities(self) -> dict[str, Any]:
        """Get cached or detect PortAudio capabilities."""
        return stream.get_portaudio_capabilities(self)

    def _create_stream_with_compatible_params(
        self, callback: Any, device: Optional[int] = None
    ) -> Optional[sd.InputStream]:
        """Create a stream with compatible parameters."""
        return stream.create_stream_with_compatible_params(self, callback, device)

    def _report_optimization_benefits(self) -> None:
        """Report the benefits of sample rate optimization."""
        stream.report_optimization_benefits(self)

    def _validate_sample_rate_compatibility(self) -> bool:
        """Validate that the sample rate is compatible with the device."""
        return stream.validate_sample_rate_compatibility(self)

    def test_portaudio_capability_detection(self) -> dict[str, Any]:
        """Test PortAudio capability detection and return results."""
        return stream.test_portaudio_capability_detection(self)

    # =========================================================================
    # Memory Management (delegates to memory_manager module)
    # =========================================================================

    def _should_stop_due_to_limits(self) -> bool:
        """Check if recording should stop due to memory or duration limits."""
        return mem.should_stop_due_to_limits(self)

    def _check_memory_usage(self) -> dict[str, Any]:
        """Check current memory usage of audio buffers."""
        return mem.check_memory_usage(self)

    def get_memory_stats(self) -> dict[str, Any]:
        """Get detailed memory statistics for the recorder."""
        return mem.get_memory_stats(self)

    def clear_audio_buffers(self) -> None:
        """Clear all audio buffers and reset counters."""
        mem.clear_audio_buffers(self)

    def _save_debug_audio(self, audio_data: np.ndarray, suffix: str = "") -> None:
        """Save audio data for debugging purposes."""
        mem.save_debug_audio(self, audio_data, suffix)

    # =========================================================================
    # Diagnostics (delegates to diagnostics module)
    # =========================================================================

    def _log_detailed_error_info(self, error: Exception, context: str = "") -> None:
        """Log detailed error information for debugging."""
        diag.log_detailed_error_info(
            error, self.device, self.sample_rate, self.channels
        )

    def _suggest_audio_fixes(self, error: Exception) -> None:
        """Suggest fixes based on the error type."""
        diag.suggest_audio_fixes(error)

    def _suggest_coreaudio_daemon_restart(self) -> None:
        """Suggest how to restart the Core Audio daemon."""
        diag.suggest_coreaudio_daemon_restart()

    def _suggest_rodecast_pro_fixes(self) -> None:
        """Suggest fixes specific to RODECaster Pro devices."""
        diag.suggest_rodecast_pro_fixes()

    def _log_system_audio_diagnostics(self) -> None:
        """Log comprehensive system audio diagnostics."""
        diag.log_system_audio_diagnostics()

    def _log_macos_audio_diagnostics(self) -> None:
        """Log macOS-specific audio diagnostics."""
        diag.log_macos_audio_diagnostics()

    def _log_coreaudio_processes(self) -> None:
        """Log Core Audio related processes."""
        diag.log_coreaudio_processes()

    def _log_system_audio_devices(self) -> None:
        """Log system-level audio device information."""
        diag.log_system_audio_devices()

    def _log_linux_audio_diagnostics(self) -> None:
        """Log Linux-specific audio diagnostics."""
        diag.log_linux_audio_diagnostics()

    def _log_windows_audio_diagnostics(self) -> None:
        """Log Windows-specific audio diagnostics."""
        diag.log_windows_audio_diagnostics()

    def debug_audio_setup(self) -> None:
        """Debug and log the current audio setup."""
        diag.debug_audio_setup(self)

    # =========================================================================
    # Error Recovery (delegates to error_recovery module)
    # =========================================================================

    def _handle_start_error(self, error: Exception) -> bool:
        """Handle errors during stream start."""
        return recovery.handle_start_error(self, error)

    def _attempt_enhanced_recovery(self, error: Exception) -> bool:
        """Attempt enhanced error recovery strategies."""
        return recovery.attempt_enhanced_recovery(self, error)

    def _cleanup_failed_start(self) -> None:
        """Clean up after a failed start attempt."""
        recovery.cleanup_failed_start(self)

    def _try_fallback_device(self) -> bool:
        """Try using a fallback device."""
        return recovery.try_fallback_device(self)

    def _try_fallback_sample_rates(self) -> bool:
        """Try fallback sample rates."""
        return recovery.try_fallback_sample_rates(self)

    def _try_alternative_configs(self) -> bool:
        """Try alternative configurations."""
        return recovery.try_alternative_configs(self)

    def _handle_portaudio_9986_error(self, device_index: Optional[int]) -> bool:
        """Handle specific PortAudio -9986 Internal Error scenarios."""
        return recovery.handle_portaudio_9986_error(self, device_index)

    def _attempt_auhal_recovery(self, device_index: Optional[int]) -> bool:
        """Attempt AUHAL-specific recovery."""
        return recovery.attempt_auhal_recovery(self, device_index)

    def _handle_audio_unit_error(self, error: Exception) -> bool:
        """Handle Audio Unit specific errors."""
        return recovery.handle_audio_unit_error(self, error)

    def _handle_rodecast_pro_errors(self, error: Exception) -> bool:
        """Handle RODECaster Pro specific errors."""
        return recovery.handle_rodecast_pro_errors(self, error)

    def test_enhanced_error_recovery(self) -> dict[str, Any]:
        """Test the enhanced error recovery system."""
        return recovery.test_enhanced_error_recovery(self)

    # =========================================================================
    # Core Recording Methods (implemented here)
    # =========================================================================

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback function for audio stream.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Time information from PortAudio.
            status: Callback flags from PortAudio.
        """
        if status:
            logger.debug(f"Audio callback status: {status}")

        if not self.recording:
            return

        # Check limits
        if self._should_stop_due_to_limits():
            logger.warning("Recording stopped due to limits")
            self.recording = False
            return

        # Apply gain
        audio = indata.copy()
        if self.gain != 1.0:
            audio = audio * self.gain

        # Convert stereo to mono if needed
        if audio.ndim > 1 and audio.shape[1] > 1 and self.channels == 1:
            audio = np.mean(audio, axis=1, keepdims=True)

        # Ensure proper shape
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        # Track memory usage
        self._total_audio_size += audio.nbytes

        # Store audio
        self.audio_data.append(audio)

        # Put in queue for visualization (non-blocking)
        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            pass  # Skip if queue is full

        # Emit signal for visualization
        self.audio_chunk_ready.emit(audio.flatten())

        # Periodic memory check
        if self.enable_buffer_monitoring:
            current_time = time.time()
            if current_time - self._last_memory_check > 5.0:
                self._check_memory_usage()
                self._last_memory_check = current_time

    def start(self) -> bool:
        """Start recording audio.

        Returns:
            True if recording started successfully, False otherwise.
        """
        if self.recording:
            logger.warning("Already recording")
            return False

        # Phase 1: Permission validation
        logger.info("Validating microphone permissions...")
        if not self._validate_microphone_permissions():
            logger.error("Microphone permissions validation failed")
            return False

        try:
            # Clear any previous data and reset counters
            self.audio_data.clear()
            self._total_audio_size = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Record start time for duration tracking
            self.recording_start_time = time.time()

            # Determine appropriate channel count for the selected device
            stream_channels = self.channels
            if self.device is not None:
                try:
                    device_info = sd.query_devices(self.device)
                    device_max_channels = int(device_info["max_input_channels"])
                    if device_max_channels >= 2 and self.channels == 1:
                        stream_channels = min(2, device_max_channels)
                        logger.debug(
                            f"Using {stream_channels} channels for device "
                            f"{self.device} (will convert to {self.channels})"
                        )
                    else:
                        stream_channels = min(self.channels, device_max_channels)
                except Exception as e:
                    logger.debug(
                        f"Could not query device info: {e}, using configured channels"
                    )
                    stream_channels = self.channels

            # Create and start stream
            self.stream = sd.InputStream(
                device=self.device,
                channels=stream_channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32,
            )

            self.stream.start()
            self.recording = True
            logger.info("Started recording successfully")
            return True

        except Exception as e:
            return self._handle_start_error(e)

    def stop(self) -> Optional[np.ndarray]:
        """Stop recording and return the recorded audio.

        Returns:
            Numpy array of recorded audio data, or None if no data.
        """
        if not self.recording:
            logger.warning("Not currently recording")
            return None

        self.recording = False

        # Close stream
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            self.stream = None

        # Combine audio data
        if not self.audio_data:
            logger.warning("No audio data recorded")
            return None

        try:
            audio = np.concatenate(self.audio_data, axis=0)

            # Calculate duration
            duration = len(audio) / self.sample_rate
            logger.info(
                f"Stopped recording: {duration:.2f}s, "
                f"{self._total_audio_size / 1024:.1f}KB"
            )

            # Save debug audio if enabled
            if self.debug_save_audio:
                self._save_debug_audio(audio.flatten(), "_recording")

            return audio.flatten()

        except Exception as e:
            logger.error(f"Error combining audio data: {e}")
            return None

        finally:
            # Clear buffers
            self.clear_audio_buffers()

    def get_devices(self) -> list[dict[str, Any]]:
        """Get list of available input devices.

        Returns:
            List of device dictionaries with index, name, and channels.
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
            logger.error(f"Failed to get devices: {e}")
            return self._fallback_device_enumeration()

    def update_device(self, device_index: Optional[int]) -> None:
        """Update the recording device.

        Args:
            device_index: New device index, or None for default.
        """
        self.device = device_index
        logger.info(f"Updated recording device to: {device_index}")

        # Re-optimize sample rate for new device
        self.sample_rate = self._select_optimal_sample_rate()
        self._report_optimization_benefits()

    def validate_audio_data(self, audio_data: np.ndarray) -> bool:
        """Validate audio data for transcription.

        Args:
            audio_data: Audio data to validate.

        Returns:
            True if audio data is valid for transcription.
        """
        if audio_data is None:
            return False

        if len(audio_data) == 0:
            return False

        # Check for minimum duration (100ms)
        min_samples = int(self.sample_rate * 0.1)
        if len(audio_data) < min_samples:
            logger.warning("Audio too short for transcription")
            return False

        # Check for silence
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.001:
            logger.warning("Audio appears to be silent")
            return False

        return True


# end src/audio/recorder.py
