# start src/audio/recorder.py
"""Audio recording functionality for SuperKeet."""

import queue
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal

from src.config.config_loader import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioRecorder(QObject):
    """Manages audio recording from the default microphone."""

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
        self.max_recording_duration = config.get("audio.max_recording_duration", 300)  # 5 minutes max
        self.buffer_size_limit = config.get("audio.buffer_size_limit", 100)  # 100 MB limit
        self.enable_buffer_monitoring = config.get("audio.enable_buffer_monitoring", True)

        # Validate configured device supports input and fallback if needed
        self._validate_and_fix_device_configuration()

        # Parakeet ASR native sample rate - our optimization target
        self.parakeet_native_rate = 16000

        # Determine optimal sample rate based on device capabilities
        self.sample_rate = self._select_optimal_sample_rate()

        # Track if we had to override user configuration
        self.rate_auto_optimized = self.sample_rate != self.configured_sample_rate

        # Audio storage with memory management
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)  # Limit queue size
        self.recording = False
        self.stream: Optional[sd.InputStream] = None
        self.audio_data: list[np.ndarray] = []
        self.recording_start_time = 0

        # Memory monitoring
        self._total_audio_size = 0
        self._last_memory_check = 0

        # Debug settings
        self.debug_save_audio = config.get("debug.save_audio_files", False)
        self.debug_audio_dir = config.get("debug.audio_debug_dir", "debug_audio")

        # Initialize debug directory if needed
        if self.debug_save_audio:
            Path(self.debug_audio_dir).mkdir(exist_ok=True)

        # Log initialization with optimization info
        if self.rate_auto_optimized:
            logger.info(
                f"🟢 AudioRecorder optimized: {self.configured_sample_rate}Hz → {self.sample_rate}Hz (device-optimal)"
            )
        else:
            logger.info(
                f"AudioRecorder initialized: {self.sample_rate}Hz, {self.channels}ch"
            )

        # Log memory management settings
        logger.info(f"Buffer limits: {self.max_recording_duration}s duration, {self.buffer_size_limit}MB size")

        # Report optimization benefits
        self._report_optimization_benefits()

        # Debug audio setup on initialization
        self.debug_audio_setup()

        # Validate sample rate compatibility
        self._validate_sample_rate_compatibility()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback function for audio stream with memory management.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Timing information.
            status: Stream status flags.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self.recording:
            # Check for buffer limits to prevent memory runaway
            if self._should_stop_due_to_limits():
                logger.warning("🟡 Stopping recording due to memory/duration limits")
                self.recording = False
                return

            # Copy data to avoid reference issues
            audio_copy = indata.copy()

            # Apply gain if configured
            if self.gain != 1.0:
                audio_copy = audio_copy * self.gain
                # Prevent clipping
                audio_copy = np.clip(audio_copy, -1.0, 1.0)

            # Add to our data list with size tracking
            chunk_size = audio_copy.nbytes
            self.audio_data.append(audio_copy)
            self._total_audio_size += chunk_size

            # Only add to queue if there's space (prevents queue overflow)
            try:
                self.audio_queue.put_nowait(audio_copy)
            except queue.Full:
                # Queue is full, just continue (we have the data in audio_data)
                pass

            # Emit signal for audio animation
            self.audio_chunk_ready.emit(audio_copy)

            # Periodic memory check (every 100 chunks to avoid overhead)
            if len(self.audio_data) % 100 == 0:
                self._check_memory_usage()

    def start(self) -> None:
        """Start recording audio."""
        if self.recording:
            logger.warning("Already recording")
            return

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
            import time
            self.recording_start_time = time.time()

            # Create and start stream
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32,
            )

            self.stream.start()
            self.recording = True
            logger.info("Started recording")

        except Exception as e:
            # Enhanced error recovery with macOS-specific handling
            error_msg = str(e).lower()

            # Check for specific PortAudio errors
            if "paerrorcode -9986" in error_msg:
                logger.error("🚫 Audio device unavailable - possible permission issue or device in use")
                self._suggest_audio_fixes()
            elif "paerrorcode -9988" in error_msg:
                logger.error("🚫 Invalid device - device may have been disconnected")
            elif "audio unit: invalid property value" in error_msg:
                logger.error("🚫 macOS Audio Unit error - trying alternative configuration")

            logger.error(f"Failed to start recording with device {self.device}: {e}")

            # If we were using a specific device, try fallback strategies
            if self.device is not None:
                logger.warning("Attempting fallback strategies...")

                # Strategy 1: Try default device
                if self._try_fallback_device(None):
                    return

                # Strategy 2: Try alternative sample rates
                if self._try_fallback_sample_rates():
                    return

                # Strategy 3: Try alternative configurations
                if self._try_alternative_configs():
                    return
            else:
                # We were already using default device - try alternative configs
                logger.warning("Default device failed - trying alternative configurations...")
                if self._try_alternative_configs():
                    return

            # All fallbacks failed
            logger.error("🚫 All audio fallback strategies failed")
            self._suggest_audio_fixes()
            raise e

    def _try_fallback_device(self, fallback_device: Optional[int]) -> bool:
        """Try to start recording with a fallback device.
        
        Args:
            fallback_device: Device to try, or None for default
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.stream = sd.InputStream(
                device=fallback_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32,
            )
            self.stream.start()
            self.recording = True

            original_device = self.device
            self.device = fallback_device

            device_name = "default" if fallback_device is None else str(fallback_device)
            logger.info(f"🟢 Successfully started recording with {device_name} device (fallback)")
            logger.warning(f"🟡 Device {original_device} failed, now using {device_name} device")
            return True

        except Exception as e:
            logger.debug(f"Fallback device {fallback_device} also failed: {e}")
            return False

    def _try_fallback_sample_rates(self) -> bool:
        """Try alternative sample rates for better device compatibility.
        
        Returns:
            True if successful, False otherwise
        """
        # Common sample rates to try in order of preference
        fallback_rates = [44100, 48000, 22050, 16000, 8000]

        for rate in fallback_rates:
            if rate == self.sample_rate:
                continue  # Already tried this rate

            try:
                logger.debug(f"Trying fallback sample rate: {rate}Hz")
                self.stream = sd.InputStream(
                    device=self.device,
                    channels=self.channels,
                    samplerate=rate,
                    blocksize=self.chunk_size,
                    callback=self._audio_callback,
                    dtype=np.float32,
                )
                self.stream.start()
                self.recording = True

                original_rate = self.sample_rate
                self.sample_rate = rate

                logger.info(f"🟢 Successfully started recording at {rate}Hz (fallback from {original_rate}Hz)")
                logger.warning(f"🟡 Sample rate changed from {original_rate}Hz to {rate}Hz for compatibility")
                return True

            except Exception as e:
                logger.debug(f"Sample rate {rate}Hz failed: {e}")
                continue

        return False

    def _try_alternative_configs(self) -> bool:
        """Try alternative audio configurations for compatibility.
        
        Returns:
            True if successful, False otherwise
        """
        # Try different configurations
        configs = [
            {"channels": 1, "blocksize": 512},   # Mono, smaller buffer
            {"channels": 1, "blocksize": 2048},  # Mono, larger buffer
            {"channels": 2, "blocksize": 1024},  # Stereo, standard buffer
        ]

        original_channels = self.channels
        original_blocksize = self.chunk_size

        for config in configs:
            if (config["channels"] == self.channels and
                config["blocksize"] == self.chunk_size):
                continue  # Already tried this config

            try:
                logger.debug(f"Trying config: {config['channels']}ch, {config['blocksize']} blocksize")
                self.stream = sd.InputStream(
                    device=self.device,
                    channels=config["channels"],
                    samplerate=self.sample_rate,
                    blocksize=config["blocksize"],
                    callback=self._audio_callback,
                    dtype=np.float32,
                )
                self.stream.start()
                self.recording = True

                self.channels = config["channels"]
                self.chunk_size = config["blocksize"]

                logger.info(f"🟢 Successfully started with alternative config: {self.channels}ch, {self.chunk_size} blocksize")
                logger.warning(f"🟡 Audio config changed from {original_channels}ch/{original_blocksize} to {self.channels}ch/{self.chunk_size}")
                return True

            except Exception as e:
                logger.debug(f"Config {config} failed: {e}")
                continue

        return False

    def _suggest_audio_fixes(self) -> None:
        """Suggest potential fixes for audio issues."""
        logger.error("🔧 SUGGESTED FIXES:")
        logger.error("   1. Check microphone permissions in System Preferences > Security & Privacy > Microphone")
        logger.error("   2. Ensure no other app is using the microphone")
        logger.error("   3. Try disconnecting and reconnecting external microphones")
        logger.error("   4. Restart the application")
        logger.error("   5. Try selecting a different audio device in settings")

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio.

        Returns:
            Numpy array containing the recorded audio.
        """
        if not self.recording:
            logger.warning("Not recording")
            return np.array([])

        try:
            self.recording = False

            # Stop and close stream FIRST to prevent more data
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Small delay to ensure stream is fully stopped
            import time

            time.sleep(0.05)

            # Collect only the audio data that was recorded during the session
            # Don't collect more from queue after stopping
            audio_chunks = list(self.audio_data)  # Copy existing data

            # Clear the queue without adding more data
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Concatenate all chunks
            if audio_chunks:
                audio_array = np.concatenate(audio_chunks, axis=0)
                # Flatten if mono
                if self.channels == 1:
                    audio_array = audio_array.flatten()

                duration = len(audio_array) / self.sample_rate
                logger.info(f"Stopped recording: {duration:.2f}s")

                # Validate audio data quality
                if not self.validate_audio_data(audio_array):
                    logger.warning(
                        "🟡 Audio data validation failed - transcription may be poor"
                    )

                return audio_array
            else:
                logger.warning("No audio data captured")
                return np.array([])

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            raise

    def get_devices(self) -> list[dict]:
        """Get list of available audio input devices.

        Returns:
            List of device information dictionaries.
        """
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "max_input_channels": device["max_input_channels"],
                        "default_samplerate": device["default_samplerate"],
                    }
                )
        return devices

    def update_device(self, device_index: Optional[int] = None) -> None:
        """Update the audio device and re-optimize sample rate.

        Args:
            device_index: Device index to use, or None for default device
        """
        logger.info(f"Updating audio device from {self.device} to {device_index}")

        # Stop current stream if recording
        was_recording = self.recording
        if was_recording:
            self.stop()

        # Update device
        self.device = device_index

        # Re-optimize sample rate for new device
        old_sample_rate = self.sample_rate
        self.sample_rate = self._select_optimal_sample_rate()

        if old_sample_rate != self.sample_rate:
            logger.info(
                f"🔧 Sample rate optimized for new device: {old_sample_rate}Hz → {self.sample_rate}Hz"
            )

        # Debug the new setup
        self.debug_audio_setup()

        # Validate sample rate compatibility with new device
        self._validate_sample_rate_compatibility()

        # Restart recording if it was active
        if was_recording:
            self.start()

        logger.info(f"Audio device updated successfully to {device_index}")

    def debug_audio_setup(self) -> None:
        """Debug current audio setup and available devices."""
        logger.info("=== AUDIO SETUP DEBUG ===")
        logger.info(f"Configured device: {self.device}")
        logger.info(
            f"Sample rate: {self.sample_rate} (configured: {self.configured_sample_rate})"
        )
        logger.info(f"Channels: {self.channels}")
        logger.info(f"Chunk size: {self.chunk_size}")
        if self.rate_auto_optimized:
            logger.info("🔧 Rate auto-optimized for better performance")

        try:
            # Query default device
            default_input = sd.query_devices(kind="input")
            logger.info(
                f"Default input device: [{default_input['index']}] {default_input['name']}"
            )

            # If specific device configured, show its details
            if self.device is not None:
                try:
                    device_info = sd.query_devices(self.device)
                    logger.info(
                        f"Configured device details: [{self.device}] {device_info['name']}"
                    )
                    logger.info(
                        f"  Max input channels: {device_info['max_input_channels']}"
                    )
                    logger.info(
                        f"  Default sample rate: {device_info['default_samplerate']}"
                    )
                except Exception as e:
                    logger.error(f"Invalid device {self.device}: {e}")

            # List all available input devices with optimization recommendations
            logger.info("Available input devices with recommendations:")
            for i, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    is_default = " (DEFAULT)" if i == default_input["index"] else ""
                    is_selected = " (SELECTED)" if i == self.device else ""

                    # Get recommendation for this device
                    recommendation = self._get_device_recommendation(device)

                    logger.info(
                        f"  [{i}] {device['name']} - {device['max_input_channels']} ch @ {device['default_samplerate']}Hz {recommendation}{is_default}{is_selected}"
                    )

        except Exception as e:
            logger.error(f"Audio device debug failed: {e}")

        logger.info("=== END AUDIO DEBUG ===")

    def _get_device_recommendation(self, device_info: dict) -> str:
        """Get recommendation for a specific device.

        Args:
            device_info: Device information dictionary

        Returns:
            Recommendation string with emoji indicator
        """
        device_rate = int(device_info["default_samplerate"])

        # Test if device supports 16kHz
        supports_16k = self._test_device_sample_rate_for_device(
            device_info["index"], 16000
        )

        if supports_16k:
            return "🟢 OPTIMAL (supports 16kHz)"
        elif device_rate >= 44100:  # Professional rates
            return "🟢 EXCELLENT (high quality)"
        elif device_rate >= 22050:  # Decent rates
            return "🟡 GOOD (acceptable quality)"
        else:
            return "🟡 POOR (low quality)"

    def _test_device_sample_rate_for_device(
        self, device_index: int, test_rate: int
    ) -> bool:
        """Test if a specific device supports a sample rate.

        Args:
            device_index: Device index to test
            test_rate: Sample rate to test

        Returns:
            True if device supports the sample rate
        """
        try:
            test_stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=test_rate,
                blocksize=64,
                dtype=np.float32,
            )
            test_stream.close()
            return True
        except Exception:
            return False

    def _select_optimal_sample_rate(self) -> int:
        """Select optimal sample rate based on device capabilities and ASR requirements.

        Priority order:
        1. If device supports 16kHz natively → use 16kHz (no resampling needed for ASR)
        2. If device supports common rates → select closest to 16kHz that minimizes resampling
        3. Use device native rate to minimize one resampling step
        4. Fall back to configured rate

        Returns:
            Optimal sample rate for the current device
        """
        try:
            # Get device info for current device
            if self.device is None:
                device_info = sd.query_devices(kind="input")
            else:
                device_info = sd.query_devices(self.device)

            device_native_rate = int(device_info["default_samplerate"])

            # Test if device supports 16kHz (Parakeet native)
            if self._test_device_sample_rate(16000):
                logger.info(
                    "🟢 OPTIMAL: Device supports 16kHz natively - no ASR resampling needed!"
                )
                return 16000

            # Device doesn't support 16kHz, find best alternative
            # Priority: minimize total resampling operations

            # If device is at a higher rate, we can downsample to 16kHz in ASR
            # This is better than upsampling device to higher rate then downsampling
            if device_native_rate >= 16000:
                logger.info(
                    f"🟡 GOOD: Using device native {device_native_rate}Hz → single downsample to 16kHz for ASR"
                )
                return device_native_rate

            # Device is below 16kHz - we need to upsample for ASR anyway
            # Use configured rate or a reasonable minimum
            logger.warning(
                f"🟡 SUBOPTIMAL: Device only supports {device_native_rate}Hz - requires upsampling for ASR"
            )

            # Use configured rate if reasonable, otherwise use a common rate
            if self.configured_sample_rate >= 16000:
                return self.configured_sample_rate
            else:
                logger.warning(
                    f"🟡 Configured rate {self.configured_sample_rate}Hz too low, using 44100Hz"
                )
                return 44100

        except Exception as e:
            logger.error(f"Sample rate optimization failed: {e}")
            logger.info(
                f"🟡 Falling back to configured rate: {self.configured_sample_rate}Hz"
            )
            return self.configured_sample_rate

    def _test_device_sample_rate(self, test_rate: int) -> bool:
        """Test if device supports a specific sample rate.

        Args:
            test_rate: Sample rate to test

        Returns:
            True if device supports the sample rate
        """
        try:
            # Try to create a test stream (don't start it)
            test_stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=test_rate,
                blocksize=64,  # Small block for quick test
                dtype=np.float32,
            )
            test_stream.close()
            return True
        except Exception:
            return False

    def _report_optimization_benefits(self) -> None:
        """Report the benefits of sample rate optimization."""
        try:
            if self.device is None:
                device_info = sd.query_devices(kind="input")
            else:
                device_info = sd.query_devices(self.device)

            device_native_rate = int(device_info["default_samplerate"])

            # Calculate resampling operations
            if self.sample_rate == self.parakeet_native_rate:
                performance = "🟢 OPTIMAL"
                detail = "No resampling needed - maximum quality and performance"
            elif device_native_rate == self.sample_rate:
                performance = "🟢 EXCELLENT"
                detail = f"Single downsample: {self.sample_rate}Hz → 16kHz for ASR"
            else:
                performance = "🟡 SUBOPTIMAL"
                detail = f"Double resample: {device_native_rate}Hz → {self.sample_rate}Hz → 16kHz"

            logger.info(f"{performance} Pipeline: {detail}")

            if self.rate_auto_optimized:
                logger.info(
                    f"🔧 Auto-optimized from {self.configured_sample_rate}Hz to {self.sample_rate}Hz"
                )

        except Exception as e:
            logger.debug(f"Could not report optimization benefits: {e}")

    def _validate_sample_rate_compatibility(self) -> None:
        """Validate and report on sample rate compatibility."""
        try:
            # Get device info for current device
            if self.device is None:
                device_info = sd.query_devices(kind="input")
            else:
                device_info = sd.query_devices(self.device)

            device_sample_rate = device_info["default_samplerate"]

            if device_sample_rate != self.sample_rate:
                # This is now expected and handled by optimization
                logger.debug(
                    f"Device native: {device_sample_rate}Hz, Using: {self.sample_rate}Hz (optimized)"
                )
            else:
                logger.info(
                    f"🟢 Perfect match: Device and capture both use {self.sample_rate}Hz"
                )

        except Exception as e:
            logger.error(f"Sample rate validation failed: {e}")

    def _validate_and_fix_device_configuration(self) -> None:
        """Validate configured device supports input and fix if needed."""
        if self.device is None:
            logger.debug("Using default device (no validation needed)")
            return

        try:
            # Check if configured device exists and supports input
            device_info = sd.query_devices(self.device)

            if device_info["max_input_channels"] == 0:
                logger.warning(
                    f"🟡 Configured device {self.device} ('{device_info['name']}') is output-only, "
                    "falling back to default device"
                )
                self.device = None
            else:
                logger.info(
                    f"🟢 Validated device {self.device}: '{device_info['name']}' "
                    f"({device_info['max_input_channels']} input channels)"
                )

        except Exception as e:
            logger.warning(
                f"🟡 Configured device {self.device} is invalid ({e}), "
                "falling back to default device"
            )
            self.device = None

    def validate_audio_data(self, audio_data: np.ndarray) -> bool:
        """Validate audio data quality before processing.

        Args:
            audio_data: Audio data to validate.

        Returns:
            True if audio data is valid, False otherwise.
        """
        if audio_data.size == 0:
            logger.warning("🟡 Audio validation failed: Empty audio data")
            return False

        # Calculate audio statistics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        duration = len(audio_data) / self.sample_rate

        # Check for silence (very low RMS)
        if rms < 0.001:
            logger.warning(
                f"🟡 Audio validation warning: Very low RMS {rms:.6f} - possible silence or wrong microphone"
            )
            return False

        # Check for clipping
        clipped_samples = np.sum(np.abs(audio_data) > 0.99)
        if clipped_samples > len(audio_data) * 0.01:  # More than 1% clipped
            logger.warning(
                f"🟡 Audio validation warning: Clipping detected in {clipped_samples} samples ({clipped_samples / len(audio_data) * 100:.1f}%)"
            )

        # Log audio quality metrics
        logger.info(
            f"🟢 Audio validation: Duration={duration:.2f}s, RMS={rms:.4f}, Peak={peak:.4f}"
        )

        # Save debug audio file if enabled
        if self.debug_save_audio:
            self._save_debug_audio(audio_data)

        return True

    def _save_debug_audio(self, audio_data: np.ndarray) -> None:
        """Save audio data for debugging purposes with automatic cleanup.

        Args:
            audio_data: Audio data to save.
        """
        try:
            from src.utils.audio_debug import AudioDebugManager

            debug_manager = AudioDebugManager(self.debug_audio_dir)
            debug_manager.save_debug_audio(audio_data, self.sample_rate)

        except Exception as e:
            logger.error(f"🛑 Failed to save debug audio: {e}")

    def _should_stop_due_to_limits(self) -> bool:
        """Check if recording should stop due to memory or duration limits.
        
        Returns:
            True if recording should stop due to limits.
        """
        import time

        # Check duration limit
        if self.max_recording_duration > 0:
            current_duration = time.time() - self.recording_start_time
            if current_duration > self.max_recording_duration:
                logger.warning(f"🟡 Recording duration limit reached: {current_duration:.1f}s")
                return True

        # Check buffer size limit
        if self.buffer_size_limit > 0:
            current_size_mb = self._total_audio_size / (1024 * 1024)
            if current_size_mb > self.buffer_size_limit:
                logger.warning(f"🟡 Audio buffer size limit reached: {current_size_mb:.1f}MB")
                return True

        return False

    def _check_memory_usage(self) -> None:
        """Check and log current memory usage."""
        if not self.enable_buffer_monitoring:
            return

        import time
        current_time = time.time()

        # Only check every 10 seconds to avoid overhead
        if current_time - self._last_memory_check < 10:
            return

        self._last_memory_check = current_time

        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            duration = current_time - self.recording_start_time
            buffer_size_mb = self._total_audio_size / (1024 * 1024)

            logger.debug(f"📊 Recording stats: {duration:.1f}s, {buffer_size_mb:.1f}MB buffer, {memory_info.rss/1024/1024:.1f}MB RSS")

            # Warn if approaching limits
            if self.max_recording_duration > 0 and duration > self.max_recording_duration * 0.8:
                logger.warning(f"🟡 Approaching recording duration limit: {duration:.1f}s / {self.max_recording_duration}s")

            if self.buffer_size_limit > 0 and buffer_size_mb > self.buffer_size_limit * 0.8:
                logger.warning(f"🟡 Approaching buffer size limit: {buffer_size_mb:.1f}MB / {self.buffer_size_limit}MB")

        except ImportError:
            logger.debug("psutil not available for memory monitoring")

    def get_memory_stats(self) -> dict:
        """Get current memory and buffer statistics.
        
        Returns:
            Dictionary with memory stats.
        """
        import time

        duration = time.time() - self.recording_start_time if self.recording else 0
        buffer_size_mb = self._total_audio_size / (1024 * 1024)

        stats = {
            "recording": self.recording,
            "duration_seconds": duration,
            "buffer_size_mb": buffer_size_mb,
            "audio_chunks": len(self.audio_data),
            "queue_size": self.audio_queue.qsize(),
            "max_duration": self.max_recording_duration,
            "max_buffer_mb": self.buffer_size_limit,
        }

        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            stats.update({
                "process_rss_mb": memory_info.rss / (1024 * 1024),
                "process_vms_mb": memory_info.vms / (1024 * 1024),
            })
        except ImportError:
            pass

        return stats

    def clear_audio_buffers(self) -> None:
        """Clear audio buffers and reset counters."""
        logger.info("🧹 Clearing audio buffers")

        # Clear buffers
        self.audio_data.clear()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Reset counters
        self._total_audio_size = 0

        # Force garbage collection
        import gc
        gc.collect()

        logger.debug("Audio buffers cleared")


# end src/audio/recorder.py
