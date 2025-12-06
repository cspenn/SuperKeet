# start src/audio/recorder.py
"""Audio recording functionality for SuperKeet."""

import queue
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal

from superkeet.config.config_loader import config
from superkeet.utils.logger import setup_logger

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
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=100
        )  # Limit queue size
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
                f"🟢 AudioRecorder optimized: {self.configured_sample_rate}Hz → {self.sample_rate}Hz (device-optimal)"  # noqa: E501
            )
        else:
            logger.info(
                f"AudioRecorder initialized: {self.sample_rate}Hz, {self.channels}ch"
            )

        # Log memory management settings
        logger.info(
            f"Buffer limits: {self.max_recording_duration}s duration, {self.buffer_size_limit}MB size"  # noqa: E501
        )

        # Report optimization benefits
        self._report_optimization_benefits()

        # Debug audio setup on initialization
        self.debug_audio_setup()

        # Validate sample rate compatibility
        self._validate_sample_rate_compatibility()

        # Validate sample rate compatibility
        self._validate_sample_rate_compatibility()

    def _validate_microphone_permissions(self) -> bool:
        """Validate microphone permissions with macOS-specific handling.

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
                return self._test_basic_microphone_access()

            logger.info("🔒 Checking microphone permissions on macOS...")

            # Method 1: Try to access microphone directly
            if not self._test_basic_microphone_access():
                logger.warning("🔒 Basic microphone access test failed")
                return False

            # Method 2: Check system permission status using tccutil (if available)
            try:
                result = subprocess.run(
                    ["tccutil", "list", "Microphone"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    logger.debug("🔒 System microphone permissions checked via tccutil")
                else:
                    logger.debug("🔒 tccutil not available or failed, using basic test")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.debug("🔒 tccutil command not available, using basic test")

            # Method 3: Test with device-appropriate configurations
            # Get device info to determine native channel count
            if self.device is not None:
                try:
                    device_info = sd.query_devices(self.device)
                    test_channels = min(2, int(device_info["max_input_channels"]))
                    logger.debug(
                        f"🔒 Device {self.device} supports {device_info['max_input_channels']} channels, testing with {test_channels}"
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
                        device=self.device,
                        # Removed never_drop_input=True - causes PortAudio -9995 error
                    )
                    test_stream.close()
                    logger.info(
                        f"🟢 Microphone permission validated with config: {config}"
                    )
                    return True
                except Exception as e:
                    logger.debug(f"Permission test failed with config {config}: {e}")
                    continue

            logger.error("🔒 All microphone permission tests failed")
            return False

        except Exception as e:
            logger.error(f"🔒 Microphone permission validation failed: {e}")
            return False

    def _test_basic_microphone_access(self) -> bool:
        """Test basic microphone access without starting recording.

        Returns:
            True if microphone is accessible, False otherwise.
        """
        try:
            # Get device info to determine native channel count
            if self.device is not None:
                device_info = sd.query_devices(self.device)
                test_channels = min(
                    2, int(device_info["max_input_channels"])
                )  # Use device native channels, max 2
            else:
                test_channels = 1  # Default device usually supports mono

            # Try to create a minimal test stream using device's native channel count
            test_stream = sd.InputStream(
                channels=test_channels,
                samplerate=16000,
                blocksize=64,
                dtype=np.float32,
                device=self.device,
                # Using basic parameters only to avoid -9995 errors
            )
            test_stream.close()
            logger.debug("🟢 Basic microphone access test passed")
            return True
        except Exception as e:
            logger.debug(f"🔒 Basic microphone access test failed: {e}")
            return False

    def _try_portaudio_reinit(self) -> bool:
        """Reinitialize PortAudio with macOS-specific parameters and retry recording.

        Returns:
            True if PortAudio reinitialization and recording start succeeded, False otherwise.
        """
        try:
            logger.info("🔄 Attempting PortAudio reinitialization...")

            # Step 1: Clean up any existing stream
            if self.stream:
                try:
                    if self.stream.active:
                        self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logger.debug(f"Error closing existing stream: {e}")
                finally:
                    self.stream = None

            # Step 2: Force PortAudio termination and reinitialization
            try:
                # Terminate current PortAudio instance if possible
                if hasattr(sd, "_terminate"):
                    sd._terminate()
                    logger.debug("🔄 PortAudio terminated")

                # Small delay to allow cleanup
                import time

                time.sleep(0.1)

                # Force reinitialization
                if hasattr(sd, "_initialize"):
                    sd._initialize()
                    logger.debug("🔄 PortAudio reinitialized")

            except Exception as e:
                logger.debug(f"PortAudio reinit attempt (not critical): {e}")

            # Step 3: Try different initialization parameters optimized for macOS
            reinit_strategies = [
                # Strategy 1: macOS preferred rate with larger buffer
                {
                    "device": None,  # Force default device
                    "channels": 1,
                    "samplerate": 44100,  # macOS preferred rate
                    "blocksize": 1024,
                    "dtype": np.float32,
                    # Removed never_drop_input and prime_output_buffers - cause -9995 errors
                },
                # Strategy 2: Lower sample rate with minimal latency
                {
                    "device": None,
                    "channels": 1,
                    "samplerate": 16000,
                    "blocksize": 512,
                    "dtype": np.float32,
                    # Removed never_drop_input - causes -9995 errors
                },
                # Strategy 3: High-quality with larger buffers
                {
                    "device": None,
                    "channels": 1,
                    "samplerate": 48000,
                    "blocksize": 2048,
                    "dtype": np.float32,
                    # Removed never_drop_input - not supported by current PortAudio
                },
            ]

            for i, strategy in enumerate(reinit_strategies, 1):
                try:
                    logger.debug(f"🔄 Trying reinit strategy {i}: {strategy}")

                    # Create stream with strategy parameters
                    self.stream = sd.InputStream(
                        device=strategy["device"],
                        channels=strategy["channels"],
                        samplerate=strategy["samplerate"],
                        blocksize=strategy["blocksize"],
                        callback=self._audio_callback,
                        dtype=strategy["dtype"],
                        # Removed never_drop_input and prime_output_buffers parameters
                        # These cause PortAudio -9995 "Invalid flag" errors
                    )

                    # Try to start the stream
                    self.stream.start()

                    # Update recorder settings to match successful strategy
                    original_device = self.device
                    original_sample_rate = self.sample_rate
                    original_channels = self.channels
                    original_chunk_size = self.chunk_size

                    self.device = strategy["device"]
                    self.sample_rate = strategy["samplerate"]
                    self.channels = strategy["channels"]
                    self.chunk_size = strategy["blocksize"]
                    self.recording = True

                    logger.info(
                        f"🟢 PortAudio reinitialization successful with strategy {i}"
                    )
                    logger.info(
                        f"🔧 Audio settings updated: {original_sample_rate}Hz→{self.sample_rate}Hz, "
                        f"device {original_device}→{self.device}, "
                        f"{original_channels}ch→{self.channels}ch, "
                        f"blocksize {original_chunk_size}→{self.chunk_size}"
                    )

                    return True

                except Exception as e:
                    logger.debug(f"Reinit strategy {i} failed: {e}")
                    if self.stream:
                        try:
                            self.stream.close()
                        except Exception:
                            pass
                        self.stream = None
                    continue

            logger.warning("🔄 All PortAudio reinitialization strategies failed")
            return False

        except Exception as e:
            logger.error(f"🔄 PortAudio reinitialization failed: {e}")
            return False

            logger.debug(f"🍎 Core Audio direct access failed: {e}")
            return False

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback function for audio stream with memory management and channel conversion.

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

            # Handle channel conversion if needed (e.g., stereo device -> mono recording)
            if audio_copy.ndim > 1 and audio_copy.shape[1] > self.channels:
                if self.channels == 1:
                    # Convert stereo to mono by averaging channels
                    audio_copy = np.mean(audio_copy, axis=1, keepdims=True)
                else:
                    # Take only the required number of channels
                    audio_copy = audio_copy[:, : self.channels]

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

    def start(self) -> bool:
        """Start recording audio with enhanced error recovery.

        Returns:
            True if recording started successfully, False otherwise.
        """
        if self.recording:
            logger.warning("Already recording")
            return False

        # Phase 1: Permission validation
        logger.info("🔒 Validating microphone permissions...")
        if not self._validate_microphone_permissions():
            logger.error("🔒 Microphone permissions validation failed")
        if not self._validate_microphone_permissions():
            logger.error("🔒 Microphone permissions validation failed")
            # Don't suggest complex fixes, just fail gracefully
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
            import time

            self.recording_start_time = time.time()

            # Determine appropriate channel count for the selected device
            stream_channels = self.channels  # Default to configured channels
            if self.device is not None:
                try:
                    device_info = sd.query_devices(self.device)
                    device_max_channels = int(device_info["max_input_channels"])
                    # Use device's native channel count, but limit to what we need
                    if device_max_channels >= 2 and self.channels == 1:
                        # Device is stereo but we want mono - record in stereo and convert in callback
                        stream_channels = min(2, device_max_channels)
                        logger.debug(
                            f"🔧 Using {stream_channels} channels for device {self.device} (will convert to {self.channels})"
                        )
                    else:
                        stream_channels = min(self.channels, device_max_channels)
                except Exception as e:
                    logger.debug(
                        f"Could not query device info: {e}, using configured channels"
                    )
                    stream_channels = self.channels

            # Create and start stream with appropriate channel count
            self.stream = sd.InputStream(
                device=self.device,
                channels=stream_channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32,
                # Using only basic, compatible PortAudio parameters
            )

            self.stream.start()
            self.recording = True
            logger.info("🟢 Started recording successfully")
            return True

        except Exception as e:
            # Enhanced error recovery with macOS-specific handling
            error_msg = str(e).lower()
            logger.error(f"❌ Failed to start recording with device {self.device}: {e}")

            # Detailed error analysis and logging
            if "paerrorcode -9986" in error_msg:
                logger.error(
                    "🚫 PortAudio Error -9986: Internal PortAudio error (macOS compatibility issue)"
                )
                self._log_detailed_error_info(e, "PortAudio -9986")

                # Attempt specific -9986 error recovery
                logger.info("🔧 Attempting PortAudio -9986 specific recovery...")
                if self._handle_portaudio_9986_error(self.device):
                    logger.info("🟢 PortAudio -9986 recovery successful")
                    return True

            elif "paerrorcode -9988" in error_msg:
                logger.error(
                    "🚫 PortAudio Error -9988: Invalid device - device may have been disconnected"
                )
                self._log_detailed_error_info(e, "PortAudio -9988")
            elif "audio unit: invalid property value" in error_msg:
                logger.error("🚫 macOS Audio Unit error: Invalid property value")
                self._log_detailed_error_info(e, "Audio Unit Error")

                # Attempt Audio Unit error recovery
                logger.info("🔧 Attempting Audio Unit error recovery...")
                if self._handle_audio_unit_error(self.device):
                    logger.info("🟢 Audio Unit error recovery successful")
                    return True
            else:
                logger.error("🚫 Unknown audio error")
                self._log_detailed_error_info(e, "Unknown")

            # Enhanced recovery strategy with multiple phases
            logger.warning("🔧 Starting enhanced error recovery...")

            # Phase 1: Permission recheck (might have changed)
            logger.info("🔒 Phase 1: Rechecking permissions...")
            if not self._validate_microphone_permissions():
                logger.error(
                    "🔒 Permission recheck failed - cannot proceed with recovery"
                )
                self._suggest_audio_fixes()
                return self._cleanup_failed_start()

            # Phase 2: PortAudio reinitialization strategy
            logger.info("🔄 Phase 2: Attempting PortAudio reinitialization...")
            if self._try_portaudio_reinit():
                logger.info("🟢 Recovery successful via PortAudio reinitialization")
                return True

            # Phase 3: Device-specific recovery
            logger.info("🔄 Phase 3: Attempting device-specific recovery...")
            if self.device is not None:
                # Check if this is a RODECaster Pro device
                if self._is_rodecast_device(self.device):
                    logger.info(
                        "🎤 Detected RODECaster Pro - attempting specific recovery..."
                    )
                    if self._handle_rodecast_pro_errors(self.device):
                        logger.info("🟢 RODECaster Pro recovery successful")
                        return True

                logger.warning("Trying fallback strategies for specific device...")

                # Strategy 1: Try default device
                if self._try_fallback_device(None):
                    logger.info("🟢 Recovery successful via default device fallback")
                    return True

                # Strategy 2: Try alternative sample rates
                if self._try_fallback_sample_rates():
                    logger.info("🟢 Recovery successful via sample rate fallback")
                    return True

                # Strategy 3: Try alternative configurations
                if self._try_alternative_configs():
                    logger.info("🟢 Recovery successful via alternative configuration")
                    return True

            logger.error("❌ Failed to start recording after recovery attempts")
            self._suggest_audio_fixes()
            return self._cleanup_failed_start()

    def _log_detailed_error_info(self, error: Exception, error_type: str) -> None:
        """Log detailed error information for debugging.

        Args:
            error: The exception that occurred
            error_type: Type/category of error for classification
        """
        import platform
        import sys

        logger.error("🔍 DETAILED ERROR ANALYSIS:")
        logger.error(f"   Error Type: {error_type}")
        logger.error(f"   Error Message: {str(error)}")
        logger.error(f"   Platform: {platform.system()} {platform.release()}")
        logger.error(f"   Python Version: {sys.version}")
        logger.error(f"   Configured Device: {self.device}")
        logger.error(f"   Sample Rate: {self.sample_rate}")
        logger.error(f"   Channels: {self.channels}")
        logger.error(f"   Chunk Size: {self.chunk_size}")

        # Get current device info if possible
        try:
            if self.device is None:
                device_info = sd.query_devices(kind="input")
                logger.error(f"   Default Input Device: {device_info['name']}")
            else:
                device_info = sd.query_devices(self.device)
                logger.error(
                    f"   Selected Device Info: {device_info['name']} ({device_info['max_input_channels']} channels)"
                )
        except Exception as device_error:
            logger.error(f"   Device Query Failed: {device_error}")

    def _cleanup_failed_start(self) -> bool:
        """Clean up after a failed start attempt.

        Returns:
            False (always, to indicate failure)
        """
        # Reset any partially initialized state
        self.recording = False
        if self.stream:
            try:
                self.stream.close()
            except Exception as cleanup_error:
                logger.debug(f"Error during cleanup: {cleanup_error}")
            self.stream = None

        # Reset alternative backend flags
        self._alternative_backend = None
        self._alternative_config = None

        return False

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
            logger.info(
                f"🟢 Successfully started recording with {device_name} device (fallback)"  # noqa: E501
            )
            logger.warning(
                f"🟡 Device {original_device} failed, now using {device_name} device"
            )
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

                logger.info(
                    f"🟢 Successfully started recording at {rate}Hz (fallback from {original_rate}Hz)"  # noqa: E501
                )
                logger.warning(
                    f"🟡 Sample rate changed from {original_rate}Hz to {rate}Hz for compatibility"  # noqa: E501
                )
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
            {"channels": 1, "blocksize": 512},  # Mono, smaller buffer
            {"channels": 1, "blocksize": 2048},  # Mono, larger buffer
            {"channels": 2, "blocksize": 1024},  # Stereo, standard buffer
        ]

        original_channels = self.channels
        original_blocksize = self.chunk_size

        for audio_config in configs:
            if (
                audio_config["channels"] == self.channels
                and audio_config["blocksize"] == self.chunk_size
            ):
                continue  # Already tried this config

            try:
                logger.debug(
                    f"Trying config: {audio_config['channels']}ch, {audio_config['blocksize']} blocksize"  # noqa: E501
                )
                self.stream = sd.InputStream(
                    device=self.device,
                    channels=audio_config["channels"],
                    samplerate=self.sample_rate,
                    blocksize=audio_config["blocksize"],
                    callback=self._audio_callback,
                    dtype=np.float32,
                )
                self.stream.start()
                self.recording = True

                self.channels = audio_config["channels"]
                self.chunk_size = audio_config["blocksize"]

                logger.info(
                    f"🟢 Successfully started with alternative config: {self.channels}ch, {self.chunk_size} blocksize"  # noqa: E501
                )
                logger.warning(
                    f"🟡 Audio config changed from {original_channels}ch/{original_blocksize} to {self.channels}ch/{self.chunk_size}"  # noqa: E501
                )
                return True

            except Exception as e:
                logger.debug(f"Config {audio_config} failed: {e}")
                continue

        return False

    def _suggest_audio_fixes(self) -> None:
        """Suggest comprehensive fixes for audio issues with enhanced macOS-specific guidance and diagnostics."""
        import platform

        logger.error("🔧 COMPREHENSIVE AUDIO TROUBLESHOOTING GUIDE:")
        logger.error("=" * 60)

        # Platform-specific suggestions
        system = platform.system()
        if system == "Darwin":  # macOS
            logger.error("🍎 macOS-SPECIFIC FIXES:")
            logger.error("   1. Check microphone permissions:")
            logger.error(
                "      → System Preferences > Security & Privacy > Privacy > Microphone"
            )
            logger.error("      → Ensure SuperKeet is listed and enabled")
            logger.error("   2. Reset microphone permissions:")
            logger.error("      → Remove SuperKeet from the list")
            logger.error("      → Restart SuperKeet to re-trigger permission request")
            logger.error("   3. Check for macOS-specific issues:")
            logger.error("      → Activity Monitor > search 'coreaudio' or 'audio'")
            logger.error("      → Quit any conflicting audio processes")
            logger.error("   4. Try Terminal command:")
            logger.error("      → Run: sudo killall coreaudiod")
            logger.error("      → This restarts the Core Audio daemon")
            logger.error("   5. macOS Big Sur+ specific:")
            logger.error(
                "      → Try running SuperKeet with Rosetta (if on Apple Silicon)"
            )
            logger.error("      → Check for PortAudio compatibility updates")

        elif system == "Linux":
            logger.error("🐧 LINUX-SPECIFIC FIXES:")
            logger.error("   1. Check ALSA/PulseAudio:")
            logger.error("      → Run: arecord -l (list recording devices)")
            logger.error("      → Run: pulseaudio --check (check PulseAudio status)")
            logger.error("   2. Permission issues:")
            logger.error(
                "      → Add user to 'audio' group: sudo usermod -a -G audio $USER"
            )
            logger.error("      → Log out and back in")

        elif system == "Windows":
            logger.error("🪟 WINDOWS-SPECIFIC FIXES:")
            logger.error("   1. Check Windows privacy settings:")
            logger.error("      → Settings > Privacy > Microphone")
            logger.error("      → Enable 'Allow apps to access your microphone'")
            logger.error("   2. Update audio drivers")
            logger.error("   3. Check Windows Sound settings")

        logger.error("")
        logger.error("🔧 GENERAL FIXES (All Platforms):")
        logger.error("   1. Restart the application completely")
        logger.error("   2. Disconnect and reconnect external microphones")
        logger.error("   3. Try a different USB port (for USB microphones)")
        logger.error("   4. Close other audio applications (Zoom, Teams, etc.)")
        logger.error("   5. Test microphone in other applications")
        logger.error("   6. Check system audio settings/levels")
        logger.error("   7. Try different audio device in SuperKeet settings")

        logger.error("")
        logger.error("🔍 DIAGNOSTIC COMMANDS:")
        if system == "Darwin":
            logger.error("   → Run: system_profiler SPAudioDataType")
            logger.error("   → Run: ps aux | grep -i audio")
        elif system == "Linux":
            logger.error("   → Run: arecord -l")
            logger.error("   → Run: lsusb (for USB devices)")
            logger.error("   → Run: cat /proc/asound/cards")

        logger.error("")
        logger.error("⚡ ADVANCED RECOVERY:")
        logger.error("   1. Try running with alternative backend:")
        logger.error("      → The system attempted multiple recovery strategies")
        logger.error("      → Check logs above for backend availability")
        logger.error("   2. If all else fails:")
        logger.error("      → Report issue with full log output")
        logger.error("      → Include system info and error details")

        logger.error("=" * 60)

        # Log current diagnostic info
        self._log_system_audio_diagnostics()

    def _handle_portaudio_9986_error(self, device_index: Optional[int]) -> bool:
        """Handle specific PortAudio -9986 Internal Error scenarios.

        Args:
            device_index: The device that failed, or None for default

        Returns:
            True if recovery succeeded, False otherwise
        """
        logger.error("🔧 HANDLING PortAudio -9986 Internal Error")
        logger.error(
            "   This typically indicates Core Audio/PortAudio interface issues"
        )

        # Step 1: Check Core Audio daemon health
        if not self._verify_coreaudio_daemon_health():
            logger.error("   Core Audio daemon appears unhealthy")
            self._suggest_coreaudio_daemon_restart()
            return False

        # Step 2: Test device accessibility at system level
        if device_index is not None:
            if not self._test_device_system_accessibility(device_index):
                logger.error(f"   Device {device_index} not accessible at system level")
                # Try alternative devices
                return self._try_alternative_input_devices()

        # Step 3: AUHAL-specific recovery
        return self._attempt_auhal_recovery(device_index)

    def _verify_coreaudio_daemon_health(self) -> bool:
        """Verify Core Audio daemon is running and responsive.

        Returns:
            True if Core Audio daemon is healthy, False otherwise.
        """
        import subprocess

        try:
            # Check if coreaudiod is running
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=5
            )
            if "coreaudiod" not in result.stdout:
                logger.error("   ❌ Core Audio daemon (coreaudiod) not found")
                return False

            # Test basic device enumeration
            try:
                devices = sd.query_devices()
                input_device_count = len(
                    [d for d in devices if d["max_input_channels"] > 0]
                )
                if input_device_count == 0:
                    logger.error("   ❌ No input devices accessible via Core Audio")
                    return False

                logger.debug(
                    f"   ✅ Core Audio daemon healthy: {input_device_count} input devices"
                )
                return True

            except Exception as e:
                logger.error(f"   ❌ Core Audio device enumeration failed: {e}")
                return False

        except Exception as e:
            logger.error(f"   ❌ Core Audio health check failed: {e}")
            return False

    def _suggest_coreaudio_daemon_restart(self) -> None:
        """Provide guidance for Core Audio daemon restart."""
        logger.error("🔧 CORE AUDIO DAEMON RECOVERY:")
        logger.error("   1. Restart Core Audio daemon:")
        logger.error("      → sudo killall coreaudiod")
        logger.error("      → System will automatically restart the daemon")
        logger.error("   2. Alternative restart methods:")
        logger.error("      → Restart SuperKeet application")
        logger.error("      → Log out and back in to macOS")
        logger.error("      → Restart macOS (if daemon issues persist)")
        logger.error("   3. Check for conflicting audio processes:")
        logger.error("      → Close Zoom, Teams, Discord, OBS")
        logger.error("      → Disconnect/reconnect USB audio devices")

    def _test_device_system_accessibility(self, device_index: int) -> bool:
        """Test if device is accessible at the system level.

        Args:
            device_index: Device index to test

        Returns:
            True if device is accessible, False otherwise.
        """
        try:
            # Try to query device info
            device_info = sd.query_devices(device_index)
            if device_info["max_input_channels"] == 0:
                logger.error(f"   Device {device_index} has no input channels")
                return False

            # Test basic stream creation (don't start it)
            test_stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=16000,
                blocksize=64,
                dtype=np.float32,
            )
            test_stream.close()
            logger.debug(
                f"   ✅ Device {device_index} system accessibility test passed"
            )
            return True

        except Exception as e:
            logger.debug(f"   Device {device_index} system accessibility failed: {e}")
            return False

    def _try_alternative_input_devices(self) -> bool:
        """Try alternative input devices when current device fails.

        Returns:
            True if an alternative device worked, False otherwise.
        """
        try:
            devices = self.get_devices()
            if not devices:
                logger.error("   No alternative input devices available")
                return False

            logger.info(f"   Trying {len(devices)} alternative input devices...")

            for device in devices:
                device_index = device["index"]
                if device_index == self.device:
                    continue  # Skip the device that failed

                try:
                    logger.debug(f"   Testing device {device_index}: {device['name']}")

                    # Test with basic configuration
                    test_stream = sd.InputStream(
                        device=device_index,
                        channels=1,
                        samplerate=16000,
                        blocksize=1024,
                        callback=self._audio_callback,
                        dtype=np.float32,
                    )
                    test_stream.start()

                    # Update configuration
                    self.stream = test_stream
                    self.device = device_index
                    self.recording = True

                    logger.info(
                        f"   ✅ Alternative device recovery: Using {device['name']}"
                    )
                    return True

                except Exception as e:
                    logger.debug(f"   Alternative device {device_index} failed: {e}")
                    continue

            logger.error("   ❌ All alternative input devices failed")
            return False

        except Exception as e:
            logger.error(f"   Alternative device recovery failed: {e}")
            return False

    def _attempt_auhal_recovery(self, device_index: Optional[int]) -> bool:
        """Attempt recovery from AUHAL (Audio Unit HAL) errors.

        Args:
            device_index: Device to attempt recovery with

        Returns:
            True if AUHAL recovery succeeded, False otherwise.
        """
        logger.info("   🔧 Attempting AUHAL error recovery...")

        # Strategy: Use safe PortAudio parameters only
        safe_configs = [
            {
                "device": device_index,
                "channels": 1,
                "samplerate": 44100,  # macOS preferred rate
                "blocksize": 1024,
                "dtype": np.float32,
                # Removed all advanced parameters that could trigger AUHAL issues
            },
            {
                "device": device_index,
                "channels": 2
                if device_index and self._device_supports_stereo(device_index)
                else 1,
                "samplerate": 48000,
                "blocksize": 2048,  # Larger buffer for stability
                "dtype": np.float32,
            },
            {
                "device": None,  # Force default device
                "channels": 1,
                "samplerate": 16000,
                "blocksize": 1024,
                "dtype": np.float32,
            },
        ]

        for i, config in enumerate(safe_configs, 1):
            try:
                logger.debug(f"   Testing AUHAL recovery config {i}: {config}")

                # Create stream with only basic, AUHAL-safe parameters
                self.stream = sd.InputStream(
                    device=config["device"],
                    channels=config["channels"],
                    samplerate=config["samplerate"],
                    blocksize=config["blocksize"],
                    callback=self._audio_callback,
                    dtype=config["dtype"],
                    # Explicitly avoid: never_drop_input, prime_output_buffers_using_stream_callback
                )

                self.stream.start()

                # Update configuration to match successful recovery
                self.device = config["device"]
                self.sample_rate = config["samplerate"]
                self.channels = config["channels"]
                self.chunk_size = config["blocksize"]
                self.recording = True

                logger.info(f"   ✅ AUHAL recovery successful with config {i}")
                return True

            except Exception as e:
                logger.debug(f"   AUHAL recovery config {i} failed: {e}")
                if self.stream:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                continue

        logger.error("   ❌ All AUHAL recovery strategies failed")
        return False

    def _device_supports_stereo(self, device_index: int) -> bool:
        """Check if device supports stereo input.

        Args:
            device_index: Device index to check

        Returns:
            True if device supports 2+ input channels, False otherwise.
        """
        try:
            device_info = sd.query_devices(device_index)
            return device_info["max_input_channels"] >= 2
        except Exception:
            return False

    def _handle_audio_unit_error(self, device_index: Optional[int]) -> bool:
        """Handle macOS Audio Unit property value errors.

        Args:
            device_index: Device that caused Audio Unit error

        Returns:
            True if recovery succeeded, False otherwise.
        """
        logger.info("   🎧 Handling Audio Unit property error...")

        # Audio Unit errors often indicate parameter incompatibility
        # Try minimal, conservative configurations
        conservative_configs = [
            {
                "device": device_index,
                "channels": 1,  # Mono is most compatible
                "samplerate": 44100,  # Standard rate
                "blocksize": 1024,  # Standard block size
                "dtype": np.float32,
            },
            {
                "device": None,  # Default device
                "channels": 1,
                "samplerate": 16000,
                "blocksize": 512,
                "dtype": np.float32,
            },
        ]

        for i, config in enumerate(conservative_configs, 1):
            try:
                logger.debug(f"   Testing Audio Unit recovery config {i}: {config}")

                self.stream = sd.InputStream(callback=self._audio_callback, **config)
                self.stream.start()

                # Update settings
                self.device = config["device"]
                self.sample_rate = config["samplerate"]
                self.channels = config["channels"]
                self.chunk_size = config["blocksize"]
                self.recording = True

                logger.info(f"   ✅ Audio Unit recovery successful with config {i}")
                return True

            except Exception as e:
                logger.debug(f"   Audio Unit recovery config {i} failed: {e}")
                if self.stream:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                continue

        return False

    def _is_rodecast_device(self, device_index: int) -> bool:
        """Check if device is a RODECaster Pro or similar RODE device.

        Args:
            device_index: Device index to check

        Returns:
            True if device appears to be a RODECaster Pro, False otherwise.
        """
        try:
            device_info = sd.query_devices(device_index)
            device_name = device_info["name"].lower()
            return any(
                keyword in device_name for keyword in ["rodecast", "rode", "rodecaster"]
            )
        except Exception:
            return False

    def _handle_rodecast_pro_errors(self, device_index: int) -> bool:
        """Handle RODECaster Pro specific configuration issues.

        Args:
            device_index: RODECaster Pro device index

        Returns:
            True if recovery succeeded, False otherwise.
        """
        try:
            device_info = sd.query_devices(device_index)
            logger.info(f"🎤 Handling RODECaster Pro: {device_info['name']}")

            # RODECaster Pro specific configurations
            rodecast_configs = [
                # Config 1: Native stereo mode
                {
                    "device": device_index,
                    "channels": 2,
                    "samplerate": 48000,  # RODECaster native rate
                    "blocksize": 1024,
                    "dtype": np.float32,
                },
                # Config 2: Mono mode (convert from stereo)
                {
                    "device": device_index,
                    "channels": 1,
                    "samplerate": 48000,
                    "blocksize": 2048,  # Larger buffer for stability
                    "dtype": np.float32,
                },
                # Config 3: Lower sample rate
                {
                    "device": device_index,
                    "channels": 2,
                    "samplerate": 44100,
                    "blocksize": 1024,
                    "dtype": np.float32,
                },
                # Config 4: Conservative fallback
                {
                    "device": device_index,
                    "channels": 1,
                    "samplerate": 16000,
                    "blocksize": 1024,
                    "dtype": np.float32,
                },
            ]

            for i, config in enumerate(rodecast_configs, 1):
                try:
                    logger.debug(f"   Testing RODECaster config {i}: {config}")

                    self.stream = sd.InputStream(
                        callback=self._audio_callback, **config
                    )
                    self.stream.start()

                    # Update settings to match successful config
                    self.device = config["device"]
                    self.channels = config["channels"]
                    self.sample_rate = config["samplerate"]
                    self.chunk_size = config["blocksize"]
                    self.recording = True

                    logger.info(
                        f"   ✅ RODECaster Pro recovery successful with config {i}"
                    )
                    return True

                except Exception as e:
                    logger.debug(f"   RODECaster config {i} failed: {e}")
                    if self.stream:
                        try:
                            self.stream.close()
                        except Exception:
                            pass
                        self.stream = None
                    continue

            # If all configs failed, provide specific guidance
            logger.error("🎤 RODECaster Pro device configuration failed")
            self._suggest_rodecast_pro_fixes()
            return False

        except Exception as e:
            logger.error(f"RODECaster Pro error handling failed: {e}")
            return False

    def _suggest_rodecast_pro_fixes(self) -> None:
        """Provide specific troubleshooting for RODECaster Pro devices."""
        logger.error("🎤 RODECAST PRO TROUBLESHOOTING:")
        logger.error("   1. Check USB connection:")
        logger.error("      → Disconnect and reconnect USB cable")
        logger.error("      → Try different USB port")
        logger.error("      → Use direct connection (no USB hub)")
        logger.error("   2. Check RODECaster Pro settings:")
        logger.error("      → Ensure device is in USB mode")
        logger.error("      → Check sample rate setting (prefer 48kHz)")
        logger.error("      → Verify channel configuration")
        logger.error("   3. macOS specific:")
        logger.error("      → Check Privacy & Security > Microphone permissions")
        logger.error("      → Restart Core Audio: sudo killall coreaudiod")
        logger.error("   4. Alternative connection:")
        logger.error("      → Try 3.5mm analog connection")
        logger.error("      → Test with different audio application first")
        logger.error("   5. Device firmware:")
        logger.error("      → Check for RODECaster Pro firmware updates")
        logger.error("      → Reset device to factory settings")

    def _log_system_audio_diagnostics(self) -> None:
        """Log comprehensive system audio diagnostics for troubleshooting."""
        import platform

        logger.error("🔬 SYSTEM AUDIO DIAGNOSTICS:")
        logger.error("-" * 40)

        try:
            system = platform.system()

            # Basic system info
            logger.error(f"Platform: {system} {platform.release()}")
            logger.error(f"Architecture: {platform.machine()}")

            # Available audio devices
            try:
                logger.error("Available Audio Input Devices:")
                devices = self.get_devices()
                if devices:
                    for i, device in enumerate(devices):
                        marker = " ← SELECTED" if device["index"] == self.device else ""
                        logger.error(
                            f"   [{device['index']}] {device['name']} "
                            f"({device['channels']} ch @ {device['default_samplerate']}Hz){marker}"
                        )
                else:
                    logger.error("   ❌ No input devices found")
            except Exception as e:
                logger.error(f"   ❌ Device enumeration failed: {e}")

            # Platform-specific diagnostics
            if system == "Darwin":  # macOS
                self._log_macos_audio_diagnostics()
            elif system == "Linux":
                self._log_linux_audio_diagnostics()
            elif system == "Windows":
                self._log_windows_audio_diagnostics()

            # PortAudio/SoundDevice info
            try:
                import sounddevice as sd

                logger.error(f"SoundDevice version: {sd.__version__}")

                # Use safe PortAudio API query
                try:
                    pa_apis = self._safe_query_hostapis()
                    logger.error(f"PortAudio host APIs: {len(pa_apis)}")
                    for api in pa_apis:
                        logger.error(
                            f"   - {api['name']} (devices: {api['device_count']})"
                        )
                except Exception as api_error:
                    logger.error(f"Safe PortAudio API query failed: {api_error}")

            except ImportError:
                logger.error("SoundDevice not available")

        except Exception as diag_error:
            logger.error(f"Diagnostics failed: {diag_error}")

        logger.error("-" * 40)

    def _safe_query_hostapis(self) -> list[dict]:
        """Safely query PortAudio host APIs with fallback for device_count errors.

        Returns:
            List of host API dictionaries with guaranteed device_count field.
        """
        try:
            host_apis = sd.query_hostapis()

            # Fix the device_count access issue found in testing
            safe_apis = []
            for api in host_apis:
                safe_api = api.copy()

                # Handle missing device_count key
                if "device_count" not in safe_api:
                    # Calculate device count manually
                    try:
                        all_devices = sd.query_devices()
                        device_count = len([d for d in all_devices if d is not None])
                        safe_api["device_count"] = device_count
                        logger.debug(
                            f"Fixed missing device_count for {api['name']}: {device_count}"
                        )
                    except Exception:
                        safe_api["device_count"] = 0

                safe_apis.append(safe_api)

            return safe_apis

        except Exception as e:
            logger.error(f"PortAudio host API query failed: {e}")
            # Return minimal fallback info
            return [
                {
                    "name": "Unknown PortAudio Host",
                    "device_count": len(self._fallback_device_enumeration()),
                    "type": -1,
                    "default_input_device": 0,
                    "default_output_device": 1,
                }
            ]

    def _fallback_device_enumeration(self) -> list[dict]:
        """Fallback device enumeration when PortAudio queries fail.

        Returns:
            List of available input device dictionaries.
        """
        try:
            # Try direct sounddevice query
            devices = sd.query_devices()
            return [d for d in devices if d and d.get("max_input_channels", 0) > 0]
        except Exception:
            # Last resort: return empty list
            logger.error("Complete device enumeration failure")
            return []

    def _log_macos_audio_diagnostics(self) -> None:
        """Log macOS-specific audio diagnostics."""
        import subprocess

        logger.error("macOS Audio System:")

        try:
            # Check Core Audio daemon
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                coreaudio_processes = [
                    line
                    for line in result.stdout.split("\n")
                    if "coreaudio" in line.lower()
                ]
                if coreaudio_processes:
                    logger.error("   Core Audio processes:")
                    for process in coreaudio_processes[:3]:  # Limit output
                        logger.error(f"     {process.strip()}")
                else:
                    logger.error("   ⚠️ No Core Audio processes found")
        except Exception:
            logger.error("   Could not check Core Audio processes")

        try:
            # Check system audio info briefly
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType", "-timeout", "5"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Extract key information
                lines = result.stdout.split("\n")
                audio_devices = [
                    line.strip()
                    for line in lines
                    if "Audio (Built-in)" in line or "USB" in line
                ][:5]
                if audio_devices:
                    logger.error("   System audio devices:")
                    for device in audio_devices:
                        logger.error(f"     {device}")
        except Exception:
            logger.error("   Could not get system audio profile")

    def _log_linux_audio_diagnostics(self) -> None:
        """Log Linux-specific audio diagnostics."""
        import subprocess

        logger.error("Linux Audio System:")

        try:
            # Check ALSA cards
            with open("/proc/asound/cards", "r") as f:
                cards = f.read().strip()
                if cards:
                    logger.error("   ALSA cards:")
                    for line in cards.split("\n")[:5]:  # Limit output
                        logger.error(f"     {line.strip()}")
        except Exception:
            logger.error("   Could not read ALSA cards")

        try:
            # Check PulseAudio
            result = subprocess.run(
                ["pulseaudio", "--check"], capture_output=True, timeout=3
            )

            if result.returncode == 0:
                logger.error("   ✅ PulseAudio is running")
            else:
                logger.error("   ❌ PulseAudio not running or available")
        except Exception:
            logger.error("   Could not check PulseAudio status")

    def _log_windows_audio_diagnostics(self) -> None:
        """Log Windows-specific audio diagnostics."""
        logger.error("Windows Audio System:")
        logger.error("   Use Device Manager to check audio devices")
        logger.error("   Check Windows Sound settings for microphone access")

    def test_enhanced_error_recovery(self) -> dict:
        """Test the enhanced error recovery mechanisms without triggering actual errors.

        Returns:
            Dictionary with test results for each recovery component.
        """
        logger.info("🧪 Testing enhanced error recovery mechanisms...")
        test_results = {}

        # Test 1: Permission validation
        logger.info("🧪 Testing permission validation...")
        try:
            permission_result = self._validate_microphone_permissions()
            test_results["permission_validation"] = {
                "status": "✅ PASSED" if permission_result else "⚠️ FAILED",
                "result": permission_result,
                "description": "Microphone permission validation",
            }
        except Exception as e:
            test_results["permission_validation"] = {
                "status": "❌ ERROR",
                "result": False,
                "error": str(e),
                "description": "Microphone permission validation",
            }

        # Test 2: Basic microphone access
        logger.info("🧪 Testing basic microphone access...")
        try:
            access_result = self._test_basic_microphone_access()
            test_results["basic_access"] = {
                "status": "✅ PASSED" if access_result else "⚠️ FAILED",
                "result": access_result,
                "description": "Basic microphone access test",
            }
        except Exception as e:
            test_results["basic_access"] = {
                "status": "❌ ERROR",
                "result": False,
                "error": str(e),
                "description": "Basic microphone access test",
            }

        # Test 3: Alternative backend availability
        logger.info("🧪 Testing alternative backend availability...")
        try:
            backend_results = {}

            # Test PyAudio availability
            try:
                pyaudio_available = self._try_pyaudio_backend()
                backend_results["pyaudio"] = (
                    "✅ Available" if pyaudio_available else "❌ Not Available"
                )
            except Exception as e:
                backend_results["pyaudio"] = f"❌ Error: {str(e)[:50]}"

            # Test Core Audio availability
            try:
                coreaudio_available = self._try_coreaudio_direct()
                backend_results["coreaudio"] = (
                    "✅ Available" if coreaudio_available else "❌ Not Available"
                )
            except Exception as e:
                backend_results["coreaudio"] = f"❌ Error: {str(e)[:50]}"

            # Test system audio units
            try:
                system_available = self._try_system_audio_units()
                backend_results["system_units"] = (
                    "✅ Available" if system_available else "❌ Not Available"
                )
            except Exception as e:
                backend_results["system_units"] = f"❌ Error: {str(e)[:50]}"

            test_results["alternative_backends"] = {
                "status": "✅ TESTED",
                "result": backend_results,
                "description": "Alternative audio backend availability",
            }
        except Exception as e:
            test_results["alternative_backends"] = {
                "status": "❌ ERROR",
                "result": {},
                "error": str(e),
                "description": "Alternative audio backend availability",
            }

        # Test 4: Device enumeration
        logger.info("🧪 Testing device enumeration...")
        try:
            devices = self.get_devices()
            test_results["device_enumeration"] = {
                "status": "✅ PASSED" if devices else "⚠️ NO DEVICES",
                "result": len(devices),
                "description": f"Found {len(devices)} audio input devices",
            }
        except Exception as e:
            test_results["device_enumeration"] = {
                "status": "❌ ERROR",
                "result": 0,
                "error": str(e),
                "description": "Audio device enumeration",
            }

        # Test 5: Sample rate compatibility
        logger.info("🧪 Testing sample rate compatibility...")
        try:
            test_rates = [8000, 16000, 22050, 44100, 48000]
            compatible_rates = []

            for rate in test_rates:
                try:
                    if self._test_device_sample_rate(rate):
                        compatible_rates.append(rate)
                except Exception:
                    pass

            test_results["sample_rate_compatibility"] = {
                "status": "✅ PASSED" if compatible_rates else "⚠️ NONE COMPATIBLE",
                "result": compatible_rates,
                "description": f"Compatible sample rates: {compatible_rates}",
            }
        except Exception as e:
            test_results["sample_rate_compatibility"] = {
                "status": "❌ ERROR",
                "result": [],
                "error": str(e),
                "description": "Sample rate compatibility testing",
            }

        # Test 6: Error logging and diagnostics
        logger.info("🧪 Testing error logging and diagnostics...")
        try:
            # Test diagnostic logging (without triggering actual errors)
            self._log_system_audio_diagnostics()
            test_results["diagnostics"] = {
                "status": "✅ PASSED",
                "result": True,
                "description": "System audio diagnostics logging",
            }
        except Exception as e:
            test_results["diagnostics"] = {
                "status": "❌ ERROR",
                "result": False,
                "error": str(e),
                "description": "System audio diagnostics logging",
            }

        # Summary
        total_tests = len(test_results)
        passed_tests = sum(
            1 for result in test_results.values() if result["status"].startswith("✅")
        )

        logger.info("🧪 TEST SUMMARY:")
        logger.info(f"   Total tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Success rate: {passed_tests / total_tests * 100:.1f}%")

        # Detailed results
        for test_name, result in test_results.items():
            logger.info(f"   {test_name}: {result['status']} - {result['description']}")
            if "error" in result:
                logger.debug(f"     Error: {result['error']}")

        return test_results

    def _detect_portaudio_capabilities(self) -> dict:
        """Detect which PortAudio features are available for future compatibility.

        Returns:
            Dictionary of supported capabilities.
        """
        if hasattr(self, "_pa_capabilities"):
            return self._pa_capabilities

        logger.info("🔍 Detecting PortAudio capabilities...")
        capabilities = {
            "never_drop_input": False,
            "prime_output_buffers_using_stream_callback": False,
            "advanced_timing": False,
            "exclusive_mode": False,
        }

        # Test each capability individually using minimal parameters
        test_configs = [
            {"param_name": "never_drop_input", "param_value": True},
            {
                "param_name": "prime_output_buffers_using_stream_callback",
                "param_value": False,
            },
            {"param_name": "exclusive_mode", "param_value": False},
        ]

        for config in test_configs:
            param_name = config["param_name"]
            param_value = config["param_value"]

            try:
                # Create minimal test stream with the specific parameter
                stream_params = {
                    "channels": 1,
                    "samplerate": 16000,
                    "blocksize": 64,
                    "dtype": np.float32,
                    param_name: param_value,
                }

                test_stream = sd.InputStream(**stream_params)
                test_stream.close()
                capabilities[param_name] = True
                logger.info(f"🟢 PortAudio capability {param_name}: Supported")

            except Exception as e:
                capabilities[param_name] = False
                logger.info(
                    f"🟡 PortAudio capability {param_name}: Not supported ({str(e)[:50]})"
                )

        # Store capabilities for future use
        self._pa_capabilities = capabilities

        # Log summary
        supported_count = sum(1 for supported in capabilities.values() if supported)
        total_count = len(capabilities)
        logger.info(
            f"🔍 PortAudio capabilities detected: {supported_count}/{total_count} advanced features supported"
        )

        return capabilities

    def get_portaudio_capabilities(self) -> dict:
        """Get detected PortAudio capabilities.

        Returns:
            Dictionary of supported capabilities.
        """
        return getattr(self, "_pa_capabilities", self._detect_portaudio_capabilities())

    def _create_stream_with_compatible_params(self, **base_params) -> sd.InputStream:
        """Create stream using only supported PortAudio parameters.

        Args:
            **base_params: Base stream parameters

        Returns:
            InputStream with only compatible parameters.
        """
        # Ensure capabilities are detected
        if not hasattr(self, "_pa_capabilities"):
            self._detect_portaudio_capabilities()

        # Start with base parameters (known to be compatible)
        stream_params = base_params.copy()

        # Add advanced parameters only if supported
        if self._pa_capabilities.get("never_drop_input", False):
            stream_params["never_drop_input"] = True
            logger.debug("🟢 Using never_drop_input parameter")

        if self._pa_capabilities.get(
            "prime_output_buffers_using_stream_callback", False
        ):
            # Only add if explicitly requested and supported
            if "prime_output_buffers_using_stream_callback" in base_params:
                stream_params["prime_output_buffers_using_stream_callback"] = (
                    base_params["prime_output_buffers_using_stream_callback"]
                )
                logger.debug(
                    "🟢 Using prime_output_buffers_using_stream_callback parameter"
                )

        return sd.InputStream(**stream_params)

    def test_portaudio_capability_detection(self) -> dict:
        """Test the PortAudio capability detection system.

        Returns:
            Dictionary with test results.
        """
        logger.info("🧪 Testing PortAudio capability detection...")
        test_results = {}

        try:
            # Test capability detection
            capabilities = self._detect_portaudio_capabilities()
            test_results["capability_detection"] = {
                "status": "✅ PASSED",
                "result": capabilities,
                "description": f"Detected {sum(1 for v in capabilities.values() if v)}/{len(capabilities)} capabilities",
            }

            # Test compatible stream creation
            try:
                test_stream = self._create_stream_with_compatible_params(
                    channels=1, samplerate=16000, blocksize=64, dtype=np.float32
                )
                test_stream.close()
                test_results["compatible_stream_creation"] = {
                    "status": "✅ PASSED",
                    "result": True,
                    "description": "Compatible stream creation works",
                }
            except Exception as e:
                test_results["compatible_stream_creation"] = {
                    "status": "❌ FAILED",
                    "result": False,
                    "error": str(e),
                    "description": "Compatible stream creation failed",
                }

        except Exception as e:
            test_results["capability_detection"] = {
                "status": "❌ ERROR",
                "result": {},
                "error": str(e),
                "description": "Capability detection system failed",
            }

        # Log results
        for test_name, result in test_results.items():
            logger.info(f"🧪 {test_name}: {result['status']} - {result['description']}")
            if "error" in result:
                logger.debug(f"   Error: {result['error']}")

        return test_results

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

    def update_device(self, device_index: Optional[int] = None) -> bool:
        """Update the audio device and re-optimize sample rate.

        Args:
            device_index: Device index to use, or None for default device

        Returns:
            True if device update was successful (and recording restarted if needed),
            False otherwise.
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
                f"🔧 Sample rate optimized for new device: {old_sample_rate}Hz → {self.sample_rate}Hz"  # noqa: E501
            )

        # Debug the new setup
        self.debug_audio_setup()

        # Validate sample rate compatibility with new device
        self._validate_sample_rate_compatibility()

        # Restart recording if it was active
        if was_recording:
            recording_restarted = self.start()
            if not recording_restarted:
                logger.error("Failed to restart recording after device change")
                return False

        logger.info(f"Audio device updated successfully to {device_index}")
        return True

    def debug_audio_setup(self) -> None:
        """Debug current audio setup and available devices."""
        logger.info("=== AUDIO SETUP DEBUG ===")
        logger.info(f"Configured device: {self.device}")
        logger.info(
            f"Sample rate: {self.sample_rate} (configured: {self.configured_sample_rate})"  # noqa: E501
        )
        logger.info(f"Channels: {self.channels}")
        logger.info(f"Chunk size: {self.chunk_size}")
        if self.rate_auto_optimized:
            logger.info("🔧 Rate auto-optimized for better performance")

        try:
            # Query default device
            default_input = sd.query_devices(kind="input")
            logger.info(
                f"Default input device: [{default_input['index']}] {default_input['name']}"  # noqa: E501
            )

            # If specific device configured, show its details
            if self.device is not None:
                try:
                    device_info = sd.query_devices(self.device)
                    logger.info(
                        f"Configured device details: [{self.device}] {device_info['name']}"  # noqa: E501
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
                        f"  [{i}] {device['name']} - {device['max_input_channels']} ch @ {device['default_samplerate']}Hz {recommendation}{is_default}{is_selected}"  # noqa: E501
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
        2. If device supports common rates → select closest to 16kHz that
           minimizes resampling
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
                    "🟢 OPTIMAL: Device supports 16kHz natively - no ASR resampling needed!"  # noqa: E501
                )
                return 16000

            # Device doesn't support 16kHz, find best alternative
            # Priority: minimize total resampling operations

            # If device is at a higher rate, we can downsample to 16kHz in ASR
            # This is better than upsampling device to higher rate then downsampling
            if device_native_rate >= 16000:
                logger.info(
                    f"🟡 GOOD: Using device native {device_native_rate}Hz → single downsample to 16kHz for ASR"  # noqa: E501
                )
                return device_native_rate

            # Device is below 16kHz - we need to upsample for ASR anyway
            # Use configured rate or a reasonable minimum
            logger.warning(
                f"🟡 SUBOPTIMAL: Device only supports {device_native_rate}Hz - requires upsampling for ASR"  # noqa: E501
            )

            # Use configured rate if reasonable, otherwise use a common rate
            if self.configured_sample_rate >= 16000:
                return self.configured_sample_rate
            else:
                logger.warning(
                    f"🟡 Configured rate {self.configured_sample_rate}Hz too low, using 44100Hz"  # noqa: E501
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
                detail = f"Double resample: {device_native_rate}Hz → {self.sample_rate}Hz → 16kHz"  # noqa: E501

            logger.info(f"{performance} Pipeline: {detail}")

            if self.rate_auto_optimized:
                logger.info(
                    f"🔧 Auto-optimized from {self.configured_sample_rate}Hz to {self.sample_rate}Hz"  # noqa: E501
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
                    f"Device native: {device_sample_rate}Hz, Using: {self.sample_rate}Hz (optimized)"  # noqa: E501
                )
            else:
                logger.info(
                    f"🟢 Perfect match: Device and capture both use {self.sample_rate}Hz"  # noqa: E501
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
                    f"🟡 Configured device {self.device} ('{device_info['name']}') is output-only, "  # noqa: E501
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
                f"🟡 Audio validation warning: Very low RMS {rms:.6f} - possible silence or wrong microphone"  # noqa: E501
            )
            return False

        # Check for clipping
        clipped_samples = np.sum(np.abs(audio_data) > 0.99)
        if clipped_samples > len(audio_data) * 0.01:  # More than 1% clipped
            logger.warning(
                f"🟡 Audio validation warning: Clipping detected in {clipped_samples} samples ({clipped_samples / len(audio_data) * 100:.1f}%)"  # noqa: E501
            )

        # Log audio quality metrics
        logger.info(
            f"🟢 Audio validation: Duration={duration:.2f}s, RMS={rms:.4f}, Peak={peak:.4f}"  # noqa: E501
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
            from superkeet.utils.audio_debug import AudioDebugManager

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
                logger.warning(
                    f"🟡 Recording duration limit reached: {current_duration:.1f}s"
                )
                return True

        # Check buffer size limit
        if self.buffer_size_limit > 0:
            current_size_mb = self._total_audio_size / (1024 * 1024)
            if current_size_mb > self.buffer_size_limit:
                logger.warning(
                    f"🟡 Audio buffer size limit reached: {current_size_mb:.1f}MB"
                )
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

            logger.debug(
                f"📊 Recording stats: {duration:.1f}s, {buffer_size_mb:.1f}MB buffer, {memory_info.rss / 1024 / 1024:.1f}MB RSS"  # noqa: E501
            )

            # Warn if approaching limits
            if (
                self.max_recording_duration > 0
                and duration > self.max_recording_duration * 0.8
            ):
                logger.warning(
                    f"🟡 Approaching recording duration limit: {duration:.1f}s / {self.max_recording_duration}s"  # noqa: E501
                )

            if (
                self.buffer_size_limit > 0
                and buffer_size_mb > self.buffer_size_limit * 0.8
            ):
                logger.warning(
                    f"🟡 Approaching buffer size limit: {buffer_size_mb:.1f}MB / {self.buffer_size_limit}MB"  # noqa: E501
                )

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
            stats.update(
                {
                    "process_rss_mb": memory_info.rss / (1024 * 1024),
                    "process_vms_mb": memory_info.vms / (1024 * 1024),
                }
            )
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
