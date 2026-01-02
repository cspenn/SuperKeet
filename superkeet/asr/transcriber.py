# start src/asr/transcriber.py
"""ASR transcription using Parakeet-MLX."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from parakeet_mlx import from_pretrained

from superkeet.config.config_loader import config
from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class ASRTranscriber:
    """Handles speech-to-text transcription using Parakeet-MLX with memory
    management."""

    def __init__(self) -> None:
        """Initialize the ASR transcriber."""
        # Updated to Parakeet v3 model
        self.model_id = config.get("asr.asr_model", "nvidia/parakeet-tdt-0.6b-v3")
        self.device = config.get("asr.device", "mps")
        self.model = None
        # Get expected sample rate from audio recorder configuration
        self.expected_sample_rate = config.get("audio.sample_rate", 16000)
        self.parakeet_native_rate = 16000  # Parakeet's native expected rate

        # Memory management settings
        self.auto_unload_timeout = config.get(
            "asr.auto_unload_timeout", 300
        )  # 5 minutes
        self.last_used_time = 0
        self._memory_monitor_timer = None

        logger.info(f"ASRTranscriber initialized with model: {self.model_id}")
        logger.info(
            f"Expected input: {self.expected_sample_rate}Hz, "
            f"Parakeet native: {self.parakeet_native_rate}Hz"
        )
        logger.info(f"Auto-unload timeout: {self.auto_unload_timeout}s")

        # Report resampling strategy
        self._report_resampling_strategy()

        # Start memory monitoring if auto-unload is enabled
        if self.auto_unload_timeout > 0:
            self._start_memory_monitor()

    def _start_memory_monitor(self) -> None:
        """Start memory monitoring timer for auto-unloading."""
        try:
            from PySide6.QtCore import QTimer

            if self._memory_monitor_timer is None:
                self._memory_monitor_timer = QTimer()
                self._memory_monitor_timer.timeout.connect(self._check_auto_unload)
                self._memory_monitor_timer.start(60000)  # Check every minute
                logger.debug("Memory monitor started for ASR model auto-unload")
        except ImportError:
            logger.debug("PySide6 not available, memory monitor disabled")

    def _check_auto_unload(self) -> None:
        """Check if model should be auto-unloaded due to inactivity."""
        if self.model is not None and self.auto_unload_timeout > 0:
            import time

            current_time = time.time()
            if current_time - self.last_used_time > self.auto_unload_timeout:
                logger.info(
                    f"Auto-unloading ASR model after "
                    f"{self.auto_unload_timeout}s of inactivity"
                )
                self.unload_model()

    def _update_last_used(self) -> None:
        """Update the last used timestamp."""
        import time

        self.last_used_time = time.time()

    def _get_cache_dir(self) -> str:
        """Get the Hugging Face cache directory.

        Returns:
            String path to the cache directory.
        """
        # Check environment variables in order of priority
        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.environ.get("HF_HOME")
        )

        if cache_dir:
            return str(Path(cache_dir))

        # Default cache location
        return str(Path.home() / ".cache" / "huggingface" / "hub")

    def _is_model_cached(self) -> bool:
        """Check if the model is already cached locally.

        Returns:
            True if model is cached, False otherwise.
        """
        cache_dir = Path(self._get_cache_dir())
        # Check if any model files exist in cache for this model ID
        model_hash = self.model_id.replace("/", "--")
        model_paths = list(cache_dir.glob(f"models--{model_hash}*"))
        return len(model_paths) > 0

    def get_memory_usage(self) -> dict:
        """Get current memory usage information.

        Returns:
            Dictionary with memory usage stats.
        """
        import gc

        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "model_loaded": self.model is not None,
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "last_used": self.last_used_time,
            "auto_unload_timeout": self.auto_unload_timeout,
            "garbage_collected_objects": len(gc.get_objects()),
        }

    def load_model(self) -> None:
        """Load the Parakeet model into memory with progress indication."""
        if self.model is not None:
            logger.info("Model already loaded in memory")
            self._update_last_used()
            return

        try:
            import gc

            from tqdm import tqdm

            # Force garbage collection before loading large model
            gc.collect()

            # Check if model is cached
            is_cached = self._is_model_cached()

            if is_cached:
                logger.info(f"Loading cached model: {self.model_id}")
                # For cached models, show a brief loading indicator
                with tqdm(
                    total=100, desc="Loading model", bar_format="{desc}: {bar}"
                ) as pbar:
                    self.model = from_pretrained(self.model_id)
                    pbar.update(100)
                logger.info("Model loaded successfully")

            else:
                # Simple, blocking download/load
                logger.info(f"Model not found in cache. Downloading: {self.model_id}")
                logger.info(
                    "This may take a few minutes for the first download (~600MB)"
                )

                try:
                    self.model = from_pretrained(self.model_id)
                    logger.info("Model downloaded and cached successfully")
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download/load model: {e}")
                    raise

            # Update usage timestamp
            self._update_last_used()

            # Log memory usage after loading
            memory_stats = self.get_memory_usage()
            logger.info(
                f"Model loaded - Memory usage: {memory_stats['rss_mb']:.1f}MB RSS"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Please check your internet connection and try again")
            raise

    def transcribe(
        self, audio_data: np.ndarray, input_sample_rate: Optional[int] = None
    ) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Numpy array containing audio samples.
            input_sample_rate: Actual sample rate of the input audio data.
                              If None, uses expected_sample_rate from config.

        Returns:
            Transcribed text string.
        """
        # Auto-reload model if it was unloaded
        if self.model is None:
            logger.info("Model not loaded - auto-reloading for transcription")
            self.load_model()

        # Determine actual input sample rate early for validation
        actual_rate = input_sample_rate or self.expected_sample_rate

        # Validate audio data before processing
        if not self._validate_audio_for_asr(audio_data, actual_rate):
            logger.warning("Audio validation failed - returning empty string")
            return ""

        # Update usage timestamp
        self._update_last_used()

        temp_path = None
        try:
            # Prepare audio file for transcription
            temp_path = self._prepare_audio_for_transcription(audio_data, actual_rate)

            # Transcribe from file
            result = self.model.transcribe(temp_path)

            # Extract text from result
            if hasattr(result, "text"):
                text = result.text
            else:
                text = str(result)

            logger.debug(f"Transcription complete: {text[:50]}...")
            return text.strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
        finally:
            # Robust cleanup of temporary file
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
                except OSError as e:
                    logger.warning(f"Failed to cleanup temporary file {temp_path}: {e}")

    def _validate_audio_for_asr(
        self, audio_data: np.ndarray, input_sample_rate: int
    ) -> bool:
        """Validate audio data before ASR processing.

        Args:
            audio_data: Audio data to validate.
            input_sample_rate: Sample rate of the input audio

        Returns:
            True if audio is valid, False otherwise
        """
        # Check None FIRST
        if audio_data is None:
            logger.warning("Audio data is None")
            return False

        # Check if it's a numpy array
        if not isinstance(audio_data, np.ndarray):
            logger.warning(f"Invalid audio type: {type(audio_data)}")
            return False

        # Check if array is empty
        if audio_data.size == 0:
            logger.warning("Audio array is empty")
            return False

        # NOW safe to do math operations
        # Calculate audio statistics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        duration = len(audio_data) / input_sample_rate

        # Check for silence (very low RMS)
        if rms < 0.001:
            logger.warning(
                f"🟡 ASR validation warning: Very low RMS {rms:.6f} - "
                f"possible silence or wrong microphone"
            )

        # Check for clipping
        clipped_samples = np.sum(np.abs(audio_data) > 0.99)
        if clipped_samples > len(audio_data) * 0.01:  # More than 1% clipped
            logger.warning(
                f"🟡 ASR validation warning: Audio clipping detected in "
                f"{clipped_samples} samples "
                f"({clipped_samples / len(audio_data) * 100:.1f}%)"
            )

        # Log audio quality metrics for ASR
        logger.debug(
            f"🟢 ASR audio validation: Duration={duration:.2f}s, RMS={rms:.4f}, Peak={peak:.4f}"  # noqa: E501
        )

        return True

    def _report_resampling_strategy(self) -> None:
        """Report the resampling strategy for this ASR configuration."""
        if self.expected_sample_rate == self.parakeet_native_rate:
            logger.info(
                f"🟢 OPTIMAL: Expected input matches Parakeet native rate ({self.parakeet_native_rate}Hz)"  # noqa: E501
            )
            logger.info("🟢 No resampling overhead - maximum performance and quality")
        else:
            logger.info(
                f"🟡 Expected input: {self.expected_sample_rate}Hz, will resample to {self.parakeet_native_rate}Hz"  # noqa: E501
            )
            logger.info("🟡 Single resampling operation - good performance")

    def _resample_audio(
        self, audio_data: np.ndarray, input_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio data from input rate to target sample rate.

        Args:
            audio_data: Input audio data
            input_rate: Input sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio data
        """
        if input_rate == target_rate:
            return audio_data

        try:
            # Use high-quality resampling when available
            try:
                from scipy.signal import resample

                # Calculate new length
                new_length = int(len(audio_data) * target_rate / input_rate)
                resampled = resample(audio_data, new_length)
                logger.debug(
                    f"HQ resample: {len(audio_data)} samples @ {input_rate}Hz → {len(resampled)} samples @ {target_rate}Hz"  # noqa: E501
                )
                return resampled.astype(np.float32)
            except ImportError:
                # Fallback to linear interpolation
                logger.debug(
                    "Using linear interpolation resampling (scipy unavailable)"
                )
                ratio = target_rate / input_rate
                new_length = int(len(audio_data) * ratio)
                resampled = np.interp(
                    np.linspace(0, len(audio_data) - 1, new_length),
                    np.arange(len(audio_data)),
                    audio_data,
                )
                return resampled.astype(np.float32)
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            logger.warning(
                "Using original audio without resampling - may cause transcription errors"  # noqa: E501
            )
            return audio_data

    def _prepare_audio_for_transcription(
        self, audio_data: np.ndarray, actual_rate: int
    ) -> str:
        """Prepare audio data for transcription by resampling and saving to temp file.

        Args:
            audio_data: Input audio data.
            actual_rate: Input sample rate.

        Returns:
            Path to the temporary audio file.
        """
        duration = len(audio_data) / actual_rate
        logger.debug(f"🐛 Transcribing audio: {duration:.2f}s @ {actual_rate}Hz")

        # Optimize resampling strategy
        if actual_rate == self.parakeet_native_rate:
            # OPTIMAL: No resampling needed!
            logger.debug("🟢 OPTIMAL: Audio already at 16kHz - no resampling needed")
            audio_data = audio_data.astype(np.float32)
            target_sample_rate = self.parakeet_native_rate
        elif actual_rate != self.parakeet_native_rate:
            # Resample to Parakeet native rate
            target_rate = self.parakeet_native_rate
            logger.debug(f"🟡 Resampling audio from {actual_rate}Hz to {target_rate}Hz")
            audio_data = self._resample_audio(
                audio_data, actual_rate, self.parakeet_native_rate
            )
            target_sample_rate = self.parakeet_native_rate
        else:
            target_sample_rate = actual_rate

        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio to [-1, 1] range if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val

        # Save audio to temporary file using target sample rate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, target_sample_rate)
            logger.debug(f"Saved audio to temporary file: {temp_path}")

        return temp_path

    def unload_model(self) -> None:
        """Unload the model from memory with garbage collection."""
        if self.model is not None:
            import gc

            # Log memory before unloading
            memory_before = self.get_memory_usage()

            self.model = None

            # Force garbage collection
            gc.collect()

            # Log memory after unloading
            memory_after = self.get_memory_usage()
            memory_freed = memory_before["rss_mb"] - memory_after["rss_mb"]

            logger.info(f"Model unloaded - Memory freed: {memory_freed:.1f}MB")

    def cleanup(self) -> None:
        """Clean up resources including model and timers."""
        if self._memory_monitor_timer is not None:
            self._memory_monitor_timer.stop()
            self._memory_monitor_timer = None
            logger.debug("Memory monitor timer stopped")

        self.unload_model()


# end src/asr/transcriber.py
