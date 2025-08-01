# start src/asr/transcriber.py
"""ASR transcription using Parakeet-MLX."""

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from parakeet_mlx import from_pretrained

from src.config.config_loader import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ASRTranscriber:
    """Handles speech-to-text transcription using Parakeet-MLX."""

    def __init__(self) -> None:
        """Initialize the ASR transcriber."""
        self.model_id = config.get("asr.model_id", "mlx-community/parakeet-tdt-0.6b-v2")
        self.device = config.get("asr.device", "mps")
        self.model = None
        self.sample_rate = 16000  # Parakeet expects 16kHz

        logger.info(f"ASRTranscriber initialized with model: {self.model_id}")

    def _get_cache_dir(self) -> Path:
        """Get the Hugging Face cache directory.

        Returns:
            Path to the cache directory.
        """
        # Check environment variables in order of priority
        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.environ.get("HF_HOME")
        )

        if cache_dir:
            return Path(cache_dir)

        # Default cache location
        return Path.home() / ".cache" / "huggingface" / "hub"

    def _is_model_cached(self) -> bool:
        """Check if the model is already cached locally.

        Returns:
            True if model is cached, False otherwise.
        """
        cache_dir = self._get_cache_dir()
        # Check if any model files exist in cache for this model ID
        model_hash = self.model_id.replace("/", "--")
        model_paths = list(cache_dir.glob(f"models--{model_hash}*"))
        return len(model_paths) > 0

    def load_model(self) -> None:
        """Load the Parakeet model into memory."""
        if self.model is not None:
            logger.info("Model already loaded in memory")
            return

        try:
            # Check if model is cached
            is_cached = self._is_model_cached()

            if is_cached:
                logger.info(f"Loading cached model: {self.model_id}")
            else:
                logger.info(f"Model not found in cache. Downloading: {self.model_id}")
                logger.info(
                    "This may take a few minutes for the first download (~600MB)"
                )

            # Load model (will download if not cached)
            self.model = from_pretrained(self.model_id)

            if not is_cached:
                logger.info("Model downloaded and cached successfully")

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Please check your internet connection and try again")
            raise

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Numpy array containing audio samples.

        Returns:
            Transcribed text string.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if audio_data.size == 0:
            logger.warning("Empty audio data provided")
            return ""

        try:
            logger.debug(
                f"Transcribing audio: {len(audio_data) / self.sample_rate:.2f}s"
            )

            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio to [-1, 1] range if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_data, self.sample_rate)
                logger.debug(f"Saved audio to temporary file: {temp_path}")

            try:
                # Transcribe from file
                result = self.model.transcribe(temp_path)

                # Extract text from result
                if hasattr(result, "text"):
                    text = result.text
                else:
                    text = str(result)

                logger.debug(f"Transcription complete: {text[:50]}...")
                return text.strip()
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"Cleaned up temporary file: {temp_path}")

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            self.model = None
            logger.info("Model unloaded")


# end src/asr/transcriber.py
