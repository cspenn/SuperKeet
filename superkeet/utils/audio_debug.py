# start src/utils/audio_debug.py
"""Audio debugging utilities for SuperKeet application."""

from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from superkeet.utils.file_manager import FileManager
from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioDebugManager:
    """Manages audio debugging functionality including file saving and cleanup."""

    def __init__(self, debug_directory: str = "debug_audio"):
        """Initialize audio debug manager.

        Args:
            debug_directory: Directory to save debug audio files
        """
        self.debug_directory = debug_directory
        self.debug_dir = Path(debug_directory)  # Add this for test compatibility
        self.file_manager = FileManager(debug_directory)

    def save_debug_audio(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Save audio data for debugging purposes with automatic cleanup.

        Args:
            audio_data: Audio data to save
            sample_rate: Audio sample rate

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate input parameters
            if audio_data.size == 0:
                logger.warning("🟡 Cannot save empty audio data")
                return False

            if sample_rate <= 0:
                logger.warning(f"🟡 Invalid sample rate: {sample_rate}")
                return False
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"debug_audio_{timestamp}.wav"
            filepath = Path(self.debug_directory) / filename

            # Ensure debug directory exists
            Path(self.debug_directory).mkdir(exist_ok=True)

            # Ensure audio is in correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize if needed to prevent clipping
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            sf.write(str(filepath), audio_data, sample_rate)
            logger.info(f"🟢 Debug audio saved: {filepath}")

            # Perform cleanup based on configuration
            self._perform_cleanup_if_needed()

            return True

        except Exception as e:
            logger.error(f"🛑 Failed to save debug audio: {e}")
            return False

    def _perform_cleanup_if_needed(self) -> None:
        """Perform cleanup of debug files if thresholds are exceeded."""
        try:
            # Get cleanup settings from config
            from superkeet.config.config_loader import config

            retention_days = config.get("debug.retention_days", 7)
            max_files = config.get("debug.max_files", 50)
            max_size_mb = config.get("debug.max_size_mb", 100.0)

            # Check if cleanup is needed
            if self.file_manager.check_size_threshold(max_size_mb):
                cleaned_count = self.file_manager.cleanup_old_files(
                    retention_days=retention_days,
                    max_files=max_files,
                    file_pattern="debug_audio_*.wav",
                )
                if cleaned_count > 0:
                    logger.info(f"🧹 Cleaned up {cleaned_count} old debug audio files")

            # Notify user if thresholds are still exceeded
            self.file_manager.notify_user_if_threshold_exceeded(max_size_mb, max_files)

        except Exception as e:
            logger.warning(f"⚠️ Debug cleanup failed: {e}")

    def get_debug_info(self) -> dict:
        """Get information about debug files.

        Returns:
            Dictionary with debug file information
        """
        try:
            total_size = self.file_manager.get_directory_size("debug_audio_*.wav")
            file_count = self.file_manager.get_file_count("debug_audio_*.wav")
            file_info = self.file_manager.get_file_info("debug_audio_*.wav")

            return {
                "directory": self.debug_directory,
                "total_size_mb": total_size / (1024 * 1024),
                "file_count": file_count,
                "files": file_info[:5],  # Show only first 5 files
            }

        except Exception as e:
            logger.error(f"🛑 Failed to get debug info: {e}")
            return {
                "directory": self.debug_directory,
                "total_size_mb": 0,
                "file_count": 0,
                "files": [],
            }

    def _generate_debug_filename(self) -> str:
        """Generate a debug filename with timestamp.

        Returns:
            Debug filename string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"debug_audio_{timestamp}.wav"


# end src/utils/audio_debug.py
