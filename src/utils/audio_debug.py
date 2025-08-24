"""Audio debugging utilities for SuperKeet application."""

from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.file_manager import FileManager
from src.utils.logger import setup_logger

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
            from src.config.config_loader import config

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


class MemoryMonitor:
    """System memory monitoring and management utility."""

    def __init__(self):
        self.logger = setup_logger("MemoryMonitor")
        self._last_gc_time = 0
        self._gc_threshold_mb = 100  # Force GC when memory increases by 100MB
        self._last_memory_check = 0

    def get_memory_info(self) -> dict:
        """Get current memory usage information.

        Returns:
            Dictionary with memory statistics.
        """
        try:
            import gc

            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "gc_objects": len(gc.get_objects()),
                "gc_collections": gc.get_count(),
            }
        except ImportError:
            return {"error": "psutil not available"}

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure.

        Returns:
            True if memory pressure is high.
        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            # Consider memory pressure high if less than 500MB available or >90% used
            return memory.available < 500 * 1024 * 1024 or memory.percent > 90
        except ImportError:
            return False

    def force_garbage_collection(self) -> int:
        """Force garbage collection and return number of objects collected.

        Returns:
            Number of objects collected.
        """
        import gc
        import time

        before_count = len(gc.get_objects())
        collected = gc.collect()
        after_count = len(gc.get_objects())

        self._last_gc_time = time.time()

        self.logger.debug(
            f"🗑️ Forced GC: {collected} collections, {before_count - after_count} objects freed"  # noqa: E501
        )
        return collected

    def periodic_cleanup(self) -> None:
        """Perform periodic memory cleanup if needed."""
        import time

        current_time = time.time()

        # Only check every 30 seconds
        if current_time - self._last_memory_check < 30:
            return

        self._last_memory_check = current_time

        memory_info = self.get_memory_info()

        # Force GC if memory usage is high or it's been a while
        should_gc = (
            memory_info.get("rss_mb", 0) > 500  # High memory usage
            or self.check_memory_pressure()  # System memory pressure
            or current_time - self._last_gc_time > 300  # Haven't GC'd in 5 minutes
        )

        if should_gc:
            self.logger.info(
                f"💾 Memory cleanup triggered - RSS: {memory_info.get('rss_mb', 0):.1f}MB"  # noqa: E501
            )
            self.force_garbage_collection()

    def log_memory_stats(self, context: str = "") -> None:
        """Log current memory statistics.

        Args:
            context: Optional context string for the log message.
        """
        memory_info = self.get_memory_info()

        if "error" in memory_info:
            self.logger.debug(f"📊 Memory stats unavailable: {memory_info['error']}")
            return

        context_str = f" ({context})" if context else ""
        self.logger.info(
            f"📊 Memory stats{context_str}: "
            f"RSS={memory_info['rss_mb']:.1f}MB, "
            f"Available={memory_info['available_mb']:.1f}MB, "
            f"Usage={memory_info['percent']:.1f}%, "
            f"Objects={memory_info['gc_objects']}"
        )


# Global memory monitor instance
memory_monitor = MemoryMonitor()
