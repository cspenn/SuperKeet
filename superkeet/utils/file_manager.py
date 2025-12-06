"""File management utilities for SuperKeet application."""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class FileManager:
    """Manages file operations including cleanup and retention policies."""

    def __init__(self, base_directory: str = "debug_audio"):
        """Initialize file manager.

        Args:
            base_directory: Base directory for file operations
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)

    def cleanup_old_files(
        self,
        retention_days: int = 7,
        max_files: Optional[int] = None,
        file_pattern: str = "*.wav",
    ) -> int:
        """Clean up old files based on retention policy.

        Args:
            retention_days: Number of days to retain files
            max_files: Maximum number of files to keep (None for unlimited)
            file_pattern: File pattern to match (e.g., "*.wav", "*.log")

        Returns:
            Number of files removed
        """
        if not self.base_directory.exists():
            return 0

        try:
            # Get all matching files
            files = list(self.base_directory.glob(file_pattern))

            if not files:
                return 0

            # Remove files older than retention period
            files_removed = self._compute_age_cleanup(files, retention_days)

            # If max_files is set, remove excess files (keep newest)
            if max_files is not None:
                files_removed += self._compute_count_cleanup(file_pattern, max_files)

            if files_removed > 0:
                logger.info(
                    f"🧹 Cleaned up {files_removed} debug files from {self.base_directory}"  # noqa: E501
                )

            return files_removed

        except Exception as e:
            logger.error(f"🛑 File cleanup failed: {e}")
            return 0

    def _compute_age_cleanup(self, files: List[Path], retention_days: int) -> int:
        """Remove files older than retention period."""
        files_removed = 0
        current_time = time.time()
        cutoff_time = current_time - (retention_days * 24 * 3600)

        # Sort by modification time (oldest first)
        files.sort(key=lambda f: f.stat().st_mtime)

        for file_path in files:
            if file_path.stat().st_mtime < cutoff_time:
                if self._safe_delete_file(file_path):
                    files_removed += 1
        return files_removed

    def _compute_count_cleanup(self, file_pattern: str, max_files: int) -> int:
        """Remove excess files beyond max_files limit."""
        files_removed = 0
        current_files = list(self.base_directory.glob(file_pattern))
        remaining_files = [f for f in current_files if f.exists()]
        remaining_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        if len(remaining_files) > max_files:
            excess_files = remaining_files[max_files:]
            for file_path in excess_files:
                if self._safe_delete_file(file_path):
                    files_removed += 1
        return files_removed

    def _safe_delete_file(self, file_path: Path) -> bool:
        """Safely delete a file and log the result.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            file_path.unlink()
            logger.debug(f"🗑️ Removed file: {file_path.name}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove {file_path.name}: {e}")
            return False

        except Exception as e:
            logger.error(f"🛑 File cleanup failed: {e}")
            return 0

    def get_directory_size(self, file_pattern: str = "*") -> int:
        """Get total size of directory in bytes.

        Args:
            file_pattern: File pattern to match

        Returns:
            Total size in bytes
        """
        if not self.base_directory.exists():
            return 0

        try:
            files = list(self.base_directory.glob(file_pattern))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            return total_size
        except Exception as e:
            logger.error(f"🛑 Failed to get directory size: {e}")
            return 0

    def get_file_count(self, file_pattern: str = "*") -> int:
        """Get file count in directory.

        Args:
            file_pattern: File pattern to match

        Returns:
            Number of files
        """
        if not self.base_directory.exists():
            return 0

        try:
            files = list(self.base_directory.glob(file_pattern))
            return len([f for f in files if f.is_file()])
        except Exception as e:
            logger.error(f"🛑 Failed to get file count: {e}")
            return 0

    def check_size_threshold(
        self, max_size_mb: float = 100.0, file_pattern: str = "*.wav"
    ) -> bool:
        """Check if directory exceeds size threshold.

        Args:
            max_size_mb: Maximum size in megabytes
            file_pattern: File pattern to match

        Returns:
            True if threshold is exceeded
        """
        total_size = self.get_directory_size(file_pattern)
        file_count = self.get_file_count(file_pattern)
        size_mb = total_size / (1024 * 1024)

        if size_mb > max_size_mb:
            logger.warning(
                f"⚠️ Directory {self.base_directory} exceeds size threshold: "
                f"{size_mb:.1f}MB > {max_size_mb}MB ({file_count} files)"
            )
            return True

        return False

    def get_file_info(self, file_pattern: str = "*.wav") -> List[dict]:
        """Get information about files in directory.

        Args:
            file_pattern: File pattern to match

        Returns:
            List of file information dictionaries
        """
        if not self.base_directory.exists():
            return []

        try:
            files = list(self.base_directory.glob(file_pattern))
            file_info = []

            for file_path in files:
                if file_path.is_file():
                    stat = file_path.stat()
                    file_info.append(
                        {
                            "name": file_path.name,
                            "path": str(file_path),
                            "size_bytes": stat.st_size,
                            "size_mb": stat.st_size / (1024 * 1024),
                            "modified": datetime.fromtimestamp(stat.st_mtime),
                            "age_days": (time.time() - stat.st_mtime) / (24 * 3600),
                        }
                    )

            # Sort by modification time (newest first)
            file_info.sort(key=lambda x: x["modified"], reverse=True)
            return file_info

        except Exception as e:
            logger.error(f"🛑 Failed to get file info: {e}")
            return []

    def notify_user_if_threshold_exceeded(
        self, max_size_mb: float = 50.0, max_files: int = 100
    ) -> bool:
        """Notify user if storage thresholds are exceeded.

        Args:
            max_size_mb: Maximum size threshold in MB
            max_files: Maximum file count threshold

        Returns:
            True if notification was shown
        """
        total_size = self.get_directory_size("*.wav")
        file_count = self.get_file_count("*.wav")
        size_mb = total_size / (1024 * 1024)

        if size_mb > max_size_mb or file_count > max_files:
            logger.warning(
                f"🔔 Debug audio storage threshold exceeded: "
                f"{size_mb:.1f}MB ({file_count} files) in {self.base_directory}"
            )
            logger.info(
                "💡 Consider enabling automatic cleanup or manually clearing old debug files"  # noqa: E501
            )
            return True

        return False
