"""Progress manager for batch transcription operations."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProgressManager(QObject):
    """Manages progress tracking and reporting for batch operations."""

    # Signals for progress updates
    progress_changed = Signal(dict)  # Progress information dictionary
    eta_updated = Signal(str)  # Estimated time remaining
    status_changed = Signal(str)  # Status message

    def __init__(self, parent: Optional[QObject] = None):
        """Initialize progress manager."""
        super().__init__(parent)

        # Progress tracking
        self.total_files = 0
        self.completed_files = 0
        self.current_file = ""
        self.start_time: Optional[datetime] = None
        self.file_start_times: Dict[str, datetime] = {}
        self.file_completion_times: Dict[str, float] = {}  # Processing time in seconds

        # Statistics
        self.successful_files = 0
        self.failed_files = 0
        self.total_bytes_processed = 0
        self.average_processing_time = 0.0

        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._emit_progress_update)
        self.update_timer.start(1000)  # Update every second

    def start_batch(self, file_paths: List[Path]) -> None:
        """Start tracking a new batch operation.

        Args:
            file_paths: List of files to be processed
        """
        self.total_files = len(file_paths)
        self.completed_files = 0
        self.successful_files = 0
        self.failed_files = 0
        self.current_file = ""
        self.start_time = datetime.now()
        self.file_start_times.clear()
        self.file_completion_times.clear()

        # Calculate total bytes
        self.total_bytes_processed = 0
        for file_path in file_paths:
            try:
                if file_path.exists():
                    self.total_bytes_processed += file_path.stat().st_size
            except Exception:
                pass  # Skip files we can't read

        logger.info(
            f"📊 Progress tracking started: {self.total_files} files, "
            f"{self.total_bytes_processed / (1024**2):.1f}MB total"
        )

        self.status_changed.emit("Starting batch processing...")
        self._emit_progress_update()

    def start_file(self, filename: str) -> None:
        """Mark the start of processing a file.

        Args:
            filename: Name of file being processed
        """
        self.current_file = filename
        self.file_start_times[filename] = datetime.now()

        logger.debug(f"⏱️ Started processing: {filename}")
        self.status_changed.emit(f"Processing: {filename}")

    def complete_file(self, filename: str, success: bool = True) -> None:
        """Mark completion of a file.

        Args:
            filename: Name of completed file
            success: Whether processing was successful
        """
        if filename in self.file_start_times:
            start_time = self.file_start_times[filename]
            processing_time = (datetime.now() - start_time).total_seconds()
            self.file_completion_times[filename] = processing_time

        self.completed_files += 1

        if success:
            self.successful_files += 1
            logger.debug(f"✅ Completed successfully: {filename}")
        else:
            self.failed_files += 1
            logger.debug(f"❌ Failed: {filename}")

        # Update average processing time
        if self.file_completion_times:
            self.average_processing_time = sum(
                self.file_completion_times.values()
            ) / len(self.file_completion_times)

        self._emit_progress_update()

    def finish_batch(self) -> Dict[str, any]:
        """Finish batch processing and return summary.

        Returns:
            Dictionary with batch processing summary
        """
        if not self.start_time:
            return {}

        total_time = (datetime.now() - self.start_time).total_seconds()

        summary = {
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "total_time_seconds": total_time,
            "total_time_formatted": self._format_duration(total_time),
            "average_processing_time": self.average_processing_time,
            "success_rate": (self.successful_files / max(1, self.completed_files))
            * 100,
            "throughput_files_per_minute": (
                self.completed_files / max(1, total_time / 60)
            ),
        }

        logger.info(
            f"📊 Batch complete: {summary['successful_files']}/{summary['total_files']} successful "  # noqa: E501
            f"({summary['success_rate']:.1f}%) in {summary['total_time_formatted']}"
        )

        self.status_changed.emit("Batch processing completed")
        return summary

    def get_progress_info(self) -> Dict[str, any]:
        """Get current progress information.

        Returns:
            Dictionary with current progress details
        """
        if not self.start_time:
            return {}

        elapsed_time = (datetime.now() - self.start_time).total_seconds()

        # Calculate progress percentage
        progress_pct = (self.completed_files / max(1, self.total_files)) * 100

        # Estimate remaining time
        eta_seconds = self._calculate_eta()
        eta_formatted = (
            self._format_duration(eta_seconds) if eta_seconds > 0 else "Calculating..."
        )

        return {
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "remaining_files": self.total_files - self.completed_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "current_file": self.current_file,
            "progress_percentage": progress_pct,
            "elapsed_time": elapsed_time,
            "elapsed_time_formatted": self._format_duration(elapsed_time),
            "eta_seconds": eta_seconds,
            "eta_formatted": eta_formatted,
            "average_processing_time": self.average_processing_time,
            "files_per_minute": (self.completed_files / max(1, elapsed_time / 60))
            if elapsed_time > 0
            else 0,
        }

    def _calculate_eta(self) -> float:
        """Calculate estimated time to completion.

        Returns:
            Estimated seconds remaining
        """
        if not self.start_time or self.completed_files == 0:
            return 0.0

        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        remaining_files = self.total_files - self.completed_files

        if remaining_files <= 0:
            return 0.0

        # Use average processing time if available, otherwise use elapsed time
        if self.average_processing_time > 0:
            return remaining_files * self.average_processing_time
        else:
            # Fallback: estimate based on overall progress
            average_time_per_file = elapsed_time / self.completed_files
            return remaining_files * average_time_per_file

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        if seconds < 0:
            return "Unknown"

        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _emit_progress_update(self) -> None:
        """Emit progress update signal with current information."""
        progress_info = self.get_progress_info()

        if progress_info:
            self.progress_changed.emit(progress_info)

            # Update ETA
            eta_formatted = progress_info.get("eta_formatted", "")
            if eta_formatted:
                self.eta_updated.emit(eta_formatted)

    def get_file_statistics(self) -> Dict[str, any]:
        """Get detailed file processing statistics.

        Returns:
            Dictionary with file processing statistics
        """
        if not self.file_completion_times:
            return {}

        times = list(self.file_completion_times.values())

        return {
            "total_files_timed": len(times),
            "fastest_file_seconds": min(times),
            "slowest_file_seconds": max(times),
            "average_file_seconds": sum(times) / len(times),
            "median_file_seconds": sorted(times)[len(times) // 2],
            "total_processing_seconds": sum(times),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.update_timer.isActive():
            self.update_timer.stop()

        logger.debug("🧹 Progress manager cleaned up")
