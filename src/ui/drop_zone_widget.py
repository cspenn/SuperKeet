"""Drop zone widget for drag-and-drop file handling."""

from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QPainter, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.config.config_loader import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DropZoneWidget(QWidget):
    """Widget that accepts drag-and-drop files for batch transcription."""

    # Signals
    files_dropped = Signal(list)  # List of Path objects

    def __init__(self, parent=None):
        """Initialize drop zone widget."""
        super().__init__(parent)

        self.supported_formats = config.get(
            "batch_processing.supported_formats",
            ["mp3", "m4a", "mp4", "mov", "wav", "flac"],
        )

        self.setAcceptDrops(True)
        self.setup_ui()
        self.apply_styles()

        logger.debug("🎯 Drop zone widget initialized")

    def setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Main drop label
        self.drop_label = QLabel("Drop audio/video files here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setWordWrap(True)

        # Supported formats label
        formats_text = "Supported: " + ", ".join(
            f".{fmt}" for fmt in self.supported_formats
        )
        self.formats_label = QLabel(formats_text)
        self.formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formats_label.setWordWrap(True)

        layout.addWidget(self.drop_label)
        layout.addWidget(self.formats_label)

        self.setLayout(layout)

    def apply_styles(self) -> None:
        """Apply styling to the widget."""
        self.setStyleSheet("""
            DropZoneWidget {
                background-color: #2A2A2A;
                border: 2px dashed #555;
                border-radius: 8px;
                min-height: 120px;
            }
            
            DropZoneWidget[drag_active="true"] {
                border-color: #007AFF;
                background-color: #1A1A2E;
            }
            
            QLabel {
                color: #EAEAEA;
                font-size: 14px;
                background: transparent;
                border: none;
            }
            
            QLabel#formats_label {
                color: #8E8E93;
                font-size: 12px;
            }
        """)

        # Set object names for styling
        self.formats_label.setObjectName("formats_label")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if self._has_supported_files(event):
            event.acceptProposedAction()
            self.setProperty("drag_active", True)
            self.style().polish(self)  # Refresh styling
            self.drop_label.setText("Drop files to start transcription")
            logger.debug("🎯 Drag enter: files detected")
        else:
            event.ignore()
            logger.debug("🟡 Drag enter: no supported files")

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Handle drag move event."""
        if self._has_supported_files(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        """Handle drag leave event."""
        self.setProperty("drag_active", False)
        self.style().polish(self)  # Refresh styling
        self.drop_label.setText("Drop audio/video files here")
        logger.debug("🎯 Drag leave")

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop event."""
        file_paths = self._extract_file_paths(event)

        if file_paths:
            logger.info(f"📁 Files dropped: {len(file_paths)} files")
            self.files_dropped.emit(file_paths)

            # Update UI
            self.setProperty("drag_active", False)
            self.style().polish(self)
            self.drop_label.setText(f"Processing {len(file_paths)} files...")

            event.acceptProposedAction()
        else:
            event.ignore()
            logger.warning("🟡 No valid files in drop")

    def _has_supported_files(self, event) -> bool:
        """Check if drag event contains supported files.

        Args:
            event: Drag event

        Returns:
            True if event contains supported files
        """
        if not event.mimeData().hasUrls():
            return False

        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = Path(url.toLocalFile())
                extension = file_path.suffix.lower().lstrip(".")

                if extension in self.supported_formats:
                    return True

        return False

    def _extract_file_paths(self, event) -> List[Path]:
        """Extract valid file paths from drop event.

        Args:
            event: Drop event

        Returns:
            List of valid file paths
        """
        file_paths = []

        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())

                    if file_path.exists() and file_path.is_file():
                        extension = file_path.suffix.lower().lstrip(".")

                        if extension in self.supported_formats:
                            file_paths.append(file_path)
                            logger.debug(f"📁 Valid file: {file_path.name}")
                        else:
                            logger.debug(
                                f"🟡 Unsupported: {file_path.name} (.{extension})"
                            )
                    else:
                        logger.warning(f"⚠️ Invalid path: {file_path}")

        return file_paths

    def reset_state(self) -> None:
        """Reset the drop zone to initial state."""
        self.setProperty("drag_active", False)
        self.style().polish(self)
        self.drop_label.setText("Drop audio/video files here")
        logger.debug("🎯 Drop zone reset")

    def show_processing_state(self, file_count: int) -> None:
        """Show processing state.

        Args:
            file_count: Number of files being processed
        """
        self.drop_label.setText(f"Processing {file_count} files...")

    def show_completion_state(self, success_count: int, total_count: int) -> None:
        """Show completion state.

        Args:
            success_count: Number of successfully processed files
            total_count: Total number of files processed
        """
        if success_count == total_count:
            self.drop_label.setText(f"✅ All {total_count} files completed!")
        else:
            self.drop_label.setText(f"⚠️ {success_count}/{total_count} files completed")

    def paintEvent(self, event) -> None:
        """Custom paint event for visual feedback."""
        super().paintEvent(event)

        # Draw additional visual indicators if needed
        if self.property("drag_active"):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Draw an additional inner border for active state
            pen = QPen(Qt.GlobalColor.blue, 1, Qt.PenStyle.DotLine)
            painter.setPen(pen)

            rect = self.rect().adjusted(5, 5, -5, -5)
            painter.drawRoundedRect(rect, 6, 6)

            painter.end()
