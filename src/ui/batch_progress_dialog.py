"""Batch progress dialog for monitoring transcription progress."""

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.batch.batch_transcriber import BatchTranscriber
from src.batch.progress_manager import ProgressManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BatchProgressDialog(QDialog):
    """Dialog for displaying batch transcription progress."""

    # Signals
    batch_cancelled = Signal()

    def __init__(self, file_paths: List[Path], parent=None):
        """Initialize batch progress dialog.

        Args:
            file_paths: List of files to process
            parent: Parent widget
        """
        super().__init__(parent)
        self.file_paths = file_paths

        # Components
        self.batch_transcriber: Optional[BatchTranscriber] = None
        self.progress_manager: Optional[ProgressManager] = None

        # State tracking
        self.is_running = False
        self.results: Dict[str, any] = {}
        self.file_results: List[Dict[str, any]] = []

        self.setup_ui()
        self.setup_components()
        self.connect_signals()

        logger.info(f"📊 Batch progress dialog initialized for {len(file_paths)} files")

    def setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Batch Transcription Progress")
        self.setModal(True)
        self.resize(700, 500)

        # Main layout
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("Batch Transcription in Progress")
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #EAEAEA; margin: 10px;"
        )
        layout.addWidget(title_label)

        # Progress section
        progress_group = self.create_progress_section()
        layout.addWidget(progress_group)

        # Results section (splitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # File list section
        file_list_group = self.create_file_list_section()
        splitter.addWidget(file_list_group)

        # Current transcript section
        transcript_group = self.create_transcript_section()
        splitter.addWidget(transcript_group)

        splitter.setSizes([350, 350])  # Equal split
        layout.addWidget(splitter)

        # Control buttons
        button_layout = QHBoxLayout()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_processing)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        self.apply_styles()

    def create_progress_section(self) -> QGroupBox:
        """Create progress tracking section."""
        group = QGroupBox("Progress")
        layout = QVBoxLayout(group)

        # Overall progress
        progress_layout = QHBoxLayout()

        self.progress_label = QLabel("Preparing...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar, 1)

        layout.addLayout(progress_layout)

        # Statistics layout
        stats_layout = QHBoxLayout()

        self.files_label = QLabel("Files: 0/0")
        self.time_label = QLabel("Time: 0s")
        self.eta_label = QLabel("ETA: Calculating...")

        stats_layout.addWidget(self.files_label)
        stats_layout.addStretch()
        stats_layout.addWidget(self.time_label)
        stats_layout.addStretch()
        stats_layout.addWidget(self.eta_label)

        layout.addLayout(stats_layout)

        # Current file
        self.current_file_label = QLabel("Current: None")
        self.current_file_label.setWordWrap(True)
        layout.addWidget(self.current_file_label)

        return group

    def create_file_list_section(self) -> QGroupBox:
        """Create file list section."""
        group = QGroupBox("Files")
        layout = QVBoxLayout(group)

        # Scroll area for file list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container widget for file items
        self.file_list_widget = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_widget)
        self.file_list_layout.setContentsMargins(5, 5, 5, 5)

        # Add file items
        self.file_items = {}
        for file_path in self.file_paths:
            file_item = self.create_file_item(file_path)
            self.file_items[str(file_path)] = file_item
            self.file_list_layout.addWidget(file_item)

        self.file_list_layout.addStretch()

        scroll_area.setWidget(self.file_list_widget)
        layout.addWidget(scroll_area)

        return group

    def create_file_item(self, file_path: Path) -> QFrame:
        """Create a file item widget.

        Args:
            file_path: Path to file

        Returns:
            QFrame containing file information
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setContentsMargins(5, 5, 5, 5)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)

        # File name
        name_label = QLabel(file_path.name)
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)

        # File size
        try:
            size_mb = file_path.stat().st_size / (1024**2)
            size_label = QLabel(f"Size: {size_mb:.1f} MB")
            size_label.setStyleSheet("font-size: 11px; color: #8E8E93;")
            layout.addWidget(size_label)
        except Exception:
            pass

        # Status
        status_label = QLabel("⏳ Waiting")
        status_label.setObjectName("status_label")
        layout.addWidget(status_label)

        frame.setProperty("file_path", str(file_path))
        frame.findChild(QLabel, "status_label") or layout.itemAt(2).widget()

        return frame

    def create_transcript_section(self) -> QGroupBox:
        """Create transcript display section."""
        group = QGroupBox("Current Transcript")
        layout = QVBoxLayout(group)

        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setPlaceholderText("Transcript will appear here...")

        layout.addWidget(self.transcript_display)

        return group

    def setup_components(self) -> None:
        """Set up batch processing components."""
        # Create batch transcriber
        self.batch_transcriber = BatchTranscriber(self)

        # Create progress manager
        self.progress_manager = ProgressManager(self)

    def connect_signals(self) -> None:
        """Connect component signals."""
        if self.batch_transcriber:
            self.batch_transcriber.progress_updated.connect(self.update_file_progress)
            self.batch_transcriber.file_completed.connect(self.on_file_completed)
            self.batch_transcriber.batch_completed.connect(self.on_batch_completed)
            self.batch_transcriber.error_occurred.connect(self.on_error_occurred)

        if self.progress_manager:
            self.progress_manager.progress_changed.connect(self.update_progress_display)
            self.progress_manager.eta_updated.connect(self.update_eta_display)
            self.progress_manager.status_changed.connect(self.update_status)

    def start_processing(self) -> None:
        """Start the batch processing."""
        if not self.batch_transcriber or not self.progress_manager:
            logger.error("🛑 Components not initialized")
            return

        logger.info(f"🚀 Starting batch processing: {len(self.file_paths)} files")

        # Initialize progress tracking
        self.progress_manager.start_batch(self.file_paths)

        # Start batch transcription
        self.batch_transcriber.start_batch(self.file_paths)

        self.is_running = True
        self.cancel_button.setText("Cancel")
        self.close_button.setEnabled(False)

    @Slot(int, int, str)
    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        """Update progress display.

        Args:
            current: Current file number
            total: Total number of files
            filename: Current filename
        """
        if self.progress_manager:
            self.progress_manager.start_file(filename)

        # Update current file display
        self.current_file_label.setText(f"Current: {filename}")

        # Update overall progress
        progress_pct = int((current / max(1, total)) * 100)
        self.progress_bar.setValue(progress_pct)

        # Highlight current file in list
        self._highlight_current_file(filename)

    @Slot(str, str, bool)
    def on_file_completed(self, filename: str, transcript: str, success: bool) -> None:
        """Handle file completion.

        Args:
            filename: Completed filename
            transcript: Generated transcript
            success: Whether processing was successful
        """
        if self.progress_manager:
            self.progress_manager.complete_file(filename, success)

        # Update file item status
        self._update_file_status(filename, success)

        # Show transcript if successful
        if success and transcript:
            self.transcript_display.setPlainText(transcript)

        # Store result
        self.file_results.append(
            {"filename": filename, "transcript": transcript, "success": success}
        )

        logger.debug(
            f"📝 File completed: {filename} ({'success' if success else 'failed'})"
        )

    @Slot(dict)
    def on_batch_completed(self, results: Dict[str, any]) -> None:
        """Handle batch completion.

        Args:
            results: Batch processing results
        """
        self.results = results
        self.is_running = False

        if self.progress_manager:
            summary = self.progress_manager.finish_batch()
            self.results.update(summary)

        # Update UI
        self.progress_bar.setValue(100)
        self.current_file_label.setText("Batch processing completed")
        self.cancel_button.setText("Cancel")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)

        # Show completion message
        success_count = results.get("successful_files", 0)
        total_count = results.get("total_files", 0)

        if success_count == total_count:
            self.progress_label.setText(
                f"✅ All {total_count} files completed successfully!"
            )
        else:
            failed_count = results.get("failed_files", 0)
            self.progress_label.setText(
                f"⚠️ {success_count} successful, {failed_count} failed"
            )

        logger.info(f"🏁 Batch completed: {success_count}/{total_count} successful")

    @Slot(str)
    def on_error_occurred(self, error_message: str) -> None:
        """Handle batch processing error.

        Args:
            error_message: Error message
        """
        self.is_running = False

        self.progress_label.setText(f"❌ Error: {error_message}")
        self.current_file_label.setText("Processing stopped due to error")

        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)

        logger.error(f"🛑 Batch error: {error_message}")

    @Slot(dict)
    def update_progress_display(self, progress_info: Dict[str, any]) -> None:
        """Update progress display with detailed information.

        Args:
            progress_info: Progress information dictionary
        """
        completed = progress_info.get("completed_files", 0)
        total = progress_info.get("total_files", 0)
        elapsed = progress_info.get("elapsed_time_formatted", "0s")

        self.files_label.setText(f"Files: {completed}/{total}")
        self.time_label.setText(f"Time: {elapsed}")

    @Slot(str)
    def update_eta_display(self, eta: str) -> None:
        """Update ETA display.

        Args:
            eta: Formatted ETA string
        """
        self.eta_label.setText(f"ETA: {eta}")

    @Slot(str)
    def update_status(self, status: str) -> None:
        """Update status display.

        Args:
            status: Status message
        """
        self.progress_label.setText(status)

    def cancel_processing(self) -> None:
        """Cancel the batch processing."""
        if self.is_running and self.batch_transcriber:
            logger.info("🛑 User cancelled batch processing")
            self.batch_transcriber.stop_batch()
            self.batch_cancelled.emit()

            self.progress_label.setText("❌ Processing cancelled by user")
            self.is_running = False
            self.cancel_button.setEnabled(False)
            self.close_button.setEnabled(True)

    def _highlight_current_file(self, filename: str) -> None:
        """Highlight the current file in the list.

        Args:
            filename: Current filename to highlight
        """
        # Reset all file items
        for file_item in self.file_items.values():
            file_item.setStyleSheet("")

        # Find and highlight current file
        for file_path, file_item in self.file_items.items():
            if Path(file_path).name == filename:
                file_item.setStyleSheet("QFrame { border: 2px solid #007AFF; }")

                # Update status
                status_label = file_item.findChild(QLabel)
                if status_label:
                    for i in range(file_item.layout().count()):
                        widget = file_item.layout().itemAt(i).widget()
                        if (
                            widget
                            and isinstance(widget, QLabel)
                            and "status_label" in widget.objectName()
                        ):
                            widget.setText("🔄 Processing")
                            break
                break

    def _update_file_status(self, filename: str, success: bool) -> None:
        """Update file status in the list.

        Args:
            filename: Filename to update
            success: Whether processing was successful
        """
        for file_path, file_item in self.file_items.items():
            if Path(file_path).name == filename:
                # Find status label and update
                for i in range(file_item.layout().count()):
                    widget = file_item.layout().itemAt(i).widget()
                    if (
                        widget
                        and isinstance(widget, QLabel)
                        and "Waiting" not in widget.text()
                        and "Processing" not in widget.text()
                    ):
                        continue
                    if widget and isinstance(widget, QLabel):
                        if success:
                            widget.setText("✅ Completed")
                            file_item.setStyleSheet(
                                "QFrame { border: 2px solid #34C759; }"
                            )
                        else:
                            widget.setText("❌ Failed")
                            file_item.setStyleSheet(
                                "QFrame { border: 2px solid #FF3B30; }"
                            )
                        break
                break

    def apply_styles(self) -> None:
        """Apply styling to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
                color: #EAEAEA;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                margin-top: 1ex;
                padding: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QLabel {
                color: #EAEAEA;
            }
            
            QProgressBar {
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                background-color: #2A2A2A;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 3px;
            }
            
            QPushButton {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 6px 12px;
                color: #EAEAEA;
            }
            
            QPushButton:hover {
                background-color: #3A3A3A;
            }
            
            QTextEdit {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                color: #EAEAEA;
            }
            
            QFrame {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                margin: 2px;
            }
            
            QScrollArea {
                background-color: transparent;
                border: none;
            }
        """)

    def closeEvent(self, event) -> None:
        """Handle close event."""
        if self.is_running:
            self.cancel_processing()

        # Cleanup
        if self.progress_manager:
            self.progress_manager.cleanup()

        super().closeEvent(event)
