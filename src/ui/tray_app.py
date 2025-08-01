# start src/ui/tray_app.py
"""System tray application for SuperKeet."""

import os
from enum import Enum
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from src.asr.transcriber import ASRTranscriber
from src.audio.recorder import AudioRecorder
from src.hotkey.listener import HotkeyListener
from src.text.injector import TextInjector
from src.ui.main_window import MainWindow
from src.ui.settings_dialog import SettingsDialog
from src.utils.logger import setup_logger
from src.utils.transcript_logger import TranscriptLogger

logger = setup_logger(__name__)


class AppState(Enum):
    """Application state enumeration."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class ASRWorker(QThread):
    """Worker thread for ASR processing."""

    transcription_complete = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, transcriber: ASRTranscriber) -> None:
        """Initialize the ASR worker.

        Args:
            transcriber: ASR transcriber instance.
        """
        super().__init__()
        self.transcriber = transcriber
        self.audio_data: Optional[np.ndarray] = None

    def set_audio(self, audio_data: np.ndarray) -> None:
        """Set audio data for transcription.

        Args:
            audio_data: Audio data to transcribe.
        """
        self.audio_data = audio_data

    def run(self) -> None:
        """Run the transcription process."""
        if self.audio_data is None:
            self.error_occurred.emit("No audio data provided")
            return

        try:
            text = self.transcriber.transcribe(self.audio_data)
            self.transcription_complete.emit(text)
        except Exception as e:
            self.error_occurred.emit(str(e))


class SuperKeetApp:
    """Main application class for SuperKeet."""

    def __init__(self) -> None:
        """Initialize the SuperKeet application."""
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])

        # Don't quit when last window closes
        self.app.setQuitOnLastWindowClosed(False)

        # Initialize components
        self.audio_recorder = AudioRecorder()
        self.asr_transcriber = ASRTranscriber()
        self.hotkey_listener = HotkeyListener()
        self.text_injector = TextInjector()

        # Initialize UI
        self.tray_icon = QSystemTrayIcon()
        self.menu = QMenu()
        self.main_window = MainWindow()

        # State
        self.state = AppState.IDLE

        # Worker thread
        self.asr_worker: Optional[ASRWorker] = None
        
        # Transcript logger
        self.transcript_logger = TranscriptLogger()

        # Create icon
        self._create_icon()

        self._setup_ui()
        self._connect_signals()

        logger.info("SuperKeetApp initialized")

    def _create_icon(self) -> None:
        """Create icon for the system tray."""
        # Try to load the parakeet SVG icon
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "assets", "parakeet.svg"
        )
        if os.path.exists(icon_path):
            self.icon = QIcon(icon_path)
            logger.info(f"Loaded icon from {icon_path}")
        else:
            # Create a simple colored square as fallback
            pixmap = QPixmap(32, 32)
            pixmap.fill("#007AFF")  # Blue color
            self.icon = QIcon(pixmap)
            logger.info("Using default colored icon")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Create menu actions
        self.status_action = QAction("Status: Ready", self.menu)
        self.status_action.setEnabled(False)

        self.show_window_action = QAction("Show SuperKeet Window", self.menu)
        self.show_window_action.triggered.connect(self._on_show_window)

        self.recent_menu = QMenu("Recent Transcriptions", self.menu)
        self.recent_transcriptions = []  # Store last 5 transcriptions

        self.settings_action = QAction("Settings...", self.menu)
        self.settings_action.triggered.connect(self._on_settings)

        self.quit_action = QAction("Quit SuperKeet", self.menu)
        self.quit_action.triggered.connect(self._on_quit)

        # Add actions to menu
        self.menu.addAction(self.status_action)
        self.menu.addAction(self.show_window_action)
        self.menu.addSeparator()
        self.menu.addMenu(self.recent_menu)
        self.menu.addAction(self.settings_action)
        self.menu.addSeparator()
        self.menu.addAction(self.quit_action)

        # Set up tray icon
        self.tray_icon.setIcon(self.icon)
        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.setVisible(True)
        self._update_status()

        # Show notification if available
        if self.tray_icon.supportsMessages():
            self.tray_icon.showMessage(
                "SuperKeet",
                "Voice-to-text is ready",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        # Connect hotkey signals
        self.hotkey_listener.signals.hotkey_pressed.connect(self._on_hotkey_pressed)
        self.hotkey_listener.signals.hotkey_released.connect(self._on_hotkey_released)
        
        # Connect audio recorder to waveform widget
        self.audio_recorder.audio_chunk_ready.connect(self.main_window.waveform_widget.update_data)

    def _update_icon(self) -> None:
        """Update tray icon based on current state."""
        if self.state == AppState.IDLE:
            # Use existing icon for idle state
            self.tray_icon.setIcon(self.icon)
        elif self.state == AppState.RECORDING:
            # Create recording icon (red circle)
            recording_pixmap = QPixmap(32, 32)
            recording_pixmap.fill("#FF3B30")  # Red for recording
            recording_icon = QIcon(recording_pixmap)
            self.tray_icon.setIcon(recording_icon)
        elif self.state == AppState.PROCESSING:
            # Create processing icon (yellow/orange)
            processing_pixmap = QPixmap(32, 32)
            processing_pixmap.fill("#FFD60A")  # Yellow for processing
            processing_icon = QIcon(processing_pixmap)
            self.tray_icon.setIcon(processing_icon)

        # Also update status text and tooltip
        self._update_status()

    def _update_status(self) -> None:
        """Update status text based on current state."""
        if self.state == AppState.IDLE:
            self.tray_icon.setToolTip("SuperKeet - Ready")
            self.status_action.setText("Status: Ready")
        elif self.state == AppState.RECORDING:
            self.tray_icon.setToolTip("SuperKeet - Recording")
            self.status_action.setText("Status: Recording...")
        elif self.state == AppState.PROCESSING:
            self.tray_icon.setToolTip("SuperKeet - Processing")
            self.status_action.setText("Status: Processing...")

        # Update main window state
        self.main_window.update_state(self.state.value)

    @Slot()
    def _on_hotkey_pressed(self) -> None:
        """Handle hotkey press event."""
        if self.state != AppState.IDLE:
            logger.warning("Hotkey pressed but not in idle state")
            return

        logger.debug("Starting recording")
        self.state = AppState.RECORDING
        self._update_icon()
        self.audio_recorder.start()

    @Slot()
    def _on_hotkey_released(self) -> None:
        """Handle hotkey release event."""
        if self.state != AppState.RECORDING:
            logger.warning("Hotkey released but not in recording state")
            return

        logger.debug("Stopping recording")
        self.state = AppState.PROCESSING
        self._update_icon()

        # Stop recording and get audio
        audio_data = self.audio_recorder.stop()

        if audio_data.size == 0:
            logger.warning("No audio captured")
            self.state = AppState.IDLE
            self._update_icon()
            return

        # Start ASR processing in worker thread
        self.asr_worker = ASRWorker(self.asr_transcriber)
        self.asr_worker.set_audio(audio_data)
        self.asr_worker.transcription_complete.connect(self._on_transcription_complete)
        self.asr_worker.error_occurred.connect(self._on_transcription_error)
        self.asr_worker.start()

    @Slot(str)
    def _on_transcription_complete(self, text: str) -> None:
        """Handle transcription completion.

        Args:
            text: Transcribed text.
        """
        logger.info(f"Transcription complete: {text[:50]}...")

        # Inject text
        if text:
            success = self.text_injector.inject(text)
            if not success:
                logger.error("Failed to inject text")
            else:
                # Update main window
                self.main_window.update_transcription(text)
                # Add to recent transcriptions
                self._add_recent_transcription(text)
                # Log transcript to disk if enabled
                self.transcript_logger.log_transcript(text)

        # Reset state
        self.state = AppState.IDLE
        self._update_status()

        # Clean up worker
        if self.asr_worker:
            self.asr_worker.quit()
            self.asr_worker.wait()
            self.asr_worker = None

    @Slot(str)
    def _on_transcription_error(self, error: str) -> None:
        """Handle transcription error.

        Args:
            error: Error message.
        """
        logger.error(f"Transcription error: {error}")

        # Show notification
        if self.tray_icon.supportsMessages():
            self.tray_icon.showMessage(
                "SuperKeet Error",
                f"Transcription failed: {error}",
                QSystemTrayIcon.MessageIcon.Critical,
                3000,
            )

        # Reset state
        self.state = AppState.IDLE
        self._update_icon()

        # Clean up worker
        if self.asr_worker:
            self.asr_worker.quit()
            self.asr_worker.wait()
            self.asr_worker = None

    @Slot()
    def _on_show_window(self) -> None:
        """Handle show window action."""
        if self.main_window.isVisible():
            self.main_window.hide()
        else:
            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()

    def _add_recent_transcription(self, text: str) -> None:
        """Add a transcription to the recent menu."""
        # Limit text length for menu display
        display_text = text[:50] + "..." if len(text) > 50 else text

        # Add to list
        self.recent_transcriptions.insert(0, text)
        if len(self.recent_transcriptions) > 5:
            self.recent_transcriptions.pop()

        # Update menu
        self.recent_menu.clear()
        for i, transcript in enumerate(self.recent_transcriptions):
            action_text = (
                transcript[:50] + "..." if len(transcript) > 50 else transcript
            )
            action = QAction(action_text, self.recent_menu)
            action.triggered.connect(
                lambda checked, t=transcript: self.text_injector.inject(t)
            )
            self.recent_menu.addAction(action)

    @Slot()
    def _on_settings(self) -> None:
        """Handle settings action."""
        logger.info("Opening settings dialog")
        settings_dialog = SettingsDialog(None)
        settings_dialog.exec()

    @Slot()
    def _on_quit(self) -> None:
        """Handle quit action."""
        logger.info("Quit requested")
        self.cleanup()
        self.app.quit()

    def run(self) -> None:
        """Run the application."""
        try:
            # Check if model needs downloading
            if not self.asr_transcriber._is_model_cached():
                if self.tray_icon.supportsMessages():
                    self.tray_icon.showMessage(
                        "SuperKeet",
                        "Downloading ASR model (~600MB). This is a one-time download...",
                        QSystemTrayIcon.MessageIcon.Information,
                        5000,
                    )

            # Load ASR model
            logger.info("Loading ASR model...")
            self.asr_transcriber.load_model()

            # Start hotkey listener
            self.hotkey_listener.start()

            # Run event loop
            logger.info("Starting application")
            self.app.exec()

        except Exception as e:
            logger.error(f"Application error: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")

        # Stop hotkey listener
        self.hotkey_listener.stop()

        # Unload model
        self.asr_transcriber.unload_model()

        # Hide tray icon
        self.tray_icon.setVisible(False)


# end src/ui/tray_app.py
