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
from src.ui.first_run_dialog import FirstRunDialog
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
        self.sample_rate: Optional[int] = None

    def set_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Set audio data and sample rate for transcription.

        Args:
            audio_data: Audio data to transcribe.
            sample_rate: Sample rate of the audio data.
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def run(self) -> None:
        """Run the transcription process."""
        if self.audio_data is None:
            self.error_occurred.emit("No audio data provided")
            return

        try:
            # Pass the actual sample rate to enable optimal resampling
            text = self.transcriber.transcribe(self.audio_data, self.sample_rate)
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
        self._setup_dock()
        self._connect_signals()

        # Check for first-run experience
        self._check_first_run()

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

    def _setup_dock(self) -> None:
        """Set up dock icon and functionality."""
        # Ensure dock icon is visible and properly named
        self.app.setApplicationDisplayName("SuperKeet")
        self.app.setApplicationName("SuperKeet")

        # Set the dock icon to use the same parakeet icon
        self.app.setWindowIcon(self.icon)

        # Create dock menu with same functionality as tray menu
        self._setup_dock_menu()

        logger.info("Dock functionality initialized")

    def _setup_dock_menu(self) -> None:
        """Set up dock context menu with proper ownership management."""
        try:
            # Create dock menu but DON'T use setAsDockMenu() to avoid ownership issues
            self.dock_menu = QMenu()

            # Add same actions as tray menu
            dock_show_action = QAction("Show SuperKeet Window", self.dock_menu)
            dock_show_action.triggered.connect(self._on_show_window)
            self.dock_menu.addAction(dock_show_action)

            self.dock_menu.addSeparator()

            # Recent transcriptions submenu - create as child of dock_menu
            self.dock_recent_menu = QMenu("Recent Transcriptions", self.dock_menu)
            self.dock_menu.addMenu(self.dock_recent_menu)

            dock_settings_action = QAction("Settings...", self.dock_menu)
            dock_settings_action.triggered.connect(self._on_settings)
            self.dock_menu.addAction(dock_settings_action)

            self.dock_menu.addSeparator()

            dock_quit_action = QAction("Quit SuperKeet", self.dock_menu)
            dock_quit_action.triggered.connect(self._on_quit)
            self.dock_menu.addAction(dock_quit_action)

            # Keep ownership of the menu - don't use setAsDockMenu()
            # This prevents the QMenu deletion error
            logger.info("Dock menu created with proper ownership management")

        except Exception as e:
            logger.warning(f"Failed to set up dock menu: {e}")
            logger.info("Application will continue without dock menu functionality")
            # Ensure these are None if setup fails
            self.dock_menu = None
            self.dock_recent_menu = None

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

        # Connect audio recorder to audio animation widget
        self.audio_recorder.audio_chunk_ready.connect(
            self.main_window.audio_animation.update_audio_level
        )

        # Connect main window microphone change signal
        self.main_window.microphone_changed.connect(
            self._on_main_window_microphone_changed
        )

        # Connect dock activation signal
        self.app.applicationStateChanged.connect(self._on_application_state_changed)

    def _check_first_run(self) -> None:
        """Check if this is first run and show setup dialog if needed."""
        from src.config.config_loader import config

        try:
            # Check if first run has been completed
            first_run_completed = config.get("app.first_run_completed", False)

            if not first_run_completed:
                logger.info("🎯 First run detected - showing setup wizard")

                # Create and show first-run dialog
                first_run_dialog = FirstRunDialog()
                first_run_dialog.setup_completed.connect(self._on_first_run_completed)

                # Show dialog modally
                first_run_dialog.exec()
            else:
                logger.info("✅ First run already completed")

        except Exception as e:
            logger.error(f"🛑 Error checking first-run state: {e}")
            # Continue with normal startup even if first-run check fails

    @Slot()
    def _on_first_run_completed(self) -> None:
        """Handle first-run setup completion."""
        logger.info("🎉 First-run setup completed successfully")

        # Show welcome notification
        if self.tray_icon.supportsMessages():
            self.tray_icon.showMessage(
                "SuperKeet Setup Complete",
                "Welcome to SuperKeet! Press Ctrl+Space to start voice transcription.",
                QSystemTrayIcon.MessageIcon.Information,
                4000,
            )

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
        # Pass both audio data and the actual sample rate used by recorder
        self.asr_worker.set_audio(audio_data, self.audio_recorder.sample_rate)
        self.asr_worker.transcription_complete.connect(self._on_transcription_complete)
        self.asr_worker.error_occurred.connect(self._on_transcription_error)
        self.asr_worker.start()

    @Slot(str)
    def _on_transcription_complete(self, text: str) -> None:
        """Handle transcription completion with robust error handling.

        Args:
            text: Transcribed text.
        """
        logger.info(f"Transcription complete: {text[:50]}...")

        try:
            # Inject text and update UI
            if text:
                success = self.text_injector.inject(text)
                if not success:
                    logger.error("Failed to inject text")
                else:
                    # Update main window
                    self.main_window.update_transcription(text)

                    # Add to recent transcriptions with error handling
                    try:
                        self._add_recent_transcription(text)
                    except Exception as e:
                        logger.warning(f"🟡 Failed to update recent menus: {e}")
                        # Continue execution - don't let menu errors break functionality

                    # Log transcript to disk if enabled
                    try:
                        self.transcript_logger.log_transcript(text)
                    except Exception as e:
                        logger.warning(f"🟡 Failed to log transcript: {e}")
        except Exception as e:
            logger.error(f"🛑 Error in transcription completion: {e}")
        finally:
            # ALWAYS reset state, regardless of errors above
            # This prevents the app from getting stuck in PROCESSING state
            logger.debug("Resetting state to IDLE")
            self.state = AppState.IDLE
            self._update_status()
            self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        """Clean up ASR worker thread safely with timeout and forced cleanup."""
        if not self.asr_worker:
            return

        try:
            logger.debug("🧹 Cleaning up ASR worker thread...")

            # First, try to signal the thread to quit gracefully
            self.asr_worker.quit()

            # Wait for thread to finish, but with a timeout
            if self.asr_worker.wait(5000):  # 5 second timeout
                logger.debug("✅ ASR worker terminated gracefully")
            else:
                logger.warning(
                    "⚠️ ASR worker didn't terminate gracefully, forcing cleanup"
                )

                # Try to terminate the thread forcefully
                try:
                    self.asr_worker.terminate()
                    if self.asr_worker.wait(2000):  # 2 second timeout for termination
                        logger.debug("✅ ASR worker terminated forcefully")
                    else:
                        logger.error("❌ ASR worker failed to terminate")
                except Exception as term_error:
                    logger.error(f"❌ Error terminating ASR worker: {term_error}")

            # Disconnect signals to prevent memory leaks
            try:
                self.asr_worker.transcription_complete.disconnect()
                self.asr_worker.error_occurred.disconnect()
                logger.debug("✅ ASR worker signals disconnected")
            except Exception as disconnect_error:
                logger.debug(f"Signal disconnect warning: {disconnect_error}")

            # Set to None regardless of termination success
            self.asr_worker = None
            logger.debug("✅ ASR worker reference cleared")

        except Exception as e:
            logger.error(f"❌ Error during ASR worker cleanup: {e}")
            # Force cleanup even if everything fails
            self.asr_worker = None

        # Force garbage collection after thread cleanup
        import gc

        gc.collect()

    @Slot(str)
    def _on_transcription_error(self, error: str) -> None:
        """Handle transcription error with robust cleanup.

        Args:
            error: Error message.
        """
        logger.error(f"🛑 Transcription error: {error}")

        try:
            # Show notification
            if self.tray_icon.supportsMessages():
                self.tray_icon.showMessage(
                    "SuperKeet Error",
                    f"Transcription failed: {error}",
                    QSystemTrayIcon.MessageIcon.Critical,
                    3000,
                )
        except Exception as e:
            logger.warning(f"🟡 Failed to show error notification: {e}")
        finally:
            # ALWAYS reset state, even if notification fails
            logger.debug("Resetting state to IDLE after error")
            self.state = AppState.IDLE
            self._update_icon()
            self._cleanup_worker()

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
        # Limit text length for menu display (unused variable removed)

        # Add to list
        self.recent_transcriptions.insert(0, text)
        if len(self.recent_transcriptions) > 5:
            self.recent_transcriptions.pop()

        # Update both tray and dock menus
        self._update_recent_menus()

    def _update_recent_menus(self) -> None:
        """Update both tray and dock recent transcription menus with error handling."""
        try:
            # Update tray menu (this should always work)
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
        except Exception as e:
            logger.warning(f"Failed to update tray recent menu: {e}")

        # Update dock menu if it exists - with comprehensive error handling
        if hasattr(self, "dock_recent_menu") and self.dock_recent_menu is not None:
            try:
                self.dock_recent_menu.clear()
                for i, transcript in enumerate(self.recent_transcriptions):
                    action_text = (
                        transcript[:50] + "..." if len(transcript) > 50 else transcript
                    )
                    action = QAction(action_text, self.dock_recent_menu)
                    action.triggered.connect(
                        lambda checked, t=transcript: self.text_injector.inject(t)
                    )
                    self.dock_recent_menu.addAction(action)
            except RuntimeError as e:
                logger.warning(
                    f"🟡 Dock menu update failed (menu deleted by system): {e}"
                )
                # Disable future dock menu updates
                self.dock_recent_menu = None
            except Exception as e:
                logger.warning(f"🟡 Unexpected dock menu error: {e}")
                self.dock_recent_menu = None

    @Slot(object)
    def _on_main_window_microphone_changed(self, device_index) -> None:
        """Handle microphone change from main window."""
        logger.info(f"🎙️ Tray app received microphone change to device: {device_index}")
        self.audio_recorder.update_device(device_index)

    @Slot()
    def _on_application_state_changed(self, state) -> None:
        """Handle application state changes (including dock clicks)."""
        # Handle dock icon activation - when user clicks dock icon
        try:
            from PySide6.QtCore import Qt

            if state == Qt.ApplicationState.ApplicationActive:
                # Show main window when dock icon is clicked and it's not visible
                if not self.main_window.isVisible():
                    logger.info("Dock icon clicked - showing main window")
                    self._on_show_window()
        except (AttributeError, ImportError):
            # Fallback for older Qt versions or import issues
            logger.debug(
                "Application state change detected, but unable to handle dock click"
            )

    @Slot()
    def _on_settings(self) -> None:
        """Handle settings action."""
        logger.info("Opening settings dialog")
        settings_dialog = SettingsDialog(None, self.audio_recorder)
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
            logger.info("🔍 Checking model cache status...")
            if not self.asr_transcriber._is_model_cached():
                if self.tray_icon.supportsMessages():
                    self.tray_icon.showMessage(
                        "SuperKeet",
                        "Downloading ASR model (~600MB). This is a one-time download...",  # noqa: E501
                        QSystemTrayIcon.MessageIcon.Information,
                        5000,
                    )

            # Load ASR model with progress feedback
            logger.info("🔄 Loading ASR model...")

            # Show loading notification
            if self.tray_icon.supportsMessages():
                if not self.asr_transcriber._is_model_cached():
                    self.tray_icon.showMessage(
                        "SuperKeet",
                        "Loading ASR model for the first time...",
                        QSystemTrayIcon.MessageIcon.Information,
                        3000,
                    )
                else:
                    self.tray_icon.showMessage(
                        "SuperKeet",
                        "Loading ASR model from cache...",
                        QSystemTrayIcon.MessageIcon.Information,
                        2000,
                    )

            # Load the model (now with tqdm progress indicators)
            self.asr_transcriber.load_model()
            logger.info("✅ ASR model loaded successfully")

            # Show completion notification
            if self.tray_icon.supportsMessages():
                self.tray_icon.showMessage(
                    "SuperKeet",
                    "ASR model loaded successfully! Ready for voice transcription.",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000,
                )

            # Start hotkey listener
            logger.info("🎹 Starting hotkey listener...")
            self.hotkey_listener.start()
            logger.info("✅ Hotkey listener started successfully")

            # Run event loop
            logger.info("🚀 Starting Qt application event loop...")
            exit_code = self.app.exec()
            logger.info(f"📋 Application event loop exited with code: {exit_code}")
            return exit_code

        except KeyboardInterrupt:
            logger.info("⛔ Application interrupted by user")
            self.cleanup()
            return 0
        except Exception as e:
            logger.error(f"🛑 Application error: {e}")
            logger.error(f"🛑 Error type: {type(e).__name__}")
            import traceback

            logger.error(f"🛑 Stack trace: {traceback.format_exc()}")
            self.cleanup()
            raise

    def cleanup(self) -> None:
        """Clean up resources with comprehensive memory management."""
        logger.info("🧹 Starting application cleanup...")

        # Clean up ASR worker thread first
        self._cleanup_worker()

        # Stop hotkey listener
        try:
            self.hotkey_listener.stop()
            logger.debug("✅ Hotkey listener stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping hotkey listener: {e}")

        # Stop audio recorder if it's running
        try:
            if (
                hasattr(self.audio_recorder, "recording")
                and self.audio_recorder.recording
            ):
                self.audio_recorder.stop()
            # Clear audio buffers
            if hasattr(self.audio_recorder, "clear_audio_buffers"):
                self.audio_recorder.clear_audio_buffers()
            logger.debug("✅ Audio recorder cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping audio recorder: {e}")

        # Clean up ASR transcriber (includes model unloading and timer cleanup)
        try:
            if hasattr(self.asr_transcriber, "cleanup"):
                self.asr_transcriber.cleanup()
            else:
                self.asr_transcriber.unload_model()
            logger.debug("✅ ASR transcriber cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up ASR transcriber: {e}")

        # Hide tray icon
        try:
            self.tray_icon.setVisible(False)
            logger.debug("✅ Tray icon hidden")
        except Exception as e:
            logger.warning(f"⚠️ Error hiding tray icon: {e}")

        # Force garbage collection
        import gc

        collected = gc.collect()
        logger.info(f"🗑️ Garbage collection: {collected} objects collected")

        logger.info("✅ Application cleanup completed")


# end src/ui/tray_app.py
