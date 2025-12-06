# start src/ui/main_window.py
"""
Main application window for SuperKeet.
Provides detailed feedback and control with a sleek, responsive UI.
"""

import logging

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from superkeet.config.config_loader import config

from .audio_animation_widget import AudioAnimationWidget
from .batch_progress_dialog import BatchProgressDialog
from .drop_zone_widget import DropZoneWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window with waveform visualization and transcription output."""

    # Signal emitted when microphone selection changes
    microphone_changed = Signal(object)  # object to handle None or integer

    def __init__(self):
        super().__init__()
        self.setWindowTitle("● SuperKeet")
        self.setFixedSize(600, 600)

        # State tracking - Initialize before setup_ui
        self._current_state = "idle"
        self._status_dot_timer = QTimer()
        self._status_dot_timer.timeout.connect(self._pulse_status_dot)
        self._pulse_state = 0

        # Clipboard feedback timer
        self._clipboard_timer = QTimer()
        self._clipboard_timer.timeout.connect(self._reset_hint_text)
        self._clipboard_timer.setSingleShot(True)

        # Set up the UI
        self._setup_ui()
        self._apply_styles()
        self._setup_shortcuts()

        logger.info("🟢 MainWindow initialized")

    def _format_hotkey_display(self, hotkey_combo) -> str:
        """Format hotkey combination for display.

        Args:
            hotkey_combo: Hotkey combination (string or list format).

        Returns:
            Formatted hotkey string for display.
        """
        key_map = {"cmd": "⌘", "ctrl": "⌃", "alt": "⌥", "shift": "⇧", "space": "Space"}

        # Handle both string and list formats
        if isinstance(hotkey_combo, str):
            keys = hotkey_combo.split("+")
        else:
            keys = hotkey_combo

        formatted = []
        for key in keys:
            formatted.append(key_map.get(key, key.upper()))

        return "+".join(formatted)

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Cmd+, (Command+Comma) for Settings - standard macOS shortcut
        settings_shortcut = QShortcut(QKeySequence("Cmd+,"), self)
        settings_shortcut.activated.connect(self._show_settings)

        logger.info("🟢 MainWindow keyboard shortcuts set up (Cmd+, for settings)")

    def _setup_ui(self):
        """Set up the UI components."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Audio animation area
        self.audio_animation = AudioAnimationWidget()
        self.audio_animation.setMinimumHeight(150)
        main_layout.addWidget(self.audio_animation)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFixedHeight(1)
        main_layout.addWidget(separator1)

        # Transcription output area
        self.transcription_output = QTextEdit()
        self.transcription_output.setReadOnly(True)
        self.transcription_output.setPlaceholderText(
            "The transcribed text will appear here in real-time."
        )
        self.transcription_output.setMinimumHeight(120)
        main_layout.addWidget(self.transcription_output)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFixedHeight(1)
        main_layout.addWidget(separator2)

        # Drop zone for batch processing
        self.drop_zone = DropZoneWidget()
        self.drop_zone.setMinimumHeight(80)
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        main_layout.addWidget(self.drop_zone)

        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFixedHeight(1)
        main_layout.addWidget(separator3)

        # Hint text - use current hotkey configuration
        hotkey_combo = config.get("hotkey.combination", "ctrl+space")
        hotkey_display = self._format_hotkey_display(hotkey_combo)
        self.hint_label = QLabel(
            f"Hold [{hotkey_display}] to speak. Release to transcribe."
        )
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setFixedHeight(40)
        main_layout.addWidget(self.hint_label)

        # Footer
        footer = self._create_footer()
        main_layout.addWidget(footer)

    def _create_header(self) -> QWidget:
        """Create the header widget with status indicator."""
        header = QWidget()
        header.setFixedHeight(40)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(10, 0, 10, 0)

        # Status dot and title
        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(20)
        layout.addWidget(self.status_dot)

        title = QLabel("SuperKeet")
        title.setFont(QFont("", 14, QFont.Bold))
        layout.addWidget(title)

        layout.addStretch()

        return header

    def _create_footer(self) -> QWidget:
        """Create the footer widget with microphone selector and settings."""
        footer = QWidget()
        footer.setFixedHeight(50)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 0, 10, 0)

        # Microphone selector
        mic_icon = QLabel("🎙️")
        layout.addWidget(mic_icon)

        self.mic_selector = QComboBox()
        self.mic_selector.setMinimumWidth(200)
        self._populate_audio_devices()
        # Connect the microphone selector to update the audio device
        self.mic_selector.currentIndexChanged.connect(self._on_microphone_changed)
        layout.addWidget(self.mic_selector)

        layout.addStretch()

        # Settings button
        self.settings_btn = QPushButton("⚙️")
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(self._show_settings)
        layout.addWidget(self.settings_btn)

        return footer

    def _populate_audio_devices(self):
        """Populate the microphone selector with available devices."""
        # Add default device option
        self.mic_selector.addItem("Default Device", None)

        # Get available audio input devices
        try:
            from superkeet.audio.recorder import AudioRecorder

            audio_recorder = AudioRecorder()
            devices = audio_recorder.get_devices()

            for device in devices:
                device_name = device.get("name", "Unknown Device")
                device_index = device.get("index", 0)
                # Add device to combo box with index as data
                self.mic_selector.addItem(device_name, device_index)

            logger.info(f"Found {len(devices)} audio input devices")

            # Set current selection based on config
            current_device = config.get("audio.device")
            if current_device is None:
                self.mic_selector.setCurrentIndex(0)  # Default device
            else:
                # Find device by index
                for i in range(self.mic_selector.count()):
                    if self.mic_selector.itemData(i) == current_device:
                        self.mic_selector.setCurrentIndex(i)
                        break

        except Exception as e:
            logger.error(f"Failed to get audio devices: {e}")

    def _apply_styles(self):
        """Apply the dark mode stylesheet."""
        stylesheet = """
        QMainWindow {
            background-color: #1E1E1E;
        }

        QLabel {
            color: #EAEAEA;
            background-color: transparent;
        }

        QLabel#status_dot_gray {
            color: #8E8E93;
        }

        QLabel#status_dot_blue {
            color: #007AFF;
        }

        QLabel#status_dot_yellow {
            color: #FFD60A;
        }

        QLabel#status_dot_red {
            color: #FF3B30;
        }

        QTextEdit {
            background-color: #2A2A2A;
            color: #EAEAEA;
            border: none;
            padding: 10px;
            font-size: 14px;
        }

        QTextEdit::placeholder {
            color: #8E8E93;
        }

        QComboBox {
            background-color: #2A2A2A;
            color: #EAEAEA;
            border: 1px solid #3A3A3A;
            padding: 5px;
            border-radius: 4px;
        }

        QComboBox::drop-down {
            border: none;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #8E8E93;
            margin-right: 5px;
        }

        QPushButton {
            background-color: #2A2A2A;
            color: #EAEAEA;
            border: 1px solid #3A3A3A;
            border-radius: 4px;
        }

        QPushButton:hover {
            background-color: #3A3A3A;
        }

        QFrame[frameShape="4"] {
            background-color: #3A3A3A;
        }

        QLabel#hint {
            color: #8E8E93;
            font-size: 13px;
        }
        """
        self.setStyleSheet(stylesheet)

        # Apply specific styles
        self.hint_label.setObjectName("hint")
        self._update_status_dot_color()

    def _update_status_dot_color(self):
        """Update the status dot color based on current state."""
        if self._current_state == "idle":
            self.status_dot.setObjectName("status_dot_gray")
            self.status_dot.setStyleSheet("color: #8E8E93;")
        elif self._current_state == "recording":
            self.status_dot.setObjectName("status_dot_blue")
            self.status_dot.setStyleSheet("color: #007AFF;")
        elif self._current_state == "processing":
            self.status_dot.setObjectName("status_dot_yellow")
            self.status_dot.setStyleSheet("color: #FFD60A;")
        elif self._current_state == "error":
            self.status_dot.setObjectName("status_dot_red")
            self.status_dot.setStyleSheet("color: #FF3B30;")

    def _pulse_status_dot(self):
        """Pulse the status dot for recording state."""
        if self._current_state != "recording":
            return

        # Create pulsing effect
        self._pulse_state = (self._pulse_state + 1) % 10
        opacity = 0.5 + 0.5 * abs(5 - self._pulse_state) / 5

        color = f"rgba(0, 122, 255, {opacity})"
        self.status_dot.setStyleSheet(f"color: {color};")

    def _start_processing_animation(self):
        """Start processing animation with animated dots."""
        if not hasattr(self, "_processing_timer"):
            self._processing_timer = QTimer()
            self._processing_timer.timeout.connect(self._animate_processing_dots)
            self._processing_dots_count = 0

        self._processing_timer.start(500)  # Update every 500ms

    def _animate_processing_dots(self):
        """Animate processing dots."""
        if self._current_state != "processing":
            if hasattr(self, "_processing_timer"):
                self._processing_timer.stop()
            return

        # Cycle through 0-3 dots
        self._processing_dots_count = (self._processing_dots_count + 1) % 4
        dots = "." * self._processing_dots_count
        self.hint_label.setText(f"Processing transcription{dots}")

    @Slot(str)
    def update_state(self, state: str):
        """Update the UI state."""
        self._current_state = state.lower()
        logger.info(f"🟢 MainWindow state changed to: {self._current_state}")

        # Update status dot
        self._update_status_dot_color()

        # Handle pulsing animation
        if self._current_state == "recording":
            self._status_dot_timer.start(100)  # Pulse every 100ms
            self.hint_label.setText("Release to transcribe.")
            self.audio_animation.start_recording()
        else:
            self._status_dot_timer.stop()

            if self._current_state == "processing":
                self.hint_label.setText("Processing transcription...")
                self.audio_animation.stop_recording()
                # Add visual feedback with animated dots
                self._start_processing_animation()
            elif self._current_state == "idle":
                # Don't immediately reset hint text if we just copied to clipboard
                if not self._clipboard_timer.isActive():
                    hotkey_combo = config.get("hotkey.combination", "ctrl+space")
                    hotkey_display = self._format_hotkey_display(hotkey_combo)
                    self.hint_label.setText(
                        f"Hold [{hotkey_display}] to speak. Release to transcribe."
                    )
                self.audio_animation.clear()

    @Slot(str)
    def update_transcription(self, text: str):
        """Update the transcription output."""
        self.transcription_output.setPlainText(text)

        # Show clipboard feedback
        self.hint_label.setText("Copied to clipboard!")
        self._clipboard_timer.start(2000)  # Reset after 2 seconds

    def _reset_hint_text(self):
        """Reset hint text to default."""
        if self._current_state == "idle":
            hotkey_combo = config.get("hotkey.combination", "ctrl+space")
            hotkey_display = self._format_hotkey_display(hotkey_combo)
            self.hint_label.setText(
                f"Hold [{hotkey_display}] to speak. Release to transcribe."
            )

    def update_microphone_list(self, devices: list):
        """Update the microphone selector with available devices."""
        self.mic_selector.clear()
        for device in devices:
            self.mic_selector.addItem(device)

    def _on_microphone_changed(self, index: int):
        """Handle microphone selection change."""
        device_index = self.mic_selector.itemData(index)
        device_name = self.mic_selector.itemText(index)
        logger.info(
            f"🎙️ Main window microphone changed to: [{device_index}] {device_name}"
        )

        # Emit signal to notify that the microphone selection has changed
        # The tray app will connect to this signal and update the audio recorder
        self.microphone_changed.emit(device_index)

    def _show_settings(self):
        """Show settings dialog."""
        # Import here to avoid circular imports
        from .settings_dialog import SettingsDialog

        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()

    def _on_files_dropped(self, file_paths):
        """Handle files dropped on the drop zone.

        Args:
            file_paths: List of Path objects representing dropped files
        """
        if not file_paths:
            logger.warning("🟡 No valid files dropped")
            return

        logger.info(f"📁 Starting batch transcription for {len(file_paths)} files")

        # Create and show progress dialog
        progress_dialog = BatchProgressDialog(file_paths, self)
        progress_dialog.batch_cancelled.connect(self._on_batch_cancelled)

        # Start processing
        progress_dialog.start_processing()

        # Show the dialog
        result = progress_dialog.exec()

        if result == progress_dialog.DialogCode.Accepted:
            logger.info("🏁 Batch transcription completed")
        else:
            logger.info("🛑 Batch transcription cancelled")

        # Reset drop zone state
        self.drop_zone.reset_state()

    def _on_batch_cancelled(self):
        """Handle batch processing cancellation."""
        logger.info("🛑 Batch processing cancelled by user")
        self.drop_zone.reset_state()

    def cleanup_timers(self) -> None:
        """Clean up all QTimer instances to prevent memory leaks."""
        logger.debug("🧹 Cleaning up MainWindow timers")

        self._cleanup_timer("_status_dot_timer", "Status dot timer")
        self._cleanup_timer("_clipboard_timer", "Clipboard timer")
        self._cleanup_timer("_processing_timer", "Processing timer")

        # Clean up audio animation widget timers
        if hasattr(self, "audio_animation") and self.audio_animation is not None:
            try:
                if hasattr(self.audio_animation, "cleanup_timers"):
                    self.audio_animation.cleanup_timers()
                logger.debug("✅ Audio animation timers cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up audio animation: {e}")

    def _cleanup_timer(self, timer_attr: str, debug_name: str) -> None:
        """Clean up a specific timer attribute.

        Args:
            timer_attr: Name of the timer attribute
            debug_name: Human readable name for logging
        """
        if hasattr(self, timer_attr):
            timer = getattr(self, timer_attr)
            if timer is not None:
                try:
                    timer.stop()
                    timer.deleteLater()
                    setattr(self, timer_attr, None)
                    logger.debug(f"✅ {debug_name} cleaned up")
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning up {debug_name}: {e}")

    def closeEvent(self, event):  # noqa: N802
        """Handle window close event with proper cleanup."""
        # Clean up timers before closing
        self.cleanup_timers()

        # Don't quit the app, just hide the window
        event.ignore()
        self.hide()


# end src/ui/main_window.py
