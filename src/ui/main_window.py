# start src/ui/main_window.py
"""
Main application window for SuperKeet.
Provides detailed feedback and control with a sleek, responsive UI.
"""

import logging

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont
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

from src.config.config_loader import config

from .waveform_widget import WaveformWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window with waveform visualization and transcription output."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("● SuperKeet")
        self.setFixedSize(600, 500)

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

        logger.info("🟢 MainWindow initialized")

    def _format_hotkey_display(self, hotkey_combo: list) -> str:
        """Format hotkey combination for display.

        Args:
            hotkey_combo: List of key names.

        Returns:
            Formatted hotkey string for display.
        """
        key_map = {"cmd": "⌘", "ctrl": "⌃", "alt": "⌥", "shift": "⇧", "space": "Space"}

        formatted = []
        for key in hotkey_combo:
            formatted.append(key_map.get(key, key.upper()))

        return "+".join(formatted)

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

        # Waveform area
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.setMinimumHeight(150)
        main_layout.addWidget(self.waveform_widget)

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

        # Hint text - use current hotkey configuration
        hotkey_combo = config.get("hotkey.combination", ["ctrl", "space"])
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
        self.mic_selector.addItem("Default Device")
        
        # Get available audio input devices
        try:
            from src.audio.recorder import AudioRecorder
            audio_recorder = AudioRecorder()
            devices = audio_recorder.get_devices()
            
            for device in devices:
                device_name = device.get("name", "Unknown Device")
                self.mic_selector.addItem(device_name)
                
            logger.info(f"Found {len(devices)} audio input devices")
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
            self.waveform_widget.start_recording()
        else:
            self._status_dot_timer.stop()

            if self._current_state == "processing":
                self.hint_label.setText("Processing...")
                self.waveform_widget.stop_recording()
            elif self._current_state == "idle":
                # Don't immediately reset hint text if we just copied to clipboard
                if not self._clipboard_timer.isActive():
                    hotkey_combo = config.get("hotkey.combination", ["ctrl", "space"])
                    hotkey_display = self._format_hotkey_display(hotkey_combo)
                    self.hint_label.setText(
                        f"Hold [{hotkey_display}] to speak. Release to transcribe."
                    )
                self.waveform_widget.clear()

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
            hotkey_combo = config.get("hotkey.combination", ["ctrl", "space"])
            hotkey_display = self._format_hotkey_display(hotkey_combo)
            self.hint_label.setText(
                f"Hold [{hotkey_display}] to speak. Release to transcribe."
            )

    def update_microphone_list(self, devices: list):
        """Update the microphone selector with available devices."""
        self.mic_selector.clear()
        for device in devices:
            self.mic_selector.addItem(device)

    def closeEvent(self, event):
        """Handle window close event."""
        # Don't quit the app, just hide the window
        event.ignore()
        self.hide()
        logger.info("🟢 MainWindow hidden")

    def _show_settings(self):
        """Show settings dialog."""
        # Import here to avoid circular imports
        from .settings_dialog import SettingsDialog

        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()


# end src/ui/main_window.py
