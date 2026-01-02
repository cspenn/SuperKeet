# start src/ui/settings_dialog.py
"""Settings dialog for SuperKeet."""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from superkeet.audio.recorder import AudioRecorder
from superkeet.config.config_loader import config
from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class HotkeyEdit(QWidget):
    """Custom widget for editing hotkey combinations."""

    hotkey_changed = Signal(str)

    def __init__(self, initial_hotkey):
        super().__init__()

        # Handle both string and list formats
        if isinstance(initial_hotkey, str):
            # Parse string format like "ctrl+space"
            self.hotkey_combo = initial_hotkey.split("+")
        else:
            # Handle legacy list format
            self.hotkey_combo = (
                initial_hotkey.copy()
                if hasattr(initial_hotkey, "copy")
                else list(initial_hotkey)
            )

        self._setup_ui()

    def _setup_ui(self):
        """Set up the hotkey editor UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.hotkey_label = QLabel(self._format_hotkey())
        self.hotkey_label.setStyleSheet(
            "border: 1px solid #3A3A3A; padding: 8px; background: #2A2A2A; color: #EAEAEA; border-radius: 4px; font-weight: bold;"  # noqa: E501
        )
        layout.addWidget(self.hotkey_label)

        self.change_button = QPushButton("Change")
        self.change_button.clicked.connect(self._change_hotkey)
        layout.addWidget(self.change_button)

    def _format_hotkey(self) -> str:
        """Format hotkey combination for display."""
        key_map = {"cmd": "⌘", "ctrl": "⌃", "alt": "⌥", "shift": "⇧", "space": "Space"}

        formatted = [key_map.get(key, key.upper()) for key in self.hotkey_combo]

        return " + ".join(formatted)

    def _change_hotkey(self):
        """Open simple hotkey change dialog."""
        # For now, cycle through common combinations
        combinations = [
            ["ctrl", "space"],
            ["cmd", "shift", "space"],
            ["alt", "space"],
            ["cmd", "space"],
        ]

        try:
            current_index = combinations.index(self.hotkey_combo)
            next_index = (current_index + 1) % len(combinations)
        except ValueError:
            next_index = 0

        self.hotkey_combo = combinations[next_index]
        self.hotkey_label.setText(self._format_hotkey())
        self.hotkey_changed.emit("+".join(self.hotkey_combo))

    def get_hotkey(self) -> str:
        """Get current hotkey combination as string format."""
        return "+".join(self.hotkey_combo)


class SettingsDialog(QDialog):
    """Settings dialog for SuperKeet configuration."""

    def __init__(self, parent=None, audio_recorder=None):
        super().__init__(parent)
        self.setWindowTitle("SuperKeet Settings")
        self.setModal(True)
        self.audio_recorder = audio_recorder

        # Set responsive sizing
        self.setMinimumSize(650, 550)
        self.resize(700, 650)

        self._setup_ui()
        self._load_current_settings()

        logger.info("Settings dialog initialized")

    def _setup_ui(self):
        """Set up the settings dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Basic settings group
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QFormLayout(basic_group)

        # Hotkey setting
        self.hotkey_edit = HotkeyEdit(config.get("hotkey.combination", "ctrl+space"))
        basic_layout.addRow("Hotkey:", self.hotkey_edit)

        # Audio device setting
        self.audio_device_combo = QComboBox()
        self._populate_audio_devices()
        # Add real-time device change notification
        self.audio_device_combo.currentIndexChanged.connect(self._on_device_changed)
        basic_layout.addRow("Audio Device:", self.audio_device_combo)

        # Log level setting
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        basic_layout.addRow("Log Level:", self.log_level_combo)

        # Text injection method
        self.injection_method_combo = QComboBox()
        self.injection_method_combo.addItem("Clipboard", "clipboard")
        self.injection_method_combo.addItem("Accessibility (Future)", "accessibility")
        self.injection_method_combo.setEnabled(
            False
        )  # Only clipboard supported for now
        basic_layout.addRow("Text Injection:", self.injection_method_combo)

        # Auto-paste checkbox
        self.auto_paste_checkbox = QCheckBox("Auto-paste after transcription")
        self.auto_paste_checkbox.setToolTip(
            "Automatically paste transcribed text (requires accessibility permissions)"
        )
        basic_layout.addRow("", self.auto_paste_checkbox)

        content_layout.addWidget(basic_group)

        # Audio storage section
        audio_storage_group = QGroupBox("Audio Storage")
        audio_storage_layout = QFormLayout(audio_storage_group)

        # Enable audio storage checkbox
        self.audio_storage_enabled = QCheckBox("Save audio recordings")
        self.audio_storage_enabled.setToolTip("Save original audio recordings to disk")
        audio_storage_layout.addRow(self.audio_storage_enabled)

        # Audio storage directory
        audio_dir_layout = QHBoxLayout()
        self.audio_storage_dir = QLineEdit()
        self.audio_storage_dir.setPlaceholderText("audio_recordings")
        audio_dir_layout.addWidget(self.audio_storage_dir)

        self.browse_audio_button = QPushButton("Browse...")
        self.browse_audio_button.clicked.connect(self._browse_audio_storage_dir)
        audio_dir_layout.addWidget(self.browse_audio_button)
        audio_storage_layout.addRow("Directory:", audio_dir_layout)

        # Audio format
        self.audio_format_combo = QComboBox()
        self.audio_format_combo.addItem("WAV", "wav")
        self.audio_format_combo.addItem("MP3 (Future)", "mp3")
        self.audio_format_combo.addItem("FLAC (Future)", "flac")
        self.audio_format_combo.setEnabled(False)  # Only WAV supported for now
        audio_storage_layout.addRow("Format:", self.audio_format_combo)

        # Retention settings
        self.retention_days = QSpinBox()
        self.retention_days.setRange(1, 365)
        self.retention_days.setValue(30)
        self.retention_days.setSuffix(" days")
        audio_storage_layout.addRow("Retention:", self.retention_days)

        # Max files
        self.max_files = QSpinBox()
        self.max_files.setRange(10, 10000)
        self.max_files.setValue(1000)
        self.max_files.setSuffix(" files")
        audio_storage_layout.addRow("Max Files:", self.max_files)

        content_layout.addWidget(audio_storage_group)

        # Transcript logging section
        transcript_group = QGroupBox("Transcript Logging")
        transcript_layout = QFormLayout(transcript_group)

        # Enable transcript logging checkbox
        self.transcript_enabled_checkbox = QCheckBox("Save transcripts to disk")
        transcript_layout.addRow(self.transcript_enabled_checkbox)

        # Transcript directory
        transcript_dir_layout = QHBoxLayout()
        self.transcript_dir_edit = QLineEdit()
        self.transcript_dir_edit.setPlaceholderText("transcripts")
        transcript_dir_layout.addWidget(self.transcript_dir_edit)

        self.browse_transcript_button = QPushButton("Browse...")
        self.browse_transcript_button.clicked.connect(self._browse_transcript_dir)
        transcript_dir_layout.addWidget(self.browse_transcript_button)
        transcript_layout.addRow("Directory:", transcript_dir_layout)

        # Transcript format
        self.transcript_format_combo = QComboBox()
        self.transcript_format_combo.addItem("Plain Text", "text")
        self.transcript_format_combo.addItem("JSON", "json")
        transcript_layout.addRow("Format:", self.transcript_format_combo)

        content_layout.addWidget(transcript_group)

        # Debug section
        debug_group = QGroupBox("Debug Settings")
        debug_layout = QFormLayout(debug_group)

        # Save debug audio files
        self.debug_audio_checkbox = QCheckBox("Save debug audio files")
        self.debug_audio_checkbox.setToolTip(
            "Save audio files for debugging transcription issues"
        )
        debug_layout.addRow(self.debug_audio_checkbox)

        # Debug audio directory
        debug_dir_layout = QHBoxLayout()
        self.debug_audio_dir = QLineEdit()
        self.debug_audio_dir.setPlaceholderText("debug_audio")
        debug_dir_layout.addWidget(self.debug_audio_dir)

        self.browse_debug_button = QPushButton("Browse...")
        self.browse_debug_button.clicked.connect(self._browse_debug_audio_dir)
        debug_dir_layout.addWidget(self.browse_debug_button)
        debug_layout.addRow("Debug Directory:", debug_dir_layout)

        content_layout.addWidget(debug_group)

        # Set content in scroll area
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)

        layout.addWidget(button_box)

        # Apply dark theme styling
        self._apply_dark_theme()

        # Connect signals for enabling/disabling related controls
        self.audio_storage_enabled.toggled.connect(self._on_audio_storage_toggled)
        self.transcript_enabled_checkbox.toggled.connect(
            self._on_transcript_logging_toggled
        )
        self.debug_audio_checkbox.toggled.connect(self._on_debug_audio_toggled)

    def _apply_dark_theme(self):
        """Apply dark theme to match the main application."""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1E1E1E;
                color: #EAEAEA;
            }

            QLabel {
                color: #EAEAEA;
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

            QComboBox QAbstractItemView {
                background-color: #2A2A2A;
                color: #EAEAEA;
                selection-background-color: #007AFF;
            }

            QPushButton {
                background-color: #2A2A2A;
                color: #EAEAEA;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 6px 12px;
            }

            QPushButton:hover {
                background-color: #3A3A3A;
            }

            QPushButton:pressed {
                background-color: #1A1A1A;
            }

            QCheckBox {
                color: #EAEAEA;
                spacing: 8px;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                background-color: #2A2A2A;
            }

            QCheckBox::indicator:checked {
                background-color: #007AFF;
                border-color: #007AFF;
            }

            QCheckBox::indicator:checked:disabled {
                background-color: #8E8E93;
                border-color: #8E8E93;
            }

            QLineEdit {
                background-color: #2A2A2A;
                color: #EAEAEA;
                border: 1px solid #3A3A3A;
                padding: 5px;
                border-radius: 4px;
            }

            QLineEdit:focus {
                border-color: #007AFF;
            }

            QGroupBox {
                color: #EAEAEA;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #007AFF;
                font-weight: bold;
            }
        """
        )

    def _browse_audio_storage_dir(self):
        """Browse for audio storage directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Audio Storage Directory", self.audio_storage_dir.text() or "."
        )
        if dir_path:
            self.audio_storage_dir.setText(dir_path)

    def _browse_transcript_dir(self):
        """Browse for transcript directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Transcript Directory", self.transcript_dir_edit.text() or "."
        )
        if dir_path:
            self.transcript_dir_edit.setText(dir_path)

    def _browse_debug_audio_dir(self):
        """Browse for debug audio directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Debug Audio Directory", self.debug_audio_dir.text() or "."
        )
        if dir_path:
            self.debug_audio_dir.setText(dir_path)

    def _on_audio_storage_toggled(self, enabled: bool):
        """Handle audio storage checkbox toggle."""
        self.audio_storage_dir.setEnabled(enabled)
        self.browse_audio_button.setEnabled(enabled)
        self.audio_format_combo.setEnabled(False)  # Keep disabled for now
        self.retention_days.setEnabled(enabled)
        self.max_files.setEnabled(enabled)

    def _on_transcript_logging_toggled(self, enabled: bool):
        """Handle transcript logging checkbox toggle."""
        self.transcript_dir_edit.setEnabled(enabled)
        self.browse_transcript_button.setEnabled(enabled)
        self.transcript_format_combo.setEnabled(enabled)

    def _on_debug_audio_toggled(self, enabled: bool):
        """Handle debug audio checkbox toggle."""
        self.debug_audio_dir.setEnabled(enabled)
        self.browse_debug_button.setEnabled(enabled)

    def _populate_audio_devices(self):
        """Populate the audio device combo box with available devices."""
        # Add default device option
        self.audio_device_combo.addItem("Default Device", None)

        # Get available audio input devices
        try:
            audio_recorder = AudioRecorder()
            devices = audio_recorder.get_devices()

            for device in devices:
                device_name = device.get("name", "Unknown Device")
                device_index = device.get("index", 0)
                device_channels = device.get("channels", 1)
                # Add device to combo box with input channel info
                display_name = f"{device_name} ({device_channels} input)"
                self.audio_device_combo.addItem(display_name, device_index)

            logger.info(f"Found {len(devices)} audio input devices")
        except Exception as e:
            logger.error(f"Failed to get audio devices: {e}")

    @Slot()
    def _on_device_changed(self):
        """Handle device selection change to provide real-time feedback."""
        device_index = self.audio_device_combo.currentData()
        device_name = self.audio_device_combo.currentText()
        logger.info(
            f"Audio device selection changed to: [{device_index}] {device_name}"
        )

    def _load_current_settings(self):
        """Load current settings into the dialog."""
        # Log level
        current_log_level = config.get("logging.level", "INFO")
        index = self.log_level_combo.findText(current_log_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)

        # Audio device
        current_device = config.get("audio.device")
        if current_device is None:
            self.audio_device_combo.setCurrentIndex(0)  # Default device
        else:
            # Find device by index
            index = self.audio_device_combo.findData(current_device)
            if index >= 0:
                self.audio_device_combo.setCurrentIndex(index)
            else:
                # Device not found, use default
                self.audio_device_combo.setCurrentIndex(0)

        # Text injection method
        current_method = config.get("text.method", "clipboard")
        index = self.injection_method_combo.findData(current_method)
        if index >= 0:
            self.injection_method_combo.setCurrentIndex(index)

        # Auto-paste preference
        auto_paste = config.get("text.auto_paste", True)  # Default to True
        self.auto_paste_checkbox.setChecked(auto_paste)

        # Audio storage settings
        audio_storage_enabled = config.get("audio_storage.enabled", False)
        self.audio_storage_enabled.setChecked(audio_storage_enabled)

        audio_storage_dir = config.get("audio_storage.directory", "audio_recordings")
        self.audio_storage_dir.setText(audio_storage_dir)

        audio_format = config.get("audio_storage.format", "wav")
        index = self.audio_format_combo.findData(audio_format)
        if index >= 0:
            self.audio_format_combo.setCurrentIndex(index)

        retention_days = config.get("audio_storage.retention_days", 30)
        self.retention_days.setValue(retention_days)

        max_files = config.get("audio_storage.max_files", 1000)
        self.max_files.setValue(max_files)

        # Transcript logging settings
        transcript_enabled = config.get("transcripts.enabled", False)
        self.transcript_enabled_checkbox.setChecked(transcript_enabled)

        transcript_dir = config.get("transcripts.directory", "transcripts")
        self.transcript_dir_edit.setText(transcript_dir)

        transcript_format = config.get("transcripts.format", "text")
        index = self.transcript_format_combo.findData(transcript_format)
        if index >= 0:
            self.transcript_format_combo.setCurrentIndex(index)

        # Debug settings
        debug_audio = config.get("debug.save_audio_files", False)
        self.debug_audio_checkbox.setChecked(debug_audio)

        debug_dir = config.get("debug.audio_debug_dir", "debug_audio")
        self.debug_audio_dir.setText(debug_dir)

        # Initialize control states
        self._on_audio_storage_toggled(audio_storage_enabled)
        self._on_transcript_logging_toggled(transcript_enabled)
        self._on_debug_audio_toggled(debug_audio)

    def _apply_settings(self):
        """Apply settings without closing dialog."""
        self._save_settings()
        logger.info("Settings applied")

    def _save_settings(self):
        """Save settings to configuration."""
        # Note: This is a simplified implementation
        # In a full implementation, you would save to the config file
        # and notify the application to reload configuration

        new_hotkey = self.hotkey_edit.get_hotkey()
        new_log_level = self.log_level_combo.currentText()
        new_device = self.audio_device_combo.currentData()
        new_method = self.injection_method_combo.currentData()
        auto_paste = self.auto_paste_checkbox.isChecked()

        # Audio storage settings
        audio_storage_enabled = self.audio_storage_enabled.isChecked()
        audio_storage_dir = self.audio_storage_dir.text() or "audio_recordings"
        audio_format = self.audio_format_combo.currentData()
        retention_days = self.retention_days.value()
        max_files = self.max_files.value()

        # Transcript settings
        transcript_enabled = self.transcript_enabled_checkbox.isChecked()
        transcript_dir = self.transcript_dir_edit.text() or "transcripts"
        transcript_format = self.transcript_format_combo.currentData()

        # Debug settings
        debug_audio = self.debug_audio_checkbox.isChecked()
        debug_dir = self.debug_audio_dir.text() or "debug_audio"

        logger.info("Would save settings:")
        logger.info(f"  Hotkey: {new_hotkey}")
        logger.info(f"  Log Level: {new_log_level}")
        logger.info(f"  Audio Device: {new_device}")
        logger.info(f"  Injection Method: {new_method}")
        logger.info(f"  Auto-paste: {auto_paste}")
        logger.info(f"  Audio storage: {audio_storage_enabled}")
        logger.info(f"  Audio storage directory: {audio_storage_dir}")
        logger.info(f"  Audio format: {audio_format}")
        logger.info(f"  Retention days: {retention_days}")
        logger.info(f"  Max files: {max_files}")
        logger.info(f"  Transcript logging: {transcript_enabled}")
        logger.info(f"  Transcript directory: {transcript_dir}")
        logger.info(f"  Transcript format: {transcript_format}")
        logger.info(f"  Debug audio: {debug_audio}")
        logger.info(f"  Debug directory: {debug_dir}")

        # Implement actual config file writing with validation and backup
        try:
            # Update configuration values
            config.set("hotkey.combination", new_hotkey)
            config.set("logging.level", new_log_level)
            config.set("text.method", new_method)
            config.set("text.auto_paste", auto_paste)

            # Audio storage settings
            config.set("audio_storage.enabled", audio_storage_enabled)
            config.set("audio_storage.directory", audio_storage_dir)
            config.set("audio_storage.format", audio_format)
            config.set("audio_storage.retention_days", retention_days)
            config.set("audio_storage.max_files", max_files)

            # Transcript settings
            config.set("transcripts.enabled", transcript_enabled)
            config.set("transcripts.directory", transcript_dir)
            config.set("transcripts.format", transcript_format)

            # Debug settings
            config.set("debug.save_audio_files", debug_audio)
            config.set("debug.audio_debug_dir", debug_dir)

            # Save audio device with validation (preserve existing validation logic)
            if new_device is not None:
                import sounddevice as sd

                device_info = sd.query_devices(new_device)
                if device_info["max_input_channels"] == 0:
                    logger.error(
                        f"🛑 Device {new_device} ({device_info['name']}) does not support input"  # noqa: E501
                    )
                    self._show_error_message(
                        "Invalid Device Selection",
                        f"The selected device '{device_info['name']}' is an output-only device.\n"  # noqa: E501
                        "Please select a device that supports audio input (microphone).",  # noqa: E501
                    )
                    return
                logger.info(
                    f"✅ Validated device {new_device}: {device_info['name']} ({device_info['max_input_channels']} input channels)"  # noqa: E501
                )
                config.set("audio.device", new_device)
            else:
                config.set("audio.device", None)

            # Apply device change immediately if audio recorder is available
            if self.audio_recorder:
                self.audio_recorder.update_device(new_device)
                logger.info(f"🔧 Applied audio device change to: {new_device}")

            # Save configuration to disk with backup and validation
            config.save()
            logger.info("💾 Configuration saved successfully")

            # Show success message to user
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Settings Saved",
                "Your settings have been saved successfully and will persist between sessions.",  # noqa: E501
            )

            self.accept()

        except Exception as e:
            logger.error(f"🛑 Failed to save settings: {e}")
            self._show_error_message(
                "Configuration Error",
                f"Failed to save settings to configuration file:\n{e}\n\n"
                "Your settings have not been saved. Please check file permissions and try again.",  # noqa: E501
            )
            return

    def _show_error_message(self, title: str, message: str) -> None:
        """Show an error message dialog to the user.

        Args:
            title: Dialog title
            message: Error message to display
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def accept(self):
        """Accept dialog and save settings."""
        self._save_settings()
        super().accept()


# end src/ui/settings_dialog.py
