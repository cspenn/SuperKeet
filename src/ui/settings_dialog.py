# start src/ui/settings_dialog.py
"""Settings dialog for SuperKeet."""

from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.config.config_loader import config
from src.utils.logger import setup_logger
from src.audio.recorder import AudioRecorder

logger = setup_logger(__name__)


class HotkeyEdit(QWidget):
    """Custom widget for editing hotkey combinations."""

    hotkey_changed = Signal(list)

    def __init__(self, initial_hotkey: List[str]):
        super().__init__()
        self.hotkey_combo = initial_hotkey.copy()
        self._setup_ui()

    def _setup_ui(self):
        """Set up the hotkey editor UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.hotkey_label = QLabel(self._format_hotkey())
        self.hotkey_label.setStyleSheet(
            "border: 1px solid #ccc; padding: 5px; background: #f9f9f9;"
        )
        layout.addWidget(self.hotkey_label)

        self.change_button = QPushButton("Change")
        self.change_button.clicked.connect(self._change_hotkey)
        layout.addWidget(self.change_button)

    def _format_hotkey(self) -> str:
        """Format hotkey combination for display."""
        key_map = {"cmd": "⌘", "ctrl": "⌃", "alt": "⌥", "shift": "⇧", "space": "Space"}

        formatted = []
        for key in self.hotkey_combo:
            formatted.append(key_map.get(key, key.upper()))

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
        self.hotkey_changed.emit(self.hotkey_combo)

    def get_hotkey(self) -> List[str]:
        """Get current hotkey combination."""
        return self.hotkey_combo.copy()


class SettingsDialog(QDialog):
    """Settings dialog for SuperKeet configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SuperKeet Settings")
        self.setModal(True)
        self.setFixedSize(400, 300)

        self._setup_ui()
        self._load_current_settings()

        logger.info("Settings dialog initialized")

    def _setup_ui(self):
        """Set up the settings dialog UI."""
        layout = QVBoxLayout(self)

        # Form layout for settings
        form_layout = QFormLayout()

        # Hotkey setting
        self.hotkey_edit = HotkeyEdit(
            config.get("hotkey.combination", ["ctrl", "space"])
        )
        form_layout.addRow("Hotkey:", self.hotkey_edit)

        # Audio device setting
        self.audio_device_combo = QComboBox()
        self._populate_audio_devices()
        form_layout.addRow("Audio Device:", self.audio_device_combo)

        # Log level setting
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        form_layout.addRow("Log Level:", self.log_level_combo)

        # Text injection method
        self.injection_method_combo = QComboBox()
        self.injection_method_combo.addItem("Clipboard", "clipboard")
        self.injection_method_combo.addItem("Accessibility (Future)", "accessibility")
        self.injection_method_combo.setEnabled(
            False
        )  # Only clipboard supported for now
        form_layout.addRow("Text Injection:", self.injection_method_combo)
        
        # Auto-paste checkbox
        from PySide6.QtWidgets import QCheckBox
        self.auto_paste_checkbox = QCheckBox("Auto-paste after transcription")
        self.auto_paste_checkbox.setToolTip(
            "Automatically paste transcribed text (requires accessibility permissions)"
        )
        form_layout.addRow("", self.auto_paste_checkbox)
        
        # Transcript logging section
        from PySide6.QtWidgets import QGroupBox, QLineEdit, QPushButton, QHBoxLayout
        transcript_group = QGroupBox("Transcript Logging")
        transcript_layout = QFormLayout()
        
        # Enable transcript logging checkbox
        self.transcript_enabled_checkbox = QCheckBox("Save transcripts to disk")
        transcript_layout.addRow(self.transcript_enabled_checkbox)
        
        # Transcript directory
        dir_layout = QHBoxLayout()
        self.transcript_dir_edit = QLineEdit()
        self.transcript_dir_edit.setPlaceholderText("transcripts")
        dir_layout.addWidget(self.transcript_dir_edit)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_transcript_dir)
        dir_layout.addWidget(self.browse_button)
        transcript_layout.addRow("Directory:", dir_layout)
        
        # Transcript format
        self.transcript_format_combo = QComboBox()
        self.transcript_format_combo.addItem("Plain Text", "text")
        self.transcript_format_combo.addItem("JSON", "json")
        transcript_layout.addRow("Format:", self.transcript_format_combo)
        
        transcript_group.setLayout(transcript_layout)
        layout.addWidget(transcript_group)

        layout.addLayout(form_layout)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)

        layout.addWidget(button_box)
    
    def _browse_transcript_dir(self):
        """Browse for transcript directory."""
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Transcript Directory",
            self.transcript_dir_edit.text() or "."
        )
        if dir_path:
            self.transcript_dir_edit.setText(dir_path)

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
                # Add device to combo box with index as data
                self.audio_device_combo.addItem(device_name, device_index)
                
            logger.info(f"Found {len(devices)} audio input devices")
        except Exception as e:
            logger.error(f"Failed to get audio devices: {e}")

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
        
        # Transcript logging settings
        transcript_enabled = config.get("transcripts.enabled", False)
        self.transcript_enabled_checkbox.setChecked(transcript_enabled)
        
        transcript_dir = config.get("transcripts.directory", "transcripts")
        self.transcript_dir_edit.setText(transcript_dir)
        
        transcript_format = config.get("transcripts.format", "text")
        index = self.transcript_format_combo.findData(transcript_format)
        if index >= 0:
            self.transcript_format_combo.setCurrentIndex(index)

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
        transcript_enabled = self.transcript_enabled_checkbox.isChecked()
        transcript_dir = self.transcript_dir_edit.text() or "transcripts"
        transcript_format = self.transcript_format_combo.currentData()

        logger.info("Would save settings:")
        logger.info(f"  Hotkey: {new_hotkey}")
        logger.info(f"  Log Level: {new_log_level}")
        logger.info(f"  Audio Device: {new_device}")
        logger.info(f"  Injection Method: {new_method}")
        logger.info(f"  Auto-paste: {auto_paste}")
        logger.info(f"  Transcript logging: {transcript_enabled}")
        logger.info(f"  Transcript directory: {transcript_dir}")
        logger.info(f"  Transcript format: {transcript_format}")

        # TODO: Implement actual config file writing
        # For now, just update the in-memory config
        config.config["hotkey"]["combination"] = new_hotkey
        config.config["logging"]["level"] = new_log_level
        if new_device:
            config.config["audio"]["device"] = new_device
        config.config["text"]["method"] = new_method
        config.config["text"]["auto_paste"] = auto_paste
        
        # Transcript settings
        if "transcripts" not in config.config:
            config.config["transcripts"] = {}
        config.config["transcripts"]["enabled"] = transcript_enabled
        config.config["transcripts"]["directory"] = transcript_dir
        config.config["transcripts"]["format"] = transcript_format

    def accept(self):
        """Accept dialog and save settings."""
        self._save_settings()
        super().accept()


# end src/ui/settings_dialog.py
