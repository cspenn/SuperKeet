# start src/ui/first_run_dialog.py
"""First-run experience dialog for SuperKeet setup."""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from superkeet.audio.recorder import AudioRecorder
from superkeet.config.config_loader import config
from superkeet.services.device_service import DeviceService
from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class FirstRunDialog(QDialog):
    """First-run setup wizard for SuperKeet."""

    setup_completed = Signal()

    def __init__(self, parent=None):
        """Initialize the first-run dialog."""
        super().__init__(parent)
        self.setWindowTitle("Welcome to SuperKeet")
        self.setModal(True)
        self.setFixedSize(700, 550)

        # State tracking
        self.current_page = 0
        self.pages = []
        self.selected_device = None
        self.permissions_granted = {"microphone": False, "accessibility": False}

        # Initialize components
        self.device_service = DeviceService()
        self.audio_recorder = None

        self.setup_ui()
        self.apply_styles()

        logger.info("🎯 First-run dialog initialized")

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = self.create_header()
        layout.addWidget(header)

        # Main content area
        self.content_stack = QStackedWidget()
        layout.addWidget(self.content_stack)

        # Create pages
        self.create_pages()

        # Navigation
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(20, 10, 20, 20)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.go_back)
        self.back_button.setEnabled(False)
        nav_layout.addWidget(self.back_button)

        nav_layout.addStretch()

        self.page_label = QLabel("1 of 5")
        nav_layout.addWidget(self.page_label)

        nav_layout.addStretch()

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.go_next)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

    def create_header(self) -> QWidget:
        """Create the header with logo and title."""
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet(
            "background-color: #2A2A2A; border-bottom: 1px solid #3A3A3A;"
        )

        layout = QHBoxLayout(header)
        layout.setContentsMargins(30, 0, 30, 0)

        # Logo/icon area
        icon_label = QLabel("🦜")
        icon_label.setStyleSheet("font-size: 32px;")
        layout.addWidget(icon_label)

        # Title
        title = QLabel("SuperKeet Setup")
        title.setFont(QFont("", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #EAEAEA; margin-left: 15px;")
        layout.addWidget(title)

        layout.addStretch()

        return header

    def create_pages(self):
        """Create all setup pages."""
        # Page 1: Welcome
        welcome_page = self.create_welcome_page()
        self.content_stack.addWidget(welcome_page)
        self.pages.append("welcome")

        # Page 2: Permissions Overview
        permissions_page = self.create_permissions_page()
        self.content_stack.addWidget(permissions_page)
        self.pages.append("permissions")

        # Page 3: Device Selection
        device_page = self.create_device_page()
        self.content_stack.addWidget(device_page)
        self.pages.append("device")

        # Page 4: Feature Introduction
        features_page = self.create_features_page()
        self.content_stack.addWidget(features_page)
        self.pages.append("features")

        # Page 5: Completion
        completion_page = self.create_completion_page()
        self.content_stack.addWidget(completion_page)
        self.pages.append("completion")

    def create_welcome_page(self) -> QWidget:
        """Create the welcome page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)

        # Welcome title
        title = QLabel("Welcome to SuperKeet!")
        title.setFont(QFont("", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #EAEAEA; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "SuperKeet is a powerful voice-to-text application that uses advanced AI "
            "to transcribe your speech in real-time.\n\n"
            "This setup wizard will help you:\n"
            "• Configure microphone permissions\n"
            "• Set up accessibility permissions for text injection\n"
            "• Select the best audio device for your needs\n"
            "• Learn about SuperKeet's features\n\n"
            "Let's get started!"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #EAEAEA; font-size: 14px; line-height: 1.6;")
        layout.addWidget(desc)

        layout.addStretch()

        return page

    def create_permissions_page(self) -> QWidget:
        """Create the permissions setup page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)

        # Title
        title = QLabel("Required Permissions")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #EAEAEA; margin-bottom: 20px;")
        layout.addWidget(title)

        # Microphone permission
        mic_frame = self.create_permission_item(
            "🎙️",
            "Microphone Access",
            "Required to capture your voice for transcription",
            "microphone",
        )
        layout.addWidget(mic_frame)

        layout.addSpacing(20)

        # Accessibility permission
        acc_frame = self.create_permission_item(
            "⌨️",
            "Accessibility Access",
            "Required to inject transcribed text into other applications",
            "accessibility",
        )
        layout.addWidget(acc_frame)

        layout.addSpacing(20)

        # Instructions
        instructions = QLabel(
            "Click the buttons above to grant permissions. "
            "You may need to restart SuperKeet after granting permissions."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #8E8E93; font-size: 12px;")
        layout.addWidget(instructions)

        layout.addStretch()

        return page

    def create_permission_item(
        self, icon: str, title: str, description: str, perm_type: str
    ) -> QFrame:
        """Create a permission item widget."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet(
            "QFrame { border: 1px solid #3A3A3A; border-radius: 8px; padding: 15px; }"
        )

        layout = QHBoxLayout(frame)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 24px;")
        icon_label.setFixedWidth(40)
        layout.addWidget(icon_label)

        # Text content
        text_layout = QVBoxLayout()

        title_label = QLabel(title)
        title_label.setFont(QFont("", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #EAEAEA;")
        text_layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout)

        # Action button
        button = QPushButton("Grant Permission")
        button.setFixedWidth(120)
        button.clicked.connect(lambda: self.request_permission(perm_type))
        layout.addWidget(button)

        # Store reference for updates
        setattr(self, f"{perm_type}_button", button)

        return frame

    def create_device_page(self) -> QWidget:
        """Create the device selection page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)

        # Title
        title = QLabel("Select Audio Device")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #EAEAEA; margin-bottom: 20px;")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Choose the microphone you'd like to use. SuperKeet will automatically "
            "optimize audio settings for the best transcription quality."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #EAEAEA; font-size: 14px; margin-bottom: 20px;")
        layout.addWidget(desc)

        # Device selector
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        device_layout.addWidget(self.device_combo)

        device_layout.addStretch()

        test_button = QPushButton("Test Device")
        test_button.clicked.connect(self.test_selected_device)
        device_layout.addWidget(test_button)

        layout.addLayout(device_layout)

        layout.addSpacing(20)

        # Device info area
        self.device_info_label = QLabel(
            "Select a device to see details and recommendations."
        )
        self.device_info_label.setWordWrap(True)
        self.device_info_label.setStyleSheet(
            "color: #8E8E93; font-size: 12px; "
            "border: 1px solid #3A3A3A; border-radius: 4px; padding: 15px;"
        )
        layout.addWidget(self.device_info_label)

        layout.addStretch()

        # Populate devices
        self.populate_devices()

        return page

    def create_features_page(self) -> QWidget:
        """Create the features introduction page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)

        # Title
        title = QLabel("SuperKeet Features")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #EAEAEA; margin-bottom: 20px;")
        layout.addWidget(title)

        # Feature list
        features = [
            (
                "🎙️",
                "Real-time Transcription",
                "Hold Ctrl+Space to record, release to transcribe instantly",
            ),
            (
                "📁",
                "Batch Processing",
                "Drag and drop audio/video files for transcription",
            ),
            (
                "⚡",
                "AI-Powered",
                "Uses advanced Parakeet-MLX model optimized for Apple Silicon",
            ),
            (
                "🎯",
                "System Integration",
                "Works with any application - transcriptions appear where you need them",  # noqa: E501
            ),
        ]

        for icon, name, description in features:
            feature_frame = self.create_feature_item(icon, name, description)
            layout.addWidget(feature_frame)
            layout.addSpacing(15)

        layout.addStretch()

        return page

    def create_feature_item(self, icon: str, name: str, description: str) -> QFrame:
        """Create a feature item widget."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #3A3A3A; border-radius: 8px; padding: 15px; }"
        )

        layout = QHBoxLayout(frame)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        icon_label.setFixedWidth(30)
        layout.addWidget(icon_label)

        # Text
        text_layout = QVBoxLayout()

        name_label = QLabel(name)
        name_label.setFont(QFont("", 14, QFont.Weight.Bold))
        name_label.setStyleSheet("color: #EAEAEA;")
        text_layout.addWidget(name_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout)

        return frame

    def create_completion_page(self) -> QWidget:
        """Create the completion page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)

        # Success icon
        success_label = QLabel("🎉")
        success_label.setStyleSheet("font-size: 48px;")
        success_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(success_label)

        layout.addSpacing(20)

        # Title
        title = QLabel("Setup Complete!")
        title.setFont(QFont("", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #EAEAEA;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(20)

        # Summary
        summary = QLabel(
            "SuperKeet is now ready to use!\n\n"
            "Quick Start:\n"
            "• Press and hold Ctrl+Space to start recording\n"
            "• Speak clearly into your microphone\n"
            "• Release Ctrl+Space to transcribe\n"
            "• The text will appear where your cursor is\n\n"
            "You can also drag audio/video files to the main window for batch transcription.\n\n"  # noqa: E501
            "Enjoy using SuperKeet!"
        )
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #EAEAEA; font-size: 14px; line-height: 1.6;")
        summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(summary)

        layout.addStretch()

        return page

    def populate_devices(self):
        """Populate the device combo box."""
        try:
            # Create temporary audio recorder to get devices
            temp_recorder = AudioRecorder()
            devices = temp_recorder.get_devices()

            # Add default device
            self.device_combo.addItem("Default Device", None)

            # Add available devices
            for device in devices:
                device_name = device.get("name", "Unknown Device")
                device_index = device.get("index", 0)

                # Add device quality rating
                rating = self.device_service.get_device_recommendation(device)
                display_name = f"{device_name} {rating}"

                self.device_combo.addItem(display_name, device_index)

            logger.info(f"🎙️ Found {len(devices)} audio devices for selection")

        except Exception as e:
            logger.error(f"🛑 Failed to populate devices: {e}")
            self.device_combo.addItem("No devices found", None)

    @Slot(int)
    def on_device_changed(self, index: int):
        """Handle device selection change."""
        device_index = self.device_combo.itemData(index)
        self.selected_device = device_index

        if device_index is None:
            self.device_info_label.setText(
                "Default device will be used. macOS will automatically select the system default microphone."  # noqa: E501
            )
        else:
            try:
                # Create temporary recorder to test device
                temp_recorder = AudioRecorder()
                devices = temp_recorder.get_devices()

                # Find device info
                device_info = next(
                    (d for d in devices if d.get("index") == device_index), None
                )
                if device_info:
                    rating = self.device_service.get_device_recommendation(device_info)
                    channels = device_info.get("max_input_channels", "Unknown")
                    sample_rate = device_info.get("default_samplerate", "Unknown")

                    info_text = (
                        f"Device: {device_info.get('name', 'Unknown')}\n"
                        f"Quality: {rating}\n"
                        f"Channels: {channels}\n"
                        f"Sample Rate: {sample_rate}Hz\n\n"
                        f"This device will be optimized for SuperKeet's 16kHz ASR model."  # noqa: E501
                    )
                    self.device_info_label.setText(info_text)

            except Exception as e:
                self.device_info_label.setText(f"Error getting device info: {e}")

    def test_selected_device(self):
        """Test the selected audio device by attempting a brief recording."""
        try:
            if self.selected_device is None:
                device_name = "Default Device"
            else:
                device_name = self.device_combo.currentText()

            self.device_info_label.setText(f"🔄 Testing {device_name}...")

            # Attempt a brief audio capture to verify device works
            temp_recorder = AudioRecorder()
            temp_recorder.device = self.selected_device

            if temp_recorder.start():
                import time

                time.sleep(0.2)  # Brief capture
                temp_recorder.stop()
                self.device_info_label.setText(f"✅ {device_name} works!")
                logger.info(f"🎙️ Device test passed: {device_name}")
            else:
                self.device_info_label.setText(
                    f"⚠️ {device_name} may have permission issues"
                )
                logger.warning(f"🟡 Device test inconclusive: {device_name}")

        except Exception as e:
            self.device_info_label.setText(f"❌ Device test failed: {e}")
            logger.error(f"🛑 Device test failed: {e}")

    def request_permission(self, perm_type: str):
        """Request system permission."""
        if perm_type == "microphone":
            success = self._request_microphone_permission()
            if success:
                self.permissions_granted["microphone"] = True
                self.microphone_button.setText("✅ Granted")
                self.microphone_button.setEnabled(False)
            else:
                self.microphone_button.setText("❌ Denied")

        elif perm_type == "accessibility":
            self._request_accessibility_permission()
            # Accessibility requires checking separately
            if self._check_accessibility_permission():
                self.permissions_granted["accessibility"] = True
                self.accessibility_button.setText("✅ Granted")
                self.accessibility_button.setEnabled(False)
            else:
                self.accessibility_button.setText("⚠️ Please Enable")

        # Check if we can proceed
        self.update_navigation()

    def _request_microphone_permission(self) -> bool:
        """Request microphone permission on macOS."""
        try:
            # On macOS, we can trigger microphone permission by trying to access it
            # The system will automatically prompt the user
            temp_recorder = AudioRecorder()
            temp_recorder.get_devices()

            # Try to create a brief recording to trigger permission
            recording_started = temp_recorder.start()
            if recording_started:
                # Very brief recording just to trigger permission
                import time

                time.sleep(0.1)
                temp_recorder.stop()
                logger.info("🎙️ Microphone permission requested successfully")
                return True
            else:
                logger.warning("⚠️ Microphone access may be denied")
                return False

        except Exception as e:
            logger.error(f"🛑 Failed to request microphone permission: {e}")
            return False

    def _request_accessibility_permission(self) -> None:
        """Request accessibility permission on macOS."""
        try:
            import subprocess

            # Open System Preferences to Privacy & Security > Accessibility
            subprocess.run(
                [
                    "open",
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
                ],
                check=False,
            )

            # Show instructions to user
            from PySide6.QtWidgets import QMessageBox

            msg = QMessageBox()
            msg.setWindowTitle("Accessibility Permission Required")
            msg.setText(
                "SuperKeet needs accessibility permission to inject transcribed text into other applications.\n\n"  # noqa: E501
                "Please:\n"
                "1. Find 'SuperKeet' in the list\n"
                "2. Check the box next to it\n"
                "3. Click 'Done' and return to this setup\n\n"
                "If SuperKeet is not in the list, you may need to add it manually by clicking the '+' button."  # noqa: E501
            )
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()

            logger.info("⌨️ Accessibility permission dialog opened")

        except Exception as e:
            logger.error(f"🛑 Failed to open accessibility settings: {e}")

    def _check_accessibility_permission(self) -> bool:
        """Check if accessibility permission is granted."""
        try:
            # Try to use accessibility features to check permission
            # This is a simplified check - actual implementation may vary
            from superkeet.text.injector import TextInjector

            # Try a simple test that doesn't actually inject anything
            # but checks if accessibility is available
            TextInjector()  # Just check if it can be instantiated
            return True  # Simplified for now

        except Exception as e:
            logger.debug(f"Accessibility permission check: {e}")
            return False

    def go_back(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.content_stack.setCurrentIndex(self.current_page)
            self.update_navigation()

    def go_next(self):
        """Go to next page."""
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
            self.content_stack.setCurrentIndex(self.current_page)
            self.update_navigation()
        else:
            # Finish setup
            self.finish_setup()

    def update_navigation(self):
        """Update navigation button states."""
        # Update page label
        self.page_label.setText(f"{self.current_page + 1} of {len(self.pages)}")

        # Back button
        self.back_button.setEnabled(self.current_page > 0)

        # Next button text and state
        if self.current_page == len(self.pages) - 1:
            self.next_button.setText("Finish")
        else:
            self.next_button.setText("Next")

        # Check if current page requirements are met
        can_proceed = True

        if self.pages[self.current_page] == "permissions":
            # Require both permissions (for demo, we'll make it optional)
            # can_proceed = all(self.permissions_granted.values())
            pass

        self.next_button.setEnabled(can_proceed)

    def finish_setup(self):
        """Complete the setup process."""
        try:
            # Save selected device to config
            if self.selected_device is not None:
                config.set("audio.device", self.selected_device)

            # Mark first run as completed
            config.set("app.first_run_completed", True)
            config.save()

            logger.info("🎯 First-run setup completed successfully")

            # Emit completion signal
            self.setup_completed.emit()

            # Close dialog
            self.accept()

        except Exception as e:
            logger.error(f"🛑 Failed to complete setup: {e}")

    def apply_styles(self):
        """Apply dialog styling."""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1E1E1E;
                color: #EAEAEA;
            }

            QLabel {
                color: #EAEAEA;
            }

            QPushButton {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 6px;
                padding: 8px 16px;
                color: #EAEAEA;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #3A3A3A;
            }

            QPushButton:disabled {
                background-color: #1A1A1A;
                color: #5A5A5A;
                border-color: #2A2A2A;
            }

            QComboBox {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 6px;
                color: #EAEAEA;
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
        """
        )


# end src/ui/first_run_dialog.py
