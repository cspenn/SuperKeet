# start src/ui/audio_animation_widget.py
"""Audio animation widget for SuperKeet - smooth visual feedback during recording."""

import logging
import math

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class AudioAnimationWidget(QWidget):
    """Smooth animated visualization for audio recording."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMaximumHeight(200)

        # Animation state
        self.is_recording = False
        self.animation_phase = 0.0
        self.pulse_intensity = 0.0

        # Colors
        self.bg_color = QColor("#1E1E1E")
        self.idle_color = QColor("#3A3A3A")
        self.recording_color = QColor("#007AFF")
        self.pulse_color = QColor("#00C7FF")

        # Audio level tracking
        self.current_audio_level = 0.0
        self.target_audio_level = 0.0
        self.audio_level_smoothing = 0.3

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_timer.setInterval(16)  # ~60 FPS

        # Start animation
        self.animation_timer.start()

    def _update_animation(self):
        """Update animation state."""
        # Update phase
        self.animation_phase += 0.02
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi

        # Smooth audio level
        self.current_audio_level += (
            self.target_audio_level - self.current_audio_level
        ) * self.audio_level_smoothing

        # Update pulse intensity
        if self.is_recording:
            self.pulse_intensity = min(1.0, self.pulse_intensity + 0.05)
        else:
            self.pulse_intensity = max(0.0, self.pulse_intensity - 0.1)
            self.target_audio_level = 0.0

        self.update()

    @Slot()
    def start_recording(self):
        """Start recording animation."""
        self.is_recording = True

    @Slot()
    def stop_recording(self):
        """Stop recording animation."""
        self.is_recording = False

    @Slot(np.ndarray)
    def update_audio_level(self, audio_chunk: np.ndarray):
        """Update visualization based on audio level."""
        if not self.is_recording:
            return

        # Calculate RMS level
        rms = np.sqrt(np.mean(audio_chunk**2))
        # Convert to 0-1 range with logarithmic scaling
        db = 20 * np.log10(max(rms, 1e-10))
        normalized = np.clip((db + 60) / 60, 0, 1)  # -60dB to 0dB range

        self.target_audio_level = normalized

    def clear(self):
        """Clear the visualization."""
        self.is_recording = False
        self.target_audio_level = 0.0
        self.current_audio_level = 0.0

    def paintEvent(self, event):  # noqa: N802
        """Paint the animation."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Clear background
        painter.fillRect(self.rect(), self.bg_color)

        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2

        if self.pulse_intensity > 0:
            # Draw animated circles
            num_circles = 5
            max_radius = min(width, height) * 0.4

            for i in range(num_circles):
                # Calculate circle properties
                phase_offset = i * (2 * math.pi / num_circles)
                current_phase = self.animation_phase + phase_offset

                # Breathing effect
                breath = (math.sin(current_phase) + 1) / 2

                # Audio reactive scaling
                audio_scale = 1 + self.current_audio_level * 0.5

                # Calculate radius
                base_radius = max_radius * (0.3 + i * 0.15)
                radius = (
                    base_radius
                    * (0.8 + breath * 0.2)
                    * audio_scale
                    * self.pulse_intensity
                )

                # Calculate opacity
                opacity = int(255 * self.pulse_intensity * (0.2 + breath * 0.3))

                # Draw circle
                color = QColor(self.recording_color)
                color.setAlpha(opacity)
                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(
                    int(center_x - radius),
                    int(center_y - radius),
                    int(radius * 2),
                    int(radius * 2),
                )

            # Draw center dot
            center_radius = 5 + self.current_audio_level * 10
            center_opacity = int(255 * self.pulse_intensity)
            center_color = QColor(self.pulse_color)
            center_color.setAlpha(center_opacity)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(center_color))
            painter.drawEllipse(
                int(center_x - center_radius),
                int(center_y - center_radius),
                int(center_radius * 2),
                int(center_radius * 2),
            )
        else:
            # Draw idle state - simple dots
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(self.idle_color))

            dot_radius = 3
            spacing = 20
            num_dots = 3

            start_x = center_x - (num_dots - 1) * spacing / 2

            for i in range(num_dots):
                x = start_x + i * spacing
                painter.drawEllipse(
                    int(x - dot_radius),
                    int(center_y - dot_radius),
                    int(dot_radius * 2),
                    int(dot_radius * 2),
                )

    def cleanup_timers(self) -> None:
        """Clean up animation timer to prevent memory leaks."""
        if hasattr(self, "animation_timer") and self.animation_timer is not None:
            try:
                self.animation_timer.stop()
                self.animation_timer.deleteLater()
                self.animation_timer = None
                logger.debug("✅ Audio animation timer cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up animation timer: {e}")

    def closeEvent(self, event):  # noqa: N802
        """Handle widget close with timer cleanup."""
        self.cleanup_timers()
        super().closeEvent(event)


# end src/ui/audio_animation_widget.py
