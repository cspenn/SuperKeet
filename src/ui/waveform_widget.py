# start src/ui/waveform_widget.py
"""
Waveform visualization widget for real-time audio display.
Uses pyqtgraph for high-performance plotting.
"""

import logging
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class WaveformWidget(QWidget):
    """Widget for displaying real-time audio waveform."""

    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 3.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)

        # Audio data buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.audio_buffer.extend([0] * self.buffer_size)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plot)
        self.update_timer.setInterval(50)  # 20 FPS

        # Recording state
        self.is_recording = False

        # Set up the UI
        self._setup_ui()

        logger.info("🟢 WaveformWidget initialized")

    def _setup_ui(self):
        """Set up the waveform plot."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1E1E1E")
        self.plot_widget.showGrid(False, False)
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.setMouseEnabled(False, False)
        self.plot_widget.setMenuEnabled(False)

        # Create plot curve with gradient
        self.plot_curve = self.plot_widget.plot(pen=None)

        # Create simple pen (gradient not supported in all pyqtgraph versions)
        pen = pg.mkPen(color="#007AFF", width=2)
        self.plot_curve.setPen(pen)

        # Set plot range
        self.plot_widget.setYRange(-1, 1, padding=0.1)
        self.plot_widget.setXRange(0, self.buffer_size, padding=0)

        layout.addWidget(self.plot_widget)

    @Slot(np.ndarray)
    def update_data(self, audio_chunk: np.ndarray):
        """Update the audio buffer with new data."""
        if not self.is_recording:
            return

        # Normalize audio data to [-1, 1] range
        if audio_chunk.size > 0:
            # Convert to mono if stereo
            if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
                audio_chunk = audio_chunk.mean(axis=1)

            # Normalize
            max_val = np.abs(audio_chunk).max()
            if max_val > 0:
                normalized = (
                    audio_chunk / max_val * 0.8
                )  # Scale to 80% to avoid clipping
            else:
                normalized = audio_chunk

            # Add to buffer
            self.audio_buffer.extend(normalized)

    def _update_plot(self):
        """Update the waveform plot."""
        if self.is_recording:
            # Convert buffer to numpy array for plotting
            data = np.array(self.audio_buffer)

            # Create time axis
            time_axis = np.arange(len(data))

            # Update plot
            self.plot_curve.setData(time_axis, data)

    def start_recording(self):
        """Start the waveform display."""
        self.is_recording = True
        self.update_timer.start()
        logger.info("🟢 Waveform recording started")

    def stop_recording(self):
        """Stop the waveform display and fade out."""
        self.is_recording = False

        # Fade out animation
        self._fade_out()

        logger.info("🟢 Waveform recording stopped")

    def _fade_out(self):
        """Fade out the waveform smoothly."""
        # Simple fade by reducing amplitude
        current_data = np.array(self.audio_buffer)

        fade_steps = 10
        fade_duration = 500  # ms
        fade_timer = QTimer()

        def fade_step():
            nonlocal fade_steps
            fade_steps -= 1

            if fade_steps <= 0:
                fade_timer.stop()
                self.update_timer.stop()
                self.clear()
            else:
                # Reduce amplitude
                fade_factor = fade_steps / 10.0
                faded_data = current_data * fade_factor
                self.plot_curve.setData(np.arange(len(faded_data)), faded_data)

        fade_timer.timeout.connect(fade_step)
        fade_timer.start(fade_duration // 10)

    def clear(self):
        """Clear the waveform display."""
        self.audio_buffer.clear()
        self.audio_buffer.extend([0] * self.buffer_size)
        self.plot_curve.setData([], [])
        logger.info("🟢 Waveform cleared")


# end src/ui/waveform_widget.py
