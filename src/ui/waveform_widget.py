# start src/ui/waveform_widget.py
"""
Waveform visualization widget for real-time audio display.
Uses pyqtgraph for high-performance plotting.
"""

import logging

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

        # Audio data buffer - use numpy array for consistency
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_write_pos = 0

        # Update timer - reduced frequency for better performance
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plot)
        self.update_timer.setInterval(100)  # 10 FPS for better performance

        # Performance optimization settings
        self.display_decimation = 4  # Show every 4th sample for performance
        self.last_update_time = 0

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

        try:
            # Normalize audio data to [-1, 1] range
            if audio_chunk.size > 0:
                # Convert to mono if stereo
                if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
                    audio_chunk = audio_chunk.mean(axis=1)

                # Ensure it's 1D and float32
                audio_chunk = audio_chunk.flatten().astype(np.float32)

                # Normalize
                max_val = np.abs(audio_chunk).max()
                if max_val > 0:
                    normalized = (
                        audio_chunk / max_val * 0.8
                    )  # Scale to 80% to avoid clipping
                else:
                    normalized = audio_chunk

                # Add to circular buffer
                self._add_to_circular_buffer(normalized)
        except Exception as e:
            logger.error(f"Error updating waveform data: {e}")

    def _add_to_circular_buffer(self, data: np.ndarray):
        """Add data to the circular buffer efficiently."""
        data_len = len(data)
        if data_len == 0:
            return

        # Pre-allocate for better performance
        if data_len > self.buffer_size:
            # If data is larger than buffer, only keep the most recent part
            data = data[-self.buffer_size :]
            data_len = len(data)

        # Handle buffer wrap-around
        end_pos = self.buffer_write_pos + data_len

        if end_pos <= self.buffer_size:
            # Simple case: data fits without wrapping
            self.audio_buffer[self.buffer_write_pos : end_pos] = data
        else:
            # Wrap-around case: split the data
            first_part_size = self.buffer_size - self.buffer_write_pos
            self.audio_buffer[self.buffer_write_pos :] = data[:first_part_size]
            self.audio_buffer[: end_pos - self.buffer_size] = data[first_part_size:]

        self.buffer_write_pos = end_pos % self.buffer_size

    def _update_plot(self):
        """Update the waveform plot with performance optimizations."""
        if self.is_recording:
            try:
                import time

                current_time = time.time()

                # Throttle updates to prevent excessive CPU usage
                if (
                    current_time - self.last_update_time < 0.05
                ):  # Minimum 50ms between updates
                    return

                self.last_update_time = current_time

                # Decimate data for visualization performance
                decimated_data = self.audio_buffer[:: self.display_decimation].copy()

                # Create time axis for decimated data
                time_axis = np.arange(len(decimated_data))

                # Update plot with decimated data
                self.plot_curve.setData(time_axis, decimated_data)

            except Exception as e:
                logger.error(f"Error updating waveform plot: {e}")

    def start_recording(self):
        """Start the waveform display."""
        self.is_recording = True
        self.last_update_time = 0  # Reset throttle timer
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
        try:
            # Simple fade by reducing amplitude
            current_data = self.audio_buffer.copy()

            fade_steps = 8  # Reduced steps for better performance
            fade_duration = 400  # ms - shorter duration
            fade_timer = QTimer()

            def fade_step():
                nonlocal fade_steps
                fade_steps -= 1

                if fade_steps <= 0:
                    fade_timer.stop()
                    self.update_timer.stop()
                    self.clear()
                else:
                    # Reduce amplitude with decimated data for performance
                    fade_factor = fade_steps / 8.0
                    decimated_data = current_data[:: self.display_decimation]
                    faded_data = decimated_data * fade_factor
                    self.plot_curve.setData(np.arange(len(faded_data)), faded_data)

            fade_timer.timeout.connect(fade_step)
            fade_timer.start(fade_duration // 8)
        except Exception as e:
            logger.error(f"Error during waveform fade out: {e}")
            # Fallback: just clear immediately
            self.update_timer.stop()
            self.clear()

    def clear(self):
        """Clear the waveform display."""
        self.audio_buffer.fill(0.0)
        self.buffer_write_pos = 0
        self.plot_curve.setData([], [])
        logger.info("Waveform cleared")


# end src/ui/waveform_widget.py
