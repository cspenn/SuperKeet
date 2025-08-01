# start src/audio/recorder.py
"""Audio recording functionality for SuperKeet."""

import queue
from typing import Optional

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal

from src.config.config_loader import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioRecorder(QObject):
    """Manages audio recording from the default microphone."""

    # Signal emitted when audio chunk is ready for visualization
    audio_chunk_ready = Signal(np.ndarray)

    def __init__(self) -> None:
        """Initialize the audio recorder."""
        super().__init__()
        self.sample_rate = config.get("audio.sample_rate", 16000)
        self.channels = config.get("audio.channels", 1)
        self.chunk_size = config.get("audio.chunk_size", 1024)
        self.device = config.get("audio.device", None)

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.recording = False
        self.stream: Optional[sd.InputStream] = None
        self.audio_data: list[np.ndarray] = []

        logger.info(
            f"AudioRecorder initialized: {self.sample_rate}Hz, {self.channels}ch"
        )

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback function for audio stream.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Timing information.
            status: Stream status flags.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self.recording:
            # Copy data to avoid reference issues
            audio_copy = indata.copy()
            self.audio_queue.put(audio_copy)
            # Emit signal for waveform visualization
            self.audio_chunk_ready.emit(audio_copy)

    def start(self) -> None:
        """Start recording audio."""
        if self.recording:
            logger.warning("Already recording")
            return

        try:
            # Clear any previous data
            self.audio_data.clear()
            while not self.audio_queue.empty():
                self.audio_queue.get()

            # Create and start stream
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32,
            )

            self.stream.start()
            self.recording = True
            logger.info("Started recording")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            raise

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio.

        Returns:
            Numpy array containing the recorded audio.
        """
        if not self.recording:
            logger.warning("Not recording")
            return np.array([])

        try:
            self.recording = False

            # Stop and close stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Collect all audio data from queue
            while not self.audio_queue.empty():
                self.audio_data.append(self.audio_queue.get())

            # Concatenate all chunks
            if self.audio_data:
                audio_array = np.concatenate(self.audio_data, axis=0)
                # Flatten if mono
                if self.channels == 1:
                    audio_array = audio_array.flatten()
                logger.info(
                    f"Stopped recording: {len(audio_array) / self.sample_rate:.2f}s"
                )
                return audio_array
            else:
                logger.warning("No audio data captured")
                return np.array([])

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            raise

    def get_devices(self) -> list[dict]:
        """Get list of available audio input devices.

        Returns:
            List of device information dictionaries.
        """
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                    }
                )
        return devices


# end src/audio/recorder.py
