# start src/audio/__init__.py
"""Audio capture and processing modules.

This package provides audio recording functionality with modular components:
- recorder: Main AudioRecorder class
- permissions: Microphone permission validation
- device_manager: Device enumeration and validation
- stream_manager: Stream creation and sample rate optimization
- memory_manager: Buffer management and memory monitoring
- diagnostics: Logging and troubleshooting
- error_recovery: Error handling and recovery strategies
"""

from superkeet.audio.recorder import AudioRecorder

__all__ = ["AudioRecorder"]

# end src/audio/__init__.py
