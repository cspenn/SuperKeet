# start src/utils/exceptions.py
"""Custom exception definitions for SuperKeet application."""


class SuperKeetError(Exception):
    """Base exception class for SuperKeet application errors."""

    pass


class AudioDeviceError(SuperKeetError):
    """Raised when audio device configuration or operation fails."""

    pass


class ASRError(SuperKeetError):
    """Raised when ASR transcription fails."""

    pass


class FileOperationError(SuperKeetError):
    """Raised when file operations (save, load, cleanup) fail."""

    pass


# end src/utils/exceptions.py
