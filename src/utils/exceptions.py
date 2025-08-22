"""Custom exception definitions for SuperKeet application."""


class SuperKeetError(Exception):
    """Base exception class for SuperKeet application errors."""

    pass


class ConfigurationError(SuperKeetError):
    """Raised when configuration is invalid or cannot be loaded/saved."""

    pass


class AudioDeviceError(SuperKeetError):
    """Raised when audio device configuration or operation fails."""

    pass


class ASRError(SuperKeetError):
    """Raised when ASR transcription fails."""

    pass


class TranscriptionError(SuperKeetError):
    """Raised when transcription processing fails."""

    pass


class HotkeyError(SuperKeetError):
    """Raised when hotkey registration or handling fails."""

    pass


class TextInjectionError(SuperKeetError):
    """Raised when text injection (clipboard/accessibility) fails."""

    pass


class FileOperationError(SuperKeetError):
    """Raised when file operations (save, load, cleanup) fail."""

    pass


class ValidationError(SuperKeetError):
    """Raised when data validation fails."""

    pass
