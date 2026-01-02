# start src/config/validators.py
"""Configuration validation schemas using Pydantic."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class HotkeyConfig(BaseModel):
    """Hotkey configuration validation."""

    combination: str = Field(
        default="cmd+shift+space", description="Hotkey combination"
    )

    @field_validator("combination")
    @classmethod
    def validate_combination(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("Hotkey combination must be a non-empty string")
        return v


class AudioConfig(BaseModel):
    """Audio configuration validation with memory management."""

    device: Optional[int] = Field(default=None, description="Audio device index")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_size: int = Field(default=8192, description="Audio chunk size")
    gain: float = Field(default=1.0, description="Audio gain multiplier")

    # Memory management settings
    max_recording_duration: int = Field(
        default=300, description="Maximum recording duration in seconds"
    )
    buffer_size_limit: int = Field(
        default=100, description="Maximum audio buffer size in MB"
    )
    enable_buffer_monitoring: bool = Field(
        default=True, description="Enable audio buffer monitoring"
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        valid_rates = (8000, 16000, 22050, 44100, 48000)
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of: {valid_rates}")
        return v

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: int) -> int:
        if v not in (1, 2):
            raise ValueError("Channels must be 1 (mono) or 2 (stereo)")
        return v

    @field_validator("gain")
    @classmethod
    def validate_gain(cls, v: float) -> float:
        if v <= 0 or v > 10:
            raise ValueError("Gain must be between 0 and 10")
        return v

    @field_validator("max_recording_duration")
    @classmethod
    def validate_max_duration(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Max recording duration must be positive")
        if v > 3600:  # 1 hour max
            raise ValueError("Max recording duration cannot exceed 1 hour")
        return v

    @field_validator("buffer_size_limit")
    @classmethod
    def validate_buffer_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Buffer size limit must be positive")
        if v > 1000:  # 1GB max
            raise ValueError("Buffer size limit cannot exceed 1GB")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration validation."""

    level: str = Field(default="INFO", description="Logging level")
    file: str = Field(default="logs/superkeet.log", description="Log file path")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class ASRConfig(BaseModel):
    """ASR (Automatic Speech Recognition) configuration validation."""

    asr_model: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="ASR model identifier (Parakeet v3)",
    )
    parakeet_native_rate: int = Field(
        default=16000, description="Parakeet native sample rate"
    )
    auto_unload_timeout: int = Field(
        default=300,
        description="Auto-unload model after N seconds of inactivity (0 to disable)",
    )
    max_audio_duration: float = Field(
        default=1440.0, description="Maximum audio duration in seconds (24 minutes)"
    )
    enable_chunked_processing: bool = Field(
        default=True, description="Enable chunked processing for large files"
    )
    chunk_duration: float = Field(
        default=60.0, description="Duration of each processing chunk in seconds"
    )

    @field_validator("parakeet_native_rate")
    @classmethod
    def validate_native_rate(cls, v: int) -> int:
        if v != 16000:
            raise ValueError("Parakeet native rate must be 16000")
        return v

    @field_validator("auto_unload_timeout")
    @classmethod
    def validate_auto_unload(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Auto-unload timeout cannot be negative")
        return v

    @field_validator("max_audio_duration")
    @classmethod
    def validate_max_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Max audio duration must be positive")
        if v > 1440:  # 24 minutes max for Parakeet v3
            raise ValueError("Max audio duration exceeds model limit (24 minutes)")
        return v

    @model_validator(mode="before")
    @classmethod
    def handle_model_id_compatibility(cls, data: Any) -> Any:
        """Handle backward compat for model_id/model_name/model_identifier."""
        if isinstance(data, dict):
            # Handle backward compatibility with multiple old field names
            if "model_id" in data and "asr_model" not in data:
                data["asr_model"] = data["model_id"]
            elif "model_name" in data and "asr_model" not in data:
                data["asr_model"] = data["model_name"]
            elif "model_identifier" in data and "asr_model" not in data:
                data["asr_model"] = data["model_identifier"]
            # Remove old fields to avoid conflicts
            data.pop("model_id", None)
            data.pop("model_name", None)
            data.pop("model_identifier", None)
        return data


class TextConfig(BaseModel):
    """Text injection configuration validation."""

    method: str = Field(default="clipboard", description="Text injection method")
    auto_paste: bool = Field(default=True, description="Auto-paste transcribed text")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid_methods = ("clipboard", "accessibility")
        if v not in valid_methods:
            raise ValueError(f"Injection method must be one of: {valid_methods}")
        return v


class BatchProcessingConfig(BaseModel):
    """Batch processing configuration validation."""

    enabled: bool = Field(default=True, description="Enable batch processing")
    max_file_size_gb: int = Field(default=1, description="Maximum file size in GB")
    supported_formats: List[str] = Field(
        default=["mp3", "m4a", "mp4", "mov", "wav", "flac"],
        description="Supported file formats",
    )
    temp_directory: str = Field(default="temp_batch", description="Temporary directory")
    chunk_duration_seconds: int = Field(
        default=300, description="Chunk duration in seconds"
    )


class SuperKeetConfig(BaseModel):
    """Main SuperKeet configuration validation."""

    app: Dict[str, Any] = Field(default_factory=dict)
    hotkey: HotkeyConfig = Field(default_factory=HotkeyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    batch_processing: BatchProcessingConfig = Field(
        default_factory=BatchProcessingConfig
    )

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
        "use_enum_values": True,
    }


def validate_config(config_dict: Dict[str, Any]) -> SuperKeetConfig:
    """Validate configuration dictionary using Pydantic schemas.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        Validated SuperKeetConfig instance

    Raises:
        ValueError: If configuration validation fails
    """
    try:
        return SuperKeetConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


# end src/config/validators.py
