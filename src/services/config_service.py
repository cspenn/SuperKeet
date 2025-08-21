"""Centralized configuration management service."""

from typing import Any, Dict

from src.config.config_loader import ConfigLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfigService:
    """Centralized service for configuration management with caching and validation."""

    def __init__(self, config_path: str = "config.yml"):
        """Initialize configuration service.

        Args:
            config_path: Path to configuration file
        """
        self._config_loader = ConfigLoader(config_path)
        self._cache = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with caching.

        Args:
            key: Configuration key in dot notation
            default: Default value if not found

        Returns:
            Configuration value
        """
        if key in self._cache:
            return self._cache[key]

        value = self._config_loader.get(key, default)
        self._cache[key] = value
        return value

    def set(self, key: str, value: Any, save_immediately: bool = False) -> None:
        """Set configuration value.

        Args:
            key: Configuration key in dot notation
            value: Value to set
            save_immediately: Whether to save to disk immediately
        """
        self._config_loader.set(key, value)
        self._cache[key] = value

        if save_immediately:
            self.save()

    def save(self) -> bool:
        """Save configuration to disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._config_loader.save()
            logger.info("💾 Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"🛑 Failed to save configuration: {e}")
            return False

    def reload(self) -> bool:
        """Reload configuration from disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._config_loader.load()
            self._cache.clear()  # Clear cache to force reload
            logger.info("🔄 Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"🛑 Failed to reload configuration: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Complete configuration dictionary
        """
        return self._config_loader.get_all()

    def validate_audio_settings(self) -> bool:
        """Validate audio configuration settings.

        Returns:
            True if valid, False otherwise
        """
        try:
            sample_rate = self.get("audio.sample_rate", 16000)
            channels = self.get("audio.channels", 1)
            chunk_size = self.get("audio.chunk_size", 1024)

            # Validate sample rate
            valid_rates = [8000, 16000, 22050, 44100, 48000]
            if sample_rate not in valid_rates:
                logger.error(f"🛑 Invalid sample rate: {sample_rate}")
                return False

            # Validate channels
            if channels not in [1, 2]:
                logger.error(f"🛑 Invalid channel count: {channels}")
                return False

            # Validate chunk size
            if chunk_size < 256 or chunk_size > 8192:
                logger.warning(f"⚠️ Unusual chunk size: {chunk_size}")

            return True

        except Exception as e:
            logger.error(f"🛑 Audio settings validation failed: {e}")
            return False

    def validate_hotkey_settings(self) -> bool:
        """Validate hotkey configuration settings.

        Returns:
            True if valid, False otherwise
        """
        try:
            combination = self.get("hotkey.combination", "cmd+shift+space")

            if not combination or not isinstance(combination, (str, list)):
                logger.error(f"🛑 Invalid hotkey combination: {combination}")
                return False

            return True

        except Exception as e:
            logger.error(f"🛑 Hotkey settings validation failed: {e}")
            return False

    def get_debug_settings(self) -> Dict[str, Any]:
        """Get debug-related settings.

        Returns:
            Dictionary of debug settings
        """
        return {
            "save_audio_files": self.get("debug.save_audio_files", True),
            "audio_debug_dir": self.get("debug.audio_debug_dir", "debug_audio"),
            "retention_days": self.get("debug.retention_days", 7),
            "max_files": self.get("debug.max_files", 50),
            "max_size_mb": self.get("debug.max_size_mb", 100.0),
        }

    def get_audio_settings(self) -> Dict[str, Any]:
        """Get audio-related settings.

        Returns:
            Dictionary of audio settings
        """
        return {
            "sample_rate": self.get("audio.sample_rate", 16000),
            "channels": self.get("audio.channels", 1),
            "chunk_size": self.get("audio.chunk_size", 1024),
            "device": self.get("audio.device", None),
            "gain": self.get("audio.gain", 1.0),
        }

    def get_text_settings(self) -> Dict[str, Any]:
        """Get text injection settings.

        Returns:
            Dictionary of text settings
        """
        return {
            "method": self.get("text.method", "clipboard"),
            "auto_paste": self.get("text.auto_paste", True),
            "delay_ms": self.get("text.delay_ms", 100),
        }

    def create_backup(self) -> bool:
        """Create backup of current configuration.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._config_loader._create_backup()
            return True
        except Exception as e:
            logger.error(f"🛑 Failed to create config backup: {e}")
            return False

    def restore_from_backup(self) -> bool:
        """Restore configuration from backup.

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self._config_loader.restore_from_backup()
            if success:
                self._cache.clear()  # Clear cache
            return success
        except Exception as e:
            logger.error(f"🛑 Failed to restore config from backup: {e}")
            return False


# Global config service instance
config_service = ConfigService()
