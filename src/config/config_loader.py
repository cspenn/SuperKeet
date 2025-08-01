# start src/config/config_loader.py
"""Configuration loader for SuperKeet."""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Loads and manages application configuration from YAML files."""

    def __init__(self, config_path: str = "config.yml") -> None:
        """Initialize the configuration loader.

        Args:
            config_path: Path to the main configuration file.
        """
        self.config_path = Path(config_path)
        self.credentials_path = Path("credentials.yml")
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML files."""
        # Load main config
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load credentials if exists
        if self.credentials_path.exists():
            with open(self.credentials_path, "r") as f:
                credentials = yaml.safe_load(f) or {}
                # Merge credentials into config
                self.config.update(credentials)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., "app.name" or "audio.sample_rate").
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.

        Returns:
            Complete configuration dictionary.
        """
        return self.config.copy()


# Global config instance
config = ConfigLoader()

# end src/config/config_loader.py
