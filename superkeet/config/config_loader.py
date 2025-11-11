# start src/config/config_loader.py
"""Configuration loader for SuperKeet."""

from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Loads and manages application configuration from YAML files."""

    def __init__(self, config_path: str = "config.yml") -> None:
        """Initialize the configuration loader.

        Args:
            config_path: Path to the main configuration file.
        """
        self.config_path = Path(config_path)
        self.credentials_path = Path("credentials.yml")
        self.backup_path = Path(f"{config_path}.backup")
        self.config: Dict[str, Any] = {}
        self.validated_config = None
        self.load()

    def load(self) -> None:
        """Load configuration from YAML files."""
        # Load main config
        if self.config_path.exists():
            try:
                import ruamel.yaml

                yaml_loader = ruamel.yaml.YAML()
                yaml_loader.preserve_quotes = True
                yaml_loader.width = 4096

                with open(self.config_path, "r") as f:
                    self.config = yaml_loader.load(f) or {}
            except ImportError:
                # Fallback to standard yaml
                import yaml

                with open(self.config_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load credentials if exists
        if self.credentials_path.exists():
            try:
                import ruamel.yaml

                yaml_loader = ruamel.yaml.YAML()
                with open(self.credentials_path, "r") as f:
                    credentials = yaml_loader.load(f) or {}
            except ImportError:
                import yaml

                with open(self.credentials_path, "r") as f:
                    credentials = yaml.safe_load(f) or {}
            # Merge credentials into config
            self.config.update(credentials)

        # Validate configuration
        self._validate_config()

    def save(self) -> None:
        """Save configuration to YAML file with atomic write and backup."""
        import shutil
        import tempfile

        try:
            # Create backup before saving
            self._create_backup()

            # Validate configuration before saving
            self._validate_config()

            # Use atomic write with temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".tmp"
            ) as temp_file:
                try:
                    import ruamel.yaml

                    yaml_loader = ruamel.yaml.YAML()
                    yaml_loader.preserve_quotes = True
                    yaml_loader.width = 4096
                    yaml_loader.dump(self.config, temp_file)
                except ImportError:
                    # Fallback to standard yaml
                    import yaml

                    yaml.safe_dump(
                        self.config, temp_file, default_flow_style=False, indent=2
                    )

                temp_file.flush()

                # Use filelock for atomic operations
                try:
                    import filelock

                    lock = filelock.FileLock(f"{self.config_path}.lock")
                    with lock.acquire(timeout=10):
                        shutil.move(temp_file.name, self.config_path)
                except ImportError:
                    # Fallback without filelock
                    shutil.move(temp_file.name, self.config_path)

        except Exception as e:
            # Clean up temp file if it exists
            if "temp_file" in locals() and Path(temp_file.name).exists():
                Path(temp_file.name).unlink()
            raise RuntimeError(f"Failed to save configuration: {e}")

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

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., "app.name" or "audio.sample_rate").
            value: Value to set.
        """
        keys = key.split(".")
        config_ref = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]

        # Set the final key
        config_ref[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.

        Returns:
            Complete configuration dictionary.
        """
        return self.config.copy()

    def _validate_config(self) -> None:
        """Validate the loaded configuration using Pydantic schemas."""
        try:
            from .validators import validate_config

            self.validated_config = validate_config(self.config)
        except ImportError:
            # Skip validation if pydantic is not available
            try:
                from superkeet.utils.logger import setup_logger

                logger = setup_logger(__name__)
                logger.warning(
                    "Pydantic not available, skipping configuration validation"
                )
            except ImportError:
                pass
        except Exception as e:
            try:
                from superkeet.utils.logger import setup_logger

                logger = setup_logger(__name__)
                logger.error(f"Configuration validation failed: {e}")
            except ImportError:
                pass
            # Continue with unvalidated config but log the error

    def _create_backup(self) -> None:
        """Create a backup of the current configuration file."""
        if self.config_path.exists():
            try:
                import shutil

                shutil.copy2(self.config_path, self.backup_path)
                try:
                    from superkeet.utils.logger import setup_logger

                    logger = setup_logger(__name__)
                    logger.debug(f"Configuration backup created: {self.backup_path}")
                except ImportError:
                    pass
            except Exception as e:
                try:
                    from superkeet.utils.logger import setup_logger

                    logger = setup_logger(__name__)
                    logger.warning(f"Failed to create configuration backup: {e}")
                except ImportError:
                    pass

    def restore_from_backup(self) -> bool:
        """Restore configuration from backup file.

        Returns:
            True if restore was successful, False otherwise.
        """
        if not self.backup_path.exists():
            try:
                from superkeet.utils.logger import setup_logger

                logger = setup_logger(__name__)
                logger.error("No backup file found for restore")
            except ImportError:
                pass
            return False

        try:
            import shutil

            shutil.copy2(self.backup_path, self.config_path)
            self.load()  # Reload the restored configuration
            try:
                from superkeet.utils.logger import setup_logger

                logger = setup_logger(__name__)
                logger.info("Configuration restored from backup successfully")
            except ImportError:
                pass
            return True
        except Exception as e:
            try:
                from superkeet.utils.logger import setup_logger

                logger = setup_logger(__name__)
                logger.error(f"Failed to restore configuration from backup: {e}")
            except ImportError:
                pass
            return False


# Global config instance
config = ConfigLoader()

# end src/config/config_loader.py
