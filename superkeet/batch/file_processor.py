"""File processor for drag-and-drop transcription handling."""

from pathlib import Path
from typing import Dict, List, Tuple

try:
    import magic
except ImportError:
    magic = None

from superkeet.config.config_loader import config
from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class FileProcessor:
    """Handles file validation and processing for batch transcription."""

    def __init__(self) -> None:
        """Initialize file processor with configuration."""
        self.max_file_size_gb: int = config.get("batch_processing.max_file_size_gb", 1)
        self.supported_formats: List[str] = config.get(
            "batch_processing.supported_formats",
            ["mp3", "m4a", "mp4", "mov", "wav", "flac"],
        )
        self.temp_directory: str = config.get(
            "batch_processing.temp_directory", "temp_batch"
        )

        # Create temp directory if it doesn't exist
        Path(self.temp_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"🟢 FileProcessor initialized: max_size={self.max_file_size_gb}GB")
        logger.info(f"🟢 Supported formats: {', '.join(self.supported_formats)}")

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate a file for batch processing.

        Args:
            file_path: Path to the file to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"

            # Check file size
            file_size_bytes = file_path.stat().st_size
            file_size_gb = file_size_bytes / (1024**3)

            if file_size_gb > self.max_file_size_gb:
                return (
                    False,
                    f"File too large: {file_size_gb:.2f}GB > {self.max_file_size_gb}GB limit",  # noqa: E501
                )

            # Check file format by extension
            file_extension = file_path.suffix.lower().lstrip(".")
            if file_extension not in self.supported_formats:
                return False, f"Unsupported format: .{file_extension}"

            # Additional MIME type validation if python-magic is available
            if magic and hasattr(magic, "Magic"):
                try:
                    mime_detector = magic.Magic(mime=True)
                    mime_type = mime_detector.from_file(str(file_path))

                    expected_mimes = {
                        "mp3": ["audio/mpeg", "audio/mp3"],
                        "m4a": ["audio/mp4", "audio/x-m4a"],
                        "mp4": ["video/mp4", "audio/mp4"],
                        "mov": ["video/quicktime"],
                        "wav": ["audio/wav", "audio/x-wav"],
                        "flac": ["audio/flac", "audio/x-flac"],
                    }

                    expected = expected_mimes.get(file_extension, [])
                    if expected and mime_type not in expected:
                        logger.warning(
                            f"🟡 MIME type mismatch: {mime_type} for .{file_extension}"
                        )
                        # Don't fail, just warn - extensions are often more reliable

                except Exception as e:
                    logger.warning(f"🟡 MIME detection failed: {e}")

            logger.info(f"✅ File validated: {file_path.name} ({file_size_gb:.2f}GB)")
            return True, ""

        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(f"🛑 {error_msg}")
            return False, error_msg

    def validate_batch(
        self, file_paths: List[Path]
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """Validate a batch of files for processing.

        Args:
            file_paths: List of file paths to validate

        Returns:
            Tuple of (valid_files, invalid_files_with_errors)
        """
        valid_files: List[Path] = []
        invalid_files: List[Tuple[Path, str]] = []

        total_size_gb = 0.0

        for file_path in file_paths:
            is_valid, error_msg = self.validate_file(file_path)

            if is_valid:
                valid_files.append(file_path)
                file_size_gb = file_path.stat().st_size / (1024**3)
                total_size_gb += file_size_gb
            else:
                invalid_files.append((file_path, error_msg))

        # Check total batch size
        if (
            total_size_gb > self.max_file_size_gb * 2
        ):  # Allow up to 2x limit for batches
            logger.warning(f"🟡 Large batch: {total_size_gb:.2f}GB total")

        logger.info(
            f"📊 Batch validation: {len(valid_files)} valid, {len(invalid_files)} invalid"  # noqa: E501
        )
        return valid_files, invalid_files

    def get_output_path(self, input_path: Path, format_type: str = "txt") -> Path:
        """Generate output path for transcript file.

        Args:
            input_path: Original input file path
            format_type: Output format (txt, json, etc.)

        Returns:
            Path for the transcript file
        """
        # Place transcript in same directory as original file
        output_dir = input_path.parent
        output_name = input_path.stem + f"_transcript.{format_type}"
        return output_dir / output_name

    def cleanup_temp_files(self) -> None:
        """Clean up temporary processing files."""
        try:
            temp_path = Path(self.temp_directory)
            if temp_path.exists():
                for temp_file in temp_path.glob("*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                            logger.debug(f"🧹 Removed temp file: {temp_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to remove {temp_file.name}: {e}")

                logger.info("🧹 Temporary files cleaned up")
        except Exception as e:
            logger.error(f"🛑 Cleanup failed: {e}")

    def get_file_info(self, file_path: Path) -> Dict[str, any]:
        """Get detailed information about a file.

        Args:
            file_path: Path to analyze

        Returns:
            Dictionary with file information
        """
        try:
            stat = file_path.stat()

            info = {
                "name": file_path.name,
                "path": str(file_path),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024**2),
                "size_gb": stat.st_size / (1024**3),
                "extension": file_path.suffix.lower(),
                "modified": stat.st_mtime,
                "is_audio": file_path.suffix.lower().lstrip(".")
                in ["mp3", "wav", "flac", "m4a"],
                "is_video": file_path.suffix.lower().lstrip(".") in ["mp4", "mov"],
            }

            # Add estimated processing time (rough estimate: 1GB = 10 minutes processing)  # noqa: E501
            info["estimated_processing_minutes"] = max(1, int(info["size_gb"] * 10))

            return info

        except Exception as e:
            logger.error(f"🛑 Failed to get file info: {e}")
            return {"error": str(e)}
