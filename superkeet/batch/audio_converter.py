# start src/batch/audio_converter.py
"""Audio converter for batch processing using FFmpeg."""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import ffmpeg
import numpy as np
import soundfile as sf

from superkeet.config.config_loader import config
from superkeet.utils.exceptions import FileOperationError
from superkeet.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioConverter:
    """Handles audio/video file conversion for batch transcription."""

    def __init__(self) -> None:
        """Initialize audio converter with configuration."""
        self.temp_directory: str = config.get(
            "batch_processing.temp_directory", "temp_batch"
        )
        self.chunk_duration: int = config.get(
            "batch_processing.chunk_duration_seconds", 300
        )
        self.target_sample_rate: int = 16000  # ASR optimal sample rate
        self.target_channels: int = 1  # Mono for ASR

        # Ensure temp directory exists
        Path(self.temp_directory).mkdir(parents=True, exist_ok=True)

        # Check FFmpeg availability
        self._check_ffmpeg()

        logger.info(
            f"🟢 AudioConverter initialized: target={self.target_sample_rate}Hz, mono"
        )

    def _check_ffmpeg(self) -> None:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.split("\n")[0]
                logger.info(f"✅ {version_line}")
            else:
                raise FileOperationError("FFmpeg not responding properly")

        except subprocess.TimeoutExpired:
            raise FileOperationError("FFmpeg check timed out")
        except FileNotFoundError:
            raise FileOperationError(
                "FFmpeg not found. Please install FFmpeg: brew install ffmpeg"
            )
        except Exception as e:
            raise FileOperationError(f"FFmpeg check failed: {e}")

    def get_audio_info(self, input_path: Path) -> dict:
        """Get audio information from file using FFprobe.

        Args:
            input_path: Path to input file

        Returns:
            Dictionary with audio information
        """
        try:
            probe = ffmpeg.probe(str(input_path))

            # Find audio stream
            audio_stream = None
            for stream in probe["streams"]:
                if stream["codec_type"] == "audio":
                    audio_stream = stream
                    break

            if not audio_stream:
                raise FileOperationError("No audio stream found in file")

            info = {
                "duration": float(audio_stream.get("duration", 0)),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "codec": audio_stream.get("codec_name", "unknown"),
                "bit_rate": int(audio_stream.get("bit_rate", 0))
                if audio_stream.get("bit_rate")
                else 0,
            }

            # Calculate estimated chunks
            info["estimated_chunks"] = max(
                1, int(info["duration"] / self.chunk_duration)
            )

            logger.info(
                f"📊 Audio info: {info['duration']:.1f}s, "
                f"{info['sample_rate']}Hz, {info['channels']}ch, {info['codec']}"
            )

            return info

        except Exception as e:
            error_msg = f"Failed to get audio info: {e}"
            logger.error(f"🛑 {error_msg}")
            raise FileOperationError(error_msg)

    def convert_to_asr_format(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """Convert audio/video file to ASR-compatible format.

        Args:
            input_path: Path to input file
            output_path: Optional output path (auto-generated if None)

        Returns:
            Path to converted audio file
        """
        if output_path is None:
            temp_name = f"converted_{input_path.stem}.wav"
            output_path = Path(self.temp_directory) / temp_name

        try:
            logger.info(f"🔄 Converting {input_path.name} to ASR format...")

            # Use FFmpeg to convert to 16kHz mono WAV
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec="pcm_s16le",  # 16-bit PCM
                ac=self.target_channels,  # Mono
                ar=self.target_sample_rate,  # 16kHz
                y=None,  # Overwrite output file
            )

            # Run conversion with progress logging
            ffmpeg.run(stream, quiet=True)

            if not output_path.exists():
                raise FileOperationError("Conversion failed - output file not created")

            # Verify the converted file
            converted_info = self.get_audio_info(output_path)
            if converted_info["sample_rate"] != self.target_sample_rate:
                logger.warning(
                    f"🟡 Sample rate mismatch: got {converted_info['sample_rate']}Hz, "
                    f"expected {self.target_sample_rate}Hz"
                )

            logger.info(f"✅ Conversion complete: {output_path.name}")
            return output_path

        except ffmpeg.Error as e:
            error_msg = (
                f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
            )
            logger.error(f"🛑 {error_msg}")
            raise FileOperationError(error_msg)
        except Exception as e:
            error_msg = f"Conversion error: {e}"
            logger.error(f"🛑 {error_msg}")
            raise FileOperationError(error_msg)

    def extract_audio_chunks(
        self, input_path: Path, chunk_duration: Optional[int] = None
    ) -> list[Path]:
        """Extract audio file into chunks for processing.

        Args:
            input_path: Path to input audio file
            chunk_duration: Chunk duration in seconds (uses config default if None)

        Returns:
            List of paths to audio chunks
        """
        if chunk_duration is None:
            chunk_duration = self.chunk_duration

        try:
            # Get audio duration
            audio_info = self.get_audio_info(input_path)
            total_duration = audio_info["duration"]

            # Calculate number of chunks needed
            num_chunks = max(1, int(np.ceil(total_duration / chunk_duration)))

            logger.info(f"📦 Creating {num_chunks} chunks of {chunk_duration}s each")

            chunk_paths = []

            for i in range(num_chunks):
                start_time = i * chunk_duration
                actual_duration = min(chunk_duration, total_duration - start_time)

                chunk_name = f"{input_path.stem}_chunk_{i:03d}.wav"
                chunk_path = Path(self.temp_directory) / chunk_name

                # Extract chunk using FFmpeg
                stream = ffmpeg.input(str(input_path), ss=start_time, t=actual_duration)
                stream = ffmpeg.output(
                    stream,
                    str(chunk_path),
                    acodec="pcm_s16le",
                    ac=self.target_channels,
                    ar=self.target_sample_rate,
                    y=None,
                )

                ffmpeg.run(stream, quiet=True)

                if chunk_path.exists():
                    chunk_paths.append(chunk_path)
                    logger.debug(f"📦 Created chunk {i + 1}/{num_chunks}: {chunk_name}")
                else:
                    logger.warning(f"⚠️ Failed to create chunk {i + 1}")

            logger.info(f"✅ Created {len(chunk_paths)} audio chunks")
            return chunk_paths

        except Exception as e:
            error_msg = f"Chunk extraction failed: {e}"
            logger.error(f"🛑 {error_msg}")
            raise FileOperationError(error_msg)

    def load_audio_data(
        self, audio_path: Path, max_duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """Load audio data from file for ASR processing with memory management.

        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration in seconds to load (for memory management)

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Get file info first to check size
            info = sf.info(str(audio_path))
            file_duration = info.duration
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)

            logger.debug(
                f"📁 Audio file info: {file_duration:.1f}s, {file_size_mb:.1f}MB"
            )

            # Check if we need to limit duration for memory management
            if max_duration and file_duration > max_duration:
                logger.warning(
                    f"⚠️ Large audio file ({file_duration:.1f}s), limiting to {max_duration:.1f}s"  # noqa: E501
                )
                frames_to_read = int(max_duration * info.samplerate)
                audio_data, sample_rate = sf.read(
                    str(audio_path), frames=frames_to_read, dtype=np.float32
                )
            else:
                # Load entire file
                audio_data, sample_rate = sf.read(str(audio_path), dtype=np.float32)

            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Calculate memory usage
            memory_mb = audio_data.nbytes / (1024 * 1024)

            logger.debug(
                f"📡 Loaded audio: {audio_data.shape[0]} samples @ {sample_rate}Hz "
                f"({audio_data.shape[0] / sample_rate:.1f}s, {memory_mb:.1f}MB)"
            )

            return audio_data, sample_rate

        except Exception as e:
            error_msg = f"Failed to load audio: {e}"
            logger.error(f"🔴 {error_msg}")
            raise FileOperationError(error_msg)

    def load_audio_chunks(
        self, audio_path: Path, chunk_duration: float = 30.0
    ) -> List[Tuple[np.ndarray, int]]:
        """Load large audio files in chunks to manage memory usage.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds

        Returns:
            List of (audio_data, sample_rate) tuples for each chunk
        """
        try:
            # Get file info
            info = sf.info(str(audio_path))
            total_duration = info.duration
            sample_rate = info.samplerate

            if total_duration <= chunk_duration:
                # File is small enough, load normally
                logger.debug(f"📄 Small file ({total_duration:.1f}s), loading normally")
                audio_data, sr = self.load_audio_data(audio_path)
                return [(audio_data, sr)]

            logger.info(
                f"📦 Large file ({total_duration:.1f}s), processing in {chunk_duration}s chunks"  # noqa: E501
            )

            chunks = []
            frames_per_chunk = int(chunk_duration * sample_rate)
            total_frames = info.frames

            # Process file in chunks
            with sf.SoundFile(str(audio_path)) as audio_file:
                while audio_file.tell() < total_frames:
                    # Read chunk
                    chunk_data = audio_file.read(frames_per_chunk, dtype=np.float32)

                    if len(chunk_data) == 0:
                        break

                    # Ensure mono
                    if len(chunk_data.shape) > 1:
                        chunk_data = np.mean(chunk_data, axis=1)

                    chunks.append((chunk_data, sample_rate))

                    # Log progress
                    progress = (audio_file.tell() / total_frames) * 100
                    logger.debug(
                        f"📊 Chunk loaded: {len(chunk_data) / sample_rate:.1f}s ({progress:.1f}% complete)"  # noqa: E501
                    )

            logger.info(f"✅ File split into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            error_msg = f"Failed to load audio chunks: {e}"
            logger.error(f"🔴 {error_msg}")
            raise FileOperationError(error_msg)

    def cleanup_chunks(self, chunk_paths: list[Path]) -> None:
        """Clean up temporary chunk files.

        Args:
            chunk_paths: List of chunk file paths to remove
        """
        removed_count = 0
        for chunk_path in chunk_paths:
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove {chunk_path.name}: {e}")

        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} chunk files")


# end src/batch/audio_converter.py
