# start src/audio/memory_manager.py
"""Memory management for audio recording.

This module handles buffer management, memory monitoring, and
resource limits for the AudioRecorder class.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from superkeet.utils.logger import setup_logger

if TYPE_CHECKING:
    from superkeet.audio.recorder import AudioRecorder

logger = setup_logger(__name__)


def should_stop_due_to_limits(recorder: "AudioRecorder") -> bool:
    """Check if recording should stop due to memory or duration limits.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        True if recording should stop due to limits.
    """
    # Check duration limit
    if recorder.max_recording_duration > 0:
        current_duration = time.time() - recorder.recording_start_time
        if current_duration > recorder.max_recording_duration:
            logger.warning(f"Recording duration limit reached: {current_duration:.1f}s")
            return True

    # Check buffer size limit
    if recorder.buffer_size_limit > 0:
        current_size_mb = recorder._total_audio_size / (1024 * 1024)
        if current_size_mb > recorder.buffer_size_limit:
            logger.warning(f"Audio buffer size limit reached: {current_size_mb:.1f}MB")
            return True

    return False


def check_memory_usage(recorder: "AudioRecorder") -> dict[str, Any]:
    """Check current memory usage of audio buffers.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        Dictionary with memory usage statistics.
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        audio_buffer_size = recorder._total_audio_size
        queue_size = recorder.audio_queue.qsize()

        stats = {
            "audio_buffer_mb": audio_buffer_size / (1024 * 1024),
            "queue_items": queue_size,
            "process_rss_mb": memory_info.rss / (1024 * 1024),
            "process_vms_mb": memory_info.vms / (1024 * 1024),
            "buffer_limit_mb": recorder.buffer_size_limit,
            "buffer_usage_percent": (
                (audio_buffer_size / (recorder.buffer_size_limit * 1024 * 1024)) * 100
                if recorder.buffer_size_limit > 0
                else 0
            ),
        }

        if recorder.enable_buffer_monitoring:
            logger.debug(
                f"Memory: audio={stats['audio_buffer_mb']:.1f}MB, "
                f"queue={stats['queue_items']}, "
                f"RSS={stats['process_rss_mb']:.1f}MB"
            )

        return stats

    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        return {
            "audio_buffer_mb": recorder._total_audio_size / (1024 * 1024),
            "queue_items": recorder.audio_queue.qsize(),
            "buffer_limit_mb": recorder.buffer_size_limit,
        }
    except Exception as e:
        logger.debug(f"Memory check failed: {e}")
        return {}


def get_memory_stats(recorder: "AudioRecorder") -> dict[str, Any]:
    """Get detailed memory statistics for the recorder.

    Args:
        recorder: The AudioRecorder instance.

    Returns:
        Dictionary with detailed memory statistics.
    """
    stats = check_memory_usage(recorder)

    # Add recording duration info
    if recorder.recording and hasattr(recorder, "recording_start_time"):
        stats["recording_duration_s"] = time.time() - recorder.recording_start_time
        stats["max_duration_s"] = recorder.max_recording_duration

    # Add buffer counts
    stats["audio_chunks_count"] = len(recorder.audio_data)

    return stats


def clear_audio_buffers(recorder: "AudioRecorder") -> None:
    """Clear all audio buffers and reset counters.

    Args:
        recorder: The AudioRecorder instance.
    """
    logger.debug("Clearing audio buffers...")

    # Clear the audio data list
    recorder.audio_data.clear()
    recorder._total_audio_size = 0

    # Empty the queue
    while not recorder.audio_queue.empty():
        try:
            recorder.audio_queue.get_nowait()
        except Exception:
            break

    logger.debug("Audio buffers cleared")


def save_debug_audio(
    recorder: "AudioRecorder", audio_data: np.ndarray, suffix: str = ""
) -> None:
    """Save audio data for debugging purposes.

    Args:
        recorder: The AudioRecorder instance.
        audio_data: The audio data to save.
        suffix: Optional suffix for the filename.
    """
    try:
        import soundfile as sf

        debug_dir = Path("debug_audio")
        debug_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{timestamp}{suffix}.wav"
        filepath = debug_dir / filename

        # Ensure proper sample rate
        sample_rate = int(recorder.sample_rate)

        sf.write(str(filepath), audio_data, sample_rate)
        logger.debug(f"Saved debug audio to {filepath}")

    except Exception as e:
        logger.debug(f"Failed to save debug audio: {e}")


# end src/audio/memory_manager.py
