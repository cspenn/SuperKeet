# start download_model.py
"""Download Parakeet-MLX model to local disk."""

import sys
from pathlib import Path

from parakeet_mlx import from_pretrained

from superkeet.utils.logger import setup_logger

# Add project root to path for absolute imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = setup_logger(__name__)


def download_model():
    """Download the Parakeet model to local disk with progress indication."""
    import threading
    import time

    from tqdm import tqdm

    model_id = "mlx-community/parakeet-tdt-0.6b-v2"

    logger.info(f"Downloading model: {model_id}")
    logger.info("This will download the model to the Hugging Face cache directory")
    logger.info("The model will be cached for future use")

    # Create a progress bar for model download
    progress_bar = tqdm(
        total=100,
        desc="Downloading model",
        unit="%",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    download_complete = threading.Event()
    download_error = None

    def simulate_progress():
        """Simulate download progress until actual download completes."""
        nonlocal download_error
        progress = 0
        while not download_complete.is_set() and progress < 95:
            time.sleep(0.5)  # Update every 500ms
            if progress < 30:
                progress += 1  # Fast initial progress
            elif progress < 60:
                progress += 0.5  # Moderate progress
            else:
                progress += 0.2  # Slower near end

            progress_bar.n = min(int(progress), 95)
            progress_bar.refresh()

    def download_worker():
        """Worker thread for actual model download."""
        nonlocal download_error
        try:
            # This will download and cache the model
            model = from_pretrained(model_id)

            # Test the model with a simple check
            if model is not None:
                logger.info("Model loaded correctly and is ready for use")
            else:
                raise ValueError("Model loaded but returned None")

        except Exception as e:
            download_error = e
        finally:
            download_complete.set()

    try:
        # Start download in background thread
        download_thread = threading.Thread(target=download_worker, daemon=True)
        progress_thread = threading.Thread(target=simulate_progress, daemon=True)

        download_thread.start()
        progress_thread.start()

        # Wait for download to complete
        download_thread.join()

        # Complete progress bar
        progress_bar.n = 100
        progress_bar.refresh()
        progress_bar.close()

        # Check for errors
        if download_error:
            raise download_error

        logger.info("Model downloaded successfully!")

    except Exception as e:
        progress_bar.close()
        logger.error(f"Failed to download model: {e}")
        raise


if __name__ == "__main__":
    download_model()

# end download_model.py
