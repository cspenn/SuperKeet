# start download_model.py
"""Download Parakeet-MLX model to local disk."""

import sys
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from parakeet_mlx import from_pretrained

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def download_model():
    """Download the Parakeet model to local disk."""
    model_id = "mlx-community/parakeet-tdt-0.6b-v2"

    logger.info(f"Downloading model: {model_id}")
    logger.info("This will download the model to the Hugging Face cache directory")
    logger.info("The model will be cached for future use")

    try:
        # This will download and cache the model
        model = from_pretrained(model_id)
        logger.info("Model downloaded successfully!")

        # Test the model with a simple check
        logger.info("Testing model loading...")
        if model is not None:
            logger.info("Model loaded correctly and is ready for use")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


if __name__ == "__main__":
    download_model()

# end download_model.py
