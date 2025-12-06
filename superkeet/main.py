# start src/main.py
"""Main entry point for SuperKeet."""

import signal
import sys

from superkeet.ui.tray_app import SuperKeetApp
from superkeet.utils.logger import setup_logger

# Add local path handling handled by poetry/pip module installation

logger = setup_logger(__name__)

# Global app instance for signal handler
app_instance = None


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    logger.info("Received interrupt signal, shutting down...")
    if app_instance:
        try:
            app_instance.cleanup()
        except Exception as e:
            logger.error(f"🛑 Error during application cleanup: {e}")
            # Continue with shutdown even if cleanup fails
    sys.exit(0)


def main() -> None:
    """Main function to run SuperKeet."""
    global app_instance

    logger.info("Starting SuperKeet...")

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Create and run application
        app_instance = SuperKeetApp()
        app_instance.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
        # Ensure proper cleanup
        if app_instance:
            try:
                app_instance.cleanup()
            except Exception as e:
                logger.error(f"🛑 Error during application cleanup: {e}")
                # Continue with shutdown even if cleanup fails
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# end src/main.py
