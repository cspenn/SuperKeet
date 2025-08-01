# start src/utils/transcript_logger.py
"""Transcript logging functionality for SuperKeet."""

import os
from datetime import datetime
from pathlib import Path

from src.config.config_loader import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TranscriptLogger:
    """Logs transcriptions to disk with timestamps."""
    
    def __init__(self):
        """Initialize the transcript logger."""
        self.enabled = config.get("transcripts.enabled", False)
        self.directory = Path(config.get("transcripts.directory", "transcripts"))
        self.format = config.get("transcripts.format", "text")  # text or json
        
        if self.enabled:
            self._ensure_directory()
            logger.info(f"TranscriptLogger initialized, saving to: {self.directory}")
        else:
            logger.info("TranscriptLogger initialized (disabled)")
    
    def _ensure_directory(self):
        """Ensure the transcript directory exists."""
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created transcript directory: {self.directory}")
    
    def log_transcript(self, text: str) -> bool:
        """Log a transcript to disk.
        
        Args:
            text: The transcribed text to log.
            
        Returns:
            True if logged successfully, False otherwise.
        """
        if not self.enabled:
            return True
        
        try:
            timestamp = datetime.now()
            
            if self.format == "text":
                return self._log_text_format(text, timestamp)
            elif self.format == "json":
                return self._log_json_format(text, timestamp)
            else:
                logger.error(f"Unknown transcript format: {self.format}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to log transcript: {e}")
            return False
    
    def _log_text_format(self, text: str, timestamp: datetime) -> bool:
        """Log transcript in plain text format."""
        # Create daily log file
        date_str = timestamp.strftime("%Y-%m-%d")
        log_file = self.directory / f"transcripts-{date_str}.txt"
        
        # Format the entry
        time_str = timestamp.strftime("%H:%M:%S")
        entry = f"[{time_str}] {text}\n"
        
        # Append to file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        
        logger.debug(f"Logged transcript to {log_file}")
        return True
    
    def _log_json_format(self, text: str, timestamp: datetime) -> bool:
        """Log transcript in JSON format."""
        import json
        
        # Create daily log file
        date_str = timestamp.strftime("%Y-%m-%d")
        log_file = self.directory / f"transcripts-{date_str}.json"
        
        # Create entry
        entry = {
            "timestamp": timestamp.isoformat(),
            "text": text,
            "length": len(text)
        }
        
        # Read existing entries or create new list
        entries = []
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except:
                logger.warning("Failed to read existing JSON file, starting fresh")
        
        # Append new entry
        entries.append(entry)
        
        # Write back
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Logged transcript to {log_file}")
        return True
    
    def update_settings(self, enabled: bool, directory: str, format: str):
        """Update transcript logger settings.
        
        Args:
            enabled: Whether transcript logging is enabled.
            directory: Directory path for transcripts.
            format: Format for transcripts (text or json).
        """
        self.enabled = enabled
        self.directory = Path(directory)
        self.format = format
        
        if self.enabled:
            self._ensure_directory()
        
        logger.info(f"Updated transcript settings: enabled={enabled}, dir={directory}, format={format}")


# end src/utils/transcript_logger.py