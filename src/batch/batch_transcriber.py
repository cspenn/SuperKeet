"""Batch transcriber for processing multiple files using existing ASR infrastructure."""

import json
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from src.asr.transcriber import ASRTranscriber
from src.batch.audio_converter import AudioConverter
from src.batch.file_processor import FileProcessor
from src.config.config_loader import config
from src.utils.exceptions import ASRError, FileOperationError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BatchTranscriptionWorker(QThread):
    """Worker thread for batch transcription processing."""

    # Signals for progress updates
    progress_updated = Signal(int, int, str)  # current, total, current_file
    file_completed = Signal(str, str, bool)  # filename, transcript, success
    batch_completed = Signal(dict)  # results summary
    error_occurred = Signal(str)  # error message

    def __init__(self, file_paths: List[Path], parent: Optional[QObject] = None):
        """Initialize batch transcription worker.

        Args:
            file_paths: List of file paths to process
            parent: Parent QObject
        """
        super().__init__(parent)
        self.file_paths = file_paths
        self.should_stop = False

        # Initialize components
        self.file_processor = FileProcessor()
        self.audio_converter = AudioConverter()
        self.transcriber: Optional[ASRTranscriber] = None

        # Results tracking
        self.results = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "transcripts": [],
            "errors": [],
        }

    def run(self) -> None:
        """Run the batch transcription process."""
        try:
            logger.info(
                f"🚀 Starting batch transcription of {len(self.file_paths)} files"
            )

            # Initialize ASR transcriber
            self._initialize_transcriber()

            # Validate files first
            valid_files, invalid_files = self.file_processor.validate_batch(
                self.file_paths
            )

            # Report invalid files
            for file_path, error in invalid_files:
                self.results["errors"].append({"file": str(file_path), "error": error})
                self.file_completed.emit(file_path.name, "", False)

            # Process valid files
            for i, file_path in enumerate(valid_files):
                if self.should_stop:
                    logger.info("🛑 Batch transcription stopped by user")
                    break

                self.progress_updated.emit(i + 1, len(valid_files), file_path.name)

                try:
                    # Process single file
                    transcript = self._process_file(file_path)

                    # Save transcript
                    self._save_transcript(file_path, transcript)

                    # Update results
                    self.results["successful_files"] += 1
                    self.results["transcripts"].append(
                        {
                            "file": str(file_path),
                            "transcript": transcript,
                            "length": len(transcript),
                        }
                    )

                    self.file_completed.emit(file_path.name, transcript, True)
                    logger.info(f"✅ Completed: {file_path.name}")

                except Exception as e:
                    error_msg = f"Processing failed for {file_path.name}: {e}"
                    logger.error(f"🛑 {error_msg}")

                    self.results["failed_files"] += 1
                    self.results["errors"].append(
                        {"file": str(file_path), "error": str(e)}
                    )

                    self.file_completed.emit(file_path.name, "", False)

                self.results["processed_files"] += 1

            # Cleanup and finish
            self._cleanup()

            logger.info(
                f"🏁 Batch complete: {self.results['successful_files']} successful, "
                f"{self.results['failed_files']} failed"
            )

            self.batch_completed.emit(self.results)

        except Exception as e:
            error_msg = f"Batch transcription failed: {e}"
            logger.error(f"🛑 {error_msg}")
            self.error_occurred.emit(error_msg)

    def stop(self) -> None:
        """Stop the batch transcription process."""
        self.should_stop = True
        logger.info("🛑 Stop requested for batch transcription")

    def _initialize_transcriber(self) -> None:
        """Initialize the ASR transcriber."""
        try:
            # Use same model and settings as real-time transcription
            # ASRTranscriber gets model_id from config internally
            self.transcriber = ASRTranscriber()
            
            # Load the model
            self.transcriber.load_model()
            
            logger.info("🟢 ASR transcriber initialized for batch processing")

        except Exception as e:
            raise ASRError(f"Failed to initialize ASR transcriber: {e}")

    def _process_file(self, file_path: Path) -> str:
        """Process a single file for transcription.

        Args:
            file_path: Path to file to process

        Returns:
            Transcribed text
        """
        logger.info(f"🔄 Processing: {file_path.name}")

        try:
            # Convert to ASR format
            converted_path = self.audio_converter.convert_to_asr_format(file_path)

            # Get audio info to decide on chunking strategy
            audio_info = self.audio_converter.get_audio_info(converted_path)
            duration = audio_info.get("duration", 0)

            if duration <= self.audio_converter.chunk_duration:
                # Process as single file
                transcript = self._transcribe_audio_file(converted_path)
            else:
                # Process in chunks
                transcript = self._transcribe_chunked_audio(converted_path)

            # Clean up converted file
            if converted_path.exists():
                converted_path.unlink()

            return transcript.strip()

        except Exception as e:
            raise FileOperationError(f"File processing failed: {e}")

    def _transcribe_audio_file(self, audio_path: Path) -> str:
        """Transcribe a single audio file using chunked processing for large files.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        if not self.transcriber:
            raise ASRError("Transcriber not initialized")

        try:
            # Get file size to determine processing strategy
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            max_chunk_duration = 60.0  # 60 seconds per chunk
            
            if file_size_mb > 50:  # Large file threshold
                logger.info(f"📦 Large file detected ({file_size_mb:.1f}MB), using chunked processing")
                return self._transcribe_large_file_chunked(audio_path, max_chunk_duration)
            else:
                # Standard processing for smaller files
                logger.debug(f"📄 Standard processing for {audio_path.name} ({file_size_mb:.1f}MB)")
                
                # Load with duration limit to prevent memory issues
                max_duration = 300.0  # 5 minutes max
                audio_data, sample_rate = self.audio_converter.load_audio_data(audio_path, max_duration)

                # Transcribe using existing ASR infrastructure
                transcript = self.transcriber.transcribe(audio_data, sample_rate)

                logger.debug(
                    f"📝 Transcribed {audio_path.name}: {len(transcript)} characters"
                )
                return transcript

        except Exception as e:
            raise ASRError(f"Transcription failed: {e}")

    def _transcribe_large_file_chunked(self, audio_path: Path, chunk_duration: float) -> str:
        """Transcribe large audio file using chunked processing.
        
        Args:
            audio_path: Path to large audio file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            Combined transcription text
        """
        try:
            # Load file in chunks
            audio_chunks = self.audio_converter.load_audio_chunks(audio_path, chunk_duration)
            
            if not audio_chunks:
                return ""
            
            transcripts = []
            total_chunks = len(audio_chunks)
            
            for i, (chunk_data, sample_rate) in enumerate(audio_chunks):
                if self.should_stop:
                    logger.warning("⚠️ Chunked transcription stopped by user")
                    break
                
                logger.debug(f"📝 Processing chunk {i+1}/{total_chunks}")
                
                try:
                    # Transcribe chunk
                    chunk_transcript = self.transcriber.transcribe(chunk_data, sample_rate)
                    transcripts.append(chunk_transcript)
                    
                    # Clear chunk data from memory immediately
                    del chunk_data
                    
                    # Force garbage collection every few chunks
                    if (i + 1) % 5 == 0:
                        import gc
                        gc.collect()
                        logger.debug(f"🗑️ Memory cleanup after chunk {i+1}")
                    
                except Exception as e:
                    logger.error(f"❌ Error transcribing chunk {i+1}: {e}")
                    transcripts.append(f"[Error in chunk {i+1}: {str(e)}]")
            
            # Combine transcripts with proper spacing
            combined_transcript = " ".join(transcript.strip() for transcript in transcripts if transcript.strip())
            
            logger.info(f"✅ Chunked transcription complete: {len(combined_transcript)} characters from {total_chunks} chunks")
            return combined_transcript
            
        except Exception as e:
            logger.error(f"❌ Chunked transcription failed: {e}")
            raise ASRError(f"Chunked transcription failed: {e}")

    def _transcribe_chunked_audio(self, audio_path: Path) -> str:
        """Transcribe audio file by processing chunks.

        Args:
            audio_path: Path to audio file

        Returns:
            Combined transcribed text
        """
        if not self.transcriber:
            raise ASRError("Transcriber not initialized")

        try:
            # Extract chunks
            chunk_paths = self.audio_converter.extract_audio_chunks(audio_path)

            if not chunk_paths:
                raise FileOperationError("No chunks created")

            transcripts = []

            # Process each chunk
            for i, chunk_path in enumerate(chunk_paths):
                if self.should_stop:
                    break

                logger.debug(f"🔄 Processing chunk {i + 1}/{len(chunk_paths)}")

                try:
                    chunk_transcript = self._transcribe_audio_file(chunk_path)
                    if chunk_transcript.strip():
                        transcripts.append(chunk_transcript.strip())

                except Exception as e:
                    logger.warning(f"⚠️ Chunk {i + 1} failed: {e}")
                    # Continue with other chunks

            # Clean up chunks
            self.audio_converter.cleanup_chunks(chunk_paths)

            # Combine transcripts
            full_transcript = " ".join(transcripts)
            logger.info(
                f"📝 Combined transcript: {len(full_transcript)} characters from {len(transcripts)} chunks"
            )

            return full_transcript

        except Exception as e:
            raise ASRError(f"Chunked transcription failed: {e}")

    def _save_transcript(self, input_path: Path, transcript: str) -> None:
        """Save transcript to output file.

        Args:
            input_path: Original input file path
            transcript: Transcribed text to save
        """
        try:
            # Get output format from config
            output_format = config.get(
                "batch_processing.output.transcript_format", "txt"
            )
            include_timestamps = config.get(
                "batch_processing.output.include_timestamps", False
            )

            # Generate output path
            output_path = self.file_processor.get_output_path(input_path, output_format)

            # Prepare content based on format
            if output_format.lower() == "json":
                content = self._format_json_output(
                    input_path, transcript, include_timestamps
                )
            else:
                content = transcript

            # Write transcript file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"💾 Transcript saved: {output_path.name}")

        except Exception as e:
            logger.error(f"🛑 Failed to save transcript: {e}")
            # Don't raise - we still have the transcript in memory

    def _format_json_output(
        self, input_path: Path, transcript: str, include_timestamps: bool
    ) -> str:
        """Format transcript as JSON output.

        Args:
            input_path: Original input file path
            transcript: Transcribed text
            include_timestamps: Whether to include timestamps

        Returns:
            JSON formatted string
        """
        output_data = {
            "input_file": str(input_path),
            "transcript": transcript,
            "length": len(transcript),
            "processing_info": {
                "model": config.get("asr.model_id", "mlx-community/parakeet-tdt-0.6b-v2"),
                "sample_rate": 16000,
                "channels": 1,
            },
        }

        if include_timestamps:
            # For now, we don't have word-level timestamps from Parakeet
            # This could be enhanced in the future
            output_data["timestamps"] = []

        return json.dumps(output_data, indent=2, ensure_ascii=False)

    def _cleanup(self) -> None:
        """Clean up resources after batch processing."""
        try:
            # Cleanup ASR transcriber
            if self.transcriber:
                self.transcriber.unload_model()
                self.transcriber = None
                logger.debug("🔧 ASR transcriber unloaded")

            # Cleanup temporary files
            self.file_processor.cleanup_temp_files()

            logger.info("🧹 Batch processing cleanup completed")

        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")


class BatchTranscriber(QObject):
    """Manager for batch transcription operations."""

    # Signals
    progress_updated = Signal(int, int, str)  # current, total, current_file
    file_completed = Signal(str, str, bool)  # filename, transcript, success
    batch_completed = Signal(dict)  # results summary
    error_occurred = Signal(str)  # error message

    def __init__(self, parent: Optional[QObject] = None):
        """Initialize batch transcriber manager."""
        super().__init__(parent)
        self.worker: Optional[BatchTranscriptionWorker] = None
        self.is_running = False

    def start_batch(self, file_paths: List[Path]) -> None:
        """Start batch transcription of files.

        Args:
            file_paths: List of file paths to process
        """
        if self.is_running:
            logger.warning("🟡 Batch transcription already running")
            return

        if not file_paths:
            self.error_occurred.emit("No files provided for batch transcription")
            return

        logger.info(f"🚀 Starting batch transcription: {len(file_paths)} files")

        # Create and configure worker
        self.worker = BatchTranscriptionWorker(file_paths)

        # Connect signals
        self.worker.progress_updated.connect(self.progress_updated.emit)
        self.worker.file_completed.connect(self.file_completed.emit)
        self.worker.batch_completed.connect(self._on_batch_completed)
        self.worker.error_occurred.connect(self._on_error_occurred)
        self.worker.finished.connect(self._on_worker_finished)

        # Start processing
        self.is_running = True
        self.worker.start()

    def stop_batch(self) -> None:
        """Stop the current batch transcription."""
        if self.worker and self.is_running:
            logger.info("🛑 Stopping batch transcription")
            self.worker.stop()
            self.worker.wait(5000)  # Wait up to 5 seconds

    def _on_batch_completed(self, results: dict) -> None:
        """Handle batch completion."""
        self.is_running = False
        self.batch_completed.emit(results)

    def _on_error_occurred(self, error: str) -> None:
        """Handle batch error."""
        self.is_running = False
        self.error_occurred.emit(error)

    def _on_worker_finished(self) -> None:
        """Handle worker thread finished."""
        self.is_running = False
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
