"""Audio format converter using FFmpeg.

Converts unsupported audio formats to 16 kHz mono PCM WAV.
"""

import subprocess
from pathlib import Path

from config.settings import SAMPLE_RATE, CHANNELS, TEMP_DIR
from utils.file_utils import generate_filename
from utils.logger import get_logger

logger = get_logger(__name__)


class ConversionError(Exception):
    """Raised when FFmpeg conversion fails."""


class AudioConverter:
    """Converts audio files to WAV format using FFmpeg."""

    @staticmethod
    def is_wav(file_path: Path) -> bool:
        """Check whether *file_path* is already a WAV file.

        Args:
            file_path: Path to the audio file.

        Returns:
            ``True`` if the extension is ``.wav``.
        """
        return file_path.suffix.lower() == ".wav"

    @staticmethod
    def needs_transcoding_to_mono_16k(file_path: Path) -> bool:
        """Use ffprobe to check if the file is already 16kHz mono.

        If ffprobe is missing or fails, assume transcoding is needed to be safe.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels,sample_rate",
            "-of", "csv=p=0",
            str(file_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                # Output format: <channels>,<sample_rate> (e.g. "2,44100" or "1,16000")
                parts = result.stdout.strip().split(",")
                if len(parts) == 2:
                    channels = int(parts[0])
                    sample_rate = int(parts[1])
                    if channels == 1 and sample_rate == 16000:
                        return False # Correct format, no transcoding needed
        except Exception as e:
            logger.debug("ffprobe check failed (falling back to safety transcode): %s", e)
        return True # Default to transcoding to be safe

    @staticmethod
    def convert_to_wav(input_path: Path, title: str = "converted") -> Path:
        """Convert an audio file to 16 kHz mono PCM WAV.

        If the file is already WAV and is mono 16kHz, it is returned unchanged.

        Args:
            input_path: Source audio file.
            title: Base name for the converted file.

        Returns:
            Path to the WAV file (original or newly converted).

        Raises:
            ConversionError: If FFmpeg is missing or conversion fails.
        """
        if AudioConverter.is_wav(input_path):
            if not AudioConverter.needs_transcoding_to_mono_16k(input_path):
                logger.info("File is already mono 16kHz WAV, skipping conversion: %s", input_path)
                return input_path
            else:
                logger.info("WAV file is not mono 16kHz, forcing transcoding: %s", input_path)

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_filename = generate_filename(title, ".wav")
        output_path = TEMP_DIR / output_filename

        cmd = [
            "ffmpeg",
            "-y",                         # overwrite without asking
            "-i", str(input_path),        # input file
            "-ar", str(SAMPLE_RATE),      # 16 kHz
            "-ac", str(CHANNELS),         # mono
            "-sample_fmt", "s16",         # 16-bit PCM
            "-f", "wav",                  # output format
            str(output_path),
        ]

        logger.info("Converting %s → %s", input_path.name, output_path.name)
        logger.debug("FFmpeg command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError:
            msg = (
                "FFmpeg not found. Please install FFmpeg and ensure it is "
                "available on your system PATH."
            )
            logger.error(msg)
            raise ConversionError(msg)
        except subprocess.TimeoutExpired:
            msg = "FFmpeg conversion timed out after 120 seconds."
            logger.error(msg)
            raise ConversionError(msg)

        if result.returncode != 0:
            stderr_snippet = result.stderr[-500:] if result.stderr else "No error output"
            msg = f"FFmpeg conversion failed (exit {result.returncode}): {stderr_snippet}"
            logger.error(msg)
            raise ConversionError(msg)

        logger.info("Conversion complete: %s (%.2f MB)", output_path, output_path.stat().st_size / 1_048_576)
        return output_path

    @staticmethod
    def compress_to_mp3(input_path: Path, title: str = "compressed") -> Path:
        """Compress an audio file to 16 kHz mono MP3 (48kbps) to reduce size for upload.

        Args:
            input_path: Source audio file.
            title: Base name for the compressed file.

        Returns:
            Path to the compressed MP3 file.
        """
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_filename = generate_filename(title, ".mp3")
        output_path = TEMP_DIR / output_filename

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-codec:a", "libmp3lame",
            "-b:a", "48k",
            "-ar", "16000",
            "-ac", "1",
            str(output_path)
        ]

        logger.info("Compressing %s → %s", input_path.name, output_path.name)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
            if result.returncode != 0:
                # Fallback without libmp3lame if not compiled in ffmpeg
                cmd_fallback = [
                    "ffmpeg",
                    "-y",
                    "-i", str(input_path),
                    "-b:a", "48k",
                    "-ar", "16000",
                    "-ac", "1",
                    str(output_path)
                ]
                logger.info("Retrying compression with fallback parameters...")
                result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=300, check=False)
        except Exception as e:
            logger.error("FFmpeg compression failed: %s", e)
            raise ConversionError(f"Compression failed: {e}")

        if result.returncode != 0:
            stderr_snippet = result.stderr[-500:] if result.stderr else "No error output"
            raise ConversionError(f"FFmpeg compression failed: {stderr_snippet}")

        logger.info("Compression complete: %s (%.2f MB)", output_path, output_path.stat().st_size / 1_048_576)
        return output_path
