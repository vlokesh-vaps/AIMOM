"""File-system utility helpers.

Handles directory creation, audio file validation, and filename generation.
"""

from datetime import datetime
from pathlib import Path

from config.settings import (
    ASSETS_DIR,
    LOGS_DIR,
    OUTPUT_DIR,
    RECORDINGS_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    TEMP_DIR,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def ensure_directories() -> None:
    """Create all required project directories if they don't exist."""
    for directory in (RECORDINGS_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", directory)


def is_supported_audio(file_path: str | Path) -> bool:
    """Check whether *file_path* has a supported audio extension.

    Args:
        file_path: Path to the audio file.

    Returns:
        ``True`` if the extension is in the supported list.
    """
    return Path(file_path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def generate_filename(title: str, extension: str = ".wav") -> str:
    """Generate a timestamped filename from a meeting title.

    Args:
        title: Meeting title (sanitized to filesystem-safe characters).
        extension: File extension including the dot.

    Returns:
        A filename string like ``my_meeting_20240101_120000.wav``.
    """
    safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
    safe_title = safe_title.strip().replace(" ", "_") or "recording"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_title}_{timestamp}{extension}"
