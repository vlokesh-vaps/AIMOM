"""Data models for recordings and transcription results."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Recording:
    """Represents a single meeting recording session.

    Attributes:
        title: User-provided meeting title.
        source_path: Path to the audio file (uploaded or recorded).
        output_dir: Directory where transcription output is saved.
        language: Language selection from UI dropdown.
        provider: Display name of the STT provider.
        model: Display name of the STT model.
        created_at: Timestamp of creation.
    """

    title: str
    source_path: Path
    output_dir: Path
    language: str = "English"
    provider: str = ""
    model: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TranscriptionResult:
    """Result returned by an STT provider after transcription.

    Attributes:
        text: The full transcribed text.
        duration_seconds: Wall-clock seconds the transcription took.
        provider: Display name of the provider used.
        model: Display name of the model used.
        language: Language code used for transcription.
    """

    text: str
    duration_seconds: float
    provider: str
    model: str
    language: str
