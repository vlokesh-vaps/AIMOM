"""Abstract base class for Speech-to-Text providers.

Every STT provider must inherit from :class:`BaseSTTProvider` and implement
all abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from models.recording import TranscriptionResult


class STTError(Exception):
    """Base exception for STT provider errors."""


class APIKeyMissingError(STTError):
    """Raised when the required API key is not configured."""


class NetworkError(STTError):
    """Raised on network or connectivity failures."""


class UnsupportedFormatError(STTError):
    """Raised when the audio format is not supported by the provider."""


class TranscriptionTimeoutError(STTError):
    """Raised when the transcription request times out."""


class BaseSTTProvider(ABC):
    """Abstract interface for speech-to-text providers.

    Subclasses must implement :meth:`transcribe`, :meth:`health_check`,
    :meth:`supported_languages`, and :meth:`supported_formats`.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model: str,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file (WAV format preferred).
            language: Display name of the language (e.g. ``"English"``).
            model: Display name of the model (e.g. ``"Deepgram Nova-3"``).

        Returns:
            A :class:`TranscriptionResult` with the transcript text.

        Raises:
            STTError: On any transcription failure.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Verify that the provider is reachable and the API key is valid.

        Returns:
            ``True`` if the provider is healthy.
        """

    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Return display names of languages this provider supports.

        Returns:
            List of language display names.
        """

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return audio file extensions this provider accepts.

        Returns:
            List of extensions like ``[".wav", ".flac"]``.
        """

    def requires_conversion_to_wav(self, audio_path: Path, model: str) -> bool:
        """Determine whether the audio file requires conversion to WAV format.

        Default implementation checks if the file extension is not in supported_formats.
        Subclasses can override this logic (e.g. if streaming mode requires WAV).
        """
        return audio_path.suffix.lower() not in self.supported_formats()
