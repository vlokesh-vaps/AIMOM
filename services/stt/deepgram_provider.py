"""Deepgram Speech-to-Text provider using the Deepgram SDK v7.

Supports Nova-3 and Nova-2 models with diarization, smart formatting,
topic detection, summarization, and entity detection.

SDK v7 uses keyword-only arguments directly on ``transcribe_file()``
instead of the older ``PrerecordedOptions`` class.
"""

import time
from pathlib import Path

from deepgram import DeepgramClient

from config.settings import DEEPGRAM_API_KEY, DEEPGRAM_LANGUAGE_MAP, DEEPGRAM_MODEL_MAP
from models.recording import TranscriptionResult
from services.stt.base import (
    APIKeyMissingError,
    BaseSTTProvider,
    NetworkError,
    STTError,
    TranscriptionTimeoutError,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class DeepgramProvider(BaseSTTProvider):
    """Deepgram ASR provider using the official Python SDK v7.

    Features enabled: ``smart_format``, ``diarize``, ``punctuate``,
    ``topics``, ``summarize v2``, ``detect_entities``.
    """

    def __init__(self) -> None:
        self._api_key: str = DEEPGRAM_API_KEY

    # ------------------------------------------------------------------
    # BaseSTTProvider implementation
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model: str,
    ) -> TranscriptionResult:
        """Transcribe audio using Deepgram.

        Args:
            audio_path: Path to the audio file.
            language: Display language name (e.g. ``"English"``).
            model: Display model name (e.g. ``"Deepgram Nova-3"``).

        Returns:
            A :class:`TranscriptionResult`.

        Raises:
            APIKeyMissingError: If ``DEEPGRAM_API_KEY`` is empty.
            STTError: On transcription failure.
        """
        self._validate_api_key()

        model_name = DEEPGRAM_MODEL_MAP.get(model)
        if model_name is None:
            raise STTError(f"Unknown Deepgram model: {model}")

        language_code = DEEPGRAM_LANGUAGE_MAP.get(language)

        logger.info(
            "Deepgram transcription — model=%s, lang=%s, file=%s",
            model_name, language_code or "auto", audio_path.name,
        )

        # Calculate a generous timeout based on file size
        file_size_mb = audio_path.stat().st_size / 1_048_576
        # 10 seconds per MB, minimum 300s (5 min), maximum 7200s (2 hr)
        timeout_seconds = max(300, min(int(file_size_mb * 10), 7200))
        logger.info(
            "File size: %.1f MB — using timeout: %ds",
            file_size_mb, timeout_seconds,
        )

        start_time = time.time()

        try:
            client = DeepgramClient(api_key=self._api_key)

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            # Build keyword arguments for the SDK v7 API
            kwargs = self._build_transcribe_kwargs(model_name, language_code)

            logger.debug("Deepgram kwargs: %s", {k: v for k, v in kwargs.items() if k != "request"})

            response = client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                request_options={"timeout_in_seconds": timeout_seconds},
                **kwargs,
            )

            transcript = self._extract_transcript(response)

        except (APIKeyMissingError, STTError):
            raise
        except Exception as exc:
            error_msg = str(exc).lower()
            if "unauthorized" in error_msg or "401" in error_msg or "403" in error_msg:
                raise APIKeyMissingError(
                    "Deepgram API key is invalid. Please check your .env file."
                ) from exc
            if "timeout" in error_msg or "timed out" in error_msg:
                raise TranscriptionTimeoutError(
                    f"Deepgram transcription timed out: {exc}"
                ) from exc
            if "connection" in error_msg or "network" in error_msg:
                raise NetworkError(f"Cannot reach Deepgram API: {exc}") from exc
            raise STTError(f"Deepgram transcription failed: {exc}") from exc

        elapsed = time.time() - start_time
        logger.info("Deepgram transcription complete — %.1f seconds", elapsed)

        return TranscriptionResult(
            text=transcript.strip(),
            duration_seconds=elapsed,
            provider="Deepgram",
            model=model,
            language=language_code or "auto",
        )

    def health_check(self) -> bool:
        """Verify the API key is configured.

        Returns:
            ``True`` if ``DEEPGRAM_API_KEY`` is set.
        """
        return bool(self._api_key)

    def supported_languages(self) -> list[str]:
        """Return supported language display names."""
        return list(DEEPGRAM_LANGUAGE_MAP.keys())

    def supported_formats(self) -> list[str]:
        """Return supported audio formats."""
        return [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm", ".mp4"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_api_key(self) -> None:
        """Raise if the Deepgram API key is not set."""
        if not self._api_key:
            raise APIKeyMissingError(
                "DEEPGRAM_API_KEY is not set. Please add it to your .env file."
            )

    @staticmethod
    def _build_transcribe_kwargs(
        model_name: str,
        language_code: str | None,
    ) -> dict:
        """Build keyword arguments for ``transcribe_file()``.

        Uses the Deepgram SDK v7 keyword-only parameter style.
        """
        kwargs: dict = {
            "model": model_name,
            "summarize": "v2",
            "topics": True,
            "custom_topic": ["IVRM,AIVRM,GOSHALA"],
            "detect_entities": True,
            "smart_format": True,
            "diarize": True,
            "paragraphs": True,
        }
        if language_code is not None:
            kwargs["language"] = language_code

        return kwargs

    @staticmethod
    def _extract_transcript(response: object) -> str:
        """Extract the transcript text from a Deepgram response, formatting by speaker if available."""
        try:
            alt = response.results.channels[0].alternatives[0]
            
            # Check if paragraph/speaker information is available
            if hasattr(alt, "paragraphs") and alt.paragraphs and alt.paragraphs.paragraphs:
                paragraphs_list = []
                for p in alt.paragraphs.paragraphs:
                    speaker = p.speaker if p.speaker is not None else 0
                    p_text = " ".join([s.text for s in p.sentences if s.text]) if p.sentences else ""
                    if p_text:
                        paragraphs_list.append(f"Speaker {speaker}: {p_text}")
                
                if paragraphs_list:
                    return "\n\n".join(paragraphs_list)
            
            # Fallback to standard transcript if paragraphs/speakers info is missing
            return alt.transcript or ""
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("Could not extract transcript from response: %s", exc)
            return str(response)


