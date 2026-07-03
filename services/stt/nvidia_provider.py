"""NVIDIA NIM Speech-to-Text provider using nvidia-riva-client (gRPC).

Supports two models via NVIDIA cloud:
- Parakeet CTC 1.1B  (streaming recognition)
- Whisper Large v3    (offline recognition)
"""

import time
import wave
from pathlib import Path

import riva.client

from config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_GRPC_SERVER,
    NVIDIA_LANGUAGE_MAP,
    NVIDIA_MODEL_MAP,
)
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


class NvidiaProvider(BaseSTTProvider):
    """NVIDIA NIM ASR provider using Riva gRPC client.

    Uses the NVIDIA cloud endpoint at ``grpc.nvcf.nvidia.com:443`` with
    function-id routing to select between Parakeet and Whisper models.
    """

    def __init__(self) -> None:
        self._api_key: str = NVIDIA_API_KEY

    # ------------------------------------------------------------------
    # BaseSTTProvider implementation
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model: str,
    ) -> TranscriptionResult:
        """Transcribe audio using NVIDIA NIM.

        Args:
            audio_path: Path to a WAV audio file.
            language: Display language name (e.g. ``"English"``).
            model: Display model name (e.g. ``"NVIDIA Parakeet CTC 1.1B"``).

        Returns:
            A :class:`TranscriptionResult`.

        Raises:
            APIKeyMissingError: If ``NVIDIA_API_KEY`` is empty.
            STTError: On transcription failure.
        """
        self._validate_api_key()

        model_config = NVIDIA_MODEL_MAP.get(model)
        if model_config is None:
            raise STTError(f"Unknown NVIDIA model: {model}")

        language_code = NVIDIA_LANGUAGE_MAP.get(language, "en-US")
        function_id = model_config["function_id"]
        mode = model_config["mode"]

        logger.info(
            "NVIDIA transcription — model=%s, lang=%s, mode=%s, file=%s",
            model, language_code, mode, audio_path.name,
        )

        start_time = time.time()

        try:
            auth = self._create_auth(function_id)
            asr_service = riva.client.ASRService(auth)

            if mode == "streaming":
                transcript = self._transcribe_streaming(asr_service, audio_path, language_code)
            else:
                transcript = self._transcribe_offline(asr_service, audio_path, language_code)

        except (APIKeyMissingError, STTError):
            raise
        except Exception as exc:
            error_msg = str(exc)
            if "StatusCode.UNAVAILABLE" in error_msg or "failed to connect" in error_msg.lower():
                raise NetworkError(f"Cannot reach NVIDIA server: {exc}") from exc
            if "StatusCode.DEADLINE_EXCEEDED" in error_msg:
                raise TranscriptionTimeoutError(f"NVIDIA transcription timed out: {exc}") from exc
            raise STTError(f"NVIDIA transcription failed: {exc}") from exc

        elapsed = time.time() - start_time
        logger.info("NVIDIA transcription complete — %.1f seconds", elapsed)

        return TranscriptionResult(
            text=transcript.strip(),
            duration_seconds=elapsed,
            provider="NVIDIA",
            model=model,
            language=language_code,
        )

    def health_check(self) -> bool:
        """Verify the API key is configured.

        Returns:
            ``True`` if ``NVIDIA_API_KEY`` is set.
        """
        return bool(self._api_key)

    def supported_languages(self) -> list[str]:
        """Return supported language display names."""
        return list(NVIDIA_LANGUAGE_MAP.keys())

    def supported_formats(self) -> list[str]:
        """Return supported audio formats."""
        return [".wav", ".ogg", ".flac"]

    def requires_conversion_to_wav(self, audio_path: Path, model: str) -> bool:
        """Determine whether the audio file requires conversion to WAV format.

        Riva expects WAV PCM 16kHz Mono.
        """
        from services.audio.converter import AudioConverter
        return AudioConverter.needs_transcoding_to_mono_16k(audio_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_api_key(self) -> None:
        """Raise if the NVIDIA API key is not set."""
        if not self._api_key:
            raise APIKeyMissingError(
                "NVIDIA_API_KEY is not set. Please add it to your .env file."
            )

    def _create_auth(self, function_id: str) -> riva.client.Auth:
        """Build a Riva Auth object for the given function ID."""
        return riva.client.Auth(
            ssl_root_cert=None,
            use_ssl=True,
            uri=NVIDIA_GRPC_SERVER,
            metadata_args=[
                ("function-id", function_id),
                ("authorization", f"Bearer {self._api_key}"),
            ],
        )

    def _transcribe_streaming(
        self,
        asr_service: riva.client.ASRService,
        audio_path: Path,
        language_code: str,
    ) -> str:
        """Transcribe using streaming recognition (Parakeet CTC).

        Reads the WAV file, chunks it, and sends via streaming gRPC.
        """
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            audio_data = wf.readframes(wf.getnframes())

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language_code,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                audio_channel_count=num_channels,
                sample_rate_hertz=sample_rate,
            ),
            interim_results=False,
        )

        # Chunk audio into 100ms segments
        bytes_per_sample = 2 * num_channels  # 16-bit PCM
        chunk_samples = sample_rate // 10  # 100ms worth of samples
        chunk_bytes = chunk_samples * bytes_per_sample

        def audio_chunk_generator():
            offset = 0
            while offset < len(audio_data):
                yield audio_data[offset : offset + chunk_bytes]
                offset += chunk_bytes

        logger.debug("Streaming %d bytes in %d-byte chunks", len(audio_data), chunk_bytes)

        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_chunk_generator(),
            streaming_config=config,
        )

        transcript_parts: list[str] = []
        for response in responses:
            for result in response.results:
                if not result.is_final:
                    continue
                if result.alternatives:
                    transcript_parts.append(result.alternatives[0].transcript)

        return " ".join(transcript_parts)

    def _transcribe_offline(
        self,
        asr_service: riva.client.ASRService,
        audio_path: Path,
        language_code: str,
    ) -> str:
        """Transcribe using offline recognition (Whisper Large v3).

        Sends the entire audio file in a single request.
        """
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        config = riva.client.RecognitionConfig(
            language_code=language_code,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        riva.client.add_audio_file_specs_to_config(config, audio_data)

        logger.debug("Offline transcription — %d bytes", len(audio_data))

        response = asr_service.offline_recognize(audio_data, config)

        transcript_parts: list[str] = []
        for result in response.results:
            if result.alternatives:
                transcript_parts.append(result.alternatives[0].transcript)

        return " ".join(transcript_parts)
