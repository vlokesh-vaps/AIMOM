"""AI Manager — 6-stage pipeline orchestrator with failover and retry.

Pipeline stages:
  1. Transcript Cleaner  (pure Python)
  2. Chunking Engine      (pure Python)
  3. Chunk Extractor      (Groq + openai/gpt-oss-20b)
  4. Merge Engine         (pure Python)
  5. Validation Layer     (pure Python)
  6. Final result          (validated summary)
"""

import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import (
    AI_MODEL,
    AI_PROVIDER,
    LLM_MAX_RETRIES,
    LLM_INITIAL_BACKOFF,
    LLM_BACKOFF_FACTOR,
    LLM_SAFETY_MARGIN,
    LLM_CHUNK_SIZE_TOKENS,
    CHUNK_EXTRACTOR_MODEL,
    GROQ_RPM_LIMIT,
    GROQ_TPM_LIMIT,
    NVIDIA_MOM_MODEL,
    NVIDIA_REQUEST_THROTTLE_SECONDS,
)
from ai.providers.base import (
    BaseAIProvider,
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderRateLimitError,
    AIProviderTruncatedResponseError,
)
from ai.providers.nvidia import NvidiaAIProvider
from ai.providers.groq import GroqAIProvider
from ai.providers.gemini import GeminiAIProvider
from ai.models.meeting import MeetingSummary
from ai.utils.checkpoints import ChunkCheckpointStore
from ai.utils.rate_limiter import ProviderRequestScheduler
from ai.utils.token_utils import estimate_tokens, clamp

# Pipeline stage imports
from ai.stages.transcript_cleaner import TranscriptCleaner
from ai.stages.chunking_engine import ChunkingEngine
from ai.stages.chunk_extractor import ChunkExtractor
from ai.stages.merge_engine import MergeEngine
from ai.validators.validation_layer import ValidationLayer
from ai.pipeline.six_agent_pipeline import SixAgentPipeline

from utils.logger import get_logger

logger = get_logger(__name__)


def chunk_transcript(transcript: str, max_chunk_tokens: int) -> List[str]:
    """Split the transcript into chunks that each fit within the max_chunk_tokens limit.

    NOTE: This is the LEGACY chunker. The new pipeline uses ChunkingEngine (Stage 2).
    Kept for backwards compatibility with any external callers.
    """
    lines = transcript.split("\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)
        if line_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            words = line.split()
            max_words = max(1, int(max_chunk_tokens / 1.33 * 0.9))
            word_chunk = []
            for word in words:
                if len(word_chunk) >= max_words:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                else:
                    word_chunk.append(word)

            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue

        if current_tokens + line_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                chunks.append(line)
                current_chunk = []
                current_tokens = 0
        else:
            current_chunk.append(line)
            current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def clamp_token_budget(value: int, minimum: int, maximum: int) -> int:
    """Clamp token budget while keeping invalid/negative calculations usable."""
    return clamp(value, minimum, maximum)


def get_model_limits(provider_name: str, model_name: str) -> tuple[int, int]:
    """Return (context_limit, tpm_limit) for a given provider and model name."""
    MODEL_LIMITS = {
        "nvidia/nemotron-3-ultra-550b-a55b": (4096, 1000000),
        "llama-3.3-70b-versatile": (128000, 12000),
        "llama-3.1-8b-instant": (128000, 30000),
        "openai/gpt-oss-20b": (128000, 12000),
        "openai/gpt-oss-120b": (128000, 12000),
        "gemini": (1048576, 1000000),
    }

    m_lower = model_name.lower()
    if "nemotron" in m_lower:
        return MODEL_LIMITS["nvidia/nemotron-3-ultra-550b-a55b"]
    elif "gpt-oss-20b" in m_lower:
        return MODEL_LIMITS["openai/gpt-oss-20b"]
    elif "gpt-oss-120b" in m_lower:
        return MODEL_LIMITS["openai/gpt-oss-120b"]
    elif "llama-3.3-70b" in m_lower or "llama3.3" in m_lower:
        return MODEL_LIMITS["llama-3.3-70b-versatile"]
    elif "llama-3.1-8b" in m_lower or "llama3.1" in m_lower:
        return MODEL_LIMITS["llama-3.1-8b-instant"]
    elif "gemini" in m_lower:
        return MODEL_LIMITS["gemini"]
    elif "qwen" in m_lower:
        return (32768, 1000000)

    p_lower = provider_name.lower()
    if p_lower == "nvidia":
        return (4096, 1000000)
    elif p_lower == "groq":
        return (128000, 12000)
    elif p_lower == "gemini":
        return (1048576, 1000000)
    return (8192, 1000000)


class AIManager:
    """Registry and runner for AI Providers with the 6-stage pipeline."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseAIProvider] = {}
        self._register_default_providers()

    def register(self, name: str, provider: BaseAIProvider) -> None:
        """Register an AI provider under a code name (e.g. 'nvidia')."""
        self._providers[name.lower()] = provider
        logger.debug("Registered AI provider: %s", name)

    def get_provider(self, name: str) -> BaseAIProvider:
        """Retrieve a registered provider by name."""
        provider = self._providers.get(name.lower())
        if not provider:
            raise ValueError(f"AI Provider '{name}' is not registered.")
        return provider

    def get_configured_providers(self) -> List[str]:
        """Return names of registered providers that are fully configured (e.g., have API keys)."""
        return [name for name, prov in self._providers.items() if prov.is_configured()]

    def _build_run_queue(self, provider_override: str | None = None) -> List[str]:
        """Build the failover queue of configured providers in order of preference.

        Preferred order: Groq -> NVIDIA -> Gemini -> others.
        If provider_override is specified, it is placed at the front.
        """
        preferred_order = ["groq", "nvidia", "gemini"]

        available_configured = [
            name for name, provider in self._providers.items()
            if provider.is_configured()
        ]

        def get_sort_key(name: str) -> int:
            try:
                return preferred_order.index(name.lower())
            except ValueError:
                return len(preferred_order)

        sorted_providers = sorted(available_configured, key=get_sort_key)

        if provider_override:
            override_name = provider_override.lower()
            if override_name in sorted_providers:
                sorted_providers.remove(override_name)
                sorted_providers.insert(0, override_name)
            elif override_name in self._providers and self._providers[override_name].is_configured():
                sorted_providers.insert(0, override_name)

        return sorted_providers

    # ------------------------------------------------------------------
    # Orchestration & Failover
    # ------------------------------------------------------------------

    def translate_to_english(
        self,
        transcript: str,
        source_language: str,
        provider_override: str | None = None,
    ) -> str:
        """Translate the transcript to English if the source language is not English.

        Uses the same failover provider queue to execute translation.
        """
        if not source_language or source_language.lower() in ("english", "auto"):
            return transcript

        logger.info("Translating transcript from '%s' to English...", source_language)

        run_queue = self._build_run_queue(provider_override)
        if not run_queue:
            logger.warning("No configured AI providers available for translation. Using original transcript.")
            return transcript

        logger.info("Translation failover queue: %s", " -> ".join(run_queue))

        system_prompt = (
            "You are a professional, accurate corporate translator. "
            f"Translate the following meeting transcript text from {source_language} to English. "
            "Preserve all speaker labels (e.g. 'Speaker 1:', 'Speaker 2:') and time tags exactly. "
            "Output ONLY the translated English text, with no preamble, intros, explanations, or notes."
        )

        for provider_name in run_queue:
            provider = self._providers[provider_name]
            logger.info("Attempting translation with provider: '%s'", provider_name)
            try:
                translated = self._execute_with_transient_retry(
                    provider=provider,
                    system_prompt=system_prompt,
                    user_prompt=transcript,
                )
                if translated and translated.strip():
                    logger.info("Translation successful with provider '%s'", provider_name)
                    return translated.strip()
            except Exception as e:
                logger.error("Translation failed on provider '%s': %s", provider_name, e)
                logger.warning("Failing over to next provider for translation...")

        logger.error("All AI providers failed to translate the transcript. Using original transcript.")
        return transcript

    def analyze_meeting(
        self,
        title: str,
        date: str,
        transcript: str,
        speaker_transcript: Optional[str] = None,
        max_retries: int = 3,
        provider_override: str | None = None,
        attendees: Optional[str] = None,
        agenda: Optional[str] = None,
    ) -> MeetingSummary:
        """Analyze meeting transcript using the 6-stage pipeline.

        Stages:
          1. Transcript Cleaner   — normalize and reduce noise
          2. Chunking Engine      — split intelligently preserving context
          3. Chunk Extractor      — structured JSON via Groq (openai/gpt-oss-20b)
          4. Merge Engine         — combine chunk outputs without loss
          5. Validation Layer     — verify completeness and consistency
          6. Final result         — return the validated summary
        """
        selected_name = (provider_override or AI_PROVIDER or "groq").lower()
        selected_provider = self._providers.get(selected_name)
        if selected_provider and selected_provider.is_configured():
            if selected_name == "nvidia":
                groq = self._providers.get("groq")
                if groq and groq.is_configured():
                    return SixAgentPipeline(
                        providers=self._providers,
                        retry_fn=self._execute_with_transient_retry,
                    ).run(
                        title=title,
                        date=date,
                        transcript=transcript,
                        attendees=attendees,
                        agenda=agenda,
                    )
                selected_provider = NvidiaAIProvider(model_override=NVIDIA_MOM_MODEL)
            elif selected_name == "groq":
                selected_provider = GroqAIProvider(model_override=CHUNK_EXTRACTOR_MODEL)
            logger.info("Stage 3 extraction provider: %s + %s", selected_name, selected_provider.get_active_model(AI_MODEL))
            return self._execute_pipeline(
                provider=selected_provider,
                title=title,
                date=date,
                transcript=transcript,
                speaker_transcript=speaker_transcript,
                attendees=attendees,
                agenda=agenda,
                chunk_checkpoint={},
                failed_providers=set(),
            )

        groq = self._providers.get("groq")
        if not groq or not groq.is_configured():
            raise AIProviderError(f"AI provider '{selected_name}' is not configured, and Groq fallback is unavailable.")

        logger.info("=" * 70)
        logger.info("6-STAGE PIPELINE — Starting meeting analysis")
        logger.info("=" * 70)
        logger.info("Stage 3 extraction provider: Groq + %s", CHUNK_EXTRACTOR_MODEL)
        return self._execute_pipeline(
            provider=GroqAIProvider(model_override=CHUNK_EXTRACTOR_MODEL),
            title=title,
            date=date,
            transcript=transcript,
            speaker_transcript=speaker_transcript,
            attendees=attendees,
            agenda=agenda,
            chunk_checkpoint={},
            failed_providers=set(),
        )

    # ------------------------------------------------------------------
    # 6-Stage Pipeline
    # ------------------------------------------------------------------

    def _execute_pipeline(
        self,
        provider: BaseAIProvider,
        title: str,
        date: str,
        transcript: str,
        speaker_transcript: Optional[str],
        attendees: Optional[str] = None,
        agenda: Optional[str] = None,
        chunk_checkpoint: Optional[dict] = None,
        failed_providers: Optional[set] = None,
    ) -> MeetingSummary:
        """Execute the full 6-stage pipeline.

        Args:
            provider: The primary provider for Stage 3 (chunk extraction).
            title: Meeting title.
            date: Meeting date.
            transcript: Raw transcript text.
            speaker_transcript: Optional structured speaker transcript.
            attendees: Optional comma-separated attendee names.
            chunk_checkpoint: Shared checkpoint dict persisting across failover.
            failed_providers: Set of provider names that have already failed.

        Returns:
            A validated and polished MeetingSummary.
        """
        if chunk_checkpoint is None:
            chunk_checkpoint = {}
        if failed_providers is None:
            failed_providers = set()

        pipeline_start = time.time()
        input_text = speaker_transcript if speaker_transcript else transcript

        # ── STAGE 1: Transcript Cleaner ──────────────────────────────────
        logger.info("─── STAGE 1: Transcript Cleaner ───")
        stage1_start = time.time()
        cleaner = TranscriptCleaner()
        cleaned_text = cleaner.clean(input_text)
        logger.info("Stage 1 complete. (%.2fs)", time.time() - stage1_start)

        # ── STAGE 2: Chunking Engine ─────────────────────────────────────
        logger.info("─── STAGE 2: Chunking Engine ───")
        stage2_start = time.time()

        chunk_budget = clamp_token_budget(LLM_CHUNK_SIZE_TOKENS, 700, 900)
        chunking_engine = ChunkingEngine(overlap_lines=3)
        chunks = chunking_engine.chunk(cleaned_text, chunk_budget)
        logger.info("Stage 2 complete: %d chunks. (%.2fs)", len(chunks), time.time() - stage2_start)

        # ── STAGE 3: Chunk Extractor (Groq / openai/gpt-oss-20b) ────────
        chunk_extractor_provider = provider
        logger.info(
            "─── STAGE 3: Chunk Extractor (%s + %s) ───",
            chunk_extractor_provider.get_name(),
            chunk_extractor_provider.get_active_model(AI_MODEL),
        )
        stage3_start = time.time()

        checkpoint_store = ChunkCheckpointStore(
            run_id=f"{title}_{date}_{hashlib.sha256(cleaned_text.encode('utf-8')).hexdigest()[:16]}",
        )
        scheduler = ProviderRequestScheduler(
            provider_name=chunk_extractor_provider.get_name(),
            rpm_limit=GROQ_RPM_LIMIT if chunk_extractor_provider.get_name() == "groq" else 60,
            tpm_limit=GROQ_TPM_LIMIT if chunk_extractor_provider.get_name() == "groq" else 1000000,
            max_concurrent=1,
            safety_margin=LLM_SAFETY_MARGIN,
        )

        extractor = ChunkExtractor(
            provider=chunk_extractor_provider,
            retry_fn=self._execute_with_transient_retry,
            scheduler=scheduler,
            checkpoint_store=checkpoint_store,
            max_output_tokens=4096,
            throttle_seconds=(NVIDIA_REQUEST_THROTTLE_SECONDS if chunk_extractor_provider.get_name() == "nvidia" else None),
        )
        extractions = extractor.extract_all(
            chunks=chunks,
            title=title,
            date=date,
            chunk_checkpoint=chunk_checkpoint,
            agenda=agenda,
            attendees=attendees,
        )
        logger.info("Stage 3 complete: %d extractions. (%.2fs)", len(extractions), time.time() - stage3_start)

        # ── STAGE 4: Merge Engine ────────────────────────────────────────
        logger.info("─── STAGE 4: Merge Engine ───")
        stage4_start = time.time()
        merge_engine = MergeEngine(
            provider=chunk_extractor_provider,
            retry_fn=self._execute_with_transient_retry,
        )
        merged_summary = merge_engine.merge(
            extractions=extractions,
            title=title,
            date=date,
            attendees=attendees,
        )
        logger.info("Stage 4 complete. (%.2fs)", time.time() - stage4_start)

        # ── STAGE 5: Validation Layer ────────────────────────────────────
        logger.info("─── STAGE 5: Validation Layer ───")
        stage5_start = time.time()
        validator = ValidationLayer()
        validation_result = validator.validate(merged_summary)
        logger.info(
            "Stage 5 complete: valid=%s, %d warnings. (%.2fs)",
            validation_result.is_valid,
            len(validation_result.warnings),
            time.time() - stage5_start,
        )

        # ── STAGE 6: Final result ────────────────────────────────────────
        final_summary = validation_result.summary
        logger.info("Stage 6 complete: using validated summary.")

        # ── Pipeline complete ────────────────────────────────────────────
        total_time = time.time() - pipeline_start
        logger.info("=" * 70)
        logger.info(
            "6-STAGE PIPELINE COMPLETE — %d discussion points, %d action items. Total: %.2fs",
            len(final_summary.discussion_points),
            len(final_summary.action_items),
            total_time,
        )
        logger.info("=" * 70)

        return final_summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_default_providers(self) -> None:
        """Register the built-in AI providers."""
        self.register("groq", GroqAIProvider())
        self.register("nvidia", NvidiaAIProvider())
        self.register("gemini", GeminiAIProvider())

    def _get_chunk_extractor_provider(self, failed_providers: set) -> BaseAIProvider:
        """Get the provider for Stage 3 chunk extraction.

        Creates a Groq provider pinned to openai/gpt-oss-20b.
        """
        groq = self._providers.get("groq")
        if groq and groq.is_configured():
            return GroqAIProvider(model_override=CHUNK_EXTRACTOR_MODEL)
        raise AIProviderError("Groq is required for chunk extraction (Stage 3).")

    def _execute_with_transient_retry(
        self,
        provider: BaseAIProvider,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Call a provider's text generation, retrying on transient errors with backoff."""
        attempt = 0
        backoff = LLM_INITIAL_BACKOFF

        while attempt <= LLM_MAX_RETRIES:
            try:
                return provider.generate_text(system_prompt, user_prompt, max_tokens=max_tokens)
            except AIProviderTruncatedResponseError as truncated_exc:
                if attempt >= LLM_MAX_RETRIES:
                    logger.error(
                        "[%s] Truncated response after %d retries.",
                        provider.get_name(),
                        LLM_MAX_RETRIES,
                    )
                    raise
                retry_tokens = None
                if max_tokens:
                    retry_tokens = min(max(int(max_tokens * 1.5), max_tokens + 512), 16384)
                logger.warning(
                    "[%s] Truncated response (attempt %d/%d). Retrying with max_tokens=%s.",
                    provider.get_name(),
                    attempt + 1,
                    LLM_MAX_RETRIES,
                    retry_tokens or "provider default",
                )
                if retry_tokens:
                    max_tokens = retry_tokens
                attempt += 1
                time.sleep(backoff)
                backoff *= LLM_BACKOFF_FACTOR
            except (AIProviderRateLimitError, AIProviderTimeoutError) as transient_exc:
                is_rate_limit = isinstance(transient_exc, AIProviderRateLimitError)
                sleep_time = max(backoff, 70.0) if is_rate_limit else backoff
                logger.warning(
                    "[%s] Transient exception: %s (attempt %d/%d). Backing off for %.2fs...",
                    provider.get_name(),
                    transient_exc,
                    attempt,
                    LLM_MAX_RETRIES,
                    sleep_time
                )
                time.sleep(sleep_time)
                attempt += 1
                backoff *= LLM_BACKOFF_FACTOR
            except AIProviderError as exc:
                exc_msg = str(exc).lower()
                is_payload_too_large = any(
                    indicator in exc_msg
                    for indicator in ("request too large", "please reduce your message size", "maximum context length", "context length")
                )
                is_transient = any(
                    indicator in exc_msg
                    for indicator in ("503", "502", "500", "504", "service unavailable", "bad gateway", "timeout", "gateway timeout", "connection", "rate_limit", "resourceexhausted")
                ) and not is_payload_too_large

                if is_transient and attempt < LLM_MAX_RETRIES:
                    is_rate_limit = any(ind in exc_msg for ind in ("rate_limit", "resourceexhausted"))
                    sleep_time = max(backoff, 70.0) if is_rate_limit else backoff
                    logger.warning(
                        "[%s] Detected transient API error: %s (attempt %d/%d). Backing off for %.2fs...",
                        provider.get_name(),
                        exc,
                        attempt,
                        LLM_MAX_RETRIES,
                        sleep_time
                    )
                    time.sleep(sleep_time)
                    attempt += 1
                    backoff *= LLM_BACKOFF_FACTOR
                else:
                    logger.error(
                        "[%s] Unrecoverable provider error or retries exhausted: %s. Propagating.",
                        provider.get_name(),
                        exc
                    )
                    raise
            except Exception as other_exc:
                if attempt < LLM_MAX_RETRIES:
                    logger.warning(
                        "[%s] Unexpected exception: %s (attempt %d/%d). Backing off for %.2fs...",
                        provider.get_name(),
                        other_exc,
                        attempt,
                        LLM_MAX_RETRIES,
                        backoff
                    )
                    time.sleep(backoff)
                    attempt += 1
                    backoff *= LLM_BACKOFF_FACTOR
                else:
                    raise

        raise AIProviderError(f"[{provider.get_name()}] Retries exhausted for transient errors.")
