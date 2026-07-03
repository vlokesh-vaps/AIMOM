"""AI Manager — selector, orchestrator, retry mechanism, and failover engine.

Coordinates LLM requests, validation, automatic JSON retries, and provider fallback.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import (
    AI_PROVIDER,
    AI_MAX_TOKENS,
    LLM_MAX_RETRIES,
    LLM_INITIAL_BACKOFF,
    LLM_BACKOFF_FACTOR,
    LLM_SAFETY_MARGIN,
    LLM_CHUNK_SIZE_TOKENS,
    LLM_THROTTLE_DELAY,
)
from ai.provider import (
    BaseAIProvider,
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderRateLimitError,
)
from ai.nvidia_provider import NvidiaAIProvider
from ai.groq_provider import GroqAIProvider
from ai.gemini_provider import GeminiAIProvider
from ai.ollama_provider import OllamaAIProvider
from ai.parser import parse_and_validate, JSONParsingError
from ai.schemas import MeetingSummary
from utils.logger import get_logger

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text block (approx 1.33 tokens per word or 4 chars per token)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(int(words * 1.33), int(chars / 4.0))


def chunk_transcript(transcript: str, max_chunk_tokens: int) -> List[str]:
    """Split the transcript into chunks that each fit within the max_chunk_tokens limit."""
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
                # Force add a single line if it alone exceeds the limit
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
    return max(minimum, min(value, maximum))


def get_model_limits(provider_name: str, model_name: str) -> tuple[int, int]:
    """Return (context_limit, tpm_limit) for a given provider and model name."""
    MODEL_LIMITS = {
        "nvidia/nemotron-3-ultra-550b-a55b": (4096, 1000000),
        "llama-3.3-70b-versatile": (128000, 12000),
        "llama-3.1-8b-instant": (128000, 30000),
        "gemini": (1048576, 1000000),
        "ollama": (8192, 1000000),
    }
    
    m_lower = model_name.lower()
    if "nemotron" in m_lower:
        return MODEL_LIMITS["nvidia/nemotron-3-ultra-550b-a55b"]
    elif "llama-3.3-70b" in m_lower or "llama3.3" in m_lower:
        return MODEL_LIMITS["llama-3.3-70b-versatile"]
    elif "llama-3.1-8b" in m_lower or "llama3.1" in m_lower:
        return MODEL_LIMITS["llama-3.1-8b-instant"]
    elif "gemini" in m_lower:
        return MODEL_LIMITS["gemini"]
    elif "ollama" in m_lower or "qwen" in m_lower:
        return (32768, 1000000)
        
    p_lower = provider_name.lower()
    if p_lower == "nvidia":
        return (4096, 1000000)
    elif p_lower == "groq":
        return (128000, 12000)
    elif p_lower == "gemini":
        return (1048576, 1000000)
    elif p_lower == "ollama":
        return (8192, 1000000)
        
    return (8192, 1000000)


class AIManager:
    """Registry and runner for AI Providers with built-in retry, failover, and summarization reliability layers."""

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

        primary_name = (provider_override or AI_PROVIDER or "nvidia").lower()
        if primary_name not in self._providers:
            available = list(self._providers.keys())
            if not available:
                return transcript
            primary_name = available[0]

        run_queue = [primary_name]
        for name in self._providers:
            if name != primary_name and self._providers[name].is_configured():
                run_queue.append(name)

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
    ) -> MeetingSummary:
        """Analyze meeting transcript to produce validated MeetingSummary JSON.

        Coordinates retries for JSON parsing errors and automatically executes
        failover to alternative providers on persistent errors.
        """
        primary_name = (provider_override or AI_PROVIDER or "nvidia").lower()
        if primary_name not in self._providers:
            available = list(self._providers.keys())
            if not available:
                raise AIProviderError("No AI providers registered in AIManager.")
            primary_name = available[0]
            logger.warning("Configured AI_PROVIDER '%s' not registered. Defaulting to '%s'", AI_PROVIDER, primary_name)

        run_queue = [primary_name]
        for name in self._providers:
            if name != primary_name and self._providers[name].is_configured():
                run_queue.append(name)

        logger.info("AI Analysis failover queue: %s", " -> ".join(run_queue))

        last_error = None

        for provider_name in run_queue:
            provider = self._providers[provider_name]
            logger.info("Attempting meeting analysis with provider: '%s'", provider_name)

            try:
                summary = self._execute_provider_with_retry(
                    provider=provider,
                    title=title,
                    date=date,
                    transcript=transcript,
                    speaker_transcript=speaker_transcript,
                    max_retries=max_retries,
                )
                logger.info("Successfully analyzed meeting transcript using provider '%s'", provider_name)
                return summary
            except Exception as exc:
                last_error = exc
                logger.error("Provider '%s' failed during execution: %s", provider_name, exc)
                logger.warning("Failing over to next available provider...")

        msg = f"All registered AI providers failed to analyze the transcript. Last error: {last_error}"
        logger.critical(msg)
        raise AIProviderError(msg) from last_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_default_providers(self) -> None:
        """Register the built-in AI providers."""
        self.register("groq", GroqAIProvider())
        self.register("nvidia", NvidiaAIProvider())
        self.register("gemini", GeminiAIProvider())
        self.register("ollama", OllamaAIProvider())

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
            except (AIProviderRateLimitError, AIProviderTimeoutError) as transient_exc:
                logger.warning(
                    "[%s] Transient exception: %s (attempt %d/%d). Backing off for %.2fs...",
                    provider.get_name(),
                    transient_exc,
                    attempt,
                    LLM_MAX_RETRIES,
                    backoff
                )
                time.sleep(backoff)
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
                    logger.warning(
                        "[%s] Detected transient API error: %s (attempt %d/%d). Backing off for %.2fs...",
                        provider.get_name(),
                        exc,
                        attempt,
                        LLM_MAX_RETRIES,
                        backoff
                    )
                    time.sleep(backoff)
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

    def _execute_provider_with_retry(
        self,
        provider: BaseAIProvider,
        title: str,
        date: str,
        transcript: str,
        speaker_transcript: Optional[str],
        max_retries: int,
    ) -> MeetingSummary:
        """Call LLM and validate Pydantic output, chunking if context limits are exceeded."""
        from config.settings import AI_MODEL
        model_name = provider.get_active_model(AI_MODEL)

        # Get limits
        context_limit, tpm_limit = get_model_limits(provider.get_name(), model_name)
        context_threshold = int(context_limit * LLM_SAFETY_MARGIN)
        tpm_threshold = int(tpm_limit * LLM_SAFETY_MARGIN)
        safety_threshold = min(context_threshold, tpm_threshold)

        # Build prompts for verification
        from ai.prompts import SYSTEM_PROMPT, format_user_prompt
        generated_at = datetime.now().isoformat()
        user_prompt_template = format_user_prompt(
            title=title,
            date=date,
            generated_at=generated_at,
            transcript=transcript,
            speaker_transcript=speaker_transcript,
        )

        system_tokens = estimate_tokens(SYSTEM_PROMPT)
        user_tokens = estimate_tokens(user_prompt_template)
        expected_output_tokens = min(AI_MAX_TOKENS, clamp_token_budget(safety_threshold // 3, 1024, 3072))
        total_estimated = system_tokens + user_tokens + expected_output_tokens

        logger.info(
            "[%s] Token estimation: Model=%s, ContextLimit=%d, TpmLimit=%d, SafetyThreshold=%d, EstimatedTotal=%d",
            provider.get_name(), model_name, context_limit, tpm_limit, safety_threshold, total_estimated
        )

        if total_estimated > safety_threshold:
            logger.warning(
                "[%s] Estimated tokens (%d) exceeds safety threshold (%d). Chunking transcript...",
                provider.get_name(), total_estimated, safety_threshold
            )
            input_text = speaker_transcript if speaker_transcript else transcript
            chunk_system = (
                "You are an assistant summarizing one section of a meeting transcript. "
                "Extract key topics, decisions, risks, questions, participants, and action items. "
                "For action items, owner means only the assignee/receiver, never the person who gave the task. "
                "If no assignee is named, mark it as information with no owner/date. "
                "Return concise plain text bullets only. Keep the summary under 700 words."
            )
            chunk_output_tokens = min(1024, expected_output_tokens)
            chunk_prompt_overhead = estimate_tokens(chunk_system) + estimate_tokens(f"Meeting Title: {title}\nDate: {date}\nChunk Transcript Content:\n")
            chunk_budget = safety_threshold - chunk_prompt_overhead - chunk_output_tokens
            chunk_budget = clamp_token_budget(chunk_budget, 500, LLM_CHUNK_SIZE_TOKENS)
            chunks = chunk_transcript(input_text, chunk_budget)
            logger.info("[%s] Split transcript into %d chunks for processing.", provider.get_name(), len(chunks))

            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                if i > 0 and LLM_THROTTLE_DELAY > 0:
                    logger.info("[%s] Throttling chunk requests: sleeping for %.2fs...", provider.get_name(), LLM_THROTTLE_DELAY)
                    time.sleep(LLM_THROTTLE_DELAY)

                logger.info("[%s] Summarizing chunk %d/%d (%d tokens)...", provider.get_name(), i + 1, len(chunks), estimate_tokens(chunk))
                chunk_user = f"Meeting Title: {title}\nDate: {date}\nChunk Transcript Content:\n{chunk}"
                
                chunk_text = self._execute_with_transient_retry(
                    provider,
                    chunk_system,
                    chunk_user,
                    max_tokens=chunk_output_tokens,
                )
                chunk_summaries.append(chunk_text)

            consolidated_text = "\n\n".join([
                f"--- Transcript Section {i+1} Summary ---\n{summary_text}"
                for i, summary_text in enumerate(chunk_summaries)
            ])
            logger.info("[%s] Successfully summarized all chunks. Consolidating intermediate summaries...", provider.get_name())

            final_user_prompt = format_user_prompt(
                title=title,
                date=date,
                generated_at=generated_at,
                transcript=consolidated_text,
                speaker_transcript=None,
            )
            final_tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(final_user_prompt) + expected_output_tokens
            if final_tokens > safety_threshold:
                logger.warning(
                    "[%s] Consolidated summaries still exceed threshold (%d > %d). Re-summarizing consolidation input...",
                    provider.get_name(), final_tokens, safety_threshold
                )
                reducer_system = (
                    "Compress these meeting section summaries into one concise master summary. "
                    "Preserve action items, assignees/receivers, dates, decisions, risks, questions, participants, and timeline facts. "
                    "Keep informational rows separate when no assignee is named. "
                    "Return plain text bullets only."
                )
                reducer_prompt = consolidated_text
                consolidated_text = self._execute_with_transient_retry(
                    provider,
                    reducer_system,
                    reducer_prompt,
                    max_tokens=min(1536, expected_output_tokens),
                )
                final_user_prompt = format_user_prompt(
                    title=title,
                    date=date,
                    generated_at=generated_at,
                    transcript=consolidated_text,
                    speaker_transcript=None,
                )
        else:
            final_user_prompt = user_prompt_template

        start_time = time.time()
        retry_count = 0

        while retry_count <= max_retries:
            try:
                raw_response = self._execute_with_transient_retry(
                    provider,
                    SYSTEM_PROMPT,
                    final_user_prompt,
                    max_tokens=expected_output_tokens,
                )
                summary = parse_and_validate(raw_response)
                
                duration = time.time() - start_time
                logger.info(
                    "[%s] AI Analysis completed and validated. Retries=%d, Latency=%.2fs",
                    provider.get_name(),
                    retry_count,
                    duration,
                )
                return summary

            except JSONParsingError as parsing_err:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error("[%s] Persistent JSON validation failure after %d attempts.", provider.get_name(), max_retries)
                    raise AIProviderError(
                        f"Failed to generate structured JSON matching the schema: {parsing_err}"
                    ) from parsing_err
                
                logger.warning(
                    "[%s] JSON parsing failed (attempt %d/%d). Retrying... Error: %s",
                    provider.get_name(),
                    retry_count,
                    max_retries,
                    parsing_err,
                )
                time.sleep(1.0 * retry_count)
