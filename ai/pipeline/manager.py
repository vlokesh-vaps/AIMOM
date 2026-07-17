"""AI Manager — Single-Extraction pipeline orchestrator.

All LLM calls route through ProviderManager for automatic failover,
retry with exponential backoff, cooldown management, and health-based
recovery.

Pipeline stages:
  1. Transcript Cleaning      — pure Python
  2. Smart Chunking           — pure Python
  3. Parallel Extraction      — async workers with semaphore
  4. Checkpoint & Resume      — SHA-256 content-hash based
  5. Incremental Merge        — pure Python dedup + normalization
  6. Final Synthesis          — single LLM call on merged JSON
  7. Validation (optional)    — never blocks
"""

from __future__ import annotations

from typing import Dict, List, Optional

from config.settings import (
    NVIDIA_MOM_MODEL,
    GROQ_FALLBACK_MODEL,
)
from ai.providers.base import (
    BaseAIProvider,
    AIProviderError,
)
from ai.providers.nvidia import NvidiaAIProvider
from ai.providers.groq import GroqAIProvider
from ai.providers.gemini import GeminiAIProvider
from ai.providers.provider_manager import ProviderManager
from ai.models.meeting import MeetingSummary
from ai.pipeline.six_agent_pipeline import SingleExtractionPipeline

from utils.logger import get_logger

logger = get_logger(__name__)


class AIManager:
    """Registry and runner for AI Providers with the single-extraction pipeline.

    Creates a centralized ProviderManager at init time and passes it
    to the SingleExtractionPipeline. Individual agents never interact with
    providers directly.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, BaseAIProvider] = {}
        self._register_default_providers()
        self._provider_manager = ProviderManager(self._providers)

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
        """Return names of registered providers that are fully configured."""
        return [name for name, prov in self._providers.items() if prov.is_configured()]

    # ------------------------------------------------------------------
    # Main orchestration entry points
    # ------------------------------------------------------------------

    def translate_to_english(
        self,
        transcript: str,
        source_language: str,
        provider_override: str | None = None,
    ) -> str:
        """Translate the transcript to English.

        Uses the centralized ProviderManager for failover.
        """
        if not source_language or source_language.lower() in ("english", "auto"):
            return transcript

        logger.info("Translating transcript from '%s' to English...", source_language)

        system_prompt = (
            "You are a professional, accurate corporate translator. "
            f"Translate the following meeting transcript text from {source_language} to English. "
            "Preserve all speaker labels (e.g. 'Speaker 1:', 'Speaker 2:') and time tags exactly. "
            "Output ONLY the translated English text, with no preamble, intros, explanations, or notes."
        )

        try:
            translated = self._provider_manager.execute(
                system_prompt=system_prompt,
                user_prompt=transcript,
                primary_model=NVIDIA_MOM_MODEL,
                fallback_model=GROQ_FALLBACK_MODEL,
                agent_name="Translation",
            )
            if translated and translated.strip():
                logger.info("Translation successful.")
                return translated.strip()
        except AIProviderError as exc:
            logger.error("Translation failed across all providers: %s", exc)

        logger.warning("Translation failed. Using original transcript.")
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
        """Analyze meeting transcript using the single-extraction pipeline.

        Stages:
          1. Transcript Cleaning      — pure Python
          2. Smart Chunking           — pure Python
          3. Parallel Extraction      — async, one LLM call per chunk
          4. Checkpoint & Resume      — hash-based fault tolerance
          5. Incremental Merge        — pure Python
          6. Final Synthesis          — single LLM call
          7. Validation (optional)    — never blocks

        All LLM requests route through ProviderManager.
        """
        configured = self.get_configured_providers()
        if not configured:
            raise AIProviderError(
                "No AI providers are configured. Please set NVIDIA_API_KEY and/or GROQ_API_KEY."
            )

        logger.info("=" * 70)
        logger.info("SINGLE EXTRACTION PIPELINE — Starting meeting analysis")
        logger.info("Configured providers: %s", configured)
        logger.info("=" * 70)

        # Use speaker_transcript if available
        analysis_transcript = speaker_transcript if speaker_transcript else transcript

        pipeline = SingleExtractionPipeline(provider_manager=self._provider_manager)
        return pipeline.run(
            title=title,
            date=date,
            transcript=analysis_transcript,
            attendees=attendees,
            agenda=agenda,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_default_providers(self) -> None:
        """Register the built-in AI providers."""
        self.register("groq", GroqAIProvider())
        self.register("nvidia", NvidiaAIProvider())
        self.register("gemini", GeminiAIProvider())
