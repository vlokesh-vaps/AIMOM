"""Abstract base class for AI Providers."""

from abc import ABC, abstractmethod


class AIProviderError(Exception):
    """Base exception for AI provider errors."""


class AIProviderTimeoutError(AIProviderError):
    """Raised when an AI provider request times out."""


class AIProviderRateLimitError(AIProviderError):
    """Raised when an AI provider rate limit is encountered."""


class BaseAIProvider(ABC):
    """Abstract interface for meeting intelligence AI providers."""

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        """Sends custom system and user prompts to the LLM and returns the raw string response.

        Args:
            system_prompt: System prompt instructions.
            user_prompt: User prompt content.
            max_tokens: Optional provider-specific completion token cap.

        Returns:
            The raw text string response.

        Raises:
            AIProviderError: On generation failure.
        """
        pass

    @abstractmethod
    def generate_summary(
        self,
        title: str,
        date: str,
        transcript: str,
        speaker_transcript: str | None = None,
    ) -> str:
        """Sends the transcript metadata and text to the LLM and returns the raw string response.

        Args:
            title: Meeting title.
            date: Meeting date.
            transcript: Full raw text transcript.
            speaker_transcript: Optional structured speaker transcript.

        Returns:
            The raw string response containing structured JSON.

        Raises:
            AIProviderError: If execution fails.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique registered display name of this provider."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Checks if the provider has all necessary API keys or configurations setup."""
        pass

    @abstractmethod
    def get_active_model(self, global_model: str) -> str:
        """Resolve the actual model identifier that will be used by this provider."""
        pass
