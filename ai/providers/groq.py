"""Groq Cloud LLM provider using OpenAI-compatible HTTP interface."""

from datetime import datetime
import json
import requests

from config.settings import (
    GROQ_API_KEY,
    AI_MODEL,
    AI_TEMPERATURE,
    AI_MAX_TOKENS,
    AI_TOP_P,
    AI_TIMEOUT,
)
from ai.providers.base import (
    BaseAIProvider,
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderRateLimitError,
    AIProviderTruncatedResponseError,
)
from ai.prompting.templates import SYSTEM_PROMPT, format_user_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


# Models hosted on Groq that are valid despite having slashes in the name.
_GROQ_HOSTED_MODELS = {
    "openai/gpt-oss-20b",
}


class GroqAIProvider(BaseAIProvider):
    """Groq Cloud Provider utilizing standard chat completions endpoint."""

    def __init__(self, model_override: str | None = None) -> None:
        self._api_key: str = GROQ_API_KEY
        self._endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
        self._model_override: str | None = model_override

    def get_name(self) -> str:
        return "groq"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        if not self.is_configured():
            raise AIProviderError("GROQ_API_KEY is not set.")

        # Use explicit override if set (e.g. for chunk extraction with gpt-oss-20b)
        if self._model_override:
            model = self._model_override
        else:
            model = AI_MODEL
            if not model or (("/" in model and model not in _GROQ_HOSTED_MODELS) or "nvidia" in model or "deepseek" in model or "gemini" in model):
                model = "llama-3.3-70b-versatile"

        effective_max_tokens = max_tokens or AI_MAX_TOKENS
        logger.info("Groq LLM query: model=%s, temp=%.2f, max_tokens=%d", model, AI_TEMPERATURE, effective_max_tokens)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": AI_TEMPERATURE,
            "max_tokens": effective_max_tokens,
            "top_p": AI_TOP_P,
        }

        try:
            response = requests.post(
                self._endpoint,
                headers=headers,
                json=payload,
                timeout=AI_TIMEOUT,
            )
        except requests.Timeout as err:
            logger.error("Groq LLM query timed out.")
            raise AIProviderTimeoutError(f"Groq API query timed out after {AI_TIMEOUT}s: {err}") from err
        except requests.RequestException as err:
            logger.error("Groq API network error: %s", err)
            raise AIProviderError(f"Groq API request failed: {err}") from err

        if response.status_code == 429:
            logger.warning("Groq API returned 429 Rate Limit.")
            raise AIProviderRateLimitError("Groq API rate limit reached.")
        elif response.status_code != 200:
            logger.error("Groq API returned error status: %d. Body: %s", response.status_code, response.text)
            raise AIProviderError(f"Groq API error {response.status_code}: {response.text}")

        try:
            result = response.json()
            choice = result["choices"][0]
            
            # Check whether the model stopped because it reached the output token limit
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                logger.error("Groq model output truncated: hit the output token limit (max_tokens).")
                raise AIProviderTruncatedResponseError("Groq response was truncated (reached output token limit).")

            completion = choice["message"]["content"]
            
            usage = result.get("usage", {})
            logger.info("Groq success: prompt_tokens=%d, completion_tokens=%d", 
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return completion
        except (KeyError, ValueError, json.JSONDecodeError) as err:
            logger.exception("Failed to parse Groq response format")
            raise AIProviderError(f"Invalid response structure from Groq API: {err}") from err

    def generate_summary(
        self,
        title: str,
        date: str,
        transcript: str,
        speaker_transcript: str | None = None,
    ) -> str:
        generated_at = datetime.now().isoformat()
        user_prompt = format_user_prompt(
            title=title,
            date=date,
            generated_at=generated_at,
            transcript=transcript,
            speaker_transcript=speaker_transcript,
        )
        return self.generate_text(SYSTEM_PROMPT, user_prompt)

    def get_active_model(self, global_model: str) -> str:
        if self._model_override:
            return self._model_override
        model = global_model
        if not model or (("/" in model and model not in _GROQ_HOSTED_MODELS) or "nvidia" in model or "deepseek" in model or "gemini" in model):
            return "llama-3.3-70b-versatile"
        return model
