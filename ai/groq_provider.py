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
from ai.provider import (
    BaseAIProvider,
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderRateLimitError,
)
from ai.prompts import SYSTEM_PROMPT, format_user_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


class GroqAIProvider(BaseAIProvider):
    """Groq Cloud Provider utilizing standard chat completions endpoint."""

    def __init__(self) -> None:
        self._api_key: str = GROQ_API_KEY
        self._endpoint: str = "https://api.groq.com/openai/v1/chat/completions"

    def get_name(self) -> str:
        return "groq"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        if not self.is_configured():
            raise AIProviderError("GROQ_API_KEY is not set.")

        model = AI_MODEL
        if not model or "/" in model or "nvidia" in model or "deepseek" in model or "gemini" in model:
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
            completion = result["choices"][0]["message"]["content"]
            
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
        model = global_model
        if not model or "/" in model or "nvidia" in model or "deepseek" in model or "gemini" in model:
            return "llama-3.3-70b-versatile"
        return model
