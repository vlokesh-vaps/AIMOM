"""Google Gemini LLM provider using the standard developer REST API.

Supports structured JSON output formatting natively.
"""

from datetime import datetime
import json
import requests

from config.settings import (
    GEMINI_API_KEY,
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


class GeminiAIProvider(BaseAIProvider):
    """Google Gemini ASR/LLM Provider using REST generateContent API."""

    def __init__(self) -> None:
        self._api_key: str = GEMINI_API_KEY

    def get_name(self) -> str:
        return "gemini"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        if not self.is_configured():
            raise AIProviderError("GEMINI_API_KEY is not set.")

        model = AI_MODEL
        if not model or "gemini" not in model.lower():
            model = "gemini-1.5-flash"

        effective_max_tokens = max_tokens or AI_MAX_TOKENS
        logger.info("Gemini LLM query: model=%s, temp=%.2f, max_tokens=%d", model, AI_TEMPERATURE, effective_max_tokens)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self._api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": AI_TEMPERATURE,
                "maxOutputTokens": effective_max_tokens,
                "topP": AI_TOP_P,
            }
        }

        # Force JSON response mode if parsing summary/Pydantic schemas
        if "matching the following schema structure exactly" in system_prompt or "{" in system_prompt:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        headers = {
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=AI_TIMEOUT,
            )
        except requests.Timeout as err:
            logger.error("Gemini LLM query timed out.")
            raise AIProviderTimeoutError(f"Gemini API query timed out after {AI_TIMEOUT}s: {err}") from err
        except requests.RequestException as err:
            logger.error("Gemini API network error: %s", err)
            raise AIProviderError(f"Gemini API request failed: {err}") from err

        if response.status_code == 429:
            logger.warning("Gemini API returned 429 Rate Limit.")
            raise AIProviderRateLimitError("Gemini API rate limit reached.")
        elif response.status_code != 200:
            logger.error("Gemini API returned error status: %d. Body: %s", response.status_code, response.text)
            raise AIProviderError(f"Gemini API error {response.status_code}: {response.text}")

        try:
            result = response.json()
            completion = result["candidates"][0]["content"]["parts"][0]["text"]
            logger.info("Gemini text analysis completed successfully")
            return completion
        except (KeyError, ValueError, IndexError, TypeError) as err:
            logger.exception("Failed to parse Gemini response format")
            raise AIProviderError(f"Invalid response structure from Gemini API: {err}") from err

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
        return global_model if global_model and "gemini" in global_model.lower() else "gemini-1.5-flash"
