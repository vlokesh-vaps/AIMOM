"""NVIDIA NIM ASR/LLM provider using OpenAI-compatible HTTP interface."""

from datetime import datetime
import json
import requests

from config.settings import (
    NVIDIA_API_KEY,
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
)
from ai.prompting.templates import SYSTEM_PROMPT, format_user_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


class NvidiaAIProvider(BaseAIProvider):
    """NVIDIA ASR/LLM Provider using the standard integrate.api.nvidia.com endpoint."""

    def __init__(self) -> None:
        self._api_key: str = NVIDIA_API_KEY
        self._endpoint: str = "https://integrate.api.nvidia.com/v1/chat/completions"

    def get_name(self) -> str:
        return "nvidia"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        if not self.is_configured():
            raise AIProviderError("NVIDIA_API_KEY is not set.")

        model = AI_MODEL if AI_MODEL and "/" in AI_MODEL else "nvidia/nemotron-3-ultra-550b-a55b"
        effective_max_tokens = max_tokens or AI_MAX_TOKENS
        logger.info("NVIDIA LLM query: model=%s, temp=%.2f, max_tokens=%d", model, AI_TEMPERATURE, effective_max_tokens)

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

        # NVIDIA NIM requests (especially for massive models like Nemotron-3-550B)
        # require a longer timeout limit than standard cloud models.
        nvidia_timeout = 150
        try:
            response = requests.post(
                self._endpoint,
                headers=headers,
                json=payload,
                timeout=nvidia_timeout,
            )
        except requests.Timeout as err:
            logger.error("NVIDIA LLM query timed out.")
            raise AIProviderTimeoutError(f"NVIDIA API query timed out after {nvidia_timeout}s: {err}") from err
        except requests.RequestException as err:
            logger.error("NVIDIA API network error: %s", err)
            raise AIProviderError(f"NVIDIA API request failed: {err}") from err

        if response.status_code == 429:
            logger.warning("NVIDIA API returned 429 Rate Limit.")
            raise AIProviderRateLimitError("NVIDIA API rate limit reached.")
        elif response.status_code == 503 and "worker local" in response.text.lower():
            logger.warning("NVIDIA API returned 503: Worker local request limit reached. Treating as transient queue limit.")
            raise AIProviderRateLimitError("NVIDIA API worker queue limit reached.")
        elif response.status_code != 200:
            logger.error("NVIDIA API returned error status: %d. Body: %s", response.status_code, response.text)
            raise AIProviderError(f"NVIDIA API error {response.status_code}: {response.text}")

        try:
            result = response.json()
            choice = result["choices"][0]
            
            # Check whether the model stopped because it reached the output token limit
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                logger.error("NVIDIA model output truncated: hit the output token limit (max_tokens).")
                raise AIProviderError("NVIDIA response was truncated (reached output token limit).")

            completion = choice["message"]["content"]
            
            # Log usage metrics
            usage = result.get("usage", {})
            logger.info("NVIDIA success: prompt_tokens=%d, completion_tokens=%d", 
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return completion
        except (KeyError, ValueError, json.JSONDecodeError) as err:
            logger.exception("Failed to parse NVIDIA response format")
            raise AIProviderError(f"Invalid response structure from NVIDIA API: {err}") from err

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
        return global_model if global_model and "/" in global_model else "nvidia/nemotron-3-ultra-550b-a55b"
