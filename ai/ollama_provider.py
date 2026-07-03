"""Ollama LLM provider supporting local instances and online endpoints."""

from datetime import datetime
import json
import requests

from config.settings import (
    OLLAMA_BASE_URL,
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


class OllamaAIProvider(BaseAIProvider):
    """Ollama LLM Provider supporting local and online API configurations."""

    def __init__(self) -> None:
        self._base_url: str = OLLAMA_BASE_URL or "http://localhost:11434"
        from config.settings import OLLAMA_API_KEY, OLLAMA_MODEL
        self._api_key: str = OLLAMA_API_KEY
        self._online_model: str = OLLAMA_MODEL

    def get_name(self) -> str:
        return "ollama"

    def is_configured(self) -> bool:
        """Ollama is considered configured if we have an API key or can reach tags."""
        if self._api_key:
            return True
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _get_first_available_model(self, include_configured: bool = True) -> str:
        """Query Ollama server /api/tags to find available models.

        Returns configured online model, first non-embedding model, or fallback.
        """
        if include_configured and self._online_model:
            logger.info("Using configured online Ollama model: '%s'", self._online_model)
            return self._online_model

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            url = f"{self._base_url}/api/tags"
            response = requests.get(url, headers=headers, timeout=3)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                if models:
                    for m in models:
                        name = m.get("name", "")
                        if name:
                            name_lower = name.lower()
                            if "embed" in name_lower or "bge-" in name_lower or "mxbai" in name_lower:
                                continue
                            logger.info("Auto-detected Ollama chat model: '%s'", name)
                            return name
                    
                    # Fallback if all are filtered out
                    first_model = models[0].get("name", "")
                    if first_model:
                        logger.info("Using first Ollama model: '%s'", first_model)
                        return first_model
        except Exception as exc:
            logger.warning("Could not auto-detect local Ollama models from /api/tags: %s", exc)
        return "llama3"

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        # Determine model
        model = self._online_model or AI_MODEL
        if not model or "/" in model or "nvidia" in model or "deepseek" in model or "gemini" in model:
            model = self._get_first_available_model()

        endpoint = f"{self._base_url}/api/chat"
        effective_max_tokens = max_tokens or AI_MAX_TOKENS
        logger.info("Ollama LLM query: model=%s, endpoint=%s, max_tokens=%d", model, endpoint, effective_max_tokens)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": AI_TEMPERATURE,
                "num_predict": effective_max_tokens,
                "top_p": AI_TOP_P,
            },
            "stream": False,
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=AI_TIMEOUT,
            )
        except requests.Timeout as err:
            logger.error("Ollama query timed out.")
            raise AIProviderTimeoutError(f"Ollama local query timed out: {err}") from err
        except requests.RequestException as err:
            logger.error("Ollama connection error: %s", err)
            raise AIProviderError(f"Ollama request failed: {err}") from err

        if response.status_code == 404 and "not found" in response.text.lower() and self._online_model:
            fallback_model = self._get_first_available_model(include_configured=False)
            if fallback_model != model:
                logger.warning("Configured Ollama model '%s' was not found. Retrying with '%s'.", model, fallback_model)
                payload["model"] = fallback_model
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=AI_TIMEOUT,
                )
                model = fallback_model

        if response.status_code == 429:
            raise AIProviderRateLimitError("Ollama rate limit reached.")
        elif response.status_code == 400 and "does not support chat" in response.text:
            raise AIProviderError(
                f"Ollama model '{model}' does not support chat/generation. "
                "Please pull a text model (e.g. 'ollama pull llama3.2' or 'ollama pull llama3') in your terminal."
            )
        elif response.status_code != 200:
            raise AIProviderError(f"Ollama error {response.status_code}: {response.text}")

        try:
            result = response.json()
            return result["message"]["content"]
        except (KeyError, ValueError) as err:
            raise AIProviderError(f"Invalid Ollama response structure: {err}") from err

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
        model = self._online_model or global_model
        if not model or "/" in model or "nvidia" in model or "deepseek" in model or "gemini" in model:
            return self._online_model or "llama3"
        return model
