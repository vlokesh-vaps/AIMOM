"""Ollama LLM provider supporting local instances and online endpoints."""

from datetime import datetime
import json
import requests

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    OLLAMA_MODEL,
    AI_MODEL,
    AI_TEMPERATURE,
    AI_MAX_TOKENS,
    AI_TOP_P,
    AI_TIMEOUT,
)

# Ollama local inference needs a much longer timeout than cloud providers.
# Cloud providers respond in seconds; Ollama may spend 60-180s on first token
# depending on hardware and model size (gemma4 is a large model).
_OLLAMA_TIMEOUT = 500  # 5 minutes; overrides global AI_TIMEOUT for local inference
_PREFERRED_MODEL = "gemma4:latest"
_EMBEDDING_KEYWORDS = ("embed", "bge-", "mxbai", "nomic", "rerank", "minilm")
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


class OllamaAIProvider(BaseAIProvider):
    """Ollama LLM Provider supporting local and online API configurations."""

    def __init__(self) -> None:
        self._base_url: str = OLLAMA_BASE_URL or "http://localhost:11434"
        self._api_key: str = OLLAMA_API_KEY
        self._online_model: str = OLLAMA_MODEL

    def get_name(self) -> str:
        return "ollama"

    def _validate_and_select_model(self) -> str:
        """Query /api/tags and select the best available local chat model.

        Prioritizes:
          1. 'gemma:4b' (or any variant with 'gemma:4b' or 'gemma4')
          2. 'gemma:2b'
          3. Configured online model (if set)
          4. First non-embedding model found

        Raises AIProviderError if Ollama is unreachable or has no usable models.
        """
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            url = f"{self._base_url}/api/tags"
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:
                raise AIProviderError(f"Ollama /api/tags returned HTTP {response.status_code}")

            data = response.json()
            available = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            logger.info("[ollama] Available local models: %s", available)

            # Prioritize gemma4:latest first (Gemma 4), then gemma3, then gemma:4b, gemma:2b
            for m in available:
                if m == "gemma4:latest" or m.startswith("gemma4:") or "gemma4" in m:
                    logger.info("[ollama] Selecting local model gemma4 match: '%s'", m)
                    return m

            for m in available:
                if m == "gemma3:4b" or m.startswith("gemma3:") or "gemma3" in m:
                    logger.info("[ollama] Selecting local model gemma3 match: '%s'", m)
                    return m

            for m in available:
                if m == "gemma:4b" or m.startswith("gemma:4") or "gemma:4" in m:
                    logger.info("[ollama] Selecting local model gemma:4b match: '%s'", m)
                    return m

            for m in available:
                if m == "gemma:2b" or m.startswith("gemma:2") or "gemma:2" in m:
                    logger.info("[ollama] Selecting local model gemma:2b match: '%s'", m)
                    return m

            # Online / fallback config
            if self._online_model and self._online_model in available:
                return self._online_model

            # Any non-embedding model
            chat_models = [
                name for name in available
                if not any(kw in name.lower() for kw in _EMBEDDING_KEYWORDS)
            ]
            if chat_models:
                logger.info("[ollama] Selecting first available non-embedding model: '%s'", chat_models[0])
                return chat_models[0]

            raise AIProviderError(
                f"No usable chat models found in local Ollama tags. Available: {available}. "
                "Please pull gemma4:latest in your terminal: 'ollama pull gemma4:latest'"
            )
        except requests.RequestException as exc:
            raise AIProviderError(f"Cannot connect to local Ollama service: {exc}") from exc


    def is_configured(self) -> bool:
        """Return True only when Ollama is reachable AND a usable chat model is available.

        A strict validation that prevents downstream errors from:
        - Ollama being offline
        - Only embedding models being available (e.g. embeddinggemma)
        - The preferred model not being pulled yet
        """
        try:
            self._validate_and_select_model()
            return True
        except Exception as exc:
            logger.debug("[ollama] is_configured=False: %s", exc)
            return False

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        # Determine model with strict validation
        model = self._validate_and_select_model()

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
                timeout=_OLLAMA_TIMEOUT,
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
                    timeout=_OLLAMA_TIMEOUT,
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
            done = result.get("done", True)
            done_reason = result.get("done_reason")
            
            # Check whether the model stopped because it reached the output token limit
            if not done or done_reason == "length":
                logger.error("Ollama model output truncated: hit the output token limit.")
                raise AIProviderTruncatedResponseError("Ollama response was truncated (reached output token limit).")
                
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
            try:
                return self._validate_and_select_model()
            except Exception:
                return _PREFERRED_MODEL
        return model
