"""NVIDIA NIM ASR/LLM provider using OpenAI-compatible HTTP interface."""

import importlib
import openai
from openai import OpenAI

from config.settings import (
    NVIDIA_API_KEY,
    AI_MODEL,
    AI_TEMPERATURE,
    AI_MAX_TOKENS,
    AI_TOP_P,
    AI_TIMEOUT,
    NVIDIA_MOM_MODEL,
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
    """NVIDIA ASR/LLM Provider."""

    def __init__(self, model_override: str | None = None) -> None:
        self._api_key: str = NVIDIA_API_KEY
        self._endpoint: str = "https://integrate.api.nvidia.com/v1"
        self._model_override = model_override
        if self._api_key:
            self.client = OpenAI(
                base_url=self._endpoint,
                api_key=self._api_key
            )
        else:
            self.client = None

    def get_name(self) -> str:
        return "nvidia"

    def is_configured(self) -> bool:
        return bool(self._api_key) and self.client is not None

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        if not self.is_configured():
            raise AIProviderError("NVIDIA_API_KEY is not set.")

        model = self._model_override or NVIDIA_MOM_MODEL
        effective_max_tokens = max_tokens or AI_MAX_TOKENS
        logger.info("NVIDIA LLM query: model=%s", model)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Determine which model strategy to use
        model_key = None
        model_lower = model.lower()
        if "deepseek" in model_lower:
            model_key = "deepseek"
        elif "glm-5.2" in model_lower:
            model_key = "glm"
        elif "nemotron" in model_lower:
            model_key = "nemotron"


        try:
            if model_key:
                # Dynamically load the model strategy
                module_name = f"ai.providers.nvidia.models.{model_key}"
                strategy_module = importlib.import_module(module_name)
                completion_text = strategy_module.execute(
                    client=self.client,
                    model=model,
                    messages=messages,
                    temperature=AI_TEMPERATURE,
                    max_tokens=effective_max_tokens,
                    top_p=AI_TOP_P
                )
            else:
                # Fallback to standard OpenAI execution if no specific strategy exists
                logger.warning(f"No specific strategy found for NVIDIA model: {model}, using default.")
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=AI_TEMPERATURE,
                    max_tokens=effective_max_tokens,
                    top_p=AI_TOP_P,
                    stream=False
                )
                completion_text = completion.choices[0].message.content

            logger.info("NVIDIA success: generated %d chars", len(completion_text))
            return completion_text
            
        except openai.RateLimitError as err:
            logger.warning("NVIDIA API returned Rate Limit.")
            raise AIProviderRateLimitError("NVIDIA API rate limit reached.") from err
        except openai.APITimeoutError as err:
            logger.error("NVIDIA LLM query timed out.")
            raise AIProviderTimeoutError(f"NVIDIA API query timed out: {err}") from err
        except openai.OpenAIError as err:
            logger.error("NVIDIA API error: %s", err)
            raise AIProviderError(f"NVIDIA API request failed: {err}") from err

    def generate_summary(
        self,
        title: str,
        date: str,
        transcript: str,
        speaker_transcript: str | None = None,
    ) -> str:
        from datetime import datetime
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
        return self._model_override or NVIDIA_MOM_MODEL
