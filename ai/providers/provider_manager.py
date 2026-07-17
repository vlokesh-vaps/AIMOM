"""Centralized LLM request gateway with dynamic scheduling and recovery.

All AI agent requests route through ProviderManager. It handles:
  - Dynamic provider selection based on real-time health metrics
  - Configurable cooldown periods for rate-limited providers
  - Exponential backoff retries on transient errors
  - Automatic failover across all configured providers
  - Health monitoring with automatic recovery
  - Structured logging of every request lifecycle event
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

from ai.providers.base import (
    BaseAIProvider,
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AIProviderTruncatedResponseError,
)
from config.settings import (
    PROVIDER_MAX_RETRIES,
    PROVIDER_INITIAL_BACKOFF,
    PROVIDER_BACKOFF_FACTOR,
    PROVIDER_HEALTH_CHECK_INTERVAL,
    PROVIDER_COOLDOWN_SECONDS,
    GROQ_FALLBACK_MODEL,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Error classification keywords
# ---------------------------------------------------------------------------
_TRANSIENT_STATUS_CODES = ("500", "502", "503", "504")
_TRANSIENT_KEYWORDS = (
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "timeout",
    "connection",
    "rate_limit",
    "resourceexhausted",
)
_PERMANENT_KEYWORDS = (
    "request too large",
    "please reduce your message size",
    "maximum context length",
    "context length",
)


@dataclass
class ProviderHealth:
    """Tracks the operational health of a single provider."""

    name: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_requests: int = 0
    total_failures: int = 0
    total_failovers: int = 0
    cooldown_until: float = 0.0  # monotonic time when cooldown expires
    avg_latency: float = 0.0  # running average latency in seconds


class ProviderManager:
    """Centralized LLM request gateway with dynamic scheduling and recovery.

    Providers are ranked dynamically based on health, cooldown state, and
    recent latency. Every request iterates from the best to the worst
    provider. Rate-limited providers enter a configurable cooldown period
    and are automatically skipped until the cooldown expires.

    Every request goes through:
        best_provider → retry → next_best_provider → retry → … → error.
    """

    def __init__(self, providers: dict[str, BaseAIProvider]) -> None:
        self._providers = providers
        self._health: dict[str, ProviderHealth] = {}
        self._lock = Lock()

        for name in providers:
            self._health[name] = ProviderHealth(name=name)

        configured = [n for n, p in providers.items() if p.is_configured()]
        logger.info(
            "[ProviderManager] Initialized with providers: %s (configured: %s)",
            list(providers.keys()),
            configured,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        system_prompt: str,
        user_prompt: str,
        primary_model: str,
        fallback_model: str | None = None,
        max_tokens: int | None = None,
        agent_name: str = "",
    ) -> str:
        """Route a request through dynamically-ranked providers.

        Tries each available provider in order of health/latency,
        respecting cooldown periods.

        Args:
            system_prompt: System instructions for the LLM.
            user_prompt: User content for the LLM.
            primary_model: Model identifier for the primary provider.
            fallback_model: Model identifier for the fallback provider.
                If None, uses GROQ_FALLBACK_MODEL.
            max_tokens: Maximum completion tokens.
            agent_name: Human-readable agent name for logging.

        Returns:
            The LLM response text.

        Raises:
            AIProviderError: When all providers fail after all retries.
        """
        request_start = time.monotonic()
        fallback_model = fallback_model or GROQ_FALLBACK_MODEL
        log_prefix = f"[ProviderManager:{agent_name}]" if agent_name else "[ProviderManager]"

        # Build ordered list of (provider_name, model) pairs
        ranked = self._rank_providers(primary_model, fallback_model)

        if not ranked:
            raise AIProviderError(
                f"{log_prefix} No configured or healthy providers available."
            )

        last_error: Exception | None = None

        for provider_name, model in ranked:
            provider = self._providers[provider_name]

            # Skip providers still in cooldown
            if self._is_in_cooldown(provider_name):
                logger.info(
                    "%s Skipping '%s' — still in cooldown (%.0fs remaining).",
                    log_prefix, provider_name,
                    self._cooldown_remaining(provider_name),
                )
                continue

            logger.info(
                "%s Routing to %s/%s",
                log_prefix, provider_name, model,
            )

            try:
                result = self._retry_with_backoff(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    log_prefix=log_prefix,
                )
                self._record_success(provider_name, time.monotonic() - request_start)
                elapsed = time.monotonic() - request_start
                logger.info(
                    "%s Success via %s/%s (%.2fs)",
                    log_prefix, provider_name, model, elapsed,
                )
                return result

            except AIProviderRateLimitError as exc:
                self._record_failure(provider_name, enter_cooldown=True)
                logger.warning(
                    "%s Provider '%s' rate-limited — entering cooldown (%.0fs). Trying next.",
                    log_prefix, provider_name, PROVIDER_COOLDOWN_SECONDS,
                )
                last_error = exc

            except AIProviderError as exc:
                self._record_failure(provider_name)
                logger.warning(
                    "%s Provider '%s' failed: %s. Trying next.",
                    log_prefix, provider_name, exc,
                )
                last_error = exc

        elapsed = time.monotonic() - request_start
        raise AIProviderError(
            f"{log_prefix} All providers failed after exhausting retries ({elapsed:.2f}s). "
            f"Last error: {last_error}"
        )

    def get_health_status(self) -> dict[str, ProviderHealth]:
        """Return a snapshot of all provider health states."""
        with self._lock:
            return {name: ProviderHealth(**vars(h)) for name, h in self._health.items()}

    # ------------------------------------------------------------------
    # Internal — provider ranking
    # ------------------------------------------------------------------

    def _rank_providers(
        self, primary_model: str, fallback_model: str,
    ) -> list[tuple[str, str]]:
        """Return an ordered list of (provider_name, model) based on health.

        Prioritizes:
          1. Healthy, not in cooldown, lowest avg latency
          2. Healthy, not in cooldown
          3. In cooldown but past recovery threshold
        """
        primary_name, fallback_name = self._resolve_providers(primary_model)
        candidates = []

        # Always try primary first, then fallback
        candidate_pairs = [
            (primary_name, primary_model),
            (fallback_name, fallback_model),
        ]

        for name, model in candidate_pairs:
            provider = self._providers.get(name)
            if not provider or not provider.is_configured():
                continue

            with self._lock:
                h = self._health[name]
                # Attempt recovery for unhealthy providers
                if not h.is_healthy and h.last_failure_time is not None:
                    elapsed = time.monotonic() - h.last_failure_time
                    if elapsed >= PROVIDER_HEALTH_CHECK_INTERVAL:
                        logger.info(
                            "[ProviderManager] Probing '%s' for recovery (%.0fs since last failure).",
                            name, elapsed,
                        )
                        h.is_healthy = True
                        h.consecutive_failures = 0

                if not h.is_healthy:
                    continue

                candidates.append((name, model, h.avg_latency))

        # Sort by average latency (lowest first), preserving primary preference
        # on ties via stable sort
        candidates.sort(key=lambda c: c[2])
        return [(name, model) for name, model, _ in candidates]

    # ------------------------------------------------------------------
    # Internal — retry logic
    # ------------------------------------------------------------------

    def _retry_with_backoff(
        self,
        provider: BaseAIProvider,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None,
        log_prefix: str,
    ) -> str:
        """Retry a provider call with exponential backoff on transient errors.

        Never waits for free-tier cooldown periods (300+ seconds).
        Fails over immediately after PROVIDER_MAX_RETRIES.
        """
        attempt = 0
        backoff = PROVIDER_INITIAL_BACKOFF
        current_max_tokens = max_tokens

        while attempt <= PROVIDER_MAX_RETRIES:
            try:
                return self._call_provider(provider, model, system_prompt, user_prompt, current_max_tokens)
            except AIProviderTruncatedResponseError as exc:
                # Retry with increased token budget
                if attempt >= PROVIDER_MAX_RETRIES:
                    logger.error(
                        "%s [%s] Truncated after %d retries. Giving up.",
                        log_prefix, provider.get_name(), PROVIDER_MAX_RETRIES,
                    )
                    raise
                if current_max_tokens:
                    current_max_tokens = min(max(int(current_max_tokens * 1.5), current_max_tokens + 512), 16384)
                logger.warning(
                    "%s [%s] Truncated (attempt %d/%d). Retrying with max_tokens=%s in %.1fs.",
                    log_prefix, provider.get_name(), attempt + 1,
                    PROVIDER_MAX_RETRIES, current_max_tokens, backoff,
                )
                time.sleep(backoff)
                attempt += 1
                backoff *= PROVIDER_BACKOFF_FACTOR

            except AIProviderRateLimitError:
                # Propagate immediately to the execute() loop which handles cooldown
                raise

            except AIProviderTimeoutError as exc:
                if attempt >= PROVIDER_MAX_RETRIES:
                    logger.error(
                        "%s [%s] %s after %d retries. Giving up.",
                        log_prefix, provider.get_name(),
                        type(exc).__name__, PROVIDER_MAX_RETRIES,
                    )
                    raise
                sleep_time = backoff
                logger.warning(
                    "%s [%s] %s (attempt %d/%d). Backing off %.1fs.",
                    log_prefix, provider.get_name(), exc,
                    attempt + 1, PROVIDER_MAX_RETRIES, sleep_time,
                )
                time.sleep(sleep_time)
                attempt += 1
                backoff *= PROVIDER_BACKOFF_FACTOR

            except AIProviderError as exc:
                if self._is_permanent_error(exc):
                    logger.error(
                        "%s [%s] Permanent error: %s. Not retrying.",
                        log_prefix, provider.get_name(), exc,
                    )
                    raise
                if self._is_transient_error(exc) and attempt < PROVIDER_MAX_RETRIES:
                    logger.warning(
                        "%s [%s] Transient error: %s (attempt %d/%d). Backing off %.1fs.",
                        log_prefix, provider.get_name(), exc,
                        attempt + 1, PROVIDER_MAX_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    attempt += 1
                    backoff *= PROVIDER_BACKOFF_FACTOR
                else:
                    logger.error(
                        "%s [%s] Unrecoverable or retries exhausted: %s.",
                        log_prefix, provider.get_name(), exc,
                    )
                    raise

        raise AIProviderError(
            f"{log_prefix} [{provider.get_name()}] Retries exhausted ({PROVIDER_MAX_RETRIES})."
        )

    def _call_provider(
        self,
        provider: BaseAIProvider,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None,
    ) -> str:
        """Make a single LLM call, dynamically selecting the correct provider instance."""
        # Create a provider instance pinned to the specific model
        provider_name = provider.get_name()
        if provider_name == "nvidia":
            from ai.providers.nvidia import NvidiaAIProvider
            pinned = NvidiaAIProvider(model_override=model)
        elif provider_name == "groq":
            from ai.providers.groq import GroqAIProvider
            pinned = GroqAIProvider(model_override=model)
        else:
            pinned = provider

        return pinned.generate_text(system_prompt, user_prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Internal — error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Return True for errors worth retrying (rate limits, server errors, timeouts)."""
        if isinstance(exc, (AIProviderRateLimitError, AIProviderTimeoutError, AIProviderTruncatedResponseError)):
            return True
        msg = str(exc).lower()
        if any(code in msg for code in _TRANSIENT_STATUS_CODES):
            return not any(kw in msg for kw in _PERMANENT_KEYWORDS)
        return any(kw in msg for kw in _TRANSIENT_KEYWORDS)

    @staticmethod
    def _is_permanent_error(exc: Exception) -> bool:
        """Return True for errors that should not be retried (payload too large, auth)."""
        msg = str(exc).lower()
        return any(kw in msg for kw in _PERMANENT_KEYWORDS)

    # ------------------------------------------------------------------
    # Internal — health & cooldown monitoring
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, provider_name: str) -> bool:
        """Check if a provider is still in its cooldown period."""
        with self._lock:
            return time.monotonic() < self._health[provider_name].cooldown_until

    def _cooldown_remaining(self, provider_name: str) -> float:
        """Return seconds remaining in cooldown (0 if not in cooldown)."""
        with self._lock:
            remaining = self._health[provider_name].cooldown_until - time.monotonic()
            return max(0.0, remaining)

    def _is_healthy(self, provider_name: str) -> bool:
        with self._lock:
            return self._health[provider_name].is_healthy

    def _record_success(self, provider_name: str, latency: float = 0.0) -> None:
        with self._lock:
            h = self._health[provider_name]
            h.is_healthy = True
            h.consecutive_failures = 0
            h.last_success_time = time.monotonic()
            h.total_requests += 1
            # Exponential moving average for latency
            if h.avg_latency == 0.0:
                h.avg_latency = latency
            else:
                h.avg_latency = h.avg_latency * 0.7 + latency * 0.3

    def _record_failure(self, provider_name: str, enter_cooldown: bool = False) -> None:
        with self._lock:
            h = self._health[provider_name]
            h.consecutive_failures += 1
            h.total_failures += 1
            h.total_requests += 1
            h.last_failure_time = time.monotonic()

            if enter_cooldown:
                h.cooldown_until = time.monotonic() + PROVIDER_COOLDOWN_SECONDS
                logger.info(
                    "[ProviderManager] Provider '%s' entering cooldown for %.0fs.",
                    provider_name, PROVIDER_COOLDOWN_SECONDS,
                )

            # Mark unhealthy after consecutive failures exceed retry count
            if h.consecutive_failures >= PROVIDER_MAX_RETRIES:
                if h.is_healthy:
                    logger.warning(
                        "[ProviderManager] Marking '%s' as UNHEALTHY after %d consecutive failures.",
                        provider_name, h.consecutive_failures,
                    )
                h.is_healthy = False

    def _attempt_recovery(self, provider_name: str) -> None:
        """Check if enough time has passed to probe a previously-failed provider."""
        with self._lock:
            h = self._health[provider_name]
            if h.is_healthy:
                return
            if h.last_failure_time is None:
                return
            elapsed = time.monotonic() - h.last_failure_time
            if elapsed < PROVIDER_HEALTH_CHECK_INTERVAL:
                return
            # Enough time has passed — optimistically mark healthy for a probe
            logger.info(
                "[ProviderManager] Probing '%s' for recovery (%.0fs since last failure).",
                provider_name, elapsed,
            )
            h.is_healthy = True
            h.consecutive_failures = 0

    # ------------------------------------------------------------------
    # Internal — provider resolution
    # ------------------------------------------------------------------

    def _resolve_providers(self, primary_model: str) -> tuple[str, str]:
        """Determine primary and fallback provider names from the model identifier.

        NVIDIA models → primary=nvidia, fallback=groq
        Groq models → primary=groq, fallback=nvidia
        """
        model_lower = primary_model.lower()

        # Groq-hosted models
        groq_prefixes = ("openai/gpt-oss", "llama-3")
        if any(model_lower.startswith(p) for p in groq_prefixes):
            return "groq", "nvidia"

        # Default: NVIDIA primary, Groq fallback
        return "nvidia", "groq"

