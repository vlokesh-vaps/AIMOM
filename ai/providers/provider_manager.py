"""Centralized LLM request gateway with adaptive scheduling, load balancing, and truncation recovery.

All AI pipeline requests route through ProviderManager. It handles:
  - Dynamic model selection based on real-time latency, health, and load
  - Custom dynamic concurrency gates per provider
  - Priority-based hybrid scheduling (priority first, then complexity)
  - Cooldown and circuit breaker temporaries
  - Mid-stream truncation recovery via recursive continuations
  - Detailed consolidated metrics and execution reporting
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.providers.base import (
    BaseAIProvider,
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AIProviderTruncatedResponseError,
)
from ai.utils.token_utils import estimate_tokens
from config.settings import PROVIDER_CONFIG_PATH, GROQ_FALLBACK_MODEL
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
    """Tracks the operational health and metrics of a single provider."""

    name: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_requests: int = 0
    total_failures: int = 0
    total_failovers: int = 0
    cooldown_until: float = 0.0  # monotonic time when cooldown expires
    avg_latency: float = 0.0  # EMA running average latency in seconds
    success_rate: float = 1.0  # calculated historical success rate
    truncations_recovered: int = 0
    retry_count: int = 0


@dataclass
class QueueItem:
    """Represents an item waiting in the scheduler queue."""

    priority: int  # lower = higher priority (0 = highest)
    complexity: int  # estimated token count
    chunk_index: int  # tie breaker
    timestamp: float
    event: asyncio.Event
    task_type: str
    user_prompt: str
    selected_provider: str | None = None
    selected_model: str | None = None


class ProviderManager:
    """Centralized LLM request gateway with dynamic scheduling and recovery.

    Routes requests to NVIDIA (primary) or Groq (fallback) based on real-time
    health metrics, capacity limits, weights, and priority queuing.
    """

    def __init__(self, providers: dict[str, BaseAIProvider]) -> None:
        self._providers = providers
        self._health: dict[str, ProviderHealth] = {}
        self._active_requests: dict[str, int] = {}
        self._active_concurrency_limits: dict[str, int] = {}
        self._queue: list[QueueItem] = []
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        
        # Load config
        self._load_config()

        # Initialize health metrics for allowed providers only
        for name in ["nvidia", "groq"]:
            if name in providers:
                self._health[name] = ProviderHealth(name=name)
                self._active_requests[name] = 0
                self._active_concurrency_limits[name] = self._concurrency_limits.get(name, 2)

        configured = [n for n, p in providers.items() if p.is_configured() and n in ["nvidia", "groq"]]
        logger.info(
            "[ProviderManager] Initialized with NVIDIA and Groq (configured: %s)",
            configured,
        )

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    async def execute_async(
        self,
        system_prompt: str,
        user_prompt: str,
        task_type: str = "extraction",
        max_tokens: int | None = None,
        priority: int = 1,
        chunk_index: int = 0,
        agent_name: str = "",
    ) -> str:
        """Route request through priority scheduler queue with dynamic reselection."""
        complexity = estimate_tokens(system_prompt + user_prompt)
        log_prefix = f"[ProviderManager:{agent_name}]" if agent_name else "[ProviderManager]"

        attempt = 0
        last_error: Exception | None = None
        max_attempts = 5

        while attempt < max_attempts:
            item = QueueItem(
                priority=priority,
                complexity=complexity,
                chunk_index=chunk_index,
                timestamp=time.monotonic(),
                event=asyncio.Event(),
                task_type=task_type,
                user_prompt=user_prompt,
            )

            async with self._lock:
                self._queue.append(item)
                self._schedule_next()

            # Wait for scheduler slot allocation
            await item.event.wait()

            if not item.selected_provider:
                raise AIProviderError(
                    f"{log_prefix} No healthy or configured providers available for task type '{task_type}'."
                )

            provider_name = item.selected_provider
            model = item.selected_model
            provider = self._providers[provider_name]

            logger.info(
                "%s [Attempt %d] Scheduled on %s/%s (priority=%d, complexity=%d tokens, active_load=%d/%d)",
                log_prefix, attempt + 1, provider_name, model,
                priority, complexity, self._active_requests[provider_name],
                self._active_concurrency_limits[provider_name],
            )

            start_call = time.monotonic()
            try:
                # Execute the call with mid-stream truncation recovery
                result = await self._execute_call_with_continuation(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    log_prefix=log_prefix,
                )

                # Call succeeded: record EMA metrics and release concurrency slot
                latency = time.monotonic() - start_call
                self._record_success(provider_name, latency)
                
                async with self._lock:
                    self._active_requests[provider_name] = max(0, self._active_requests[provider_name] - 1)
                    # Success check: dynamically scale up concurrency limit to configured max if healthy
                    cfg_max = self._concurrency_limits.get(provider_name, 2)
                    if self._active_concurrency_limits[provider_name] < cfg_max:
                        self._active_concurrency_limits[provider_name] += 1
                        logger.info("[ProviderManager] Concurrency limit for '%s' scaled UP to %d", provider_name, self._active_concurrency_limits[provider_name])
                    self._schedule_next()
                    self._condition.notify_all()

                return result

            except Exception as exc:
                # Call failed: release concurrency slot, record failure, scale down concurrency
                latency = time.monotonic() - start_call
                is_rate_limit = (
                    isinstance(exc, AIProviderRateLimitError)
                    or "429" in str(exc)
                    or "resourceexhausted" in str(exc).lower()
                )

                async with self._lock:
                    self._active_requests[provider_name] = max(0, self._active_requests[provider_name] - 1)
                    self._record_failure(provider_name, enter_cooldown=is_rate_limit)
                    
                    # Rate limit scaling: drop active concurrency threshold to throttle load
                    if is_rate_limit:
                        self._active_concurrency_limits[provider_name] = max(1, self._active_concurrency_limits[provider_name] - 1)
                        logger.warning("[ProviderManager] Concurrency limit for '%s' scaled DOWN to %d due to rate limits", provider_name, self._active_concurrency_limits[provider_name])
                    
                    self._schedule_next()
                    self._condition.notify_all()

                logger.warning(
                    "%s Call to %s/%s failed (%.2fs): %s. Reselecting candidate provider...",
                    log_prefix, provider_name, model, latency, exc,
                )
                last_error = exc
                attempt += 1

        raise AIProviderError(
            f"{log_prefix} All provider selections failed after all re-schedulings ({max_attempts} attempts). "
            f"Last error: {last_error}"
        )

    def execute(
        self,
        system_prompt: str,
        user_prompt: str,
        primary_model: str,
        fallback_model: str | None = None,
        max_tokens: int | None = None,
        agent_name: str = "",
    ) -> str:
        """Synchronous legacy wrapper for execute_async (used by translation and tests)."""
        # Guess task type based on agent_name
        task_type = "extraction"
        if "synthesis" in agent_name.lower():
            task_type = "synthesis"
        elif "translation" in agent_name.lower():
            task_type = "translation"

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    self.execute_async(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        task_type=task_type,
                        max_tokens=max_tokens,
                        priority=0,  # Sync requests are high priority
                        agent_name=agent_name,
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self.execute_async(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    task_type=task_type,
                    max_tokens=max_tokens,
                    priority=0,
                    agent_name=agent_name,
                )
            )

    def generate_execution_report(self, total_duration: float, checkpoints_cached: int = 0) -> str:
        """Consolidate current run stats into a detailed Markdown observability report."""
        report = []
        report.append("# AIMOM Pipeline Execution Report")
        report.append(f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"- **Total Processing Time**: {total_duration:.1f} seconds")
        report.append(f"- **Checkpoints Cached / Resumed**: {checkpoints_cached}")
        report.append("")
        report.append("## Provider Metrics & Utilization")
        report.append("| Provider | Success Rate | Requests | Successes | Failures | Failovers | Average Latency | Retries | Truncations Recovered |")
        report.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        recommendations = []
        for name in ["nvidia", "groq"]:
            h = self._health.get(name)
            if not h:
                continue
            report.append(
                f"| {name.upper()} | {h.success_rate * 100:.1f}% | {h.total_requests} | "
                f"{h.total_requests - h.total_failures} | {h.total_failures} | {h.total_failovers} | "
                f"{h.avg_latency:.2f}s | {h.retry_count} | {h.truncations_recovered} |"
            )
            # Gather recommendations
            if h.total_failures > 0:
                if name == "nvidia":
                    recommendations.append("- **NVIDIA NIM throttling**: Consider reducing `MAX_CONCURRENT_EXTRACTIONS` in settings, or switch primary provider weights in `config/provider_config.json` to balance load with Groq.")
                elif name == "groq":
                    recommendations.append("- **Groq TPM limit reached**: Groq limits free-tier TPM. Consider increasing `PROVIDER_COOLDOWN_SECONDS` or reducing input chunk sizes.")
            if h.truncations_recovered > 0:
                recommendations.append(f"- **{name.upper()} Output Truncation**: Model hit output token limit. Consider increasing `EXTRACTION_MAX_TOKENS` in configurations.")

        if not recommendations:
            recommendations.append("- Everything executed cleanly. Concurrency levels and provider load ratios are optimally configured.")

        report.append("")
        report.append("## Observations & Recommendations")
        report.extend(recommendations)
        report.append("")
        
        return "\n".join(report)

    # ------------------------------------------------------------------
    # Scheduler Core
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        """Evaluate waiting items and dispatch them to the best available slots.

        Sorted by priority, complexity (estimated size), chunk index, and timestamp.
        """
        if not self._queue:
            return

        # Sort: priority class (0=highest), complexity (cheaper/shorter first), chunk index, timestamp
        self._queue.sort(key=lambda x: (x.priority, x.complexity, x.chunk_index, x.timestamp))

        still_waiting = []

        for item in self._queue:
            # Find configured, healthy providers for this task
            candidates = self._get_candidates(item.task_type)
            
            # Score each candidate
            scored = []
            for name, model in candidates:
                score = self._compute_score(name, model, item.complexity)
                scored.append((score, name, model))
            
            # Sort descending by score
            scored.sort(key=lambda x: x[0], reverse=True)

            # Find a candidate with concurrency slot available
            dispatched = False
            for _, name, model in scored:
                active_limit = self._active_concurrency_limits.get(name, 2)
                if self._active_requests[name] < active_limit:
                    # Allocate slot
                    self._active_requests[name] += 1
                    item.selected_provider = name
                    item.selected_model = model
                    item.event.set()
                    dispatched = True
                    break

            if not dispatched:
                # Keep item in queue to be scheduled later
                still_waiting.append(item)

        self._queue = still_waiting

    def _get_candidates(self, task_type: str) -> list[tuple[str, str]]:
        """Get candidates from weights config file, excluding those in cooldown/unhealthy."""
        prefs = self._config.get("model_preferences", {}).get(task_type, [])
        candidates = []

        for item in prefs:
            name = item.get("provider", "")
            model = item.get("model", "")
            
            # Only NVIDIA and Groq allowed
            if name not in ["nvidia", "groq"]:
                continue

            provider = self._providers.get(name)
            if not provider or not provider.is_configured():
                continue

            # Skip providers in active cooldown
            if time.monotonic() < self._health[name].cooldown_until:
                continue

            # Proactive recovery check
            h = self._health[name]
            if not h.is_healthy and h.last_failure_time is not None:
                elapsed = time.monotonic() - h.last_failure_time
                if elapsed >= 30.0:  # health check recovery time
                    h.is_healthy = True
                    h.consecutive_failures = 0

            if h.is_healthy:
                candidates.append((name, model))

        return candidates

    def _compute_score(self, provider_name: str, model: str, complexity: int) -> float:
        """Compute scheduling suitability score for a provider/model combo."""
        h = self._health[provider_name]
        
        # Get base weight from configuration
        base_weight = 50.0
        for task_type in self._config.get("model_preferences", {}):
            for m in self._config["model_preferences"][task_type]:
                if m.get("provider") == provider_name and m.get("model") == model:
                    base_weight = m.get("weight", 50.0)
                    break

        score = base_weight

        # 1. Capacity Penalty: heavily penalize saturated providers
        active = self._active_requests.get(provider_name, 0)
        limit = self._active_concurrency_limits.get(provider_name, 2)
        utilization = active / limit if limit > 0 else 1.0
        score -= utilization * 40.0

        # 2. Health Penalty: drop points for consecutive failures
        score -= h.consecutive_failures * 15.0

        # 3. Latency Penalty: penalize slower models
        score -= h.avg_latency * 1.5

        # 4. Token limit feasibility:
        # Groq has a strict free-tier TPM constraint (8000 tokens).
        # If the chunk complexity + estimated output exceeds 7000 tokens, penalize Groq
        if provider_name == "groq" and complexity > 6500:
            score -= 100.0  # Exclude or push to bottom of the preference queue

        return score

    # ------------------------------------------------------------------
    # LLM request & Truncation Continuation recovery
    # ------------------------------------------------------------------

    async def _execute_call_with_continuation(
        self,
        provider: BaseAIProvider,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None,
        log_prefix: str,
    ) -> str:
        """Call provider, and if truncated, recursively retrieve continuations."""
        loop = asyncio.get_running_loop()

        try:
            res = await loop.run_in_executor(
                None,
                self._call_provider,
                provider, model, system_prompt, user_prompt, max_tokens,
            )
            return res
        except AIProviderTruncatedResponseError as exc:
            partial = exc.partial_response
            if not partial:
                raise  # No partial content to continue from, propagate

            # Increment count
            self._health[provider.get_name()].truncations_recovered += 1
            logger.warning(
                "%s Output truncated. Activating recursive continuation recovery...",
                log_prefix,
            )

            accumulated = [partial]
            current_prompt = user_prompt
            current_partial = partial

            # Continuously request next fragments (up to 3 continuations)
            for cont_idx in range(3):
                # Request next segment starting directly after the previous output
                continuation_instruction = (
                    f"{current_prompt}\n\n"
                    f"--- CONTINUATION ---\n"
                    f"The previous output was truncated mid-stream. Here is the last 500 characters of what was generated:\n"
                    f"\"\"\"\n{current_partial[-500:]}\n\"\"\"\n"
                    f"Continue outputting the remaining content exactly from where it was cut off. "
                    f"Do NOT wrap in JSON code blocks or markdown fences. Just output the continuation text."
                )

                try:
                    cont_res = await loop.run_in_executor(
                        None,
                        self._call_provider,
                        provider, model, system_prompt, continuation_instruction, max_tokens,
                    )
                    accumulated.append(cont_res)
                    break  # Success
                except AIProviderTruncatedResponseError as next_exc:
                    logger.warning(
                        "%s Continuation fragment %d truncated. Recovering further...",
                        log_prefix, cont_idx + 1,
                    )
                    accumulated.append(next_exc.partial_response)
                    current_partial = next_exc.partial_response

            return "".join(accumulated)

    def _call_provider(
        self,
        provider: BaseAIProvider,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None,
    ) -> str:
        """Make a single LLM call, dynamically resolving the provider subclass instance."""
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
    # Metrics, Health, and Recovery tracking
    # ------------------------------------------------------------------

    def _record_success(self, provider_name: str, latency: float) -> None:
        h = self._health[provider_name]
        h.is_healthy = True
        h.consecutive_failures = 0
        h.last_success_time = time.monotonic()
        
        # EMA for Latency running average
        if h.avg_latency == 0.0:
            h.avg_latency = latency
        else:
            h.avg_latency = h.avg_latency * 0.7 + latency * 0.3

        h.total_requests += 1
        h.success_rate = (h.total_requests - h.total_failures) / h.total_requests

    def _record_failure(self, provider_name: str, enter_cooldown: bool = False) -> None:
        h = self._health[provider_name]
        h.consecutive_failures += 1
        h.total_failures += 1
        h.total_requests += 1
        h.last_failure_time = time.monotonic()
        h.success_rate = (h.total_requests - h.total_failures) / h.total_requests

        if enter_cooldown:
            duration = self._cooldown_durations.get(provider_name, 30.0)
            h.cooldown_until = time.monotonic() + duration
            h.total_failovers += 1
            logger.warning(
                "[ProviderManager] Provider '%s' entered rate-limit cooldown for %.1fs.",
                provider_name, duration,
            )
            try:
                asyncio.create_task(self._cooldown_timer(provider_name, duration))
            except RuntimeError:
                pass

        # Mark unhealthy after failures exceed limit
        retry_policy = self._retry_policies.get(provider_name, {"max_retries": 3})
        if h.consecutive_failures >= retry_policy.get("max_retries", 3):
            if h.is_healthy:
                logger.error(
                    "[ProviderManager] Provider '%s' marked UNHEALTHY after %d failures.",
                    provider_name, h.consecutive_failures,
                )
            h.is_healthy = False

    async def _cooldown_timer(self, provider_name: str, duration: float) -> None:
        """Wait for the cooldown duration to expire, then trigger scheduling."""
        await asyncio.sleep(duration)
        logger.info("[ProviderManager] Cooldown for '%s' expired. Re-evaluating schedule queue.", provider_name)
        async with self._lock:
            h = self._health.get(provider_name)
            if h and not h.is_healthy:
                h.is_healthy = True
                h.consecutive_failures = 0
            self._schedule_next()
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Config File loader
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load weights and policies configuration from provider_config.json."""
        if PROVIDER_CONFIG_PATH.exists():
            try:
                with open(PROVIDER_CONFIG_PATH, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
            except Exception as exc:
                logger.error("[ProviderManager] Failed to read provider_config.json: %s", exc)
                self._config = self._get_default_config()
        else:
            self._config = self._get_default_config()
            try:
                # Write default configuration to disk
                PROVIDER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(PROVIDER_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(self._config, f, indent=2)
            except Exception as exc:
                logger.warning("[ProviderManager] Failed to write default provider_config.json: %s", exc)

        # Extract limits
        self._concurrency_limits = self._config.get("concurrency_limits", {"nvidia": 3, "groq": 2})
        self._cooldown_durations = self._config.get("cooldown_durations", {"nvidia": 30.0, "groq": 60.0})
        self._retry_policies = self._config.get("retry_policies", {"nvidia": {"max_retries": 3}})

    def _get_default_config(self) -> dict[str, Any]:
        """Built-in default config dictionary for NVIDIA & Groq fallback."""
        return {
          "concurrency_limits": {
            "nvidia": 3,
            "groq": 2
          },
          "cooldown_durations": {
            "nvidia": 30.0,
            "groq": 60.0
          },
          "retry_policies": {
            "nvidia": { "max_retries": 3, "initial_backoff": 2.0, "backoff_factor": 2.0 },
            "groq": { "max_retries": 3, "initial_backoff": 2.0, "backoff_factor": 2.0 }
          },
          "model_preferences": {
            "extraction": [
              { "provider": "nvidia", "model": "deepseek-ai/deepseek-v4-flash", "weight": 90 },
              { "provider": "nvidia", "model": "z-ai/glm-5.2", "weight": 70 },
              { "provider": "groq", "model": "openai/gpt-oss-120b", "weight": 60 }
            ],
            "synthesis": [
              { "provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b", "weight": 95 },
              { "provider": "groq", "model": "openai/gpt-oss-120b", "weight": 80 }
            ],
            "translation": [
              { "provider": "nvidia", "model": "z-ai/glm-5.2", "weight": 90 },
              { "provider": "groq", "model": "openai/gpt-oss-120b", "weight": 70 }
            ]
          }
        }
