"""Provider-aware request scheduler for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderStats:
    """Sliding-window provider usage statistics."""

    requests: int = 0
    tokens: int = 0
    rate_limit_hits: int = 0
    failures: int = 0
    window_started_at: float = field(default_factory=time.monotonic)


class ProviderRequestScheduler:
    """Throttle provider calls by RPM, TPM, and concurrency.

    The current pipeline is sequential, but concurrency accounting keeps this
    reusable for future parallel chunk extraction.
    """

    def __init__(
        self,
        provider_name: str,
        rpm_limit: int = 60,
        tpm_limit: int = 12_000,
        max_concurrent: int = 1,
        safety_margin: float = 0.85,
    ) -> None:
        self.provider_name = provider_name
        self.rpm_limit = max(1, rpm_limit)
        self.tpm_limit = max(1, tpm_limit)
        self.max_concurrent = max(1, max_concurrent)
        self.safety_margin = max(0.1, min(safety_margin, 1.0))
        self.stats = ProviderStats()
        self._lock = threading.Lock()
        self._in_flight = 0

    def acquire(self, tokens_needed: int) -> None:
        """Wait until a request can be sent within provider limits."""
        tokens_needed = max(1, tokens_needed)
        while True:
            with self._lock:
                self._reset_window_if_needed()
                rpm_budget = int(self.rpm_limit * self.safety_margin)
                tpm_budget = int(self.tpm_limit * self.safety_margin)
                can_send = (
                    self._in_flight < self.max_concurrent
                    and self.stats.requests + 1 <= rpm_budget
                    and self.stats.tokens + tokens_needed <= tpm_budget
                )
                if can_send:
                    self._in_flight += 1
                    self.stats.requests += 1
                    self.stats.tokens += tokens_needed
                    return

                elapsed = time.monotonic() - self.stats.window_started_at
                window_wait = max(1.0, 60.0 - elapsed)
                pressure = max(
                    self.stats.requests / max(rpm_budget, 1),
                    self.stats.tokens / max(tpm_budget, 1),
                )
                wait = min(window_wait, max(1.0, 2.0 ** min(int(pressure * 3), 5)))

            logger.info(
                "[%s] Scheduler delaying request %.1fs (rpm=%d/%d, tpm=%d/%d, active=%d/%d).",
                self.provider_name,
                wait,
                self.stats.requests,
                rpm_budget,
                self.stats.tokens,
                tpm_budget,
                self._in_flight,
                self.max_concurrent,
            )
            time.sleep(wait)

    def release(self, failed: bool = False, rate_limited: bool = False) -> None:
        """Release one in-flight slot and record request outcome."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if failed:
                self.stats.failures += 1
            if rate_limited:
                self.stats.rate_limit_hits += 1

    def _reset_window_if_needed(self) -> None:
        if time.monotonic() - self.stats.window_started_at >= 60.0:
            self.stats.requests = 0
            self.stats.tokens = 0
            self.stats.window_started_at = time.monotonic()
