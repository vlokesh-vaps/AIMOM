"""Stage 6 - Ollama final reviewer.

The reviewer is a copy-editing pass only. If it drops facts or returns invalid
JSON, the validated merged summary is kept unchanged.
"""

from __future__ import annotations

import json
import re

from ai.prompting.templates import FINAL_REVIEW_SYSTEM_PROMPT
from ai.providers.base import AIProviderError, BaseAIProvider
from ai.models.meeting import MeetingSummary
from utils.logger import get_logger

logger = get_logger(__name__)


class FinalReviewer:
    """Polish final meeting minutes JSON with local Ollama."""

    def __init__(self, provider: BaseAIProvider, retry_fn) -> None:
        self._provider = provider
        self._retry_fn = retry_fn

    def review(self, summary: MeetingSummary) -> MeetingSummary:
        if not self._provider.is_configured():
            logger.warning("[FinalReviewer] Ollama unavailable. Skipping final review.")
            return summary

        summary_json = json.dumps(summary.model_dump(), indent=2, ensure_ascii=False)
        user_prompt = (
            "Review this meeting minutes JSON. Preserve all facts and keys.\n\n"
            f"{summary_json}\n\n"
            "Return only the improved JSON."
        )

        try:
            raw_response = self._retry_fn(
                provider=self._provider,
                system_prompt=FINAL_REVIEW_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=4096,
            )
            reviewed = self._parse_reviewed_summary(raw_response)
            if not self._preserves_required_facts(original=summary, reviewed=reviewed):
                return summary
            reviewed.generated_at = summary.generated_at
            return reviewed
        except AIProviderError as exc:
            logger.warning("[FinalReviewer] Ollama review failed: %s. Keeping original.", exc)
            return summary
        except Exception as exc:
            logger.warning("[FinalReviewer] Unexpected review error: %s. Keeping original.", exc)
            return summary

    @staticmethod
    def _parse_reviewed_summary(raw_response: str) -> MeetingSummary:
        cleaned = raw_response.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return MeetingSummary(**json.loads(cleaned))

    @staticmethod
    def _preserves_required_facts(original: MeetingSummary, reviewed: MeetingSummary) -> bool:
        checks = {
            "discussion_points": len(reviewed.discussion_points) >= len(original.discussion_points),
            "action_items": len(reviewed.action_items) >= len(original.action_items),
            "decisions": len(reviewed.decisions) >= len(original.decisions),
            "risks": len(reviewed.risks) >= len(original.risks),
            "questions": len(reviewed.questions) >= len(original.questions),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            logger.warning("[FinalReviewer] Review removed required facts from: %s", ", ".join(failed))
            return False
        return True
