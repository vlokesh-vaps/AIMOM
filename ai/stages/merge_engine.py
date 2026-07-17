"""Stage — Incremental Python Merge Engine.

Combines structured JSON outputs from parallel chunk extraction workers into a
single consolidated structure. Performs advanced normalization, deduplication,
and field validation — all in pure Python with no LLM calls.

Designed to accept chunk results incrementally (as they arrive) to reduce
peak memory usage and simplify recovery after interruptions.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


class MergeEngine:
    """Incrementally merges chunk extraction results into a unified structure.

    Usage::

        engine = MergeEngine()
        engine.add_chunk(chunk_0_data, chunk_index=0)
        engine.add_chunk(chunk_1_data, chunk_index=1)
        ...
        merged = engine.finalize()
    """

    def __init__(self) -> None:
        self._chunks: list[tuple[int, dict[str, Any]]] = []  # (index, data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunk(self, data: dict[str, Any], chunk_index: int) -> None:
        """Add a chunk's extraction result for later merging.

        Args:
            data: Structured JSON output from the extraction agent.
            chunk_index: The chronological index of this chunk.
        """
        self._chunks.append((chunk_index, data))
        logger.debug(
            "[MergeEngine] Added chunk %d (%d topics, %d actions).",
            chunk_index,
            len(data.get("topics", [])),
            len(data.get("action_items", [])),
        )

    def finalize(self) -> dict[str, Any]:
        """Merge all added chunks into a single consolidated structure.

        Returns:
            A dict with keys: topics, discussion_points, action_items,
            decisions, risks, open_questions.
        """
        # Sort by chunk index to preserve chronological order
        self._chunks.sort(key=lambda c: c[0])

        merged_topics: list[str] = []
        merged_discussions: list[dict] = []
        merged_actions: list[dict] = []
        merged_decisions: list[str] = []
        merged_risks: list[str] = []
        merged_questions: list[str] = []

        for idx, data in self._chunks:
            # --- Topics / Agenda ---
            for topic in data.get("topics", []):
                if isinstance(topic, dict):
                    topic_name = str(topic.get("agenda_item") or topic.get("topic", ""))
                elif isinstance(topic, str):
                    topic_name = topic
                else:
                    continue
                topic_name = self._normalize_text(topic_name)
                if topic_name and not self._is_duplicate_string(topic_name, merged_topics):
                    merged_topics.append(topic_name)

            # --- Discussion points ---
            for disc in data.get("discussion_points", []):
                if not isinstance(disc, dict):
                    continue
                disc = self._normalize_discussion(disc)
                if not self._is_duplicate_discussion(disc, merged_discussions):
                    merged_discussions.append(disc)

            # --- Action items ---
            for action in data.get("action_items", []):
                if not isinstance(action, dict):
                    continue
                action = self._normalize_action(action)
                if action.get("task"):
                    merged_actions.append(action)

            # --- Decisions ---
            for decision in data.get("decisions", []):
                text = self._normalize_text(str(decision) if not isinstance(decision, str) else decision)
                if text and not self._is_duplicate_string(text, merged_decisions):
                    merged_decisions.append(text)

            # --- Risks ---
            for risk in data.get("risks", []):
                text = self._normalize_text(str(risk) if not isinstance(risk, str) else risk)
                if text and not self._is_duplicate_string(text, merged_risks):
                    merged_risks.append(text)

            # --- Open questions ---
            for q in data.get("open_questions", []):
                text = self._normalize_text(str(q) if not isinstance(q, str) else q)
                if text and not self._is_duplicate_string(text, merged_questions):
                    merged_questions.append(text)

        # Final deduplication pass on action items
        merged_actions = self._deduplicate_actions(merged_actions)

        # Validate missing fields
        merged_discussions = self._validate_discussions(merged_discussions)
        merged_actions = self._validate_actions(merged_actions)

        logger.info(
            "[MergeEngine] Merged %d chunks → %d topics, %d discussions, "
            "%d actions, %d decisions, %d risks, %d open questions.",
            len(self._chunks), len(merged_topics), len(merged_discussions),
            len(merged_actions), len(merged_decisions), len(merged_risks),
            len(merged_questions),
        )

        return {
            "topics": merged_topics,
            "discussion_points": merged_discussions,
            "action_items": merged_actions,
            "decisions": merged_decisions,
            "risks": merged_risks,
            "open_questions": merged_questions,
        }

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace and strip a string."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize participant/assignee names to title-case."""
        name = name.strip()
        if not name or name.lower() in ("not specified", "n/a", "none", "tbd", ""):
            return ""
        return name.title()

    @staticmethod
    def _normalize_priority(priority: str) -> str:
        """Normalize priority to High/Medium/Low."""
        p = priority.strip().lower()
        if p in ("high", "urgent", "critical"):
            return "High"
        if p in ("low", "minor"):
            return "Low"
        return "Medium"

    @staticmethod
    def _normalize_date(date: str) -> str:
        """Normalize date strings — pass through valid formats, empty otherwise."""
        date = date.strip()
        if not date or date.lower() in ("not specified", "n/a", "none", "tbd", ""):
            return ""
        return date

    def _normalize_discussion(self, disc: dict) -> dict:
        """Normalize a single discussion point dict."""
        disc["agenda_item"] = self._normalize_text(
            str(disc.get("agenda_item", "Off Agenda Discussion"))
        )
        disc["point"] = self._normalize_text(str(disc.get("point", "")))
        disc["detailed_summary"] = self._normalize_text(
            str(disc.get("detailed_summary", ""))
        )
        disc["decision"] = self._normalize_text(
            str(disc.get("decision", "No Decision Taken"))
        )
        return disc

    def _normalize_action(self, action: dict) -> dict:
        """Normalize a single action item dict."""
        action["task"] = self._normalize_text(str(action.get("task", "")))
        action["owner"] = self._normalize_name(str(action.get("owner", "")))
        action["target_date"] = self._normalize_date(
            str(action.get("target_date", action.get("deadline", "")))
        )
        action["priority"] = self._normalize_priority(
            str(action.get("priority", "Medium"))
        )
        action["agenda_item"] = self._normalize_text(
            str(action.get("agenda_item", "Off Agenda Discussion"))
        )
        return action

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_duplicate_string(text: str, existing: list[str], threshold: float = 0.85) -> bool:
        """Check if text is a near-duplicate of any existing string."""
        for ex in existing:
            if SequenceMatcher(None, text.lower(), ex.lower()).ratio() > threshold:
                return True
        return False

    @staticmethod
    def _is_duplicate_discussion(disc: dict, existing: list[dict], threshold: float = 0.80) -> bool:
        """Check if a discussion point is a near-duplicate."""
        point = disc.get("point", "").lower()
        summary = disc.get("detailed_summary", "").lower()[:200]
        for ex in existing:
            ex_point = ex.get("point", "").lower()
            ex_summary = ex.get("detailed_summary", "").lower()[:200]
            if (
                SequenceMatcher(None, point, ex_point).ratio() > threshold
                and SequenceMatcher(None, summary, ex_summary).ratio() > threshold
            ):
                # Merge: keep the longer summary
                if len(disc.get("detailed_summary", "")) > len(ex.get("detailed_summary", "")):
                    ex["detailed_summary"] = disc["detailed_summary"]
                return True
        return False

    @staticmethod
    def _deduplicate_actions(actions: list[dict]) -> list[dict]:
        """Deduplicate action items based on task similarity."""
        deduped: list[dict] = []
        for action in actions:
            task = str(action.get("task", "")).strip()
            if not task:
                continue

            owner = str(action.get("owner", "")).strip()
            target_date = str(action.get("target_date", "")).strip()

            is_duplicate = False
            for existing in deduped:
                existing_task = str(existing.get("task", "")).strip()
                existing_owner = str(existing.get("owner", "")).strip()

                similarity = SequenceMatcher(None, task.lower(), existing_task.lower()).ratio()
                if similarity > 0.85 or (task == existing_task and owner == existing_owner):
                    is_duplicate = True
                    # Enrich the existing entry with any missing info
                    if not existing_owner and owner:
                        existing["owner"] = owner
                    if not existing.get("target_date") and target_date:
                        existing["target_date"] = target_date
                    if len(task) > len(existing_task):
                        existing["task"] = task
                    break

            if not is_duplicate:
                deduped.append(action)

        return deduped

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_discussions(discussions: list[dict]) -> list[dict]:
        """Ensure every discussion has required fields with fallback values."""
        for d in discussions:
            d.setdefault("point", "Untitled discussion")
            d.setdefault("detailed_summary", "")
            d.setdefault("decision", "No Decision Taken")
            d.setdefault("agenda_item", "Off Agenda Discussion")
            d.setdefault("status", "Open")
        return discussions

    @staticmethod
    def _validate_actions(actions: list[dict]) -> list[dict]:
        """Ensure every action has required fields with fallback values."""
        for a in actions:
            a.setdefault("task", "")
            a.setdefault("owner", "")
            a.setdefault("target_date", "")
            a.setdefault("priority", "Medium")
            a.setdefault("status", "Pending")
            a.setdefault("agenda_item", "Off Agenda Discussion")
        return [a for a in actions if a.get("task")]
