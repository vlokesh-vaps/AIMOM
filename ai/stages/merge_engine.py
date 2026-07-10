"""Stage 4 - deterministic Python merge engine."""

from __future__ import annotations

import difflib
from datetime import datetime
from typing import List

from ai.models.chunk import ChunkActionItem, ChunkExtraction
from ai.models.meeting import ActionItem, DiscussionPoint, MeetingSummary
from utils.logger import get_logger

logger = get_logger(__name__)

_SIMILARITY_THRESHOLD = 0.80


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _dedup_strings(items: List[str], threshold: float = _SIMILARITY_THRESHOLD) -> List[str]:
    result: List[str] = []
    for item in items:
        text = (item or "").strip()
        if not text:
            continue
        duplicate_index = None
        for i, existing in enumerate(result):
            if _similarity(text, existing) >= threshold:
                duplicate_index = i
                break
        if duplicate_index is None:
            result.append(text)
        elif len(text) > len(result[duplicate_index]):
            result[duplicate_index] = text
    return result


def _join_unique(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    if b in a:
        return a
    if a in b:
        return b
    return f"{a}; {b}"


class MergeEngine:
    """Combine chunk extractions without LLM calls."""

    def merge(
        self,
        extractions: List[ChunkExtraction],
        title: str,
        date: str,
        attendees: str | None = None,
    ) -> MeetingSummary:
        if not extractions:
            return MeetingSummary(
                meeting_title=title,
                executive_summary="No content extracted.",
                generated_at=datetime.now().isoformat(),
            )

        discussion_points = self._merge_discussion_points(extractions)
        action_items = self._merge_action_items(extractions)
        decisions = _dedup_strings(self._flatten(extractions, "decisions"))
        risks = _dedup_strings(self._flatten(extractions, "risks") + self._flatten(extractions, "blockers"))
        questions = _dedup_strings(self._flatten(extractions, "questions"))
        deadlines = _dedup_strings(self._flatten(extractions, "deadlines"))
        participants = list(dict.fromkeys(self._flatten(extractions, "participants")))
        topics = _dedup_strings([dp.point for dp in discussion_points])

        attendees_list = [a.strip() for a in attendees.split(",") if a.strip()] if attendees else []

        summary = MeetingSummary(
            meeting_title=title,
            executive_summary=self._build_executive_summary(title, discussion_points, action_items, decisions),
            meeting_type="General",
            overall_sentiment="Neutral",
            topics=topics,
            decisions=decisions,
            risks=risks,
            questions=questions,
            action_items=action_items,
            discussion_points=discussion_points,
            participants=participants,
            attendees=attendees_list,
            pending_items=questions,
            parking_lot=[],
            timeline=topics,
            keywords=[],
            followups=_dedup_strings([item.task for item in action_items]),
            meeting_duration="Unknown",
            generated_at=datetime.now().isoformat(),
        )
        if deadlines:
            summary.pending_items = _dedup_strings(summary.pending_items + [f"Deadline noted: {d}" for d in deadlines])

        logger.info(
            "[MergeEngine] Merged %d chunks into %d discussions, %d actions, %d decisions, %d risks.",
            len(extractions),
            len(discussion_points),
            len(action_items),
            len(decisions),
            len(risks),
        )
        return summary

    def _merge_discussion_points(self, extractions: List[ChunkExtraction]) -> List[DiscussionPoint]:
        merged: List[DiscussionPoint] = []
        for ext in extractions:
            for chunk_point in ext.discussion_points:
                point = DiscussionPoint(
                    point=chunk_point.topic,
                    detailed_summary=chunk_point.details,
                    decision=self._related_decision(chunk_point.topic, ext.decisions),
                    task=self._related_action(chunk_point.topic, ext.action_items),
                    assigned_to=self._related_owner(chunk_point.topic, ext.action_items),
                    deadline=self._related_deadline(chunk_point.topic, ext.action_items, ext.deadlines),
                    risks_or_concerns=self._related_text(chunk_point.topic, ext.risks + ext.blockers),
                    notes=", ".join(chunk_point.speakers),
                )
                existing_index = self._find_similar_discussion(point, merged)
                if existing_index is None:
                    merged.append(point)
                else:
                    merged[existing_index] = self._merge_two_discussions(merged[existing_index], point)
        return merged

    def _merge_action_items(self, extractions: List[ChunkExtraction]) -> List[ActionItem]:
        merged: List[ActionItem] = []
        for ext in extractions:
            for chunk_action in ext.action_items:
                item = ActionItem(
                    task=chunk_action.task,
                    owner=chunk_action.owner or "",
                    target_date=chunk_action.deadline or "",
                    notes=chunk_action.topic or "",
                )
                existing_index = self._find_similar_action(item, merged)
                if existing_index is None:
                    merged.append(item)
                else:
                    merged[existing_index] = self._merge_two_actions(merged[existing_index], item)
        return merged

    @staticmethod
    def _find_similar_discussion(point: DiscussionPoint, existing: List[DiscussionPoint]) -> int | None:
        for i, item in enumerate(existing):
            if _similarity(point.point, item.point) >= _SIMILARITY_THRESHOLD:
                return i
        return None

    @staticmethod
    def _merge_two_discussions(a: DiscussionPoint, b: DiscussionPoint) -> DiscussionPoint:
        return DiscussionPoint(
            point=a.point if len(a.point) >= len(b.point) else b.point,
            detailed_summary=_join_unique(a.detailed_summary, b.detailed_summary),
            decision=b.decision if a.decision == "No Decision Taken" and b.decision != "No Decision Taken" else a.decision,
            task=b.task if a.task == "No Action Item" and b.task != "No Action Item" else a.task,
            assigned_to=b.assigned_to if a.assigned_to == "Not Specified" and b.assigned_to != "Not Specified" else a.assigned_to,
            deadline=b.deadline if a.deadline == "Not Specified" and b.deadline != "Not Specified" else a.deadline,
            risks_or_concerns=_join_unique(a.risks_or_concerns, b.risks_or_concerns),
            notes=_join_unique(a.notes, b.notes),
        )

    @staticmethod
    def _find_similar_action(item: ActionItem, existing: List[ActionItem]) -> int | None:
        for i, candidate in enumerate(existing):
            if _similarity(item.task, candidate.task) < _SIMILARITY_THRESHOLD:
                continue
            if item.owner and candidate.owner and _similarity(item.owner, candidate.owner) < 0.5:
                continue
            return i
        return None

    @staticmethod
    def _merge_two_actions(a: ActionItem, b: ActionItem) -> ActionItem:
        return ActionItem(
            task=a.task if len(a.task) >= len(b.task) else b.task,
            owner=a.owner or b.owner,
            target_date=a.target_date or b.target_date,
            priority=b.priority if a.priority == "Medium" and b.priority != "Medium" else a.priority,
            status=b.status if a.status == "Pending" and b.status != "Pending" else a.status,
            notes=_join_unique(a.notes, b.notes),
        )

    @staticmethod
    def _flatten(extractions: List[ChunkExtraction], field: str) -> List[str]:
        values: List[str] = []
        for extraction in extractions:
            values.extend(getattr(extraction, field, []))
        return [value for value in values if value]

    @staticmethod
    def _related_decision(topic: str, decisions: List[str]) -> str:
        related = [d for d in decisions if _similarity(topic, d) >= 0.35 or topic.lower() in d.lower()]
        return related[0] if related else "No Decision Taken"

    @staticmethod
    def _related_text(topic: str, items: List[str]) -> str:
        related = [i for i in items if _similarity(topic, i) >= 0.35 or topic.lower() in i.lower()]
        return "; ".join(_dedup_strings(related))

    @staticmethod
    def _related_action(topic: str, actions: List[ChunkActionItem]) -> str:
        for action in actions:
            if action.topic and _similarity(topic, action.topic) >= 0.60:
                return action.task
        return "No Action Item"

    @staticmethod
    def _related_owner(topic: str, actions: List[ChunkActionItem]) -> str:
        for action in actions:
            if action.topic and action.owner and _similarity(topic, action.topic) >= 0.60:
                return action.owner
        return "Not Specified"

    @staticmethod
    def _related_deadline(topic: str, actions: List[ChunkActionItem], deadlines: List[str]) -> str:
        for action in actions:
            if action.topic and action.deadline and _similarity(topic, action.topic) >= 0.60:
                return action.deadline
        return deadlines[0] if deadlines else "Not Specified"

    @staticmethod
    def _build_executive_summary(
        title: str,
        discussions: List[DiscussionPoint],
        actions: List[ActionItem],
        decisions: List[str],
    ) -> str:
        parts = [f"Meeting: {title}."]
        if discussions:
            parts.append(f"Key topics discussed: {'; '.join(d.point for d in discussions[:5])}.")
        if decisions:
            parts.append(f"{len(decisions)} decision(s) recorded.")
        if actions:
            parts.append(f"{len(actions)} action item(s) identified.")
        return " ".join(parts)
