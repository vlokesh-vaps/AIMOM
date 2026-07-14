"""Stage 5 — Validation Layer.

Post-merge validation and auto-repair for the MeetingSummary.
Pure Python — no LLM calls. Checks completeness, consistency,
and schema conformance, auto-repairing trivially broken fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from pydantic import ValidationError

from ai.models.chunk import ChunkExtraction
from ai.models.meeting import MeetingSummary
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_PRIORITIES = {"High", "Medium", "Low"}
_VALID_SENTIMENTS = {"Positive", "Neutral", "Negative", "Mixed"}
_VALID_DP_STATUSES = {"Open", "In Progress", "Completed", "Pending"}
_VALID_AI_STATUSES = {"Pending", "Completed", "In Progress", "Information"}


@dataclass
class ValidationResult:
    """Output of the validation layer."""

    summary: MeetingSummary
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True


class ValidationLayer:
    """Stage 5 — validate and auto-repair the merged MeetingSummary."""

    def validate_chunk(self, data: dict) -> ChunkExtraction:
        """Validate one Stage-3 chunk extraction."""
        try:
            return ChunkExtraction(**data)
        except ValidationError:
            logger.warning("[ValidationLayer] Chunk extraction failed schema validation.")
            raise

    def validate(self, summary: MeetingSummary) -> ValidationResult:
        """Run all validation checks and return a repaired result.

        Args:
            summary: The merged MeetingSummary from Stage 4.

        Returns:
            ValidationResult with the (possibly repaired) summary,
            a list of warnings, and a validity flag.
        """
        warnings: List[str] = []

        # ── Schema conformance (re-validate Pydantic) ──
        # This should always pass since we built via Pydantic, but catch edge cases.
        try:
            summary = MeetingSummary(**summary.model_dump())
        except Exception as e:
            warnings.append(f"Re-validation failed: {e}")

        # ── Metadata completeness ──
        summary, w = self._check_metadata(summary)
        warnings.extend(w)

        # ── Discussion point completeness ──
        summary, w = self._check_discussion_points(summary)
        warnings.extend(w)

        # ── Action item completeness ──
        summary, w = self._check_action_items(summary)
        warnings.extend(w)

        # ── Participant consistency ──
        summary, w = self._check_participant_consistency(summary)
        warnings.extend(w)

        # ── Enum value consistency ──
        summary, w = self._check_enum_values(summary)
        warnings.extend(w)

        is_valid = len(warnings) == 0

        if warnings:
            logger.warning(
                "[ValidationLayer] %d warnings found. Auto-repaired where possible.",
                len(warnings),
            )
            for w_msg in warnings:
                logger.warning("[ValidationLayer]   • %s", w_msg)
        else:
            logger.info("[ValidationLayer] All checks passed — summary is valid.")

        return ValidationResult(
            summary=summary,
            warnings=warnings,
            is_valid=is_valid,
        )

    # ── Check helpers ────────────────────────────────────────────────────

    @staticmethod
    def _check_metadata(summary: MeetingSummary) -> tuple[MeetingSummary, List[str]]:
        """Check and repair metadata fields."""
        warnings = []

        if not summary.meeting_title or not summary.meeting_title.strip():
            summary.meeting_title = "Untitled Meeting"
            warnings.append("meeting_title was empty — set to 'Untitled Meeting'.")

        if not summary.executive_summary or not summary.executive_summary.strip():
            summary.executive_summary = "No executive summary generated."
            warnings.append("executive_summary was empty — set to placeholder.")

        if not summary.generated_at or not summary.generated_at.strip():
            summary.generated_at = datetime.now().isoformat()
            warnings.append("generated_at was empty — set to current timestamp.")

        if not summary.meeting_type or not summary.meeting_type.strip():
            summary.meeting_type = "General"
            warnings.append("meeting_type was empty — set to 'General'.")

        if summary.overall_sentiment not in _VALID_SENTIMENTS:
            warnings.append(
                f"overall_sentiment '{summary.overall_sentiment}' is invalid — set to 'Neutral'."
            )
            summary.overall_sentiment = "Neutral"

        return summary, warnings

    @staticmethod
    def _check_discussion_points(summary: MeetingSummary) -> tuple[MeetingSummary, List[str]]:
        """Check that every discussion point has required fields."""
        warnings = []

        for i, dp in enumerate(summary.discussion_points):
            if not dp.point or not dp.point.strip():
                dp.point = f"Discussion Point {i + 1}"
                warnings.append(f"discussion_points[{i}].point was empty — auto-filled.")

            if not dp.detailed_summary or not dp.detailed_summary.strip():
                dp.detailed_summary = dp.point
                warnings.append(
                    f"discussion_points[{i}].detailed_summary was empty — copied from point."
                )

        return summary, warnings

    @staticmethod
    def _check_action_items(summary: MeetingSummary) -> tuple[MeetingSummary, List[str]]:
        """Check that every action item has a non-empty task."""
        warnings = []

        valid_items = []
        for i, ai in enumerate(summary.action_items):
            if not ai.task or not ai.task.strip():
                warnings.append(
                    f"action_items[{i}].task was empty — removed from list."
                )
                continue
            valid_items.append(ai)

        summary.action_items = valid_items
        return summary, warnings

    @staticmethod
    def _check_participant_consistency(summary: MeetingSummary) -> tuple[MeetingSummary, List[str]]:
        """Check that action-item owners appear in participants or attendees."""
        warnings = []

        known_people = set()
        for p in summary.participants:
            known_people.add(p.lower().strip())
        for a in summary.attendees:
            known_people.add(a.lower().strip())

        if not summary.participants:
            warnings.append("participants list is empty — no speakers detected.")

        attendee_names = {a.lower().strip() for a in summary.attendees}

        for ai in summary.action_items:
            if ai.owner and ai.owner.strip():
                owner_lower = ai.owner.lower().strip()
                if summary.attendees:
                    if owner_lower not in attendee_names:
                        warnings.append(f"Action item owner '{ai.owner}' not in attendees list. Set to 'To Be Assigned'.")
                        ai.owner = "To Be Assigned"
                else:
                    if owner_lower not in known_people:
                        summary.participants.append(ai.owner.strip())
                        known_people.add(owner_lower)
                        warnings.append(f"Action item owner '{ai.owner}' was not in participants — auto-added.")

        for dp in summary.discussion_points:
            if (
                dp.assigned_to
                and dp.assigned_to not in ("Not Specified", "")
                and dp.assigned_to.lower().strip() not in known_people
            ):
                summary.participants.append(dp.assigned_to.strip())
                known_people.add(dp.assigned_to.lower().strip())
                warnings.append(
                    f"Discussion assignee '{dp.assigned_to}' was not in participants — auto-added."
                )

        return summary, warnings

    @staticmethod
    def _check_enum_values(summary: MeetingSummary) -> tuple[MeetingSummary, List[str]]:
        """Validate and repair enum-like fields (priority, status)."""
        warnings = []

        for i, dp in enumerate(summary.discussion_points):
            if dp.priority not in _VALID_PRIORITIES:
                warnings.append(
                    f"discussion_points[{i}].priority '{dp.priority}' invalid — set to 'Medium'."
                )
                dp.priority = "Medium"

            if dp.status not in _VALID_DP_STATUSES:
                warnings.append(
                    f"discussion_points[{i}].status '{dp.status}' invalid — set to 'Open'."
                )
                dp.status = "Open"

        for i, ai in enumerate(summary.action_items):
            if ai.priority not in _VALID_PRIORITIES:
                warnings.append(
                    f"action_items[{i}].priority '{ai.priority}' invalid — set to 'Medium'."
                )
                ai.priority = "Medium"

            if ai.status not in _VALID_AI_STATUSES:
                warnings.append(
                    f"action_items[{i}].status '{ai.status}' invalid — set to 'Pending'."
                )
                ai.status = "Pending"

        return summary, warnings
