"""Minimal per-chunk schemas for Stage 3 extraction.

These models intentionally capture facts only. Report prose is generated after
deterministic merging, never during chunk extraction.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_optional(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class ChunkActionItem(BaseModel):
    """One action item extracted from a transcript chunk."""

    model_config = ConfigDict(extra="ignore")

    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    topic: Optional[str] = None
    agenda_item: Optional[str] = None
    authority_context: Optional[str] = None
    tone_and_consequence: Optional[str] = None

    @field_validator("task", "owner", "deadline", "topic", "agenda_item", "authority_context", "tone_and_consequence", mode="before")
    @classmethod
    def clean_text(cls, value: object) -> Optional[str]:
        return _clean_optional(value)


class ChunkDiscussionPoint(BaseModel):
    """One factual discussion point extracted from a transcript chunk."""

    model_config = ConfigDict(extra="ignore")

    topic: str
    details: str
    speakers: List[str] = Field(default_factory=list)
    agenda_item: Optional[str] = None
    authority_context: Optional[str] = None
    tone_and_consequence: Optional[str] = None
    cross_topic_context: Optional[str] = None
    implicit_decision: Optional[str] = None

    @field_validator("topic", "details", "agenda_item", "authority_context", "tone_and_consequence", "cross_topic_context", "implicit_decision", mode="before")
    @classmethod
    def clean_text(cls, value: object) -> Optional[str]:
        return _clean_optional(value)


class ChunkExtraction(BaseModel):
    """Structured extraction from a single transcript chunk."""

    model_config = ConfigDict(extra="ignore")

    discussion_points: List[ChunkDiscussionPoint] = Field(default_factory=list)
    action_items: List[ChunkActionItem] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    deadlines: List[str] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    cross_topic_context: List[str] = Field(default_factory=list)
    implicit_decisions: List[str] = Field(default_factory=list)
    tone_and_consequences: List[str] = Field(default_factory=list)

    @field_validator(
        "decisions",
        "risks",
        "blockers",
        "questions",
        "deadlines",
        "participants",
        "cross_topic_context",
        "implicit_decisions",
        "tone_and_consequences",
        mode="before",
    )
    @classmethod
    def clean_string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        cleaned = []
        for item in value:
            text = _clean_optional(item)
            if text:
                cleaned.append(text)
        return cleaned
