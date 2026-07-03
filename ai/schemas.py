"""Pydantic schemas for structured AI Meeting Minutes output."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """Pydantic model representing an extracted action item."""

    task: str = Field(
        ...,
        description="The detailed description of the action item task."
    )
    owner: str = Field(
        "Unknown",
        description="Only the person assigned to complete/receive this task. Empty if no assignee is named."
    )
    target_date: str = Field(
        "",
        description="The due date / target date for this task (format: YYYY-MM-DD or empty if none detected)."
    )
    priority: str = Field(
        "Medium",
        description="Priority of the task: High, Medium, or Low."
    )
    status: str = Field(
        "Pending",
        description="Status of the task: Pending, Completed, In Progress, or Information."
    )
    notes: str = Field(
        "",
        description="Additional context, constraints, or notes regarding the task."
    )


class MeetingSummary(BaseModel):
    """Pydantic model representing the full analyzed meeting intelligence output."""

    meeting_title: str = Field(
        ...,
        description="Title of the meeting."
    )
    executive_summary: str = Field(
        ...,
        description="A concise professional summary of the meeting."
    )
    meeting_type: str = Field(
        "General",
        description="Type of the meeting (e.g. Status Update, Design Review, Standup, Retrospective)."
    )
    overall_sentiment: str = Field(
        "Neutral",
        description="Overall sentiment of the meeting: Positive, Neutral, Negative, or Mixed."
    )
    topics: List[str] = Field(
        default_factory=list,
        description="List of key discussion topics detected."
    )
    decisions: List[str] = Field(
        default_factory=list,
        description="Extract all decisions made during the meeting."
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Identify project risks, blockers, dependencies, or unresolved issues."
    )
    questions: List[str] = Field(
        default_factory=list,
        description="List of questions raised during the meeting."
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="List of extracted action items/tasks."
    )
    participants: List[str] = Field(
        default_factory=list,
        description="Extract speaker names. If names are unavailable, use Speaker 1, Speaker 2, etc."
    )
    timeline: List[str] = Field(
        default_factory=list,
        description="Chronological meeting events or flow of discussion."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Important keywords or tags relating to the meeting."
    )
    followups: List[str] = Field(
        default_factory=list,
        description="List of generated follow-up activities."
    )
    meeting_duration: str = Field(
        "Unknown",
        description="Inferred or specified meeting duration (e.g., '30 minutes')."
    )
    generated_at: str = Field(
        ...,
        description="ISO timestamp of when the summary was generated."
    )
