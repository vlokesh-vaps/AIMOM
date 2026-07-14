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
        "",
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
    agenda_item: str = Field(
        "Off Agenda Discussion",
        description=(
            "The agenda item this task belongs to. Defaults to "
            "'Off Agenda Discussion' when no explicit agenda match exists."
        )
    )
    authority_context: str = Field(
        "",
        description="Who requested or authorized this action and the authority context stated in the transcript."
    )
    tone_and_consequence: str = Field(
        "",
        description="Tone, urgency, warning, or consequence attached to this action, if explicitly stated."
    )


class DiscussionPoint(BaseModel):
    """Pydantic model representing one distinct discussion point in the meeting.

    Every discussion point — however small — must be captured with all 12 fields.
    If a field was not mentioned during the meeting, use the explicit fallback value
    documented in each field description.
    """

    point: str = Field(
        ...,
        description="Short heading / title of the discussion topic."
    )
    detailed_summary: str = Field(
        ...,
        description=(
            "Full narrative of everything discussed under this topic: ideas, "
            "questions asked, answers given, clarifications, approvals, disagreements, "
            "and each contributor's contribution. Must be comprehensive — do not omit details."
        )
    )
    decision: str = Field(
        "No Decision Taken",
        description="Decision made during this discussion, or 'No Decision Taken' if none."
    )
    task: str = Field(
        "No Action Item",
        description="Specific action item arising from this discussion, or 'No Action Item' if none."
    )
    assigned_to: str = Field(
        "Not Specified",
        description=(
            "Person responsible for the task. Only the assignee/receiver, never the requester. "
            "Use 'Not Specified' if no assignee was named."
        )
    )
    deadline: str = Field(
        "Not Specified",
        description="Deadline for the task in YYYY-MM-DD format, or 'Not Specified' if none mentioned."
    )
    priority: str = Field(
        "Medium",
        description="Priority of this action item: High, Medium, or Low."
    )
    status: str = Field(
        "Open",
        description="Status: Open, In Progress, Completed, or Pending."
    )
    risks_or_concerns: str = Field(
        "",
        description="Any risks, blockers, concerns, or dependencies raised in this discussion. Empty string if none."
    )
    suggestions: str = Field(
        "",
        description="Suggestions or recommendations made during this discussion. Empty string if none."
    )
    follow_up_required: str = Field(
        "No",
        description="Whether follow-up is needed: 'Yes — <what follow-up>' or 'No'."
    )
    notes: str = Field(
        "",
        description="Additional notes, context, or miscellaneous information for this discussion point."
    )
    agenda_item: str = Field(
        "Off Agenda Discussion",
        description=(
            "The agenda item this discussion maps to. Only set when the agenda is provided "
            "and there is explicit evidence the discussion relates to a specific agenda item. "
            "Defaults to 'Off Agenda Discussion' when no match exists."
        )
    )
    authority_context: str = Field(
        "",
        description="Who said what, their role or authority, and whether they proposed, directed, approved, or challenged it."
    )
    tone_and_consequence: str = Field(
        "",
        description="Explicit tone and consequence language such as final warning, last chance, escalation, or deadline impact."
    )
    cross_topic_context: str = Field(
        "",
        description="Links to related agenda topics when this discussion or escalation spans more than one topic."
    )
    implicit_decision: str = Field(
        "",
        description="Inferred consensus only when the transcript contains explicit agreement, acceptance, or no-objection evidence; otherwise empty."
    )


class MeetingSummary(BaseModel):
    """Pydantic model representing the full analyzed meeting intelligence output."""

    meeting_title: str = Field(
        ...,
        description="Title of the meeting."
    )
    meeting_date: str = Field(
        "",
        description="Meeting date as displayed in the exported MoM reports."
    )
    chaired_by: str = Field(
        "",
        description="Name of the meeting chairperson for the report header."
    )
    organization: str = Field(
        "",
        description="Organization name for the report header."
    )
    absents: str = Field(
        "Nil",
        description="Absentees text shown in the report header."
    )
    executive_summary: str = Field(
        ...,
        description="A concise professional executive-level summary of the entire meeting."
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
        description="All decisions made during the meeting."
    )
    implicit_decisions: List[str] = Field(
        default_factory=list,
        description="Consensus signals explicitly supported by agreement, acceptance, or no-objection language and labeled as inferred."
    )
    cross_topic_context: List[str] = Field(
        default_factory=list,
        description="Important links and escalations that span multiple agenda topics."
    )
    tone_and_consequences: List[str] = Field(
        default_factory=list,
        description="Material tone, urgency, warning, and consequence statements from the meeting."
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Project risks, blockers, dependencies, or unresolved issues."
    )
    questions: List[str] = Field(
        default_factory=list,
        description="Open questions raised during the meeting."
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="Summary list of all extracted action items/tasks across the entire meeting."
    )
    discussion_points: List[DiscussionPoint] = Field(
        default_factory=list,
        description=(
            "Detailed discussion points — one entry per distinct topic discussed. "
            "Every discussion must be captured here in full, in chronological order. "
            "Never skip any discussion, even if it appears minor."
        )
    )
    participants: List[str] = Field(
        default_factory=list,
        description=(
            "AI-detected speaker names. Use 'Unknown Speaker' or 'Unknown Speaker N' "
            "for any speaker whose name cannot be confidently identified."
        )
    )
    attendees: List[str] = Field(
        default_factory=list,
        description=(
            "Manually-provided attendee list (from the user). This is the authoritative "
            "list of meeting attendees and should take precedence over AI-detected participants."
        )
    )
    pending_items: List[str] = Field(
        default_factory=list,
        description=(
            "Items that were raised but not resolved, deferred decisions, "
            "or topics that need follow-up in a later meeting."
        )
    )
    parking_lot: List[str] = Field(
        default_factory=list,
        description=(
            "Topics that were mentioned but intentionally set aside for a future meeting "
            "or out-of-scope for this meeting."
        )
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
