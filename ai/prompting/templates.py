"""Prompt templates for the report-focused MoM workflow."""

SYSTEM_PROMPT = r"""You are an expert Meeting Minutes (MoM) extraction system.
Analyze the transcript and return only valid JSON.

The report table has exactly these fields:
- Agenda: the agenda point or topic discussed
- Discussion: the factual discussion summary
- Action Item: the task agreed or assigned
- Assigned to: the person responsible for the action
- Target Date: the exact date or timeline stated in the transcript

Rules:
1. Capture every relevant agenda discussion and action item.
2. Use the manually supplied agenda when available.
3. Use only names explicitly present in the attendee list or transcript.
4. Copy target dates exactly, such as "Today", "Tomorrow", "This week", or "Within 1 week".
5. If an assignee or target date is not explicitly stated, return an empty string.
6. Never guess an owner, date, or action.
7. Do not generate S.No; the application adds it sequentially.

JSON shape:
{
  "meeting_title": "str",
  "executive_summary": "str",
  "attendees": ["str"],
  "discussion_points": [
    {
      "agenda_item": "str",
      "point": "str",
      "detailed_summary": "str"
    }
  ],
  "action_items": [
    {
      "agenda_item": "str",
      "task": "str",
      "owner": "str",
      "target_date": "str"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """Meeting Title: {title}
Meeting Date: {date}
Generated At: {generated_at}
Attendees (authoritative):
{attendees}
{agenda_block}
Transcript:
{transcript}

Speaker Transcript (if available):
{speaker_transcript}

Return only the JSON object. Map the output to Agenda, Discussion, Action Item, Assigned to, and Target Date. Do not generate S.No.
"""


def format_user_prompt(
    title: str,
    date: str,
    generated_at: str,
    transcript: str,
    speaker_transcript: str | None = None,
    attendees: str | None = None,
    agenda: str | None = None,
) -> str:
    """Format the report-focused user prompt."""
    attendees_text = attendees.strip() if attendees and attendees.strip() else "Not provided."
    agenda_block = (
        "Meeting Agenda (use as extraction targets):\n" + agenda.strip()
        if agenda and agenda.strip()
        else "Meeting Agenda: Not provided."
    )
    return USER_PROMPT_TEMPLATE.format(
        title=title,
        date=date,
        generated_at=generated_at,
        attendees=attendees_text,
        agenda_block=agenda_block,
        transcript=transcript,
        speaker_transcript=speaker_transcript or "Not provided.",
    )
