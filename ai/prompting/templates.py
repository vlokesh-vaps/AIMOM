"""Prompt templates for AI Meeting Minutes (MoM) generation.

The system prompt enforces comprehensive, zero-omission extraction:
  - Every discussion point captured in full with all 12 MoM fields
  - Unknown speakers labeled explicitly
  - Pending items and parking lot topics tracked
  - Manual attendees list used when provided
"""

SYSTEM_PROMPT = r"""You are an expert AI Meeting Minutes (MoM) system.
Analyze the transcript and return ONLY a valid JSON matching the schema below. No explanation, preamble, or markdown.
STRICT EXTRACTION RULES:
1. Do not skip any discussion. Capture all ideas, decisions, questions, actions, suggestions, risks, and follow-ups.
2. Maintain original chronological order. Do not merge unrelated points.
3. For action items, 'assigned_to' is the assignee only, never the requester.
4. Label unidentified speakers as 'Unknown Speaker N' (N = 1, 2, 3...).
5. If manual attendees are provided, use them for the 'attendees' field.
6. MINIMIZE JSON SIZE: Omit keys in `discussion_points` and `action_items` if they would contain default values:
   - For discussion_points, omit: "decision" (if "No Decision Taken"), "task" (if "No Action Item"), "assigned_to"/"deadline" (if "Not Specified"), "priority" (if "Medium"), "status" (if "Open"), "risks_or_concerns"/"suggestions"/"notes" (if ""), "follow_up_required" (if "No").
   - For action_items, omit: "owner"/"target_date"/"notes" (if ""), "priority" (if "Medium"), "status" (if "Pending").
JSON SCHEMA:
{
    "meeting_title": "str",
    "executive_summary": "str (concise, executive-level summary)",
    "topics": ["str"],
    "questions": ["str"],
    "action_items": [
        {
            "task": "str (REQUIRED)",
            "owner": "str (Optional)",
            "target_date": "str (Optional)",
            "status": "Pending | Completed | In Progress | Information (Optional)",
        }
    ],
    "discussion_points": [
        {
            "point": "str (REQUIRED)",
            "detailed_summary": "str (REQUIRED, detailed narrative of contributions/debates)",
            "decision": "str (Optional)",
            "task": "str (Optional)",
            "assigned_to": "str (Optional)",
            "deadline": "str (Optional)",
            "priority": "High | Medium | Low (Optional)",
            "status": "Open | In Progress | Completed | Pending (Optional)",
            "follow_up_required": "str (Optional)",
            "notes": "str (Optional)"
        }
    ]
}
"""

USER_PROMPT_TEMPLATE = """Meeting Title: {title}
Meeting Date: {date}
Generated At (ISO): {generated_at}
Manually Provided Attendees (authoritative — use this list for the 'attendees' field):
{attendees}
Transcript:
{transcript}

Speaker Transcript (if available):
{speaker_transcript}
"""


def format_user_prompt(
    title: str,
    date: str,
    generated_at: str,
    transcript: str,
    speaker_transcript: str | None = None,
    attendees: str | None = None,
) -> str:
    """Format the user prompt for the LLM with meeting metadata, transcript, and optional attendees."""
    attendees_text = attendees.strip() if attendees and attendees.strip() else "Not provided by user."
    return USER_PROMPT_TEMPLATE.format(
        title=title,
        date=date,
        generated_at=generated_at,
        attendees=attendees_text,
        transcript=transcript,
        speaker_transcript=speaker_transcript or "Not provided.",
    )


# ---------------------------------------------------------------------------
# Stage 3 — Chunk Extraction Prompt (Groq / openai/gpt-oss-20b)
# ---------------------------------------------------------------------------

CHUNK_EXTRACTION_SYSTEM_PROMPT = r"""You extract factual meeting data from ONE transcript chunk.
Return valid JSON only. No markdown, no prose outside JSON.

Do not summarize the whole meeting. Do not write polished report text.
Extract only facts explicitly present in this chunk. If unknown, use null.
For action items, owner means assignee only, not requester.
The overlap context is reference only; do not re-extract items from it.

Required JSON shape:
{
  "discussion_points": [{"topic": "str", "details": "str", "speakers": ["str"]}],
  "action_items": [{"task": "str", "owner": null, "deadline": null, "topic": null}],
  "decisions": ["str"],
  "risks": ["str"],
  "blockers": ["str"],
  "questions": ["str"],
  "deadlines": ["str"],
  "participants": ["str"]
}

Keep each string concise. Omit empty list items. Use [] for no items.
"""


# ---------------------------------------------------------------------------
# Stage 6 — Final Review Prompt (Ollama / gemma4:latest)
# ---------------------------------------------------------------------------

FINAL_REVIEW_SYSTEM_PROMPT = r"""You are a meeting-minutes copy reviewer.
You receive complete meeting minutes JSON.

Improve wording, grammar, readability, and consistency only.
Do not invent information. Do not remove information. Do not alter facts.
Preserve every discussion point, action item, deadline, owner, risk, blocker,
question, and decision. Preserve generated_at.

Return only valid JSON matching the input schema. No markdown or explanation.
"""
