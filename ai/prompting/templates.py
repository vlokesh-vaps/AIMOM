"""Prompt templates for AI Meeting Minutes (MoM) generation.

The system prompt enforces comprehensive, zero-omission extraction:
  - Every discussion point captured in full with all 12 MoM fields
  - Unknown speakers labeled explicitly
  - Pending items and parking lot topics tracked
  - Manual attendees list used when provided
"""

SYSTEM_PROMPT = r"""You are an expert AI Meeting Minutes system.
Analyze the transcript and return ONLY valid JSON matching the schema below.
RULES:
1. Capture all key discussions, decisions, and action items.
2. For action items, 'owner' is the assignee.
3. MINIMIZE JSON SIZE: Omit keys if they are empty or default values (e.g., "No Decision Taken", "Not Specified", "Open", "Medium", "").
JSON SCHEMA:
{
    "meeting_title": "str",
    "executive_summary": "str (clear, concise executive-level summary)",
    "topics": ["str"],
    "action_items": [
        {
            "task": "str (REQUIRED)",
            "owner": "str",
            "target_date": "str",
            "priority": "High | Medium | Low",
            "status": "Pending | Completed | In Progress",
            "notes": "str",
            "agenda_item": "str"
        }
    ],
    "discussion_points": [
        {
            "point": "str (REQUIRED)",
            "detailed_summary": "str (REQUIRED, clear and concise summary of the discussion)",
            "agenda_item": "str",
            "decision": "str",
            "task": "str",
            "assigned_to": "str",
            "Data": "str",
        }
    ],
    "attendees": ["str"]
}
"""

USER_PROMPT_TEMPLATE = """Meeting Title: {title}
Meeting Date: {date}
Generated At (ISO): {generated_at}
Manually Provided Attendees (authoritative — use this list for the 'attendees' field):
{attendees}
{agenda_block}
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
    agenda: str | None = None,
) -> str:
    """Format the user prompt for the LLM with meeting metadata, transcript, and optional attendees/agenda."""
    attendees_text = attendees.strip() if attendees and attendees.strip() else "Not provided by user."
    if agenda and agenda.strip():
        agenda_block = (
            "Meeting Agenda (use these points as active extraction targets — for each point, find all relevant discussions, decisions, and action items in the transcript):\n"
            f"{agenda.strip()}"
        )
    else:
        agenda_block = ""
    return USER_PROMPT_TEMPLATE.format(
        title=title,
        date=date,
        generated_at=generated_at,
        attendees=attendees_text,
        agenda_block=agenda_block,
        transcript=transcript,
        speaker_transcript=speaker_transcript or "Not provided.",
    )


# ---------------------------------------------------------------------------
# Stage 3 — Chunk Extraction Prompt (Groq / openai/gpt-oss-20b)
# ---------------------------------------------------------------------------

CHUNK_EXTRACTION_SYSTEM_PROMPT = r"""You are an expert meeting minutes extraction engine working on ONE chunk of a larger meeting transcript. Your job is to extract factual meeting data from this chunk and map it precisely to the provided agenda points. You must actively search the chunk for content related to each agenda point — do not wait for the content to be obvious. Read carefully and pull out every relevant discussion, decision, action item, and deadline.
You will be given a list of agenda points and a list of known attendees. Use the agenda points as your extraction targets — for each agenda point, find all relevant content in this chunk. Use the attendees list to fill owner and assigned_to fields — never invent names and never leave owner as null if a name is mentioned anywhere in the chunk.
Extraction rules you must follow without exception. First, map every discussion point to its closest agenda item — if a discussion clearly belongs to an agenda point, tag it. If a chunk contains content for multiple agenda points, extract separately for each. Second, for action items, the task must start with an action verb such as complete, prepare, present, fix, review, submit, or schedule. The owner must be a real name from the attendees list. The deadline must capture the exact specific timeline context mentioned (e.g., "Monday (Next Day)", "Tomorrow EOD", "This Week", "Within 1 Week (Plan)") — do not paraphrase generically or invent deadlines, capture specific timelines. Third, for decisions, only capture what was explicitly decided or directed — do not capture opinions or suggestions as decisions. Fourth, if a discussion point has no clear agenda match, still extract it under agenda_item as null — do not discard it. Fifth, keep all strings concise and factual. Do not add commentary, interpretation, or filler.
Return only valid compact JSON with no markdown, no explanation, and no prose outside the JSON block. If a list has no items return an empty array. Omit keys entirely if their value would be null or empty.

Required JSON shape:
{
  "chunk_index": "int",
  "agenda_coverage": ["list of agenda point names found in this chunk"],
  "discussion_points": [
    {
      "topic": "short topic name",
      "details": "clear and concise factual summary of what was discussed",
      "agenda_item": "exact agenda point name or null",
      "decision": "what was decided if any",
      "authority_context": "who said it and their role or authority",
      "tone_and_consequence": "explicit urgency, warning, or consequence",
      "cross_topic_context": "related agenda topics or escalation spanning topics",
      "implicit_decision": "inferred consensus only with explicit agreement, acceptance, or no-objection evidence; otherwise null",
      "status": "Open or Resolved or Escalated or Pending",
      "authority_context": "who said it and their authority",
      "tone_and_consequence": "explicit tone, warning, urgency, or consequence",
      "cross_topic_context": "related topics or spanning escalation",
      "implicit_decision": "inferred consensus backed by explicit agreement evidence"
    }
  ],
  "action_items": [
    {
      "task": "starts with action verb",
      "owner": "name from attendees list",
      "deadline": "specific timeline context (e.g., Monday (Next Day), Tomorrow EOD)",
      "priority": "High or Medium or Low",
      "agenda_item": "exact agenda point name or null",
      "authority_context": "who assigned or authorized the action",
      "tone_and_consequence": "explicit urgency, warning, or consequence"
    }
  ],
  "decisions": ["list of explicit decisions made"],
  "participants_mentioned": ["names from attendees list mentioned in this chunk"],
  "cross_topic_context": ["links or escalations spanning agenda topics"],
  "implicit_decisions": ["inferred consensus with explicit agreement evidence only"],
  "tone_and_consequences": ["explicit tone, warning, urgency, or consequence statements"]
}
"""

CHUNK_EXTRACTION_USER_PROMPT = """You are processing chunk {chunk_index} of {total_chunks} from this meeting transcript.

Meeting Title: {title}
Meeting Date: {date}

Known Attendees (use ONLY these names for owner and assigned_to fields — do not invent any name):
{attendees_list}

Agenda Points to Extract For (these are your extraction targets — actively search for each one in the chunk below):
{agenda_points_numbered}

Chunk Text:
{chunk_text}

For each agenda point listed above, search this chunk and extract all relevant discussions, decisions, and action items. If an agenda point has no content in this chunk, skip it. Do not assume content is present — only extract what is explicitly in the chunk text above."""

MERGE_SUMMARY_SYSTEM_PROMPT = r"""You are a meeting minutes synthesis engine. You will receive structured JSON data extracted from multiple chunks of a single meeting transcript. Your job is to merge all chunk data into one clean, complete, and deduplicated meeting summary.

Merging rules you must follow. First, group all discussion points by agenda item — combine related points from different chunks that discuss the same topic into a single unified discussion point. Do not keep duplicates. Second, for action items, deduplicate by task meaning — if two chunks extracted the same action item with slightly different wording, keep the more detailed one. Never lose an action item during merge. Third, for decisions, deduplicate by meaning — keep the clearest and most complete version. Fourth, the executive summary must be written fresh from the merged data — it must cover the key decisions made, major action items with owners and deadlines, critical issues raised, and any escalations. Write it as a clear paragraph that a senior executive can read in thirty seconds. Fifth, the attendees field must use the manually provided attendees list — do not modify it based on participants mentioned in chunks.

Return only valid JSON with no markdown and no explanation outside the JSON block.

Required JSON shape:
{
  "meeting_title": "str",
  "meeting_date": "str",
  "executive_summary": "str — a clear paragraph covering key decisions, major actions, critical issues, and escalations",
  "attendees": ["from manually provided list only"],
  "topics_covered": ["list of agenda points that had content"],
  "discussion_points": [
    {
      "agenda_item": "exact agenda point name",
      "summary": "unified summary of all discussion on this agenda point",
      "decision": "what was decided",
      "status": "Open or Resolved or Escalated or Pending"
    }
  ],
  "action_items": [
    {
      "task": "starts with action verb",
      "owner": "name from attendees list",
      "deadline": "specific timeline context (e.g., Monday (Next Day), Tomorrow EOD)",
      "priority": "High or Medium or Low",
      "agenda_item": "exact agenda point name",
      "authority_context": "who assigned or authorized the action",
      "tone_and_consequence": "explicit tone or consequence"
    }
  ],
  "decisions": ["list of all decisions from all chunks deduplicated"],
  "pending_items": ["items raised but not resolved or assigned"],
  "escalations": ["items explicitly escalated to senior management"],
  "implicit_decisions": ["inferred consensus with explicit agreement evidence only"],
  "cross_topic_context": ["links or escalations spanning agenda topics"],
  "tone_and_consequences": ["explicit tone, warnings, urgency, and consequences"]
}
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
