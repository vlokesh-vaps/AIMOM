"""Six-agent MoM pipeline.

Each agent has one focused responsibility and returns small JSON. The final
result is converted to the existing MeetingSummary model, so report code does
not need to know which LLM produced the data.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

from ai.models.meeting import ActionItem, DiscussionPoint, MeetingSummary
from ai.providers.base import BaseAIProvider
from ai.stages.chunking_engine import ChunkingEngine
from ai.stages.transcript_cleaner import TranscriptCleaner
from ai.utils.token_utils import estimate_tokens
from config.settings import (
    AGENT1_MODEL,
    AGENT2_MODEL,
    AGENT3_MODEL,
    AGENT4_MODEL,
    AGENT5_MODEL,
    AGENT6_MODEL,
    NVIDIA_REQUEST_THROTTLE_SECONDS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentSpec:
    number: int
    name: str
    provider: str
    model: str
    max_tokens: int


AGENTS = (
    AgentSpec(1, "Transcript cleanup", "groq", AGENT1_MODEL, 1800),
    AgentSpec(2, "Topic segmentation", "nvidia", AGENT2_MODEL, 1200),
    AgentSpec(3, "Discussion extraction", "nvidia", AGENT3_MODEL, 1800),
    AgentSpec(4, "Action extraction", "nvidia", AGENT4_MODEL, 1400),
    AgentSpec(5, "Decision synthesis", "nvidia", AGENT5_MODEL, 1800),
    AgentSpec(6, "Validation", "groq", AGENT6_MODEL, 1200),
)


class SixAgentPipeline:
    """Run the configured six-agent workflow with deterministic fallbacks."""

    def __init__(
        self,
        providers: dict[str, BaseAIProvider],
        retry_fn: Callable,
    ) -> None:
        self._providers = providers
        self._retry_fn = retry_fn

    def run(
        self,
        title: str,
        date: str,
        transcript: str,
        attendees: str | None = None,
        agenda: str | None = None,
    ) -> MeetingSummary:
        logger.info("Six-agent MoM pipeline started.")
        cleaned = TranscriptCleaner().clean(transcript)
        chunks = ChunkingEngine(overlap_lines=3).chunk(cleaned, 900)
        if not chunks:
            return MeetingSummary(meeting_title=title, meeting_date=date, executive_summary="No content extracted.")

        # Agent 1 cleans each bounded chunk so the request remains within context limits.
        clean_chunks = []
        for chunk in chunks:
            prompt = self._prompt(1, title, date, agenda, attendees, chunk.text)
            response = self._call(AGENTS[0], prompt)
            clean_chunks.append(self._extract_text(response) or chunk.text)
        clean_text = "\n\n".join(clean_chunks)

        topics: list[dict] = []
        discussions: list[dict] = []
        actions: list[dict] = []
        for index, chunk_text in enumerate(clean_chunks, start=1):
            topic_response = self._call(AGENTS[1], self._prompt(2, title, date, agenda, attendees, chunk_text))
            chunk_topics = self._json_list(topic_response, "topics")
            topics.extend(chunk_topics)

            context = json.dumps({"topics": chunk_topics, "transcript": chunk_text}, ensure_ascii=False)
            discussion_response = self._call(AGENTS[2], self._prompt(3, title, date, agenda, attendees, context))
            discussions.extend(self._json_list(discussion_response, "discussion_points"))

            action_response = self._call(AGENTS[3], self._prompt(4, title, date, agenda, attendees, context))
            actions.extend(self._json_list(action_response, "action_items"))
            logger.info("Agents 2-4 completed chunk %d/%d.", index, len(clean_chunks))

        actions = self._deduplicate_action_dicts(actions)

        decisions_input = json.dumps(
            {"topics": topics, "discussion_points": discussions, "action_items": actions},
            ensure_ascii=False,
        )
        decision_response = self._call(AGENTS[4], self._prompt(5, title, date, agenda, attendees, decisions_input))
        decision_data = self._json_object(decision_response)

        summary = self._build_summary(
            title=title,
            date=date,
            attendees=attendees,
            topics=topics,
            discussions=discussions,
            actions=actions,
            decisions=decision_data,
        )

        validation_input = json.dumps(summary.model_dump(), ensure_ascii=False)
        validation_response = self._call(AGENTS[5], self._prompt(6, title, date, agenda, attendees, validation_input))
        validation = self._json_object(validation_response)
        if validation.get("valid") is False:
            logger.warning("Agent 6 validation found issues: %s", validation.get("issues", []))
        logger.info("Six-agent MoM pipeline completed: %d discussions, %d actions.", len(summary.discussion_points), len(summary.action_items))
        return summary

    def _call(self, spec: AgentSpec, user_prompt: str) -> str:
        provider = self._providers.get(spec.provider)
        if provider is None or not provider.is_configured():
            raise RuntimeError(f"Agent {spec.number} requires configured provider '{spec.provider}'.")
        if spec.provider == "nvidia" and NVIDIA_REQUEST_THROTTLE_SECONDS > 0:
            time.sleep(NVIDIA_REQUEST_THROTTLE_SECONDS)
        system_prompt = self._system_prompt(spec.number)
        logger.info("Agent %d %s: %s/%s", spec.number, spec.name, spec.provider, spec.model)
        # Provider model overrides are resolved by the provider classes.
        if spec.provider == "nvidia":
            from ai.providers.nvidia import NvidiaAIProvider
            provider = NvidiaAIProvider(model_override=spec.model)
        elif spec.provider == "groq":
            from ai.providers.groq import GroqAIProvider
            provider = GroqAIProvider(model_override=spec.model)
        return self._retry_fn(provider=provider, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=spec.max_tokens)

    @staticmethod
    def _system_prompt(number: int) -> str:
        prompts = {
            1: "Clean meeting transcript text. Preserve speaker names, facts, dates, and meaning. Return only JSON: {\"cleaned_text\": \"...\"}.",
            2: "Segment topics for the MoM report. Return JSON with topics containing the exact agenda field and cross_topic_context. Never invent links.",
            3: "Extract every discussion for the MoM report. Return JSON with discussion_points containing agenda_item (report Agenda), point and detailed_summary (report Discussion), and decision. Keep the discussion factual and complete. Do not invent information.",
            4: "Extract every action for the MoM report. Return JSON with action_items containing agenda_item (report Agenda), task (report Action Item), owner (report Assigned), and target_date (report Target Date). The owner must be a real name from the attendee list when assigned. Copy the exact date or timeline stated in the transcript. If no owner or date is explicitly given, return an empty string. Never guess.",
            5: "Synthesize the final MoM facts for the report fields Agenda, Discussion, Action Item, Assigned, and Target Date. Verify that every action has the correct assigned person and exact stated date. Do not invent missing owners or dates. Do not generate S.No; the application assigns it.",
            6: "Validate the supplied meeting summary for missing or inconsistent facts. Return only JSON: {\"valid\": true, \"issues\": [], \"suggestions\": []}. Do not rewrite the summary.",
        }
        return prompts[number]

    @staticmethod
    def _prompt(number: int, title: str, date: str, agenda: str | None, attendees: str | None, content: str) -> str:
        return (
            f"Meeting title: {title}\nMeeting date: {date}\n"
            f"Attendees: {attendees or 'Not provided'}\nAgenda:\n{agenda or 'Not provided'}\n"
            f"Agent task {number} input:\n{content}"
        )

    @staticmethod
    def _extract_text(raw: str) -> str:
        data = SixAgentPipeline._json_object(raw)
        return str(data.get("cleaned_text", "")).strip()

    @staticmethod
    def _json_object(raw: str) -> dict:
        cleaned = (raw or "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", cleaned[start:end + 1]))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _json_list(raw: str, key: str) -> list[dict]:
        value = SixAgentPipeline._json_object(raw).get(key, [])
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    @staticmethod
    def _deduplicate_action_dicts(actions: list[dict]) -> list[dict]:
        from difflib import SequenceMatcher
        deduped = []
        for action in actions:
            task = str(action.get("task", "")).strip()
            if not task:
                continue
            
            owner = str(action.get("owner", "")).strip()
            target_date = str(action.get("target_date", action.get("deadline", ""))).strip()
            
            is_duplicate = False
            for existing in deduped:
                existing_task = str(existing.get("task", "")).strip()
                existing_owner = str(existing.get("owner", "")).strip()
                
                similarity = SequenceMatcher(None, task.lower(), existing_task.lower()).ratio()
                if similarity > 0.85 or (task == existing_task and owner == existing_owner):
                    is_duplicate = True
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

    @staticmethod
    def _build_summary(title, date, attendees, topics, discussions, actions, decisions) -> MeetingSummary:
        discussion_points = [DiscussionPoint(
            point=str(item.get("point", "Untitled discussion")),
            detailed_summary=str(item.get("detailed_summary", "")),
            agenda_item=str(item.get("agenda_item", "Off Agenda Discussion")),
            decision=str(item.get("decision", "No Decision Taken")),
            status=str(item.get("status", "Open")),
            authority_context=str(item.get("authority_context", "")),
            tone_and_consequence=str(item.get("tone_and_consequence", "")),
            cross_topic_context=str(item.get("cross_topic_context", "")),
            implicit_decision=str(item.get("implicit_decision", "")),
        ) for item in discussions]
        action_items = [ActionItem(
            task=str(item.get("task", "")),
            owner=str(item.get("owner", "")),
            target_date=str(item.get("target_date", item.get("deadline", ""))),
            priority=str(item.get("priority", "Medium")),
            status=str(item.get("status", "Pending")),
            agenda_item=str(item.get("agenda_item", "Off Agenda Discussion")),
            authority_context=str(item.get("authority_context", "")),
            tone_and_consequence=str(item.get("tone_and_consequence", "")),
        ) for item in actions if item.get("task")]
        topic_names = list(dict.fromkeys(str(item.get("agenda_item") or item.get("topic")) for item in topics if item.get("topic") or item.get("agenda_item")))
        return MeetingSummary(
            meeting_title=title,
            meeting_date=date,
            executive_summary=str(decisions.get("executive_summary", "Meeting summary generated from the transcript.")),
            topics=topic_names or decisions.get("topics_covered", []),
            decisions=decisions.get("decisions", []),
            implicit_decisions=decisions.get("implicit_decisions", []),
            cross_topic_context=decisions.get("cross_topic_context", []),
            tone_and_consequences=decisions.get("tone_and_consequences", []),
            risks=decisions.get("risks", []),
            questions=decisions.get("pending_items", []),
            pending_items=decisions.get("pending_items", []),
            action_items=action_items,
            discussion_points=discussion_points,
            attendees=[name.strip() for name in (attendees or "").split(",") if name.strip()],
            participants=[],
            followups=[item.task for item in action_items],
            timeline=topic_names,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
