"""Async Parallel Single-Extraction MoM Pipeline.

Architecture:
  1. Transcript Cleaning   — pure Python
  2. Smart Chunking        — pure Python
  3. Parallel Extraction   — asyncio, one LLM call per chunk via ProviderManager
  4. Checkpoint Manager    — SHA-256 hash-based save/resume
  5. Incremental Merge     — pure Python dedup + normalization
  6. Final Synthesis       — single LLM call for high-level summary metadata
  7. Validation            — optional, never blocks

All LLM calls route through ProviderManager for adaptive scheduling,
load-balancing, and truncation recovery.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any, TYPE_CHECKING

from ai.models.meeting import ActionItem, DiscussionPoint, MeetingSummary
from ai.stages.checkpoint_manager import CheckpointManager
from ai.stages.chunking_engine import ChunkingEngine, TranscriptChunk
from ai.stages.merge_engine import MergeEngine
from ai.stages.transcript_cleaner import TranscriptCleaner
from config.settings import (
    EXTRACTION_MAX_TOKENS,
    EXTRACTION_MODEL,
    GROQ_FALLBACK_MODEL,
    LLM_CHUNK_SIZE_TOKENS,
    SYNTHESIS_MAX_TOKENS,
    SYNTHESIS_MODEL,
)
from utils.logger import get_logger

if TYPE_CHECKING:
    from ai.providers.provider_manager import ProviderManager

logger = get_logger(__name__)


# ── Kept for backward compatibility with manager.py import ────────────────
FourAgentPipeline = None  # type alias set at module bottom


class SingleExtractionPipeline:
    """Production-grade async parallel MoM pipeline.

    Processes each transcript chunk exactly once (one LLM call) in parallel,
    saves checkpoints for fault tolerance, merges results locally in Python
    to preserve 100% of granular details, and uses a final LLM call only
    for executive summary, decisions, and metadata synthesis.
    """

    def __init__(self, provider_manager: ProviderManager) -> None:
        self._pm = provider_manager

    # ------------------------------------------------------------------
    # Main entry point (sync — called from manager.py)
    # ------------------------------------------------------------------

    def run(
        self,
        title: str,
        date: str,
        transcript: str,
        attendees: str | None = None,
        agenda: str | None = None,
    ) -> MeetingSummary:
        """Execute the full pipeline synchronously (wraps the async core)."""
        logger.info("=" * 70)
        logger.info("SINGLE EXTRACTION PIPELINE — Starting")
        logger.info("=" * 70)
        pipeline_start = time.monotonic()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._run_async(title, date, transcript, attendees, agenda),
                )
                result = future.result()
        else:
            result = asyncio.run(
                self._run_async(title, date, transcript, attendees, agenda)
            )

        elapsed = time.monotonic() - pipeline_start
        logger.info(
            "SINGLE EXTRACTION PIPELINE — Completed in %.1fs "
            "(%d discussions, %d actions).",
            elapsed, len(result.discussion_points), len(result.action_items),
        )
        return result

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _run_async(
        self,
        title: str,
        date: str,
        transcript: str,
        attendees: str | None,
        agenda: str | None,
    ) -> MeetingSummary:
        pipeline_start = time.monotonic()

        # ── Stage 1: Transcript Cleaning (pure Python) ────────────────
        logger.info("[Stage 1/6] Cleaning transcript...")
        cleaned = TranscriptCleaner().clean(transcript)

        # ── Stage 2: Smart Chunking (pure Python) ─────────────────────
        logger.info("[Stage 2/6] Chunking transcript...")
        chunks = ChunkingEngine(overlap_lines=3).chunk(cleaned, LLM_CHUNK_SIZE_TOKENS)
        if not chunks:
            return self._empty_summary(title, date)

        logger.info("[Stage 2/6] Created %d chunks.", len(chunks))

        # ── Checkpoint setup ──────────────────────────────────────────
        session_id = hashlib.sha256(
            f"{title}:{date}:{len(cleaned)}".encode()
        ).hexdigest()[:12]
        checkpoint_mgr = CheckpointManager(session_id)

        # Compute content hashes for each chunk
        chunk_hashes = [CheckpointManager.content_hash(c.text) for c in chunks]

        # Load existing checkpoints (resume support)
        existing = checkpoint_mgr.load_all_completed(chunk_hashes)

        # ── Stage 3: Parallel Extraction (async via scheduler) ────────
        logger.info(
            "[Stage 3/6] Extracting from %d chunks (%d cached)...",
            len(chunks), len(existing),
        )
        extraction_start = time.monotonic()

        tasks = []
        for chunk in chunks:
            chunk_hash = chunk_hashes[chunk.index]
            tasks.append(
                self._extract_single_chunk(
                    chunk, chunk_hash, title, date, attendees, agenda,
                    checkpoint_mgr, existing,
                )
            )

        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results and log failures
        extraction_data: list[tuple[int, dict[str, Any]]] = []
        failed_count = 0
        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                logger.error(
                    "[Stage 3/6] Chunk %d FAILED: %s", i, result,
                )
                failed_count += 1
            elif result is not None:
                extraction_data.append(result)

        extraction_elapsed = time.monotonic() - extraction_start
        logger.info(
            "[Stage 3/6] Extraction complete: %d/%d succeeded, %d failed (%.1fs).",
            len(extraction_data), len(chunks), failed_count, extraction_elapsed,
        )

        if not extraction_data:
            logger.error("[Stage 3/6] All chunks failed — returning empty summary.")
            return self._empty_summary(title, date)

        # ── Stage 4: Incremental Merge (pure Python) ──────────────────
        logger.info("[Stage 4/6] Merging %d chunk results...", len(extraction_data))
        merge_engine = MergeEngine()
        for chunk_index, data in extraction_data:
            merge_engine.add_chunk(data, chunk_index)
        merged = merge_engine.finalize()

        # ── Stage 5: Final Synthesis (single LLM call for summary metadata) ──
        logger.info("[Stage 5/6] Final synthesis on merged JSON...")
        summary = await self._final_synthesis(
            title=title,
            date=date,
            attendees=attendees,
            agenda=agenda,
            merged=merged,
        )

        # ── Stage 6: Validation (optional, never blocks) ──────────────
        await self._validate(summary)

        # Cleanup checkpoints on full success
        if failed_count == 0:
            checkpoint_mgr.cleanup()

        # Generate observability execution report
        total_time = time.monotonic() - pipeline_start
        report_md = self._pm.generate_execution_report(
            total_duration=total_time,
            checkpoints_cached=len(existing),
        )

        # Save Markdown report to output folder
        from config.settings import OUTPUT_DIR
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = OUTPUT_DIR / f"execution_report_{timestamp}.md"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info("[ProviderManager] Saved pipeline execution report: %s", report_path)
        except OSError as exc:
            logger.warning("[ProviderManager] Failed to save execution report: %s", exc)

        logger.info("\n" + "=" * 70 + "\nPIPELINE OBSERVABILITY REPORT SUMMARY\n" + "=" * 70 + f"\n{report_md}\n" + "=" * 70)

        return summary

    # ------------------------------------------------------------------
    # Stage 3 — Per-chunk extraction
    # ------------------------------------------------------------------

    async def _extract_single_chunk(
        self,
        chunk: TranscriptChunk,
        chunk_hash: str,
        title: str,
        date: str,
        attendees: str | None,
        agenda: str | None,
        checkpoint_mgr: CheckpointManager,
        existing: dict[str, dict[str, Any]],
    ) -> tuple[int, dict[str, Any]] | None:
        """Extract structured data from a single chunk using priority scheduler."""
        idx = chunk.index

        # Check for existing checkpoint
        if chunk_hash in existing:
            cached = existing[chunk_hash]
            logger.info(
                "[Worker %d] Using cached checkpoint %s (provider=%s).",
                idx, chunk_hash, cached.get("metadata", {}).get("provider", "?"),
            )
            return (idx, cached["data"])

        # Build compact extraction prompt
        system_prompt = self._extraction_system_prompt()
        user_prompt = self._extraction_user_prompt(
            title, date, attendees, agenda, chunk.text, idx,
        )

        worker_start = time.monotonic()
        logger.info("[Worker %d] Dispatching chunk extraction (hash=%s)...", idx, chunk_hash)

        try:
            raw_response = await self._pm.execute_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task_type="extraction",
                max_tokens=EXTRACTION_MAX_TOKENS,
                priority=1,  # Extraction has normal priority
                chunk_index=idx,
                agent_name=f"Extraction-Chunk-{idx}",
            )
        except Exception as exc:
            logger.error("[Worker %d] Extraction failed: %s", idx, exc)
            raise

        # Parse the structured JSON
        data = self._parse_json_object(raw_response)
        if not data:
            logger.warning(
                "[Worker %d] Empty JSON response — treating as empty extraction.", idx,
            )
            data = {
                "topics": [], "discussion_points": [], "action_items": [],
                "decisions": [], "risks": [], "open_questions": [],
            }

        elapsed = time.monotonic() - worker_start
        logger.info("[Worker %d] Extraction complete in %.1fs (hash=%s).", idx, elapsed, chunk_hash)

        # Save checkpoint
        checkpoint_mgr.save(
            chunk_hash=chunk_hash,
            chunk_index=idx,
            data=data,
            provider="dynamic",
            model=EXTRACTION_MODEL,
        )

        return (idx, data)

    # ------------------------------------------------------------------
    # Stage 5 — Final Synthesis
    # ------------------------------------------------------------------

    async def _final_synthesis(
        self,
        title: str,
        date: str,
        attendees: str | None,
        agenda: str | None,
        merged: dict[str, Any],
    ) -> MeetingSummary:
        """Synthesize high-level metadata (executive summary, decisions, risks, pending)."""
        system_prompt = (
            "You are a professional meeting minutes writer. Synthesize the final high-level "
            "meeting outcomes, decisions, risks, pending items, and the executive summary. "
            "Output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "executive_summary": "...",\n'
            '  "topics_covered": ["..."],\n'
            '  "decisions": ["..."],\n'
            '  "risks": ["..."],\n'
            '  "pending_items": ["..."]\n'
            "}\n"
            "Rules:\n"
            "- Write a polished, detailed executive summary (2-3 paragraphs) in professional business language.\n"
            "- Synthesize and list all key decisions made during the meeting.\n"
            "- Identify and list any noted risks, concerns, or mitigations.\n"
            "- List open questions, follow-ups, or pending items.\n"
            "- Output ONLY the JSON object, no markdown fences, no preamble, and no other keys."
        )
        user_prompt = (
            f"Meeting title: {title}\n"
            f"Meeting date: {date}\n"
            f"Attendees: {attendees or 'Not provided'}\n"
            f"Agenda: {agenda or 'Not provided'}\n\n"
            f"Structured extraction data:\n{json.dumps(merged, ensure_ascii=False)}"
        )

        synthesis_start = time.monotonic()

        try:
            raw = await self._pm.execute_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task_type="synthesis",
                max_tokens=SYNTHESIS_MAX_TOKENS,
                priority=0,  # Synthesis has high priority
                chunk_index=0,
                agent_name="Final-Synthesis",
            )
        except Exception as exc:
            logger.error("[Stage 5/6] Final synthesis failed: %s", exc)
            logger.info("[Stage 5/6] Falling back to empty synthesis.")
            raw = None

        elapsed = time.monotonic() - synthesis_start
        logger.info("[Stage 5/6] Final synthesis completed in %.1fs.", elapsed)

        if raw:
            synthesized = self._parse_json_object(raw)
        else:
            synthesized = {}

        # Build the MeetingSummary, taking discussion points and actions directly from merged
        return self._build_summary(
            title=title,
            date=date,
            attendees=attendees,
            merged=merged,
            synthesized=synthesized,
        )

    # ------------------------------------------------------------------
    # Stage 6 — Validation (optional)
    # ------------------------------------------------------------------

    async def _validate(self, summary: MeetingSummary) -> None:
        """Optional validation — logs issues but never blocks the pipeline."""
        try:
            issues = []
            if not summary.discussion_points:
                issues.append("No discussion points extracted.")
            if not summary.action_items:
                issues.append("No action items extracted.")
            for ai in summary.action_items:
                if not ai.task:
                    issues.append("Action item with empty task found.")

            if issues:
                logger.warning("[Stage 6/6] Validation issues: %s", issues)
            else:
                logger.info("[Stage 6/6] Validation passed.")
        except Exception as exc:
            logger.warning("[Stage 6/6] Validation failed (non-blocking): %s", exc)

    # ------------------------------------------------------------------
    # Prompt templates — compact extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extraction_system_prompt() -> str:
        return (
            "Extract structured information from a meeting transcript chunk. "
            "Return ONLY a valid JSON object with these keys:\n"
            "{\n"
            '  "topics": [{"agenda_item": "...", "topic": "..."}],\n'
            '  "discussion_points": [{"agenda_item": "...", "point": "...", '
            '"detailed_summary": "...", "decision": "No Decision Taken"}],\n'
            '  "action_items": [{"agenda_item": "...", "task": "...", '
            '"owner": "Name of assignee", "target_date": "Target date / timeline", "priority": "Medium"}],\n'
            '  "decisions": ["..."],\n'
            '  "risks": ["..."],\n'
            '  "open_questions": ["..."]\n'
            "}\n"
            "Rules:\n"
            "- Be concise but complete. Do not omit any discussed item.\n"
            "- owner: Extract the name of the specific person assigned to do the task (e.g. 'Ganesh'). "
            "Do NOT use the name of the person who requested or chaired it. If no owner is assigned, use ''.\n"
            "- target_date: Extract the specific timeline mentioned (e.g. 'immediately', 'by tomorrow', "
            "'next review meeting', '2026-07-20'). Do NOT leave empty if any verbal date/timeline is stated. "
            "If no timeline is mentioned at all, use ''.\n"
            "- decision: if no decision was taken, use 'No Decision Taken'.\n"
            "- Output ONLY the JSON object — no markdown fences, no preamble."
        )

    @staticmethod
    def _extraction_user_prompt(
        title: str,
        date: str,
        attendees: str | None,
        agenda: str | None,
        chunk_text: str,
        chunk_index: int,
    ) -> str:
        return (
            f"Meeting: {title} | Date: {date}\n"
            f"Attendees: {attendees or 'Not provided'}\n"
            f"Agenda: {agenda or 'Not provided'}\n"
            f"Chunk {chunk_index}:\n{chunk_text}"
        )

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_object(raw: str) -> dict:
        """Extract and parse a JSON object from an LLM response."""
        cleaned = (raw or "").strip()
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            # Fix trailing commas
            fragment = re.sub(r",\s*([}\]])", r"\1", cleaned[start:end + 1])
            return json.loads(fragment)
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------
    # Build MeetingSummary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        title: str,
        date: str,
        attendees: str | None,
        merged: dict[str, Any],
        synthesized: dict[str, Any],
    ) -> MeetingSummary:
        """Build the final MeetingSummary matching the existing schema."""
        # Detailed items preserved directly from python-merged data (Fixes truncation!)
        disc_source = merged.get("discussion_points", [])
        action_source = merged.get("action_items", [])

        action_items = [
            ActionItem(
                task=str(item.get("task", "")),
                owner=str(item.get("owner", "")),
                target_date=str(item.get("target_date", item.get("deadline", ""))),
                priority=str(item.get("priority", "Medium")),
                status=str(item.get("status", "Pending")),
                agenda_item=str(item.get("agenda_item", "Off Agenda Discussion")),
                authority_context=str(item.get("authority_context", "")),
            )
            for item in action_source
            if isinstance(item, dict) and item.get("task")
        ]

        discussion_points = []
        for item in disc_source:
            if not isinstance(item, dict):
                continue
            
            agenda_item = str(item.get("agenda_item", "Off Agenda Discussion"))
            point = str(item.get("point", "Untitled discussion"))
            
            # Find matching action item for this discussion point (same agenda item or matching task)
            matched_action = None
            for ai in action_items:
                if ai.agenda_item == agenda_item:
                    matched_action = ai
                    break

            assigned_to = matched_action.owner if matched_action else "Not Specified"
            deadline = matched_action.target_date if matched_action else "Not Specified"
            priority = matched_action.priority if matched_action else "Medium"
            task = matched_action.task if matched_action else "No Action Item"

            discussion_points.append(
                DiscussionPoint(
                    point=point,
                    detailed_summary=str(item.get("detailed_summary", "")),
                    agenda_item=agenda_item,
                    decision=str(item.get("decision", "No Decision Taken")),
                    task=task,
                    assigned_to=assigned_to or "Not Specified",
                    deadline=deadline or "Not Specified",
                    priority=priority,
                    status=str(item.get("status", "Open"))
                )
            )

        topic_names = (
            synthesized.get("topics_covered")
            or merged.get("topics", [])
        )

        return MeetingSummary(
            meeting_title=title,
            meeting_date=date,
            executive_summary=str(
                synthesized.get(
                    "executive_summary",
                    "Meeting summary generated from the transcript.",
                )
            ),
            topics=topic_names if isinstance(topic_names, list) else [],
            decisions=synthesized.get("decisions") or merged.get("decisions", []),
            implicit_decisions=synthesized.get("implicit_decisions", []),
            cross_topic_context=synthesized.get("cross_topic_context", []),
            tone_and_consequences=synthesized.get("tone_and_consequences", []),
            risks=synthesized.get("risks") or merged.get("risks", []),
            questions=merged.get("open_questions", []),
            pending_items=synthesized.get("pending_items") or merged.get("open_questions", []),
            action_items=action_items,
            discussion_points=discussion_points,
            attendees=[
                name.strip()
                for name in (attendees or "").split(",")
                if name.strip()
            ],
            participants=[],
            followups=[item.task for item in action_items],
            timeline=topic_names if isinstance(topic_names, list) else [],
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    @staticmethod
    def _empty_summary(title: str, date: str) -> MeetingSummary:
        """Return an empty MeetingSummary when no content is available."""
        return MeetingSummary(
            meeting_title=title,
            meeting_date=date,
            executive_summary="No content extracted.",
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )


# Backward compatibility alias
FourAgentPipeline = SingleExtractionPipeline
