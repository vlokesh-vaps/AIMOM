"""Stage 3 - Groq chunk extractor.

This stage extracts compact factual JSON from each chunk using Groq
openai/gpt-oss-20b. It never asks for polished report sections.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from pydantic import ValidationError

from ai.utils.checkpoints import ChunkCheckpointStore
from ai.models.chunk import ChunkExtraction
from ai.stages.chunking_engine import TranscriptChunk
from ai.providers.base import (
    AIProviderError,
    AIProviderTruncatedResponseError,
    BaseAIProvider,
)
from ai.prompting.templates import CHUNK_EXTRACTION_SYSTEM_PROMPT
from ai.utils.rate_limiter import ProviderRequestScheduler
from ai.utils.token_utils import estimate_tokens
from ai.validators.validation_layer import ValidationLayer
from utils.logger import get_logger

logger = get_logger(__name__)


class ChunkExtractionError(Exception):
    """Raised when one chunk cannot be recovered after all split retries."""


class ChunkExtractor:
    """Extract minimal structured facts from transcript chunks via Groq."""

    def __init__(
        self,
        provider: BaseAIProvider,
        retry_fn,
        scheduler: ProviderRequestScheduler,
        checkpoint_store: ChunkCheckpointStore | None = None,
        max_output_tokens: int = 900,
        min_split_tokens: int = 180,
    ) -> None:
        self._provider = provider
        self._retry_fn = retry_fn
        self._scheduler = scheduler
        self._checkpoint_store = checkpoint_store
        self._max_output_tokens = max_output_tokens
        self._min_split_tokens = min_split_tokens
        self._validator = ValidationLayer()

    def extract_all(
        self,
        chunks: list[TranscriptChunk],
        title: str,
        date: str,
        chunk_checkpoint: Optional[dict[int, dict]] = None,
    ) -> list[ChunkExtraction]:
        """Extract and validate all chunks in order."""
        if chunk_checkpoint is None:
            chunk_checkpoint = {}

        extractions: list[ChunkExtraction] = []
        total = len(chunks)

        for chunk in chunks:
            cached = self._load_checkpoint(chunk.index, chunk_checkpoint)
            if cached is not None:
                logger.info("[ChunkExtractor] Chunk %d/%d loaded from checkpoint.", chunk.index + 1, total)
                extractions.append(cached)
                continue

            logger.info(
                "[ChunkExtractor] Extracting chunk %d/%d (%d tokens) via %s.",
                chunk.index + 1,
                total,
                chunk.estimated_tokens,
                self._provider.get_name(),
            )
            extraction = self._extract_with_recovery(chunk, title, date, depth=0)
            self._save_checkpoint(chunk.index, extraction, chunk_checkpoint)
            extractions.append(extraction)

        logger.info("[ChunkExtractor] Completed %d chunk extractions.", len(extractions))
        return extractions

    def extract_single(self, chunk: TranscriptChunk, title: str, date: str) -> ChunkExtraction:
        """Extract one chunk without using the batch checkpoint dictionary."""
        return self._extract_with_recovery(chunk, title, date, depth=0)

    def _extract_with_recovery(
        self,
        chunk: TranscriptChunk,
        title: str,
        date: str,
        depth: int,
    ) -> ChunkExtraction:
        """Retry a chunk by halving input when output is truncated or invalid."""
        try:
            return self._call_and_parse(chunk, title, date)
        except (AIProviderTruncatedResponseError, ValidationError, json.JSONDecodeError, ChunkExtractionError) as exc:
            if chunk.estimated_tokens <= self._min_split_tokens:
                logger.warning(
                    "[ChunkExtractor] Chunk %d too small to split after error: %s. Returning empty extraction.",
                    chunk.index,
                    exc,
                )
                return ChunkExtraction()

            logger.warning(
                "[ChunkExtractor] Chunk %d extraction hit size/JSON limit. Splitting by ~50%% and retrying.",
                chunk.index,
            )
            left, right = self._split_chunk(chunk)
            left_result = self._extract_with_recovery(left, title, date, depth + 1)
            right_result = self._extract_with_recovery(right, title, date, depth + 1)
            return self._combine_extractions([left_result, right_result])

    def _call_and_parse(self, chunk: TranscriptChunk, title: str, date: str) -> ChunkExtraction:
        prompt = self._build_user_prompt(chunk, title, date)
        request_tokens = estimate_tokens(CHUNK_EXTRACTION_SYSTEM_PROMPT) + estimate_tokens(prompt) + self._max_output_tokens

        self._scheduler.acquire(request_tokens)
        failed = False
        rate_limited = False
        try:
            logger.info("[ChunkExtractor] Sleeping 20 seconds to prevent exceeding rate limits...")
            time.sleep(20)
            raw_response = self._retry_fn(
                provider=self._provider,
                system_prompt=CHUNK_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=self._max_output_tokens,
            )
            extraction = self._parse_extraction(raw_response, self._validator)
            logger.info(
                "[ChunkExtractor] Extracted %d discussion points and %d actions from chunk %d.",
                len(extraction.discussion_points),
                len(extraction.action_items),
                chunk.index,
            )
            return extraction
        except AIProviderError as exc:
            failed = True
            rate_limited = "rate" in str(exc).lower() and "limit" in str(exc).lower()
            raise
        except Exception:
            failed = True
            raise
        finally:
            self._scheduler.release(failed=failed, rate_limited=rate_limited)

    @staticmethod
    def _build_user_prompt(chunk: TranscriptChunk, title: str, date: str) -> str:
        parts = [
            f"Meeting title: {title}",
            f"Meeting date: {date}",
            f"Chunk: {chunk.index + 1}",
        ]
        if chunk.speakers:
            parts.append(f"Speakers: {', '.join(chunk.speakers)}")
        if chunk.overlap_prefix:
            parts.append(
                "Overlap context, reference only:\n"
                f"{chunk.overlap_prefix}\n"
                "End overlap context."
            )
        parts.append(f"Transcript chunk:\n{chunk.text}")
        return "\n".join(parts)

    @staticmethod
    def _parse_extraction(raw_response: str, validator: ValidationLayer) -> ChunkExtraction:
        cleaned = raw_response.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ChunkExtractionError("No complete JSON object found in chunk response.")

        cleaned = cleaned[start : end + 1]
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        data = json.loads(cleaned)
        return validator.validate_chunk(data)

    @staticmethod
    def _split_chunk(chunk: TranscriptChunk) -> tuple[TranscriptChunk, TranscriptChunk]:
        lines = [line for line in chunk.text.splitlines() if line.strip()]
        if len(lines) < 2:
            words = chunk.text.split()
            midpoint = max(1, len(words) // 2)
            left_text = " ".join(words[:midpoint])
            right_text = " ".join(words[midpoint:])
        else:
            midpoint = max(1, len(lines) // 2)
            left_text = "\n".join(lines[:midpoint])
            right_text = "\n".join(lines[midpoint:])

        left = TranscriptChunk(
            index=chunk.index,
            text=left_text,
            speakers=chunk.speakers,
            estimated_tokens=estimate_tokens(left_text),
            overlap_prefix=chunk.overlap_prefix,
        )
        right = TranscriptChunk(
            index=chunk.index,
            text=right_text,
            speakers=chunk.speakers,
            estimated_tokens=estimate_tokens(right_text),
            overlap_prefix=left_text.splitlines()[-1] if left_text else "",
        )
        return left, right

    @staticmethod
    def _combine_extractions(items: list[ChunkExtraction]) -> ChunkExtraction:
        combined = ChunkExtraction()
        for item in items:
            combined.discussion_points.extend(item.discussion_points)
            combined.action_items.extend(item.action_items)
            combined.decisions.extend(item.decisions)
            combined.risks.extend(item.risks)
            combined.blockers.extend(item.blockers)
            combined.questions.extend(item.questions)
            combined.deadlines.extend(item.deadlines)
            combined.participants.extend(item.participants)
        return combined

    def _load_checkpoint(
        self,
        index: int,
        memory_checkpoint: dict[int, dict],
    ) -> ChunkExtraction | None:
        data = memory_checkpoint.get(index)
        if data is None and self._checkpoint_store is not None:
            data = self._checkpoint_store.load(index)
        if data is None:
            return None
        try:
            return ChunkExtraction(**data)
        except Exception as exc:
            logger.warning("[ChunkExtractor] Ignoring invalid checkpoint for chunk %d: %s", index, exc)
            return None

    def _save_checkpoint(
        self,
        index: int,
        extraction: ChunkExtraction,
        memory_checkpoint: dict[int, dict],
    ) -> None:
        data = extraction.model_dump()
        memory_checkpoint[index] = data
        if self._checkpoint_store is not None:
            self._checkpoint_store.save(index, data)
