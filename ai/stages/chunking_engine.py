"""Stage 2 — Chunking Engine.

Context-aware transcript splitter that preserves speaker turns
and topic coherence, with configurable overlap for boundary context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Token estimator (mirrors manager.estimate_tokens) ─────────────────────
def _estimate_tokens(text: str) -> int:
    """Estimate token count (≈1.33 tokens per word or 4 chars per token)."""
    if not text:
        return 0
    return max(int(len(text.split()) * 1.33), int(len(text) / 4.0))


# ── Speaker-turn detection ────────────────────────────────────────────────
_SPEAKER_LINE_RE = re.compile(r"^(Speaker\s+\d+)\s*:", re.IGNORECASE | re.MULTILINE)

# ── Topic-shift heuristic keywords ───────────────────────────────────────
_TOPIC_SHIFT_CUES = re.compile(
    r"(?:moving on|next topic|let's talk about|switching to|another point|"
    r"on a different note|next item|let's discuss|agenda item|"
    r"now let's|coming to|regarding|about the|the next thing)",
    re.IGNORECASE,
)


@dataclass
class TranscriptChunk:
    """One context-preserving segment of the transcript."""

    index: int
    text: str
    speakers: List[str] = field(default_factory=list)
    estimated_tokens: int = 0
    overlap_prefix: str = ""  # trailing context from the previous chunk


class ChunkingEngine:
    """Context-aware chunker (Stage 2 of the pipeline).

    Splitting priorities (highest → lowest):
      1. Topic-shift cues (keywords like "next topic", "moving on")
      2. Speaker-turn boundaries (never split mid-turn)
      3. Paragraph / blank-line boundaries
      4. Hard token limit (last resort — split at sentence end)

    Each chunk carries a small overlap prefix from the previous chunk
    so the LLM has boundary context and doesn't miss cross-boundary info.
    """

    DEFAULT_OVERLAP_LINES = 3

    def __init__(self, overlap_lines: int = DEFAULT_OVERLAP_LINES) -> None:
        self._overlap_lines = overlap_lines

    def chunk(
        self,
        transcript: str,
        max_chunk_tokens: int,
    ) -> List[TranscriptChunk]:
        """Split the transcript into context-preserving chunks.

        Args:
            transcript: Cleaned transcript text (output of Stage 1).
            max_chunk_tokens: Maximum token budget per chunk (excluding overlap).

        Returns:
            Ordered list of TranscriptChunk objects.
        """
        if not transcript or not transcript.strip():
            return []

        # 1. Segment into speaker turns (groups of lines belonging to the same speaker)
        turns = self._segment_into_turns(transcript)

        # 2. Pack turns into chunks respecting the token budget
        chunks = self._pack_turns(turns, max_chunk_tokens)

        logger.info(
            "[ChunkingEngine] Split transcript into %d chunks (max %d tokens each, overlap=%d lines)",
            len(chunks),
            max_chunk_tokens,
            self._overlap_lines,
        )
        return chunks

    # ── Internal helpers ──────────────────────────────────────────────────

    def _segment_into_turns(self, transcript: str) -> List[str]:
        """Break transcript into speaker turns (or paragraph blocks)."""
        lines = transcript.split("\n")
        turns: List[str] = []
        current_turn_lines: List[str] = []

        for line in lines:
            # A new speaker label starts a new turn
            if _SPEAKER_LINE_RE.match(line) and current_turn_lines:
                turns.append("\n".join(current_turn_lines))
                current_turn_lines = [line]
            # A blank line after content can also be a turn break
            elif not line.strip() and current_turn_lines and any(l.strip() for l in current_turn_lines):
                current_turn_lines.append(line)
                # Check if next non-blank starts a new speaker — we'll handle that above
            else:
                current_turn_lines.append(line)

        if current_turn_lines:
            turns.append("\n".join(current_turn_lines))

        # Filter out fully-empty turns
        turns = [t for t in turns if t.strip()]
        return turns

    def _pack_turns(
        self,
        turns: List[str],
        max_chunk_tokens: int,
    ) -> List[TranscriptChunk]:
        """Greedily pack speaker turns into chunks within the token budget."""
        chunks: List[TranscriptChunk] = []
        current_turns: List[str] = []
        current_tokens = 0
        chunk_index = 0
        previous_tail_lines: List[str] = []

        for turn in turns:
            turn_tokens = _estimate_tokens(turn)

            # If a single turn exceeds the budget, it gets its own chunk
            if turn_tokens > max_chunk_tokens:
                # Flush current accumulator first
                if current_turns:
                    chunks.append(self._make_chunk(
                        chunk_index, current_turns, previous_tail_lines,
                    ))
                    previous_tail_lines = self._get_tail_lines(current_turns)
                    chunk_index += 1
                    current_turns = []
                    current_tokens = 0

                # Split the oversized turn at sentence boundaries
                sub_chunks = self._split_oversized_turn(turn, max_chunk_tokens)
                for sub in sub_chunks:
                    chunks.append(self._make_chunk(
                        chunk_index, [sub], previous_tail_lines,
                    ))
                    previous_tail_lines = self._get_tail_lines([sub])
                    chunk_index += 1
                continue

            # Check for topic-shift cue at the start of this turn
            is_topic_shift = bool(_TOPIC_SHIFT_CUES.search(turn[:200]))

            # Would adding this turn exceed the budget, or is it a topic shift with content?
            if current_tokens + turn_tokens > max_chunk_tokens or (
                is_topic_shift and current_turns and current_tokens > max_chunk_tokens * 0.3
            ):
                # Flush current chunk
                if current_turns:
                    chunks.append(self._make_chunk(
                        chunk_index, current_turns, previous_tail_lines,
                    ))
                    previous_tail_lines = self._get_tail_lines(current_turns)
                    chunk_index += 1

                current_turns = [turn]
                current_tokens = turn_tokens
            else:
                current_turns.append(turn)
                current_tokens += turn_tokens

        # Flush remaining
        if current_turns:
            chunks.append(self._make_chunk(
                chunk_index, current_turns, previous_tail_lines,
            ))

        return chunks

    def _make_chunk(
        self,
        index: int,
        turns: List[str],
        previous_tail_lines: List[str],
    ) -> TranscriptChunk:
        """Build a TranscriptChunk from accumulated turns."""
        text = "\n".join(turns)
        overlap = "\n".join(previous_tail_lines) if previous_tail_lines else ""

        # Extract speakers mentioned in this chunk
        speakers = list(set(_SPEAKER_LINE_RE.findall(text)))

        return TranscriptChunk(
            index=index,
            text=text,
            speakers=speakers,
            estimated_tokens=_estimate_tokens(text),
            overlap_prefix=overlap,
        )

    def _get_tail_lines(self, turns: List[str]) -> List[str]:
        """Extract the last N lines from the turns for overlap context."""
        all_lines = "\n".join(turns).split("\n")
        return all_lines[-self._overlap_lines:] if len(all_lines) >= self._overlap_lines else all_lines

    @staticmethod
    def _split_oversized_turn(turn: str, max_tokens: int) -> List[str]:
        """Split a single oversized turn at sentence boundaries."""
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", turn)
        sub_chunks: List[str] = []
        current: List[str] = []
        current_tok = 0

        for sentence in sentences:
            s_tok = _estimate_tokens(sentence)
            if current_tok + s_tok > max_tokens and current:
                sub_chunks.append(" ".join(current))
                current = [sentence]
                current_tok = s_tok
            else:
                current.append(sentence)
                current_tok += s_tok

        if current:
            sub_chunks.append(" ".join(current))

        return sub_chunks
