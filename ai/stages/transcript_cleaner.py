"""Stage 1 — Transcript Cleaner.

Normalizes raw transcript text before any LLM processing:
  • Encoding fixes (BOM, mojibake, Unicode NFC)
  • Whitespace collapse (blank lines, trailing spaces, indentation)
  • Filler-word removal (um, uh, you know, basically, …)
  • Speaker-label standardization (→ "Speaker N:" format)
  • Timestamp normalization (→ "[HH:MM:SS]" format)
  • Stutter / repeated-word removal
  • Empty-turn pruning
"""

import re
import unicodedata
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Filler patterns ──────────────────────────────────────────────────────
# Matches common English filler words/phrases at word boundaries.
# Anchored so they don't eat real words like "human" (contains "um").
_FILLER_WORDS = [
    r"\bum+\b",
    r"\buh+\b",
    r"\bah+\b",
    r"\bhmm+\b",
    r"\beh+\b",
    r"\byou know\b",
    r"\blike\b(?=\s*,)",         # "like," as filler, not "I like cats"
    r"\bbasically\b",
    r"\bi mean\b",
    r"\bso+\b(?=\s*,)",          # "so," as filler
    r"\bactually\b(?=\s*,)",     # "actually," as filler
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bright\b(?=\s*[,?])",     # "right," / "right?" as filler tag
]
_FILLER_RE = re.compile("|".join(_FILLER_WORDS), re.IGNORECASE)

# ── Speaker-label patterns ────────────────────────────────────────────────
# Matches variants like "Speaker1:", "SPEAKER 1:", "speaker_1:", "[Speaker 1]"
_SPEAKER_PATTERNS = [
    # [Speaker 1] or [SPEAKER 1]
    re.compile(r"\[\s*speaker[\s_-]*(\d+)\s*\]", re.IGNORECASE),
    # Speaker1: or SPEAKER_1: or speaker-1:
    re.compile(r"(?:^|\n)\s*speaker[\s_-]*(\d+)\s*:", re.IGNORECASE),
]

# ── Timestamp patterns ───────────────────────────────────────────────────
# Matches (5:30), [5:30], 0:05:30, [00:05:30], (00:05:30)
_TIMESTAMP_RE = re.compile(
    r"[\[\(]?\s*(\d{0,2}):?(\d{1,2}):(\d{2})\s*[\]\)]?"
)

# ── Repeated words ────────────────────────────────────────────────────────
_STUTTER_RE = re.compile(r"\b(\w+)(?:\s+\1){1,}\b", re.IGNORECASE)


class TranscriptCleaner:
    """Pure-Python transcript normalizer (Stage 1 of the pipeline)."""

    def clean(self, raw_transcript: str) -> str:
        """Run the full cleaning pipeline and return normalized text."""
        if not raw_transcript or not raw_transcript.strip():
            return ""

        text = raw_transcript

        text = self._fix_encoding(text)
        text = self._normalize_speaker_labels(text)
        text = self._normalize_timestamps(text)
        text = self._remove_fillers(text)
        text = self._remove_stutters(text)
        text = self._collapse_whitespace(text)
        text = self._remove_empty_turns(text)

        logger.info(
            "[TranscriptCleaner] Cleaned transcript: %d chars -> %d chars (%.1f%% reduction)",
            len(raw_transcript),
            len(text),
            (1 - len(text) / max(len(raw_transcript), 1)) * 100,
        )
        return text.strip()

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Strip BOM, normalize to NFC Unicode."""
        # Remove BOM if present
        text = text.lstrip("\ufeff")
        # NFC normalization (composed form - most interoperable)
        text = unicodedata.normalize("NFC", text)
        # Replace common mojibake sequences using Unicode escapes
        replacements = {
            "\u00e2\u0080\u0099": "'",
            "\u00e2\u0080\u009c": '"',
            "\u00e2\u0080\u009d": '"',
            "\u00e2\u0080\u0094": "-",
            "\u00e2\u0080\u0093": "-",
            "\u00e2\u0080\u00a6": "...",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    @staticmethod
    def _normalize_speaker_labels(text: str) -> str:
        """Standardize speaker labels to 'Speaker N:' format."""
        # Handle [Speaker N] → Speaker N:
        text = re.sub(
            r"\[\s*speaker[\s_-]*(\d+)\s*\]",
            lambda m: f"Speaker {m.group(1)}:",
            text,
            flags=re.IGNORECASE,
        )
        # Handle SPEAKER_1: or speaker-1: → Speaker 1: (line-start only)
        text = re.sub(
            r"^(\s*)speaker[\s_-]*(\d+)\s*:",
            lambda m: f"{m.group(1)}Speaker {m.group(2)}:",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return text

    @staticmethod
    def _normalize_timestamps(text: str) -> str:
        """Standardize timestamps to [HH:MM:SS] format."""
        def _fmt(m: re.Match) -> str:
            hours = m.group(1).zfill(2) if m.group(1) else "00"
            minutes = m.group(2).zfill(2)
            seconds = m.group(3).zfill(2)
            return f"[{hours}:{minutes}:{seconds}]"

        return _TIMESTAMP_RE.sub(_fmt, text)

    @staticmethod
    def _remove_fillers(text: str) -> str:
        """Remove filler words/phrases."""
        cleaned = _FILLER_RE.sub("", text)
        # Collapse double-spaces left by removals
        cleaned = re.sub(r"  +", " ", cleaned)
        # Remove orphan commas left after filler removal: ", ,"  → ","
        cleaned = re.sub(r",\s*,", ",", cleaned)
        return cleaned

    @staticmethod
    def _remove_stutters(text: str) -> str:
        """Collapse repeated adjacent words: 'the the the' → 'the'."""
        return _STUTTER_RE.sub(r"\1", text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse multiple blank lines, strip trailing spaces per line."""
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        # Collapse 3+ consecutive blank lines into 1
        result = []
        blank_count = 0
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 1:
                    result.append("")
            else:
                blank_count = 0
                result.append(line)
        return "\n".join(result)

    @staticmethod
    def _remove_empty_turns(text: str) -> str:
        """Remove speaker turns that have no content after the label."""
        # Matches lines like "Speaker 1:" followed by nothing meaningful
        return re.sub(
            r"^Speaker\s+\d+:\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )
