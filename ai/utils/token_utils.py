"""Shared token estimation helpers for pipeline sizing and scheduling."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate token count using a conservative character/word heuristic."""
    if not text:
        return 0
    return max(int(len(text.split()) * 1.33), int(len(text) / 4.0))


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer value to an inclusive range."""
    return max(minimum, min(value, maximum))
