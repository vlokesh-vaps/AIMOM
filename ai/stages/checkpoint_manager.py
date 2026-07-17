"""Stage — Checkpoint Manager.

Provides content-hash based checkpointing for the async extraction pipeline.
Each completed chunk is saved as a JSON file with rich metadata so the pipeline
can resume from the last successful chunk after interruption.

Checkpoint files are named by a SHA-256 hash of the chunk text, so if only part
of the transcript changes, only the affected chunks are reprocessed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config.settings import CHECKPOINT_DIR, PROMPT_VERSION
from utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages chunk-level checkpoints for the extraction pipeline.

    Each checkpoint is identified by a content hash (SHA-256) of the chunk text,
    and stores both the extracted JSON data and rich metadata for debugging and
    deterministic recovery.
    """

    def __init__(self, session_id: str) -> None:
        """Initialize the checkpoint manager for a pipeline session.

        Args:
            session_id: Unique identifier for this pipeline run (e.g. meeting title hash).
                        Creates a subdirectory under CHECKPOINT_DIR.
        """
        self._session_dir = CHECKPOINT_DIR / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[CheckpointManager] Session dir: %s", self._session_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def content_hash(text: str) -> str:
        """Generate a SHA-256 hash of chunk text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def exists(self, chunk_hash: str) -> bool:
        """Check whether a checkpoint exists for the given chunk hash."""
        return self._checkpoint_path(chunk_hash).exists()

    def load(self, chunk_hash: str) -> dict[str, Any] | None:
        """Load a checkpoint by its content hash.

        Returns:
            The full checkpoint dict (with 'data' and 'metadata' keys),
            or None if the checkpoint doesn't exist or is incompatible.
        """
        path = self._checkpoint_path(chunk_hash)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            # Invalidate checkpoint if prompt version has changed
            stored_version = checkpoint.get("metadata", {}).get("prompt_version")
            if stored_version != PROMPT_VERSION:
                logger.info(
                    "[CheckpointManager] Checkpoint %s invalidated (prompt version %s != %s).",
                    chunk_hash, stored_version, PROMPT_VERSION,
                )
                return None

            logger.debug("[CheckpointManager] Loaded checkpoint %s.", chunk_hash)
            return checkpoint
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[CheckpointManager] Failed to load checkpoint %s: %s", chunk_hash, exc,
            )
            return None

    def save(
        self,
        chunk_hash: str,
        chunk_index: int,
        data: dict[str, Any],
        provider: str = "",
        model: str = "",
        retry_count: int = 0,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Save a checkpoint for a completed chunk.

        Args:
            chunk_hash: SHA-256 content hash of the chunk.
            chunk_index: Sequential index of the chunk (for ordering).
            data: The structured JSON extraction result.
            provider: Which provider fulfilled the request.
            model: Which model was used.
            retry_count: How many retries were needed.
            token_usage: Optional dict with prompt_tokens, completion_tokens.
        """
        checkpoint = {
            "data": data,
            "metadata": {
                "chunk_hash": chunk_hash,
                "chunk_index": chunk_index,
                "provider": provider,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "retry_count": retry_count,
                "token_usage": token_usage or {},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }

        path = self._checkpoint_path(chunk_hash)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            logger.info(
                "[CheckpointManager] Saved checkpoint %s (chunk %d, provider=%s).",
                chunk_hash, chunk_index, provider,
            )
        except OSError as exc:
            logger.error(
                "[CheckpointManager] Failed to save checkpoint %s: %s", chunk_hash, exc,
            )

    def load_all_completed(self, expected_hashes: list[str]) -> dict[str, dict[str, Any]]:
        """Load all existing checkpoints that match the expected chunk hashes.

        Args:
            expected_hashes: List of chunk hashes that are expected for the current run.

        Returns:
            Dict mapping chunk_hash → checkpoint data for existing valid checkpoints.
        """
        completed: dict[str, dict[str, Any]] = {}
        for h in expected_hashes:
            checkpoint = self.load(h)
            if checkpoint is not None:
                completed[h] = checkpoint
        if completed:
            logger.info(
                "[CheckpointManager] Resuming — found %d/%d completed checkpoints.",
                len(completed), len(expected_hashes),
            )
        return completed

    def cleanup(self) -> None:
        """Remove all checkpoints for this session (called after successful completion)."""
        try:
            for f in self._session_dir.glob("*.json"):
                f.unlink()
            self._session_dir.rmdir()
            logger.info("[CheckpointManager] Cleaned up session dir: %s", self._session_dir)
        except OSError as exc:
            logger.warning("[CheckpointManager] Cleanup failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _checkpoint_path(self, chunk_hash: str) -> Path:
        return self._session_dir / f"chunk_{chunk_hash}.json"
