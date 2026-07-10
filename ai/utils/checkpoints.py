"""Disk-backed checkpoint storage for chunk extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import TEMP_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class ChunkCheckpointStore:
    """Persist each completed chunk extraction immediately as JSON."""

    def __init__(self, run_id: str, base_dir: Path | None = None) -> None:
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_id)
        self.directory = (base_dir or TEMP_DIR) / "ai_checkpoints" / safe_id
        self.directory.mkdir(parents=True, exist_ok=True)

    def has(self, index: int) -> bool:
        return self._path(index).exists()

    def load(self, index: int) -> dict[str, Any] | None:
        path = self._path(index)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[Checkpoint] Could not read %s: %s", path, exc)
            return None

    def save(self, index: int, data: dict[str, Any]) -> None:
        path = self._path(index)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _path(self, index: int) -> Path:
        return self.directory / f"chunk_{index:04d}.json"
