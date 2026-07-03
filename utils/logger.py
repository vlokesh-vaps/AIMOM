"""Centralized logging configuration.

Provides a rotating file handler (logs/app.log) and a console handler.
Every module calls ``get_logger(__name__)`` to obtain a child logger.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES

_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
_INITIALIZED: bool = False


def _init_root_logger() -> None:
    """Configure the root logger once on first call."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    log_dir: Path = LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initializing root config on first use.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    _init_root_logger()
    return logging.getLogger(name)
