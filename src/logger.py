"""
logger.py
---------
Centralised logging setup for the Card-Retention-Intelligence pipeline.
Uses loguru for structured, timestamped logs written to both
the console (stdout) and a persistent log file.

Every module in src/ imports get_logger() from here — this ensures
all phases write to the same log file and follow the same format.
"""

import sys
from pathlib import Path
from loguru import logger


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # project root
LOG_DIR  = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Log format ────────────────────────────────────────────────────────────────
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)


def get_logger(name: str = "pipeline"):
    """
    Returns a loguru logger bound to a specific module name.
    Configures:
      - Console handler  : INFO level, coloured output
      - File handler     : DEBUG level, full detail, auto-rotated at 10 MB
    Safe to call multiple times — handlers are only added once.
    """
    # Remove any default handlers loguru adds at import time
    logger.remove()

    # Console — INFO and above (clean output during runs)
    logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level="INFO",
        colorize=True,
    )

    # File — DEBUG and above (full detail for post-mortem inspection)
    logger.add(
        LOG_FILE,
        format=LOG_FORMAT,
        level="DEBUG",
        rotation="10 MB",      # start a new file after 10 MB
        retention="30 days",   # keep logs for 30 days
        compression="zip",     # compress rotated logs
        enqueue=True,          # thread-safe writes
    )

    return logger.bind(module=name)
