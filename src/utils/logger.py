"""
Centralised logging configuration for the entire project.
All modules import `get_logger(__name__)` — never print() directly.
"""
import sys
import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a logger that writes to both the console and a dated log file.

    On Windows the default console encoding (cp1252) cannot represent Unicode
    characters used in log messages (arrows, tick marks, etc.).
    reconfigure() forces UTF-8 so those characters print correctly.

    Args:
        name:    Module name  (use __name__ when calling).
        log_dir: Directory where log files are stored.

    Returns:
        A configured logging.Logger instance.
    """
    # ── Force UTF-8 on Windows console ──────────────────────────────────────
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (INFO+) ──────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler (DEBUG+) — always UTF-8 regardless of OS ────────────────
    log_file = os.path.join(
        log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_drivable_area.log"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
