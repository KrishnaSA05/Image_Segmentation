"""
Centralised logging configuration for the entire project.
All modules import `get_logger(__name__)` — never print() directly.
"""
import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a logger that writes to both the console and a dated log file.

    Args:
        name:    Module name  (use __name__ when calling).
        log_dir: Directory where log files are stored.

    Returns:
        A configured logging.Logger instance.
    """
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
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler (DEBUG+) ────────────────────────────────────────────────
    log_file = os.path.join(
        log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_drivable_area.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
