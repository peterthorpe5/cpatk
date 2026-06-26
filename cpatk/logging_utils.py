"""Logging helpers for CPATK."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    *,
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    logger_name: str = "cpatk",
) -> logging.Logger:
    """Configure and return a CPATK logger.

    Parameters
    ----------
    log_file:
        Optional path to a log file. Parent directories are created when
        needed.
    log_level:
        Logging level name, for example ``INFO`` or ``DEBUG``.
    logger_name:
        Name of the logger to configure.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(level=getattr(logging, log_level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(fmt=formatter)
    stream_handler.setLevel(level=logger.level)
    logger.addHandler(hdlr=stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file, mode="w")
        file_handler.setFormatter(fmt=formatter)
        file_handler.setLevel(level=logger.level)
        logger.addHandler(hdlr=file_handler)

    return logger
