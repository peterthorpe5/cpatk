"""Optional AI/CLIPn integration helpers.

This module deliberately treats AI workflows as optional. Classical profiling,
distance, clustering and MOA workflows should remain usable without CLIPn or any
other AI dependency being installed.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class AiAvailability:
    """Describe whether an optional AI backend is available."""

    backend_name: str
    available: bool
    message: str


def check_backend_availability(*, backend_name: str = "clipn") -> AiAvailability:
    """Check whether an optional AI backend can be imported.

    Parameters
    ----------
    backend_name:
        Python module name for the backend.

    Returns
    -------
    AiAvailability
        Availability status and message.
    """
    try:
        importlib.import_module(name=backend_name)
        return AiAvailability(
            backend_name=backend_name,
            available=True,
            message=f"Backend {backend_name} is importable.",
        )
    except Exception as exc:
        return AiAvailability(
            backend_name=backend_name,
            available=False,
            message=f"Backend {backend_name} is not available: {exc}",
        )


def make_ai_status_table(*, backend_name: str = "clipn") -> pd.DataFrame:
    """Create a one-row table describing optional AI backend availability.

    Parameters
    ----------
    backend_name:
        Backend module name.

    Returns
    -------
    pandas.DataFrame
        AI backend status table.
    """
    status = check_backend_availability(backend_name=backend_name)
    return pd.DataFrame.from_records(
        [
            {
                "backend_name": status.backend_name,
                "available": status.available,
                "message": status.message,
            }
        ]
    )


def run_optional_clipn_placeholder(
    *,
    input_tables: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """Placeholder entry point for future CLIPn execution.

    Parameters
    ----------
    input_tables:
        Named input tables.
    config:
        Optional CLIPn configuration.
    logger:
        Optional logger.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Tables describing why CLIPn was or was not run.

    Notes
    -----
    The first CPATK release keeps CLIPn integration optional and explicit.
    This function is intentionally conservative so the package can be used
    without AI dependencies. A future version can replace this with a concrete
    adapter around the preferred CLIPn implementation.
    """
    status = check_backend_availability(backend_name="clipn")
    if logger is not None:
        logger.info(status.message)
    summary = pd.DataFrame.from_records(
        [
            {
                "mode": "optional_ai_clipn",
                "backend_available": status.available,
                "n_input_tables": len(input_tables),
                "message": status.message,
                "next_step": "Use cpatk classical workflow now; add configured CLIPn adapter in a later CPATK release.",
            }
        ]
    )
    return {"ai_status": summary}
