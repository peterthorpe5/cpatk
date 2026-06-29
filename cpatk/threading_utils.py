"""Thread-control helpers for CPATK command-line workflows."""

from __future__ import annotations

import logging
import os
from typing import Optional

_THREADPOOL_LIMITER = None


def normalise_thread_count(*, value: object, default: int = 1) -> int:
    """Return a safe positive thread count from a user-supplied value.

    Parameters
    ----------
    value:
        User supplied value, environment value, or ``None``.
    default:
        Fallback thread count when ``value`` is missing or invalid.

    Returns
    -------
    int
        A positive integer thread count.
    """
    try:
        threads = int(value)
    except (TypeError, ValueError):
        threads = int(default)
    return max(1, threads)


def configure_threading(
    *,
    n_threads: int,
    logger: Optional[logging.Logger] = None,
    use_threadpoolctl: bool = True,
) -> int:
    """Configure common native-library thread pools for a CPATK process.

    This function is intentionally conservative.  It sets common BLAS/OpenMP
    environment variables and, when ``threadpoolctl`` is available, applies a
    process-level limit to already-loaded thread pools.

    Parameters
    ----------
    n_threads:
        Requested maximum thread count.
    logger:
        Optional logger.
    use_threadpoolctl:
        Whether to try applying a live threadpoolctl limit.

    Returns
    -------
    int
        The normalised thread count actually requested.
    """
    global _THREADPOOL_LIMITER
    threads = normalise_thread_count(value=n_threads, default=1)
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ]
    for key in keys:
        os.environ[key] = str(threads)
    if use_threadpoolctl:
        try:
            from threadpoolctl import threadpool_limits  # type: ignore

            _THREADPOOL_LIMITER = threadpool_limits(limits=threads)
            _THREADPOOL_LIMITER.__enter__()
        except Exception as exc:  # pragma: no cover - optional dependency
            if logger is not None:
                logger.warning("threadpoolctl limit could not be applied: %s", exc)
    if logger is not None:
        logger.info("Configured CPATK thread count: %d", threads)
    return threads


def configure_torch_threads(*, n_threads: int, logger: Optional[logging.Logger] = None) -> int:
    """Configure PyTorch CPU thread pools when PyTorch is available.

    Parameters
    ----------
    n_threads:
        Requested PyTorch CPU thread count.
    logger:
        Optional logger.

    Returns
    -------
    int
        The normalised thread count requested from PyTorch.
    """
    threads = normalise_thread_count(value=n_threads, default=1)
    try:
        import torch  # type: ignore

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(threads, 4)))
        except RuntimeError:
            # PyTorch only allows changing inter-op threads before parallel work
            # has started.  This is harmless for CPATK runs that have already
            # initialised torch elsewhere.
            pass
    except Exception as exc:  # pragma: no cover - torch is optional
        if logger is not None:
            logger.warning("PyTorch thread configuration was skipped: %s", exc)
    if logger is not None:
        logger.info("Configured PyTorch thread count: %d", threads)
    return threads
