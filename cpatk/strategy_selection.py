"""Preprocessing/normalisation strategy comparison helpers for CPATK."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cpatk.io import read_table, write_table


def _read_optional_table(*, path: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read a table if it exists, otherwise return an empty data frame."""
    if path.exists() and path.stat().st_size > 0:
        try:
            return read_table(path=path, logger=logger)
        except Exception as exc:  # pragma: no cover - defensive for damaged partial runs
            if logger is not None:
                logger.warning("Could not read strategy table %s: %s", path, exc)
    return pd.DataFrame()


def _summary_value(summary: pd.DataFrame, item: str, default: object = np.nan) -> object:
    """Extract an item/value entry from a CPATK summary table."""
    if summary.empty or not {"item", "value"}.issubset(summary.columns):
        return default
    hits = summary.loc[summary["item"].astype(str) == item, "value"]
    if hits.empty:
        return default
    return hits.iloc[0]


def _to_float(value: object, default: float = np.nan) -> float:
    """Convert a possibly string value to float."""
    try:
        return float(value)
    except Exception:
        return default


def _median_replicate_correlation(table: pd.DataFrame) -> float:
    """Return the median after-batch replicate correlation where available."""
    if table.empty or "median_correlation" not in table.columns:
        return np.nan
    data = table.copy()
    if "stage" in data.columns:
        data = data.loc[data["stage"].astype(str) == "after_batch_correction"]
    values = pd.to_numeric(data["median_correlation"], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _eta_for_column(table: pd.DataFrame, column: str) -> float:
    """Return the mean PC1/PC2 eta-squared for a metadata column."""
    if table.empty or "metadata_column" not in table.columns or "eta_squared" not in table.columns:
        return np.nan
    status = table["status"].astype(str) if "status" in table.columns else pd.Series("tested", index=table.index)
    data = table.loc[
        (table["metadata_column"].astype(str) == column)
        & (status == "tested")
    ].copy()
    if "stage" in data.columns:
        data = data.loc[data["stage"].astype(str) == "after_batch_correction"]
    values = pd.to_numeric(data["eta_squared"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _control_status(table: pd.DataFrame) -> str:
    """Summarise reference-control QC status."""
    if table.empty or "status" not in table.columns:
        return "not_reported"
    values = table["status"].dropna().astype(str).str.lower().tolist()
    if not values:
        return "not_reported"
    if all(value == "ok" for value in values):
        return "ok"
    if any("fail" in value or "missing" in value or "not" in value for value in values):
        return "review"
    return "mixed"


def summarise_preprocessing_strategies(
    *,
    strategy_root: Path,
    batch_column: str = "Metadata_Plate",
    compound_column: str = "Metadata_Compound",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Create a compact comparison table for preprocessing strategies.

    Parameters
    ----------
    strategy_root:
        Directory containing one subdirectory per preprocessing strategy.
    batch_column:
        Metadata column used as the primary technical/batch association metric.
    compound_column:
        Metadata column used as the primary biological/treatment association metric.
    logger:
        Optional logger.

    Returns
    -------
    pandas.DataFrame
        Strategy-level metrics and a cautious qualitative recommendation.
    """
    strategy_root = Path(strategy_root)
    if not strategy_root.exists():
        return pd.DataFrame.from_records(
            [{"strategy": "", "status": "not_tested_strategy_root_missing", "strategy_root": str(strategy_root)}]
        )
    records: list[dict[str, object]] = []
    for strategy_dir in sorted(path for path in strategy_root.iterdir() if path.is_dir()):
        summary = _read_optional_table(path=strategy_dir / "preprocessing_summary.tsv", logger=logger)
        matrix = _read_optional_table(path=strategy_dir / "final_matrix_validation.tsv", logger=logger)
        control = _read_optional_table(path=strategy_dir / "control_qc_before_normalisation.tsv", logger=logger)
        rep = _read_optional_table(path=strategy_dir / "before_after_replicate_summary.tsv", logger=logger)
        batch = _read_optional_table(path=strategy_dir / "before_after_batch_pc_association.tsv", logger=logger)
        rows_input = _to_float(_summary_value(summary, "n_rows_input"))
        rows_final = _to_float(_summary_value(summary, "n_rows_passing_qc"))
        features_input = _to_float(_summary_value(summary, "n_features_input"))
        features_final = _to_float(_summary_value(summary, "n_features_after_correlation_filter"))
        matrix_status = "not_reported"
        if not matrix.empty and "status" in matrix.columns:
            matrix_status = ";".join(sorted(matrix["status"].dropna().astype(str).unique()))
        batch_eta = _eta_for_column(batch, batch_column)
        compound_eta = _eta_for_column(batch, compound_column)
        replicate_corr = _median_replicate_correlation(rep)
        score = 0.0
        score += 2.0 if matrix_status == "ok" else 0.0
        score += 1.0 if _control_status(control) == "ok" else 0.0
        if np.isfinite(replicate_corr):
            score += max(min((replicate_corr + 1.0) / 2.0, 1.0), 0.0)
        if np.isfinite(batch_eta):
            score += max(1.0 - min(batch_eta, 1.0), 0.0)
        if np.isfinite(compound_eta):
            score += min(compound_eta, 1.0)
        if matrix_status != "ok":
            recommendation = "do_not_use_matrix_not_ok"
        elif _control_status(control) == "review":
            recommendation = "review_reference_controls"
        elif np.isfinite(batch_eta) and batch_eta > 0.25:
            recommendation = "review_batch_structure"
        else:
            recommendation = "candidate_strategy"
        records.append(
            {
                "strategy": strategy_dir.name,
                "matrix_status": matrix_status,
                "control_qc_status": _control_status(control),
                "rows_input": rows_input,
                "rows_final": rows_final,
                "row_retention_fraction": rows_final / rows_input if rows_input else np.nan,
                "features_input": features_input,
                "features_final": features_final,
                "feature_retention_fraction": features_final / features_input if features_input else np.nan,
                "median_replicate_correlation_after": replicate_corr,
                f"mean_eta_squared_{batch_column}": batch_eta,
                f"mean_eta_squared_{compound_column}": compound_eta,
                "selection_score_heuristic": score,
                "recommendation": recommendation,
            }
        )
    if not records:
        return pd.DataFrame.from_records(
            [{"strategy": "", "status": "not_tested_no_strategy_directories", "strategy_root": str(strategy_root)}]
        )
    result = pd.DataFrame.from_records(records)
    result = result.sort_values("selection_score_heuristic", ascending=False, na_position="last")
    result.insert(0, "rank", range(1, result.shape[0] + 1))
    return result


def write_preprocessing_strategy_summary(
    *,
    strategy_root: Path,
    output_path: Path,
    batch_column: str = "Metadata_Plate",
    compound_column: str = "Metadata_Compound",
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write a preprocessing strategy comparison table."""
    table = summarise_preprocessing_strategies(
        strategy_root=strategy_root,
        batch_column=batch_column,
        compound_column=compound_column,
        logger=logger,
    )
    return write_table(data_frame=table, path=output_path, logger=logger)
