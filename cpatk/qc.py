"""Quality-control utilities for Cell Painting profiles."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def calculate_feature_qc(
    *,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate feature-level QC metrics.

    Parameters
    ----------
    features:
        Numeric feature matrix.

    Returns
    -------
    pandas.DataFrame
        Feature-level QC table.
    """
    records = []
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        records.append(
            {
                "feature": column,
                "missing_fraction": float(values.isna().mean()),
                "variance": float(values.var(skipna=True)),
                "sd": float(values.std(skipna=True)),
                "median": float(values.median(skipna=True)),
                "mad": float((values - values.median(skipna=True)).abs().median(skipna=True)),
                "n_unique": int(values.nunique(dropna=True)),
                "near_zero_variance": bool(values.var(skipna=True) <= 1e-12),
                "all_missing": bool(values.notna().sum() == 0),
            }
        )
    return pd.DataFrame.from_records(records)


def calculate_sample_qc(
    *,
    features: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Calculate sample- or object-level QC metrics.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    metadata:
        Optional metadata to append to the QC table.

    Returns
    -------
    pandas.DataFrame
        Row-level QC table.
    """
    qc = pd.DataFrame(
        {
            "row_index": np.arange(features.shape[0]),
            "missing_fraction": features.isna().mean(axis=1).to_numpy(),
            "n_missing_features": features.isna().sum(axis=1).to_numpy(),
            "n_non_missing_features": features.notna().sum(axis=1).to_numpy(),
        }
    )
    if metadata is not None:
        qc = pd.concat([metadata.reset_index(drop=True), qc], axis=1)
    return qc


def select_features_by_qc(
    *,
    feature_qc: pd.DataFrame,
    max_missing_fraction: float = 0.2,
    min_variance: float = 1e-12,
    min_unique_values: int = 2,
) -> Tuple[List[str], pd.DataFrame]:
    """Select features passing missingness and variance thresholds.

    Parameters
    ----------
    feature_qc:
        Feature-level QC table from :func:`calculate_feature_qc`.
    max_missing_fraction:
        Maximum allowed fraction of missing values.
    min_variance:
        Minimum allowed variance.
    min_unique_values:
        Minimum number of unique non-missing values.

    Returns
    -------
    tuple[list[str], pandas.DataFrame]
        Selected feature names and feature QC table with status columns.
    """
    qc = feature_qc.copy()
    qc["pass_missingness"] = qc["missing_fraction"] <= max_missing_fraction
    qc["pass_variance"] = qc["variance"].fillna(0) > min_variance
    qc["pass_unique_values"] = qc["n_unique"] >= min_unique_values
    qc["feature_qc_pass"] = qc[
        ["pass_missingness", "pass_variance", "pass_unique_values"]
    ].all(axis=1)
    selected = qc.loc[qc["feature_qc_pass"], "feature"].astype(str).tolist()
    return selected, qc


def flag_samples_by_qc(
    *,
    sample_qc: pd.DataFrame,
    max_missing_fraction: float = 0.5,
) -> pd.DataFrame:
    """Flag rows passing sample-level QC.

    Parameters
    ----------
    sample_qc:
        Sample-level QC table from :func:`calculate_sample_qc`.
    max_missing_fraction:
        Maximum allowed row missingness.

    Returns
    -------
    pandas.DataFrame
        QC table with pass/fail columns.
    """
    qc = sample_qc.copy()
    qc["pass_missingness"] = qc["missing_fraction"] <= max_missing_fraction
    qc["sample_qc_pass"] = qc["pass_missingness"]
    return qc


def robust_z_score(
    *,
    values: pd.Series,
    epsilon: float = 1e-12,
) -> pd.Series:
    """Calculate robust z-scores using median and MAD.

    Parameters
    ----------
    values:
        Numeric values.
    epsilon:
        Small constant to avoid division by zero.

    Returns
    -------
    pandas.Series
        Robust z-scores.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median(skipna=True)
    mad = (numeric - median).abs().median(skipna=True)
    denominator = max(float(mad), epsilon)
    return 0.6745 * (numeric - median) / denominator


def flag_profile_outliers(
    *,
    data_frame: pd.DataFrame,
    metric_columns: Sequence[str],
    group_columns: Optional[Sequence[str]] = None,
    robust_z_threshold: float = 5.0,
) -> pd.DataFrame:
    """Flag outlying profiles using robust z-scores.

    Parameters
    ----------
    data_frame:
        Profile table containing metrics to inspect.
    metric_columns:
        Numeric metric columns to score.
    group_columns:
        Optional grouping columns. Robust z-scores are calculated within group.
    robust_z_threshold:
        Absolute robust z-score threshold.

    Returns
    -------
    pandas.DataFrame
        Table with one robust z-score and flag per metric.
    """
    output = data_frame.copy()
    valid_metric_columns = [column for column in metric_columns if column in output.columns]
    if not group_columns:
        for column in valid_metric_columns:
            z_column = f"{column}_robust_z"
            output[z_column] = robust_z_score(values=output[column])
            output[f"{column}_outlier"] = output[z_column].abs() > robust_z_threshold
        return output

    group_columns = [column for column in group_columns if column in output.columns]
    for column in valid_metric_columns:
        z_column = f"{column}_robust_z"
        output[z_column] = np.nan
        for _, index_values in output.groupby(group_columns, dropna=False).groups.items():
            output.loc[index_values, z_column] = robust_z_score(values=output.loc[index_values, column])
        output[f"{column}_outlier"] = output[z_column].abs() > robust_z_threshold
    return output


def summarise_qc(
    *,
    feature_qc: pd.DataFrame,
    sample_qc: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise feature and sample QC results.

    Parameters
    ----------
    feature_qc:
        Feature QC table.
    sample_qc:
        Sample QC table.

    Returns
    -------
    pandas.DataFrame
        Compact QC summary.
    """
    summary = [
        {"item": "n_features_total", "value": int(feature_qc.shape[0])},
        {
            "item": "n_features_passing_qc",
            "value": int(feature_qc.get("feature_qc_pass", pd.Series(dtype=bool)).sum()),
        },
        {"item": "n_samples_total", "value": int(sample_qc.shape[0])},
        {
            "item": "n_samples_passing_qc",
            "value": int(sample_qc.get("sample_qc_pass", pd.Series(dtype=bool)).sum()),
        },
    ]
    return pd.DataFrame.from_records(summary)
