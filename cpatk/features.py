"""Feature and metadata column handling for generic Cell Painting tables."""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_METADATA_PATTERNS = (
    "metadata",
    "plate",
    "well",
    "site",
    "image",
    "imagenumber",
    "objectnumber",
    "compound",
    "treatment",
    "dose",
    "concentration",
    "donor",
    "batch",
    "replicate",
    "moa",
    "mechanism",
    "class",
    "label",
    "group",
    "barcode",
    "file",
    "path",
)

DEFAULT_FEATURE_PATTERNS = (
    "intensity",
    "texture",
    "radialdistribution",
    "areashape",
    "neighbors",
    "granularity",
    "correlation",
    "location",
    "number_object",
)


def infer_metadata_columns(
    *,
    data_frame: pd.DataFrame,
    additional_metadata_columns: Optional[Sequence[str]] = None,
    metadata_patterns: Sequence[str] = DEFAULT_METADATA_PATTERNS,
) -> List[str]:
    """Infer likely metadata columns from names and data types.

    Parameters
    ----------
    data_frame:
        Input data frame.
    additional_metadata_columns:
        User-specified columns that must be treated as metadata when present.
    metadata_patterns:
        Case-insensitive name patterns used to detect metadata columns.

    Returns
    -------
    list[str]
        Metadata column names.
    """
    metadata_columns = []
    pattern = re.compile("|".join(re.escape(item) for item in metadata_patterns), re.IGNORECASE)
    requested = set(additional_metadata_columns or [])
    for column in data_frame.columns:
        is_requested = column in requested
        is_named_metadata = bool(pattern.search(string=str(column)))
        is_non_numeric = not pd.api.types.is_numeric_dtype(data_frame[column])
        if is_requested or is_named_metadata or is_non_numeric:
            metadata_columns.append(column)
    return metadata_columns


def infer_feature_columns(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    feature_patterns: Sequence[str] = DEFAULT_FEATURE_PATTERNS,
    numeric_only: bool = True,
) -> List[str]:
    """Infer likely Cell Painting feature columns.

    Parameters
    ----------
    data_frame:
        Input data frame.
    metadata_columns:
        Columns to exclude from the feature set.
    feature_patterns:
        Case-insensitive name patterns used to detect Cell Painting features.
    numeric_only:
        Whether to require numeric dtype.

    Returns
    -------
    list[str]
        Inferred feature column names.
    """
    metadata_set = set(metadata_columns or [])
    pattern = re.compile("|".join(re.escape(item) for item in feature_patterns), re.IGNORECASE)
    feature_columns = []
    for column in data_frame.columns:
        if column in metadata_set:
            continue
        if numeric_only and not pd.api.types.is_numeric_dtype(data_frame[column]):
            continue
        if pattern.search(string=str(column)) or not metadata_set:
            feature_columns.append(column)
    if not feature_columns:
        feature_columns = [
            column for column in data_frame.columns
            if column not in metadata_set and pd.api.types.is_numeric_dtype(data_frame[column])
        ]
    return feature_columns


def split_metadata_and_features(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    additional_metadata_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Split a table into metadata and numeric feature matrices.

    Parameters
    ----------
    data_frame:
        Input data frame.
    metadata_columns:
        Optional explicit metadata columns.
    feature_columns:
        Optional explicit feature columns.
    additional_metadata_columns:
        Extra metadata columns for automatic inference.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, list[str], list[str]]
        Metadata table, feature table, metadata column names, feature column
        names.
    """
    if metadata_columns is None:
        metadata_columns = infer_metadata_columns(
            data_frame=data_frame,
            additional_metadata_columns=additional_metadata_columns,
        )
    else:
        metadata_columns = [column for column in metadata_columns if column in data_frame.columns]

    if feature_columns is None:
        feature_columns = infer_feature_columns(
            data_frame=data_frame,
            metadata_columns=metadata_columns,
        )
    else:
        feature_columns = [column for column in feature_columns if column in data_frame.columns]

    metadata = data_frame.loc[:, list(metadata_columns)].copy()
    features = data_frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    return metadata, features, list(metadata_columns), list(feature_columns)


def make_column_inventory(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """Create a column inventory table.

    Parameters
    ----------
    data_frame:
        Input data frame.

    Returns
    -------
    pandas.DataFrame
        Column-level inventory.
    """
    records = []
    for column in data_frame.columns:
        series = data_frame[column]
        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
            }
        )
    return pd.DataFrame.from_records(records)


def validate_columns_present(
    *,
    data_frame: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """Raise an informative error if required columns are absent.

    Parameters
    ----------
    data_frame:
        Input data frame.
    required_columns:
        Required columns.
    """
    missing = [column for column in required_columns if column not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def summarise_feature_matrix(
    *,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise a numeric feature matrix.

    Parameters
    ----------
    features:
        Numeric feature matrix.

    Returns
    -------
    pandas.DataFrame
        Summary table with feature-level statistics.
    """
    records = []
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        records.append(
            {
                "feature": column,
                "n_values": int(values.notna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "mean": float(values.mean(skipna=True)),
                "median": float(values.median(skipna=True)),
                "sd": float(values.std(skipna=True)),
                "variance": float(values.var(skipna=True)),
                "mad": float((values - values.median(skipna=True)).abs().median(skipna=True)),
                "min": float(values.min(skipna=True)),
                "max": float(values.max(skipna=True)),
            }
        )
    return pd.DataFrame.from_records(records)


def parse_column_list(*, value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-free or comma-separated command-line column list.

    Parameters
    ----------
    value:
        String containing column names separated by commas, semicolons or tabs.

    Returns
    -------
    list[str] or None
        Parsed column names, or None when no value is supplied.
    """
    if value is None or not str(value).strip():
        return None
    parts = re.split(pattern=r"[,;\t]", string=str(value))
    return [part.strip() for part in parts if part.strip()]
