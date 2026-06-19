"""Feature and metadata column handling for generic Cell Painting tables.

This module is intentionally conservative.  Cell Painting exports often contain
many numeric columns that are not biological morphology features, for example
image identifiers, object identifiers, execution times, file checksums, image
heights, image widths and object counts.  CPATK therefore separates columns into
metadata, feature and QC/excluded roles before preprocessing.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_METADATA_PATTERNS = (
    "metadata",
    "plate",
    "well",
    "site",
    "field",
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
    "library",
    "barcode",
    "sourcew",
    "cpd_id",
    "cpd_type",
)

DEFAULT_FEATURE_PATTERNS = (
    "areashape",
    "children",
    "correlation",
    "granularity",
    "intensity",
    "location",
    "neighbors",
    "radialdistribution",
    "texture",
    "zernike",
)

DEFAULT_EXCLUDED_FEATURE_PREFIXES = (
    "executiontime",
    "filename",
    "pathname",
    "url",
    "md5digest",
    "height",
    "width",
    "channel",
    "group_index",
    "group_length",
    "group_number",
    "imageid",
    "imagename",
    "imagenumber",
    "imageseries",
    "objectnumber",
    "number_object_number",
    "parent_",
    "mean_parent_",
)

DEFAULT_QC_NUMERIC_PATTERNS = (
    "count_",
    "mean_count_",
    "children_",
    "mean_children_",
)


def _normalise_text(value: object) -> str:
    """Return a lowercase alphanumeric-ish representation for matching."""
    return str(value).strip().lower().replace(" ", "_")


def parse_column_list(*, value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-, semicolon-, tab- or newline-separated column list."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = re.split(r"[,;\t\n]+", value)
    else:
        parts = list(value)
    parsed = [str(item).strip() for item in parts if str(item).strip()]
    return parsed or None


def read_column_list_file(*, path: Optional[str]) -> Optional[List[str]]:
    """Read one column name per line from a text file when supplied."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return lines or None


def infer_metadata_columns(
    *,
    data_frame: pd.DataFrame,
    additional_metadata_columns: Optional[Sequence[str]] = None,
    metadata_patterns: Sequence[str] = DEFAULT_METADATA_PATTERNS,
) -> List[str]:
    """Infer likely metadata columns from names and data types.

    Non-numeric columns are treated as metadata.  Numeric columns are treated as
    metadata only when their names match known metadata patterns or when the user
    explicitly requests them.  This prevents concentration/dose columns from
    silently entering the image-feature matrix.
    """
    metadata_columns: List[str] = []
    requested = set(additional_metadata_columns or [])
    pattern = re.compile("|".join(re.escape(item) for item in metadata_patterns), re.IGNORECASE)
    for column in data_frame.columns:
        column_text = _normalise_text(column)
        is_requested = column in requested
        is_non_numeric = not pd.api.types.is_numeric_dtype(data_frame[column])
        is_named_metadata = bool(pattern.search(string=column_text))
        if is_requested or is_non_numeric or is_named_metadata:
            metadata_columns.append(column)
    return metadata_columns


def looks_like_cellpainting_feature(
    *,
    column: str,
    feature_patterns: Sequence[str] = DEFAULT_FEATURE_PATTERNS,
) -> bool:
    """Return whether a column name looks like a CellProfiler feature."""
    column_text = _normalise_text(column)
    return any(pattern in column_text for pattern in feature_patterns)


def looks_like_excluded_numeric_column(
    *,
    column: str,
    excluded_prefixes: Sequence[str] = DEFAULT_EXCLUDED_FEATURE_PREFIXES,
    qc_patterns: Sequence[str] = DEFAULT_QC_NUMERIC_PATTERNS,
) -> tuple[bool, str]:
    """Classify numeric columns that should not be default analysis features."""
    column_text = _normalise_text(column)
    for prefix in excluded_prefixes:
        if column_text.startswith(prefix):
            return True, f"excluded_prefix:{prefix}"
    for pattern in qc_patterns:
        if pattern in column_text or column_text.startswith(pattern):
            return True, f"qc_numeric_pattern:{pattern}"
    return False, ""


def infer_feature_columns(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    feature_patterns: Sequence[str] = DEFAULT_FEATURE_PATTERNS,
    numeric_only: bool = True,
    include_qc_numeric: bool = False,
) -> List[str]:
    """Infer likely Cell Painting feature columns.

    By default this selects numeric CellProfiler-style morphology/intensity
    columns and excludes obvious identifiers, file/provenance fields, execution
    times and object-count QC columns.  Users can always override this with an
    explicit feature-column list.
    """
    metadata_set = set(metadata_columns or [])
    feature_columns: List[str] = []
    for column in data_frame.columns:
        if column in metadata_set:
            continue
        if numeric_only and not pd.api.types.is_numeric_dtype(data_frame[column]):
            continue
        excluded, _ = looks_like_excluded_numeric_column(column=str(column))
        if excluded and not include_qc_numeric:
            continue
        if looks_like_cellpainting_feature(column=str(column), feature_patterns=feature_patterns):
            feature_columns.append(column)

    if not feature_columns:
        for column in data_frame.columns:
            if column in metadata_set:
                continue
            if numeric_only and not pd.api.types.is_numeric_dtype(data_frame[column]):
                continue
            excluded, _ = looks_like_excluded_numeric_column(column=str(column))
            if excluded and not include_qc_numeric:
                continue
            feature_columns.append(column)
    return feature_columns


def assign_column_roles(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    additional_metadata_columns: Optional[Sequence[str]] = None,
    include_qc_numeric: bool = False,
) -> pd.DataFrame:
    """Create an auditable column-role table.

    Returns a table explaining whether each column was treated as metadata,
    feature, excluded numeric QC/provenance, or ignored non-feature content.
    """
    inferred_metadata = list(metadata_columns) if metadata_columns is not None else infer_metadata_columns(
        data_frame=data_frame,
        additional_metadata_columns=additional_metadata_columns,
    )
    inferred_features = list(feature_columns) if feature_columns is not None else infer_feature_columns(
        data_frame=data_frame,
        metadata_columns=inferred_metadata,
        include_qc_numeric=include_qc_numeric,
    )
    metadata_set = set(column for column in inferred_metadata if column in data_frame.columns)
    feature_set = set(column for column in inferred_features if column in data_frame.columns)
    records = []
    for column in data_frame.columns:
        series = data_frame[column]
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        excluded, reason = looks_like_excluded_numeric_column(column=str(column))
        if column in metadata_set:
            role = "metadata"
            role_reason = "metadata_column"
        elif column in feature_set:
            role = "feature"
            role_reason = "selected_feature"
        elif is_numeric and excluded:
            role = "excluded_numeric_qc_or_provenance"
            role_reason = reason
        elif is_numeric:
            role = "numeric_not_selected"
            role_reason = "numeric but not selected as feature"
        else:
            role = "non_numeric_not_selected"
            role_reason = "non-numeric and not selected as metadata"
        records.append(
            {
                "column": column,
                "role": role,
                "reason": role_reason,
                "dtype": str(series.dtype),
                "is_numeric": is_numeric,
                "n_missing": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame.from_records(records)


def split_metadata_and_features(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    additional_metadata_columns: Optional[Sequence[str]] = None,
    include_qc_numeric: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Split a table into metadata and numeric feature matrices."""
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
            include_qc_numeric=include_qc_numeric,
        )
    else:
        feature_columns = [column for column in feature_columns if column in data_frame.columns]

    metadata = data_frame.loc[:, list(metadata_columns)].copy()
    features = data_frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    return metadata, features, list(metadata_columns), list(feature_columns)


def make_column_inventory(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level inventory table."""
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


def validate_columns_present(*, data_frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Raise an informative error if required columns are absent."""
    missing = [column for column in required_columns if column not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def summarise_feature_matrix(*, features: pd.DataFrame) -> pd.DataFrame:
    """Summarise a numeric feature matrix."""
    records = []
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        median = values.median(skipna=True)
        records.append(
            {
                "feature": column,
                "n_values": int(values.notna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "mean": float(values.mean(skipna=True)) if values.notna().any() else np.nan,
                "median": float(median) if values.notna().any() else np.nan,
                "sd": float(values.std(skipna=True)) if values.notna().sum() > 1 else np.nan,
                "variance": float(values.var(skipna=True)) if values.notna().sum() > 1 else np.nan,
                "mad": float((values - median).abs().median(skipna=True)) if values.notna().any() else np.nan,
                "min": float(values.min(skipna=True)) if values.notna().any() else np.nan,
                "max": float(values.max(skipna=True)) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)
