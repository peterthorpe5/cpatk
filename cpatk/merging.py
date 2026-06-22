"""Build Cell Painting profile tables from folders of raw exports.

This module provides a conservative, auditable profile-building layer for
CellProfiler/Cell Painting projects.  Many projects arrive as a folder of
separate image-level, object-level and metadata tables rather than as one clean
analysis-ready profile table.  CPATK therefore separates this step from feature
preprocessing.

The safest default is:

* use the Image table as the row-level backbone where possible;
* aggregate each object table to ``ImageNumber`` before merging;
* do not blindly merge object tables by ``ObjectNumber`` across compartments;
* merge external plate/well metadata using canonical metadata aliases; and
* write every table role, merge key, aggregation and unmatched-row decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from cpatk.features import (
    infer_feature_columns,
    infer_metadata_columns,
    make_column_inventory,
    parse_column_list,
)
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.metadata import (
    drop_unnamed_index_columns,
    normalise_column_names,
    standardise_metadata_aliases,
)
from cpatk.reporting import make_html_report

SUPPORTED_TABLE_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tsv.gz",
    ".parquet",
    ".xlsx",
    ".xls",
)

IMAGE_HINTS = ("image", "images")
OBJECT_HINTS = ("cell", "cells", "cytoplasm", "nuclei", "nucleus", "object", "objects")
METADATA_HINTS = ("metadata", "meta", "plate_map", "platemap", "layout", "compound", "annotation")


@dataclass
class ProfileBuildResult:
    """Container for a profile-build result and audit tables."""

    profiles: pd.DataFrame
    tables: Dict[str, pd.DataFrame]


def _normalise_file_label(path: Path) -> str:
    """Return a safe table label derived from a file name."""
    name = path.name
    for suffix in [".csv.gz", ".tsv.gz", ".csv", ".tsv", ".parquet", ".xlsx", ".xls"]:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    label = re.sub(pattern=r"[^A-Za-z0-9]+", repl="_", string=name).strip("_")
    return label or "table"


def _has_supported_suffix(path: Path) -> bool:
    """Return whether a path is a supported table file."""
    suffixes = "".join(path.suffixes).lower()
    return any(suffixes.endswith(suffix) for suffix in SUPPORTED_TABLE_SUFFIXES)


def _resolve_input_path(*, input_dir: Path, path_value: str | Path) -> Path:
    """Resolve an explicit input path, accepting paths relative to input_dir."""
    path = Path(path_value)
    if path.exists():
        return path
    candidate = input_dir / path
    if candidate.exists():
        return candidate
    return path


def discover_table_files(*, input_dir: Path | str, recursive: bool = False) -> pd.DataFrame:
    """Discover supported table files in a folder.

    Parameters
    ----------
    input_dir:
        Folder containing Cell Painting outputs.
    recursive:
        Whether to search recursively.

    Returns
    -------
    pandas.DataFrame
        File inventory with one row per supported table.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    pattern = "**/*" if recursive else "*"
    records: List[Dict[str, object]] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file() or not _has_supported_suffix(path):
            continue
        records.append(
            {
                "path": str(path),
                "file_name": path.name,
                "table_label": _normalise_file_label(path),
                "size_bytes": int(path.stat().st_size),
                "suffixes": "".join(path.suffixes),
            }
        )
    return pd.DataFrame.from_records(records)


def read_table_columns(*, path: Path | str) -> List[str]:
    """Read only table column names where possible."""
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return list(pd.read_csv(filepath_or_buffer=path, nrows=0, encoding="utf-8-sig").columns)
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return list(pd.read_csv(filepath_or_buffer=path, sep="\t", nrows=0).columns)
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return list(pd.read_excel(io=path, nrows=0).columns)
    if suffixes.endswith(".parquet"):
        return list(pd.read_parquet(path=path).columns)
    raise ValueError(f"Unsupported input table format: {path}")


def infer_table_role(*, file_name: str, columns: Sequence[str]) -> Tuple[str, str]:
    """Infer whether a table is image-level, object-level, metadata or profile.

    The role is deliberately conservative.  Tables with both ``ImageNumber`` and
    ``ObjectNumber`` are object-level.  Tables with ``ImageNumber`` but no
    ``ObjectNumber`` are image-level unless they look like an already merged
    profile table.  Tables with plate/well/compound metadata and no ImageNumber
    are metadata tables.
    """
    lowered_columns = {str(column).strip().lower() for column in columns}
    lowered_name = file_name.lower()
    has_image_number = "imagenumber" in lowered_columns
    has_object_number = "objectnumber" in lowered_columns or "number_object_number" in lowered_columns
    has_plate = any("plate" in column for column in lowered_columns)
    has_well = any("well" in column for column in lowered_columns)
    has_compound = any("compound" in column or "cpd" in column or "moa" in column for column in lowered_columns)
    feature_like_count = sum(
        any(token in str(column).lower() for token in ("areashape", "intensity", "texture", "granularity", "radialdistribution", "correlation"))
        for column in columns
    )
    if has_image_number and has_object_number:
        return "object", "contains ImageNumber and ObjectNumber; aggregate to ImageNumber before merging"
    if has_image_number:
        if any(hint in lowered_name for hint in IMAGE_HINTS):
            return "image", "contains ImageNumber without ObjectNumber and file name suggests Image table"
        if feature_like_count > 0 and (has_plate or has_well or has_compound):
            return "profile", "contains ImageNumber plus metadata/features; appears profile-like"
        return "image", "contains ImageNumber without ObjectNumber"
    if has_plate and has_well and (has_compound or any(hint in lowered_name for hint in METADATA_HINTS)):
        return "metadata", "contains plate/well metadata without ImageNumber"
    if has_well and (has_compound or any(hint in lowered_name for hint in METADATA_HINTS)):
        return "metadata", "contains well/compound metadata without ImageNumber"
    if feature_like_count > 0:
        return "profile", "contains Cell Painting feature-like columns"
    return "unknown", "no robust role could be inferred"


def inspect_folder_tables(*, input_dir: Path | str, recursive: bool = False) -> pd.DataFrame:
    """Inspect a folder and infer table roles from headers."""
    inventory = discover_table_files(input_dir=input_dir, recursive=recursive)
    records: List[Dict[str, object]] = []
    for _, row in inventory.iterrows():
        path = Path(str(row["path"]))
        try:
            columns = read_table_columns(path=path)
            role, reason = infer_table_role(file_name=path.name, columns=columns)
            records.append(
                {
                    **row.to_dict(),
                    "n_columns": len(columns),
                    "inferred_role": role,
                    "role_reason": reason,
                    "has_ImageNumber": "ImageNumber" in columns,
                    "has_ObjectNumber": "ObjectNumber" in columns,
                    "readable": True,
                    "error": "",
                }
            )
        except Exception as exc:  # defensive inspection; continue other files
            records.append(
                {
                    **row.to_dict(),
                    "n_columns": 0,
                    "inferred_role": "unreadable",
                    "role_reason": "header read failed",
                    "has_ImageNumber": False,
                    "has_ObjectNumber": False,
                    "readable": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame.from_records(records)


def _coerce_and_clean_table(*, data_frame: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean column names, drop index artefacts and create metadata aliases."""
    cleaned, column_report = normalise_column_names(data_frame=data_frame)
    cleaned, dropped_report = drop_unnamed_index_columns(data_frame=cleaned, logger=logger)
    cleaned, alias_report = standardise_metadata_aliases(data_frame=cleaned, logger=logger)
    inventory = make_column_inventory(data_frame=cleaned)
    return cleaned, column_report, dropped_report, alias_report, inventory


def _aggregate_object_table(
    *,
    data_frame: pd.DataFrame,
    table_label: str,
    statistic: str = "median",
    include_qc_numeric_features: bool = False,
    group_keys: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate one object-level table to image-level profiles.

    Parameters
    ----------
    data_frame:
        Object-level CellProfiler table.
    table_label:
        Prefix used for aggregated feature names.
    statistic:
        Aggregation statistic, either ``median`` or ``mean``.
    include_qc_numeric_features:
        Whether to allow numeric QC/count features as biological features.
    group_keys:
        Keys defining one image/profile. Multi-plate projects should usually
        use ``Metadata_Plate`` + ``ImageNumber`` rather than ``ImageNumber``
        alone.
    logger:
        Optional logger.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Aggregated image-level table and an audit report.
    """
    group_keys = list(group_keys or ["ImageNumber"])
    missing_keys = [key for key in group_keys if key not in data_frame.columns]
    if missing_keys:
        raise ValueError(f"Object table {table_label} is missing merge keys: {missing_keys}")
    if "ImageNumber" not in group_keys:
        raise ValueError("Object aggregation keys must include ImageNumber.")
    metadata_columns = infer_metadata_columns(
        data_frame=data_frame,
        additional_metadata_columns=[*group_keys, "ObjectNumber"],
    )
    for identifier in [*group_keys, "ObjectNumber", "Number_Object_Number"]:
        if identifier in data_frame.columns and identifier not in metadata_columns:
            metadata_columns.append(identifier)
    feature_columns = infer_feature_columns(
        data_frame=data_frame,
        metadata_columns=metadata_columns,
        include_qc_numeric=include_qc_numeric_features,
    )
    if not feature_columns:
        raise ValueError(f"No object-level feature columns were found in {table_label}.")
    features = data_frame[[*group_keys, *feature_columns]].copy()
    for feature in feature_columns:
        features[feature] = pd.to_numeric(features[feature], errors="coerce")
    grouped = features.groupby(group_keys, dropna=False)
    if statistic == "median":
        aggregated = grouped.median(numeric_only=True).reset_index()
    elif statistic == "mean":
        aggregated = grouped.mean(numeric_only=True).reset_index()
    else:
        raise ValueError("Object aggregation statistic must be 'median' or 'mean'.")
    rename_map = {feature: f"{table_label}__{feature}" for feature in feature_columns}
    aggregated = aggregated.rename(columns=rename_map)
    object_counts = data_frame.groupby(group_keys, dropna=False).size().reset_index(name=f"{table_label}__n_objects")
    aggregated = aggregated.merge(object_counts, on=group_keys, how="left", validate="one_to_one")
    report = pd.DataFrame.from_records(
        [
            {
                "table_label": table_label,
                "n_input_rows": int(data_frame.shape[0]),
                "n_input_columns": int(data_frame.shape[1]),
                "n_feature_columns_aggregated": int(len(feature_columns)),
                "n_image_profiles": int(aggregated.shape[0]),
                "aggregation_statistic": statistic,
                "merge_key": ";".join(group_keys),
                "object_merge_caution": (
                    "Object-level tables were aggregated to ImageNumber before merging. "
                    "CPATK did not merge separate object compartments by ObjectNumber."
                ),
            }
        ]
    )
    if logger is not None:
        logger.info(
            "Aggregated object table %s: %s rows, %s features -> %s image profiles",
            table_label,
            data_frame.shape[0],
            len(feature_columns),
            aggregated.shape[0],
        )
    return aggregated, report



def _collapse_duplicate_rows(
    *,
    data_frame: pd.DataFrame,
    keys: Sequence[str],
    duplicate_policy: str,
    context: str,
) -> pd.DataFrame:
    """Collapse duplicate key rows only when explicitly allowed.

    Parameters
    ----------
    data_frame:
        Table to inspect.
    keys:
        Key columns that should be unique.
    duplicate_policy:
        One of ``error``, ``identical`` or ``first``.
    context:
        Human-readable context for error messages.

    Returns
    -------
    pandas.DataFrame
        Table with duplicate keys resolved according to policy.
    """
    duplicate_policy = duplicate_policy.lower()
    if duplicate_policy not in {"error", "identical", "first"}:
        raise ValueError("duplicate_policy must be one of: error, identical, first.")
    keys = list(keys)
    duplicated = data_frame.duplicated(subset=keys, keep=False)
    if not duplicated.any():
        return data_frame.copy()
    if duplicate_policy == "error":
        preview = data_frame.loc[duplicated, keys].head(10).to_dict(orient="records")
        raise ValueError(
            f"Duplicate {context} keys were found for {keys}. Preview: {preview}. "
            "Fix the input table or rerun with duplicate_policy='identical' or 'first'."
        )
    if duplicate_policy == "first":
        return data_frame.drop_duplicates(subset=keys, keep="first").copy()
    non_key_columns = [column for column in data_frame.columns if column not in keys]
    problem_keys = []
    for key_values, block in data_frame.loc[duplicated].groupby(keys, dropna=False):
        conflicting = [
            column for column in non_key_columns
            if block[column].dropna().astype(str).nunique() > 1
        ]
        if conflicting:
            problem_keys.append((key_values, conflicting[:5]))
    if problem_keys:
        raise ValueError(
            f"Duplicate {context} keys were not identical. Preview: {problem_keys[:5]}. "
            "Use duplicate_policy='first' only after reviewing the duplicate-key report."
        )
    return data_frame.drop_duplicates(subset=keys, keep="first").copy()


def _parse_merge_key_list(value: Optional[Sequence[str] | str]) -> Optional[List[str]]:
    """Parse an optional merge-key specification.

    Parameters
    ----------
    value:
        Sequence or comma/semicolon separated string of key names.

    Returns
    -------
    list[str] or None
        Parsed key names, or ``None`` when no keys were supplied.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_column_list(value=value)
    else:
        parsed = [str(item).strip() for item in value if str(item).strip()]
    return parsed or None


def choose_profile_merge_keys(
    *,
    left: pd.DataFrame,
    right: Optional[pd.DataFrame] = None,
    requested_keys: Optional[Sequence[str] | str] = None,
    require_image_number: bool = True,
) -> List[str]:
    """Choose safe composite keys for image/profile table merging.

    CellProfiler ``ImageNumber`` values are frequently unique only within one
    export or plate.  For multi-plate projects CPATK therefore prefers a
    composite key such as ``Metadata_Plate`` + ``ImageNumber`` when both sides
    contain these columns.  Source/robot wells are deliberately not used here;
    only assay/profile keys are considered.

    Parameters
    ----------
    left:
        Left table, usually the image/profile backbone.
    right:
        Optional right table to be merged onto ``left``.
    requested_keys:
        Explicit keys supplied by the user.
    require_image_number:
        Whether at least one selected key must be ``ImageNumber``.

    Returns
    -------
    list[str]
        Merge keys present in the required table(s).

    Raises
    ------
    ValueError
        If no safe keys can be selected or requested keys are absent.
    """
    requested = _parse_merge_key_list(requested_keys)
    tables = [left] + ([right] if right is not None else [])
    if requested:
        missing = [
            key
            for key in requested
            if any(key not in table.columns for table in tables)
        ]
        if missing:
            raise ValueError(
                f"Requested image/profile merge keys are missing from one or more tables: {missing}"
            )
        if require_image_number and "ImageNumber" not in requested:
            raise ValueError("Image/profile merge keys must include ImageNumber.")
        return list(requested)

    candidate_sets = [
        ["Metadata_Plate", "Metadata_Well", "ImageNumber"],
        ["Metadata_Plate", "ImageNumber"],
        ["ImageNumber"],
    ]
    for candidate in candidate_sets:
        if all(all(key in table.columns for key in candidate) for table in tables):
            if candidate == ["ImageNumber"]:
                # Object-level right tables are expected to contain repeated
                # ImageNumber values before aggregation. The unsafe case is a
                # profile/image backbone with repeated ImageNumber values and
                # no plate/export key to disambiguate them.
                if left.duplicated(subset=candidate, keep=False).any():
                    raise ValueError(
                        "ImageNumber is duplicated in the profile backbone and no shared Metadata_Plate column is available. "
                        "Build profiles per plate/export or ensure object and image tables contain an assay plate column."
                    )
            return list(candidate)
    raise ValueError(
        "Could not choose safe image/profile merge keys. Expected ImageNumber, ideally with Metadata_Plate for multi-plate exports."
    )



def _prepare_image_or_profile_table(
    *,
    data_frame: pd.DataFrame,
    table_label: str,
    include_qc_numeric_features: bool = False,
    duplicate_image_policy: str = "error",
    image_merge_keys: Optional[Sequence[str] | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare an image-level/profile table without object aggregation."""
    n_rows_input = int(data_frame.shape[0])
    merge_keys = choose_profile_merge_keys(
        left=data_frame,
        requested_keys=image_merge_keys,
        require_image_number="ImageNumber" in data_frame.columns,
    ) if "ImageNumber" in data_frame.columns else []
    n_duplicate_image_rows = 0
    if merge_keys:
        n_duplicate_image_rows = int(data_frame.duplicated(subset=merge_keys, keep=False).sum())
        data_frame = _collapse_duplicate_rows(
            data_frame=data_frame,
            keys=merge_keys,
            duplicate_policy=duplicate_image_policy,
            context="image/profile composite key",
        )
    report = pd.DataFrame.from_records(
        [
            {
                "table_label": table_label,
                "n_rows_input": n_rows_input,
                "n_rows": int(data_frame.shape[0]),
                "n_duplicate_ImageNumber_rows_removed": n_duplicate_image_rows,
                "n_columns": int(data_frame.shape[1]),
                "contains_ImageNumber": bool("ImageNumber" in data_frame.columns),
                "contains_Metadata_Plate": bool("Metadata_Plate" in data_frame.columns),
                "contains_Metadata_Well": bool("Metadata_Well" in data_frame.columns),
                "duplicate_image_policy": duplicate_image_policy,
                "image_merge_keys": ";".join(merge_keys),
                "note": "Image/profile table used as profile backbone. Duplicate composite-key rows fail by default unless an explicit permissive policy is chosen.",
            }
        ]
    )
    return data_frame.copy(), report


def _select_paths(
    *,
    input_dir: Path,
    inspection: pd.DataFrame,
    explicit_paths: Optional[Sequence[str]],
    role: str,
) -> List[Path]:
    """Select explicit paths or inferred paths for a table role."""
    if explicit_paths:
        return [_resolve_input_path(input_dir=input_dir, path_value=path) for path in explicit_paths]
    if inspection.empty:
        return []
    return [Path(path) for path in inspection.loc[inspection["inferred_role"] == role, "path"].tolist()]


def _pick_backbone_path(*, input_dir: Path, inspection: pd.DataFrame, explicit_image_table: Optional[str]) -> Optional[Path]:
    """Pick the safest profile backbone table."""
    if explicit_image_table:
        return _resolve_input_path(input_dir=input_dir, path_value=explicit_image_table)
    for role in ["image", "profile"]:
        matches = inspection.loc[inspection["inferred_role"] == role, "path"].tolist()
        if matches:
            return Path(matches[0])
    return None


def _merge_external_metadata(
    *,
    profiles: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_label: str,
    metadata_duplicate_policy: str = "error",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge external metadata onto profiles using canonical keys."""
    profiles, _, _, _, _ = _coerce_and_clean_table(data_frame=profiles)
    metadata, _, _, _, _ = _coerce_and_clean_table(data_frame=metadata)
    candidate_keys = ["Metadata_Plate", "Metadata_Well"]
    keys = [key for key in candidate_keys if key in profiles.columns and key in metadata.columns]
    if not keys and "Metadata_Well" in profiles.columns and "Metadata_Well" in metadata.columns:
        keys = ["Metadata_Well"]
    if not keys:
        report = pd.DataFrame.from_records(
            [
                {
                    "metadata_label": metadata_label,
                    "status": "skipped",
                    "merge_keys": "",
                    "reason": "No shared canonical plate/well metadata keys were available.",
                    "n_profiles_before": int(profiles.shape[0]),
                    "n_profiles_after": int(profiles.shape[0]),
                    "n_metadata_rows": int(metadata.shape[0]),
                    "n_unmatched_profiles": int(profiles.shape[0]),
                }
            ]
        )
        return profiles, report
    duplicated = metadata.duplicated(subset=keys, keep=False)
    metadata_unique = _collapse_duplicate_rows(
        data_frame=metadata,
        keys=keys,
        duplicate_policy=metadata_duplicate_policy,
        context=f"metadata table {metadata_label}",
    )
    before = profiles.shape[0]
    metadata_columns_to_add = [column for column in metadata_unique.columns if column not in keys and column not in profiles.columns]
    merged = profiles.merge(
        metadata_unique.loc[:, [*keys, *metadata_columns_to_add]],
        how="left",
        on=keys,
        validate="many_to_one",
    )
    unmatched = int(merged[metadata_columns_to_add].isna().all(axis=1).sum()) if metadata_columns_to_add else 0
    report = pd.DataFrame.from_records(
        [
            {
                "metadata_label": metadata_label,
                "status": "merged",
                "merge_keys": ";".join(keys),
                "reason": "Merged external metadata using canonical keys.",
                "n_profiles_before": int(before),
                "n_profiles_after": int(merged.shape[0]),
                "n_metadata_rows": int(metadata.shape[0]),
                "n_metadata_duplicate_key_rows": int(duplicated.sum()),
                "metadata_duplicate_policy": metadata_duplicate_policy,
                "n_metadata_columns_added": int(len(metadata_columns_to_add)),
                "n_unmatched_profiles": unmatched,
            }
        ]
    )
    return merged, report


def build_profiles_from_folder(
    *,
    input_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    recursive: bool = False,
    image_table: Optional[str] = None,
    object_tables: Optional[Sequence[str]] = None,
    metadata_table: Optional[str] = None,
    aggregate_statistic: str = "median",
    include_qc_numeric_features: bool = False,
    duplicate_image_policy: str = "error",
    metadata_duplicate_policy: str = "error",
    image_merge_keys: Optional[Sequence[str] | str] = None,
    logger: Optional[logging.Logger] = None,
) -> ProfileBuildResult:
    """Build an analysis-ready profile table from a folder of Cell Painting files.

    Parameters
    ----------
    input_dir:
        Directory containing raw Cell Painting exports.
    output_dir:
        Optional output directory for audit tables and report.
    recursive:
        Whether to discover input files recursively.
    image_table:
        Optional explicit image/profile backbone table path.
    object_tables:
        Optional explicit object-level table paths. If omitted, inferred object
        tables are aggregated and merged.
    metadata_table:
        Optional explicit external metadata/platemap table path.
    aggregate_statistic:
        Object aggregation statistic, ``median`` or ``mean``.
    include_qc_numeric_features:
        Whether count/QC columns can be aggregated as features.
    duplicate_image_policy:
        Policy for duplicate ImageNumber rows in the backbone table: error, identical or first.
    metadata_duplicate_policy:
        Policy for duplicate external metadata keys: error, identical or first.
    image_merge_keys:
        Optional explicit image/object merge keys. Multi-plate projects often
        require ``Metadata_Plate,ImageNumber``.
    logger:
        Optional logger.

    Returns
    -------
    ProfileBuildResult
        Merged profile table and audit tables.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    inspection = inspect_folder_tables(input_dir=input_dir, recursive=recursive)
    if inspection.empty:
        raise ValueError(f"No supported Cell Painting tables found in {input_dir}.")
    if logger is not None:
        logger.info("Discovered %s supported input tables", inspection.shape[0])

    explicit_object_paths = [str(path) for path in parse_column_list(value=",".join(object_tables or [])) or []]
    object_paths = _select_paths(input_dir=input_dir, inspection=inspection, explicit_paths=explicit_object_paths, role="object")
    backbone_path = _pick_backbone_path(input_dir=input_dir, inspection=inspection, explicit_image_table=image_table)
    if backbone_path is None:
        raise ValueError(
            "No Image/profile backbone table was found. Provide --image_table or include an Image/Profile table with ImageNumber."
        )

    if logger is not None:
        logger.info("Using profile backbone table: %s", backbone_path)
    backbone_raw = read_table(path=backbone_path, logger=logger)
    backbone_clean, backbone_column_report, backbone_dropped, backbone_alias, backbone_inventory = _coerce_and_clean_table(
        data_frame=backbone_raw,
        logger=logger,
    )
    profiles, backbone_report = _prepare_image_or_profile_table(
        data_frame=backbone_clean,
        table_label=_normalise_file_label(backbone_path),
        include_qc_numeric_features=include_qc_numeric_features,
        duplicate_image_policy=duplicate_image_policy,
        image_merge_keys=image_merge_keys,
    )

    aggregation_reports: List[pd.DataFrame] = []
    object_column_reports: List[pd.DataFrame] = []
    for object_path in object_paths:
        if object_path.resolve() == backbone_path.resolve():
            continue
        table_label = _normalise_file_label(object_path)
        object_raw = read_table(path=object_path, logger=logger)
        object_clean, column_report, dropped_report, alias_report, inventory = _coerce_and_clean_table(
            data_frame=object_raw,
            logger=logger,
        )
        column_report.insert(0, "table_label", table_label)
        object_column_reports.append(column_report)
        object_merge_keys = choose_profile_merge_keys(
            left=profiles,
            right=object_clean,
            requested_keys=image_merge_keys,
            require_image_number=True,
        )
        aggregated, report = _aggregate_object_table(
            data_frame=object_clean,
            table_label=table_label,
            statistic=aggregate_statistic,
            include_qc_numeric_features=include_qc_numeric_features,
            group_keys=object_merge_keys,
            logger=logger,
        )
        aggregation_reports.append(report)
        before_rows = profiles.shape[0]
        profiles = profiles.merge(aggregated, on=object_merge_keys, how="left", validate="one_to_one")
        if logger is not None:
            logger.info(
                "Merged object aggregate %s on %s: %s -> %s rows",
                table_label,
                ";".join(object_merge_keys),
                before_rows,
                profiles.shape[0],
            )

    metadata_reports: List[pd.DataFrame] = []
    explicit_metadata_paths = parse_column_list(value=metadata_table) if metadata_table else None
    metadata_paths: List[Path]
    if explicit_metadata_paths:
        metadata_paths = [_resolve_input_path(input_dir=input_dir, path_value=path) for path in explicit_metadata_paths]
    else:
        metadata_paths = _select_paths(input_dir=input_dir, inspection=inspection, explicit_paths=None, role="metadata")
    for metadata_path in metadata_paths:
        metadata_raw = read_table(path=metadata_path, logger=logger)
        profiles, metadata_report = _merge_external_metadata(
            profiles=profiles,
            metadata=metadata_raw,
            metadata_label=_normalise_file_label(metadata_path),
            metadata_duplicate_policy=metadata_duplicate_policy,
        )
        metadata_reports.append(metadata_report)

    profiles, final_column_report, final_dropped, final_alias, final_inventory = _coerce_and_clean_table(
        data_frame=profiles,
        logger=logger,
    )
    feature_columns = infer_feature_columns(
        data_frame=profiles,
        metadata_columns=infer_metadata_columns(data_frame=profiles),
        include_qc_numeric=include_qc_numeric_features,
    )
    summary = pd.DataFrame.from_records(
        [
            {"item": "n_input_tables_discovered", "value": int(inspection.shape[0])},
            {"item": "profile_backbone_table", "value": str(backbone_path)},
            {"item": "n_object_tables_aggregated", "value": int(len(aggregation_reports))},
            {"item": "object_aggregation_statistic", "value": aggregate_statistic},
            {"item": "duplicate_image_policy", "value": duplicate_image_policy},
            {"item": "metadata_duplicate_policy", "value": metadata_duplicate_policy},
            {"item": "image_merge_keys", "value": ";".join(_parse_merge_key_list(image_merge_keys) or [])},
            {"item": "external_metadata_tables", "value": ";".join(str(path) for path in metadata_paths)},
            {"item": "n_profiles", "value": int(profiles.shape[0])},
            {"item": "n_columns", "value": int(profiles.shape[1])},
            {"item": "n_inferred_feature_columns", "value": int(len(feature_columns))},
            {
                "item": "object_merge_policy",
                "value": "object tables aggregated to ImageNumber before merging; no cross-compartment ObjectNumber merge",
            },
        ]
    )
    tables: Dict[str, pd.DataFrame] = {
        "profile_build_summary": summary,
        "input_table_inventory": inspection,
        "backbone_table_report": backbone_report,
        "backbone_column_name_report": backbone_column_report,
        "backbone_metadata_alias_report": backbone_alias,
        "backbone_column_inventory": backbone_inventory,
        "final_profile_column_name_report": final_column_report,
        "final_profile_metadata_alias_report": final_alias,
        "final_profile_column_inventory": final_inventory,
        "retained_profile_features": pd.DataFrame({"feature": feature_columns}),
        "object_aggregation_report": pd.concat(aggregation_reports, ignore_index=True) if aggregation_reports else pd.DataFrame(),
        "object_column_name_report": pd.concat(object_column_reports, ignore_index=True) if object_column_reports else pd.DataFrame(),
        "metadata_merge_report": pd.concat(metadata_reports, ignore_index=True) if metadata_reports else pd.DataFrame(),
    }
    if output_path is not None:
        _write_profile_build_outputs(profiles=profiles, tables=tables, output_dir=output_path, logger=logger)
    return ProfileBuildResult(profiles=profiles, tables=tables)


def _write_profile_build_outputs(
    *,
    profiles: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Write profile-build tables, workbook and report."""
    try:
        write_table(data_frame=profiles, path=output_dir / "merged_profiles.parquet", logger=logger)
    except ImportError as exc:
        if logger is not None:
            logger.warning("Parquet writing unavailable; writing TSV.GZ fallback: %s", exc)
        write_table(data_frame=profiles, path=output_dir / "merged_profiles.tsv.gz", logger=logger)
    for name, table in tables.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    excel_tables = {**tables, "merged_profiles_preview": profiles.head(5000)}
    write_excel_workbook(tables=excel_tables, path=output_dir / "profile_build_summary.xlsx", logger=logger)
    warnings = [
        "Object-level tables are aggregated to ImageNumber before merging. This avoids unsafe cross-compartment ObjectNumber joins.",
        "Review the metadata_merge_report for unmatched profiles or duplicate plate/well metadata keys.",
    ]
    make_html_report(
        title="CPATK profile-build report",
        output_path=output_dir / "profile_build_report.html",
        summary_tables={**tables, "Merged profile preview": profiles.head(50)},
        narrative=(
            "CPATK built an analysis-ready profile table from a folder of Cell Painting exports. "
            "The workflow used an image/profile table as the row-level backbone, aggregated object-level tables to ImageNumber, "
            "merged optional external plate/well metadata, and wrote audit tables for every decision."
        ),
        methods_text=(
            "Raw Cell Painting exports often contain separate Image, Cell, Cytoplasm, Nuclei and metadata files. "
            "CPATK does not assume that ObjectNumber is comparable across compartments. Object-level tables are therefore summarised "
            "within each ImageNumber using the requested statistic, then merged to the image/profile backbone."
        ),
        warnings=warnings,
    )
