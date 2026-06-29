"""Build Cell Painting profile tables from folders of raw exports.

This module provides a conservative, auditable profile-building layer for
CellProfiler/Cell Painting projects.  Many projects arrive as a folder of
separate image-level, object-level and metadata tables rather than as one clean
analysis-ready profile table.  CPATK therefore separates this step from feature
preprocessing.

The safest default is:

* use the Image table as the row-level backbone where possible;
* propagate image-level assay keys onto object tables when safe;
* aggregate each object table to image/profile-level before merging;
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

import numpy as np
import pandas as pd

from cpatk.features import (
    infer_feature_columns,
    infer_metadata_columns,
    make_column_inventory,
    parse_column_list,
)
from cpatk.io import (
    is_ignored_sidecar_path,
    read_table,
    write_excel_workbook,
    write_table,
)
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
        if (
            not path.is_file()
            or is_ignored_sidecar_path(path=path)
            or not _has_supported_suffix(path)
        ):
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



def _can_propagate_object_key(column: str) -> bool:
    """Return whether an image-level key is safe to stamp onto object rows."""
    if column in {"Metadata_Plate", "Metadata_Well", "Plate_Metadata", "Well_Metadata"}:
        return True
    return column.startswith("Metadata_") and not column.startswith("Metadata_Source_")


def _candidate_object_key_propagation_columns(
    *,
    profiles: pd.DataFrame,
    object_table: pd.DataFrame,
    requested_keys: Optional[Sequence[str] | str] = None,
) -> Tuple[List[str], bool]:
    """Choose image-level keys that may be propagated to an object table.

    Returns the missing key columns and whether they came from an explicit user
    request. Explicit keys are treated more strictly because the user has asked
    for a particular merge policy, usually for multi-plate safety.
    """
    requested = _parse_merge_key_list(requested_keys)
    explicit = requested is not None
    if requested is None:
        requested = ["Metadata_Plate", "Metadata_Well", "ImageNumber"]
    missing = [
        key
        for key in requested
        if key != "ImageNumber"
        and key in profiles.columns
        and key not in object_table.columns
        and _can_propagate_object_key(key)
    ]
    return missing, explicit


def _propagate_image_keys_to_object_table(
    *,
    object_table: pd.DataFrame,
    profiles: pd.DataFrame,
    table_label: str,
    requested_keys: Optional[Sequence[str] | str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate image-level assay keys to an object table when safe.

    CellProfiler object tables commonly contain ``ImageNumber`` and
    ``ObjectNumber`` but do not repeat assay-level metadata such as
    ``Metadata_Plate`` and ``Metadata_Well``. When the image/profile backbone
    has a unique mapping from ``ImageNumber`` to these metadata keys, CPATK can
    safely stamp the keys onto object rows before aggregation. This allows
    multi-plate composite keys such as ``Metadata_Plate,ImageNumber`` without
    requiring the raw object tables to already contain ``Metadata_Plate``.

    The function deliberately refuses ambiguous mappings. If ``ImageNumber`` is
    associated with more than one plate/well value in the backbone, there is no
    safe way to infer the missing key for an object row that only has
    ``ImageNumber``.
    """
    missing_keys, explicit = _candidate_object_key_propagation_columns(
        profiles=profiles,
        object_table=object_table,
        requested_keys=requested_keys,
    )
    base_record = {
        "table_label": table_label,
        "requested_keys": ";".join(_parse_merge_key_list(requested_keys) or []),
        "propagated_keys": ";".join(missing_keys),
        "n_object_rows": int(object_table.shape[0]),
        "n_profile_rows": int(profiles.shape[0]),
    }
    if not missing_keys:
        report = pd.DataFrame.from_records([
            {
                **base_record,
                "status": "not_needed",
                "reason": "No missing image-level merge keys needed propagation.",
                "n_mapping_rows": 0,
                "n_conflicting_image_numbers": 0,
                "n_object_rows_missing_propagated_keys": 0,
            }
        ])
        return object_table, report
    if "ImageNumber" not in object_table.columns or "ImageNumber" not in profiles.columns:
        message = (
            f"Object table {table_label} is missing image-level keys {missing_keys}, "
            "but ImageNumber is not available on both the object table and profile backbone."
        )
        if explicit:
            raise ValueError(message)
        report = pd.DataFrame.from_records([
            {
                **base_record,
                "status": "skipped_missing_ImageNumber",
                "reason": message,
                "n_mapping_rows": 0,
                "n_conflicting_image_numbers": 0,
                "n_object_rows_missing_propagated_keys": 0,
            }
        ])
        return object_table, report
    mapping = profiles.loc[:, ["ImageNumber", *missing_keys]].drop_duplicates()
    conflicting_mask = mapping.duplicated(subset=["ImageNumber"], keep=False)
    n_conflicting = int(mapping.loc[conflicting_mask, "ImageNumber"].nunique())
    if n_conflicting:
        message = (
            f"Cannot safely propagate {missing_keys} to object table {table_label}: "
            f"{n_conflicting} ImageNumber value(s) map to multiple image-level key values. "
            "Object tables must contain the plate/export key, or the input should be split per plate/export."
        )
        if explicit:
            raise ValueError(message)
        report = pd.DataFrame.from_records([
            {
                **base_record,
                "status": "skipped_ambiguous_ImageNumber_mapping",
                "reason": message,
                "n_mapping_rows": int(mapping.shape[0]),
                "n_conflicting_image_numbers": n_conflicting,
                "n_object_rows_missing_propagated_keys": 0,
            }
        ])
        if logger is not None:
            logger.warning(message)
        return object_table, report
    propagated = object_table.merge(mapping, on="ImageNumber", how="left", validate="many_to_one")
    missing_after = int(propagated.loc[:, missing_keys].isna().any(axis=1).sum())
    report = pd.DataFrame.from_records([
        {
            **base_record,
            "status": "propagated",
            "reason": "Image-level keys were propagated from the profile backbone using a unique ImageNumber mapping.",
            "n_mapping_rows": int(mapping.shape[0]),
            "n_conflicting_image_numbers": 0,
            "n_object_rows_missing_propagated_keys": missing_after,
        }
    ])
    if logger is not None:
        logger.info(
            "Propagated image-level keys to object table %s using ImageNumber: %s",
            table_label,
            ";".join(missing_keys),
        )
        if missing_after:
            logger.warning(
                "Object table %s has %s row(s) with missing propagated image-level keys.",
                table_label,
                missing_after,
            )
    return propagated, report


def trim_object_table_by_robust_distance(
    *,
    data_frame: pd.DataFrame,
    table_label: str,
    group_keys: Sequence[str],
    feature_columns: Sequence[str],
    keep_central_fraction: float = 0.90,
    metric: str = "q95",
    trim_quantile: float = 0.95,
    max_feature_missing_fraction: float = 0.70,
    min_feature_fraction_per_object: float = 0.25,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trim extreme object rows using a multifeature robust-distance score.

    The method is deliberately optional.  Within each group, usually one image
    or one well/profile key, CPATK computes feature-wise medians and median
    absolute deviations, converts object rows to robust z-scores, summarises
    each object by a multifeature distance, and keeps the central requested
    fraction.  This is not per-feature top/bottom trimming; it removes the most
    extreme objects according to the combined feature profile.
    """
    if not (0.0 < float(keep_central_fraction) <= 1.0):
        raise ValueError("keep_central_fraction must be in (0, 1].")
    if not (0.0 < float(trim_quantile) <= 1.0):
        raise ValueError("trim_quantile must be in (0, 1].")
    if not (0.0 <= float(max_feature_missing_fraction) <= 1.0):
        raise ValueError("max_feature_missing_fraction must be in [0, 1].")
    if not (0.0 < float(min_feature_fraction_per_object) <= 1.0):
        raise ValueError("min_feature_fraction_per_object must be in (0, 1].")
    if metric not in {"q", "q95", "l2", "max"}:
        raise ValueError("metric must be one of: q, q95, l2, max.")

    group_keys = list(group_keys)
    feature_columns = [column for column in feature_columns if column in data_frame.columns]
    missing_keys = [key for key in group_keys if key not in data_frame.columns]
    if missing_keys:
        raise ValueError(f"Trimming group keys are missing from {table_label}: {missing_keys}")
    if not feature_columns:
        empty_summary = pd.DataFrame.from_records([
            {
                "table_label": table_label,
                "status": "skipped_no_features",
                "n_objects_before": int(data_frame.shape[0]),
                "n_objects_after": int(data_frame.shape[0]),
                "n_objects_removed": 0,
                "fraction_removed": 0.0,
                "keep_central_fraction": float(keep_central_fraction),
                "trim_metric": metric,
                "trim_quantile": float(trim_quantile),
                "warning": "No numeric object features were available for trimming.",
            }
        ])
        return data_frame.copy(), empty_summary, pd.DataFrame()

    keep_masks: List[pd.Series] = []
    group_rows: List[Dict[str, object]] = []
    if group_keys:
        group_iterator = data_frame.groupby(group_keys, dropna=False, sort=False)
    else:
        group_iterator = [("__global__", data_frame)]
    for key_values, group in group_iterator:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_record = {group_key: key_value for group_key, key_value in zip(group_keys, key_values)}
        if not key_record:
            key_record = {"trim_group": "__global__"}
        group_index = group.index
        values = group.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
        missing_fraction = values.isna().mean(axis=0)
        usable_features = missing_fraction[missing_fraction <= float(max_feature_missing_fraction)].index.tolist()
        n_too_missing = int(len(feature_columns) - len(usable_features))
        if not usable_features:
            keep_mask = pd.Series(True, index=group_index)
            group_rows.append({
                "table_label": table_label,
                **key_record,
                "status": "kept_no_usable_features",
                "n_objects_before": int(group.shape[0]),
                "n_objects_after": int(group.shape[0]),
                "n_objects_removed": 0,
                "fraction_removed": 0.0,
                "n_features_total": int(len(feature_columns)),
                "n_features_usable": 0,
                "n_features_too_missing": n_too_missing,
                "n_features_zero_mad": 0,
                "cutoff_distance": np.nan,
            })
            keep_masks.append(keep_mask)
            continue
        matrix = values.loc[:, usable_features].to_numpy(dtype="float32")
        medians = np.nanmedian(matrix, axis=0)
        abs_dev = np.abs(matrix - medians)
        mads = np.nanmedian(abs_dev, axis=0)
        nonzero = mads > 0
        n_zero_mad = int((~nonzero).sum())
        if not np.any(nonzero):
            keep_mask = pd.Series(True, index=group_index)
            group_rows.append({
                "table_label": table_label,
                **key_record,
                "status": "kept_zero_mad",
                "n_objects_before": int(group.shape[0]),
                "n_objects_after": int(group.shape[0]),
                "n_objects_removed": 0,
                "fraction_removed": 0.0,
                "n_features_total": int(len(feature_columns)),
                "n_features_usable": int(len(usable_features)),
                "n_features_too_missing": n_too_missing,
                "n_features_zero_mad": n_zero_mad,
                "cutoff_distance": np.nan,
            })
            keep_masks.append(keep_mask)
            continue
        matrix = matrix[:, nonzero]
        medians = medians[nonzero]
        mads = mads[nonzero]
        with np.errstate(invalid="ignore", divide="ignore"):
            robust_z = (matrix - medians) / mads
        valid_counts = np.sum(np.isfinite(robust_z), axis=1)
        min_required = max(1, int(np.ceil(float(min_feature_fraction_per_object) * robust_z.shape[1])))
        enough_features = valid_counts >= min_required
        with np.errstate(invalid="ignore", divide="ignore"):
            if metric in {"q", "q95"}:
                distances = np.nanquantile(np.abs(robust_z), float(trim_quantile), axis=1)
            elif metric == "l2":
                distances = np.sqrt(np.nanmean(robust_z * robust_z, axis=1))
            else:
                distances = np.nanmax(np.abs(robust_z), axis=1)
        finite = np.isfinite(distances) & enough_features
        if not np.any(finite):
            keep_mask_array = np.ones(group.shape[0], dtype=bool)
            cutoff = np.nan
            status = "kept_no_finite_distances"
        else:
            cutoff = float(np.nanquantile(distances[finite], float(keep_central_fraction)))
            keep_mask_array = (distances <= cutoff) & finite
            status = "trimmed"
        keep_mask = pd.Series(keep_mask_array, index=group_index)
        n_after = int(keep_mask.sum())
        n_before = int(group.shape[0])
        group_rows.append({
            "table_label": table_label,
            **key_record,
            "status": status,
            "n_objects_before": n_before,
            "n_objects_after": n_after,
            "n_objects_removed": int(n_before - n_after),
            "fraction_removed": float((n_before - n_after) / max(n_before, 1)),
            "n_features_total": int(len(feature_columns)),
            "n_features_usable": int(len(usable_features)),
            "n_features_too_missing": n_too_missing,
            "n_features_zero_mad": n_zero_mad,
            "cutoff_distance": cutoff,
        })
        keep_masks.append(keep_mask)

    combined_keep = pd.concat(keep_masks).reindex(data_frame.index).fillna(False).astype(bool)
    trimmed = data_frame.loc[combined_keep].copy()
    by_group = pd.DataFrame.from_records(group_rows)
    n_before = int(data_frame.shape[0])
    n_after = int(trimmed.shape[0])
    summary = pd.DataFrame.from_records([
        {
            "table_label": table_label,
            "status": "enabled",
            "n_objects_before": n_before,
            "n_objects_after": n_after,
            "n_objects_removed": int(n_before - n_after),
            "fraction_removed": float((n_before - n_after) / max(n_before, 1)),
            "n_groups": int(by_group.shape[0]),
            "n_groups_removed_gt_25pct": int((by_group.get("fraction_removed", pd.Series(dtype=float)) > 0.25).sum()),
            "n_groups_removed_gt_50pct": int((by_group.get("fraction_removed", pd.Series(dtype=float)) > 0.50).sum()),
            "keep_central_fraction": float(keep_central_fraction),
            "trim_metric": metric,
            "trim_quantile": float(trim_quantile),
            "max_feature_missing_fraction": float(max_feature_missing_fraction),
            "min_feature_fraction_per_object": float(min_feature_fraction_per_object),
            "warning": (
                "Object trimming was enabled. This can reduce segmentation/debris artefacts but may remove true extreme phenotypes. "
                "Compare trimmed and untrimmed runs for sensitive biological conclusions."
            ),
        }
    ])
    if logger is not None:
        logger.info(
            "Object trimming %s: kept %s / %s rows (removed %.1f%%).",
            table_label,
            n_after,
            n_before,
            100.0 * (n_before - n_after) / max(n_before, 1),
        )
    return trimmed, summary, by_group

def _aggregate_object_table(
    *,
    data_frame: pd.DataFrame,
    table_label: str,
    statistic: str = "median",
    include_qc_numeric_features: bool = False,
    group_keys: Optional[Sequence[str]] = None,
    trim_objects: bool = False,
    trim_keep_central_fraction: float = 0.90,
    trim_metric: str = "q95",
    trim_quantile: float = 0.95,
    trim_max_feature_missing_fraction: float = 0.70,
    trim_min_feature_fraction_per_object: float = 0.25,
    trim_group_keys: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    trim_summary = pd.DataFrame.from_records([
        {
            "table_label": table_label,
            "status": "disabled",
            "n_objects_before": int(data_frame.shape[0]),
            "n_objects_after": int(data_frame.shape[0]),
            "n_objects_removed": 0,
            "fraction_removed": 0.0,
            "keep_central_fraction": float(trim_keep_central_fraction),
            "trim_metric": trim_metric,
            "trim_quantile": float(trim_quantile),
            "warning": "Object trimming was not enabled.",
        }
    ])
    trim_by_group = pd.DataFrame()
    if trim_objects:
        data_frame, trim_summary, trim_by_group = trim_object_table_by_robust_distance(
            data_frame=data_frame,
            table_label=table_label,
            group_keys=list(trim_group_keys or group_keys),
            feature_columns=feature_columns,
            keep_central_fraction=trim_keep_central_fraction,
            metric=trim_metric,
            trim_quantile=trim_quantile,
            max_feature_missing_fraction=trim_max_feature_missing_fraction,
            min_feature_fraction_per_object=trim_min_feature_fraction_per_object,
            logger=logger,
        )
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
                "trimming_enabled": bool(trim_objects),
                "n_object_rows_after_trimming": int(data_frame.shape[0]),
                "n_object_rows_removed_by_trimming": int(trim_summary.loc[0, "n_objects_removed"]),
                "fraction_object_rows_removed_by_trimming": float(trim_summary.loc[0, "fraction_removed"]),
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
    return aggregated, report, trim_summary, trim_by_group



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
    trim_objects: bool = False,
    trim_keep_central_fraction: float = 0.90,
    trim_scope: str = "image",
    trim_metric: str = "q95",
    trim_quantile: float = 0.95,
    trim_max_feature_missing_fraction: float = 0.70,
    trim_min_feature_fraction_per_object: float = 0.25,
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
    trim_objects:
        Whether to trim extreme object rows before object-table aggregation.
        Disabled by default because trimming may remove true extreme phenotypes.
    trim_keep_central_fraction:
        Fraction of object rows to retain within each trimming group.
    trim_scope:
        ``image`` uses the image/profile merge keys. ``plate`` uses Metadata_Plate
        when present. ``global`` trims all rows in a table as one group.
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
    object_trimming_summaries: List[pd.DataFrame] = []
    object_trimming_by_group_reports: List[pd.DataFrame] = []
    object_column_reports: List[pd.DataFrame] = []
    object_key_propagation_reports: List[pd.DataFrame] = []
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
        object_clean, key_propagation_report = _propagate_image_keys_to_object_table(
            object_table=object_clean,
            profiles=profiles,
            table_label=table_label,
            requested_keys=image_merge_keys,
            logger=logger,
        )
        object_key_propagation_reports.append(key_propagation_report)
        object_merge_keys = choose_profile_merge_keys(
            left=profiles,
            right=object_clean,
            requested_keys=image_merge_keys,
            require_image_number=True,
        )
        if trim_scope == "global":
            object_trim_group_keys = []
        elif trim_scope == "plate" and "Metadata_Plate" in object_clean.columns:
            object_trim_group_keys = ["Metadata_Plate"]
        elif trim_scope in {"image", "per_image", "per_profile", "per_well"}:
            object_trim_group_keys = object_merge_keys
        else:
            object_trim_group_keys = object_merge_keys
        aggregated, report, trim_summary, trim_by_group = _aggregate_object_table(
            data_frame=object_clean,
            table_label=table_label,
            statistic=aggregate_statistic,
            include_qc_numeric_features=include_qc_numeric_features,
            group_keys=object_merge_keys,
            trim_objects=trim_objects,
            trim_group_keys=object_trim_group_keys,
            trim_keep_central_fraction=trim_keep_central_fraction,
            trim_metric=trim_metric,
            trim_quantile=trim_quantile,
            trim_max_feature_missing_fraction=trim_max_feature_missing_fraction,
            trim_min_feature_fraction_per_object=trim_min_feature_fraction_per_object,
            logger=logger,
        )
        aggregation_reports.append(report)
        object_trimming_summaries.append(trim_summary)
        if not trim_by_group.empty:
            object_trimming_by_group_reports.append(trim_by_group)
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
            {"item": "object_trimming_enabled", "value": bool(trim_objects)},
            {"item": "object_trimming_scope", "value": trim_scope},
            {"item": "object_trimming_keep_central_fraction", "value": float(trim_keep_central_fraction)},
            {"item": "object_trimming_metric", "value": trim_metric},
            {"item": "external_metadata_tables", "value": ";".join(str(path) for path in metadata_paths)},
            {"item": "n_profiles", "value": int(profiles.shape[0])},
            {"item": "n_columns", "value": int(profiles.shape[1])},
            {"item": "n_inferred_feature_columns", "value": int(len(feature_columns))},
            {
                "item": "n_object_tables_with_image_key_propagation",
                "value": int(sum(
                    1
                    for report in object_key_propagation_reports
                    if not report.empty and str(report.loc[0, "status"]) == "propagated"
                )),
            },
            {
                "item": "object_merge_policy",
                "value": (
                    "object tables are first stamped with image-level assay keys when a unique ImageNumber mapping is available, "
                    "then aggregated to image/profile-level; no cross-compartment ObjectNumber merge"
                ),
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
        "object_trimming_summary": pd.concat(object_trimming_summaries, ignore_index=True) if object_trimming_summaries else pd.DataFrame(),
        "object_trimming_by_group": pd.concat(object_trimming_by_group_reports, ignore_index=True) if object_trimming_by_group_reports else pd.DataFrame(),
        "object_key_propagation_report": pd.concat(object_key_propagation_reports, ignore_index=True) if object_key_propagation_reports else pd.DataFrame(),
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
        "Object-level tables are stamped with image-level assay keys before aggregation when a unique ImageNumber mapping is available.",
        "If ImageNumber is ambiguous across plates and object tables lack plate/export keys, CPATK refuses to infer those keys.",
        "Review object_key_propagation_report and metadata_merge_report before interpreting profile-level outputs.",
        "If object trimming was enabled, review object_trimming_summary and object_trimming_by_group; trimming may remove true extreme phenotypes as well as artefacts.",
    ]
    make_html_report(
        title="CPATK profile-build report",
        output_path=output_dir / "profile_build_report.html",
        summary_tables={**tables, "Merged profile preview": profiles.head(50)},
        narrative=(
            "CPATK built an analysis-ready profile table from a folder of Cell Painting exports. "
            "The workflow used an image/profile table as the row-level backbone, propagated image-level assay keys "
            "to object tables when this could be done through a unique ImageNumber mapping, aggregated object-level tables "
            "to image/profile level, merged optional external plate/well metadata, and wrote audit tables for every decision."
        ),
        methods_text=(
            "Raw Cell Painting exports often contain separate Image, Cell, Cytoplasm, Nuclei and metadata files. "
            "CPATK does not assume that ObjectNumber is comparable across compartments. Object-level tables are therefore summarised "
            "within each image/profile key using the requested statistic, then merged to the image/profile backbone. When object tables "
            "lack Metadata_Plate or Metadata_Well but the Image table provides a unique ImageNumber mapping, those keys are propagated first."
        ),
        warnings=warnings,
    )
