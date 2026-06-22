"""Metadata validation and annotation merging for CPATK.

This module implements the recommended first step for messy Cell Painting
projects: inspect, standardise and validate metadata before profile building or
preprocessing. The goal is not to hide problematic metadata, but to produce a
clean formatted table plus audit reports that make assumptions explicit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.metadata import (
    canonicalise_well_value,
    drop_unnamed_index_columns,
    normalise_column_names,
    standardise_metadata_aliases,
)
from cpatk.reporting import make_html_report

PRIMARY_KEY_COLUMNS = ["Metadata_Plate", "Metadata_Well"]
SOURCE_KEY_COLUMNS = ["Metadata_Source_Plate", "Metadata_Source_Well"]
LEGACY_COMPATIBILITY_COLUMNS: Mapping[str, str] = {
    "Metadata_Plate": "Plate_Metadata",
    "Metadata_Well": "Well_Metadata",
    "Metadata_Compound": "cpd_id",
    "Metadata_MOA": "cpd_type",
    "Metadata_Batch": "Library",
}


def _normalise_text_series(series: pd.Series) -> pd.Series:
    """Strip whitespace while preserving missing values.

    Parameters
    ----------
    series:
        Input series to clean.

    Returns
    -------
    pandas.Series
        Cleaned series with empty strings converted to missing values.
    """
    output = series.astype("string").str.strip()
    output = output.mask(output.eq(""), pd.NA)
    return output


def _normalise_key_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise common metadata key columns while preserving raw values.

    The raw well/plate strings can matter when robotic source-plate maps use a
    different naming convention from CellProfiler assay outputs. CPATK therefore
    keeps ``__raw`` audit columns before canonicalising assay and source keys.
    """
    output = data_frame.copy()
    for column in ["Metadata_Well", "Metadata_Source_Well", "Well_Metadata"]:
        if column in output.columns:
            raw_column = f"{column}__raw"
            if raw_column not in output.columns:
                output[raw_column] = output[column]
            output[column] = output[column].map(lambda value: canonicalise_well_value(value=value))
    for column in ["Metadata_Plate", "Metadata_Source_Plate", "Plate_Metadata"]:
        if column in output.columns:
            raw_column = f"{column}__raw"
            if raw_column not in output.columns:
                output[raw_column] = output[column]
            output[column] = _normalise_text_series(output[column])
    return output


def _case_insensitive_column_lookup(data_frame: pd.DataFrame) -> dict[str, str]:
    """Return a case-insensitive lookup for existing columns."""
    return {str(column).strip().lower(): str(column) for column in data_frame.columns}


def _resolve_explicit_column(*, data_frame: pd.DataFrame, column_name: Optional[str]) -> Optional[str]:
    """Resolve an explicitly requested column name after header cleaning.

    Parameters
    ----------
    data_frame:
        Table whose columns have already been cleaned.
    column_name:
        User-supplied column name or ``None``.

    Returns
    -------
    str or None
        Actual column name in ``data_frame``.

    Raises
    ------
    ValueError
        If the user supplied a column name that is not present.
    """
    if column_name is None:
        return None
    requested = str(column_name).strip()
    if not requested:
        return None
    lookup = _case_insensitive_column_lookup(data_frame=data_frame)
    resolved = lookup.get(requested.lower())
    if resolved is None:
        raise ValueError(
            f"Explicit metadata column '{column_name}' was not found. "
            f"Available columns are: {', '.join(map(str, data_frame.columns))}"
        )
    return resolved


def apply_explicit_key_columns(
    *,
    data_frame: pd.DataFrame,
    plate_column: Optional[str] = None,
    well_column: Optional[str] = None,
    source_plate_column: Optional[str] = None,
    source_well_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply user-specified assay and source plate/well columns.

    Explicit assay columns define the plate/well keys that should be used when
    merging CellProfiler profiles. Explicit source columns are kept separately
    for library/robot annotation merges and are never promoted to assay keys.
    """
    output = data_frame.copy()
    mapping = {
        "Metadata_Plate": plate_column,
        "Metadata_Well": well_column,
        "Metadata_Source_Plate": source_plate_column,
        "Metadata_Source_Well": source_well_column,
    }
    records = []
    for target, requested in mapping.items():
        source = _resolve_explicit_column(data_frame=output, column_name=requested)
        if source is None:
            records.append(
                {
                    "target_column": target,
                    "explicit_source_column": "",
                    "action": "not_requested",
                    "n_non_missing": 0,
                }
            )
            continue
        if target in output.columns:
            raw_target = f"{target}__before_explicit_override"
            if raw_target not in output.columns:
                output[raw_target] = output[target]
        explicit_raw_target = f"{target}__raw"
        if explicit_raw_target not in output.columns:
            output[explicit_raw_target] = output[source]
        output[target] = output[source]
        records.append(
            {
                "target_column": target,
                "explicit_source_column": source,
                "action": "set_from_explicit_column",
                "n_non_missing": int(output[target].notna().sum()),
            }
        )
    return output, pd.DataFrame.from_records(records)


def _find_source_like_primary_aliases(alias_report: pd.DataFrame) -> pd.DataFrame:
    """Return primary assay key aliases that came from source-like columns."""
    if alias_report.empty:
        return pd.DataFrame()
    if not {"canonical_column", "matched_source_column", "action"}.issubset(alias_report.columns):
        return pd.DataFrame()
    primary_keys = {"Metadata_Plate", "Metadata_Well"}
    source_like = alias_report["matched_source_column"].astype(str).str.contains(
        r"(?i)source|^spb$|^sw2?$"
    )
    created = alias_report["action"].astype(str).isin(["created", "created_from_alias"])
    primary = alias_report["canonical_column"].isin(primary_keys)
    return alias_report.loc[primary & source_like & created].copy()


def add_legacy_compatibility_columns(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """Add legacy CPATK/project aliases without overwriting existing columns.

    Parameters
    ----------
    data_frame:
        Metadata table with canonical ``Metadata_*`` columns.

    Returns
    -------
    pandas.DataFrame
        Metadata table with optional legacy aliases such as ``cpd_id`` and
        ``Well_Metadata`` added when they are not already present.
    """
    output = data_frame.copy()
    for source, target in LEGACY_COMPATIBILITY_COLUMNS.items():
        if source in output.columns and target not in output.columns:
            output[target] = output[source]
    return output


def prepare_metadata_table(
    *,
    data_frame: pd.DataFrame,
    table_label: str = "metadata",
    plate_column: Optional[str] = None,
    well_column: Optional[str] = None,
    source_plate_column: Optional[str] = None,
    source_well_column: Optional[str] = None,
    require_assay_keys: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Clean and standardise one metadata or annotation table.

    Source-plate/source-well columns are deliberately kept separate from the
    primary assay ``Metadata_Plate``/``Metadata_Well`` keys. If the raw file is
    ambiguous, provide explicit ``plate_column`` and ``well_column`` arguments.

    Parameters
    ----------
    data_frame:
        Raw metadata table.
    table_label:
        Human-readable label used in reports.
    plate_column, well_column:
        Optional explicit assay plate/well columns.
    source_plate_column, source_well_column:
        Optional explicit source-library plate/well columns.
    require_assay_keys:
        If True, raise an error unless assay plate and well keys are available.
    logger:
        Optional logger.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, pandas.DataFrame]]
        Cleaned metadata table and audit tables.
    """
    cleaned, column_report = normalise_column_names(data_frame=data_frame)
    cleaned, dropped_report = drop_unnamed_index_columns(data_frame=cleaned, logger=logger)
    cleaned, explicit_report = apply_explicit_key_columns(
        data_frame=cleaned,
        plate_column=plate_column,
        well_column=well_column,
        source_plate_column=source_plate_column,
        source_well_column=source_well_column,
    )
    cleaned, alias_report = standardise_metadata_aliases(data_frame=cleaned, logger=logger)
    unsafe_alias_report = _find_source_like_primary_aliases(alias_report=alias_report)
    if not unsafe_alias_report.empty:
        raise ValueError(
            "Source-like columns were about to be used as primary assay plate/well keys. "
            "Please provide explicit plate_column/well_column for the CellProfiler assay keys "
            "and source_plate_column/source_well_column for library/robot annotation keys."
        )
    cleaned = _normalise_key_columns(cleaned)
    cleaned = add_legacy_compatibility_columns(data_frame=cleaned)
    if require_assay_keys:
        missing = [column for column in PRIMARY_KEY_COLUMNS if column not in cleaned.columns]
        if missing:
            raise ValueError(
                "Formatted metadata is missing required assay key columns: "
                f"{missing}. Provide explicit plate_column and well_column values."
            )
    summary = pd.DataFrame.from_records(
        [
            {
                "table_label": table_label,
                "n_rows": int(cleaned.shape[0]),
                "n_columns": int(cleaned.shape[1]),
                "has_Metadata_Plate": bool("Metadata_Plate" in cleaned.columns),
                "has_Metadata_Well": bool("Metadata_Well" in cleaned.columns),
                "has_Metadata_Source_Plate": bool("Metadata_Source_Plate" in cleaned.columns),
                "has_Metadata_Source_Well": bool("Metadata_Source_Well" in cleaned.columns),
                "has_Metadata_Compound": bool("Metadata_Compound" in cleaned.columns),
                "has_cpd_id": bool("cpd_id" in cleaned.columns),
            }
        ]
    )
    if logger is not None:
        logger.info(
            "Prepared metadata table %s: %s rows, %s columns",
            table_label,
            cleaned.shape[0],
            cleaned.shape[1],
        )
    reports = {
        "column_name_report": column_report,
        "dropped_index_column_report": dropped_report,
        "metadata_alias_report": alias_report,
        "explicit_key_column_report": explicit_report,
        "metadata_prepare_summary": summary,
    }
    return cleaned, reports


def choose_merge_keys(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    requested_keys: Optional[Sequence[str]] = None,
    allow_well_only: bool = False,
) -> list[str]:
    """Choose safe shared keys for a metadata/annotation merge.

    Parameters
    ----------
    left, right:
        Tables to merge.
    requested_keys:
        Optional explicit key list. All requested keys must be present in both
        tables.
    allow_well_only:
        Whether to allow a one-column ``Metadata_Well`` merge. This is disabled
        by default because it is unsafe for multi-plate data.

    Returns
    -------
    list[str]
        Merge keys.

    Raises
    ------
    ValueError
        If no safe merge keys can be selected.
    """
    if requested_keys:
        keys = [str(key).strip() for key in requested_keys if str(key).strip()]
        missing = [key for key in keys if key not in left.columns or key not in right.columns]
        if missing:
            raise ValueError(f"Requested merge keys are not present in both tables: {missing}")
        return keys

    for candidate in (SOURCE_KEY_COLUMNS, PRIMARY_KEY_COLUMNS):
        if all(key in left.columns and key in right.columns for key in candidate):
            return list(candidate)
    if allow_well_only and "Metadata_Well" in left.columns and "Metadata_Well" in right.columns:
        return ["Metadata_Well"]
    raise ValueError(
        "No safe shared merge keys found. Use explicit --merge_keys, or provide "
        "plate+well or source-plate+source-well columns."
    )


def duplicate_key_report(*, data_frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Report duplicated metadata keys.

    Parameters
    ----------
    data_frame:
        Metadata table to inspect.
    keys:
        Key columns.

    Returns
    -------
    pandas.DataFrame
        Duplicate-key summary and value previews.
    """
    keys = list(keys)
    missing = [key for key in keys if key not in data_frame.columns]
    if missing:
        return pd.DataFrame.from_records(
            [{"status": "skipped_missing_keys", "keys": ";".join(keys), "missing_keys": ";".join(missing)}]
        )
    duplicated = data_frame.duplicated(subset=keys, keep=False)
    if not duplicated.any():
        return pd.DataFrame.from_records(
            [{"status": "ok_unique_keys", "keys": ";".join(keys), "n_duplicate_rows": 0, "n_duplicate_groups": 0}]
        )
    grouped = data_frame.loc[duplicated].groupby(keys, dropna=False)
    rows = []
    for key_values, block in grouped:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "status": "duplicate_key",
                "keys": ";".join(keys),
                "n_rows_for_key": int(block.shape[0]),
                "columns_with_conflicting_values": ";".join(_conflicting_columns(block=block, keys=keys)),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _conflicting_columns(*, block: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    """Return non-key columns with more than one non-missing value."""
    conflicts = []
    for column in block.columns:
        if column in keys:
            continue
        if block[column].dropna().astype(str).nunique() > 1:
            conflicts.append(str(column))
    return conflicts


def collapse_duplicate_keys(
    *,
    data_frame: pd.DataFrame,
    keys: Sequence[str],
    duplicate_policy: str = "error",
) -> pd.DataFrame:
    """Collapse duplicated keys only when the chosen policy allows it.

    Parameters
    ----------
    data_frame:
        Table to collapse.
    keys:
        Key columns.
    duplicate_policy:
        One of ``error``, ``first`` or ``identical``.

    Returns
    -------
    pandas.DataFrame
        Table with one row per key.
    """
    duplicate_policy = duplicate_policy.lower()
    if duplicate_policy not in {"error", "first", "identical"}:
        raise ValueError("duplicate_policy must be one of: error, first, identical.")
    duplicated = data_frame.duplicated(subset=list(keys), keep=False)
    if not duplicated.any():
        return data_frame.copy()
    if duplicate_policy == "error":
        raise ValueError(
            "Duplicate metadata/annotation keys were found. Review duplicate reports, "
            "or rerun with --duplicate_policy identical or --duplicate_policy first."
        )
    if duplicate_policy == "first":
        return data_frame.drop_duplicates(subset=list(keys), keep="first").copy()
    problem_groups = []
    for key_values, block in data_frame.loc[duplicated].groupby(list(keys), dropna=False):
        conflicts = _conflicting_columns(block=block, keys=keys)
        if conflicts:
            problem_groups.append((key_values, conflicts))
    if problem_groups:
        preview = "; ".join(f"{key}: {cols[:5]}" for key, cols in problem_groups[:5])
        raise ValueError(f"Duplicate keys have conflicting annotation values: {preview}")
    return data_frame.drop_duplicates(subset=list(keys), keep="first").copy()


def merge_annotation_tables(
    *,
    metadata: pd.DataFrame,
    annotations: Sequence[pd.DataFrame],
    annotation_labels: Optional[Sequence[str]] = None,
    requested_keys: Optional[Sequence[str]] = None,
    duplicate_policy: str = "error",
    allow_well_only: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge one or more annotation tables into a metadata table.

    Parameters
    ----------
    metadata:
        Cleaned metadata table.
    annotations:
        Cleaned annotation tables.
    annotation_labels:
        Optional labels for reports.
    requested_keys:
        Optional explicit merge keys.
    duplicate_policy:
        How to handle duplicate keys in annotations.
    allow_well_only:
        Whether to allow well-only annotation merges.
    plate_column, well_column:
        Explicit assay plate/well columns for the main metadata table.
    source_plate_column, source_well_column:
        Explicit source-library plate/well columns for the main metadata table.
    annotation_plate_column, annotation_well_column:
        Explicit assay plate/well columns for annotation tables.
    annotation_source_plate_column, annotation_source_well_column:
        Explicit source-library plate/well columns for annotation tables.
    require_assay_keys:
        Whether the main metadata table must contain assay plate/well keys.
    logger:
        Optional logger.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Merged metadata, merge report and duplicate-key report.
    """
    output = metadata.copy()
    merge_records = []
    duplicate_reports = []
    labels = list(annotation_labels or [f"annotation_{i + 1}" for i in range(len(annotations))])
    for label, annotation in zip(labels, annotations):
        keys = choose_merge_keys(
            left=output,
            right=annotation,
            requested_keys=requested_keys,
            allow_well_only=allow_well_only,
        )
        dup_report = duplicate_key_report(data_frame=annotation, keys=keys)
        dup_report.insert(0, "annotation_label", label)
        duplicate_reports.append(dup_report)
        annotation_unique = collapse_duplicate_keys(
            data_frame=annotation,
            keys=keys,
            duplicate_policy=duplicate_policy,
        )
        columns_to_add = [column for column in annotation_unique.columns if column not in keys]
        rename_map = {}
        for column in columns_to_add:
            if column in output.columns:
                rename_map[column] = f"{label}__{column}"
        annotation_unique = annotation_unique.rename(columns=rename_map)
        columns_to_add = [rename_map.get(column, column) for column in columns_to_add]
        before_cols = output.shape[1]
        output = output.merge(
            annotation_unique.loc[:, [*keys, *columns_to_add]],
            on=keys,
            how="left",
            validate="many_to_one",
        )
        unmatched = int(output[columns_to_add].isna().all(axis=1).sum()) if columns_to_add else 0
        merge_records.append(
            {
                "annotation_label": label,
                "status": "merged",
                "merge_keys": ";".join(keys),
                "n_annotation_rows_input": int(annotation.shape[0]),
                "n_annotation_rows_after_duplicate_policy": int(annotation_unique.shape[0]),
                "n_columns_added": int(output.shape[1] - before_cols),
                "n_metadata_rows": int(output.shape[0]),
                "n_rows_without_annotation_match": unmatched,
                "duplicate_policy": duplicate_policy,
            }
        )
        if logger is not None:
            logger.info("Merged annotation %s using keys %s", label, ";".join(keys))
    merge_report = pd.DataFrame.from_records(merge_records)
    duplicate_report = pd.concat(duplicate_reports, ignore_index=True) if duplicate_reports else pd.DataFrame()
    return output, merge_report, duplicate_report


def validate_metadata_keys(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """Create key-level validation summaries for a formatted metadata table."""
    records = []
    for keys, key_name in [(PRIMARY_KEY_COLUMNS, "assay_plate_well"), (SOURCE_KEY_COLUMNS, "source_plate_well")]:
        present = [key for key in keys if key in data_frame.columns]
        if len(present) != len(keys):
            records.append(
                {
                    "key_name": key_name,
                    "status": "missing_key_columns",
                    "key_columns": ";".join(keys),
                    "n_unique_keys": 0,
                    "n_duplicate_rows": 0,
                    "n_missing_key_rows": int(data_frame.shape[0]),
                }
            )
            continue
        missing_key_rows = int(data_frame.loc[:, keys].isna().any(axis=1).sum())
        duplicate_rows = int(data_frame.duplicated(subset=keys, keep=False).sum())
        records.append(
            {
                "key_name": key_name,
                "status": "ok" if duplicate_rows == 0 and missing_key_rows == 0 else "review",
                "key_columns": ";".join(keys),
                "n_unique_keys": int(data_frame.loc[:, keys].drop_duplicates().shape[0]),
                "n_duplicate_rows": duplicate_rows,
                "n_missing_key_rows": missing_key_rows,
            }
        )
    return pd.DataFrame.from_records(records)


def run_metadata_validation_workflow(
    *,
    metadata_table: Path,
    output_dir: Path,
    annotation_tables: Optional[Sequence[Path]] = None,
    merge_keys: Optional[Sequence[str]] = None,
    duplicate_policy: str = "error",
    allow_well_only: bool = False,
    plate_column: Optional[str] = None,
    well_column: Optional[str] = None,
    source_plate_column: Optional[str] = None,
    source_well_column: Optional[str] = None,
    annotation_plate_column: Optional[str] = None,
    annotation_well_column: Optional[str] = None,
    annotation_source_plate_column: Optional[str] = None,
    annotation_source_well_column: Optional[str] = None,
    require_assay_keys: bool = True,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Run metadata validation, formatting and optional annotation merging.

    Parameters
    ----------
    metadata_table:
        Raw metadata table.
    output_dir:
        Directory for formatted metadata and reports.
    annotation_tables:
        Optional annotation tables to merge.
    merge_keys:
        Optional explicit merge keys used for every annotation merge.
    duplicate_policy:
        Duplicate-key policy for annotation tables.
    allow_well_only:
        Whether to allow well-only annotation merges.
    plate_column, well_column:
        Explicit assay plate/well columns for the main metadata table.
    source_plate_column, source_well_column:
        Explicit source-library plate/well columns for the main metadata table.
    annotation_plate_column, annotation_well_column:
        Explicit assay plate/well columns for annotation tables.
    annotation_source_plate_column, annotation_source_well_column:
        Explicit source-library plate/well columns for annotation tables.
    require_assay_keys:
        Whether the main metadata table must contain assay plate/well keys.
    logger:
        Optional logger.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Output and report tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metadata = read_table(path=metadata_table, logger=logger)
    formatted, reports = prepare_metadata_table(
        data_frame=raw_metadata,
        table_label=Path(metadata_table).name,
        plate_column=plate_column,
        well_column=well_column,
        source_plate_column=source_plate_column,
        source_well_column=source_well_column,
        require_assay_keys=require_assay_keys,
        logger=logger,
    )
    annotation_prepare_reports = []
    annotation_labels = []
    annotations = []
    for annotation_path in annotation_tables or []:
        raw_annotation = read_table(path=annotation_path, logger=logger)
        label = Path(annotation_path).stem
        annotation, annotation_reports = prepare_metadata_table(
            data_frame=raw_annotation,
            table_label=label,
            plate_column=annotation_plate_column,
            well_column=annotation_well_column,
            source_plate_column=annotation_source_plate_column,
            source_well_column=annotation_source_well_column,
            require_assay_keys=False,
            logger=logger,
        )
        annotations.append(annotation)
        annotation_labels.append(label)
        for report_name, table in annotation_reports.items():
            table = table.copy()
            table.insert(0, "annotation_label", label)
            table.insert(1, "report_name", report_name)
            annotation_prepare_reports.append(table)
    if annotations:
        formatted, merge_report, annotation_duplicate_report = merge_annotation_tables(
            metadata=formatted,
            annotations=annotations,
            annotation_labels=annotation_labels,
            requested_keys=merge_keys,
            duplicate_policy=duplicate_policy,
            allow_well_only=allow_well_only,
            logger=logger,
        )
    else:
        merge_report = pd.DataFrame()
        annotation_duplicate_report = pd.DataFrame()
    formatted = _normalise_key_columns(formatted)
    formatted = add_legacy_compatibility_columns(data_frame=formatted)
    key_validation = validate_metadata_keys(data_frame=formatted)
    validation_summary = pd.DataFrame.from_records(
        [
            {
                "item": "n_rows_formatted_metadata",
                "value": int(formatted.shape[0]),
            },
            {
                "item": "n_columns_formatted_metadata",
                "value": int(formatted.shape[1]),
            },
            {
                "item": "n_annotation_tables_merged",
                "value": int(len(annotations)),
            },
            {
                "item": "duplicate_policy",
                "value": duplicate_policy,
            },
            {
                "item": "well_canonicalisation",
                "value": "A1-style assay/source wells are converted to A01-style wells where possible, with __raw columns retained",
            },
            {
                "item": "assay_key_policy",
                "value": "source plate/well columns are never promoted to primary assay keys; use explicit plate/well arguments when ambiguous",
            },
        ]
    )
    all_reports = {
        "formatted_metadata": formatted,
        "metadata_validation_summary": validation_summary,
        "metadata_key_validation": key_validation,
        "metadata_annotation_merge_report": merge_report,
        "metadata_annotation_duplicate_key_report": annotation_duplicate_report,
        **reports,
    }
    if annotation_prepare_reports:
        all_reports["annotation_prepare_reports"] = pd.concat(annotation_prepare_reports, ignore_index=True)
    for name, table in all_reports.items():
        if name == "formatted_metadata":
            write_table(data_frame=table, path=output_dir / "formatted_metadata.tsv", logger=logger)
        else:
            write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(
        tables={name: table.head(5000) for name, table in all_reports.items()},
        path=output_dir / "metadata_validation_summary.xlsx",
        logger=logger,
    )
    warnings = []
    if (key_validation.get("status", pd.Series(dtype=str)) == "review").any():
        warnings.append("One or more metadata key checks require review. Check missing or duplicated plate/well keys.")
    if annotations and not merge_report.empty:
        if pd.to_numeric(merge_report.get("n_rows_without_annotation_match"), errors="coerce").fillna(0).gt(0).any():
            warnings.append("One or more annotation merges left metadata rows without matching annotation values.")
    make_html_report(
        title="CPATK metadata validation report",
        output_path=output_dir / "metadata_validation_report.html",
        summary_tables={
            "Metadata validation summary": validation_summary,
            "Metadata key validation": key_validation,
            "Annotation merge report": merge_report,
            "Annotation duplicate-key report": annotation_duplicate_report.head(2000),
            "Metadata alias report": reports.get("metadata_alias_report", pd.DataFrame()),
            "Explicit key column report": reports.get("explicit_key_column_report", pd.DataFrame()),
            "Dropped index column report": reports.get("dropped_index_column_report", pd.DataFrame()),
            "Formatted metadata preview": formatted.head(50),
        },
        narrative=(
            "CPATK checked and formatted the supplied metadata before analysis. "
            "Well identifiers were canonicalised where possible with raw values retained, common metadata aliases were standardised, "
            "source-library keys were kept separate from assay keys, and optional annotation tables were merged using strict key checks."
        ),
        warnings=warnings,
    )
    return all_reports
