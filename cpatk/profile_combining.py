"""Combine pre-built CPATK profile tables across plates or exports.

The profile combiner is deliberately separate from raw CellProfiler profile
building. It is intended for production multi-plate projects where each plate
or export has already been built and checked, then the resulting profile tables
need to be stacked into one joint analysis matrix.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from cpatk.features import infer_feature_columns, infer_metadata_columns
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.metadata import normalise_column_names, standardise_metadata_aliases
from cpatk.reporting import make_html_report


COMBINE_SOURCE_COLUMN = "Metadata_Profile_Source"

PROFILE_KEY_PRIORITY_COLUMNS = [
    COMBINE_SOURCE_COLUMN,
    "Metadata_Plate",
    "ImageNumber",
    "Metadata_ImageNumber",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Field",
    "Metadata_FieldID",
    "Metadata_SiteID",
    "Metadata_Frame",
    "Metadata_Tile",
]


def _deduplicate_preserving_order(*, values: Sequence[str]) -> list[str]:
    """Return non-empty strings once, preserving their first appearance."""
    seen = set()
    deduplicated = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduplicated.append(cleaned)
    return deduplicated


def _normalise_key_columns(
    *,
    key_columns: Optional[Sequence[str]],
    tables: Sequence[pd.DataFrame],
) -> list[str]:
    """Return requested key columns that are present in at least one table."""
    if key_columns is None:
        requested = [
            COMBINE_SOURCE_COLUMN,
            "Metadata_Plate",
            "ImageNumber",
            "Metadata_Well",
            "Metadata_Site",
        ]
    else:
        requested = list(key_columns)
    available = set().union(*(set(table.columns) for table in tables))
    return [
        column
        for column in _deduplicate_preserving_order(values=requested)
        if column in available
    ]


def _count_duplicate_key_rows(
    *,
    data_frame: pd.DataFrame,
    key_columns: Sequence[str],
) -> tuple[int, int]:
    """Count duplicated rows and duplicated groups for a candidate key set."""
    key_columns = [column for column in key_columns if column in data_frame.columns]
    if not key_columns:
        return 0, 0
    duplicated = data_frame.duplicated(subset=key_columns, keep=False)
    if not duplicated.any():
        return 0, 0
    n_duplicate_rows = int(duplicated.sum())
    n_duplicate_groups = int(data_frame.loc[duplicated, key_columns].drop_duplicates().shape[0])
    return n_duplicate_rows, n_duplicate_groups


def make_key_candidate_report(
    *,
    data_frame: pd.DataFrame,
    key_columns: Sequence[str],
    max_extra_columns: int = 3,
) -> pd.DataFrame:
    """Assess likely combined-profile key sets.

    Parameters
    ----------
    data_frame:
        Combined profile table to check.
    key_columns:
        Current or requested key columns.
    max_extra_columns:
        Maximum number of extra candidate columns to append to the current
        key when looking for a unique profile identifier.

    Returns
    -------
    pandas.DataFrame
        Candidate key audit table. ``status`` is ``unique`` when the key set
        identifies each row once, otherwise ``duplicate_key``.
    """
    base_columns = [column for column in key_columns if column in data_frame.columns]
    base_columns = _deduplicate_preserving_order(values=base_columns)
    available_columns = set(data_frame.columns)
    priority_columns = [
        column
        for column in PROFILE_KEY_PRIORITY_COLUMNS
        if column in available_columns and column not in base_columns
    ]
    records = []

    candidate_sets = [base_columns]
    for extra_count in range(1, max_extra_columns + 1):
        for extra_columns in combinations(priority_columns, extra_count):
            candidate_sets.append([*base_columns, *extra_columns])

    seen_sets = set()
    for candidate_columns in candidate_sets:
        candidate_columns = _deduplicate_preserving_order(values=candidate_columns)
        key = tuple(candidate_columns)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        if not candidate_columns:
            records.append(
                {
                    "status": "skipped_no_key_columns",
                    "key_columns": "",
                    "added_columns": "",
                    "n_key_columns": 0,
                    "n_duplicate_rows": 0,
                    "n_duplicate_groups": 0,
                    "n_missing_key_rows": 0,
                }
            )
            continue
        n_duplicate_rows, n_duplicate_groups = _count_duplicate_key_rows(
            data_frame=data_frame,
            key_columns=candidate_columns,
        )
        missing_key_rows = int(data_frame[list(candidate_columns)].isna().any(axis=1).sum())
        added_columns = [column for column in candidate_columns if column not in base_columns]
        records.append(
            {
                "status": "unique" if n_duplicate_rows == 0 else "duplicate_key",
                "key_columns": ";".join(candidate_columns),
                "added_columns": ";".join(added_columns),
                "n_key_columns": int(len(candidate_columns)),
                "n_duplicate_rows": n_duplicate_rows,
                "n_duplicate_groups": n_duplicate_groups,
                "n_missing_key_rows": missing_key_rows,
            }
        )

    report = pd.DataFrame.from_records(records)
    if report.empty:
        return report
    report["is_requested_key"] = report["key_columns"] == ";".join(base_columns)
    report["is_unique"] = report["status"] == "unique"
    report = report.sort_values(
        by=["is_unique", "n_key_columns", "n_duplicate_rows"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return report


def choose_default_key_columns(
    *,
    tables: Sequence[pd.DataFrame],
    logger: Optional[logging.Logger] = None,
) -> list[str]:
    """Choose a sensible default key set for combined profile tables.

    CPATK profile tables are usually image-level rows. In that case, well-level
    keys are too coarse because one well often contains many images or sites.
    The default therefore prefers source, plate and image identity where those
    columns are available.
    """
    key_columns = _normalise_key_columns(key_columns=None, tables=tables)
    if logger is not None:
        logger.info("Using default combined-profile key columns: %s", ";".join(key_columns))
    return key_columns


def _format_unique_key_suggestions(*, candidate_report: pd.DataFrame) -> str:
    """Return compact suggested unique key sets for an exception message."""
    if candidate_report.empty or "is_unique" not in candidate_report.columns:
        return "No unique candidate key set could be inferred."
    unique_candidates = candidate_report.loc[candidate_report["is_unique"], "key_columns"].head(3)
    if unique_candidates.empty:
        return "No unique candidate key set could be inferred."
    joined = " | ".join(str(value) for value in unique_candidates.tolist())
    return f"Candidate unique key set(s): {joined}."


def _normalise_label(*, path: Path, label: Optional[str], index: int) -> str:
    """Return a safe source label for a profile table."""
    if label is not None and str(label).strip():
        return str(label).strip()
    stem = path.name
    for suffix in [".tsv.gz", ".csv.gz", ".parquet", ".tsv", ".csv", ".xlsx", ".xls"]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or f"profile_{index + 1}"


def _prepare_profile_table(
    *,
    data_frame: pd.DataFrame,
    source_label: str,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalise one pre-built profile table before combining.

    Parameters
    ----------
    data_frame:
        Input profile table.
    source_label:
        Label added to ``Metadata_Profile_Source``.
    logger:
        Optional logger.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Cleaned profile table, column-name report and metadata-alias report.
    """
    cleaned, column_report = normalise_column_names(data_frame=data_frame)
    cleaned, alias_report = standardise_metadata_aliases(data_frame=cleaned, logger=logger)
    if COMBINE_SOURCE_COLUMN in cleaned.columns:
        cleaned[f"{COMBINE_SOURCE_COLUMN}__input"] = cleaned[COMBINE_SOURCE_COLUMN]
    cleaned[COMBINE_SOURCE_COLUMN] = source_label
    return cleaned, column_report, alias_report


def _select_columns_for_combination(
    *,
    tables: Sequence[pd.DataFrame],
    feature_join: str,
    key_columns: Sequence[str],
) -> tuple[list[str], pd.DataFrame]:
    """Choose columns to retain when stacking profile tables."""
    feature_join = feature_join.lower()
    if feature_join not in {"union", "intersection"}:
        raise ValueError("feature_join must be one of: union, intersection.")
    metadata_union: list[str] = []
    feature_sets: list[set[str]] = []
    records = []
    for table_index, table in enumerate(tables):
        metadata_columns = infer_metadata_columns(
            data_frame=table,
            additional_metadata_columns=[
                *key_columns,
                *PROFILE_KEY_PRIORITY_COLUMNS,
                COMBINE_SOURCE_COLUMN,
            ],
        )
        feature_columns = infer_feature_columns(data_frame=table, metadata_columns=metadata_columns)
        feature_sets.append(set(feature_columns))
        for column in metadata_columns:
            if column not in metadata_union:
                metadata_union.append(column)
        for feature in feature_columns:
            records.append(
                {
                    "table_index": table_index,
                    "feature": feature,
                    "present": True,
                }
            )
    if not feature_sets:
        raise ValueError("No profile tables were supplied.")
    if feature_join == "intersection":
        selected_features = sorted(set.intersection(*feature_sets))
    else:
        selected_features = sorted(set.union(*feature_sets))
    selected_columns = [column for column in metadata_union if column not in selected_features] + selected_features
    feature_presence = pd.DataFrame.from_records(records)
    if not feature_presence.empty:
        feature_presence = (
            feature_presence.assign(value=1)
            .pivot_table(index="feature", columns="table_index", values="value", fill_value=0, aggfunc="max")
            .reset_index()
        )
    return selected_columns, feature_presence


def check_combined_key_uniqueness(
    *,
    data_frame: pd.DataFrame,
    key_columns: Sequence[str],
    duplicate_policy: str = "error",
    candidate_report: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Check duplicate profile keys in a combined table.

    Parameters
    ----------
    data_frame:
        Combined profile table.
    key_columns:
        Columns expected to define unique profiles.
    duplicate_policy:
        ``error`` or ``allow``.
    candidate_report:
        Optional candidate-key report used to make duplicate-key errors more
        actionable.

    Returns
    -------
    pandas.DataFrame
        Duplicate-key report.
    """
    key_columns = [column for column in key_columns if column in data_frame.columns]
    if not key_columns:
        return pd.DataFrame.from_records(
            [
                {
                    "status": "skipped_no_key_columns",
                    "key_columns": "",
                    "n_duplicate_rows": 0,
                    "n_duplicate_groups": 0,
                }
            ]
        )
    duplicated = data_frame.duplicated(subset=key_columns, keep=False)
    if not duplicated.any():
        return pd.DataFrame.from_records(
            [
                {
                    "status": "ok_unique_keys",
                    "key_columns": ";".join(key_columns),
                    "n_duplicate_rows": 0,
                    "n_duplicate_groups": 0,
                }
            ]
        )
    duplicate_groups = data_frame.loc[duplicated, key_columns].drop_duplicates()
    report = duplicate_groups.copy()
    report["status"] = "duplicate_key"
    report["key_columns"] = ";".join(key_columns)
    report["n_duplicate_rows_total"] = int(duplicated.sum())
    report["n_duplicate_groups_total"] = int(duplicate_groups.shape[0])
    if duplicate_policy == "error":
        preview = report.head(10).to_dict(orient="records")
        suggestion_text = ""
        if candidate_report is not None:
            suggestion_text = " " + _format_unique_key_suggestions(candidate_report=candidate_report)
        raise ValueError(
            f"Combined profile keys are not unique for {key_columns}. Preview: {preview}."
            f"{suggestion_text} Use a more specific key set, or rerun only after "
            "reviewing with duplicate_policy='allow'."
        )
    if duplicate_policy != "allow":
        raise ValueError("duplicate_policy must be either error or allow.")
    return report


def combine_profile_tables(
    *,
    profile_tables: Sequence[pd.DataFrame],
    profile_paths: Optional[Sequence[Path | str]] = None,
    source_labels: Optional[Sequence[str]] = None,
    key_columns: Optional[Sequence[str]] = None,
    feature_join: str = "union",
    duplicate_policy: str = "error",
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Combine multiple already-built profile tables.

    Parameters
    ----------
    profile_tables:
        Input profile tables.
    profile_paths:
        Optional paths used for reporting and source-label defaults.
    source_labels:
        Optional labels, one per table.
    key_columns:
        Columns expected to define unique profiles after combining. If omitted,
        CPATK prefers source, plate and image identity where present.
    feature_join:
        ``union`` keeps all features and fills missing table-specific features
        as missing values; ``intersection`` keeps only features present in all
        tables.
    duplicate_policy:
        ``error`` or ``allow`` for duplicate combined keys.
    logger:
        Optional logger.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, pandas.DataFrame]]
        Combined table and audit tables.
    """
    if len(profile_tables) < 2:
        raise ValueError("At least two profile tables are required for cpatk-combine-profiles.")
    paths = [Path(path) for path in (profile_paths or [f"profile_{i + 1}" for i in range(len(profile_tables))])]
    labels = list(source_labels or [])
    if labels and len(labels) != len(profile_tables):
        raise ValueError("source_labels must have one value per profile table.")
    prepared_tables = []
    column_reports = []
    alias_reports = []
    input_records = []
    for index, table in enumerate(profile_tables):
        label = _normalise_label(
            path=paths[index],
            label=labels[index] if labels else None,
            index=index,
        )
        cleaned, column_report, alias_report = _prepare_profile_table(
            data_frame=table,
            source_label=label,
            logger=logger,
        )
        column_report.insert(0, "source_label", label)
        alias_report.insert(0, "source_label", label)
        column_reports.append(column_report)
        alias_reports.append(alias_report)
        prepared_tables.append(cleaned)
        input_records.append(
            {
                "source_label": label,
                "path": str(paths[index]),
                "n_rows": int(cleaned.shape[0]),
                "n_columns": int(cleaned.shape[1]),
            }
        )
    if key_columns is None:
        key_columns = choose_default_key_columns(tables=prepared_tables, logger=logger)
    else:
        key_columns = _normalise_key_columns(key_columns=key_columns, tables=prepared_tables)
    key_columns = list(key_columns or [])
    selected_columns, feature_presence = _select_columns_for_combination(
        tables=prepared_tables,
        feature_join=feature_join,
        key_columns=key_columns,
    )
    aligned = [table.reindex(columns=selected_columns) for table in prepared_tables]
    combined = pd.concat(aligned, axis=0, ignore_index=True)
    key_candidate_report = make_key_candidate_report(
        data_frame=combined,
        key_columns=key_columns,
    )
    duplicate_report = check_combined_key_uniqueness(
        data_frame=combined,
        key_columns=key_columns,
        duplicate_policy=duplicate_policy,
        candidate_report=key_candidate_report,
    )
    feature_columns = infer_feature_columns(
        data_frame=combined,
        metadata_columns=infer_metadata_columns(data_frame=combined),
    )
    summary = pd.DataFrame.from_records(
        [
            {"item": "n_profile_tables", "value": int(len(profile_tables))},
            {"item": "n_combined_rows", "value": int(combined.shape[0])},
            {"item": "n_combined_columns", "value": int(combined.shape[1])},
            {"item": "n_combined_features", "value": int(len(feature_columns))},
            {"item": "feature_join", "value": feature_join},
            {"item": "key_columns", "value": ";".join(key_columns)},
            {"item": "duplicate_policy", "value": duplicate_policy},
        ]
    )
    if logger is not None:
        logger.info(
            "Combined %d profile tables into %d rows and %d feature columns",
            len(profile_tables),
            combined.shape[0],
            len(feature_columns),
        )
    reports = {
        "combine_profile_summary": summary,
        "input_profile_report": pd.DataFrame.from_records(input_records),
        "combined_duplicate_key_report": duplicate_report,
        "combined_key_candidate_report": key_candidate_report,
        "feature_presence_matrix": feature_presence,
        "column_name_report": pd.concat(column_reports, ignore_index=True),
        "metadata_alias_report": pd.concat(alias_reports, ignore_index=True),
        "retained_combined_features": pd.DataFrame({"feature": feature_columns}),
    }
    return combined, reports


def run_combine_profiles_workflow(
    *,
    profile_paths: Sequence[Path | str],
    output_dir: Path | str,
    source_labels: Optional[Sequence[str]] = None,
    key_columns: Optional[Sequence[str]] = None,
    feature_join: str = "union",
    duplicate_policy: str = "error",
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run the command-line combine-profiles workflow."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_tables = [read_table(path=path, logger=logger) for path in profile_paths]
    combined, reports = combine_profile_tables(
        profile_tables=profile_tables,
        profile_paths=profile_paths,
        source_labels=source_labels,
        key_columns=key_columns,
        feature_join=feature_join,
        duplicate_policy=duplicate_policy,
        logger=logger,
    )
    write_table(data_frame=combined, path=output_dir / "combined_profiles.tsv.gz", logger=logger)
    for name, table in reports.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=reports, path=output_dir / "combine_profiles_summary.xlsx", logger=logger)
    make_html_report(
        title="CPATK combined profile report",
        output_path=output_dir / "combine_profiles_report.html",
        summary_tables=reports,
        narrative=(
            "CPATK combined multiple already-built profile tables into one joint analysis table. "
            "Use this step when plates or exports have been built and reviewed separately, especially when "
            "CellProfiler ImageNumber values may restart in each export."
        ),
        warnings=[],
    )
    return combined, reports
