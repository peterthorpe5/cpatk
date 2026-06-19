"""Metadata normalisation helpers for generic Cell Painting projects.

The functions in this module are deliberately conservative. They do not try to
force every project into a single schema, but they make common Cell Painting
metadata variants easier to recognise and report.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

CANONICAL_METADATA_ALIASES: Mapping[str, Sequence[str]] = {
    "Metadata_Plate": (
        "Metadata_Plate",
        "Plate_Metadata",
        "Plate",
        "plate",
        "plate_id",
        "PlateID",
        "Image_Metadata_Plate",
        "Metadata_PlateID",
        "Assay_Plate_Barcode",
        "Source_Plate_Barcode",
    ),
    "Metadata_Well": (
        "Metadata_Well",
        "Well_Metadata",
        "Well",
        "well",
        "well_id",
        "Image_Metadata_Well",
        "DestinationWell",
        "Destination_Well",
        "well_position",
    ),
    "Metadata_Row": (
        "Metadata_Row",
        "Row_Metadata",
        "Row",
        "row",
        "Plate_Row",
        "Image_Metadata_Row",
    ),
    "Metadata_Column": (
        "Metadata_Column",
        "Column_Metadata",
        "Column",
        "column",
        "Plate_Column",
        "Image_Metadata_Column",
    ),
    "Metadata_Site": (
        "Metadata_Site",
        "Field_Metadata",
        "Site",
        "site",
        "Image_Metadata_Site",
        "ImageNumber_Metadata_Site",
        "Field",
    ),
    "Metadata_Compound": (
        "Metadata_Compound",
        "Compound",
        "compound",
        "name",
        "Name",
        "cpd_id",
        "compound_id",
        "COMPOUND_NUMBER",
        "pert_iname",
        "treatment",
        "Treatment",
    ),
    "Metadata_MOA": (
        "Metadata_MOA",
        "MOA",
        "moa",
        "Mechanism",
        "mechanism",
        "cpd_type",
        "class",
        "Class",
        "mode_of_action",
        "ModeOfAction",
    ),
    "Metadata_Dose": (
        "Metadata_Dose",
        "Dose",
        "dose",
        "Destination Concentration",
        "Destination_Concentration",
        "concentration",
        "Concentration",
        "pert_dose",
    ),
    "Metadata_Dose_Unit": (
        "Metadata_Dose_Unit",
        "Destination Concentration Units",
        "Destination_Concentration_Units",
        "Dose_Unit",
        "dose_unit",
        "Concentration_Units",
    ),
    "Metadata_Batch": (
        "Metadata_Batch",
        "Batch",
        "batch",
        "Library",
        "library",
        "Source_Plate_Barcode",
        "Experiment",
        "experiment",
    ),
}


def clean_column_name(*, column_name: object) -> str:
    """Return a readable and reproducible column name.

    CellProfiler feature names are already meaningful, so this function avoids
    aggressive renaming. It removes BOM characters, strips whitespace and
    collapses repeated whitespace while preserving punctuation used in feature
    names.
    """
    name = str(column_name).replace("\ufeff", "").strip()
    name = re.sub(pattern=r"\s+", repl=" ", string=name)
    return name


def normalise_column_names(*, data_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean column names and make duplicates unique."""
    cleaned_names = []
    counts: Dict[str, int] = {}
    records = []
    for original in data_frame.columns:
        cleaned = clean_column_name(column_name=original)
        if cleaned in counts:
            counts[cleaned] += 1
            unique = f"{cleaned}__duplicate_{counts[cleaned]}"
        else:
            counts[cleaned] = 0
            unique = cleaned
        cleaned_names.append(unique)
        records.append(
            {
                "original_column": str(original),
                "cleaned_column": unique,
                "changed": bool(str(original) != unique),
            }
        )
    output = data_frame.copy()
    output.columns = cleaned_names
    return output, pd.DataFrame.from_records(records)


def canonicalise_well_value(*, value: object) -> object:
    """Return a canonical well identifier where possible.

    Examples include converting ``A1`` to ``A01`` while leaving missing or
    non-standard values unchanged.
    """
    if pd.isna(value):
        return value
    text = str(value).strip()
    match = re.fullmatch(pattern=r"([A-Za-z])0*([0-9]{1,2})", string=text)
    if match:
        return f"{match.group(1).upper()}{int(match.group(2)):02d}"
    return text


def canonicalise_row_value(*, value: object) -> object:
    """Return an uppercase one-letter plate row when possible."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    if re.fullmatch(pattern=r"[A-Za-z]", string=text):
        return text.upper()
    return text


def canonicalise_column_value(*, value: object) -> object:
    """Return an integer-like plate column when possible."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    if re.fullmatch(pattern=r"0*[0-9]{1,3}", string=text):
        return int(text)
    return text


def derive_well_from_row_column(
    *,
    data_frame: pd.DataFrame,
    row_column: str = "Metadata_Row",
    column_column: str = "Metadata_Column",
    well_column: str = "Metadata_Well",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create ``Metadata_Well`` from row/column metadata when needed.

    This is useful for CellProfiler Image tables that contain separate row and
    column metadata but no single well identifier. Existing non-missing
    ``Metadata_Well`` values are preserved.
    """
    output = data_frame.copy()
    records = []
    if row_column not in output.columns or column_column not in output.columns:
        return output, pd.DataFrame.from_records(
            [
                {
                    "action": "not_created",
                    "target_column": well_column,
                    "reason": "row_or_column_metadata_missing",
                    "n_values_created": 0,
                }
            ]
        )
    row_values = output[row_column].map(lambda value: canonicalise_row_value(value=value))
    column_values = output[column_column].map(lambda value: canonicalise_column_value(value=value))
    derived = []
    for row_value, column_value in zip(row_values, column_values):
        if pd.isna(row_value) or pd.isna(column_value):
            derived.append(pd.NA)
            continue
        if isinstance(column_value, int):
            derived.append(f"{row_value}{column_value:02d}")
        else:
            derived.append(canonicalise_well_value(value=f"{row_value}{column_value}"))
    derived_series = pd.Series(derived, index=output.index, dtype="object")
    if well_column in output.columns:
        missing_mask = output[well_column].isna() | (output[well_column].astype(str).str.strip() == "")
        output.loc[missing_mask, well_column] = derived_series.loc[missing_mask]
        action = "filled_missing_existing"
        n_created = int(missing_mask.sum())
    else:
        output[well_column] = derived_series
        action = "created"
        n_created = int(derived_series.notna().sum())
    output[well_column] = output[well_column].map(lambda value: canonicalise_well_value(value=value))
    records.append(
        {
            "action": action,
            "target_column": well_column,
            "source_row_column": row_column,
            "source_column_column": column_column,
            "reason": "derived_from_plate_row_and_column",
            "n_values_created": n_created,
        }
    )
    return output, pd.DataFrame.from_records(records)


def standardise_metadata_aliases(
    *,
    data_frame: pd.DataFrame,
    alias_map: Mapping[str, Sequence[str]] = CANONICAL_METADATA_ALIASES,
    overwrite_existing: bool = False,
    derive_well: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add canonical metadata aliases when recognised source columns exist."""
    output = data_frame.copy()
    lower_to_column = {str(column).lower(): column for column in output.columns}
    records = []
    for canonical, aliases in alias_map.items():
        matched = None
        for alias in aliases:
            if alias.lower() in lower_to_column:
                matched = lower_to_column[alias.lower()]
                break
        action = "not_found"
        if matched is not None:
            if canonical not in output.columns or overwrite_existing:
                output[canonical] = output[matched]
                action = "created" if canonical != matched else "already_present"
            else:
                action = "kept_existing"
            if canonical == "Metadata_Well":
                output[canonical] = output[canonical].map(lambda value: canonicalise_well_value(value=value))
            if canonical == "Metadata_Row":
                output[canonical] = output[canonical].map(lambda value: canonicalise_row_value(value=value))
            if canonical == "Metadata_Column":
                output[canonical] = output[canonical].map(lambda value: canonicalise_column_value(value=value))
        records.append(
            {
                "canonical_column": canonical,
                "matched_source_column": matched if matched is not None else "",
                "action": action,
            }
        )
    report = pd.DataFrame.from_records(records)
    if derive_well:
        output, derive_report = derive_well_from_row_column(data_frame=output)
        for _, row in derive_report.iterrows():
            report = pd.concat(
                [
                    report,
                    pd.DataFrame.from_records(
                        [
                            {
                                "canonical_column": row.get("target_column", "Metadata_Well"),
                                "matched_source_column": f"{row.get('source_row_column', '')},{row.get('source_column_column', '')}",
                                "action": row.get("action", "not_created"),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    if logger is not None:
        logger.info("Standardised %s metadata aliases", int((report["action"] == "created").sum()))
    return output, report


def detect_unnamed_index_columns(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """Identify likely accidental index columns from CSV exports."""
    records = []
    for column in data_frame.columns:
        column_text = str(column)
        if column_text.lower().startswith("unnamed"):
            series = pd.to_numeric(data_frame[column], errors="coerce")
            sequential = bool(series.notna().all() and series.reset_index(drop=True).equals(pd.Series(range(len(series)), dtype=series.dtype)))
            records.append(
                {
                    "column": column,
                    "reason": "unnamed_csv_index",
                    "looks_sequential": sequential,
                }
            )
    return pd.DataFrame.from_records(records)


def drop_unnamed_index_columns(
    *,
    data_frame: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns that look like accidental CSV index exports."""
    report = detect_unnamed_index_columns(data_frame=data_frame)
    if report.empty:
        return data_frame.copy(), report
    columns_to_drop = report.loc[report["looks_sequential"], "column"].tolist()
    output = data_frame.drop(columns=columns_to_drop, errors="ignore").copy()
    report["dropped"] = report["column"].isin(columns_to_drop)
    if logger is not None and columns_to_drop:
        logger.info("Dropped %s likely CSV index columns: %s", len(columns_to_drop), columns_to_drop)
    return output, report
