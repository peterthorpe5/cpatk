"""Plate-layout diagnostics for Cell Painting profiles.

This module is deliberately generic. It does not assume a specific assay,
cell type, stain, or treatment layout. It provides helpers for extracting
plate row/column information from well names and summarising metric patterns
across plate positions, treatments, donors, batches, or other metadata labels.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from cpatk.plotting import plot_heatmap


WELL_PATTERN = re.compile(r"^([A-Za-z]+)0*([0-9]+)$")


def normalise_well_name(*, value: object) -> Optional[str]:
    """Return a normalised well name such as ``A01``.

    Parameters
    ----------
    value:
        Well value to normalise.

    Returns
    -------
    str or None
        Normalised well name, or ``None`` when the value cannot be parsed.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = WELL_PATTERN.match(text)
    if match is None:
        return text if text else None
    row, column = match.groups()
    return f"{row.upper()}{int(column):02d}"


def add_plate_position_columns(
    *,
    data_frame: pd.DataFrame,
    well_column: str = "Well_Metadata",
    row_column: str = "plate_row",
    column_column: str = "plate_column",
) -> pd.DataFrame:
    """Add parsed plate row and column fields from a well column.

    Parameters
    ----------
    data_frame:
        Input table.
    well_column:
        Column containing well names.
    row_column:
        Output plate-row column name.
    column_column:
        Output plate-column column name.

    Returns
    -------
    pandas.DataFrame
        Copy of input table with parsed plate-position fields.
    """
    if well_column not in data_frame.columns:
        raise ValueError(f"Well column is missing: {well_column}")
    output = data_frame.copy()
    normalised = output[well_column].map(lambda item: normalise_well_name(value=item))
    output[well_column] = normalised
    rows = []
    columns = []
    for well in normalised:
        match = WELL_PATTERN.match(str(well)) if well is not None else None
        if match is None:
            rows.append(np.nan)
            columns.append(np.nan)
        else:
            row, column = match.groups()
            rows.append(row.upper())
            columns.append(int(column))
    output[row_column] = rows
    output[column_column] = columns
    return output


def summarise_plate_metric(
    *,
    data_frame: pd.DataFrame,
    metric_column: str,
    well_column: str = "Well_Metadata",
    row_column: str = "plate_row",
    column_column: str = "plate_column",
) -> pd.DataFrame:
    """Create a plate-layout matrix for a numeric metric.

    Parameters
    ----------
    data_frame:
        Input table with one or more rows per well.
    metric_column:
        Numeric metric to summarise by well.
    well_column:
        Well column.
    row_column:
        Plate-row column.
    column_column:
        Plate-column column.

    Returns
    -------
    pandas.DataFrame
        Row-by-column matrix of median metric values.
    """
    if metric_column not in data_frame.columns:
        raise ValueError(f"Metric column is missing: {metric_column}")
    table = add_plate_position_columns(
        data_frame=data_frame,
        well_column=well_column,
        row_column=row_column,
        column_column=column_column,
    )
    table[metric_column] = pd.to_numeric(table[metric_column], errors="coerce")
    grouped = (
        table.groupby([row_column, column_column], dropna=False)[metric_column]
        .median()
        .reset_index()
    )
    matrix = grouped.pivot(index=row_column, columns=column_column, values=metric_column)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    matrix.columns = [f"{int(column):02d}" if pd.notna(column) else "NA" for column in matrix.columns]
    return matrix


def summarise_layout_axis(
    *,
    data_frame: pd.DataFrame,
    metric_columns: Sequence[str],
    axis_columns: Sequence[str],
) -> pd.DataFrame:
    """Summarise metrics by layout or metadata axes.

    Parameters
    ----------
    data_frame:
        Input table.
    metric_columns:
        Numeric metrics to summarise.
    axis_columns:
        Metadata/layout columns used as one-variable-at-a-time axes.

    Returns
    -------
    pandas.DataFrame
        Long-format summary of metric medians, means and standard deviations.
    """
    records = []
    valid_metrics = [column for column in metric_columns if column in data_frame.columns]
    for axis_column in axis_columns:
        if axis_column not in data_frame.columns:
            continue
        for axis_value, group in data_frame.groupby(axis_column, dropna=False):
            record = {
                "layout_axis": axis_column,
                "layout_value": axis_value,
                "n_rows": int(group.shape[0]),
            }
            for metric_column in valid_metrics:
                values = pd.to_numeric(group[metric_column], errors="coerce")
                record[f"median_{metric_column}"] = float(values.median(skipna=True))
                record[f"mean_{metric_column}"] = float(values.mean(skipna=True))
                record[f"sd_{metric_column}"] = float(values.std(skipna=True))
            records.append(record)
    return pd.DataFrame.from_records(records)


def run_plate_layout_diagnostics(
    *,
    data_frame: pd.DataFrame,
    output_dir: Path,
    well_column: str,
    metric_columns: Sequence[str],
    grouping_columns: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Mapping[str, pd.DataFrame]:
    """Run plate-layout diagnostics and write heatmaps.

    Parameters
    ----------
    data_frame:
        Input profile or metric table.
    output_dir:
        Output directory.
    well_column:
        Well metadata column.
    metric_columns:
        Numeric metric columns to inspect.
    grouping_columns:
        Optional metadata columns such as treatment, donor or batch.
    logger:
        Optional logger.

    Returns
    -------
    mapping[str, pandas.DataFrame]
        Diagnostic summary tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = add_plate_position_columns(data_frame=data_frame, well_column=well_column)
    valid_metrics = [column for column in metric_columns if column in table.columns]
    matrices = {}
    for metric_column in valid_metrics:
        matrix = summarise_plate_metric(
            data_frame=table,
            metric_column=metric_column,
            well_column=well_column,
        )
        matrices[f"plate_matrix_{metric_column}"] = matrix.reset_index()
        try:
            plot_heatmap(
                matrix=matrix,
                output_path_base=output_dir / f"plate_heatmap_{metric_column}",
                title=f"Plate-layout heatmap: {metric_column}",
                value_label=metric_column,
                logger=logger,
            )
        except Exception as exc:  # pragma: no cover - plotting robustness
            if logger is not None:
                logger.warning("Could not write plate heatmap for %s: %s", metric_column, exc)
    axis_columns = ["plate_row", "plate_column", *(grouping_columns or [])]
    axis_summary = summarise_layout_axis(
        data_frame=table,
        metric_columns=valid_metrics,
        axis_columns=axis_columns,
    )
    warning = pd.DataFrame.from_records(
        [
            {
                "warning": (
                    "Plate-layout diagnostics make row, column, batch and treatment patterns visible. "
                    "They do not prove whether an apparent position effect is biological or technical. "
                    "If treatment, donor, batch or dose is arranged systematically by row or column, "
                    "strong causal interpretation requires a randomised or blocked repeat experiment."
                )
            }
        ]
    )
    result: dict[str, pd.DataFrame] = {
        "plate_layout_table": table,
        "layout_axis_summary": axis_summary,
        "plate_layout_warning": warning,
    }
    result.update(matrices)
    return result
