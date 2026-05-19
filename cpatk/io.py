"""Input and output helpers for CPATK.

The package intentionally writes tab-separated, Parquet, Excel and HTML files.
Comma-separated output is not used by CPATK writing helpers.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment

TableLike = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]


def read_table(
    *,
    path: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Read a table from TSV, CSV, Excel or Parquet.

    Parameters
    ----------
    path:
        Input table path.
    sheet_name:
        Sheet name or index for Excel input.
    logger:
        Optional logger.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if logger is not None:
        logger.info("Reading table: %s", path)

    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path=path)
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(filepath_or_buffer=path, sep="\t")
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(filepath_or_buffer=path)
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(io=path, sheet_name=sheet_name or 0)

    raise ValueError(f"Unsupported input table format: {path}")


def write_table(
    *,
    data_frame: pd.DataFrame,
    path: Union[str, Path],
    index: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write a table to TSV, TSV.GZ, Parquet or Excel.

    Parameters
    ----------
    data_frame:
        Data frame to write.
    path:
        Output path. Supported suffixes are ``.tsv``, ``.tsv.gz``,
        ``.parquet`` and ``.xlsx``.
    index:
        Whether to write the data-frame index.
    logger:
        Optional logger.

    Returns
    -------
    pathlib.Path
        Written output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()
    if logger is not None:
        logger.info("Writing table: %s", path)

    if suffixes.endswith(".parquet"):
        data_frame.to_parquet(path=path, index=index)
    elif suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        data_frame.to_csv(path_or_buf=path, sep="\t", index=index)
    elif suffixes.endswith(".xlsx"):
        write_excel_workbook(
            tables={"Sheet1": data_frame},
            path=path,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported output table format: {path}")
    return path


def sanitise_sheet_name(*, sheet_name: str) -> str:
    """Return a valid Excel sheet name no longer than 31 characters.

    Parameters
    ----------
    sheet_name:
        Proposed sheet name.

    Returns
    -------
    str
        Sanitised sheet name.
    """
    invalid = set("[]:*?/\\")
    cleaned = "".join("_" if character in invalid else character for character in sheet_name)
    cleaned = cleaned.strip() or "Sheet"
    return cleaned[:31]


def write_excel_workbook(
    *,
    tables: Mapping[str, pd.DataFrame],
    path: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write a formatted Excel workbook with one sheet per table.

    Parameters
    ----------
    tables:
        Mapping of sheet names to data frames.
    path:
        Output workbook path.
    logger:
        Optional logger.

    Returns
    -------
    pathlib.Path
        Written workbook path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.info("Writing formatted Excel workbook: %s", path)

    with pd.ExcelWriter(path=path, engine="openpyxl") as writer:
        used_names = set()
        for sheet_name, table in tables.items():
            safe_name = sanitise_sheet_name(sheet_name=sheet_name)
            base_name = safe_name
            counter = 1
            while safe_name in used_names:
                suffix = f"_{counter}"
                safe_name = f"{base_name[:31 - len(suffix)]}{suffix}"
                counter += 1
            used_names.add(safe_name)
            table.to_excel(excel_writer=writer, sheet_name=safe_name, index=False)
            worksheet = writer.sheets[safe_name]
            format_worksheet(worksheet=worksheet, data_frame=table)
    return path


def format_worksheet(*, worksheet, data_frame: pd.DataFrame) -> None:
    """Apply readable formatting to an openpyxl worksheet.

    Parameters
    ----------
    worksheet:
        Openpyxl worksheet object.
    data_frame:
        Data frame written to the worksheet.
    """
    header_fill = PatternFill(fill_type="solid", fgColor="ECEFF1")
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for column_index, column_name in enumerate(data_frame.columns, start=1):
        values = data_frame[column_name].astype(str).head(200).tolist()
        max_length = max([len(str(column_name)), *[len(value) for value in values]])
        width = min(max(max_length + 2, 10), 45)
        worksheet.column_dimensions[get_column_letter(column_index)].width = width

    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=False)


def data_frame_to_html_table(
    *,
    data_frame: pd.DataFrame,
    max_rows: int = 20,
) -> str:
    """Convert a data frame to a small HTML table.

    Parameters
    ----------
    data_frame:
        Data frame to render.
    max_rows:
        Maximum number of rows to show.

    Returns
    -------
    str
        HTML table string.
    """
    shown = data_frame.head(n=max_rows).copy()
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in shown.columns)
    rows = []
    for _, row in shown.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def list_supported_tables(*, input_dir: Union[str, Path]) -> pd.DataFrame:
    """List recognised table files in a directory.

    Parameters
    ----------
    input_dir:
        Directory to scan.

    Returns
    -------
    pandas.DataFrame
        Inventory of candidate table files.
    """
    input_dir = Path(input_dir)
    patterns = ["*.tsv", "*.tsv.gz", "*.csv", "*.csv.gz", "*.parquet", "*.xlsx"]
    records = []
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            records.append(
                {
                    "path": str(path),
                    "file_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "suffixes": "".join(path.suffixes),
                }
            )
    return pd.DataFrame.from_records(records)
