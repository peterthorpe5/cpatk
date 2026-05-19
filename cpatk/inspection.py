"""Inspection workflow for generic Cell Painting tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from cpatk.features import make_column_inventory
from cpatk.io import list_supported_tables, read_table


def inspect_table_file(
    *,
    path: Union[str, Path],
    max_preview_rows: int = 10,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Inspect one table file.

    Parameters
    ----------
    path:
        Input table path.
    max_preview_rows:
        Number of preview rows to keep.
    logger:
        Optional logger.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Inspection tables.
    """
    data_frame = read_table(path=path, logger=logger)
    summary = pd.DataFrame.from_records(
        [
            {
                "path": str(path),
                "n_rows": int(data_frame.shape[0]),
                "n_columns": int(data_frame.shape[1]),
            }
        ]
    )
    return {
        "summary": summary,
        "column_inventory": make_column_inventory(data_frame=data_frame),
        "preview": data_frame.head(n=max_preview_rows),
    }


def inspect_directory(
    *,
    input_dir: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Inspect all recognised tables in a directory.

    Parameters
    ----------
    input_dir:
        Directory to inspect.
    logger:
        Optional logger.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Directory-level inspection tables.
    """
    inventory = list_supported_tables(input_dir=input_dir)
    summaries = []
    columns = []
    for path_string in inventory.get("path", []):
        result = inspect_table_file(path=path_string, logger=logger)
        summary = result["summary"].copy()
        summaries.append(summary)
        column_inventory = result["column_inventory"].copy()
        column_inventory.insert(loc=0, column="file_name", value=Path(path_string).name)
        columns.append(column_inventory)
    if summaries:
        summary_table = pd.concat(summaries, ignore_index=True)
        column_table = pd.concat(columns, ignore_index=True)
    else:
        summary_table = pd.DataFrame(columns=["path", "n_rows", "n_columns"])
        column_table = pd.DataFrame()
    return {
        "file_inventory": inventory,
        "file_summary": summary_table,
        "column_inventory": column_table,
    }
