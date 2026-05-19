"""Static and interactive plotting helpers for CPATK."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_current_figure(
    *,
    output_path_base: Path,
    formats: Sequence[str] = ("pdf", "svg"),
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Save the current matplotlib figure in one or more formats.

    Parameters
    ----------
    output_path_base:
        Path without suffix.
    formats:
        Output formats.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    output_path_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for file_format in formats:
        path = output_path_base.with_suffix(f".{file_format}")
        plt.savefig(fname=path, bbox_inches="tight")
        written.append(path)
        if logger is not None:
            logger.info("Wrote plot: %s", path)
    plt.close()
    return written


def plot_embedding(
    *,
    embedding: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    x_column: str,
    y_column: str,
    colour_column: Optional[str],
    output_path_base: Path,
    title: str = "Cell Painting embedding",
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Create a static 2D embedding plot.

    Parameters
    ----------
    embedding:
        Embedding table.
    metadata:
        Optional metadata aligned to embedding rows.
    x_column:
        X-axis embedding column.
    y_column:
        Y-axis embedding column.
    colour_column:
        Optional metadata column for point colours.
    output_path_base:
        Output path without suffix.
    title:
        Plot title.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    plot_table = embedding.reset_index(drop=True).copy()
    if metadata is not None:
        plot_table = pd.concat([metadata.reset_index(drop=True), plot_table], axis=1)
    plt.figure(figsize=(7, 5))
    if colour_column and colour_column in plot_table.columns:
        groups = plot_table[colour_column].astype(str).fillna("NA")
        for group_name, group_table in plot_table.groupby(groups, dropna=False):
            plt.scatter(
                x=group_table[x_column],
                y=group_table[y_column],
                label=str(group_name),
                alpha=0.75,
                s=30,
            )
        plt.legend(fontsize=8, loc="best")
    else:
        plt.scatter(x=plot_table[x_column], y=plot_table[y_column], alpha=0.75, s=30)
    plt.xlabel(xlabel=x_column)
    plt.ylabel(ylabel=y_column)
    plt.title(label=title)
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_heatmap(
    *,
    matrix: pd.DataFrame,
    output_path_base: Path,
    title: str,
    value_label: str,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Create a simple matrix heatmap.

    Parameters
    ----------
    matrix:
        Matrix with row and column labels.
    output_path_base:
        Output path without suffix.
    title:
        Plot title.
    value_label:
        Colour-bar label.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    values = matrix.to_numpy(dtype=float)
    plt.figure(figsize=(max(6, matrix.shape[1] * 0.5), max(4, matrix.shape[0] * 0.4)))
    image = plt.imshow(values, aspect="auto")
    plt.colorbar(mappable=image, label=value_label)
    plt.xticks(ticks=np.arange(matrix.shape[1]), labels=matrix.columns, rotation=90)
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=matrix.index)
    plt.title(label=title)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                plt.text(
                    x=column_index,
                    y=row_index,
                    s=f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_metric_by_group(
    *,
    data_frame: pd.DataFrame,
    group_column: str,
    metric_column: str,
    output_path_base: Path,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Create a box-and-point style plot for a metric by group.

    Parameters
    ----------
    data_frame:
        Input table.
    group_column:
        Grouping column.
    metric_column:
        Numeric metric column.
    output_path_base:
        Output path without suffix.
    title:
        Optional plot title.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    groups = [group for group, _ in data_frame.groupby(group_column, dropna=False)]
    values = [
        pd.to_numeric(data_frame.loc[data_frame[group_column] == group, metric_column], errors="coerce").dropna().to_numpy()
        for group in groups
    ]
    plt.figure(figsize=(max(7, len(groups) * 0.45), 5))
    plt.boxplot(x=values, labels=[str(group) for group in groups], showfliers=False)
    for index, value_array in enumerate(values, start=1):
        jitter = np.linspace(-0.08, 0.08, num=max(len(value_array), 1))[: len(value_array)]
        plt.scatter(x=np.repeat(index, len(value_array)) + jitter, y=value_array, alpha=0.75, s=20)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel=metric_column)
    plt.xlabel(xlabel=group_column)
    plt.title(label=title or f"{metric_column} by {group_column}")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def write_interactive_embedding_html(
    *,
    embedding: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    x_column: str,
    y_column: str,
    colour_column: Optional[str],
    output_path: Path,
    title: str = "Interactive Cell Painting embedding",
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Write an interactive Plotly embedding if Plotly is installed.

    Parameters
    ----------
    embedding:
        Embedding table.
    metadata:
        Optional metadata aligned to embedding rows.
    x_column:
        X-axis embedding column.
    y_column:
        Y-axis embedding column.
    colour_column:
        Optional colour column.
    output_path:
        Output HTML path.
    title:
        Plot title.
    logger:
        Optional logger.

    Returns
    -------
    pathlib.Path or None
        Written path, or None when Plotly is unavailable.
    """
    try:
        import plotly.express as px  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger is not None:
            logger.warning("Plotly unavailable; skipping interactive embedding: %s", exc)
        return None
    plot_table = embedding.reset_index(drop=True).copy()
    if metadata is not None:
        plot_table = pd.concat([metadata.reset_index(drop=True), plot_table], axis=1)
    figure = px.scatter(
        data_frame=plot_table,
        x=x_column,
        y=y_column,
        color=colour_column if colour_column in plot_table.columns else None,
        hover_data=plot_table.columns.tolist(),
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(file=str(output_path), include_plotlyjs="cdn")
    if logger is not None:
        logger.info("Wrote interactive plot: %s", output_path)
    return output_path
