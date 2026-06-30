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
    n_jobs: int = 1,
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
    n_jobs:
        Accepted for compatibility with threaded command-line entry points.
        Plot rendering itself is single-threaded. Native numerical libraries may
        still use their configured thread pools outside this function.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    _ = n_jobs
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
    plt.boxplot(x=values, tick_labels=[str(group) for group in groups], showfliers=False)
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


def set_publication_theme(*, font_size: int = 10) -> None:
    """Apply a simple publication-oriented matplotlib theme.

    Parameters
    ----------
    font_size:
        Base font size.
    """
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_pca_variance(
    *,
    explained_variance: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot PCA explained variance.

    Parameters
    ----------
    explained_variance:
        PCA explained variance table.
    output_path_base:
        Output path without suffix.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    set_publication_theme()
    table = explained_variance.copy()
    plt.figure(figsize=(7, 4.5))
    plt.bar(x=table["component"].astype(str), height=table["explained_variance_ratio"].astype(float))
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component")
    plt.title("PCA explained variance")
    plt.xticks(rotation=45, ha="right")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_feature_importance(
    *,
    importance_table: pd.DataFrame,
    value_column: str,
    output_path_base: Path,
    feature_column: str = "feature",
    top_n: int = 30,
    title: str = "Feature importance",
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot top feature-importance scores.

    Parameters
    ----------
    importance_table:
        Importance table.
    value_column:
        Numeric importance column.
    output_path_base:
        Output path without suffix.
    feature_column:
        Feature-name column.
    top_n:
        Number of features to show.
    title:
        Plot title.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    if feature_column not in importance_table.columns or value_column not in importance_table.columns:
        raise ValueError("Importance table is missing required columns.")
    set_publication_theme()
    table = importance_table.copy()
    table[value_column] = pd.to_numeric(table[value_column], errors="coerce")
    table = table.sort_values(value_column, ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, top_n * 0.22)))
    plt.barh(y=table[feature_column].astype(str), width=table[value_column])
    plt.xlabel(value_column)
    plt.ylabel("Feature")
    plt.title(title)
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_confusion_matrix(
    *,
    confusion_table: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot a confusion matrix table produced by CPATK.

    Parameters
    ----------
    confusion_table:
        Confusion table with ``true_class`` plus predicted-class columns.
    output_path_base:
        Output path without suffix.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    if "true_class" not in confusion_table.columns:
        raise ValueError("Confusion table must contain true_class.")
    matrix = confusion_table.set_index("true_class")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return plot_heatmap(
        matrix=matrix,
        output_path_base=output_path_base,
        title="MOA classifier confusion matrix",
        value_label="Count",
        logger=logger,
    )


def write_interactive_heatmap_html(
    *,
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Write an interactive heatmap if Plotly is installed.

    Parameters
    ----------
    matrix:
        Matrix to plot.
    output_path:
        Output HTML path.
    title:
        Plot title.
    logger:
        Optional logger.

    Returns
    -------
    pathlib.Path or None
        Written output path, or ``None`` when Plotly is unavailable.
    """
    try:
        import plotly.express as px  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger is not None:
            logger.warning("Plotly unavailable; skipping interactive heatmap: %s", exc)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = px.imshow(matrix, aspect="auto", title=title)
    figure.write_html(file=str(output_path), include_plotlyjs="cdn")
    if logger is not None:
        logger.info("Wrote interactive heatmap: %s", output_path)
    return output_path


def plot_all_zero_row_summary(
    *,
    all_zero_row_report: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot counts of profiles flagged by the all-zero row filter."""
    if all_zero_row_report.empty or "all_zero_feature_row" not in all_zero_row_report.columns:
        return []
    set_publication_theme()
    labels = all_zero_row_report["all_zero_feature_row"].map({True: "all observed features zero", False: "has non-zero feature evidence"})
    counts = labels.value_counts().reindex(["has non-zero feature evidence", "all observed features zero"]).fillna(0)
    plt.figure(figsize=(7, 4.5))
    plt.bar(x=counts.index.astype(str), height=counts.to_numpy(dtype=float))
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Number of profiles")
    plt.title("All-zero feature-row QC")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_missingness_histogram(
    *,
    qc_table: pd.DataFrame,
    missing_column: str,
    output_path_base: Path,
    title: str,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot a missingness histogram from a QC table."""
    if missing_column not in qc_table.columns:
        raise ValueError(f"Missing column not found in QC table: {missing_column}")
    set_publication_theme()
    values = pd.to_numeric(qc_table[missing_column], errors="coerce").dropna()
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=min(40, max(5, int(np.sqrt(max(len(values), 1))))))
    plt.xlabel(missing_column)
    plt.ylabel("Count")
    plt.title(title)
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_feature_variance_histogram(
    *,
    feature_qc: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot log10 feature variance distribution."""
    if "variance" not in feature_qc.columns:
        raise ValueError("feature_qc must contain a variance column.")
    set_publication_theme()
    values = pd.to_numeric(feature_qc["variance"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = np.log10(values.clip(lower=1e-300))
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=min(40, max(5, int(np.sqrt(max(len(values), 1))))))
    plt.xlabel("log10(feature variance)")
    plt.ylabel("Number of features")
    plt.title("Feature variance distribution")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_feature_qc_status(
    *,
    feature_qc: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot counts of features passing/failing QC."""
    if "feature_qc_pass" not in feature_qc.columns:
        raise ValueError("feature_qc must contain feature_qc_pass.")
    set_publication_theme()
    counts = feature_qc["feature_qc_pass"].map({True: "pass", False: "fail"}).value_counts().reindex(["pass", "fail"]).fillna(0)
    plt.figure(figsize=(5, 4))
    plt.bar(x=counts.index.astype(str), height=counts.to_numpy(dtype=float))
    plt.ylabel("Number of features")
    plt.title("Feature QC retention")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_preprocessing_retention(
    *,
    summary: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot key row and feature retention counts from preprocessing summary."""
    if not {"item", "value"}.issubset(summary.columns):
        raise ValueError("summary must contain item and value columns.")
    wanted = [
        "n_rows_input",
        "n_all_zero_feature_rows_removed",
        "n_rows_passing_qc",
        "n_features_input",
        "n_features_after_qc",
        "n_features_after_correlation_filter",
    ]
    table = summary.loc[summary["item"].isin(wanted), ["item", "value"]].copy()
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    set_publication_theme()
    plt.figure(figsize=(8, 4.5))
    plt.bar(x=table["item"].astype(str), height=table["value"].astype(float))
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Count")
    plt.title("Preprocessing retention summary")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_feature_family_summary(
    *,
    feature_family_summary: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot retained feature counts by broad feature family."""
    if feature_family_summary.empty:
        return []
    if not {"feature_family", "n_features"}.issubset(feature_family_summary.columns):
        raise ValueError("feature_family_summary must contain feature_family and n_features.")
    table = feature_family_summary.sort_values("n_features", ascending=True)
    set_publication_theme()
    plt.figure(figsize=(8, max(4, 0.35 * len(table))))
    plt.barh(y=table["feature_family"].astype(str), width=pd.to_numeric(table["n_features"], errors="coerce"))
    plt.xlabel("Number of retained features")
    plt.ylabel("Feature family")
    plt.title("Retained features by family")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_correlation_filter_summary(
    *,
    correlation_report: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot distribution of correlations removed by the correlation filter."""
    if correlation_report.empty or "correlation" not in correlation_report.columns:
        return []
    set_publication_theme()
    values = pd.to_numeric(correlation_report["correlation"], errors="coerce").dropna()
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=min(30, max(5, int(np.sqrt(max(len(values), 1))))))
    plt.xlabel("Absolute Pearson correlation")
    plt.ylabel("Number of removed features")
    plt.title("Correlation filter: removed-feature correlations")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_column_role_summary(
    *,
    column_role_report: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot counts of metadata, feature and excluded columns."""
    if column_role_report.empty or "role" not in column_role_report.columns:
        return []
    set_publication_theme()
    counts = column_role_report["role"].astype(str).value_counts().sort_values(ascending=True)
    plt.figure(figsize=(8, max(4, 0.35 * len(counts))))
    plt.barh(y=counts.index, width=counts.to_numpy(dtype=float))
    plt.xlabel("Number of columns")
    plt.ylabel("Assigned role")
    plt.title("Column role assignment")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_imputation_missingness_top_features(
    *,
    imputation_report: pd.DataFrame,
    output_path_base: Path,
    top_n: int = 30,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot the features with the most missing values before imputation."""
    required = {"feature", "missing_fraction_before"}
    if imputation_report.empty or not required.issubset(imputation_report.columns):
        return []
    set_publication_theme()
    table = imputation_report.copy()
    table["missing_fraction_before"] = pd.to_numeric(table["missing_fraction_before"], errors="coerce")
    table = table.sort_values("missing_fraction_before", ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.25 * len(table))))
    plt.barh(y=table["feature"].astype(str), width=table["missing_fraction_before"])
    plt.xlabel("Missing fraction before imputation")
    plt.ylabel("Feature")
    plt.title("Top missing features before imputation")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_model_summary(
    *,
    summary: pd.DataFrame,
    metric_column: str,
    output_path_base: Path,
    model_column: str = "model_name",
    title: str = "Model comparison",
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot model-performance summary values."""
    if summary.empty or model_column not in summary.columns or metric_column not in summary.columns:
        return []
    table = summary.copy()
    table[metric_column] = pd.to_numeric(table[metric_column], errors="coerce")
    table = table.sort_values(metric_column, ascending=True)
    set_publication_theme()
    plt.figure(figsize=(8, max(4, 0.35 * len(table))))
    plt.barh(y=table[model_column].astype(str), width=table[metric_column])
    plt.xlabel(metric_column)
    plt.ylabel(model_column)
    plt.title(title)
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_prediction_confidence(
    *,
    predictions: pd.DataFrame,
    confidence_column: str,
    output_path_base: Path,
    title: str = "Prediction confidence distribution",
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot a histogram of prediction confidence scores."""
    if predictions.empty or confidence_column not in predictions.columns:
        return []
    values = pd.to_numeric(predictions[confidence_column], errors="coerce").dropna()
    if values.empty:
        return []
    set_publication_theme()
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))))
    plt.xlabel(confidence_column)
    plt.ylabel("Number of profiles")
    plt.title(title)
    return save_current_figure(output_path_base=output_path_base, logger=logger)
