"""Generic visualisation utilities for CPATK.

This module provides reusable plotting workflows for processed Cell Painting
profiles and CLIPn/PCA latent spaces. It deliberately avoids assuming any
project-specific metadata names beyond a small set of common aliases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from cpatk.io import read_table, write_table
from cpatk.plotting import save_current_figure

METADATA_EXACT = {
    "cpd_id",
    "compound",
    "compound_id",
    "treatment",
    "Metadata_Compound",
    "Metadata_MOA",
    "known_moa",
    "moa",
    "cpd_type",
    "Library",
    "Dataset",
    "Sample",
    "Plate_Metadata",
    "Well_Metadata",
    "Metadata_Plate",
    "Metadata_Well",
    "ImageNumber",
    "ObjectNumber",
    "TableNumber",
    "Number_Object_Number",
    "Metadata_Batch",
    "batch",
    "dose",
    "concentration",
    "Metadata_Dose",
}

METADATA_PREFIXES = (
    "Metadata_",
    "FileName_",
    "PathName_",
    "URL_",
    "MD5Digest_",
    "ExecutionTime_",
    "Group_",
    "Image_FileName_",
    "Image_PathName_",
)


def is_metadata_like(column: str) -> bool:
    """Return whether a column should be excluded from feature matrices.

    Parameters
    ----------
    column:
        Candidate column name.

    Returns
    -------
    bool
        True when the column looks like metadata, provenance or technical ID.
    """
    name = str(column)
    lower = name.lower()
    if name in METADATA_EXACT or lower in {x.lower() for x in METADATA_EXACT}:
        return True
    if any(name.startswith(prefix) for prefix in METADATA_PREFIXES):
        return True
    if lower.endswith("_metadata") or lower.startswith("metadata"):
        return True
    return False


def select_feature_columns(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    latent_prefix: Optional[str] = None,
    digit_named_latents: bool = False,
) -> list[str]:
    """Select numeric feature columns for embeddings and heatmaps.

    Parameters
    ----------
    data_frame:
        Input table.
    metadata_columns:
        Additional user-specified metadata columns to exclude.
    latent_prefix:
        Optional prefix for latent-space columns.
    digit_named_latents:
        If True, only use columns named like ``0``, ``1`` and so on.

    Returns
    -------
    list[str]
        Feature column names in input order.
    """
    user_metadata = set(metadata_columns or [])
    numeric_cols = data_frame.select_dtypes(include=[np.number]).columns.tolist()
    features = []
    for column in numeric_cols:
        name = str(column)
        if column in user_metadata or is_metadata_like(name):
            continue
        if latent_prefix is not None and not name.startswith(latent_prefix):
            continue
        if digit_named_latents and not name.isdigit():
            continue
        features.append(column)
    return features


def aggregate_profiles(
    *,
    data_frame: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    metadata_columns: Optional[Sequence[str]] = None,
    method: str = "median",
) -> pd.DataFrame:
    """Aggregate replicate rows to one profile per identifier.

    Parameters
    ----------
    data_frame:
        Input table.
    id_column:
        Identifier to aggregate by.
    feature_columns:
        Numeric features to aggregate.
    metadata_columns:
        Metadata columns to preserve using the first observed value.
    method:
        ``median`` or ``mean``.

    Returns
    -------
    pandas.DataFrame
        Aggregated table with metadata first, then features.
    """
    if id_column not in data_frame.columns:
        raise KeyError(f"Identifier column not found: {id_column}")
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'.")

    feature_columns = list(feature_columns)
    if len(feature_columns) == 0:
        raise ValueError("No feature columns supplied for aggregation.")

    metadata_keep = [id_column]
    for column in metadata_columns or []:
        if column in data_frame.columns and column not in metadata_keep:
            metadata_keep.append(column)
    for column in data_frame.columns:
        if column in feature_columns or column in metadata_keep:
            continue
        if is_metadata_like(str(column)):
            metadata_keep.append(column)

    grouped = data_frame.groupby(id_column, dropna=False, sort=False)
    if method == "median":
        feature_block = grouped[feature_columns].median(numeric_only=True)
    else:
        feature_block = grouped[feature_columns].mean(numeric_only=True)

    metadata_block = grouped[metadata_keep].first()
    metadata_block.index = feature_block.index
    out = pd.concat([metadata_block, feature_block], axis=1).reset_index(drop=True)
    return out


def l2_norm_summary(*, features: pd.DataFrame) -> pd.DataFrame:
    """Summarise row-wise L2 norms for a feature matrix.

    Parameters
    ----------
    features:
        Numeric feature matrix.

    Returns
    -------
    pandas.DataFrame
        One-row summary of L2 norms.
    """
    values = features.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    return pd.DataFrame(
        [
            {
                "n_vectors": int(norms.size),
                "min_norm": float(np.nanmin(norms)) if norms.size else np.nan,
                "max_norm": float(np.nanmax(norms)) if norms.size else np.nan,
                "mean_norm": float(np.nanmean(norms)) if norms.size else np.nan,
                "median_norm": float(np.nanmedian(norms)) if norms.size else np.nan,
                "std_norm": float(np.nanstd(norms)) if norms.size else np.nan,
                "n_zero_norm": int(np.sum(norms == 0.0)),
            }
        ]
    )


def plot_l2_norm_histogram(
    *,
    features: pd.DataFrame,
    output_path_base: Path,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot a histogram of row-wise L2 norms.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    output_path_base:
        Path without suffix.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    norms = np.linalg.norm(features.to_numpy(dtype=float), axis=1)
    plt.figure(figsize=(7, 4.5))
    plt.hist(norms, bins=min(50, max(10, int(np.sqrt(max(norms.size, 1))))))
    plt.axvline(1.0, linestyle="--", linewidth=1.0, label="Norm = 1")
    plt.xlabel("L2 norm")
    plt.ylabel("Number of profiles")
    plt.title("Latent/profile vector L2 norms")
    plt.legend(loc="best")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def calculate_pca(
    *,
    features: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate PCA coordinates and variance summary.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    n_components:
        Number of principal components.
    random_state:
        Kept for API consistency; PCA itself is deterministic here.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Coordinates and component variance table.
    """
    _ = random_state
    n_components = min(int(n_components), features.shape[1], features.shape[0])
    if n_components < 1:
        raise ValueError("PCA requires at least one component.")
    model = PCA(n_components=n_components)
    coords = model.fit_transform(features.to_numpy(dtype=float))
    coord_df = pd.DataFrame(
        coords,
        index=features.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    variance = pd.DataFrame(
        {
            "component": coord_df.columns,
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
    )
    return coord_df, variance


def calculate_umap(
    *,
    features: pd.DataFrame,
    n_neighbours: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 0,
) -> pd.DataFrame:
    """Calculate UMAP coordinates when umap-learn is installed.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    n_neighbours:
        UMAP nearest-neighbour parameter.
    min_dist:
        UMAP minimum distance.
    metric:
        UMAP distance metric.
    random_state:
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Two-column UMAP coordinate table.
    """
    try:
        import umap  # type: ignore
    except Exception as exc:
        raise ImportError("umap-learn is not installed; UMAP cannot be run.") from exc

    n_rows = features.shape[0]
    if n_rows < 3:
        raise ValueError("UMAP requires at least three rows.")
    n_neighbours = max(2, min(int(n_neighbours), n_rows - 1))
    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbours,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    coords = model.fit_transform(features.to_numpy(dtype=float))
    return pd.DataFrame(coords, index=features.index, columns=["UMAP1", "UMAP2"])


def calculate_phate(
    *,
    features: pd.DataFrame,
    knn: int = 15,
    random_state: int = 0,
) -> pd.DataFrame:
    """Calculate PHATE coordinates when phate is installed.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    knn:
        PHATE neighbourhood parameter.
    random_state:
        Random seed where supported.

    Returns
    -------
    pandas.DataFrame
        Two-column PHATE coordinate table.
    """
    try:
        import phate  # type: ignore
    except Exception as exc:
        raise ImportError("phate is not installed; PHATE cannot be run.") from exc

    n_rows = features.shape[0]
    if n_rows < 3:
        raise ValueError("PHATE requires at least three rows.")
    model = phate.PHATE(knn=max(2, min(int(knn), n_rows - 1)), random_state=random_state)
    coords = model.fit_transform(features.to_numpy(dtype=float))
    return pd.DataFrame(coords, index=features.index, columns=["PHATE1", "PHATE2"])


def plot_coordinates(
    *,
    coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
    x_column: str,
    y_column: str,
    colour_column: Optional[str],
    label_column: Optional[str],
    output_path_base: Path,
    title: str,
    max_labels: int = 0,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot two-dimensional coordinates with optional labels.

    Parameters
    ----------
    coordinates:
        Coordinate table.
    metadata:
        Metadata aligned to coordinates.
    x_column, y_column:
        Coordinate columns to plot.
    colour_column:
        Optional colour grouping column.
    label_column:
        Optional label column.
    output_path_base:
        Path without suffix.
    title:
        Plot title.
    max_labels:
        Maximum labels to draw; zero disables labels.
    logger:
        Optional logger.

    Returns
    -------
    list[pathlib.Path]
        Written plot paths.
    """
    table = pd.concat(
        [metadata.reset_index(drop=True), coordinates.reset_index(drop=True)],
        axis=1,
    )
    plt.figure(figsize=(8, 6))
    if colour_column and colour_column in table.columns:
        groups = table[colour_column].astype(str).fillna("NA")
        for group, block in table.groupby(groups, dropna=False):
            plt.scatter(block[x_column], block[y_column], s=32, alpha=0.78, label=str(group))
        plt.legend(loc="best", fontsize=8)
    else:
        plt.scatter(table[x_column], table[y_column], s=32, alpha=0.78)
    if label_column and label_column in table.columns and max_labels > 0:
        for _, row in table.head(max_labels).iterrows():
            plt.text(row[x_column], row[y_column], str(row[label_column]), fontsize=7)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def write_interactive_coordinates(
    *,
    coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
    x_column: str,
    y_column: str,
    colour_column: Optional[str],
    output_path: Path,
    title: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Write interactive Plotly coordinate plot when Plotly is installed."""
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        if logger is not None:
            logger.warning("Plotly is not installed; skipping interactive plot: %s", output_path)
        return None
    table = pd.concat(
        [metadata.reset_index(drop=True), coordinates.reset_index(drop=True)],
        axis=1,
    )
    hover_cols = [c for c in metadata.columns[:20] if c in table.columns]
    fig = px.scatter(
        table,
        x=x_column,
        y=y_column,
        color=colour_column if colour_column in table.columns else None,
        hover_data=hover_cols,
        title=title,
        template="simple_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    if logger is not None:
        logger.info("Wrote interactive plot: %s", output_path)
    return output_path


def zscore_matrix(*, matrix: pd.DataFrame, mode: str = "feature") -> pd.DataFrame:
    """Z-score a matrix by feature/column or by row.

    Parameters
    ----------
    matrix:
        Input numeric matrix.
    mode:
        ``none``, ``feature`` or ``row``.

    Returns
    -------
    pandas.DataFrame
        Z-scored matrix.
    """
    mode = mode.lower()
    out = matrix.astype(float).copy()
    if mode == "none":
        return out
    if mode == "feature":
        centre = out.mean(axis=0)
        scale = out.std(axis=0, ddof=0).replace(0.0, 1.0)
        return (out - centre) / scale
    if mode == "row":
        centre = out.mean(axis=1)
        scale = out.std(axis=1, ddof=0).replace(0.0, 1.0)
        return out.subtract(centre, axis=0).divide(scale, axis=0)
    raise ValueError("mode must be one of: none, feature, row")


def clustered_order(
    *,
    matrix: pd.DataFrame,
    axis: int,
    distance: str = "cosine",
    linkage_method: str = "average",
) -> tuple[list[int], Optional[np.ndarray]]:
    """Return hierarchical clustering order for rows or columns."""
    data = matrix.to_numpy(dtype=float) if axis == 0 else matrix.to_numpy(dtype=float).T
    if data.shape[0] < 2:
        return list(range(data.shape[0])), None
    if linkage_method == "ward" and distance != "euclidean":
        linkage_method = "average"
    condensed = pdist(data, metric=distance)
    if not np.isfinite(condensed).all():
        finite = condensed[np.isfinite(condensed)]
        replacement = float(np.max(finite)) if finite.size else 1.0
        condensed = np.nan_to_num(condensed, nan=replacement, posinf=replacement)
    z_matrix = linkage(condensed, method=linkage_method)
    return leaves_list(z_matrix).tolist(), z_matrix


def draw_clustered_heatmap(
    *,
    matrix: pd.DataFrame,
    output_path_base: Path,
    title: str,
    cluster_rows: bool = True,
    cluster_columns: bool = True,
    distance: str = "cosine",
    linkage_method: str = "average",
    zscore: str = "feature",
    max_rows: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    """Create a clustered heatmap and return ordered matrix/order tables.

    Parameters
    ----------
    matrix:
        Matrix with profiles in rows and features in columns.
    output_path_base:
        Output path without suffix.
    title:
        Plot title.
    cluster_rows, cluster_columns:
        Whether to hierarchically cluster rows/columns.
    distance:
        Distance metric for clustering.
    linkage_method:
        Linkage method.
    zscore:
        Z-score mode: ``feature``, ``row`` or ``none``.
    max_rows:
        Optional cap on number of plotted rows after ordering.
    logger:
        Optional logger.

    Returns
    -------
    tuple
        Ordered matrix, row-order table, column-order table, written plots.
    """
    work = zscore_matrix(matrix=matrix, mode=zscore)
    row_order, _ = clustered_order(matrix=work, axis=0, distance=distance, linkage_method=linkage_method) if cluster_rows else (list(range(work.shape[0])), None)
    col_order, _ = clustered_order(matrix=work, axis=1, distance=distance, linkage_method=linkage_method) if cluster_columns else (list(range(work.shape[1])), None)
    ordered = work.iloc[row_order, col_order]
    if max_rows is not None and ordered.shape[0] > max_rows:
        ordered = ordered.iloc[:max_rows, :]
    row_df = pd.DataFrame({"row_rank": np.arange(1, len(ordered.index) + 1), "row_id": ordered.index.astype(str)})
    col_df = pd.DataFrame({"column_rank": np.arange(1, len(ordered.columns) + 1), "feature": ordered.columns.astype(str)})

    fig_w = max(7.0, min(24.0, 0.16 * ordered.shape[1] + 4.0))
    fig_h = max(6.0, min(30.0, 0.18 * ordered.shape[0] + 3.0))
    plt.figure(figsize=(fig_w, fig_h))
    image = plt.imshow(ordered.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    plt.colorbar(image, label=f"value ({zscore})")
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Profiles")
    if ordered.shape[1] <= 80:
        plt.xticks(ticks=np.arange(ordered.shape[1]), labels=ordered.columns.astype(str), rotation=90, fontsize=7)
    else:
        plt.xticks([])
    if ordered.shape[0] <= 120:
        plt.yticks(ticks=np.arange(ordered.shape[0]), labels=ordered.index.astype(str), fontsize=7)
    else:
        plt.yticks([])
    written = save_current_figure(output_path_base=output_path_base, logger=logger)
    return ordered, row_df, col_df, written


def build_knn_topology(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    id_column: Optional[str],
    colour_column: Optional[str],
    n_neighbours: int = 10,
    metric: str = "cosine",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a k-nearest-neighbour topology graph.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    metadata:
        Metadata aligned to features.
    id_column:
        Optional identifier column for node labels.
    colour_column:
        Optional metadata column for node colour value.
    n_neighbours:
        Number of neighbours per node.
    metric:
        Distance metric.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Node table, edge table and PCA coordinates for node layout.
    """
    n_rows = features.shape[0]
    if n_rows < 2:
        raise ValueError("At least two rows are required for kNN topology.")
    k = max(1, min(int(n_neighbours), n_rows - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(features.to_numpy(dtype=float))
    distances, indices = nbrs.kneighbors(features.to_numpy(dtype=float))

    if id_column and id_column in metadata.columns:
        ids = metadata[id_column].astype(str).tolist()
    else:
        ids = [str(x) for x in features.index]

    nodes = pd.DataFrame(
        {
            "node_index": np.arange(n_rows),
            "node_id": ids,
            "colour_value": metadata[colour_column].astype(str).tolist()
            if colour_column and colour_column in metadata.columns
            else ["all"] * n_rows,
            "degree_requested": k,
        }
    )
    edge_rows = []
    seen = set()
    for i in range(n_rows):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            a, b = sorted((int(i), int(j)))
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            edge_rows.append(
                {
                    "source_index": a,
                    "target_index": b,
                    "source_id": ids[a],
                    "target_id": ids[b],
                    "distance": float(dist),
                }
            )
    edges = pd.DataFrame(edge_rows)
    coords, _ = calculate_pca(features=features, n_components=2)
    coords.columns = ["x", "y"]
    coords.insert(0, "node_id", ids)
    return nodes, edges, coords


def plot_knn_topology(
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    coordinates: pd.DataFrame,
    output_path_base: Path,
    title: str = "kNN topology graph",
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot a kNN topology graph using supplied two-dimensional coordinates."""
    coord_lookup = coordinates.set_index("node_id")[["x", "y"]]
    plt.figure(figsize=(8, 6))
    for _, edge in edges.iterrows():
        source = str(edge["source_id"])
        target = str(edge["target_id"])
        if source not in coord_lookup.index or target not in coord_lookup.index:
            continue
        xs = [coord_lookup.loc[source, "x"], coord_lookup.loc[target, "x"]]
        ys = [coord_lookup.loc[source, "y"], coord_lookup.loc[target, "y"]]
        plt.plot(xs, ys, linewidth=0.35, alpha=0.35)
    merged = nodes.merge(coordinates, on="node_id", how="left")
    for value, block in merged.groupby("colour_value", dropna=False):
        plt.scatter(block["x"], block["y"], s=40, alpha=0.85, label=str(value))
    if merged["colour_value"].nunique(dropna=False) <= 20:
        plt.legend(loc="best", fontsize=8)
    plt.xlabel("PCA layout 1")
    plt.ylabel("PCA layout 2")
    plt.title(title)
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def write_interactive_topology(
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    coordinates: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Write an interactive topology graph with Plotly when available."""
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        if logger is not None:
            logger.warning("Plotly is not installed; skipping topology HTML.")
        return None
    coord = coordinates.set_index("node_id")[["x", "y"]]
    edge_x: list[Optional[float]] = []
    edge_y: list[Optional[float]] = []
    for _, edge in edges.iterrows():
        source = str(edge["source_id"])
        target = str(edge["target_id"])
        if source not in coord.index or target not in coord.index:
            continue
        edge_x.extend([float(coord.loc[source, "x"]), float(coord.loc[target, "x"]), None])
        edge_y.extend([float(coord.loc[source, "y"]), float(coord.loc[target, "y"]), None])
    merged = nodes.merge(coordinates, on="node_id", how="left")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 0.5},
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged["x"],
            y=merged["y"],
            mode="markers",
            text=merged["node_id"],
            marker={"size": 8},
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(template="simple_white", title="Interactive kNN topology graph")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    if logger is not None:
        logger.info("Wrote interactive topology graph: %s", output_path)
    return output_path


def run_visualisation_workflow(
    *,
    input_table: Path,
    output_dir: Path,
    metadata_columns: Optional[Sequence[str]] = None,
    id_column: Optional[str] = None,
    colour_columns: Optional[Sequence[str]] = None,
    latent_prefix: Optional[str] = None,
    digit_named_latents: bool = False,
    aggregate_by_id: bool = False,
    aggregate_method: str = "median",
    make_pca: bool = True,
    make_umap: bool = True,
    make_phate: bool = False,
    make_heatmap: bool = True,
    make_topology: bool = True,
    interactive: bool = True,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Path]:
    """Run a generic CPATK visualisation workflow."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(path=input_table, logger=logger)
    feature_cols = select_feature_columns(
        data_frame=df,
        metadata_columns=metadata_columns,
        latent_prefix=latent_prefix,
        digit_named_latents=digit_named_latents,
    )
    if not feature_cols:
        raise ValueError("No numeric feature columns detected for visualisation.")
    metadata_cols = [c for c in df.columns if c not in feature_cols]
    work = df.copy()
    if aggregate_by_id and id_column:
        work = aggregate_profiles(
            data_frame=work,
            id_column=id_column,
            feature_columns=feature_cols,
            metadata_columns=metadata_cols,
            method=aggregate_method,
        )
        feature_cols = [c for c in feature_cols if c in work.columns]
        metadata_cols = [c for c in work.columns if c not in feature_cols]
    metadata = work[metadata_cols].copy()
    features = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.median(numeric_only=True))
    features = features.fillna(0.0)
    if id_column and id_column in metadata.columns:
        features.index = metadata[id_column].astype(str).tolist()
    else:
        features.index = work.index.astype(str)

    outputs: dict[str, Path] = {}
    outputs["visualisation_feature_columns"] = write_table(
        data_frame=pd.DataFrame({"feature": feature_cols}),
        path=output_dir / "visualisation_feature_columns.tsv",
        logger=logger,
    )
    norm_summary = l2_norm_summary(features=features)
    outputs["latent_norm_summary"] = write_table(
        data_frame=norm_summary,
        path=output_dir / "latent_norm_summary.tsv",
        logger=logger,
    )
    norm_plots = plot_l2_norm_histogram(
        features=features,
        output_path_base=plots_dir / "latent_norm_histogram",
        logger=logger,
    )
    for index, path in enumerate(norm_plots, start=1):
        outputs[f"latent_norm_histogram_{index}"] = path

    colour_columns = list(colour_columns or [])
    colour_for_default = colour_columns[0] if colour_columns else None

    if make_pca:
        pca_coords, pca_var = calculate_pca(features=features, n_components=2)
        outputs["pca_coordinates"] = write_table(
            data_frame=pca_coords.reset_index(names="profile_id"),
            path=output_dir / "pca_coordinates.tsv",
            logger=logger,
        )
        outputs["pca_variance"] = write_table(
            data_frame=pca_var,
            path=output_dir / "pca_variance.tsv",
            logger=logger,
        )
        for colour in colour_columns or [None]:
            suffix = str(colour or "uncoloured")
            pca_plot_paths = plot_coordinates(
                coordinates=pca_coords,
                metadata=metadata,
                x_column="PC1",
                y_column="PC2",
                colour_column=colour,
                label_column=id_column,
                output_path_base=plots_dir / f"pca_{suffix}",
                title=f"PCA coloured by {suffix}",
                logger=logger,
            )
            for index, path in enumerate(pca_plot_paths, start=1):
                outputs[f"pca_{suffix}_plot_{index}"] = path
            if interactive:
                interactive_path = write_interactive_coordinates(
                    coordinates=pca_coords,
                    metadata=metadata,
                    x_column="PC1",
                    y_column="PC2",
                    colour_column=colour,
                    output_path=plots_dir / f"pca_{suffix}.html",
                    title=f"Interactive PCA coloured by {suffix}",
                    logger=logger,
                )
                if interactive_path is not None:
                    outputs[f"pca_{suffix}_interactive"] = interactive_path

    if make_umap:
        try:
            umap_coords = calculate_umap(features=features)
            outputs["umap_coordinates"] = write_table(
                data_frame=umap_coords.reset_index(names="profile_id"),
                path=output_dir / "umap_coordinates.tsv",
                logger=logger,
            )
            for colour in colour_columns or [None]:
                suffix = str(colour or "uncoloured")
                umap_plot_paths = plot_coordinates(
                    coordinates=umap_coords,
                    metadata=metadata,
                    x_column="UMAP1",
                    y_column="UMAP2",
                    colour_column=colour,
                    label_column=id_column,
                    output_path_base=plots_dir / f"umap_{suffix}",
                    title=f"UMAP coloured by {suffix}",
                    logger=logger,
                )
                for index, path in enumerate(umap_plot_paths, start=1):
                    outputs[f"umap_{suffix}_plot_{index}"] = path
                if interactive:
                    interactive_path = write_interactive_coordinates(
                        coordinates=umap_coords,
                        metadata=metadata,
                        x_column="UMAP1",
                        y_column="UMAP2",
                        colour_column=colour,
                        output_path=plots_dir / f"umap_{suffix}.html",
                        title=f"Interactive UMAP coloured by {suffix}",
                        logger=logger,
                    )
                    if interactive_path is not None:
                        outputs[f"umap_{suffix}_interactive"] = interactive_path
        except Exception as exc:
            if logger is not None:
                logger.warning("UMAP output skipped: %s", exc)

    if make_phate:
        try:
            phate_coords = calculate_phate(features=features)
            outputs["phate_coordinates"] = write_table(
                data_frame=phate_coords.reset_index(names="profile_id"),
                path=output_dir / "phate_coordinates.tsv",
                logger=logger,
            )
            phate_plot_paths = plot_coordinates(
                coordinates=phate_coords,
                metadata=metadata,
                x_column="PHATE1",
                y_column="PHATE2",
                colour_column=colour_for_default,
                label_column=id_column,
                output_path_base=plots_dir / "phate",
                title="PHATE projection",
                logger=logger,
            )
            for index, path in enumerate(phate_plot_paths, start=1):
                outputs[f"phate_plot_{index}"] = path
        except Exception as exc:
            if logger is not None:
                logger.warning("PHATE output skipped: %s", exc)

    if make_heatmap:
        ordered, row_order, col_order, _ = draw_clustered_heatmap(
            matrix=features,
            output_path_base=plots_dir / "profile_feature_heatmap",
            title="Profile feature heatmap",
            max_rows=120,
            logger=logger,
        )
        outputs["heatmap_matrix"] = write_table(
            data_frame=ordered.reset_index(names="profile_id"),
            path=output_dir / "heatmap_matrix.tsv",
            logger=logger,
        )
        outputs["heatmap_row_order"] = write_table(
            data_frame=row_order,
            path=output_dir / "heatmap_row_order.tsv",
            logger=logger,
        )
        outputs["heatmap_column_order"] = write_table(
            data_frame=col_order,
            path=output_dir / "heatmap_column_order.tsv",
            logger=logger,
        )
        heatmap_plots = sorted(plots_dir.glob("profile_feature_heatmap.*"))
        for index, path in enumerate(heatmap_plots, start=1):
            outputs[f"profile_feature_heatmap_{index}"] = path

    if make_topology:
        nodes, edges, coords = build_knn_topology(
            features=features,
            metadata=metadata,
            id_column=id_column,
            colour_column=colour_for_default,
        )
        outputs["topology_nodes"] = write_table(data_frame=nodes, path=output_dir / "topology_nodes.tsv", logger=logger)
        outputs["topology_edges"] = write_table(data_frame=edges, path=output_dir / "topology_edges.tsv", logger=logger)
        outputs["topology_coordinates"] = write_table(data_frame=coords, path=output_dir / "topology_coordinates.tsv", logger=logger)
        topology_plots = plot_knn_topology(
            nodes=nodes,
            edges=edges,
            coordinates=coords,
            output_path_base=plots_dir / "topology_graph",
            logger=logger,
        )
        for index, path in enumerate(topology_plots, start=1):
            outputs[f"topology_graph_{index}"] = path
        if interactive:
            interactive_path = write_interactive_topology(
                nodes=nodes,
                edges=edges,
                coordinates=coords,
                output_path=plots_dir / "topology_graph.html",
                logger=logger,
            )
            if interactive_path is not None:
                outputs["topology_graph_interactive"] = interactive_path
    return outputs
