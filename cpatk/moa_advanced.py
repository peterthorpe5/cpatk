"""Advanced mechanism-of-action scoring and pseudo-anchor utilities.

The functions in this module are deliberately generic. They operate on any
numeric embedding or feature matrix with an identifier column, including CLIPn
latent outputs and preprocessed CellProfiler profiles.  The module implements
pseudo-anchor creation, consensus K-means stability, centroid/sub-centroid MOA
scoring, CSLS scoring, anchor-permutation tests and compact plotting helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score

LOGGER = logging.getLogger(__name__)


def simple_kmeans_labels(
    *,
    values: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster rows with a small deterministic NumPy K-means implementation.

    This avoids relying on optional threaded K-means backends inside test and
    HPC environments while keeping behaviour deterministic and auditable. Rows
    are treated as L2-normalised embeddings, so cosine assignment is used.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        raise ValueError("values must be two-dimensional.")
    n_rows = x.shape[0]
    n_clusters = int(n_clusters)
    if n_clusters < 1 or n_clusters > n_rows:
        raise ValueError("n_clusters must be between 1 and the number of rows.")
    rng = np.random.default_rng(seed=random_state)
    initial = rng.choice(n_rows, size=n_clusters, replace=False)
    centres = x[initial, :].copy()
    centres = l2_normalise_matrix(values=centres)
    labels = np.zeros(n_rows, dtype=int)
    for _ in range(max_iter):
        old_labels = labels.copy()
        scores = x @ centres.T
        labels = np.argmax(scores, axis=1).astype(int)
        new_centres = centres.copy()
        for cluster in range(n_clusters):
            mask = labels == cluster
            if not np.any(mask):
                new_centres[cluster, :] = x[int(rng.integers(0, n_rows)), :]
            else:
                new_centres[cluster, :] = np.mean(x[mask, :], axis=0)
        new_centres = l2_normalise_matrix(values=new_centres)
        shift = float(np.max(np.abs(new_centres - centres)))
        centres = new_centres
        if np.array_equal(labels, old_labels) or shift <= tol:
            break
    return labels, centres


def l2_normalise_matrix(*, values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a row-wise L2-normalised copy of a numeric matrix."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return x / norms


def _mode_or_first(series: pd.Series) -> object:
    """Return the modal non-null value from a Series, or NA if unavailable."""
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    modes = clean.mode(dropna=True)
    if modes.empty:
        return clean.iloc[0]
    return modes.iloc[0]


def aggregate_profiles(
    *,
    table: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    metadata_columns: Optional[Sequence[str]] = None,
    method: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate replicate rows to one vector per identifier.

    Parameters
    ----------
    table:
        Input table containing identifiers, metadata and numeric features.
    id_column:
        Column used to define replicate groups, usually ``cpd_id``.
    feature_columns:
        Numeric columns to aggregate.
    metadata_columns:
        Optional metadata columns to carry forward as modal values.
    method:
        ``median`` or ``mean``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Aggregated feature table and an aggregation summary.
    """
    if id_column not in table.columns:
        raise KeyError(f"Identifier column is missing: {id_column}")
    if not feature_columns:
        raise ValueError("At least one feature column is required for aggregation.")
    method = method.lower()
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'.")

    metadata_columns = [
        column for column in (metadata_columns or [])
        if column in table.columns and column != id_column
    ]
    work = table[[id_column, *metadata_columns, *feature_columns]].copy()
    work[id_column] = work[id_column].astype(str)
    grouped = work.groupby(id_column, sort=False, dropna=False)
    if method == "median":
        features = grouped[list(feature_columns)].median(numeric_only=True)
    else:
        features = grouped[list(feature_columns)].mean(numeric_only=True)

    metadata_frames = []
    for column in metadata_columns:
        metadata_frames.append(grouped[column].apply(_mode_or_first).rename(column))
    if metadata_frames:
        metadata = pd.concat(metadata_frames, axis=1)
        out = pd.concat([metadata, features], axis=1).reset_index()
    else:
        out = features.reset_index()

    summary = grouped.size().reset_index(name="n_replicate_rows")
    summary["aggregation_method"] = method
    return out, summary


def prepare_embedding_matrix(
    *,
    table: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    metadata_columns: Optional[Sequence[str]] = None,
    aggregate_method: str = "median",
    normalise: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    """Build an aggregated embedding table and aligned numeric matrix."""
    aggregated, summary = aggregate_profiles(
        table=table,
        id_column=id_column,
        feature_columns=feature_columns,
        metadata_columns=metadata_columns,
        method=aggregate_method,
    )
    feature_columns = [column for column in feature_columns if column in aggregated.columns]
    x = aggregated.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        medians = x.median(axis=0, skipna=True)
        x = x.fillna(medians)
        x = x.fillna(0.0)
    x_values = x.to_numpy(dtype=float)
    if normalise:
        x_values = l2_normalise_matrix(values=x_values)
        aggregated.loc[:, feature_columns] = x_values
    return aggregated, x_values, feature_columns, summary


def choose_k_by_silhouette(
    *,
    values: np.ndarray,
    k_values: Sequence[int],
    random_state: int = 0,
) -> tuple[int, pd.DataFrame]:
    """Choose a K-means cluster number using cosine silhouette score."""
    x = np.asarray(values, dtype=float)
    n = x.shape[0]
    rows = []
    for k in sorted(set(int(k) for k in k_values)):
        if k < 2 or k >= n:
            rows.append({"k": k, "silhouette": np.nan, "status": "skipped_invalid_k"})
            continue
        try:
            labels, _ = simple_kmeans_labels(values=x, n_clusters=k, random_state=random_state)
            score = silhouette_score(x, labels, metric="cosine")
            rows.append({"k": k, "silhouette": float(score), "status": "tested"})
        except Exception as exc:  # pragma: no cover - defensive branch
            rows.append({"k": k, "silhouette": np.nan, "status": f"failed:{exc}"})
    table = pd.DataFrame(rows)
    tested = table.loc[table["status"].eq("tested") & table["silhouette"].notna(), :]
    if tested.empty:
        fallback = max(2, min(int(np.sqrt(max(n, 2))), n - 1))
        return fallback, table
    best = int(tested.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return best, table


def bootstrap_kmeans_stability(
    *,
    values: np.ndarray,
    k_values: Sequence[int],
    n_bootstraps: int = 50,
    subsample_fraction: float = 0.8,
    random_state: int = 0,
) -> tuple[int, pd.DataFrame]:
    """Evaluate K-means stability using bootstrap ARI and silhouette metrics.

    A final K-means model is fitted on the full matrix for each candidate k.
    Bootstrap models are fitted to subsampled rows, and unobserved rows are
    assigned to their nearest bootstrap centroid by cosine similarity.  Stability
    is the mean adjusted Rand index between the full-data labels and the
    bootstrap-extended labels.  This is more defensible than relying on a single
    visually pleasing UMAP.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        raise ValueError("values must be two-dimensional.")
    n = x.shape[0]
    if n < 3:
        raise ValueError("At least three rows are needed for bootstrap stability.")
    if not (0.0 < float(subsample_fraction) <= 1.0):
        raise ValueError("subsample_fraction must be in (0, 1].")
    n_bootstraps = max(1, int(n_bootstraps))
    rng = np.random.default_rng(seed=random_state)
    rows = []
    for k in sorted(set(int(k) for k in k_values)):
        if k < 2 or k >= n:
            rows.append(
                {
                    "k": k,
                    "mean_bootstrap_ari": np.nan,
                    "sd_bootstrap_ari": np.nan,
                    "mean_bootstrap_silhouette": np.nan,
                    "full_silhouette": np.nan,
                    "n_successful_bootstraps": 0,
                    "status": "skipped_invalid_k",
                }
            )
            continue
        full_labels, _ = simple_kmeans_labels(values=x, n_clusters=k, random_state=random_state)
        try:
            full_silhouette = silhouette_score(x, full_labels, metric="cosine")
        except Exception:
            full_silhouette = np.nan
        ari_values = []
        sil_values = []
        sample_size = max(k + 1, int(round(float(subsample_fraction) * n)))
        sample_size = min(sample_size, n)
        for _ in range(n_bootstraps):
            idx = np.sort(rng.choice(n, size=sample_size, replace=False))
            x_sub = x[idx, :]
            try:
                labels_sub, centres = simple_kmeans_labels(
                    values=x_sub,
                    n_clusters=k,
                    random_state=int(rng.integers(0, 1_000_000_000)),
                )
                scores = x @ centres.T
                labels_full = np.argmax(scores, axis=1)
                labels_full[idx] = labels_sub
                ari_values.append(adjusted_rand_score(full_labels, labels_full))
                sil_values.append(silhouette_score(x, labels_full, metric="cosine"))
            except Exception:
                continue
        rows.append(
            {
                "k": k,
                "mean_bootstrap_ari": float(np.mean(ari_values)) if ari_values else np.nan,
                "sd_bootstrap_ari": float(np.std(ari_values, ddof=1)) if len(ari_values) > 1 else 0.0,
                "mean_bootstrap_silhouette": float(np.mean(sil_values)) if sil_values else np.nan,
                "full_silhouette": float(full_silhouette),
                "n_successful_bootstraps": int(len(ari_values)),
                "status": "tested" if ari_values else "failed",
            }
        )
    table = pd.DataFrame(rows)
    tested = table.loc[table["status"].eq("tested"), :].copy()
    if tested.empty:
        best_k, _ = choose_k_by_silhouette(values=x, k_values=k_values, random_state=random_state)
        return best_k, table
    tested["rank_score"] = tested["mean_bootstrap_ari"].fillna(-np.inf) + (
        0.1 * tested["mean_bootstrap_silhouette"].fillna(-np.inf)
    )
    best = int(tested.sort_values(["rank_score", "k"], ascending=[False, True]).iloc[0]["k"])
    return best, table


def make_pseudo_anchors(
    *,
    table: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    metadata_columns: Optional[Sequence[str]] = None,
    aggregate_method: str = "median",
    n_clusters: int = 30,
    auto_k: bool = False,
    k_values: Optional[Sequence[int]] = None,
    bootstrap: bool = False,
    n_bootstraps: int = 50,
    subsample_fraction: float = 0.8,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate pseudo-MOA anchors from numeric embeddings/features."""
    emb, x, _, aggregation_summary = prepare_embedding_matrix(
        table=table,
        id_column=id_column,
        feature_columns=feature_columns,
        metadata_columns=metadata_columns,
        aggregate_method=aggregate_method,
        normalise=True,
    )
    n = x.shape[0]
    if n < 2:
        raise ValueError("At least two unique identifiers are needed for pseudo-anchors.")
    if k_values is None:
        root = max(2, int(np.sqrt(n)))
        k_values = sorted(set([2, root, min(n - 1, max(root + 1, int(1.5 * root))), min(n - 1, int(n_clusters))]))
    if auto_k or bootstrap:
        if bootstrap:
            selected_k, k_selection = bootstrap_kmeans_stability(
                values=x,
                k_values=k_values,
                n_bootstraps=n_bootstraps,
                subsample_fraction=subsample_fraction,
                random_state=random_state,
            )
        else:
            selected_k, k_selection = choose_k_by_silhouette(
                values=x,
                k_values=k_values,
                random_state=random_state,
            )
    else:
        selected_k = max(2, min(int(n_clusters), n - 1))
        k_selection = pd.DataFrame(
            [{"k": selected_k, "status": "fixed", "silhouette": np.nan}]
        )
    labels, _ = simple_kmeans_labels(
        values=x,
        n_clusters=selected_k,
        random_state=random_state,
    )
    unique_labels = sorted(np.unique(labels))
    mapping = {label: f"PseudoMOA_{idx + 1:04d}" for idx, label in enumerate(unique_labels)}
    pseudo = [mapping[int(label)] for label in labels]
    anchors = pd.DataFrame({id_column: emb[id_column].astype(str), "pseudo_moa": pseudo})
    clusters = pd.DataFrame(
        {
            id_column: emb[id_column].astype(str),
            "cluster": labels.astype(int),
            "pseudo_moa": pseudo,
        }
    )
    summary = clusters.groupby("pseudo_moa", dropna=False).size().reset_index(name="n_compounds")
    summary["selected_k"] = selected_k
    summary["bootstrap_used"] = bool(bootstrap)
    summary["aggregate_method"] = aggregate_method
    return anchors, summary, clusters, k_selection.merge(aggregation_summary.head(0), how="cross") if False else k_selection


def build_moa_centroids(
    *,
    embedding_table: pd.DataFrame,
    anchors: pd.DataFrame,
    id_column: str,
    moa_column: str,
    feature_columns: Sequence[str],
    centroid_method: str = "median",
    n_subcentroids: int = 1,
    shrinkage: float = 0.0,
    adaptive_shrinkage: bool = False,
    adaptive_c: float = 0.5,
    adaptive_max: float = 0.3,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one or more normalised centroids for each anchor MOA."""
    if id_column not in embedding_table.columns:
        raise KeyError(f"Identifier column is missing from embeddings: {id_column}")
    if id_column not in anchors.columns or moa_column not in anchors.columns:
        raise KeyError(f"Anchors must contain {id_column!r} and {moa_column!r}.")
    if centroid_method not in {"median", "mean"}:
        raise ValueError("centroid_method must be 'median' or 'mean'.")
    n_subcentroids = max(1, int(n_subcentroids))
    shrinkage = max(0.0, min(1.0, float(shrinkage)))
    rng = np.random.default_rng(seed=random_state)
    emb = embedding_table.copy()
    emb[id_column] = emb[id_column].astype(str)
    anchors = anchors[[id_column, moa_column]].copy()
    anchors[id_column] = anchors[id_column].astype(str)
    anchors[moa_column] = anchors[moa_column].astype(str)
    emb_map = {cid: pos for pos, cid in enumerate(emb[id_column].astype(str).tolist())}
    x = emb.loc[:, feature_columns].to_numpy(dtype=float)
    global_mean = x.mean(axis=0)
    global_norm = np.linalg.norm(global_mean)
    if global_norm > 0:
        global_mean = global_mean / global_norm
    rows = []
    summary_rows = []

    def effective_alpha(n_members: int) -> float:
        alpha = shrinkage
        if adaptive_shrinkage and n_members > 0:
            alpha += min(float(adaptive_max), float(adaptive_c) / float(n_members))
        return max(0.0, min(1.0, alpha))

    for moa, sub in anchors.groupby(moa_column, sort=False, dropna=False):
        ids = sub[id_column].astype(str).tolist()
        indices = [emb_map[item] for item in ids if item in emb_map]
        if not indices:
            summary_rows.append(
                {
                    "moa": str(moa),
                    "centroid_index": pd.NA,
                    "n_members": 0,
                    "status": "no_matching_anchors",
                }
            )
            continue
        x_moa = x[indices, :]
        n_members = x_moa.shape[0]
        if n_subcentroids <= 1 or n_members < 2:
            cluster_labels = np.zeros(n_members, dtype=int)
        else:
            k = min(n_subcentroids, n_members)
            try:
                cluster_labels, _ = simple_kmeans_labels(
                    values=x_moa,
                    n_clusters=k,
                    random_state=int(rng.integers(0, 1_000_000_000)),
                )
            except Exception:
                cluster_labels = np.zeros(n_members, dtype=int)
        for sub_index in sorted(np.unique(cluster_labels)):
            sub_matrix = x_moa[cluster_labels == sub_index, :]
            if sub_matrix.size == 0:
                continue
            if centroid_method == "median":
                vec = np.median(sub_matrix, axis=0)
            else:
                vec = np.mean(sub_matrix, axis=0)
            alpha = effective_alpha(sub_matrix.shape[0])
            if alpha > 0:
                vec = ((1.0 - alpha) * vec) + (alpha * global_mean)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            centroid_id = f"{moa}__centroid_{int(sub_index) + 1:03d}"
            row = {"centroid_id": centroid_id, "moa": str(moa)}
            row.update({column: value for column, value in zip(feature_columns, vec)})
            rows.append(row)
            summary_rows.append(
                {
                    "moa": str(moa),
                    "centroid_id": centroid_id,
                    "centroid_index": int(sub_index),
                    "n_members": int(sub_matrix.shape[0]),
                    "shrinkage": float(alpha),
                    "method": centroid_method if n_subcentroids <= 1 else f"kmeans/{centroid_method}",
                    "status": "ok",
                }
            )
    centroids = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    if centroids.empty:
        raise ValueError("No centroids could be built from the supplied anchors.")
    return centroids, summary


def cosine_scores(*, query: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return cosine-similarity scores for L2-normalised rows."""
    return np.asarray(query, dtype=float) @ np.asarray(centroids, dtype=float).T


def csls_scores(*, query: np.ndarray, centroids: np.ndarray, k: int = 10) -> np.ndarray:
    """Return cross-domain similarity local scaling scores."""
    q = np.asarray(query, dtype=float)
    c = np.asarray(centroids, dtype=float)
    if q.size == 0 or c.size == 0:
        return np.zeros((q.shape[0], c.shape[0]), dtype=float)
    scores = q @ c.T
    kq = max(1, min(int(k), c.shape[0]))
    kc = max(1, min(int(k), q.shape[0]))
    rq = np.partition(scores, kth=scores.shape[1] - kq, axis=1)[:, -kq:].mean(axis=1, keepdims=True)
    rc = np.partition(scores, kth=scores.shape[0] - kc, axis=0)[-kc:, :].mean(axis=0, keepdims=True)
    return (2.0 * scores) - rq - rc


def _softmax(values: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """Return a numerically stable softmax vector."""
    temp = max(float(temperature), 1e-12)
    z = np.asarray(values, dtype=float) / temp
    z = z - np.nanmax(z)
    exp_z = np.exp(z)
    denom = np.nansum(exp_z)
    if not np.isfinite(denom) or denom <= 0.0:
        return np.repeat(1.0 / len(values), len(values))
    return exp_z / denom


def collapse_centroid_scores_to_moa(
    *,
    scores: np.ndarray,
    centroid_table: pd.DataFrame,
    moa_column: str = "moa",
    collapse: str = "max",
) -> tuple[np.ndarray, list[str]]:
    """Collapse centroid scores to MOA-level scores."""
    if moa_column not in centroid_table.columns:
        raise KeyError(f"Centroid table is missing MOA column: {moa_column}")
    collapse = collapse.lower()
    if collapse not in {"max", "mean", "median"}:
        raise ValueError("collapse must be 'max', 'mean' or 'median'.")
    labels = centroid_table[moa_column].astype(str).tolist()
    unique = list(dict.fromkeys(labels))
    out = np.zeros((scores.shape[0], len(unique)), dtype=float)
    for j, label in enumerate(unique):
        idx = [pos for pos, item in enumerate(labels) if item == label]
        sub = scores[:, idx]
        if collapse == "max":
            out[:, j] = np.max(sub, axis=1)
        elif collapse == "mean":
            out[:, j] = np.mean(sub, axis=1)
        else:
            out[:, j] = np.median(sub, axis=1)
    return out, unique


def score_against_moa_centroids(
    *,
    embedding_table: pd.DataFrame,
    centroid_table: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    score_method: str = "cosine",
    csls_k: int = 10,
    top_n: int = 5,
    collapse: str = "max",
    confidence_temperature: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score compounds against MOA centroids and return long/top tables."""
    if id_column not in embedding_table.columns:
        raise KeyError(f"Identifier column is missing: {id_column}")
    if "moa" not in centroid_table.columns:
        raise KeyError("centroid_table must contain a 'moa' column.")
    xq = embedding_table.loc[:, feature_columns].to_numpy(dtype=float)
    xc = centroid_table.loc[:, feature_columns].to_numpy(dtype=float)
    xq = l2_normalise_matrix(values=xq)
    xc = l2_normalise_matrix(values=xc)
    if score_method == "cosine":
        centroid_scores = cosine_scores(query=xq, centroids=xc)
    elif score_method == "csls":
        centroid_scores = csls_scores(query=xq, centroids=xc, k=csls_k)
    else:
        raise ValueError("score_method must be 'cosine' or 'csls'.")
    moa_scores, moa_labels = collapse_centroid_scores_to_moa(
        scores=centroid_scores,
        centroid_table=centroid_table,
        collapse=collapse,
    )
    records = []
    top_n = max(1, min(int(top_n), len(moa_labels)))
    ids = embedding_table[id_column].astype(str).tolist()
    for row_idx, cid in enumerate(ids):
        order = np.argsort(-moa_scores[row_idx, :])
        probs = _softmax(moa_scores[row_idx, :], temperature=confidence_temperature)
        best = order[0]
        second = order[1] if len(order) > 1 else order[0]
        margin = moa_scores[row_idx, best] - moa_scores[row_idx, second]
        for rank, pos in enumerate(order[:top_n], start=1):
            records.append(
                {
                    id_column: cid,
                    "rank": rank,
                    "predicted_moa": moa_labels[pos],
                    "moa_score": float(moa_scores[row_idx, pos]),
                    "softmax_confidence": float(probs[pos]),
                    "top1_score_margin": float(margin),
                    "score_method": score_method,
                    "score_collapse": collapse,
                }
            )
    long_scores = pd.DataFrame.from_records(records)
    top = long_scores.loc[long_scores["rank"].eq(1), :].reset_index(drop=True)
    return long_scores, top


def score_matrix_table(
    *,
    embedding_table: pd.DataFrame,
    centroid_table: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    score_method: str = "cosine",
    csls_k: int = 10,
    collapse: str = "max",
) -> pd.DataFrame:
    """Return a wide compound-by-MOA score matrix."""
    xq = l2_normalise_matrix(values=embedding_table.loc[:, feature_columns].to_numpy(dtype=float))
    xc = l2_normalise_matrix(values=centroid_table.loc[:, feature_columns].to_numpy(dtype=float))
    if score_method == "csls":
        centroid_scores = csls_scores(query=xq, centroids=xc, k=csls_k)
    else:
        centroid_scores = cosine_scores(query=xq, centroids=xc)
    moa_scores, moa_labels = collapse_centroid_scores_to_moa(
        scores=centroid_scores,
        centroid_table=centroid_table,
        collapse=collapse,
    )
    out = pd.DataFrame(moa_scores, columns=moa_labels)
    out.insert(0, id_column, embedding_table[id_column].astype(str).to_numpy())
    return out


def anchor_permutation_test(
    *,
    embedding_table: pd.DataFrame,
    anchors: pd.DataFrame,
    id_column: str,
    moa_column: str,
    feature_columns: Sequence[str],
    centroid_method: str = "median",
    n_subcentroids: int = 1,
    shrinkage: float = 0.0,
    adaptive_shrinkage: bool = False,
    score_method: str = "cosine",
    csls_k: int = 10,
    collapse: str = "max",
    n_permutations: int = 100,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation test by shuffling anchor MOA labels and rebuilding centroids.

    This avoids the common pitfall of merely shuffling labels on an already-built
    score matrix, which does not change the maximum score when there is one
    centroid per class.  Here, the null distribution is created by rebuilding
    random anchor centroids with the same anchor IDs and permuted MOA labels.
    """
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        empty = pd.DataFrame(columns=[id_column, "observed_top_score", "empirical_p_value"])
        return empty, pd.DataFrame()
    rng = np.random.default_rng(seed=random_state)
    observed_centroids, _ = build_moa_centroids(
        embedding_table=embedding_table,
        anchors=anchors,
        id_column=id_column,
        moa_column=moa_column,
        feature_columns=feature_columns,
        centroid_method=centroid_method,
        n_subcentroids=n_subcentroids,
        shrinkage=shrinkage,
        adaptive_shrinkage=adaptive_shrinkage,
        random_state=random_state,
    )
    observed_matrix = score_matrix_table(
        embedding_table=embedding_table,
        centroid_table=observed_centroids,
        id_column=id_column,
        feature_columns=feature_columns,
        score_method=score_method,
        csls_k=csls_k,
        collapse=collapse,
    )
    score_cols = [column for column in observed_matrix.columns if column != id_column]
    observed_top = observed_matrix[score_cols].max(axis=1).to_numpy(dtype=float)
    ids = embedding_table[id_column].astype(str).tolist()
    null_rows = []
    exceed = np.zeros(len(ids), dtype=int)
    anchors_base = anchors[[id_column, moa_column]].copy()
    label_values = anchors_base[moa_column].astype(str).to_numpy()
    for iteration in range(n_permutations):
        permuted = anchors_base.copy()
        shuffled = label_values.copy()
        rng.shuffle(shuffled)
        permuted[moa_column] = shuffled
        try:
            null_centroids, _ = build_moa_centroids(
                embedding_table=embedding_table,
                anchors=permuted,
                id_column=id_column,
                moa_column=moa_column,
                feature_columns=feature_columns,
                centroid_method=centroid_method,
                n_subcentroids=n_subcentroids,
                shrinkage=shrinkage,
                adaptive_shrinkage=adaptive_shrinkage,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            null_matrix = score_matrix_table(
                embedding_table=embedding_table,
                centroid_table=null_centroids,
                id_column=id_column,
                feature_columns=feature_columns,
                score_method=score_method,
                csls_k=csls_k,
                collapse=collapse,
            )
            null_score_cols = [column for column in null_matrix.columns if column != id_column]
            null_top = null_matrix[null_score_cols].max(axis=1).to_numpy(dtype=float)
        except Exception:
            null_top = np.repeat(np.nan, len(ids))
        exceed += np.where(np.isfinite(null_top) & (null_top >= observed_top), 1, 0)
        for cid, score in zip(ids, null_top):
            null_rows.append(
                {
                    "permutation_index": int(iteration),
                    id_column: cid,
                    "null_top_score": float(score) if np.isfinite(score) else np.nan,
                }
            )
    p_values = (exceed + 1.0) / (n_permutations + 1.0)
    summary = pd.DataFrame(
        {
            id_column: ids,
            "observed_top_score": observed_top,
            "empirical_p_value": p_values,
            "n_permutations": int(n_permutations),
            "null_exceedances": exceed.astype(int),
        }
    )
    return summary, pd.DataFrame.from_records(null_rows)


def pairwise_distance_outputs(
    *,
    embedding_table: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    metrics: Sequence[str] = ("cosine", "correlation"),
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Create pairwise distance matrices and nearest-neighbour tables."""
    x = embedding_table.loc[:, feature_columns].to_numpy(dtype=float)
    ids = embedding_table[id_column].astype(str).tolist()
    outputs: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        if metric == "spearman":
            ranks = pd.DataFrame(x).rank(axis=1, method="average").to_numpy(dtype=float)
            dist = pairwise_distances(ranks, metric="correlation")
        else:
            dist = pairwise_distances(x, metric=metric)
        dist = np.where(np.isfinite(dist), dist, 1.0)
        np.fill_diagonal(dist, 0.0)
        matrix = pd.DataFrame(dist, columns=ids)
        matrix.insert(0, id_column, ids)
        outputs[f"pairwise_distance_{metric}"] = matrix
        rows = []
        for i, cid in enumerate(ids):
            d = dist[i, :].copy()
            d[i] = np.inf
            order = np.argsort(d)[: min(int(top_n), len(ids) - 1)]
            for rank, j in enumerate(order, start=1):
                rows.append(
                    {
                        id_column: cid,
                        "rank": rank,
                        "neighbour_id": ids[j],
                        "distance": float(dist[i, j]),
                        "metric": metric,
                    }
                )
        outputs[f"nearest_neighbours_{metric}"] = pd.DataFrame.from_records(rows)
    return outputs


def plot_k_selection(*, table: pd.DataFrame, output_path: Path) -> list[Path]:
    """Plot K-selection diagnostics and return written paths."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_col = "mean_bootstrap_ari" if "mean_bootstrap_ari" in table.columns else "silhouette"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plot_df = table.dropna(subset=["k", y_col]).sort_values("k")
    if not plot_df.empty:
        ax.plot(plot_df["k"], plot_df[y_col], marker="o")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title("Pseudo-anchor K selection")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    svg_path = output_path.with_suffix(".svg")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if not plot_df.empty:
        ax.plot(plot_df["k"], plot_df[y_col], marker="o")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title("Pseudo-anchor K selection")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(svg_path)
    plt.close(fig)
    return [output_path, svg_path]


def plot_prediction_score_distribution(*, predictions: pd.DataFrame, output_path: Path) -> list[Path]:
    """Plot distribution of top MOA scores and confidence margins."""
    import matplotlib.pyplot as plt

    written: list[Path] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for column in ["moa_score", "top1_score_margin", "softmax_confidence"]:
        if column not in predictions.columns:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").dropna()
        path = output_path.with_name(f"{output_path.stem}_{column}{output_path.suffix}")
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(max(len(values), 1))))))
        ax.set_xlabel(column.replace("_", " "))
        ax.set_ylabel("Count")
        ax.set_title(f"MOA prediction {column.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def plot_score_heatmap(
    *,
    score_matrix: pd.DataFrame,
    id_column: str,
    output_path: Path,
    max_items: int = 250,
) -> list[Path]:
    """Plot a clustered compound-by-MOA score heatmap."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if score_matrix.shape[0] == 0 or score_matrix.shape[1] <= 1:
        return []
    plot_df = score_matrix.copy()
    if plot_df.shape[0] > max_items:
        plot_df = plot_df.head(max_items)
    ids = plot_df[id_column].astype(str).tolist()
    values = plot_df.drop(columns=[id_column]).to_numpy(dtype=float)
    col_labels = [c for c in plot_df.columns if c != id_column]
    row_order = np.arange(values.shape[0])
    if values.shape[0] >= 3:
        try:
            dist = pdist(values, metric="cosine")
            row_order = leaves_list(linkage(dist, method="average"))
        except Exception:
            row_order = np.arange(values.shape[0])
    values = values[row_order, :]
    ids = [ids[i] for i in row_order]
    width = max(8.0, min(18.0, 0.28 * len(col_labels) + 4.0))
    height = max(6.0, min(22.0, 0.08 * len(ids) + 4.0))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    if len(ids) <= 80:
        ax.set_yticks(np.arange(len(ids)))
        ax.set_yticklabels(ids, fontsize=6)
    else:
        ax.set_yticks([])
    ax.set_title("Compound-by-MOA score heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    svg_path = output_path.with_suffix(".svg")
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    if len(ids) <= 80:
        ax.set_yticks(np.arange(len(ids)))
        ax.set_yticklabels(ids, fontsize=6)
    else:
        ax.set_yticks([])
    ax.set_title("Compound-by-MOA score heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Score")
    fig.tight_layout()
    fig.savefig(svg_path)
    plt.close(fig)
    return [output_path, svg_path]


def project_embedding_with_centroids(
    *,
    embedding_table: pd.DataFrame,
    centroid_table: pd.DataFrame,
    id_column: str,
    feature_columns: Sequence[str],
    method: str = "pca",
    random_state: int = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project compounds and centroids into the same 2D space."""
    x = embedding_table.loc[:, feature_columns].to_numpy(dtype=float)
    c = centroid_table.loc[:, feature_columns].to_numpy(dtype=float)
    x = l2_normalise_matrix(values=x)
    c = l2_normalise_matrix(values=c)
    stacked = np.vstack([x, c])
    split = x.shape[0]
    method = method.lower()
    if method == "pca":
        model = PCA(n_components=2, random_state=random_state)
        coords = model.fit_transform(stacked)
    elif method == "umap":
        try:
            import umap  # type: ignore
        except Exception as exc:
            raise ImportError("umap-learn is required for UMAP projection.") from exc
        n_neighbors = min(max(2, int(umap_n_neighbors)), stacked.shape[0] - 1)
        model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=float(umap_min_dist),
            random_state=random_state,
        )
        coords = model.fit_transform(stacked)
    else:
        raise ValueError("method must be 'pca' or 'umap'.")
    comp = pd.DataFrame(
        {
            id_column: embedding_table[id_column].astype(str).to_numpy(),
            f"{method.upper()}1": coords[:split, 0],
            f"{method.upper()}2": coords[:split, 1],
            "point_type": "compound",
        }
    )
    cent = pd.DataFrame(
        {
            "centroid_id": centroid_table["centroid_id"].astype(str).to_numpy(),
            "moa": centroid_table["moa"].astype(str).to_numpy(),
            f"{method.upper()}1": coords[split:, 0],
            f"{method.upper()}2": coords[split:, 1],
            "point_type": "centroid",
        }
    )
    return comp, cent, pd.concat([comp, cent], ignore_index=True, sort=False)


def plot_projection(
    *,
    compound_coords: pd.DataFrame,
    centroid_coords: pd.DataFrame,
    id_column: str,
    method: str,
    output_path: Path,
    colour_series: Optional[pd.Series] = None,
    interactive: bool = False,
) -> list[Path]:
    """Plot compound and centroid coordinates, with optional Plotly HTML."""
    import matplotlib.pyplot as plt

    method_upper = method.upper()
    x_col = f"{method_upper}1"
    y_col = f"{method_upper}2"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if colour_series is not None:
        labels = colour_series.astype(str).reset_index(drop=True)
        for label in sorted(labels.unique()):
            mask = labels.eq(label).to_numpy()
            ax.scatter(
                compound_coords.loc[mask, x_col],
                compound_coords.loc[mask, y_col],
                s=30,
                alpha=0.8,
                label=label,
            )
        ax.legend(fontsize=7, frameon=True, loc="best")
    else:
        ax.scatter(compound_coords[x_col], compound_coords[y_col], s=30, alpha=0.8)
    ax.scatter(
        centroid_coords[x_col],
        centroid_coords[y_col],
        s=140,
        marker="X",
        edgecolors="black",
        linewidths=0.6,
        label="Centroid",
    )
    for _, row in centroid_coords.iterrows():
        ax.text(row[x_col], row[y_col], str(row["moa"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"MOA centroid map ({method_upper})")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    written.append(output_path)
    svg_path = output_path.with_suffix(".svg")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(compound_coords[x_col], compound_coords[y_col], s=30, alpha=0.8)
    ax.scatter(centroid_coords[x_col], centroid_coords[y_col], s=140, marker="X", edgecolors="black", linewidths=0.6)
    for _, row in centroid_coords.iterrows():
        ax.text(row[x_col], row[y_col], str(row["moa"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"MOA centroid map ({method_upper})")
    fig.tight_layout()
    fig.savefig(svg_path)
    plt.close(fig)
    written.append(svg_path)
    if interactive:
        try:
            import plotly.graph_objects as go  # type: ignore
            fig_i = go.Figure()
            fig_i.add_trace(
                go.Scattergl(
                    x=compound_coords[x_col],
                    y=compound_coords[y_col],
                    mode="markers",
                    text=compound_coords[id_column],
                    name="Compounds",
                    marker={"size": 6, "opacity": 0.75},
                )
            )
            fig_i.add_trace(
                go.Scatter(
                    x=centroid_coords[x_col],
                    y=centroid_coords[y_col],
                    mode="markers+text",
                    text=centroid_coords["moa"],
                    name="Centroids",
                    marker={"size": 14, "symbol": "x"},
                    textposition="top center",
                )
            )
            fig_i.update_layout(
                title=f"MOA centroid map ({method_upper})",
                template="plotly_white",
                xaxis_title=x_col,
                yaxis_title=y_col,
            )
            html_path = output_path.with_suffix(".html")
            fig_i.write_html(str(html_path))
            written.append(html_path)
        except Exception:
            LOGGER.warning("Interactive MOA projection plot failed.", exc_info=True)
    return written
