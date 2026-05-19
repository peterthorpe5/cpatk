"""Dimensionality reduction helpers for CPATK."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_pca(
    *,
    features: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run PCA and return scores plus explained variance.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    n_components:
        Number of PCA components.
    random_state:
        Random seed.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        PCA scores and explained variance table.
    """
    model = PCA(n_components=n_components, random_state=random_state)
    values = model.fit_transform(X=features)
    score_columns = [f"PC{index + 1}" for index in range(values.shape[1])]
    scores = pd.DataFrame(data=values, columns=score_columns, index=features.index)
    explained = pd.DataFrame(
        {
            "component": score_columns,
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
    )
    return scores, explained


def run_umap_or_pca(
    *,
    features: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Run UMAP if available, otherwise fall back to PCA.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    n_components:
        Number of embedding dimensions.
    n_neighbors:
        UMAP neighbourhood size.
    min_dist:
        UMAP minimum distance.
    random_state:
        Random seed.
    logger:
        Optional logger.

    Returns
    -------
    pandas.DataFrame
        Embedding table.
    """
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        values = reducer.fit_transform(X=features)
        prefix = "UMAP"
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        if logger is not None:
            logger.warning("UMAP unavailable, falling back to PCA: %s", exc)
        scores, _ = run_pca(features=features, n_components=n_components, random_state=random_state)
        values = scores.to_numpy()
        prefix = "PCA_fallback"
    columns = [f"{prefix}{index + 1}" for index in range(values.shape[1])]
    return pd.DataFrame(data=values, columns=columns, index=features.index)


def run_tsne(
    *,
    features: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run t-SNE embedding.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    n_components:
        Number of t-SNE dimensions.
    perplexity:
        t-SNE perplexity.
    random_state:
        Random seed.

    Returns
    -------
    pandas.DataFrame
        t-SNE embedding table.
    """
    model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    values = model.fit_transform(X=features)
    return pd.DataFrame(
        data=values,
        columns=[f"TSNE{index + 1}" for index in range(values.shape[1])],
        index=features.index,
    )
