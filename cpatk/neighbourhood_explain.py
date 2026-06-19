"""Neighbourhood-level feature explanation for Cell Painting profiles.

This module complements the global supervised feature-attribution workflow in
``cpatk.explainability``.  It is designed for the common Cell Painting question:
"why is this query compound close to, or different from, its nearest
neighbours?"

The functions are deliberately table-based and auditable.  They can be used
from the command line, in notebooks, or in tests without relying on global
state.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cpatk.plotting import save_current_figure, set_publication_theme


METADATA_LIKE_RE = re.compile(
    r"""(?ix)
    (^metadata($|_)|_metadata$|^filename_|^pathname_|^url_|^md5digest_|^executiontime_|
     ^channel_|^height_|^width_|^imageid$|^imagename$|^imageseries$|
     (^|_)imagenumber$|^objectnumber$|^number_object_number$|^tablenumber$|
     ^parent_|^children_|^count_|^group_)
    """
)


def _normalise_id(value: object) -> str:
    """Return a stable uppercase identifier string for matching."""
    return str(value).strip().upper()


def _is_metadata_like(column: object) -> bool:
    """Return True if ``column`` looks like metadata/provenance rather than a feature."""
    return bool(METADATA_LIKE_RE.search(str(column).strip().lower()))


def benjamini_hochberg(*, p_values: Sequence[float]) -> np.ndarray:
    """Calculate Benjamini-Hochberg adjusted q-values.

    Parameters
    ----------
    p_values:
        Raw p-values.

    Returns
    -------
    numpy.ndarray
        Adjusted q-values in the original order.
    """
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values
    values = np.where(np.isfinite(values), values, 1.0)
    order = np.argsort(values)
    ranked = values[order]
    n_values = float(values.size)
    q_ranked = ranked * n_values / np.arange(1, values.size + 1, dtype=float)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    output = np.empty_like(q_ranked)
    output[order] = q_ranked
    return output


def parse_query_ids(*, query_ids: Optional[Sequence[str]] = None, query_file: Optional[str | Path] = None) -> list[str]:
    """Parse query identifiers from inline values and/or a text/CSV/TSV file."""
    parsed: list[str] = []
    if query_file is not None:
        query_path = Path(query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        raw_lines = [line.strip() for line in query_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()]
        raw_lines = [line for line in raw_lines if line and not line.startswith("#")]
        # Plain one-ID-per-line files are common; do not let pandas treat the
        # first identifier as a header and silently drop it.
        if raw_lines and all(("," not in line and "\t" not in line) for line in raw_lines):
            parsed.extend([line for line in raw_lines if line.lower() not in {"cpd_id", "query_id", "id"}])
        else:
            try:
                table = pd.read_csv(query_path, sep=None, engine="python")
                column = "cpd_id" if "cpd_id" in table.columns else ("query_id" if "query_id" in table.columns else table.columns[0])
                parsed.extend(table[column].dropna().astype(str).tolist())
            except Exception:
                parsed.extend(raw_lines)
    for item in query_ids or []:
        parsed.extend([part.strip() for part in str(item).split(",") if part.strip()])
    clean = []
    seen = set()
    for item in parsed:
        if not item or str(item).startswith("#"):
            continue
        key = _normalise_id(item)
        if key not in seen:
            seen.add(key)
            clean.append(str(item).strip())
    return clean


def select_neighbour_ids(
    *,
    neighbour_table: pd.DataFrame,
    query_id: str,
    n_neighbours: int = 5,
    query_column_candidates: Sequence[str] = ("cpd_id", "query_id"),
    neighbour_column: str = "neighbour_id",
    distance_column: str = "distance",
) -> list[str]:
    """Select top-neighbour identifiers for a query from a nearest-neighbour table."""
    query_column = next((column for column in query_column_candidates if column in neighbour_table.columns), None)
    if query_column is None:
        raise ValueError(f"Neighbour table needs one of {list(query_column_candidates)}")
    if neighbour_column not in neighbour_table.columns:
        raise ValueError(f"Neighbour table is missing required column: {neighbour_column}")
    hits = neighbour_table.loc[
        neighbour_table[query_column].map(_normalise_id) == _normalise_id(query_id)
    ].copy()
    if distance_column in hits.columns:
        hits[distance_column] = pd.to_numeric(hits[distance_column], errors="coerce")
        hits = hits.sort_values(distance_column, ascending=True)
    ids = []
    seen = {_normalise_id(query_id)}
    for value in hits[neighbour_column].dropna().astype(str):
        key = _normalise_id(value)
        if key not in seen:
            seen.add(key)
            ids.append(value)
        if len(ids) >= int(n_neighbours):
            break
    return ids


def clean_numeric_feature_matrix(
    *,
    features: pd.DataFrame,
    explicit_feature_columns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return numeric feature matrix and an audit table for selected/excluded columns.

    This mirrors the package-wide principle that provenance/QC columns should not
    silently enter biological feature-attribution models.
    """
    if explicit_feature_columns is not None:
        candidate_columns = [column for column in explicit_feature_columns if column in features.columns]
    else:
        candidate_columns = list(features.columns)
    records = []
    selected = []
    for column in candidate_columns:
        series = features[column]
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        metadata_like = _is_metadata_like(column)
        if is_numeric and not metadata_like:
            selected.append(column)
            role = "feature"
            reason = "numeric_non_metadata"
        elif not is_numeric:
            role = "excluded"
            reason = "non_numeric"
        else:
            role = "excluded"
            reason = "metadata_or_qc_like_name"
        records.append(
            {
                "column": column,
                "role": role,
                "reason": reason,
                "is_numeric": is_numeric,
                "metadata_like_name": metadata_like,
            }
        )
    if not selected:
        raise ValueError("No usable numeric feature columns remain for neighbourhood explanation.")
    matrix = features.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    # Local explanation should not run a new complex preprocessing pipeline; it
    # uses median filling only as a final safety net after CPATK preprocessing.
    medians = matrix.median(axis=0, skipna=True)
    matrix = matrix.fillna(medians).fillna(0.0).astype(float)
    return matrix, pd.DataFrame.from_records(records)


def make_binary_neighbourhood_dataset(
    *,
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    id_column: str,
    query_id: str,
    neighbour_ids: Sequence[str],
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build a binary query-vs-neighbour feature matrix for SHAP/explanation."""
    if id_column not in metadata.columns:
        raise ValueError(f"Metadata table is missing id_column: {id_column}")
    ids = metadata[id_column].map(_normalise_id)
    query_key = _normalise_id(query_id)
    neighbour_keys = {_normalise_id(value) for value in neighbour_ids}
    mask = (ids == query_key) | ids.isin(neighbour_keys)
    if not mask.any():
        raise ValueError(f"No profiles found for query {query_id} or its neighbours.")
    subset_features = features.loc[mask.to_numpy(), :].reset_index(drop=True)
    subset_metadata = metadata.loc[mask.to_numpy(), :].reset_index(drop=True)
    clean_x, column_audit = clean_numeric_feature_matrix(
        features=subset_features,
        explicit_feature_columns=feature_columns,
    )
    y = (subset_metadata[id_column].map(_normalise_id) == query_key).astype(int)
    if y.nunique() < 2:
        raise ValueError("Neighbourhood explanation requires both query and neighbour profiles.")
    return clean_x, y, subset_metadata, column_audit


def calculate_two_sample_feature_statistics(
    *,
    features: pd.DataFrame,
    mask_a: Sequence[bool],
    mask_b: Sequence[bool],
    comparison_name: str,
    test: str = "mw",
) -> pd.DataFrame:
    """Compare two profile groups feature-by-feature.

    The output includes robust effect-size metrics and FDR correction across all
    tested features for the comparison.
    """
    if test not in {"mw", "ks"}:
        raise ValueError("test must be 'mw' or 'ks'")
    clean_x, _ = clean_numeric_feature_matrix(features=features)
    mask_a_array = np.asarray(mask_a, dtype=bool)
    mask_b_array = np.asarray(mask_b, dtype=bool)
    if mask_a_array.shape[0] != clean_x.shape[0] or mask_b_array.shape[0] != clean_x.shape[0]:
        raise ValueError("Masks must match the number of rows in features.")
    records = []
    for feature in clean_x.columns:
        a = clean_x.loc[mask_a_array, feature].dropna().to_numpy(dtype=float)
        b = clean_x.loc[mask_b_array, feature].dropna().to_numpy(dtype=float)
        if a.size == 0 or b.size == 0:
            p_value = 1.0
            median_a = np.nan
            median_b = np.nan
            emd = np.nan
        else:
            median_a = float(np.median(a))
            median_b = float(np.median(b))
            emd = float(wasserstein_distance(a, b))
            if test == "mw":
                try:
                    p_value = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except Exception:
                    p_value = 1.0
            else:
                try:
                    p_value = float(ks_2samp(a, b, alternative="two-sided", mode="auto").pvalue)
                except Exception:
                    p_value = 1.0
        diff = median_a - median_b if np.isfinite(median_a) and np.isfinite(median_b) else np.nan
        records.append(
            {
                "comparison": comparison_name,
                "feature": feature,
                "n_group_a": int(a.size),
                "n_group_b": int(b.size),
                "median_group_a": median_a,
                "median_group_b": median_b,
                "median_difference_group_a_minus_group_b": diff,
                "absolute_median_difference": abs(diff) if np.isfinite(diff) else np.nan,
                "wasserstein_distance": emd,
                "test": test,
                "p_value": p_value,
            }
        )
    table = pd.DataFrame.from_records(records)
    table["q_value"] = benjamini_hochberg(p_values=table["p_value"].to_numpy(dtype=float))
    table = table.sort_values(["q_value", "absolute_median_difference"], ascending=[True, False])
    return table


def calculate_query_background_statistics(
    *,
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    id_column: str,
    query_id: str,
    background_column: Optional[str] = None,
    background_values: Optional[Sequence[str]] = None,
    test: str = "mw",
) -> pd.DataFrame:
    """Compare one query compound against a background/control set."""
    if id_column not in metadata.columns:
        raise ValueError(f"Metadata table is missing id_column: {id_column}")
    query_mask = metadata[id_column].map(_normalise_id) == _normalise_id(query_id)
    if background_column and background_column in metadata.columns and background_values:
        accepted = {_normalise_id(value) for value in background_values}
        background_mask = metadata[background_column].map(_normalise_id).isin(accepted)
        comparison = f"{query_id}_vs_{background_column}_{'_'.join(background_values)}"
    else:
        background_mask = ~query_mask
        comparison = f"{query_id}_vs_non_query_background"
    if int(query_mask.sum()) == 0:
        raise ValueError(f"No profiles found for query_id: {query_id}")
    if int(background_mask.sum()) == 0:
        raise ValueError("No background profiles available for comparison.")
    return calculate_two_sample_feature_statistics(
        features=features,
        mask_a=query_mask.to_numpy(),
        mask_b=background_mask.to_numpy(),
        comparison_name=comparison,
        test=test,
    )


def _choose_neighbourhood_model(*, n_rows: int, random_state: int, n_jobs: int = 1):
    """Choose a simple classifier for query-vs-neighbour SHAP."""
    if n_rows < 30:
        return LogisticRegression(max_iter=2000, random_state=random_state, class_weight="balanced")
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=max(1, int(n_jobs)),
    )


def _normalise_shap_values(*, shap_raw: object, n_features: int, prefer_class_index: int = 1) -> np.ndarray:
    """Normalise SHAP outputs across SHAP/model versions to samples x features."""
    if isinstance(shap_raw, list):
        if len(shap_raw) > prefer_class_index:
            array = np.asarray(shap_raw[prefer_class_index])
        else:
            array = np.asarray(shap_raw[0])
    else:
        array = np.asarray(getattr(shap_raw, "values", shap_raw))
    array = np.squeeze(array)
    if array.ndim == 3:
        # samples x features x classes
        if array.shape[1] == n_features:
            class_index = min(prefer_class_index, array.shape[2] - 1)
            array = array[:, :, class_index]
        # samples x classes x features
        elif array.shape[2] == n_features:
            class_index = min(prefer_class_index, array.shape[1] - 1)
            array = array[:, class_index, :]
    if array.ndim != 2 or array.shape[1] != n_features:
        raise ValueError(f"SHAP array shape {array.shape} is incompatible with {n_features} features.")
    return array.astype(float)


def calculate_neighbourhood_shap(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    query_id: str,
    n_top_features: int = 20,
    max_background: int = 200,
    max_explain: int = 200,
    random_state: int = 42,
    n_jobs: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Mapping[str, pd.DataFrame | np.ndarray | object]:
    """Explain a query-vs-neighbour classifier using SHAP.

    Returns a mapping with ``top_features``, ``low_contribution_features``,
    ``sample_feature_shap_values``, ``status`` and the fitted ``model``.  SHAP is
    optional; failures are captured in the status table rather than crashing a
    complete CPATK run.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "top_features": pd.DataFrame(columns=["feature", "mean_absolute_shap"]),
            "low_contribution_features": pd.DataFrame(columns=["feature", "mean_absolute_shap"]),
            "sample_feature_shap_values": pd.DataFrame(),
            "status": pd.DataFrame.from_records([{"status": "not_available", "message": str(exc)}]),
            "model": None,
            "shap_array": np.empty((0, 0)),
            "explained_x": pd.DataFrame(),
        }
    clean_x = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean_x = clean_x.fillna(clean_x.median(axis=0, skipna=True)).fillna(0.0).astype(float)
    clean_y = y.astype(int).reset_index(drop=True)
    if clean_y.nunique() < 2 or clean_y.value_counts().min() < 1:
        raise ValueError("Neighbourhood SHAP requires both query and neighbour classes.")
    model = _choose_neighbourhood_model(n_rows=clean_x.shape[0], random_state=random_state, n_jobs=n_jobs)
    model.fit(clean_x, clean_y)
    rng = np.random.default_rng(seed=random_state)
    background_n = min(int(max_background), clean_x.shape[0])
    explain_n = min(int(max_explain), clean_x.shape[0])
    background_indices = np.sort(rng.choice(np.arange(clean_x.shape[0]), size=background_n, replace=False))
    explain_indices = np.sort(rng.choice(np.arange(clean_x.shape[0]), size=explain_n, replace=False))
    background_x = clean_x.iloc[background_indices, :]
    explain_x = clean_x.iloc[explain_indices, :]
    try:
        if isinstance(model, RandomForestClassifier):
            try:
                explainer = shap.TreeExplainer(
                    model,
                    data=background_x,
                    feature_perturbation="interventional",
                    model_output="probability",
                )
                shap_raw = explainer.shap_values(explain_x, check_additivity=False)
            except Exception:
                explainer = shap.TreeExplainer(model)
                shap_raw = explainer.shap_values(explain_x, check_additivity=False)
        else:
            # LinearExplainer is much faster and more predictable than the
            # generic SHAP dispatcher for small logistic-regression
            # neighbourhood models.
            try:
                explainer = shap.LinearExplainer(model, background_x)
                shap_raw = explainer.shap_values(explain_x)
            except Exception:
                explainer = shap.Explainer(model, background_x, algorithm="linear")
                shap_raw = explainer(explain_x)
        shap_array = _normalise_shap_values(shap_raw=shap_raw, n_features=clean_x.shape[1])
        mean_abs = np.mean(np.abs(shap_array), axis=0)
        ranking = np.argsort(mean_abs)[::-1]
        low_ranking = np.argsort(mean_abs)
        top_n = min(max(1, int(n_top_features)), clean_x.shape[1])
        top = pd.DataFrame(
            {
                "feature": clean_x.columns[ranking[:top_n]],
                "mean_absolute_shap": mean_abs[ranking[:top_n]],
                "importance_rank": np.arange(1, top_n + 1),
            }
        )
        low = pd.DataFrame(
            {
                "feature": clean_x.columns[low_ranking[:top_n]],
                "mean_absolute_shap": mean_abs[low_ranking[:top_n]],
                "low_contribution_rank": np.arange(1, top_n + 1),
            }
        )
        shap_value_table = pd.DataFrame(shap_array, columns=clean_x.columns)
        shap_value_table.insert(0, "explained_row_position", explain_indices)
        status = pd.DataFrame.from_records(
            [
                {
                    "status": "ok",
                    "query_id": query_id,
                    "model_type": type(model).__name__,
                    "n_profiles": int(clean_x.shape[0]),
                    "n_features": int(clean_x.shape[1]),
                    "n_background": int(background_x.shape[0]),
                    "n_explained": int(explain_x.shape[0]),
                    "message": "Neighbourhood SHAP completed successfully.",
                }
            ]
        )
        if logger is not None:
            logger.info("Neighbourhood SHAP completed for %s", query_id)
        return {
            "top_features": top,
            "low_contribution_features": low,
            "sample_feature_shap_values": shap_value_table,
            "status": status,
            "model": model,
            "shap_array": shap_array,
            "explained_x": explain_x,
        }
    except Exception as exc:
        if logger is not None:
            logger.warning("Neighbourhood SHAP failed for %s: %s", query_id, exc)
        return {
            "top_features": pd.DataFrame(columns=["feature", "mean_absolute_shap"]),
            "low_contribution_features": pd.DataFrame(columns=["feature", "mean_absolute_shap"]),
            "sample_feature_shap_values": pd.DataFrame(),
            "status": pd.DataFrame.from_records(
                [{"status": "failed", "query_id": query_id, "model_type": type(model).__name__, "message": str(exc)}]
            ),
            "model": model,
            "shap_array": np.empty((0, 0)),
            "explained_x": pd.DataFrame(),
        }


def plot_signed_feature_statistics(
    *,
    stats_table: pd.DataFrame,
    output_path_base: Path,
    top_n: int = 15,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot top positive and negative median feature shifts."""
    required = {"feature", "median_difference_group_a_minus_group_b"}
    if not required.issubset(stats_table.columns) or stats_table.empty:
        return []
    set_publication_theme()
    table = stats_table.copy()
    table["effect"] = pd.to_numeric(table["median_difference_group_a_minus_group_b"], errors="coerce")
    positive = table.sort_values("effect", ascending=False).head(top_n)
    negative = table.sort_values("effect", ascending=True).head(top_n)
    plot_table = pd.concat([negative, positive], ignore_index=True).drop_duplicates("feature")
    plot_table = plot_table.sort_values("effect", ascending=True)
    plt.figure(figsize=(9, max(5, 0.22 * plot_table.shape[0])))
    plt.barh(plot_table["feature"].astype(str), plot_table["effect"].astype(float))
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Median difference: group A minus group B")
    plt.ylabel("Feature")
    plt.title("Top signed feature shifts")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def plot_shap_outputs(
    *,
    shap_array: np.ndarray,
    explained_x: pd.DataFrame,
    top_features: pd.DataFrame,
    output_path_base: Path,
    max_display: int = 20,
    n_dependence: int = 5,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Write SHAP bar, beeswarm, heatmap and dependence plots when SHAP is available."""
    written: list[Path] = []
    if shap_array.size == 0 or explained_x.empty:
        return written
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger is not None:
            logger.warning("SHAP plotting skipped because SHAP is unavailable: %s", exc)
        return written
    output_path_base.parent.mkdir(parents=True, exist_ok=True)
    feature_names = explained_x.columns.tolist()
    explanation = shap.Explanation(values=shap_array, data=explained_x.to_numpy(), feature_names=feature_names)
    try:
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_array, explained_x, plot_type="bar", show=False, max_display=max_display)
        written.extend(save_current_figure(output_path_base=output_path_base.with_name(output_path_base.name + "_summary_bar"), logger=logger))
    except Exception as exc:
        if logger is not None:
            logger.warning("SHAP summary bar plot failed: %s", exc)
    try:
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_array, explained_x, plot_type="dot", show=False, max_display=max_display)
        written.extend(save_current_figure(output_path_base=output_path_base.with_name(output_path_base.name + "_summary_beeswarm"), logger=logger))
    except Exception as exc:
        if logger is not None:
            logger.warning("SHAP beeswarm plot failed: %s", exc)
    try:
        plt.figure(figsize=(9, 6))
        shap.plots.bar(explanation, max_display=max_display, show=False)
        written.extend(save_current_figure(output_path_base=output_path_base.with_name(output_path_base.name + "_bar"), logger=logger))
    except Exception as exc:
        if logger is not None:
            logger.warning("SHAP bar plot failed: %s", exc)
    try:
        plt.figure(figsize=(10, 7))
        shap.plots.heatmap(explanation, max_display=max_display, show=False)
        written.extend(save_current_figure(output_path_base=output_path_base.with_name(output_path_base.name + "_heatmap"), logger=logger))
    except Exception as exc:
        if logger is not None:
            logger.warning("SHAP heatmap failed: %s", exc)
    top_feature_names = top_features.get("feature", pd.Series(dtype=str)).astype(str).head(n_dependence).tolist()
    dependence_dir = output_path_base.parent / f"{output_path_base.name}_dependence_plots"
    dependence_dir.mkdir(parents=True, exist_ok=True)
    for index, feature in enumerate(top_feature_names, start=1):
        if feature not in explained_x.columns:
            continue
        try:
            shap.dependence_plot(feature, shap_array, explained_x, show=False)
            written.extend(
                save_current_figure(
                    output_path_base=dependence_dir / f"{index:02d}_{feature[:80]}",
                    logger=logger,
                )
            )
        except Exception as exc:
            if logger is not None:
                logger.warning("SHAP dependence plot failed for %s: %s", feature, exc)
    return written


def group_importance_by_feature_family(
    *,
    importance_table: pd.DataFrame,
    value_column: str = "mean_absolute_shap",
) -> pd.DataFrame:
    """Summarise local/global importance by broad CellProfiler feature family."""
    if "feature" not in importance_table.columns or value_column not in importance_table.columns:
        return pd.DataFrame(columns=["feature_family", "n_features", "total_importance", "mean_importance"])
    text = importance_table["feature"].astype(str)
    family = np.select(
        [
            text.str.contains("Intensity", case=False, regex=False),
            text.str.contains("Texture", case=False, regex=False),
            text.str.contains("AreaShape", case=False, regex=False),
            text.str.contains("Granularity", case=False, regex=False),
            text.str.contains("RadialDistribution", case=False, regex=False),
            text.str.contains("Correlation", case=False, regex=False),
            text.str.contains("Neighbors", case=False, regex=False),
            text.str.contains("Location", case=False, regex=False),
        ],
        ["Intensity", "Texture", "AreaShape", "Granularity", "RadialDistribution", "Correlation", "Neighbors", "Location"],
        default="Other",
    )
    table = importance_table.copy()
    table["feature_family"] = family
    table[value_column] = pd.to_numeric(table[value_column], errors="coerce")
    return (
        table.groupby("feature_family", dropna=False)[value_column]
        .agg(n_features="count", total_importance="sum", mean_importance="mean")
        .reset_index()
        .sort_values("total_importance", ascending=False)
    )
