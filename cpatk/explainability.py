"""Feature-attribution utilities including permutation importance and SHAP."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from cpatk.ml import ModelName, build_classifier


def _clean_supervised_inputs(*, features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Remove rows with missing labels and ensure numeric feature data."""
    clean_labels = labels.astype("object")
    mask = clean_labels.notna() & (clean_labels.astype(str).str.len() > 0)
    clean_features = features.loc[mask.to_numpy(), :].apply(pd.to_numeric, errors="coerce")
    clean_features = clean_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    clean_labels = clean_labels.loc[mask].astype(str).reset_index(drop=True)
    clean_features = clean_features.reset_index(drop=True)
    if clean_features.empty:
        raise ValueError("No labelled rows remain after cleaning supervised inputs.")
    if clean_labels.nunique() < 2:
        raise ValueError("At least two classes are required for feature attribution.")
    return clean_features, clean_labels


def calculate_permutation_feature_importance(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: ModelName = "random_forest",
    n_repeats: int = 10,
    test_size: float = 0.3,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate held-out permutation feature importance.

    This is the safest default feature-attribution method because it estimates
    the drop in held-out balanced accuracy after each feature is shuffled.  It
    still explains the fitted model, not direct biological causality.
    """
    clean_features, clean_labels = _clean_supervised_inputs(features=features, labels=labels)
    counts = clean_labels.value_counts()
    stratify = clean_labels if counts.min() >= 2 else None
    train_x, test_x, train_y, test_y = train_test_split(
        clean_features,
        clean_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    model = build_classifier(model_name=model_name, random_state=random_state)
    model.fit(X=train_x, y=train_y)
    predicted = model.predict(X=test_x)
    baseline = balanced_accuracy_score(y_true=test_y, y_pred=predicted)
    importance = permutation_importance(
        estimator=model,
        X=test_x,
        y=test_y,
        n_repeats=max(1, int(n_repeats)),
        random_state=random_state,
        scoring="balanced_accuracy",
    )
    table = pd.DataFrame(
        {
            "feature": clean_features.columns.tolist(),
            "permutation_importance_mean": importance.importances_mean,
            "permutation_importance_sd": importance.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    table["importance_rank"] = np.arange(1, table.shape[0] + 1)
    summary = pd.DataFrame.from_records(
        [
            {
                "model_name": model_name,
                "n_features": int(clean_features.shape[1]),
                "n_train": int(train_x.shape[0]),
                "n_test": int(test_x.shape[0]),
                "n_classes": int(clean_labels.nunique()),
                "balanced_accuracy": float(baseline),
                "n_repeats": int(n_repeats),
                "interpretation": (
                    "Permutation importance is model-based and held-out. It is useful for ranking features that support "
                    "classification, but it does not prove a causal mechanism."
                ),
            }
        ]
    )
    if logger is not None:
        logger.info("Permutation importance calculated for %s features", clean_features.shape[1])
    return table, summary


def calculate_tree_native_importance(*, model: object, feature_names: Sequence[str]) -> pd.DataFrame:
    """Extract native tree-model feature importance when available."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "native_importance"])
    values = getattr(model, "feature_importances_")
    table = pd.DataFrame({"feature": list(feature_names), "native_importance": values})
    table = table.sort_values("native_importance", ascending=False)
    table["importance_rank"] = np.arange(1, table.shape[0] + 1)
    return table


def _sample_indices_stratified(*, labels: pd.Series, max_rows: int, random_state: int) -> np.ndarray:
    """Return a reproducible class-aware subset of row indices."""
    rng = np.random.default_rng(seed=random_state)
    n_rows = len(labels)
    if n_rows <= max_rows:
        return np.arange(n_rows)
    per_class = max(1, max_rows // max(1, labels.nunique()))
    chosen: list[int] = []
    for _, group_index in labels.groupby(labels).groups.items():
        group_index = np.asarray(list(group_index), dtype=int)
        size = min(per_class, len(group_index))
        chosen.extend(rng.choice(group_index, size=size, replace=False).tolist())
    remaining = [idx for idx in range(n_rows) if idx not in set(chosen)]
    if len(chosen) < max_rows and remaining:
        extra = rng.choice(np.asarray(remaining), size=min(max_rows - len(chosen), len(remaining)), replace=False)
        chosen.extend(extra.tolist())
    return np.asarray(sorted(chosen), dtype=int)


def _shap_values_to_tables(
    *,
    shap_values: object,
    feature_names: Sequence[str],
    class_names: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert SHAP output into global and class-level importance tables."""
    values = getattr(shap_values, "values", shap_values)
    array = np.asarray(values)
    global_records = []
    class_records = []
    if array.ndim == 3:
        # Common shapes are samples x features x classes or samples x classes x features.
        if array.shape[1] == len(feature_names):
            sample_feature_class = array
        elif array.shape[2] == len(feature_names):
            sample_feature_class = np.moveaxis(array, 1, 2)
        else:
            raise ValueError(f"Cannot align SHAP array shape {array.shape} to {len(feature_names)} features.")
        mean_abs_by_feature = np.mean(np.abs(sample_feature_class), axis=(0, 2))
        n_classes = sample_feature_class.shape[2]
        class_names = list(class_names or [f"class_{idx}" for idx in range(n_classes)])
        for class_index in range(n_classes):
            class_values = np.mean(np.abs(sample_feature_class[:, :, class_index]), axis=0)
            for feature, value in zip(feature_names, class_values):
                class_records.append(
                    {
                        "class_name": str(class_names[class_index]) if class_index < len(class_names) else f"class_{class_index}",
                        "feature": feature,
                        "mean_absolute_shap": float(value),
                    }
                )
    elif array.ndim == 2:
        if array.shape[1] != len(feature_names):
            raise ValueError(f"Cannot align SHAP array shape {array.shape} to {len(feature_names)} features.")
        mean_abs_by_feature = np.mean(np.abs(array), axis=0)
    else:
        raise ValueError(f"Unsupported SHAP value array shape: {array.shape}")
    for feature, value in zip(feature_names, mean_abs_by_feature):
        global_records.append({"feature": feature, "mean_absolute_shap": float(value)})
    global_table = pd.DataFrame.from_records(global_records).sort_values("mean_absolute_shap", ascending=False)
    global_table["importance_rank"] = np.arange(1, global_table.shape[0] + 1)
    class_table = pd.DataFrame.from_records(class_records)
    if not class_table.empty:
        class_table = class_table.sort_values(["class_name", "mean_absolute_shap"], ascending=[True, False])
        class_table["class_importance_rank"] = class_table.groupby("class_name").cumcount() + 1
    return global_table, class_table


def calculate_shap_importance(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: ModelName = "random_forest",
    max_background: int = 200,
    max_explain: int = 200,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate global SHAP feature importance when SHAP is installed.

    This function preserves the v0.2 API and returns global importance plus a
    status table.  Use :func:`calculate_shap_importance_detailed` when class-level
    SHAP tables are also required.
    """
    global_table, _, status = calculate_shap_importance_detailed(
        features=features,
        labels=labels,
        model_name=model_name,
        max_background=max_background,
        max_explain=max_explain,
        random_state=random_state,
        logger=logger,
    )
    return global_table, status


def calculate_shap_importance_detailed(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: ModelName = "random_forest",
    max_background: int = 200,
    max_explain: int = 200,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate global and class-level SHAP feature importance when possible."""
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        status = pd.DataFrame.from_records([{"status": "not_available", "message": f"SHAP is not available: {exc}"}])
        return pd.DataFrame(columns=["feature", "mean_absolute_shap"]), pd.DataFrame(), status

    try:
        clean_features, clean_labels = _clean_supervised_inputs(features=features, labels=labels)
        background_indices = _sample_indices_stratified(
            labels=clean_labels,
            max_rows=max_background,
            random_state=random_state,
        )
        explain_indices = _sample_indices_stratified(
            labels=clean_labels,
            max_rows=max_explain,
            random_state=random_state + 1,
        )
        background_x = clean_features.iloc[background_indices, :]
        explain_x = clean_features.iloc[explain_indices, :]
        model = build_classifier(model_name=model_name, random_state=random_state)
        # Fit on all available labelled profiles, not just the SHAP background
        # subset. The background subset is only used to keep explainers tractable.
        model.fit(X=clean_features, y=clean_labels)
        if model_name in {"random_forest", "extra_trees", "gradient_boosting"}:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(explain_x)
            except Exception:
                explainer = shap.Explainer(model, background_x)
                shap_values = explainer(explain_x)
        else:
            explainer = shap.Explainer(model, background_x)
            shap_values = explainer(explain_x)
        if isinstance(shap_values, list):
            array = np.stack([np.asarray(item) for item in shap_values], axis=2)
            shap_values = array
        class_names = [str(item) for item in getattr(model, "classes_", [])]
        global_table, class_table = _shap_values_to_tables(
            shap_values=shap_values,
            feature_names=clean_features.columns.tolist(),
            class_names=class_names,
        )
        status = pd.DataFrame.from_records(
            [
                {
                    "status": "ok",
                    "model_name": model_name,
                    "n_background": int(background_x.shape[0]),
                    "n_explained": int(explain_x.shape[0]),
                    "n_features": int(clean_features.shape[1]),
                    "n_classes": int(clean_labels.nunique()),
                    "message": "SHAP importance calculated successfully.",
                }
            ]
        )
    except Exception as exc:  # pragma: no cover - SHAP model-specific robustness
        if logger is not None:
            logger.warning("SHAP calculation failed: %s", exc)
        global_table = pd.DataFrame(columns=["feature", "mean_absolute_shap"])
        class_table = pd.DataFrame(columns=["class_name", "feature", "mean_absolute_shap"])
        status = pd.DataFrame.from_records([{"status": "failed", "model_name": model_name, "message": str(exc)}])
    return global_table, class_table, status


def group_feature_importance_by_family(
    *,
    importance_table: pd.DataFrame,
    feature_column: str = "feature",
    value_column: str = "permutation_importance_mean",
) -> pd.DataFrame:
    """Summarise importance by CellProfiler-style feature family."""
    if feature_column not in importance_table.columns or value_column not in importance_table.columns:
        return pd.DataFrame(columns=["feature_family", "n_features", "total_importance", "mean_importance"])
    table = importance_table.copy()
    feature_text = table[feature_column].astype(str)
    family = np.select(
        [
            feature_text.str.contains("Intensity", case=False, regex=False),
            feature_text.str.contains("Texture", case=False, regex=False),
            feature_text.str.contains("AreaShape", case=False, regex=False),
            feature_text.str.contains("Granularity", case=False, regex=False),
            feature_text.str.contains("RadialDistribution", case=False, regex=False),
            feature_text.str.contains("Correlation", case=False, regex=False),
            feature_text.str.contains("Neighbors", case=False, regex=False),
            feature_text.str.contains("Location", case=False, regex=False),
            feature_text.str.startswith("MissingIndicator__"),
        ],
        [
            "Intensity",
            "Texture",
            "AreaShape",
            "Granularity",
            "RadialDistribution",
            "Correlation",
            "Neighbors",
            "Location",
            "MissingnessIndicator",
        ],
        default="Other",
    )
    table["feature_family"] = family
    table[value_column] = pd.to_numeric(table[value_column], errors="coerce")
    grouped = table.groupby("feature_family", dropna=False)[value_column].agg(["count", "sum", "mean"]).reset_index()
    grouped = grouped.rename(
        columns={"count": "n_features", "sum": "total_importance", "mean": "mean_importance"}
    ).sort_values("total_importance", ascending=False)
    return grouped
