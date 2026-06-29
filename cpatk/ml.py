"""Machine-learning classifiers and calibration utilities for CPATK."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence
import inspect

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ModelName = Literal[
    "knn",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "logistic_regression",
    "linear_svm",
]


def _make_calibrated_classifier(*, base_estimator: object):
    """Create a calibrated classifier across scikit-learn versions.

    Parameters
    ----------
    base_estimator:
        Base classifier to calibrate.

    Returns
    -------
    sklearn.calibration.CalibratedClassifierCV
        Calibrated classifier.
    """
    signature = inspect.signature(CalibratedClassifierCV)
    if "estimator" in signature.parameters:
        return CalibratedClassifierCV(estimator=base_estimator, cv=3)
    return CalibratedClassifierCV(base_estimator=base_estimator, cv=3)


def build_classifier(
    *,
    model_name: ModelName = "random_forest",
    random_state: int = 42,
    n_neighbours: int = 5,
    calibrate: bool = True,
    n_jobs: int = 1,
):
    """Build a supported classifier.

    Parameters
    ----------
    model_name:
        Name of the classifier.
    random_state:
        Random seed.
    n_neighbours:
        Neighbour count for KNN.
    calibrate:
        Whether to wrap margin-based models in probability calibration.
    n_jobs:
        Number of jobs for estimators that support parallel execution.

    Returns
    -------
    sklearn estimator
        Configured classifier.
    """
    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=n_neighbours,
            weights="distance",
            n_jobs=max(1, int(n_jobs)),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=max(1, int(n_jobs)),
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=max(1, int(n_jobs)),
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced"
        )
    if model_name == "linear_svm":
        base = LinearSVC(random_state=random_state, class_weight="balanced", max_iter=5000)
        if calibrate:
            return _make_calibrated_classifier(base_estimator=base)
        return base
    raise ValueError(f"Unsupported model_name: {model_name}")


def cross_validate_classifier(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: ModelName = "random_forest",
    n_splits: int = 5,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-validate a classifier and return metrics and predictions.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    labels:
        Class labels.
    model_name:
        Classifier name.
    n_splits:
        Maximum number of folds.
    random_state:
        Random seed.
    logger:
        Optional logger.
    n_jobs:
        Number of jobs for supported estimators and cross-validation.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Summary, per-profile predictions, and confusion matrix.
    """
    clean_labels = labels.astype(str).reset_index(drop=True)
    counts = clean_labels.value_counts()
    if counts.shape[0] < 2 or counts.min() < 2:
        raise ValueError("At least two classes with at least two profiles each are required.")
    folds = min(n_splits, int(counts.min()))
    model = build_classifier(
        model_name=model_name,
        random_state=random_state,
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    predicted = cross_val_predict(
        estimator=model,
        X=features,
        y=clean_labels,
        cv=cv,
        n_jobs=max(1, int(n_jobs)),
    )
    summary = pd.DataFrame.from_records(
        [
            {
                "model_name": model_name,
                "n_profiles": int(features.shape[0]),
                "n_classes": int(counts.shape[0]),
                "n_splits": int(folds),
                "accuracy": float(accuracy_score(y_true=clean_labels, y_pred=predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true=clean_labels, y_pred=predicted)),
                "macro_f1": float(f1_score(y_true=clean_labels, y_pred=predicted, average="macro")),
                "probability_status": "not_calculated_in_fast_cv",
                "cross_validated_log_loss": np.nan,
            }
        ]
    )
    predictions = pd.DataFrame(
        {
            "row_index": features.index,
            "true_class": clean_labels.to_numpy(),
            "predicted_class": predicted,
            "correct": clean_labels.to_numpy() == predicted,
        }
    )
    labels_sorted = sorted(clean_labels.unique())
    matrix = confusion_matrix(y_true=clean_labels, y_pred=predicted, labels=labels_sorted)
    confusion = pd.DataFrame(data=matrix, index=labels_sorted, columns=labels_sorted).reset_index()
    confusion = confusion.rename(columns={"index": "true_class"})
    if logger is not None:
        logger.info("%s balanced accuracy: %.3f", model_name, summary["balanced_accuracy"].iloc[0])
    return summary, predictions, confusion


def train_predict_classifier(
    *,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    query_features: pd.DataFrame,
    model_name: ModelName = "random_forest",
    random_state: int = 42,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, object]:
    """Train a classifier and predict query classes with confidence scores.

    Parameters
    ----------
    train_features:
        Training features.
    train_labels:
        Training labels.
    query_features:
        Query features.
    model_name:
        Classifier name.
    random_state:
        Random seed.
    n_jobs:
        Number of jobs for supported estimators.

    Returns
    -------
    tuple[pandas.DataFrame, object]
        Prediction table and fitted model.
    """
    shared = [column for column in train_features.columns if column in query_features.columns]
    if not shared:
        raise ValueError("No shared feature columns between training and query tables.")
    model = build_classifier(
        model_name=model_name,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X=train_features.loc[:, shared], y=train_labels.astype(str))
    predicted = model.predict(X=query_features.loc[:, shared])
    result = pd.DataFrame({"query_index": query_features.index, "predicted_class": predicted})
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X=query_features.loc[:, shared])
        classes = [str(item) for item in model.classes_]
        result["max_probability"] = probabilities.max(axis=1)
        for class_index, class_name in enumerate(classes):
            result[f"probability_{class_name}"] = probabilities[:, class_index]
    return result, model


def compare_moa_models(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    model_names: Optional[Sequence[ModelName]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare multiple MOA classifiers by cross-validation.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    labels:
        Known labels.
    model_names:
        Models to compare.
    n_splits:
        Maximum number of folds.
    random_state:
        Random seed.
    logger:
        Optional logger.
    n_jobs:
        Number of jobs for supported estimators and cross-validation.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Model summary and long-format predictions.
    """
    model_names = list(model_names or ["knn", "random_forest", "extra_trees", "logistic_regression"])
    summaries = []
    predictions = []
    for model_name in model_names:
        summary, prediction, _ = cross_validate_classifier(
            features=features,
            labels=labels,
            model_name=model_name,
            n_splits=n_splits,
            random_state=random_state,
            logger=logger,
            n_jobs=n_jobs,
        )
        prediction["model_name"] = model_name
        summaries.append(summary)
        predictions.append(prediction)
    return pd.concat(summaries, ignore_index=True), pd.concat(predictions, ignore_index=True)
