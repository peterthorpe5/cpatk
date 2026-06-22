"""Preprocessing workflows for generic Cell Painting data.

The preprocessing defaults are intentionally conservative.  They are designed
for CellProfiler-style image/object/profile tables where metadata can be messy,
numeric identifiers can masquerade as features and missing values can be caused
by segmentation, measurement or merge artefacts.  The workflow therefore writes
an auditable trail of column roles, QC decisions, imputation, normalisation,
scaling and feature filtering.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from cpatk.features import assign_column_roles, split_metadata_and_features
from cpatk.metadata import (
    drop_unnamed_index_columns,
    normalise_column_names,
    standardise_metadata_aliases,
)
from cpatk.reproducibility import calculate_replicate_correlations, summarise_replicate_correlations
from cpatk.qc import (
    calculate_feature_qc,
    calculate_sample_qc,
    flag_samples_by_qc,
    select_features_by_qc,
)


def validate_preprocessing_parameters(
    *,
    max_feature_missing_fraction: float,
    max_sample_missing_fraction: float,
    max_absolute_correlation: float,
    min_unique_values: int,
    max_zero_fraction: float = 1.0,
    all_zero_row_tolerance: float = 0.0,
) -> None:
    """Validate preprocessing thresholds before expensive work begins."""
    if not 0 <= max_feature_missing_fraction <= 1:
        raise ValueError("max_feature_missing_fraction must be between 0 and 1.")
    if not 0 <= max_sample_missing_fraction <= 1:
        raise ValueError("max_sample_missing_fraction must be between 0 and 1.")
    if not 0 < max_absolute_correlation <= 1:
        raise ValueError("max_absolute_correlation must be in the interval (0, 1].")
    if min_unique_values < 1:
        raise ValueError("min_unique_values must be at least 1.")
    if not 0 <= max_zero_fraction <= 1:
        raise ValueError("max_zero_fraction must be between 0 and 1.")
    if all_zero_row_tolerance < 0:
        raise ValueError("all_zero_row_tolerance must be non-negative.")



def replace_nonfinite_with_nan(
    *,
    features: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
    max_abs_finite_value: float = 1e10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replace unsafe non-finite or implausibly huge feature values with NaN.

    CellProfiler tables can occasionally contain infinite values after ratios or
    malformed numeric values such as ``#DIV/0!``. Legacy project scripts also
    treated extremely large finite values as unsafe, because they usually reflect
    calculation or export artefacts rather than interpretable morphology. CPATK
    therefore converts ``inf``, ``-inf`` and finite values whose absolute value
    exceeds ``max_abs_finite_value`` to missing values before QC and imputation.
    """
    output = features.copy()
    records = []
    for feature in output.columns:
        values = pd.to_numeric(output[feature], errors="coerce")
        pos_inf = int(np.isposinf(values).sum())
        neg_inf = int(np.isneginf(values).sum())
        finite = np.isfinite(values.to_numpy(dtype=float, copy=False))
        not_missing = values.notna().to_numpy()
        nonfinite = int((~finite & not_missing).sum())
        extreme_mask = values.abs().gt(max_abs_finite_value) & np.isfinite(values)
        n_extreme = int(extreme_mask.sum())
        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.mask(extreme_mask, np.nan)
        output[feature] = values
        records.append(
            {
                "feature": feature,
                "n_positive_infinity": pos_inf,
                "n_negative_infinity": neg_inf,
                "n_extreme_finite_values_replaced": n_extreme,
                "n_nonfinite_replaced": nonfinite,
                "n_total_values_replaced": nonfinite + n_extreme,
                "max_abs_finite_value": max_abs_finite_value,
            }
        )
    report = pd.DataFrame.from_records(records)
    if logger is not None:
        logger.info(
            "Replaced %s non-finite/extreme feature values with missing values",
            int(report["n_total_values_replaced"].sum()),
        )
    return output, report


def calculate_all_zero_row_report(
    *,
    features: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    tolerance: float = 0.0,
    metadata_preview_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Identify rows with no non-zero observed feature values.

    This QC step is intentionally applied after all CellProfiler output tables
    have been merged into a single feature matrix and after feature columns have
    been selected.  A row is flagged when it has at least one observed feature
    value and every observed feature value is zero, or within ``tolerance`` of
    zero.  Missing values do not count as non-zero evidence; rows that are
    entirely missing are handled separately by the sample/profile missingness
    filter.

    Parameters
    ----------
    features:
        Numeric feature matrix after feature-level QC selection.
    metadata:
        Optional metadata table aligned to ``features``. A small set of useful
        metadata columns is copied into the report for auditability.
    tolerance:
        Absolute tolerance used when deciding whether a value is zero.
    metadata_preview_columns:
        Optional metadata columns to include in the report. If omitted, common
        metadata identifiers such as plate, well, compound and batch are used
        when available.

    Returns
    -------
    pandas.DataFrame
        One row per input profile with zero-row counts and flags.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    observed = numeric.notna()
    zero_like = numeric.abs().le(tolerance) & observed
    n_observed = observed.sum(axis=1).astype(int)
    n_zero_like = zero_like.sum(axis=1).astype(int)
    n_nonzero = (observed & ~zero_like).sum(axis=1).astype(int)
    all_zero = (n_observed > 0) & (n_nonzero == 0)
    report = pd.DataFrame(
        {
            "row_index": features.index.to_numpy(),
            "n_features_examined": int(features.shape[1]),
            "n_observed_feature_values": n_observed.to_numpy(),
            "n_zero_or_near_zero_observed_feature_values": n_zero_like.to_numpy(),
            "n_nonzero_observed_feature_values": n_nonzero.to_numpy(),
            "all_zero_feature_row": all_zero.to_numpy(dtype=bool),
            "removed_by_all_zero_row_filter": False,
        }
    )
    if metadata is not None and not metadata.empty:
        default_columns = [
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_Compound",
            "Metadata_MOA",
            "Metadata_Batch",
            "cpd_id",
            "cpd_type",
        ]
        wanted = list(metadata_preview_columns or default_columns)
        for column in wanted:
            if column in metadata.columns:
                report[column] = metadata[column].reset_index(drop=True)
    return report


def add_missingness_indicators(
    *,
    features: pd.DataFrame,
    max_indicators: int = 500,
    minimum_missing_fraction: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create binary missingness-indicator features.

    Missingness indicators can be useful when missingness itself carries
    information, but they should not be added blindly to extremely sparse data.
    The number of indicators is therefore capped and every added indicator is
    reported.
    """
    records = []
    indicator_columns = []
    missing_fractions = features.isna().mean(axis=0).sort_values(ascending=False)
    eligible = missing_fractions[missing_fractions > minimum_missing_fraction]
    eligible = eligible.head(max_indicators)
    for feature, missing_fraction in eligible.items():
        indicator_name = f"MissingIndicator__{feature}"
        indicator_columns.append(features[feature].isna().astype(int).rename(indicator_name))
        records.append(
            {
                "source_feature": feature,
                "indicator_feature": indicator_name,
                "missing_fraction": float(missing_fraction),
            }
        )
    indicators = pd.concat(indicator_columns, axis=1) if indicator_columns else pd.DataFrame(index=features.index)
    if logger is not None:
        logger.info("Created %s missingness indicator features", indicators.shape[1])
    return indicators, pd.DataFrame.from_records(records)


def summarise_imputation(*, before: pd.DataFrame, after: pd.DataFrame, method: str) -> pd.DataFrame:
    """Summarise missingness before and after imputation."""
    records = []
    for feature in before.columns:
        before_values = pd.to_numeric(before[feature], errors="coerce")
        if feature in after.columns:
            after_values = pd.to_numeric(after[feature], errors="coerce")
        else:
            after_values = pd.Series(dtype=float)
        records.append(
            {
                "feature": feature,
                "imputation_method": method,
                "n_missing_before": int(before_values.isna().sum()),
                "missing_fraction_before": float(before_values.isna().mean()),
                "n_missing_after": int(after_values.isna().sum()),
                "missing_fraction_after": float(after_values.isna().mean()) if len(after_values) else np.nan,
                "median_before": float(before_values.median(skipna=True)) if before_values.notna().any() else np.nan,
                "median_after": float(after_values.median(skipna=True)) if after_values.notna().any() else np.nan,
                "sd_before": float(before_values.std(skipna=True)) if before_values.notna().sum() > 1 else np.nan,
                "sd_after": float(after_values.std(skipna=True)) if after_values.notna().sum() > 1 else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _groupwise_fill(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    group_columns: Sequence[str],
    statistic: str,
) -> pd.DataFrame:
    """Fill missing values using group-wise summary statistics."""
    valid_groups = [column for column in group_columns if column in metadata.columns]
    if not valid_groups:
        raise ValueError("Group-wise imputation requested but no valid group columns were supplied.")
    output = features.copy()
    grouped_metadata = metadata.loc[:, valid_groups].astype(str).agg("|".join, axis=1)
    for feature in output.columns:
        values = pd.to_numeric(output[feature], errors="coerce")
        if statistic == "median":
            group_values = values.groupby(grouped_metadata).transform("median")
            global_value = values.median(skipna=True)
        elif statistic == "mean":
            group_values = values.groupby(grouped_metadata).transform("mean")
            global_value = values.mean(skipna=True)
        else:
            raise ValueError(f"Unsupported group-wise imputation statistic: {statistic}")
        output[feature] = values.fillna(group_values).fillna(global_value).fillna(0)
    return output


def impute_features(
    *,
    features: pd.DataFrame,
    method: str = "median",
    n_neighbors: int = 5,
    metadata: Optional[pd.DataFrame] = None,
    group_columns: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Impute missing values in a feature matrix.

    Median imputation is the safest default for Cell Painting profiles because
    it is robust and does not borrow morphology across treatments.  KNN
    imputation is available for exploratory use but should be interpreted more
    cautiously because it can smooth real perturbation or batch structure.
    """
    method = method.lower()
    if logger is not None:
        logger.info("Imputing features using method=%s", method)

    if method in {"group_median", "group_mean"}:
        if metadata is None:
            raise ValueError("metadata is required for group-wise imputation.")
        statistic = "median" if method == "group_median" else "mean"
        return _groupwise_fill(
            features=features,
            metadata=metadata,
            group_columns=group_columns or [],
            statistic=statistic,
        )

    if method in {"median", "mean"}:
        imputer = SimpleImputer(strategy=method)
    elif method == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif method == "knn":
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1 for KNN imputation.")
        if features.shape[0] < 2:
            if logger is not None:
                logger.warning(
                    "KNN imputation requested for fewer than two rows; using median imputation instead."
                )
            imputer = SimpleImputer(strategy="median")
        else:
            effective_neighbors = min(int(n_neighbors), max(1, features.shape[0] - 1))
            if effective_neighbors != n_neighbors and logger is not None:
                logger.warning(
                    "Capped KNN n_neighbors from %s to %s for %s rows.",
                    n_neighbors,
                    effective_neighbors,
                    features.shape[0],
                )
            imputer = KNNImputer(n_neighbors=effective_neighbors)
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    values = imputer.fit_transform(X=features)
    return pd.DataFrame(data=values, columns=features.columns, index=features.index)


def winsorise_features(
    *,
    features: pd.DataFrame,
    lower_quantile: Optional[float] = None,
    upper_quantile: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally clip extreme feature values to empirical quantiles."""
    if lower_quantile is None and upper_quantile is None:
        return features.copy(), pd.DataFrame(columns=["feature", "lower", "upper", "n_clipped_low", "n_clipped_high"])
    lower_quantile = 0.0 if lower_quantile is None else float(lower_quantile)
    upper_quantile = 1.0 if upper_quantile is None else float(upper_quantile)
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Winsorisation quantiles must satisfy 0 <= lower < upper <= 1.")
    output = features.copy()
    records = []
    for feature in output.columns:
        values = pd.to_numeric(output[feature], errors="coerce")
        lower = values.quantile(lower_quantile)
        upper = values.quantile(upper_quantile)
        n_low = int((values < lower).sum()) if pd.notna(lower) else 0
        n_high = int((values > upper).sum()) if pd.notna(upper) else 0
        output[feature] = values.clip(lower=lower, upper=upper)
        records.append(
            {
                "feature": feature,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_value": float(lower) if pd.notna(lower) else np.nan,
                "upper_value": float(upper) if pd.notna(upper) else np.nan,
                "n_clipped_low": n_low,
                "n_clipped_high": n_high,
            }
        )
    if logger is not None:
        logger.info("Winsorisation applied to %s features", output.shape[1])
    return output, pd.DataFrame.from_records(records)


def normalise_features_to_reference(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    reference_column: Optional[str] = None,
    reference_values: Optional[Sequence[str]] = None,
    group_columns: Optional[Sequence[str]] = None,
    method: str = "none",
    min_reference_profiles: int = 2,
    epsilon: float = 1e-9,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise profiles to control/reference profiles.

    Supported methods are ``none``, ``robust_z``, ``median_center`` and
    ``zscore``.  When a reference column/value is provided, reference profiles
    define the baseline within each requested group, such as each plate.  This
    is often a defensible Cell Painting normalisation strategy, but it should
    only be enabled when real controls are present and named correctly.
    """
    method = method.lower()
    if method == "none":
        return features.copy(), pd.DataFrame(columns=["group", "feature", "status"])
    if reference_column is None or reference_column not in metadata.columns:
        raise ValueError("Reference normalisation requires a valid reference_column in metadata.")
    if not reference_values:
        raise ValueError("Reference normalisation requires at least one reference value.")
    valid_groups = [column for column in (group_columns or []) if column in metadata.columns]
    output = features.copy()
    records = []
    ref_values = {str(value) for value in reference_values}
    if valid_groups:
        group_series = metadata.loc[:, valid_groups].astype(str).agg("|".join, axis=1)
    else:
        group_series = pd.Series("__global__", index=metadata.index)
    ref_mask_all = metadata[reference_column].astype(str).isin(ref_values)
    for group_name, row_index in group_series.groupby(group_series, dropna=False).groups.items():
        row_index = list(row_index)
        group_ref_mask = ref_mask_all.loc[row_index].to_numpy(dtype=bool)
        if int(group_ref_mask.sum()) < min_reference_profiles:
            if logger is not None:
                logger.warning(
                    "Skipping reference normalisation for group %s because only %s reference profiles were found",
                    group_name,
                    int(group_ref_mask.sum()),
                )
            for feature in output.columns:
                records.append({"group": group_name, "feature": feature, "status": "skipped_insufficient_reference"})
            continue
        group_features = features.loc[row_index, :]
        ref_features = group_features.loc[group_ref_mask, :]
        for feature in output.columns:
            ref_values_numeric = pd.to_numeric(ref_features[feature], errors="coerce")
            if method == "robust_z":
                centre = ref_values_numeric.median(skipna=True)
                scale = (ref_values_numeric - centre).abs().median(skipna=True)
            elif method == "median_center":
                centre = ref_values_numeric.median(skipna=True)
                scale = 1.0
            elif method == "zscore":
                centre = ref_values_numeric.mean(skipna=True)
                scale = ref_values_numeric.std(skipna=True)
            else:
                raise ValueError(f"Unsupported reference normalisation method: {method}")
            scale = float(scale) if pd.notna(scale) else 0.0
            if abs(scale) < epsilon:
                scale = 1.0
                status = "ok_scale_replaced_by_one"
            else:
                status = "ok"
            output.loc[row_index, feature] = (pd.to_numeric(group_features[feature], errors="coerce") - centre) / scale
            records.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "status": status,
                    "reference_column": reference_column,
                    "reference_values": ";".join(sorted(ref_values)),
                    "n_reference_profiles": int(group_ref_mask.sum()),
                    "centre": float(centre) if pd.notna(centre) else np.nan,
                    "scale": float(scale),
                    "method": method,
                }
            )
    if logger is not None:
        logger.info("Reference normalisation method=%s complete", method)
    return output, pd.DataFrame.from_records(records)


def batch_center_features(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_columns: Optional[Sequence[str]] = None,
    method: str = "none",
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally centre features within batch groups.

    This is a simple diagnostic/engineering correction, not a replacement for a
    fully blocked design.  It is disabled by default.
    """
    method = method.lower()
    if method == "none" or not batch_columns:
        return features.copy(), pd.DataFrame(columns=["group", "feature", "status"])
    valid_groups = [column for column in batch_columns if column in metadata.columns]
    if not valid_groups:
        raise ValueError("Batch centering requested but no valid batch columns were supplied.")
    output = features.copy()
    group_series = metadata.loc[:, valid_groups].astype(str).agg("|".join, axis=1)
    records = []
    for group_name, row_index in group_series.groupby(group_series, dropna=False).groups.items():
        row_index = list(row_index)
        group_features = output.loc[row_index, :]
        for feature in output.columns:
            values = pd.to_numeric(group_features[feature], errors="coerce")
            if method == "median_center":
                centre = values.median(skipna=True)
            elif method == "mean_center":
                centre = values.mean(skipna=True)
            else:
                raise ValueError(f"Unsupported batch centering method: {method}")
            output.loc[row_index, feature] = values - centre
            records.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "status": "ok",
                    "method": method,
                    "centre": float(centre) if pd.notna(centre) else np.nan,
                }
            )
    if logger is not None:
        logger.info("Batch centering method=%s complete", method)
    return output, pd.DataFrame.from_records(records)


def calculate_reference_control_qc(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    reference_column: Optional[str],
    reference_values: Optional[Sequence[str]],
    group_columns: Optional[Sequence[str]] = None,
    method: str = "none",
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Summarise reference/control profiles before normalisation.

    Parameters
    ----------
    features:
        Feature matrix before reference normalisation.
    metadata:
        Metadata aligned to ``features``.
    reference_column:
        Metadata column identifying controls, for example ``Metadata_Compound``.
    reference_values:
        Values treated as reference controls, for example ``DMSO``.
    group_columns:
        Optional grouping columns, usually ``Metadata_Plate`` for per-plate
        normalisation.
    method:
        Requested reference normalisation method.
    epsilon:
        Small value below which a robust scale is considered zero.

    Returns
    -------
    pandas.DataFrame
        Per-group control-count and scale-quality report.
    """
    columns = [
        "group",
        "status",
        "method",
        "reference_column",
        "reference_values",
        "n_profiles",
        "n_reference_profiles",
        "reference_fraction",
        "n_features",
        "n_features_zero_or_near_zero_mad",
        "fraction_features_zero_or_near_zero_mad",
        "median_reference_missing_fraction",
    ]
    if method.lower() == "none":
        return pd.DataFrame(columns=columns)
    if reference_column is None or not reference_values:
        return pd.DataFrame.from_records(
            [
                {
                    "group": "__all__",
                    "status": "not_tested_missing_reference_definition",
                    "method": method,
                    "reference_column": str(reference_column),
                    "reference_values": ";".join(reference_values or []),
                    "n_profiles": int(features.shape[0]),
                    "n_reference_profiles": 0,
                    "reference_fraction": 0.0,
                    "n_features": int(features.shape[1]),
                    "n_features_zero_or_near_zero_mad": 0,
                    "fraction_features_zero_or_near_zero_mad": 0.0,
                    "median_reference_missing_fraction": float("nan"),
                }
            ]
        )
    if reference_column not in metadata.columns:
        raise ValueError(f"Reference column is missing from metadata: {reference_column}")
    valid_groups = [column for column in (group_columns or []) if column in metadata.columns]
    aligned_metadata = metadata.reset_index(drop=True)
    aligned_features = features.reset_index(drop=True)
    reference_set = {str(value) for value in reference_values}
    if valid_groups:
        group_series = aligned_metadata.loc[:, valid_groups].astype(str).agg("|".join, axis=1)
    else:
        group_series = pd.Series(["__all__"] * aligned_metadata.shape[0])
    records = []
    ref_mask_all = aligned_metadata[reference_column].astype(str).isin(reference_set)
    for group_name, row_index in group_series.groupby(group_series, dropna=False).groups.items():
        rows = list(row_index)
        group_ref = ref_mask_all.iloc[rows].to_numpy(dtype=bool)
        n_ref = int(group_ref.sum())
        ref_features = aligned_features.iloc[rows, :].loc[group_ref, :]
        if n_ref == 0:
            status = "failed_no_reference_profiles"
            n_zero_mad = 0
            median_missing = float("nan")
        else:
            mad = ref_features.apply(
                lambda values: (pd.to_numeric(values, errors="coerce") - pd.to_numeric(values, errors="coerce").median(skipna=True)).abs().median(skipna=True),
                axis=0,
            )
            n_zero_mad = int((mad.fillna(0.0).abs() <= epsilon).sum())
            median_missing = float(ref_features.isna().mean(axis=0).median(skipna=True))
            status = "ok" if n_ref >= 2 else "review_fewer_than_two_reference_profiles"
        records.append(
            {
                "group": group_name,
                "status": status,
                "method": method,
                "reference_column": reference_column,
                "reference_values": ";".join(sorted(reference_set)),
                "n_profiles": int(len(rows)),
                "n_reference_profiles": n_ref,
                "reference_fraction": float(n_ref / max(len(rows), 1)),
                "n_features": int(aligned_features.shape[1]),
                "n_features_zero_or_near_zero_mad": n_zero_mad,
                "fraction_features_zero_or_near_zero_mad": float(n_zero_mad / max(aligned_features.shape[1], 1)),
                "median_reference_missing_fraction": median_missing,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def assess_batch_confounding(
    *,
    metadata: pd.DataFrame,
    batch_column: Optional[str],
    protected_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Assess whether biological labels are confounded with batch.

    The report is deliberately conservative. It flags cases where a protected
    label appears in only one batch, or where each batch contains only one value
    for a protected label. Such designs make batch correction hard to interpret.
    """
    columns = [
        "protected_column",
        "status",
        "batch_column",
        "n_batches",
        "n_labels",
        "n_labels_observed_in_one_batch",
        "n_batches_with_one_label",
        "interpretation",
    ]
    if batch_column is None or batch_column not in metadata.columns:
        return pd.DataFrame.from_records(
            [
                {
                    "protected_column": "",
                    "status": "not_tested_missing_batch_column",
                    "batch_column": str(batch_column),
                    "n_batches": 0,
                    "n_labels": 0,
                    "n_labels_observed_in_one_batch": 0,
                    "n_batches_with_one_label": 0,
                    "interpretation": "Batch correction was not assessed for confounding because no valid batch column was supplied.",
                }
            ]
        )
    protected = [column for column in (protected_columns or []) if column in metadata.columns and column != batch_column]
    labels_batch = metadata[batch_column].astype(str)
    n_batches = int(labels_batch.nunique(dropna=False))
    if not protected:
        return pd.DataFrame.from_records(
            [
                {
                    "protected_column": "",
                    "status": "not_tested_no_protected_columns",
                    "batch_column": batch_column,
                    "n_batches": n_batches,
                    "n_labels": 0,
                    "n_labels_observed_in_one_batch": 0,
                    "n_batches_with_one_label": 0,
                    "interpretation": "No biological/protected columns were supplied for confounding assessment.",
                }
            ]
        )
    records = []
    for column in protected:
        labels = metadata[column].astype(str)
        batches_per_label = labels_batch.groupby(labels, dropna=False).nunique(dropna=False)
        labels_in_one_batch = int((batches_per_label <= 1).sum())
        labels_per_batch = labels.groupby(labels_batch, dropna=False).nunique(dropna=False)
        batches_with_one_label = int((labels_per_batch <= 1).sum())
        if labels_in_one_batch == int(labels.nunique(dropna=False)) or batches_with_one_label == n_batches:
            status = "high_risk_confounded"
        elif labels_in_one_batch > 0 or batches_with_one_label > 0:
            status = "review_partial_confounding"
        else:
            status = "ok_mixed_design"
        records.append(
            {
                "protected_column": column,
                "status": status,
                "batch_column": batch_column,
                "n_batches": n_batches,
                "n_labels": int(labels.nunique(dropna=False)),
                "n_labels_observed_in_one_batch": labels_in_one_batch,
                "n_batches_with_one_label": batches_with_one_label,
                "interpretation": (
                    "High-risk confounding means batch correction may remove biology or fail to remove batch. "
                    "Prefer balanced designs and inspect before/after QC."
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def combat_style_location_scale_correction(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_column: Optional[str],
    protected_columns: Optional[Sequence[str]] = None,
    method: str = "none",
    min_batch_size: int = 3,
    epsilon: float = 1e-8,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply optional ComBat-style location/scale batch correction.

    This is an auditable location/scale harmonisation inspired by the practical
    goal of ComBat: adjust feature distributions across batches while retaining
    a global feature scale. It is not an empirical-Bayes implementation and the
    report names it clearly as ``combat_location_scale``.
    """
    method = method.lower()
    empty_report = pd.DataFrame(columns=["batch", "feature", "status", "method"])
    confounding = assess_batch_confounding(
        metadata=metadata,
        batch_column=batch_column,
        protected_columns=protected_columns,
    )
    if method == "none":
        return features.copy(), empty_report, confounding
    if method != "combat_location_scale":
        raise ValueError("batch_correction_method must be none or combat_location_scale.")
    if batch_column is None or batch_column not in metadata.columns:
        raise ValueError("combat_location_scale correction requires a valid batch_column.")
    if min_batch_size < 2:
        raise ValueError("min_batch_size must be at least 2 for batch correction.")
    output = features.copy()
    numeric = features.apply(pd.to_numeric, errors="coerce")
    global_centre = numeric.mean(axis=0, skipna=True)
    global_scale = numeric.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)
    batches = metadata[batch_column].astype(str).reset_index(drop=True)
    records = []
    for batch, row_index in batches.groupby(batches, dropna=False).groups.items():
        rows = list(row_index)
        if len(rows) < min_batch_size:
            for feature in numeric.columns:
                records.append(
                    {
                        "batch": batch,
                        "feature": feature,
                        "status": "skipped_small_batch",
                        "method": method,
                        "n_batch_profiles": int(len(rows)),
                        "batch_centre": float("nan"),
                        "batch_scale": float("nan"),
                        "global_centre": float(global_centre[feature]) if pd.notna(global_centre[feature]) else float("nan"),
                        "global_scale": float(global_scale[feature]) if pd.notna(global_scale[feature]) else float("nan"),
                    }
                )
            continue
        block = numeric.iloc[rows, :]
        batch_centre = block.mean(axis=0, skipna=True)
        batch_scale = block.std(axis=0, skipna=True)
        for feature in numeric.columns:
            scale = batch_scale[feature]
            status = "ok"
            if pd.isna(scale) or abs(float(scale)) <= epsilon:
                scale = 1.0
                status = "ok_scale_replaced_by_one"
            adjusted = ((block[feature] - batch_centre[feature]) / float(scale)) * global_scale[feature] + global_centre[feature]
            output.loc[rows, feature] = adjusted.to_numpy()
            records.append(
                {
                    "batch": batch,
                    "feature": feature,
                    "status": status,
                    "method": method,
                    "n_batch_profiles": int(len(rows)),
                    "batch_centre": float(batch_centre[feature]) if pd.notna(batch_centre[feature]) else float("nan"),
                    "batch_scale": float(scale),
                    "global_centre": float(global_centre[feature]) if pd.notna(global_centre[feature]) else float("nan"),
                    "global_scale": float(global_scale[feature]) if pd.notna(global_scale[feature]) else float("nan"),
                }
            )
    if logger is not None:
        logger.info("ComBat-style location/scale batch correction method=%s complete", method)
    return output, pd.DataFrame.from_records(records), confounding


def _summarise_stage_replicates(
    *,
    stage: str,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    replicate_group_columns: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate replicate-pair and replicate-summary tables for one stage."""
    valid = [column for column in (replicate_group_columns or []) if column in metadata.columns]
    if not valid:
        empty_pairs = pd.DataFrame(columns=["stage", "replicate_group", "correlation"])
        empty_summary = pd.DataFrame.from_records(
            [{"stage": stage, "status": "not_tested_no_valid_replicate_group_columns"}]
        )
        return empty_pairs, empty_summary
    pairs = calculate_replicate_correlations(
        features=features,
        metadata=metadata,
        replicate_group_columns=valid,
    )
    if not pairs.empty:
        pairs.insert(0, "stage", stage)
    summary = summarise_replicate_correlations(
        replicate_correlations=pairs.drop(columns=["stage"], errors="ignore"),
        group_columns=valid,
    )
    if summary.empty:
        summary = pd.DataFrame.from_records([{column: "" for column in valid}])
        summary["status"] = "not_tested_no_groups_with_replicates"
    summary.insert(0, "stage", stage)
    return pairs, summary


def _summarise_stage_batch_association(
    *,
    stage: str,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_report_columns: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Summarise simple PC1/PC2 association with batch/report columns."""
    valid = [column for column in (batch_report_columns or []) if column in metadata.columns]
    if not valid:
        return pd.DataFrame.from_records(
            [{"stage": stage, "metadata_column": "", "status": "not_tested_no_valid_batch_report_columns"}]
        )
    numeric = features.apply(pd.to_numeric, errors="coerce").fillna(features.median(numeric_only=True))
    if numeric.shape[0] < 3 or numeric.shape[1] < 1:
        return pd.DataFrame.from_records(
            [{"stage": stage, "metadata_column": "", "status": "not_tested_too_few_profiles_or_features"}]
        )
    values = numeric.to_numpy(dtype=float)
    values = values - np.nanmean(values, axis=0)
    try:
        _, singular_values, vt = np.linalg.svd(values, full_matrices=False)
        scores = values @ vt[: min(2, vt.shape[0])].T
    except np.linalg.LinAlgError:
        return pd.DataFrame.from_records(
            [{"stage": stage, "metadata_column": "", "status": "not_tested_svd_failed"}]
        )
    records = []
    for column in valid:
        labels = metadata[column].astype(str).reset_index(drop=True)
        for component_index in range(scores.shape[1]):
            component = scores[:, component_index]
            grand_mean = float(np.nanmean(component))
            total_ss = float(np.nansum((component - grand_mean) ** 2))
            between_ss = 0.0
            for _, row_index in labels.groupby(labels, dropna=False).groups.items():
                group_values = component[list(row_index)]
                between_ss += len(group_values) * (float(np.nanmean(group_values)) - grand_mean) ** 2
            records.append(
                {
                    "stage": stage,
                    "metadata_column": column,
                    "component": f"PC{component_index + 1}",
                    "status": "tested",
                    "eta_squared": float(between_ss / total_ss) if total_ss > 0 else float("nan"),
                    "n_groups": int(labels.nunique(dropna=False)),
                    "n_profiles": int(features.shape[0]),
                }
            )
    return pd.DataFrame.from_records(records)


def make_before_after_qc_reports(
    *,
    before_features: pd.DataFrame,
    after_features: pd.DataFrame,
    metadata: pd.DataFrame,
    replicate_group_columns: Optional[Sequence[str]] = None,
    batch_report_columns: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Create before/after replicate and batch-association reports."""
    rep_pairs = []
    rep_summaries = []
    batch_summaries = []
    for stage, stage_features in [("before_batch_correction", before_features), ("after_batch_correction", after_features)]:
        pairs, summary = _summarise_stage_replicates(
            stage=stage,
            features=stage_features,
            metadata=metadata,
            replicate_group_columns=replicate_group_columns,
        )
        rep_pairs.append(pairs)
        rep_summaries.append(summary)
        batch_summaries.append(
            _summarise_stage_batch_association(
                stage=stage,
                features=stage_features,
                metadata=metadata,
                batch_report_columns=batch_report_columns,
            )
        )
    return {
        "before_after_replicate_correlations": pd.concat(rep_pairs, ignore_index=True),
        "before_after_replicate_summary": pd.concat(rep_summaries, ignore_index=True),
        "before_after_batch_pc_association": pd.concat(batch_summaries, ignore_index=True),
    }



def scale_features(
    *,
    features: pd.DataFrame,
    method: str = "robust",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Scale a feature matrix."""
    method = method.lower()
    if logger is not None:
        logger.info("Scaling features using method=%s", method)
    if method == "none":
        return features.copy()
    if method == "robust":
        scaler = RobustScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
    values = scaler.fit_transform(X=features)
    return pd.DataFrame(data=values, columns=features.columns, index=features.index)


def remove_correlated_features(
    *,
    features: pd.DataFrame,
    max_absolute_correlation: float = 0.95,
    max_features_for_correlation: int = 5000,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove one feature from each highly correlated pair.

    Features with more missing values and lower variance are preferentially
    removed.  This is more defensible than removing whichever feature happens to
    appear later in the table.
    """
    report_columns = [
        "status",
        "removed_feature",
        "retained_feature",
        "correlation",
        "n_features",
        "max_features_for_correlation",
    ]
    if features.shape[1] <= 1:
        return features.copy(), pd.DataFrame(columns=report_columns)
    if features.shape[1] > max_features_for_correlation:
        if logger is not None:
            logger.warning(
                "Skipping full correlation filtering for %s features because this exceeds max_features_for_correlation=%s.",
                features.shape[1],
                max_features_for_correlation,
            )
        return features.copy(), pd.DataFrame.from_records(
            [
                {
                    "status": "skipped_too_many_features",
                    "removed_feature": "",
                    "retained_feature": "",
                    "correlation": np.nan,
                    "n_features": int(features.shape[1]),
                    "max_features_for_correlation": int(max_features_for_correlation),
                }
            ]
        )
    correlation = features.corr(method="pearson").abs()
    variances = features.var(axis=0, skipna=True).fillna(0)
    missing = features.isna().mean(axis=0)
    to_remove = set()
    records = []
    columns = list(features.columns)
    for i, first in enumerate(columns):
        if first in to_remove:
            continue
        for second in columns[i + 1:]:
            if second in to_remove:
                continue
            corr_value = correlation.loc[first, second]
            if pd.isna(corr_value) or corr_value <= max_absolute_correlation:
                continue
            first_score = (float(missing[first]), -float(variances[first]))
            second_score = (float(missing[second]), -float(variances[second]))
            if first_score >= second_score:
                removed, retained = first, second
            else:
                removed, retained = second, first
            to_remove.add(removed)
            records.append(
                {
                    "status": "removed",
                    "removed_feature": removed,
                    "retained_feature": retained,
                    "correlation": float(corr_value),
                    "n_features": int(features.shape[1]),
                    "max_features_for_correlation": int(max_features_for_correlation),
                    "removed_missing_fraction": float(missing[removed]),
                    "retained_missing_fraction": float(missing[retained]),
                    "removed_variance": float(variances[removed]),
                    "retained_variance": float(variances[retained]),
                }
            )
            if first in to_remove:
                break
    if logger is not None:
        logger.info("Removed %s highly correlated features", len(to_remove))
    retained_columns = [column for column in columns if column not in to_remove]
    return features.loc[:, retained_columns].copy(), pd.DataFrame.from_records(records)



def validate_final_feature_matrix(
    *,
    features: pd.DataFrame,
    context: str = "final feature matrix",
) -> pd.DataFrame:
    """Validate that a downstream feature matrix is finite and non-empty.

    Parameters
    ----------
    features:
        Numeric feature matrix to validate.
    context:
        Human-readable label used in error messages.

    Returns
    -------
    pandas.DataFrame
        One-row validation report.

    Raises
    ------
    ValueError
        If the matrix has no rows, no columns, missing values or non-finite
        values.
    """
    if features.shape[0] == 0:
        raise ValueError(f"{context} has no rows after preprocessing.")
    if features.shape[1] == 0:
        raise ValueError(f"{context} has no feature columns after preprocessing.")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    n_nan = int(np.isnan(values).sum())
    n_pos_inf = int(np.isposinf(values).sum())
    n_neg_inf = int(np.isneginf(values).sum())
    n_nonfinite = int((~np.isfinite(values)).sum())
    report = pd.DataFrame.from_records(
        [
            {
                "context": context,
                "n_rows": int(features.shape[0]),
                "n_features": int(features.shape[1]),
                "n_nan": n_nan,
                "n_positive_infinity": n_pos_inf,
                "n_negative_infinity": n_neg_inf,
                "n_nonfinite": n_nonfinite,
                "status": "ok" if n_nonfinite == 0 else "failed",
            }
        ]
    )
    if n_nonfinite:
        raise ValueError(
            f"{context} contains {n_nonfinite} non-finite values "
            f"({n_nan} NaN, {n_pos_inf} +inf, {n_neg_inf} -inf)."
        )
    return report


def aggregate_profiles(
    *,
    data_frame: pd.DataFrame,
    group_columns: Sequence[str],
    feature_columns: Sequence[str],
    statistic: str = "median",
) -> pd.DataFrame:
    """Aggregate object-level measurements to profile-level summaries."""
    valid_groups = [column for column in group_columns if column in data_frame.columns]
    valid_features = [column for column in feature_columns if column in data_frame.columns]
    if not valid_groups:
        raise ValueError("At least one valid group column is required.")
    if not valid_features:
        raise ValueError("At least one valid feature column is required.")
    if statistic == "median":
        aggregated = data_frame.groupby(valid_groups, dropna=False)[valid_features].median().reset_index()
    elif statistic == "mean":
        aggregated = data_frame.groupby(valid_groups, dropna=False)[valid_features].mean().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation statistic: {statistic}")
    counts = data_frame.groupby(valid_groups, dropna=False).size().reset_index(name="n_objects")
    return counts.merge(right=aggregated, on=valid_groups, how="left")


def summarise_feature_families(*, feature_names: Sequence[str]) -> pd.DataFrame:
    """Summarise retained features by broad CellProfiler-style family."""
    records = []
    for feature in feature_names:
        text = str(feature)
        lowered = text.lower()
        if "intensity" in lowered:
            family = "Intensity"
        elif "texture" in lowered:
            family = "Texture"
        elif "areashape" in lowered or "zernike" in lowered:
            family = "AreaShape"
        elif "granularity" in lowered:
            family = "Granularity"
        elif "radialdistribution" in lowered:
            family = "RadialDistribution"
        elif "correlation" in lowered:
            family = "Correlation"
        elif "neighbors" in lowered:
            family = "Neighbors"
        elif "location" in lowered:
            family = "Location"
        elif text.startswith("MissingIndicator__"):
            family = "MissingnessIndicator"
        else:
            family = "Other"
        parts = text.split("_")
        compartment = "unknown"
        for candidate in ["Cell", "Cells", "Nuclei", "Cytoplasm", "Image"]:
            if candidate.lower() in lowered:
                compartment = candidate
                break
        channel = parts[-1] if len(parts) > 1 else "unknown"
        records.append(
            {
                "feature": text,
                "feature_family": family,
                "feature_compartment": compartment,
                "feature_suffix": channel,
            }
        )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return pd.DataFrame(columns=["feature_family", "n_features"])
    return table.groupby("feature_family", dropna=False).size().reset_index(name="n_features")


def _decision_log(records: List[dict], step: str, message: str, **kwargs: object) -> None:
    """Append a standard preprocessing decision record."""
    record = {"step": step, "message": message}
    record.update(kwargs)
    records.append(record)


def preprocess_profiles(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    additional_metadata_columns: Optional[Sequence[str]] = None,
    max_feature_missing_fraction: float = 0.2,
    max_sample_missing_fraction: float = 0.5,
    min_feature_variance: float = 1e-12,
    min_unique_values: int = 2,
    max_zero_fraction: float = 1.0,
    remove_all_zero_rows: bool = True,
    all_zero_row_tolerance: float = 0.0,
    remove_correlated: bool = True,
    max_absolute_correlation: float = 0.95,
    max_features_for_correlation: int = 5000,
    imputation_method: str = "median",
    imputation_group_columns: Optional[Sequence[str]] = None,
    add_missing_indicators: bool = False,
    include_missing_indicators_in_correlation_filter: bool = False,
    max_missing_indicators: int = 500,
    minimum_missing_indicator_fraction: float = 0.0,
    scaling_method: str = "robust",
    standardise_metadata: bool = True,
    drop_unnamed_indexes: bool = True,
    include_qc_numeric_features: bool = False,
    winsorise_lower_quantile: Optional[float] = None,
    winsorise_upper_quantile: Optional[float] = None,
    reference_normalisation_method: str = "none",
    reference_column: Optional[str] = None,
    reference_values: Optional[Sequence[str]] = None,
    reference_group_columns: Optional[Sequence[str]] = None,
    batch_centering_method: str = "none",
    batch_centering_columns: Optional[Sequence[str]] = None,
    batch_correction_method: str = "none",
    batch_column: Optional[str] = None,
    batch_protect_columns: Optional[Sequence[str]] = None,
    batch_correction_min_batch_size: int = 3,
    replicate_group_columns: Optional[Sequence[str]] = None,
    batch_report_columns: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """Run a defensive generic Cell Painting preprocessing workflow."""
    validate_preprocessing_parameters(
        max_feature_missing_fraction=max_feature_missing_fraction,
        max_sample_missing_fraction=max_sample_missing_fraction,
        max_absolute_correlation=max_absolute_correlation,
        min_unique_values=min_unique_values,
        max_zero_fraction=max_zero_fraction,
        all_zero_row_tolerance=all_zero_row_tolerance,
    )
    decisions: List[dict] = []
    if logger is not None:
        logger.info("Starting CPATK preprocessing workflow")
        logger.info("Input table shape: rows=%s columns=%s", data_frame.shape[0], data_frame.shape[1])

    working = data_frame.copy()
    working, column_name_report = normalise_column_names(data_frame=working)
    _decision_log(decisions, "column_names", "Normalised column names", n_changed=int(column_name_report.get("changed", pd.Series(dtype=bool)).sum()))

    if drop_unnamed_indexes:
        working, dropped_index_report = drop_unnamed_index_columns(data_frame=working, logger=logger)
    else:
        dropped_index_report = pd.DataFrame(columns=["column", "reason", "looks_sequential", "dropped"])
    _decision_log(decisions, "index_columns", "Dropped likely accidental index columns", n_dropped=int(dropped_index_report.get("dropped", pd.Series(dtype=bool)).sum()))

    if standardise_metadata:
        working, metadata_alias_report = standardise_metadata_aliases(data_frame=working, logger=logger)
    else:
        metadata_alias_report = pd.DataFrame(columns=["canonical_column", "matched_source_column", "action"])
    _decision_log(decisions, "metadata_aliases", "Standardised common metadata aliases", n_actions=int(metadata_alias_report.shape[0]))

    column_role_report = assign_column_roles(
        data_frame=working,
        metadata_columns=metadata_columns,
        feature_columns=feature_columns,
        additional_metadata_columns=additional_metadata_columns,
        include_qc_numeric=include_qc_numeric_features,
    )
    metadata, features, metadata_names, feature_names = split_metadata_and_features(
        data_frame=working,
        metadata_columns=metadata_columns,
        feature_columns=feature_columns,
        additional_metadata_columns=additional_metadata_columns,
        include_qc_numeric=include_qc_numeric_features,
    )
    if logger is not None:
        logger.info("Detected %s metadata columns", len(metadata_names))
        logger.info("Detected %s feature columns", len(feature_names))
    _decision_log(decisions, "column_roles", "Separated metadata and feature columns", n_metadata=len(metadata_names), n_features=len(feature_names))
    if features.shape[1] == 0:
        raise ValueError(
            "No feature columns were detected. Supply --feature_columns explicitly, "
            "or review whether the input table is metadata-only or already reduced to non-CellProfiler measurements."
        )

    sample_qc_before_feature_qc = calculate_sample_qc(
        features=features,
        metadata=metadata.reset_index(drop=True),
    )
    _decision_log(
        decisions,
        "sample_qc_before_feature_qc",
        "Calculated profile/sample QC before feature-level filtering",
        n_rows=int(sample_qc_before_feature_qc.shape[0]),
    )

    features, nonfinite_report = replace_nonfinite_with_nan(features=features, logger=logger)
    _decision_log(
        decisions,
        "nonfinite_values",
        "Converted positive/negative infinity to missing values before QC",
        n_values_replaced=int(nonfinite_report.get("n_total_values_replaced", nonfinite_report.get("n_nonfinite_replaced", pd.Series(dtype=int))).sum()) if not nonfinite_report.empty else 0,
    )

    feature_qc = calculate_feature_qc(features=features)
    selected_features, feature_qc = select_features_by_qc(
        feature_qc=feature_qc,
        max_missing_fraction=max_feature_missing_fraction,
        min_variance=min_feature_variance,
        min_unique_values=min_unique_values,
        max_zero_fraction=max_zero_fraction,
    )
    if not selected_features:
        raise ValueError("No features passed preprocessing QC. Review metadata/feature columns or relax thresholds.")
    _decision_log(decisions, "feature_qc", "Applied feature missingness/variance/uniqueness filters", n_retained=len(selected_features), n_input=len(feature_names))

    selected_for_sample_qc = features.loc[:, selected_features]
    all_zero_row_report = calculate_all_zero_row_report(
        features=selected_for_sample_qc,
        metadata=metadata.reset_index(drop=True),
        tolerance=all_zero_row_tolerance,
    )
    if remove_all_zero_rows:
        keep_non_zero_rows = ~all_zero_row_report["all_zero_feature_row"].to_numpy(dtype=bool)
        all_zero_row_report.loc[all_zero_row_report["all_zero_feature_row"], "removed_by_all_zero_row_filter"] = True
        n_all_zero_removed = int((~keep_non_zero_rows).sum())
        metadata_for_sample_qc = metadata.loc[keep_non_zero_rows, :].reset_index(drop=True)
        selected_for_sample_qc = selected_for_sample_qc.loc[keep_non_zero_rows, :].reset_index(drop=True)
        _decision_log(
            decisions,
            "all_zero_row_filter",
            "Removed profiles whose observed retained feature values were all zero after merged-table feature QC",
            n_removed=n_all_zero_removed,
            tolerance=all_zero_row_tolerance,
        )
        if logger is not None and n_all_zero_removed:
            logger.warning(
                "Removed %s all-zero feature rows after merged profile construction and before imputation",
                n_all_zero_removed,
            )
    else:
        metadata_for_sample_qc = metadata.reset_index(drop=True)
        _decision_log(
            decisions,
            "all_zero_row_filter",
            "All-zero row filter disabled",
            n_flagged=int(all_zero_row_report["all_zero_feature_row"].sum()),
            tolerance=all_zero_row_tolerance,
        )

    sample_qc = calculate_sample_qc(features=selected_for_sample_qc, metadata=metadata_for_sample_qc)
    sample_qc = flag_samples_by_qc(sample_qc=sample_qc, max_missing_fraction=max_sample_missing_fraction)
    passed_rows = sample_qc["sample_qc_pass"].to_numpy(dtype=bool)
    selected_metadata = metadata_for_sample_qc.loc[passed_rows, :].reset_index(drop=True)
    selected_feature_matrix = selected_for_sample_qc.loc[passed_rows, :].reset_index(drop=True)
    _decision_log(decisions, "sample_qc", "Applied profile/sample missingness filter", n_retained=int(passed_rows.sum()), n_input=len(passed_rows))

    winsorised, winsorisation_report = winsorise_features(
        features=selected_feature_matrix,
        lower_quantile=winsorise_lower_quantile,
        upper_quantile=winsorise_upper_quantile,
        logger=logger,
    )
    if not winsorisation_report.empty:
        _decision_log(decisions, "winsorisation", "Clipped extreme values to requested quantiles", n_features=int(winsorisation_report.shape[0]))

    reference_control_qc_before_normalisation = calculate_reference_control_qc(
        features=winsorised,
        metadata=selected_metadata,
        reference_column=reference_column,
        reference_values=reference_values,
        group_columns=reference_group_columns,
        method=reference_normalisation_method,
    )
    normalised_for_imputation, reference_normalisation_report = normalise_features_to_reference(
        features=winsorised,
        metadata=selected_metadata,
        reference_column=reference_column,
        reference_values=reference_values,
        group_columns=reference_group_columns,
        method=reference_normalisation_method,
        logger=logger,
    )
    if reference_normalisation_method.lower() != "none":
        _decision_log(
            decisions,
            "reference_normalisation",
            "Normalised features to reference/control profiles before imputation so control statistics are not biased by imputed values",
            method=reference_normalisation_method,
        )

    indicators, indicator_report = add_missingness_indicators(
        features=normalised_for_imputation,
        max_indicators=max_missing_indicators,
        minimum_missing_fraction=minimum_missing_indicator_fraction,
        logger=logger,
    ) if add_missing_indicators else (
        pd.DataFrame(index=normalised_for_imputation.index),
        pd.DataFrame(columns=["source_feature", "indicator_feature", "missing_fraction"]),
    )
    if not indicator_report.empty:
        indicator_report["feature_role"] = "missingness_indicator"
        indicator_report["included_in_correlation_filter"] = bool(include_missing_indicators_in_correlation_filter)

    imputed = impute_features(
        features=normalised_for_imputation,
        method=imputation_method,
        metadata=selected_metadata,
        group_columns=imputation_group_columns,
        logger=logger,
    )
    imputation_report = summarise_imputation(before=normalised_for_imputation, after=imputed, method=imputation_method)
    _decision_log(
        decisions,
        "imputation",
        "Imputed remaining missing feature values after optional reference normalisation",
        method=imputation_method,
        n_missing_before=int(normalised_for_imputation.isna().sum().sum()),
        n_missing_after=int(imputed.isna().sum().sum()),
    )

    batch_centered, batch_centering_report = batch_center_features(
        features=imputed,
        metadata=selected_metadata,
        batch_columns=batch_centering_columns,
        method=batch_centering_method,
        logger=logger,
    )
    if batch_centering_method.lower() != "none":
        _decision_log(decisions, "batch_centering", "Applied optional batch centering", method=batch_centering_method)

    batch_corrected, batch_correction_report, batch_confounding_report = combat_style_location_scale_correction(
        features=batch_centered,
        metadata=selected_metadata,
        batch_column=batch_column,
        protected_columns=batch_protect_columns,
        method=batch_correction_method,
        min_batch_size=batch_correction_min_batch_size,
        logger=logger,
    )
    if batch_correction_method.lower() != "none":
        _decision_log(
            decisions,
            "batch_correction",
            "Applied optional ComBat-style location/scale batch correction after imputation and before final scaling",
            method=batch_correction_method,
            batch_column=str(batch_column),
        )

    before_after_qc = make_before_after_qc_reports(
        before_features=batch_centered,
        after_features=batch_corrected,
        metadata=selected_metadata,
        replicate_group_columns=replicate_group_columns,
        batch_report_columns=batch_report_columns,
    )

    scaled_biological = scale_features(features=batch_corrected, method=scaling_method, logger=logger)
    _decision_log(decisions, "scaling", "Scaled biological Cell Painting features", method=scaling_method)

    if include_missing_indicators_in_correlation_filter:
        features_for_correlation = pd.concat(
            [scaled_biological.reset_index(drop=True), indicators.reset_index(drop=True)],
            axis=1,
        )
    else:
        features_for_correlation = scaled_biological.copy()

    if remove_correlated:
        filtered_features, correlation_report = remove_correlated_features(
            features=features_for_correlation,
            max_absolute_correlation=max_absolute_correlation,
            max_features_for_correlation=max_features_for_correlation,
            logger=logger,
        )
        n_removed_corr = int((correlation_report.get("status", pd.Series(dtype=str)) == "removed").sum())
        _decision_log(
            decisions,
            "correlation_filter",
            "Removed redundant highly correlated biological features",
            n_removed=n_removed_corr,
        )
    else:
        filtered_features = features_for_correlation.copy()
        correlation_report = pd.DataFrame(columns=["status", "removed_feature", "retained_feature", "correlation"])

    if include_missing_indicators_in_correlation_filter:
        final_features = filtered_features.copy()
    else:
        final_features = pd.concat(
            [filtered_features.reset_index(drop=True), indicators.reset_index(drop=True)],
            axis=1,
        )

    final_matrix_validation = validate_final_feature_matrix(
        features=final_features,
        context="preprocessed feature matrix",
    )
    preprocessed = pd.concat([selected_metadata.reset_index(drop=True), final_features.reset_index(drop=True)], axis=1)
    imputed_unscaled = pd.concat([selected_metadata.reset_index(drop=True), imputed.reset_index(drop=True)], axis=1)
    retained_features = pd.DataFrame(
        {
            "feature": final_features.columns.tolist(),
            "feature_role": [
                "missingness_indicator" if str(feature).startswith("MissingIndicator__") else "biological_feature"
                for feature in final_features.columns
            ],
        }
    )
    feature_family_summary = summarise_feature_families(feature_names=final_features.columns.tolist())

    config = pd.DataFrame.from_records(
        [
            {"parameter": "imputation_method", "value": imputation_method},
            {"parameter": "scaling_method", "value": scaling_method},
            {"parameter": "remove_correlated", "value": str(remove_correlated)},
            {"parameter": "max_absolute_correlation", "value": str(max_absolute_correlation)},
            {"parameter": "max_zero_fraction", "value": str(max_zero_fraction)},
            {"parameter": "remove_all_zero_rows", "value": str(remove_all_zero_rows)},
            {"parameter": "all_zero_row_tolerance", "value": str(all_zero_row_tolerance)},
            {"parameter": "reference_normalisation_method", "value": reference_normalisation_method},
            {"parameter": "reference_column", "value": str(reference_column)},
            {"parameter": "reference_values", "value": ";".join(reference_values or [])},
            {"parameter": "batch_centering_method", "value": batch_centering_method},
            {"parameter": "batch_correction_method", "value": batch_correction_method},
            {"parameter": "batch_column", "value": str(batch_column)},
            {"parameter": "batch_protect_columns", "value": ";".join(batch_protect_columns or [])},
            {"parameter": "batch_correction_min_batch_size", "value": str(batch_correction_min_batch_size)},
            {"parameter": "replicate_group_columns", "value": ";".join(replicate_group_columns or [])},
            {"parameter": "batch_report_columns", "value": ";".join(batch_report_columns or [])},
            {"parameter": "max_features_for_correlation", "value": str(max_features_for_correlation)},
            {"parameter": "include_qc_numeric_features", "value": str(include_qc_numeric_features)},
            {"parameter": "include_missing_indicators_in_correlation_filter", "value": str(include_missing_indicators_in_correlation_filter)},
        ]
    )
    summary = pd.DataFrame.from_records(
        [
            {"item": "n_rows_input", "value": int(data_frame.shape[0])},
            {"item": "n_columns_input", "value": int(data_frame.shape[1])},
            {"item": "n_rows_passing_qc", "value": int(preprocessed.shape[0])},
            {"item": "n_metadata_columns", "value": int(len(metadata_names))},
            {"item": "n_features_input", "value": int(len(feature_names))},
            {"item": "n_features_after_qc", "value": int(len(selected_features))},
            {"item": "n_nonfinite_or_extreme_feature_values_replaced", "value": int(nonfinite_report.get("n_total_values_replaced", nonfinite_report.get("n_nonfinite_replaced", pd.Series(dtype=int))).sum()) if not nonfinite_report.empty else 0},
            {"item": "n_all_zero_feature_rows_flagged", "value": int(all_zero_row_report["all_zero_feature_row"].sum())},
            {"item": "n_all_zero_feature_rows_removed", "value": int(all_zero_row_report["removed_by_all_zero_row_filter"].sum())},
            {"item": "n_missing_feature_values_before_imputation", "value": int(winsorised.isna().sum().sum())},
            {"item": "n_missing_feature_values_after_preprocessing", "value": int(final_features.isna().sum().sum())},
            {"item": "n_missing_indicator_features_added", "value": int(indicators.shape[1])},
            {"item": "n_features_after_correlation_filter", "value": int(len(final_features.columns))},
            {"item": "n_correlated_features_removed", "value": int((correlation_report.get("status", pd.Series(dtype=str)) == "removed").sum())},
            {"item": "n_excluded_numeric_qc_or_provenance_columns", "value": int((column_role_report["role"] == "excluded_numeric_qc_or_provenance").sum())},
            {"item": "imputation_method", "value": imputation_method},
            {"item": "imputation_group_columns", "value": ";".join(imputation_group_columns or [])},
            {"item": "scaling_method", "value": scaling_method},
            {"item": "reference_normalisation_method", "value": reference_normalisation_method},
            {"item": "batch_centering_method", "value": batch_centering_method},
            {"item": "batch_correction_method", "value": batch_correction_method},
            {"item": "batch_column", "value": str(batch_column)},
            {"item": "max_feature_missing_fraction", "value": max_feature_missing_fraction},
            {"item": "max_sample_missing_fraction", "value": max_sample_missing_fraction},
            {"item": "max_absolute_correlation", "value": max_absolute_correlation},
            {"item": "max_zero_fraction", "value": max_zero_fraction},
            {"item": "remove_all_zero_rows", "value": str(remove_all_zero_rows)},
            {"item": "all_zero_row_tolerance", "value": all_zero_row_tolerance},
        ]
    )
    return {
        "preprocessed": preprocessed,
        "imputed_unscaled_features_with_metadata": imputed_unscaled,
        "feature_qc": feature_qc,
        "sample_qc_before_feature_qc": sample_qc_before_feature_qc,
        "sample_qc_after_feature_qc": sample_qc,
        "sample_qc": sample_qc,
        "all_zero_row_report": all_zero_row_report,
        "imputation_report": imputation_report,
        "missingness_indicator_report": indicator_report,
        "metadata_alias_report": metadata_alias_report,
        "column_name_report": column_name_report,
        "column_role_report": column_role_report,
        "dropped_index_column_report": dropped_index_report,
        "nonfinite_value_report": nonfinite_report,
        "winsorisation_report": winsorisation_report,
        "reference_control_qc_before_normalisation": reference_control_qc_before_normalisation,
        "reference_normalisation_report": reference_normalisation_report,
        "batch_centering_report": batch_centering_report,
        "batch_correction_report": batch_correction_report,
        "batch_confounding_report": batch_confounding_report,
        "before_after_replicate_correlations": before_after_qc["before_after_replicate_correlations"],
        "before_after_replicate_summary": before_after_qc["before_after_replicate_summary"],
        "before_after_batch_pc_association": before_after_qc["before_after_batch_pc_association"],
        "correlation_filter_report": correlation_report,
        "final_matrix_validation": final_matrix_validation,
        "retained_features": retained_features,
        "feature_family_summary": feature_family_summary,
        "preprocessing_config": config,
        "preprocessing_decision_log": pd.DataFrame.from_records(decisions),
        "preprocessing_summary": summary,
    }
