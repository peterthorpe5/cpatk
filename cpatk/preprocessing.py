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
        imputer = KNNImputer(n_neighbors=n_neighbors)
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
                    "reference_values": ",".join(sorted(ref_values)),
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
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove one feature from each highly correlated pair.

    Features with more missing values and lower variance are preferentially
    removed.  This is more defensible than removing whichever feature happens to
    appear later in the table.
    """
    if features.shape[1] <= 1:
        return features.copy(), pd.DataFrame(columns=["removed_feature", "retained_feature", "correlation"])
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
                    "removed_feature": removed,
                    "retained_feature": retained,
                    "correlation": float(corr_value),
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
    imputation_method: str = "median",
    imputation_group_columns: Optional[Sequence[str]] = None,
    add_missing_indicators: bool = False,
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

    imputed = impute_features(
        features=winsorised,
        method=imputation_method,
        metadata=selected_metadata,
        group_columns=imputation_group_columns,
        logger=logger,
    )
    imputation_report = summarise_imputation(before=winsorised, after=imputed, method=imputation_method)
    _decision_log(decisions, "imputation", "Imputed remaining missing feature values", method=imputation_method, n_missing_before=int(winsorised.isna().sum().sum()), n_missing_after=int(imputed.isna().sum().sum()))

    indicators, indicator_report = add_missingness_indicators(
        features=winsorised,
        max_indicators=max_missing_indicators,
        minimum_missing_fraction=minimum_missing_indicator_fraction,
        logger=logger,
    ) if add_missing_indicators else (
        pd.DataFrame(index=winsorised.index),
        pd.DataFrame(columns=["source_feature", "indicator_feature", "missing_fraction"]),
    )

    normalised, reference_normalisation_report = normalise_features_to_reference(
        features=imputed,
        metadata=selected_metadata,
        reference_column=reference_column,
        reference_values=reference_values,
        group_columns=reference_group_columns,
        method=reference_normalisation_method,
        logger=logger,
    )
    if reference_normalisation_method.lower() != "none":
        _decision_log(decisions, "reference_normalisation", "Normalised features to reference/control profiles", method=reference_normalisation_method)

    batch_centered, batch_centering_report = batch_center_features(
        features=normalised,
        metadata=selected_metadata,
        batch_columns=batch_centering_columns,
        method=batch_centering_method,
        logger=logger,
    )
    if batch_centering_method.lower() != "none":
        _decision_log(decisions, "batch_centering", "Applied optional batch centering", method=batch_centering_method)

    with_indicators = pd.concat([batch_centered.reset_index(drop=True), indicators.reset_index(drop=True)], axis=1)
    scaled = scale_features(features=with_indicators, method=scaling_method, logger=logger)
    _decision_log(decisions, "scaling", "Scaled features", method=scaling_method)

    if remove_correlated:
        final_features, correlation_report = remove_correlated_features(
            features=scaled,
            max_absolute_correlation=max_absolute_correlation,
            logger=logger,
        )
        _decision_log(decisions, "correlation_filter", "Removed redundant highly correlated features", n_removed=int(correlation_report.shape[0]))
    else:
        final_features = scaled.copy()
        correlation_report = pd.DataFrame(columns=["removed_feature", "retained_feature", "correlation"])

    preprocessed = pd.concat([selected_metadata.reset_index(drop=True), final_features.reset_index(drop=True)], axis=1)
    imputed_unscaled = pd.concat([selected_metadata.reset_index(drop=True), imputed.reset_index(drop=True)], axis=1)
    retained_features = pd.DataFrame({"feature": final_features.columns.tolist()})
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
            {"parameter": "reference_values", "value": ",".join(reference_values or [])},
            {"parameter": "batch_centering_method", "value": batch_centering_method},
            {"parameter": "include_qc_numeric_features", "value": str(include_qc_numeric_features)},
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
            {"item": "n_correlated_features_removed", "value": int(correlation_report.shape[0])},
            {"item": "n_excluded_numeric_qc_or_provenance_columns", "value": int((column_role_report["role"] == "excluded_numeric_qc_or_provenance").sum())},
            {"item": "imputation_method", "value": imputation_method},
            {"item": "imputation_group_columns", "value": ",".join(imputation_group_columns or [])},
            {"item": "scaling_method", "value": scaling_method},
            {"item": "reference_normalisation_method", "value": reference_normalisation_method},
            {"item": "batch_centering_method", "value": batch_centering_method},
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
        "reference_normalisation_report": reference_normalisation_report,
        "batch_centering_report": batch_centering_report,
        "correlation_filter_report": correlation_report,
        "retained_features": retained_features,
        "feature_family_summary": feature_family_summary,
        "preprocessing_config": config,
        "preprocessing_decision_log": pd.DataFrame.from_records(decisions),
        "preprocessing_summary": summary,
    }
