# CPATK correlation filtering

CPATK removes redundant features after the profile table has been built, QC has been applied, missing values have been handled, optional reference normalisation has been applied and features have been scaled.

The default correlation filter is now:

```text
correlation_method = spearman
correlation_filter_strategy = variance
```

This matches the practical behaviour used in the earlier Cell Painting preprocessing scripts: first remove globally low-variance features, then calculate a full all-vs-all absolute correlation matrix, then remove redundant features from highly correlated sets while keeping the highest-variance representative. Spearman correlation is the default because Cell Painting features are often non-normal and monotonic rather than strictly linearly related.

## Why keep the highest-variance feature?

If two or more features carry nearly the same information, keeping all of them gives downstream PCA, nearest-neighbour, MOA and ML steps unnecessary duplicate signal. Retaining the feature with the greatest variance is a simple, auditable way to keep the representative that still contains the most spread across profiles after normalisation and scaling.

This does not replace the global variance filter. The two filters do different things:

1. the variance filter removes near-constant features that carry little information;
2. the correlation filter removes redundant copies of informative features.

## Options

`cpatk-preprocess` exposes:

```text
--max_absolute_correlation
--max_features_for_correlation
--correlation_method {pearson,spearman,kendall}
--correlation_filter_strategy {variance,min_redundancy,table_order}
```

Recommended default for general Cell Painting:

```text
--correlation_method spearman
--correlation_filter_strategy variance
```

Use `pearson` only when linear relationships are specifically desired. Use `kendall` only for small datasets because it can be slow. Use `min_redundancy` as a sensitivity analysis if you want to retain features with lower average correlation to all other features rather than the highest-variance representative.

## Audit outputs

The correlation filter report records the method, strategy, removed feature, retained feature, correlation value, missing fractions, variances and priority ranks. This means the user can see which feature was dropped and why.
