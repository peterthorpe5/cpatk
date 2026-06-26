# CPATK v0.2.18 review hardening

This is a critical-review patch after the larger v0.2.17 reporting and explainability update.

## Main decisions confirmed

- CLIPn can accept real literal zero values.
- CPATK should not epsilon-replace zeros before CLIPn.
- Missing, NA, NaN, infinity and non-finite values are the values that must be cleaned before CLIPn.
- All-zero rows and all-zero features are still removed by default as empty-signal QC, not because CLIPn cannot accept zeros.
- MOA on preprocessed feature space remains the main MOA route.
- If CLIPn runs and writes a latent table, the stress-test shells now also run a separate latent-space MOA analysis.
- Latent-space MOA is labelled separately and should not replace feature-space MOA.

## Fixes in this pass

- Removed the deprecated `--clipn_zero_epsilon` argument from the recommended malaria and mitotox stress-test shells.
- Updated CLIPn report wording so it no longer implies that literal zeros are removed for CLIPn compatibility.
- Added clearer query-vs-background SHAP bookkeeping in `query_neighbourhood_summary.tsv`.
- Closed old logging handlers when reconfiguring the CPATK logger, avoiding leaked file-handler warnings during repeated in-process CLI tests.
- Added unit tests for the reviewed CLIPn zero policy, logging cleanup and query-vs-background summary metadata.

## Tests

The v0.2.18 tests include the v0.2.17 behaviour plus extra review-hardening tests. The package still needs to be rerun on the malaria and mitotox cluster data, because those are the real stress tests for run time and filesystem behaviour.
