"""Regression tests for CPATK v0.2.31 synthetic validation suite."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cpatk.synthetic_latent import (
    COMPREHENSIVE_SYNTHETIC_SCENARIOS,
    LatentBenchmarkConfig,
    SyntheticCellPaintingConfig,
    config_for_scenario,
    generate_synthetic_cell_painting_profiles,
    run_synthetic_latent_benchmark,
    summarise_comprehensive_validation,
)


class SyntheticComprehensiveValidationTests(unittest.TestCase):
    """Test the comprehensive synthetic validation additions."""

    def test_comprehensive_scenarios_are_registered(self) -> None:
        """Comprehensive mode should include the hard stress scenarios."""
        expected = {
            "label_noise_25pct",
            "high_missingness",
            "missing_compartment",
            "segmentation_outliers",
            "compound_plate_confounding",
            "no_biology_high_batch_negative_control",
        }
        self.assertTrue(expected.issubset(set(COMPREHENSIVE_SYNTHETIC_SCENARIOS)))

    def test_perturbation_flags_and_training_labels_are_created(self) -> None:
        """Synthetic generator should expose noisy-label and artefact flags."""
        base = SyntheticCellPaintingConfig(
            scenario_name="label_noise_25pct",
            n_compounds=8,
            n_moa_classes=2,
            n_batches=2,
            n_datasets=2,
            replicates_per_compound_dataset=3,
            n_features=24,
            n_informative_features=12,
            random_state=7,
        )
        config = config_for_scenario(scenario_name="label_noise_25pct", base_config=base)
        profiles, _truth, _scenario = generate_synthetic_cell_painting_profiles(config=config)
        self.assertIn("training_cpd_id", profiles.columns)
        self.assertIn("synthetic_label_noised", profiles.columns)
        self.assertGreater(int(profiles["synthetic_label_noised"].sum()), 0)
        self.assertTrue((profiles.loc[profiles["synthetic_label_noised"], "training_cpd_id"] != profiles.loc[profiles["synthetic_label_noised"], "cpd_id"]).all())

    def test_summarise_comprehensive_validation_negative_control(self) -> None:
        """Decision summary should fail high held-out signal in a negative control."""
        metrics = pd.DataFrame(
            [
                {
                    "scenario": "no_biology_negative_control",
                    "seed": 1,
                    "method": "cpatk_contrastive",
                    "metric": "validation_to_train_top1_same_compound_rate",
                    "value": 0.25,
                },
                {
                    "scenario": "no_biology_negative_control",
                    "seed": 1,
                    "method": "raw_scaled_features",
                    "metric": "validation_to_train_top1_same_compound_rate",
                    "value": 0.0,
                },
                {
                    "scenario": "no_biology_negative_control",
                    "seed": 1,
                    "method": "pca",
                    "metric": "validation_to_train_top1_same_compound_rate",
                    "value": 0.0,
                },
            ]
        )
        pass_fail = pd.DataFrame(
            [
                {
                    "scenario": "no_biology_negative_control",
                    "seed": 1,
                    "check": "native_validation_to_train_top1_same_compound_rate",
                    "passed": False,
                }
            ]
        )
        summary = summarise_comprehensive_validation(metrics=metrics, pass_fail=pass_fail)
        self.assertEqual(summary.loc[0, "decision"], "fail")

    def test_quick_multiseed_benchmark_without_native(self) -> None:
        """Benchmark runner should support repeated seeds and decision outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LatentBenchmarkConfig(
                output_dir=Path(temp_dir),
                scenarios=["clean_biology", "no_biology_negative_control"],
                n_compounds=8,
                n_moa_classes=2,
                n_batches=2,
                n_datasets=2,
                replicates_per_compound_dataset=2,
                n_features=32,
                n_informative_features=12,
                latent_dim=4,
                epochs=2,
                batch_size=32,
                steps_per_epoch=1,
                validation_fraction=0.25,
                random_state=11,
                seed_values=[11, 12],
                threads=1,
                run_native_contrastive=False,
                run_pca=True,
            )
            outputs = run_synthetic_latent_benchmark(config=config)
            self.assertIn("synthetic_validation_decision_summary", outputs)
            manifest = outputs["synthetic_benchmark_manifest"]
            self.assertEqual(manifest.shape[0], 4)
            self.assertTrue((Path(temp_dir) / "synthetic_validation_decision_summary.tsv").exists())


if __name__ == "__main__":
    unittest.main()
