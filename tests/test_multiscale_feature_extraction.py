from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from scripts.multiscale_feature_extraction import (
    MULTISCALE_FEATURE_NAMES,
    extract_multiscale_features_for_obs,
    run_multiscale_pipeline,
)


class TestMultiscaleFeatureExtraction(unittest.TestCase):

    def setUp(self):
        self.search_img = np.random.default_rng(20260913).integers(
            0, 256, size=(200, 200), dtype=np.uint8
        )
        self.ref_img = np.random.default_rng(20260913).integers(
            0, 256, size=(100, 100), dtype=np.uint8
        )
        self.sample_candidates = [
            {"x": 100.0, "y": 100.0, "score": 0.9, "generator": "dog"},
            {"x": 120.0, "y": 120.0, "score": 0.85, "generator": "ncc"},
            {"x": 50.0, "y": 50.0, "score": 0.7, "generator": "gradient"},
        ]

    def test_multiscale_feature_schema(self):
        feats = extract_multiscale_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        self.assertEqual(len(MULTISCALE_FEATURE_NAMES), 33)
        self.assertEqual(feats.shape, (len(self.sample_candidates), 33))

    def test_multiscale_feature_values_are_finite(self):
        feats = extract_multiscale_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        self.assertTrue(np.isfinite(feats).all())

    def test_deterministic_multiscale_feature_generation(self):
        feats1 = extract_multiscale_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        feats2 = extract_multiscale_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        np.testing.assert_array_equal(feats1, feats2)

    def test_target_coordinate_non_leakage(self):
        params = inspect.signature(extract_multiscale_features_for_obs).parameters
        self.assertNotIn("target_x", params)
        self.assertNotIn("target_y", params)
        self.assertNotIn("gt_x", params)
        self.assertNotIn("gt_y", params)

    def test_empty_candidate_pool_handling(self):
        feats = extract_multiscale_features_for_obs(self.search_img, self.ref_img, [])
        self.assertEqual(len(feats), 0)

    def test_reproducibility_seed(self):
        clf1 = XGBClassifier(n_estimators=10, random_state=20260913)
        clf2 = XGBClassifier(n_estimators=10, random_state=20260913)
        X = np.random.randn(20, 33).astype(np.float32)
        y = np.array([1]*5 + [0]*15, dtype=np.int32)
        clf1.fit(X, y)
        clf2.fit(X, y)
        p1 = clf1.predict_proba(X)
        p2 = clf2.predict_proba(X)
        np.testing.assert_array_equal(p1, p2)

    def test_multiscale_pipeline_integration(self):
        dataset_root = Path("dataset_v0.1")
        phase6_summary = Path("validation/phase6/robust_candidate_generation/robust_candidate_generation_summary.json")

        if dataset_root.exists() and phase6_summary.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "multiscale"
                summary = run_multiscale_pipeline(
                    dataset_root=dataset_root,
                    phase6_summary_path=phase6_summary,
                    output_dir=out_dir,
                )
                self.assertIn("feature_count", summary)
                self.assertEqual(summary["feature_count"], 33)
                self.assertIn("decision", summary)
                self.assertTrue((out_dir / "feature_ablation.csv").exists())
                self.assertTrue((out_dir / "importance.json").exists())
                self.assertTrue((out_dir / "summary.json").exists())
                self.assertTrue((out_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
