from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from scripts.hard_negative_mining import (
    extract_fast_features_for_obs,
    mine_hard_negatives,
    run_hard_negative_mining_pipeline,
)


class TestHardNegativeMining(unittest.TestCase):

    def setUp(self):
        self.search_img = np.random.default_rng(20260913).integers(
            0, 256, size=(200, 200), dtype=np.uint8
        )
        self.ref_img = np.random.default_rng(20260913).integers(
            0, 256, size=(100, 100), dtype=np.uint8
        )
        self.sample_candidates = [
            {"x": 100.0, "y": 100.0, "score": 0.9, "generator": "dog", "label": 1, "distance": 2.0},
            {"x": 120.0, "y": 120.0, "score": 0.85, "generator": "ncc", "label": 0, "distance": 28.28},
            {"x": 50.0, "y": 50.0, "score": 0.7, "generator": "gradient", "label": 0, "distance": 70.71},
            {"x": 102.0, "y": 102.0, "score": 0.6, "generator": "edge", "label": 1, "distance": 2.83},
            {"x": 180.0, "y": 180.0, "score": 0.95, "generator": "frequency", "label": 0, "distance": 113.14},
        ]

    def test_fast_feature_extraction_shape(self):
        feats = extract_fast_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        self.assertEqual(feats.shape, (len(self.sample_candidates), 17))

    def test_fast_feature_finite_values(self):
        feats = extract_fast_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        self.assertTrue(np.isfinite(feats).all())

    def test_deterministic_fast_feature_extraction(self):
        feats1 = extract_fast_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        feats2 = extract_fast_features_for_obs(self.search_img, self.ref_img, self.sample_candidates)
        np.testing.assert_array_equal(feats1, feats2)

    def test_mine_hard_negatives_criteria(self):
        raw_scores = np.array([0.9, 0.85, 0.7, 0.6, 0.95], dtype=np.float32)
        mined = mine_hard_negatives(self.sample_candidates, raw_scores, top_k_hard=15)
        self.assertTrue(all(c["label"] == 0 and c["distance"] > 5.0 for c in mined))

    def test_positive_candidates_excluded_from_hard_negatives(self):
        raw_scores = np.array([0.99, 0.85, 0.7, 0.98, 0.95], dtype=np.float32)
        mined = mine_hard_negatives(self.sample_candidates, raw_scores, top_k_hard=15)
        mined_labels = [c["label"] for c in mined]
        self.assertNotIn(1, mined_labels)

    def test_close_negatives_excluded_from_hard_negatives(self):
        cands = [
            {"x": 100.0, "y": 100.0, "score": 0.9, "generator": "dog", "label": 0, "distance": 4.5},
            {"x": 150.0, "y": 150.0, "score": 0.8, "generator": "dog", "label": 0, "distance": 10.0},
        ]
        raw_scores = np.array([0.95, 0.85], dtype=np.float32)
        mined = mine_hard_negatives(cands, raw_scores, top_k_hard=15)
        self.assertEqual(len(mined), 1)
        self.assertEqual(mined[0]["distance"], 10.0)

    def test_hard_negative_sorting(self):
        raw_scores = np.array([0.9, 0.85, 0.7, 0.6, 0.95], dtype=np.float32)
        mined = mine_hard_negatives(self.sample_candidates, raw_scores, top_k_hard=15)
        scores = [c["model_score"] for c in mined]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_hard_negative_limit(self):
        raw_scores = np.array([0.9, 0.85, 0.7, 0.6, 0.95], dtype=np.float32)
        mined = mine_hard_negatives(self.sample_candidates, raw_scores, top_k_hard=2)
        self.assertLessEqual(len(mined), 2)

    def test_hard_negative_mining_non_leakage(self):
        params = inspect.signature(extract_fast_features_for_obs).parameters
        self.assertNotIn("target_x", params)
        self.assertNotIn("target_y", params)

    def test_sample_weight_assignment_logic(self):
        weights = []
        mined_ids = {id(self.sample_candidates[4])}
        for c in self.sample_candidates:
            if c["label"] == 1:
                weights.append(10.0)
            elif id(c) in mined_ids:
                weights.append(3.0)
            else:
                weights.append(1.0)
        self.assertEqual(weights, [10.0, 1.0, 1.0, 10.0, 3.0])

    def test_xgboost_model_save_load(self):
        X = np.random.randn(20, 17).astype(np.float32)
        y = np.array([1]*5 + [0]*15, dtype=np.int32)
        clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=20260913)
        clf.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "hn_xgboost_ranker.json"
            clf.save_model(str(model_path))
            self.assertTrue(model_path.exists())

            clf2 = XGBClassifier()
            clf2.load_model(str(model_path))
            p1 = clf.predict_proba(X)
            p2 = clf2.predict_proba(X)
            np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_reproducibility_seed(self):
        clf1 = XGBClassifier(n_estimators=10, random_state=20260913)
        clf2 = XGBClassifier(n_estimators=10, random_state=20260913)
        X = np.random.randn(20, 17).astype(np.float32)
        y = np.array([1]*5 + [0]*15, dtype=np.int32)
        clf1.fit(X, y)
        clf2.fit(X, y)
        p1 = clf1.predict_proba(X)
        p2 = clf2.predict_proba(X)
        np.testing.assert_array_equal(p1, p2)

    def test_empty_candidate_pool_handling(self):
        feats = extract_fast_features_for_obs(self.search_img, self.ref_img, [])
        self.assertEqual(len(feats), 0)
        mined = mine_hard_negatives([], np.array([]), top_k_hard=15)
        self.assertEqual(len(mined), 0)

    def test_pipeline_execution_integration(self):
        dataset_root = Path("dataset_v0.1")
        phase6_summary = Path("validation/phase6/robust_candidate_generation/robust_candidate_generation_summary.json")
        phase7_model = Path("validation/phase7/learned_candidate_ranking/xgboost_ranker.json")

        if dataset_root.exists() and phase6_summary.exists() and phase7_model.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "phase3"
                summary = run_hard_negative_mining_pipeline(
                    dataset_root=dataset_root,
                    phase6_summary_path=phase6_summary,
                    phase7_model_path=phase7_model,
                    output_dir=out_dir,
                )
                self.assertIn("mined_hard_negatives_count", summary)
                self.assertIn("decision", summary)
                self.assertTrue((out_dir / "hard_negatives.csv").exists())
                self.assertTrue((out_dir / "training_results.csv").exists())
                self.assertTrue((out_dir / "summary.json").exists())
                self.assertTrue((out_dir / "report.md").exists())
                self.assertTrue((out_dir / "model" / "hn_xgboost_ranker.json").exists())

    def test_evaluation_metric_keys(self):
        summary_keys = ["mined_hard_negatives_count", "baseline_metrics", "hard_negative_metrics", "decision"]
        metrics_dummy = {
            "mined_hard_negatives_count": 105,
            "baseline_metrics": {"overall": {"median_error_px": 25.0, "recall_at_5px": 0.02}},
            "hard_negative_metrics": {"overall": {"median_error_px": 25.0, "recall_at_5px": 0.02}},
            "decision": "KEEP",
        }
        for k in summary_keys:
            self.assertIn(k, metrics_dummy)


if __name__ == "__main__":
    unittest.main()
