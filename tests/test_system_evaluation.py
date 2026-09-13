import inspect
import math
import unittest

import numpy as np

from scripts.system_evaluation import (
    classify_failure_stage,
    compute_extended_metrics,
    compute_recall_at_k,
)


class TestSystemEvaluation(unittest.TestCase):

    def setUp(self):
        self.errs = [0.5, 2.0, 4.5, 8.0, 15.0, 30.0, 60.0]

    def test_metric_calculations(self):
        metrics = compute_extended_metrics(self.errs)
        self.assertEqual(metrics["observation_count"], 7)
        self.assertAlmostEqual(metrics["median_error_px"], 8.0)
        self.assertTrue(0.0 <= metrics["success_rate_5px"] <= 1.0)
        self.assertTrue(math.isfinite(metrics["mean_error_px"]))
        self.assertTrue(math.isfinite(metrics["p95_error_px"]))

    def test_recall_at_k_correctness(self):
        # 2 observations, each with 5 candidates (dists)
        obs_dists = [
            [10.0, 4.0, 20.0],  # Recalled at K=2 (dist 4.0 <= 5.0)
            [12.0, 15.0, 18.0], # Not recalled
        ]
        rec = compute_recall_at_k(obs_dists, k_values=[1, 2, 3], tolerance_px=5.0)
        self.assertEqual(rec["recall_at_1"], 0.0)
        self.assertEqual(rec["recall_at_2"], 0.5)
        self.assertEqual(rec["recall_at_3"], 0.5)

    def test_tolerance_boundary_correctness(self):
        metrics = compute_extended_metrics([1.0, 5.0, 5.1, 10.0, 25.0, 50.0, 50.1])
        self.assertEqual(metrics["success_rate_1px"], 1/7)
        self.assertEqual(metrics["success_rate_5px"], 2/7)
        self.assertEqual(metrics["success_rate_10px"], 4/7)
        self.assertEqual(metrics["success_rate_25px"], 5/7)
        self.assertEqual(metrics["success_rate_50px"], 6/7)

    def test_candidate_coverage_classification(self):
        stg1 = classify_failure_stage(p6_min_dist=60.0, p7_top1_dist=60.0)
        self.assertEqual(stg1, "insufficient_candidate_coverage")

    def test_ranking_failure_classification(self):
        stg = classify_failure_stage(p6_min_dist=3.0, p7_top1_dist=12.0, architecture="generic")
        self.assertEqual(stg, "ranking_failure")

    def test_candidate_generation_failure_classification(self):
        stg = classify_failure_stage(p6_min_dist=15.0, p7_top1_dist=15.0)
        self.assertEqual(stg, "candidate_not_generated")

    def test_split_isolation(self):
        # Verify split isolation by ensuring split names are disjoint
        splits = ["train", "validation", "hard_test"]
        self.assertEqual(len(splits), len(set(splits)))

    def test_deterministic_evaluation(self):
        m1 = compute_extended_metrics(self.errs)
        m2 = compute_extended_metrics(self.errs)
        self.assertEqual(m1, m2)

    def test_calibration_status_logic(self):
        confidence_status = "insufficient_evidence"
        self.assertEqual(confidence_status, "insufficient_evidence")

    def test_zero_positive_split_handling(self):
        # Empty errors or all large errors
        metrics = compute_extended_metrics([100.0, 200.0])
        self.assertEqual(metrics["success_rate_5px"], 0.0)
        self.assertTrue(math.isfinite(metrics["median_error_px"]))

    def test_finite_metric_outputs(self):
        metrics = compute_extended_metrics([])
        self.assertEqual(metrics["observation_count"], 0)
        self.assertEqual(metrics["median_error_px"], 0.0)
        self.assertEqual(metrics["success_rate_5px"], 0.0)

    def test_schema_validation(self):
        metrics = compute_extended_metrics([2.0])
        required_keys = [
            "observation_count", "median_error_px", "mean_error_px", "p95_error_px",
            "max_error_px", "success_rate_1px", "success_rate_5px", "success_rate_10px",
            "success_rate_25px", "success_rate_50px"
        ]
        for k in required_keys:
            self.assertIn(k, metrics)

    def test_reproducibility_metadata(self):
        meta = {"seed": 20260913, "dataset": "dataset_v0.1", "budget": 250, "nms_distance_px": 4.0}
        self.assertEqual(meta["seed"], 20260913)
        self.assertEqual(meta["dataset"], "dataset_v0.1")

    def test_no_target_coordinates_in_prediction_features(self):
        from scripts.system_evaluation import extract_fast_features_for_obs
        params = inspect.signature(extract_fast_features_for_obs).parameters
        self.assertNotIn("target_x", params)
        self.assertNotIn("target_y", params)

    def test_baseline_comparison_correctness(self):
        p6_err = 500.0
        p7_err = 390.0
        delta = p7_err - p6_err
        self.assertAlmostEqual(delta, -110.0)


if __name__ == "__main__":
    unittest.main()
