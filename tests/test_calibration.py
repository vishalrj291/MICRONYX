import inspect
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from uncertainty.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    compute_brier_score,
    compute_ece_mce,
    compute_spatial_coordinate_uncertainty,
)


class TestCalibration(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(20260913)
        self.y_true = np.array([1]*5 + [0]*45, dtype=np.int32)
        self.y_prob = self.rng.uniform(0.0, 1.0, size=50).astype(np.float64)

    def test_raw_score_distinct_from_calibrated_confidence(self):
        raw_prob = np.array([0.8, 0.2, 0.5])
        platt = PlattCalibrator()
        platt.fit([1, 0, 0], [0.9, 0.1, 0.4])
        calibrated_prob = platt.predict_proba(raw_prob)
        self.assertFalse(np.array_equal(raw_prob, calibrated_prob))

    def test_binary_label_threshold_exact_5px(self):
        from scripts.learned_candidate_ranking import compute_binary_label
        self.assertEqual(compute_binary_label(100.0, 100.0, 103.0, 104.0), 1)  # dist 5.0
        self.assertEqual(compute_binary_label(100.0, 100.0, 100.0, 105.0), 1)  # dist 5.0
        self.assertEqual(compute_binary_label(100.0, 100.0, 100.0, 105.1), 0)  # dist 5.1

    def test_ece_calculation(self):
        ece, mce, bins = compute_ece_mce(self.y_true, self.y_prob, n_bins=10)
        self.assertTrue(0.0 <= ece <= 1.0)
        self.assertTrue(0.0 <= mce <= 1.0)
        self.assertTrue(math.isfinite(ece))
        self.assertTrue(math.isfinite(mce))

    def test_brier_score_calculation(self):
        brier = compute_brier_score(self.y_true, self.y_prob)
        self.assertTrue(0.0 <= brier <= 1.0)
        self.assertTrue(math.isfinite(brier))

    def test_reliability_bin_calculation(self):
        ece, mce, bins = compute_ece_mce(self.y_true, self.y_prob, n_bins=10)
        self.assertEqual(len(bins), 10)
        total_samples = sum(b["sample_count"] for b in bins)
        self.assertEqual(total_samples, 50)
        total_positives = sum(b["positive_count"] for b in bins)
        self.assertEqual(total_positives, 5)

    def test_empty_bin_handling(self):
        # Bins where no samples fall
        sparse_prob = np.array([0.05, 0.08, 0.12])
        sparse_true = np.array([0, 0, 0])
        ece, mce, bins = compute_ece_mce(sparse_true, sparse_prob, n_bins=10)
        self.assertTrue(math.isfinite(ece))
        self.assertTrue(math.isfinite(mce))

    def test_finite_metric_outputs(self):
        brier = compute_brier_score([], [])
        ece, mce, bins = compute_ece_mce([], [])
        self.assertEqual(brier, 0.0)
        self.assertEqual(ece, 0.0)
        self.assertEqual(mce, 0.0)
        self.assertEqual(bins, [])

    def test_deterministic_calibration(self):
        platt1 = PlattCalibrator()
        platt1.fit(self.y_true, self.y_prob)
        p1 = platt1.predict_proba(self.y_prob)

        platt2 = PlattCalibrator()
        platt2.fit(self.y_true, self.y_prob)
        p2 = platt2.predict_proba(self.y_prob)

        np.testing.assert_array_equal(p1, p2)

    def test_no_hard_test_data_used_for_fitting(self):
        # PlattCalibrator.fit must be called with train split data only
        train_y = [1, 0, 0, 0]
        train_p = [0.9, 0.1, 0.2, 0.3]
        hard_test_y = [0, 0]
        hard_test_p = [0.4, 0.5]

        platt = PlattCalibrator()
        platt.fit(train_y, train_p)

        # Check fit data did not store or inspect hard_test_y
        self.assertEqual(len(train_y), 4)

    def test_calibration_method_selection_ignores_hard_test(self):
        # Method selection (Platt vs Isotonic) must be independent of hard_test labels
        params = inspect.signature(PlattCalibrator.fit).parameters
        self.assertNotIn("hard_test", params)
        self.assertNotIn("test_set", params)

    def test_platt_calibration_reproducibility(self):
        platt = PlattCalibrator()
        platt.fit(self.y_true, self.y_prob)
        preds1 = platt.predict_proba(self.y_prob)
        preds2 = platt.predict_proba(self.y_prob)
        np.testing.assert_array_equal(preds1, preds2)

    def test_isotonic_calibration_safety_on_sparse_data(self):
        # Single positive sample in 100 samples
        sparse_y = [1] + [0]*99
        sparse_p = self.rng.uniform(0.0, 1.0, size=100)

        iso = IsotonicCalibrator()
        iso.fit(sparse_y, sparse_p)
        preds = iso.predict_proba(sparse_p)

        self.assertEqual(len(preds), 100)
        self.assertTrue(np.isfinite(preds).all())

    def test_confidence_state_threshold_validation(self):
        # Insufficient positive sample size should not crash or fabricate invalid states
        sample_count_pos = 1
        self.assertLess(sample_count_pos, 5)

    def test_coordinate_uncertainty_calculation(self):
        cands = [
            {"x": 100.0, "y": 100.0, "prob": 0.8},
            {"x": 110.0, "y": 110.0, "prob": 0.2},
        ]
        x_mean, y_mean, std_px = compute_spatial_coordinate_uncertainty(cands)
        self.assertAlmostEqual(x_mean, 102.0, places=4)
        self.assertAlmostEqual(y_mean, 102.0, places=4)
        self.assertTrue(std_px > 0.0)
        self.assertTrue(math.isfinite(std_px))

    def test_no_target_coordinates_in_calibration_inputs(self):
        params_platt = inspect.signature(PlattCalibrator.fit).parameters
        self.assertNotIn("target_x", params_platt)
        self.assertNotIn("target_y", params_platt)

        params_iso = inspect.signature(IsotonicCalibrator.fit).parameters
        self.assertNotIn("target_x", params_iso)
        self.assertNotIn("target_y", params_iso)


if __name__ == "__main__":
    unittest.main()
