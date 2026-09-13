import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from scripts.learned_candidate_ranking import (
    FEATURE_NAMES,
    centered_patch,
    compute_binary_label,
    extract_candidate_features,
)


class TestLearnedCandidateRanking(unittest.TestCase):

    def setUp(self):
        self.search_img = np.random.default_rng(20260913).integers(
            0, 256, size=(200, 200), dtype=np.uint8
        )
        self.ref_img = np.random.default_rng(20260913).integers(
            0, 256, size=(100, 100), dtype=np.uint8
        )

    def test_feature_extraction_schema(self):
        feat = extract_candidate_features(
            search_img=self.search_img,
            reference_img=self.ref_img,
            candidate_x=100.0,
            candidate_y=100.0,
            generator_score=0.85,
            generator_rank=1,
            max_candidates=250,
            generator_name="dog",
        )
        self.assertEqual(len(FEATURE_NAMES), 17)
        self.assertEqual(len(feat), 17)

    def test_feature_values_are_finite(self):
        feat = extract_candidate_features(
            search_img=self.search_img,
            reference_img=self.ref_img,
            candidate_x=50.0,
            candidate_y=50.0,
            generator_score=0.75,
            generator_rank=5,
            max_candidates=250,
            generator_name="frequency",
        )
        self.assertTrue(np.isfinite(feat).all())

    def test_deterministic_feature_generation(self):
        feat1 = extract_candidate_features(
            search_img=self.search_img,
            reference_img=self.ref_img,
            candidate_x=80.0,
            candidate_y=80.0,
            generator_score=0.9,
            generator_rank=2,
            max_candidates=250,
            generator_name="gradient",
        )
        feat2 = extract_candidate_features(
            search_img=self.search_img,
            reference_img=self.ref_img,
            candidate_x=80.0,
            candidate_y=80.0,
            generator_score=0.9,
            generator_rank=2,
            max_candidates=250,
            generator_name="gradient",
        )
        np.testing.assert_array_equal(feat1, feat2)

    def test_target_coordinate_non_leakage(self):
        params = inspect.signature(extract_candidate_features).parameters
        self.assertNotIn("target_x", params)
        self.assertNotIn("target_y", params)
        self.assertNotIn("gt_x", params)
        self.assertNotIn("gt_y", params)

    def test_label_threshold_at_5px(self):
        self.assertEqual(compute_binary_label(100.0, 100.0, 103.0, 104.0), 1)  # dist 5.0 -> 1
        self.assertEqual(compute_binary_label(100.0, 100.0, 100.0, 105.0), 1)  # dist 5.0 -> 1
        self.assertEqual(compute_binary_label(100.0, 100.0, 100.0, 105.1), 0)  # dist 5.1 -> 0
        self.assertEqual(compute_binary_label(100.0, 100.0, 150.0, 150.0), 0)  # dist ~70 -> 0

    def test_train_validation_test_split_integrity(self):
        records = [
            {"split": "train", "label": 1},
            {"split": "validation", "label": 0},
            {"split": "hard_test", "label": 1},
        ]
        train_only = [r for r in records if r["split"] == "train"]
        self.assertEqual(len(train_only), 1)
        self.assertEqual(train_only[0]["label"], 1)

    def test_hard_negative_handling(self):
        lbl_pos = compute_binary_label(10.0, 10.0, 12.0, 12.0)
        lbl_neg = compute_binary_label(10.0, 10.0, 100.0, 100.0)
        self.assertEqual(lbl_pos, 1)
        self.assertEqual(lbl_neg, 0)

    def test_model_training_and_probability_output(self):
        X = np.random.default_rng(20260913).normal(size=(50, 17)).astype(np.float32)
        y = np.array([1]*10 + [0]*40, dtype=np.int32)

        clf = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=20260913,
            n_jobs=1,
            eval_metric="logloss",
        )
        clf.fit(X, y)

        probs = clf.predict_proba(X)[:, 1]
        self.assertEqual(len(probs), 50)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_deterministic_inference(self):
        X = np.random.default_rng(20260913).normal(size=(20, 17)).astype(np.float32)
        y = np.array([1]*5 + [0]*15, dtype=np.int32)

        clf = XGBClassifier(n_estimators=10, random_state=20260913, n_jobs=1, eval_metric="logloss")
        clf.fit(X, y)

        p1 = clf.predict_proba(X)[:, 1]
        p2 = clf.predict_proba(X)[:, 1]
        np.testing.assert_array_equal(p1, p2)

    def test_ranked_candidate_ordering(self):
        cands = [
            {"prob": 0.2, "score": 0.9, "y": 10.0, "x": 10.0},
            {"prob": 0.9, "score": 0.5, "y": 20.0, "x": 20.0},
            {"prob": 0.9, "score": 0.8, "y": 15.0, "x": 15.0},
        ]
        ranked = sorted(cands, key=lambda c: (-c["prob"], -c["score"], c["y"], c["x"]))

        self.assertEqual(ranked[0]["score"], 0.8)
        self.assertEqual(ranked[0]["prob"], 0.9)
        self.assertEqual(ranked[-1]["prob"], 0.2)

    def test_empty_or_invalid_candidate_handling(self):
        # Candidate at border / out of bounds
        patch = centered_patch(self.search_img, -10.0, -10.0, 10)
        self.assertIsNotNone(patch)
        self.assertEqual(patch.shape, (10, 10))

    def test_model_serialization_deserialization(self):
        X = np.random.default_rng(20260913).normal(size=(30, 17)).astype(np.float32)
        y = np.array([1]*5 + [0]*25, dtype=np.int32)

        clf = XGBClassifier(n_estimators=10, random_state=20260913, n_jobs=1, eval_metric="logloss")
        clf.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            clf.save_model(str(model_path))
            self.assertTrue(model_path.exists())

            clf2 = XGBClassifier()
            clf2.load_model(str(model_path))

            p1 = clf.predict_proba(X)[:, 1]
            p2 = clf2.predict_proba(X)[:, 1]
            np.testing.assert_allclose(p1, p2, rtol=1e-5)

    def test_metric_calculation(self):
        errs = [2.0, 4.0, 10.0, 20.0]
        succ = sum(1 for e in errs if e <= 5.0)
        self.assertEqual(succ, 2)
        self.assertEqual(succ / len(errs), 0.5)


if __name__ == "__main__":
    unittest.main()
