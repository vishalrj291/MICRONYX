from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier, XGBRanker

from scripts.advanced_ranking import run_advanced_ranking_pipeline


class TestAdvancedRanking(unittest.TestCase):

    def setUp(self):
        self.X_dummy = np.random.randn(30, 17).astype(np.float32)
        self.y_dummy = np.array([1]*3 + [0]*27, dtype=np.int32)
        self.qids_dummy = np.array([0]*10 + [1]*10 + [2]*10, dtype=np.int32)

    def test_pointwise_xgboost_training(self):
        clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=20260913)
        clf.fit(self.X_dummy, self.y_dummy)
        probs = clf.predict_proba(self.X_dummy)[:, 1]
        self.assertEqual(len(probs), 30)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_pairwise_xgboost_ranker_training(self):
        unique_qids, qid_counts = np.unique(self.qids_dummy, return_counts=True)
        ranker = XGBRanker(objective="rank:pairwise", n_estimators=10, max_depth=3, random_state=20260913)
        ranker.fit(self.X_dummy, self.y_dummy, group=qid_counts)
        scores = ranker.predict(self.X_dummy)
        self.assertEqual(len(scores), 30)
        self.assertTrue(np.isfinite(scores).all())

    def test_class_balanced_xgboost_training(self):
        pos_weight = float(27 / 3)
        clf = XGBClassifier(n_estimators=10, max_depth=3, scale_pos_weight=pos_weight, random_state=20260913)
        clf.fit(self.X_dummy, self.y_dummy)
        probs = clf.predict_proba(self.X_dummy)[:, 1]
        self.assertEqual(len(probs), 30)

    def test_reproducibility_seed(self):
        clf1 = XGBClassifier(n_estimators=10, random_state=20260913)
        clf2 = XGBClassifier(n_estimators=10, random_state=20260913)
        clf1.fit(self.X_dummy, self.y_dummy)
        clf2.fit(self.X_dummy, self.y_dummy)
        p1 = clf1.predict_proba(self.X_dummy)
        p2 = clf2.predict_proba(self.X_dummy)
        np.testing.assert_array_equal(p1, p2)

    def test_advanced_ranking_pipeline_integration(self):
        dataset_root = Path("dataset_v0.1")
        phase6_summary = Path("validation/phase6/robust_candidate_generation/robust_candidate_generation_summary.json")

        if dataset_root.exists() and phase6_summary.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "advanced_ranking"
                summary = run_advanced_ranking_pipeline(
                    dataset_root=dataset_root,
                    phase6_summary_path=phase6_summary,
                    output_dir=out_dir,
                )
                self.assertIn("ranking_approaches_evaluated", summary)
                self.assertIn("decision", summary)
                self.assertTrue((out_dir / "comparison.csv").exists())
                self.assertTrue((out_dir / "summary.json").exists())
                self.assertTrue((out_dir / "report.md").exists())
                self.assertTrue((out_dir / "models" / "pairwise_xgboost_ranker.json").exists())


if __name__ == "__main__":
    unittest.main()
