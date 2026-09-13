from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.structural_verification import (
    compute_structural_similarity_score,
    run_structural_verification_pipeline,
)


class TestStructuralVerification(unittest.TestCase):

    def setUp(self):
        self.search_img = np.random.default_rng(20260913).integers(
            0, 256, size=(200, 200), dtype=np.uint8
        )
        self.ref_img = np.random.default_rng(20260913).integers(
            0, 256, size=(100, 100), dtype=np.uint8
        )

    def test_structural_score_range(self):
        score = compute_structural_similarity_score(self.search_img, self.ref_img, 100.0, 100.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deterministic_structural_score(self):
        s1 = compute_structural_similarity_score(self.search_img, self.ref_img, 80.0, 80.0)
        s2 = compute_structural_similarity_score(self.search_img, self.ref_img, 80.0, 80.0)
        self.assertEqual(s1, s2)

    def test_boundary_patch_structural_score(self):
        score = compute_structural_similarity_score(self.search_img, self.ref_img, 1.0, 1.0)
        self.assertTrue(np.isfinite(score))

    def test_structural_pipeline_integration(self):
        dataset_root = Path("dataset_v0.1")
        phase6_summary = Path("validation/phase6/robust_candidate_generation/robust_candidate_generation_summary.json")
        phase7_model = Path("validation/phase7/learned_candidate_ranking/xgboost_ranker.json")

        if dataset_root.exists() and phase6_summary.exists() and phase7_model.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "structural_verification"
                summary = run_structural_verification_pipeline(
                    dataset_root=dataset_root,
                    phase6_summary_path=phase6_summary,
                    phase7_model_path=phase7_model,
                    output_dir=out_dir,
                )
                self.assertIn("alpha_ml_weight", summary)
                self.assertIn("decision", summary)
                self.assertTrue((out_dir / "scores.csv").exists())
                self.assertTrue((out_dir / "summary.json").exists())
                self.assertTrue((out_dir / "report.md").exists())
                self.assertTrue((out_dir / "visualizations").exists())


if __name__ == "__main__":
    unittest.main()
