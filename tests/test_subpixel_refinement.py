from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.subpixel_refinement import (
    compute_subpixel_refined_location,
    fit_subpixel_parabola,
    run_subpixel_refinement_pipeline,
)


class TestSubpixelRefinement(unittest.TestCase):

    def setUp(self):
        self.search_img = np.random.default_rng(20260913).integers(
            0, 256, size=(200, 200), dtype=np.uint8
        )
        self.ref_img = np.random.default_rng(20260913).integers(
            0, 256, size=(100, 100), dtype=np.uint8
        )

    def test_parabola_fitting_peak(self):
        grid = np.array([
            [0.5, 0.7, 0.5],
            [0.8, 1.0, 0.8],
            [0.5, 0.7, 0.5],
        ], dtype=np.float32)
        dx, dy = fit_subpixel_parabola(grid)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

    def test_parabola_subpixel_shift_bounds(self):
        grid = np.array([
            [0.2, 0.4, 0.9],
            [0.3, 0.6, 1.0],
            [0.1, 0.3, 0.8],
        ], dtype=np.float32)
        dx, dy = fit_subpixel_parabola(grid)
        self.assertTrue(-1.0 <= dx <= 1.0)
        self.assertTrue(-1.0 <= dy <= 1.0)

    def test_deterministic_subpixel_location(self):
        rx1, ry1, dx1, dy1 = compute_subpixel_refined_location(self.search_img, self.ref_img, 100.0, 100.0)
        rx2, ry2, dx2, dy2 = compute_subpixel_refined_location(self.search_img, self.ref_img, 100.0, 100.0)
        self.assertEqual(rx1, rx2)
        self.assertEqual(ry1, ry2)
        self.assertEqual(dx1, dx2)
        self.assertEqual(dy1, dy2)

    def test_subpixel_pipeline_integration(self):
        dataset_root = Path("dataset_v0.1")
        phase6_summary = Path("validation/phase6/robust_candidate_generation/robust_candidate_generation_summary.json")
        phase7_model = Path("validation/phase7/learned_candidate_ranking/xgboost_ranker.json")

        if dataset_root.exists() and phase6_summary.exists() and phase7_model.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "subpixel_refinement"
                summary = run_subpixel_refinement_pipeline(
                    dataset_root=dataset_root,
                    phase6_summary_path=phase6_summary,
                    phase7_model_path=phase7_model,
                    output_dir=out_dir,
                )
                self.assertIn("original_metrics", summary)
                self.assertIn("subpixel_refined_metrics", summary)
                self.assertIn("decision", summary)
                self.assertTrue((out_dir / "comparison.csv").exists())
                self.assertTrue((out_dir / "summary.json").exists())
                self.assertTrue((out_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
