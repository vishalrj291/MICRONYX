import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.robust_candidate_generation import (
    Candidate,
    allocate_budget,
    build_weights_from_summary,
    evaluate_recall,
    generate_candidates,
    load_manifest,
    rank_candidates,
    spatial_deduplicate,
    validate_generators,
    _extract_target,
    ncc_response,
    dog_response,
    gradient_response,
    edge_response,
    frequency_response,
)


class TestRobustCandidateGeneration(unittest.TestCase):

    def test_generator_validation_accepts_all_generators(self):
        generators = validate_generators(
            [
                "ncc",
                "dog",
                "gradient",
                "edge",
                "frequency",
            ]
        )

        self.assertEqual(
            len(generators),
            5,
        )

    def test_generator_validation_rejects_unknown(self):
        with self.assertRaises(ValueError):
            validate_generators(
                ["ncc", "not_a_generator"]
            )

    def test_budget_is_fully_allocated(self):
        allocation = allocate_budget(
            [
                "ncc",
                "dog",
                "gradient",
                "edge",
                "frequency",
            ],
            250,
        )

        self.assertEqual(
            sum(allocation.values()),
            250,
        )

        for value in allocation.values():
            self.assertGreaterEqual(
                value,
                1,
            )

    def test_weighted_budget_prefers_higher_weight(self):
        allocation = allocate_budget(
            [
                "dog",
                "frequency",
            ],
            100,
            {
                "dog": 1.0,
                "frequency": 4.0,
            },
        )

        self.assertGreater(
            allocation["frequency"],
            allocation["dog"],
        )

    def test_zero_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            allocate_budget(
                ["dog", "frequency"],
                100,
                {
                    "dog": 0.0,
                    "frequency": 0.0,
                },
            )

    def test_invalid_weight_values_rejected(self):
        with self.assertRaises(ValueError):
            allocate_budget(
                ["dog", "frequency"],
                100,
                {
                    "dog": -1.0,
                    "frequency": 2.0,
                },
            )

    def test_candidate_schema_valid(self):
        cand = Candidate(
            x=150.5,
            y=200.25,
            score=0.85,
            generator="dog",
        )
        d = cand.as_dict()

        self.assertEqual(d["x"], 150.5)
        self.assertEqual(d["y"], 200.25)
        self.assertEqual(d["score"], 0.85)
        self.assertEqual(d["generator"], "dog")
        self.assertTrue(np.isfinite(d["x"]))
        self.assertTrue(np.isfinite(d["y"]))

    def test_spatial_deduplication(self):
        candidates = [
            Candidate(
                x=100.0,
                y=100.0,
                score=0.9,
                generator="dog",
            ),
            Candidate(
                x=101.0,
                y=100.0,
                score=0.8,
                generator="ncc",
            ),
            Candidate(
                x=120.0,
                y=100.0,
                score=0.7,
                generator="frequency",
            ),
        ]

        result = spatial_deduplicate(
            candidates,
            min_distance_px=4.0,
        )

        self.assertEqual(
            len(result),
            2,
        )

        self.assertEqual(
            result[0].generator,
            "dog",
        )

    def test_deterministic_ordering(self):
        candidates = [
            Candidate(x=10.0, y=20.0, score=0.5, generator="ncc"),
            Candidate(x=10.0, y=20.0, score=0.9, generator="dog"),
            Candidate(x=15.0, y=25.0, score=0.5, generator="edge"),
            Candidate(x=10.0, y=20.0, score=0.5, generator="dog"),
        ]

        ranked1 = rank_candidates(candidates)
        ranked2 = rank_candidates(candidates)

        self.assertEqual(ranked1, ranked2)
        self.assertEqual(ranked1[0].generator, "dog")
        self.assertEqual(ranked1[0].score, 0.9)

    def test_recall_is_measured_before_ranking_semantics(self):
        candidates = [
            Candidate(
                x=20.0,
                y=20.0,
                score=0.9,
                generator="ncc",
            ),
            Candidate(
                x=50.0,
                y=50.0,
                score=0.8,
                generator="dog",
            ),
            Candidate(
                x=100.0,
                y=100.0,
                score=0.7,
                generator="frequency",
            ),
        ]

        evaluation = evaluate_recall(
            candidates,
            target_x=100.0,
            target_y=100.0,
            tolerance_px=5.0,
        )

        self.assertTrue(
            evaluation.recall_at_10
        )

        self.assertAlmostEqual(
            evaluation.best_error_px,
            0.0,
            places=12,
        )

        self.assertEqual(
            evaluation.best_generator,
            "frequency",
        )

    def test_empty_candidate_pool(self):
        evaluation = evaluate_recall(
            [],
            target_x=10.0,
            target_y=10.0,
        )

        self.assertEqual(
            evaluation.candidate_count,
            0,
        )

        self.assertFalse(
            evaluation.recall_at_1
        )

        self.assertTrue(
            np.isinf(
                evaluation.best_error_px
            )
        )

    def test_target_coordinates_not_used_in_generator(self):
        generators_fn = [
            ncc_response,
            dog_response,
            gradient_response,
            edge_response,
            frequency_response,
            generate_candidates,
        ]

        for fn in generators_fn:
            params = inspect.signature(fn).parameters
            self.assertNotIn("target_x", params)
            self.assertNotIn("target_y", params)
            self.assertNotIn("gt_x", params)
            self.assertNotIn("gt_y", params)

    def test_manifest_csv_parsing_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_csv = tmp / "manifest.csv"
            manifest_csv.write_text(
                "pair_id,split,architecture,target_x,target_y,hard_case,seed\n"
                "TEST_00001,train,FinFET,500,300,False,12345\n",
                encoding="utf-8",
            )

            records = load_manifest(tmp)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["pair_id"], "TEST_00001")

            tx, ty = _extract_target(records[0])
            self.assertEqual(tx, 500.0)
            self.assertEqual(ty, 300.0)

    def test_build_weights_from_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_summary = Path(tmpdir) / "summary.json"
            tmp_summary.write_text(
                '{\n'
                '  "generator_benchmark": {\n'
                '    "ncc": {"median_error_px": 500.0},\n'
                '    "dog": {"median_error_px": 250.0},\n'
                '    "gradient": {"median_error_px": 250.0},\n'
                '    "edge": {"median_error_px": 500.0},\n'
                '    "frequency": {"median_error_px": 250.0}\n'
                '  }\n'
                '}\n',
                encoding="utf-8",
            )

            weights = build_weights_from_summary(tmp_summary)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
            self.assertGreater(weights["dog"], weights["ncc"])


if __name__ == "__main__":
    unittest.main()