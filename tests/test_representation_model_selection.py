"""
MICRONYX Phase B tests.

Run:

    python -m unittest discover -s tests \
        -p "test_representation_model_selection.py" -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

from scripts import representation_model_selection


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from scripts import representation_model_selection as rms


class TestRepresentationModelSelection(unittest.TestCase):

    def test_generator_list_is_complete(self):

        self.assertEqual(
            set(rms.GENERATORS),
            {
                "ncc",
                "dog",
                "gradient",
                "edge",
                "frequency",
            },
        )

    def test_weights_normalize(self):

        weights = rms.normalize_weights({
            "ncc": 2,
            "dog": 3,
            "gradient": 1,
            "edge": 2,
            "frequency": 2,
        })

        self.assertTrue(
        np.isclose(
                float(np.sum(list(weights.values()))),
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
        )


        for value in weights.values():

            self.assertGreaterEqual(
                value,
                0.0,
            )

    def test_zero_weights_are_rejected(self):

        with self.assertRaises(
            ValueError
        ):

            rms.normalize_weights({
                "ncc": 0,
                "dog": 0,
                "gradient": 0,
                "edge": 0,
                "frequency": 0,
            })

    def test_rule_policy_only_uses_eda(self):

        row = {
            "search_scene_structure_class":
                "periodic",

            "search_periodicity_score":
                0.8,

            "search_spectral_entropy":
                0.4,

            "search_edge_density":
                0.1,
        }

        weights = rms.rule_policy(
            row
        )

        self.assertAlmostEqual(
            sum(weights.values()),
            1.0,
            places=12,
        )

        self.assertIn(
            "dog",
            weights,
        )

    def test_target_coordinates_are_not_eda_features(self):

        self.assertNotIn(
            "target_x",
            rms.EDA_FEATURES,
        )

        self.assertNotIn(
            "target_y",
            rms.EDA_FEATURES,
        )

        self.assertNotIn(
            "gt_x",
            rms.EDA_FEATURES,
        )

        self.assertNotIn(
            "gt_y",
            rms.EDA_FEATURES,
        )

    def test_reference_is_scaled_by_factor_ten(self):

        reference = np.zeros(
            (1000, 1000),
            dtype=np.uint8,
        )

        template = rms.resize_reference(
            reference,
            (1000, 1000),
            10.0,
        )

        self.assertEqual(
            template.shape,
            (100, 100),
        )

    def test_template_matching_finds_known_location(self):

        search = np.zeros(
            (500, 500),
            dtype=np.uint8,
        )

        reference = np.zeros(
            (100, 100),
            dtype=np.uint8,
        )

        rng = np.random.default_rng(
            123
        )

        reference[
            20:80,
            20:80
        ] = rng.integers(
            20,
            240,
            size=(60, 60),
            dtype=np.uint8,
        )

        template = reference.copy()

        search[
            200:300,
            150:250
        ] = template

        score, location = (
            rms.normalized_template_match(
                search,
                template,
            )
        )

        self.assertGreater(
            score,
            0.99,
        )

        self.assertEqual(
            location,
            (150, 200),
        )

    def test_gradient_representation(self):

        image = np.zeros(
            (100, 100),
            dtype=np.uint8,
        )

        image[
            25:75,
            40:60
        ] = 255

        gradient = (
            rms.gradient_image(
                image
            )
        )

        self.assertEqual(
            gradient.shape,
            image.shape,
        )

        self.assertGreater(
            float(
                np.max(gradient)
            ),
            0.0,
        )

    def test_dog_representation(self):

        image = np.zeros(
            (100, 100),
            dtype=np.uint8,
        )

        image[
            25:75,
            40:60
        ] = 255

        dog = (
            rms.dog_image(
                image
            )
        )

        self.assertEqual(
            dog.shape,
            image.shape,
        )

        self.assertTrue(
            np.isfinite(
                dog
            ).all()
        )

    def test_edge_representation(self):

        image = np.zeros(
            (100, 100),
            dtype=np.uint8,
        )

        image[
            25:75,
            40:60
        ] = 255

        edge = (
            rms.edge_image(
                image
            )
        )

        self.assertEqual(
            edge.shape,
            image.shape,
        )

    def test_frequency_representation(self):

        image = np.zeros(
            (128, 128),
            dtype=np.uint8,
        )

        yy, xx = np.indices(
            image.shape
        )

        image = (
            127.5
            + 100
            * np.sin(
                2
                * np.pi
                * xx
                / 16
            )
        ).astype(
            np.uint8
        )

        frequency = (
            rms.frequency_representation(
                image
            )
        )

        self.assertEqual(
            frequency.shape,
            image.shape,
        )

        self.assertTrue(
            np.isfinite(
                frequency
            ).all()
        )

    def test_best_generator_prefers_lower_error(self):

        results = [
            rms.GeneratorResult(
                generator="ncc",
                score=0.90,
                predicted_x=100,
                predicted_y=100,
                error_px=5.0,
                runtime_ms=1.0,
            ),
            rms.GeneratorResult(
                generator="dog",
                score=0.80,
                predicted_x=100,
                predicted_y=100,
                error_px=1.0,
                runtime_ms=2.0,
            ),
            rms.GeneratorResult(
                generator="gradient",
                score=0.80,
                predicted_x=100,
                predicted_y=100,
                error_px=3.0,
                runtime_ms=1.0,
            ),
            rms.GeneratorResult(
                generator="edge",
                score=0.70,
                predicted_x=100,
                predicted_y=100,
                error_px=4.0,
                runtime_ms=1.0,
            ),
            rms.GeneratorResult(
                generator="frequency",
                score=0.70,
                predicted_x=100,
                predicted_y=100,
                error_px=2.0,
                runtime_ms=2.0,
            ),
        ]

        self.assertEqual(
            rms.best_generator(
                results
            ),
            "dog",
        )

    def test_override_parser(self):

        weights = (
            rms.parse_override(
                "dog:0.5,ncc:0.2,"
                "gradient:0.1,edge:0.1,"
                "frequency:0.1"
            )
        )

        self.assertAlmostEqual(
            sum(weights.values()),
            1.0,
            places=12,
        )

        self.assertAlmostEqual(
            weights["dog"],
            0.5,
            places=12,
        )

    def test_override_rejects_unknown_generator(self):

        with self.assertRaises(
            ValueError
        ):

            rms.parse_override(
                "magic:1.0"
            )

    def test_softmax_is_valid_distribution(self):

        values = np.asarray([
            1.0,
            2.0,
            3.0,
        ])

        result = rms.softmax(
            values
        )

        self.assertAlmostEqual(
            float(
                np.sum(result)
            ),
            1.0,
            places=12,
        )

        self.assertTrue(
            np.all(
                result > 0
            )
        )


if __name__ == "__main__":
    unittest.main()