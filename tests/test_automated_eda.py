"""
MICRONYX Phase 4A tests.

Run from repository root:

    python -m unittest discover -s tests -p "test_automated_eda.py" -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import automated_eda as eda


class TestAutomatedEDA(unittest.TestCase):

    def test_schema_and_finite_values(self):

        image = np.zeros(
            (128, 128),
            dtype=np.uint8,
        )

        image[
            32:96,
            48:80,
        ] = 255

        features = eda.analyze_image(
            image
        )

        required = {
            "mean",
            "std",
            "global_contrast",
            "dynamic_range",
            "gradient_mean",
            "edge_density",
            "laplacian_energy",
            "fft_high_ratio",
            "spectral_entropy",
            "periodicity_indicator",
            "correlation_length_x_px",
            "correlation_length_y_px",
            "dominant_period_x_px",
            "dominant_period_y_px",
            "local_std_mean",
            "highpass_residual_std",
            "structure_class",
            "spectral_class",
            "edge_class",
            "texture_class",
            "contrast_class",
        }

        self.assertTrue(
            required.issubset(
                features.keys()
            )
        )

        for key, value in features.items():

            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):

                self.assertTrue(
                    np.isfinite(value),
                    key,
                )

    def test_constant_image_is_stable(self):

        image = np.full(
            (64, 64),
            127,
            dtype=np.uint8,
        )

        features = eda.analyze_image(
            image
        )

        self.assertAlmostEqual(
            features["std"],
            0.0,
            places=12,
        )

        self.assertAlmostEqual(
            features["global_contrast"],
            0.0,
            places=12,
        )

        self.assertTrue(
            np.isfinite(
                features["spectral_entropy"]
            )
        )

    def test_sampling_ratio_uses_metadata(self):

        ratio, source = (
            eda.estimate_sampling(
                (1000, 1000),
                (1000, 1000),
                {
                    "scale_relationship_nominal":
                        10.0
                },
            )
        )

        self.assertEqual(
            ratio,
            10.0,
        )

        self.assertEqual(
            source,
            "metadata",
        )

    def test_dimensions_do_not_fake_sampling_ratio(self):

        ratio, source = (
            eda.estimate_sampling(
                (1000, 1000),
                (1000, 1000),
                {},
            )
        )

        self.assertIsNone(
            ratio
        )

        self.assertEqual(
            source,
            "not_inferable_from_dimensions",
        )

    def test_target_coordinates_are_not_required(self):

        image = np.random.default_rng(
            123
        ).integers(
            0,
            256,
            size=(96, 96),
            dtype=np.uint8,
        )

        features = eda.analyze_image(
            image
        )

        self.assertIn(
            "periodicity_indicator",
            features,
        )


if __name__ == "__main__":
    unittest.main()