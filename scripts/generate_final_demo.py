"""
MICRONYX PS02 — Final Demo Pair Generator
==========================================

Generates the final canonical 1000x1000 search/reference pair
using canonical_renderer.py.

IMPORTANT:
- Search: 1000 x 1000
- Reference: 1000 x 1000
- Search sampling: 5 px/unit
- Reference sampling: 50 px/unit
- Sampling ratio: 10x
- No legacy generator
- No target fingerprint
- No alternate observation model
"""

from pathlib import Path
import cv2

import canonical_renderer as cr


# ============================================================
# OUTPUT
# ============================================================

PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CANONICAL TEST SCENE
# ============================================================

TX = cr.DEFAULT_TX
TY = cr.DEFAULT_TY

SEED = 20260816


# ============================================================
# GENERATE
# ============================================================

def main():

    print("=" * 76)
    print("MICRONYX PS02 — FINAL DEMO PAIR GENERATOR")
    print("=" * 76)
    print()

    print("Canonical acquisition:")
    print(
        f"  Search:       "
        f"{cr.SEARCH_WIDTH} x {cr.SEARCH_HEIGHT}"
    )

    print(
        f"  Reference:    "
        f"{cr.REFERENCE_WIDTH} x {cr.REFERENCE_HEIGHT}"
    )

    print(
        f"  Search PPU:   "
        f"{cr.SEARCH_PIXELS_PER_UNIT}"
    )

    print(
        f"  Reference PPU:"
        f" {cr.REFERENCE_PIXELS_PER_UNIT}"
    )

    print(
        f"  Sampling:     "
        f"{cr.REFERENCE_PIXELS_PER_UNIT / cr.SEARCH_PIXELS_PER_UNIT:.1f}x"
    )

    print()

    for scene_type in (
        "periodic",
        "quasiperiodic",
    ):

        print(
            f"Generating {scene_type} scene..."
        )

        observation = cr.generate_observation(
            TX,
            TY,
            scene_type,
            SEED,
        )

        search = observation[
            "search"
        ]

        reference = observation[
            "reference"
        ]

        template = observation[
            "template"
        ]

        gt_x = observation[
            "gt_x"
        ]

        gt_y = observation[
            "gt_y"
        ]

        print(
            "  Search shape:",
            search.shape,
        )

        print(
            "  Reference shape:",
            reference.shape,
        )

        print(
            "  Template shape:",
            template.shape,
        )

        print(
            "  GT top-left:",
            (gt_x, gt_y),
        )

        # ----------------------------------------------------
        # Hard validation
        # ----------------------------------------------------

        assert search.shape == (
            1000,
            1000,
        )

        assert reference.shape == (
            1000,
            1000,
        )

        assert template.shape == (
            100,
            100,
        )

        assert search.dtype.name == "uint8"
        assert reference.dtype.name == "uint8"
        assert template.dtype.name == "uint8"

        # ----------------------------------------------------
        # Save only the first scene as the canonical demo pair.
        #
        # The final localizer takes one search/reference pair.
        # ----------------------------------------------------

        if scene_type == "periodic":

            search_path = (
                OUTPUT_DIR
                / "clean_search.png"
            )

            reference_path = (
                OUTPUT_DIR
                / "clean_reference.png"
            )

            if not cv2.imwrite(
                str(search_path),
                search,
            ):
                raise RuntimeError(
                    "Failed to write search image."
                )

            if not cv2.imwrite(
                str(reference_path),
                reference,
            ):
                raise RuntimeError(
                    "Failed to write reference image."
                )

            print()
            print(
                "Saved canonical demo pair:"
            )

            print(
                f"  Search:    {search_path}"
            )

            print(
                f"  Reference: {reference_path}"
            )

    print()
    print("=" * 76)
    print("FINAL DEMO PAIR GENERATION: PASS")
    print("=" * 76)


if __name__ == "__main__":
    main()