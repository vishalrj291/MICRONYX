from pathlib import Path
import csv

import cv2
import numpy as np

from canonical_renderer import (
    generate_observation,
)


# ============================================================
# MICRONYX STEP 21 — CORRECTED
#
# MULTI-FEATURE DESCRIPTOR BENCHMARK
#
# IMPORTANT:
# Uses canonical_renderer.py.
#
# Therefore:
#
#   Search and reference originate from exactly the same
#   continuous physical scene.
#
#   Only sensor sampling differs.
# ============================================================


PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "multifeature_descriptor_v2"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "multifeature_results.csv"
)


SEEDS = range(
    20260850,
    20260880,
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

TEMPLATE_SIZE = 10


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    image,
):
    image = image.astype(
        np.float32
    )

    mean = np.mean(
        image
    )

    std = np.std(
        image
    )

    if std < 1e-8:

        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    return (
        image - mean
    ) / std


# ============================================================
# FEATURES
# ============================================================

def extract_features(
    image,
):
    image = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw = normalize(
        image
    )

    # --------------------------------------------------------
    # DOG
    # --------------------------------------------------------

    g1 = cv2.GaussianBlur(
        image,
        (0, 0),
        0.8,
    )

    g2 = cv2.GaussianBlur(
        image,
        (0, 0),
        2.5,
    )

    dog = normalize(
        g1 - g2
    )

    # --------------------------------------------------------
    # GRADIENT
    # --------------------------------------------------------

    gx_raw = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy_raw = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude_raw = np.sqrt(
        gx_raw * gx_raw
        + gy_raw * gy_raw
    )

    gx = normalize(
        gx_raw
    )

    gy = normalize(
        gy_raw
    )

    magnitude = normalize(
        magnitude_raw
    )

    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    orientation = np.arctan2(
        gy_raw,
        gx_raw,
    )

    orientation_sin = (
        np.sin(
            orientation
        ).astype(
            np.float32
        )
    )

    orientation_cos = (
        np.cos(
            orientation
        ).astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # LAPLACIAN
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_32F,
    )

    laplacian = normalize(
        laplacian
    )

    return {
        "raw": raw,
        "dog": dog,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "orientation_sin":
            orientation_sin,
        "orientation_cos":
            orientation_cos,
        "laplacian": laplacian,
    }


# ============================================================
# TEMPLATE MATCH
# ============================================================

def match(
    search,
    template,
):
    return cv2.matchTemplate(
        search.astype(
            np.float32
        ),
        template.astype(
            np.float32
        ),
        cv2.TM_CCOEFF_NORMED,
    )


# ============================================================
# MULTI-FEATURE RESPONSE
# ============================================================

def multi_response(
    search_features,
    reference_features,
):
    """
    Equal / fixed weights.

    We intentionally DO NOT optimize these on the test set.
    """

    weights = {
        "raw": 1.00,
        "dog": 1.00,
        "gx": 0.75,
        "gy": 0.75,
        "magnitude": 0.75,
        "orientation_sin": 0.50,
        "orientation_cos": 0.50,
        "laplacian": 0.75,
    }

    total = sum(
        weights.values()
    )

    response = None

    for name, weight in weights.items():

        current = match(
            search_features[name],
            reference_features[name],
        )

        if response is None:

            response = (
                weight
                * current
            )

        else:

            response += (
                weight
                * current
            )

    response /= total

    return response


# ============================================================
# RANK + ERROR
# ============================================================

def evaluate_response(
    response,
    gt_x,
    gt_y,
):
    gt_score = float(
        response[
            gt_y,
            gt_x,
        ]
    )

    rank = (
        1
        + int(
            np.sum(
                response
                > gt_score
            )
        )
    )

    _, best_score, _, best_loc = (
        cv2.minMaxLoc(
            response
        )
    )

    error = float(
        np.hypot(
            best_loc[0] - gt_x,
            best_loc[1] - gt_y,
        )
    )

    return (
        gt_score,
        rank,
        float(best_score),
        error,
    )


# ============================================================
# SCENE
# ============================================================

def run_scene(
    scene_type,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    tx = (
        75.25
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    ty = (
        113.75
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    observation = generate_observation(
        tx,
        ty,
        scene_type,
        seed,
    )

    search = observation[
        "search"
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

    search_features = extract_features(
        search
    )

    template_features = extract_features(
        template
    )

    results = {}

    # --------------------------------------------------------
    # Individual features
    # --------------------------------------------------------

    methods = [
        "raw",
        "dog",
        "magnitude",
        "laplacian",
    ]

    for method in methods:

        response = match(
            search_features[method],
            template_features[method],
        )

        (
            score,
            rank,
            best_score,
            error,
        ) = evaluate_response(
            response,
            gt_x,
            gt_y,
        )

        results[
            f"{method}_score"
        ] = score

        results[
            f"{method}_rank"
        ] = rank

        results[
            f"{method}_error"
        ] = error

    # --------------------------------------------------------
    # Multi-feature
    # --------------------------------------------------------

    response = multi_response(
        search_features,
        template_features,
    )

    (
        score,
        rank,
        best_score,
        error,
    ) = evaluate_response(
        response,
        gt_x,
        gt_y,
    )

    results[
        "multifeature_score"
    ] = score

    results[
        "multifeature_rank"
    ] = rank

    results[
        "multifeature_error"
    ] = error

    return results


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    results,
    scene_type,
):
    subset = [
        r
        for r in results
        if r["scene"]
        == scene_type
    ]

    methods = [
        "raw",
        "dog",
        "magnitude",
        "laplacian",
        "multifeature",
    ]

    print()
    print(
        scene_type.upper()
    )

    print(
        "-" * 76
    )

    for method in methods:

        ranks = np.array(
            [
                r[
                    f"{method}_rank"
                ]
                for r in subset
            ]
        )

        errors = np.array(
            [
                r[
                    f"{method}_error"
                ]
                for r in subset
            ]
        )

        scores = np.array(
            [
                r[
                    f"{method}_score"
                ]
                for r in subset
            ]
        )

        print(
            f"{method:<14}"
            f"Top1="
            f"{np.mean(ranks == 1) * 100:6.2f}% "
            f"Top5="
            f"{np.mean(ranks <= 5) * 100:6.2f}% "
            f"<=5px="
            f"{np.mean(errors <= 5) * 100:6.2f}% "
            f"MedErr="
            f"{np.median(errors):8.3f}px "
            f"MedRank="
            f"{np.median(ranks):8.1f} "
            f"MedScore="
            f"{np.median(scores):.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 76)
    print(
        "MICRONYX STEP 21 — CORRECTED"
    )
    print(
        "CANONICAL MULTI-FEATURE BENCHMARK"
    )
    print("=" * 76)

    print()
    print(
        "Canonical observation model:"
    )

    print(
        "continuous physical scene"
    )

    print(
        "        ↓"
    )

    print(
        "supersampled rendering"
    )

    print(
        "        ↓"
    )

    print(
        "sensor area integration"
    )

    print(
        "        ↓"
    )

    print(
        "search / reference observations"
    )

    print()
    print(
        "Scenes:",
        len(SEEDS)
        * len(SCENE_TYPES),
    )

    results = []

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    counter = 0

    for scene_type in SCENE_TYPES:

        print()
        print(
            f"SCENE: "
            f"{scene_type.upper()}"
        )

        print(
            "-" * 76
        )

        for seed in SEEDS:

            counter += 1

            result = run_scene(
                scene_type,
                seed,
            )

            result[
                "scene"
            ] = scene_type

            result[
                "seed"
            ] = seed

            results.append(
                result
            )

            print(
                f"Scene "
                f"{counter:02d}/{total}"
            )

    # ========================================================
    # CSV
    # ========================================================

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 76)
    print(
        "SUMMARY"
    )
    print("=" * 76)

    for scene_type in SCENE_TYPES:

        summarize(
            results,
            scene_type,
        )

    print()
    print("=" * 76)

    print(
        "Saved:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 76)


if __name__ == "__main__":
    main()