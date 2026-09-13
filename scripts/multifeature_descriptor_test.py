from pathlib import Path
import csv

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 21
# MULTI-FEATURE DESCRIPTOR BENCHMARK
#
# Goal:
# Compare individual image representations against a
# handcrafted multi-feature descriptor.
#
# Representations:
#   raw
#   dog
#   gradient
#   laplacian
#   multifeature
#
# No learned weights.
# No test-set optimization.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "multifeature_descriptor"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "multifeature_results.csv"
)


SEARCH_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

TEMPLATE_SIZE = 10

BASE_TARGET_X = 75.25
BASE_TARGET_Y = 113.75

BASE_PITCH = 0.50
BASE_LINE_WIDTH = 0.20

SEEDS = range(
    20260850,
    20260880,
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]


# ============================================================
# SCENE GENERATION
# ============================================================

def periodic_scene(x, y):

    px = np.mod(
        x,
        BASE_PITCH,
    )

    py = np.mod(
        y,
        BASE_PITCH,
    )

    vertical = (
        px < BASE_LINE_WIDTH
    )

    horizontal = (
        py < BASE_LINE_WIDTH
    )

    return np.where(
        vertical | horizontal,
        235.0,
        45.0,
    )


def quasiperiodic_scene(
    x,
    y,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    variation = rng.uniform(
        0.03,
        0.08,
    )

    phase_x = rng.uniform(
        0,
        2 * np.pi,
    )

    phase_y = rng.uniform(
        0,
        2 * np.pi,
    )

    pitch = (
        BASE_PITCH
        * (
            1.0
            + variation
            * np.sin(
                0.025 * x
                + phase_x
            )
            * np.cos(
                0.021 * y
                + phase_y
            )
        )
    )

    warped_x = (
        x
        + 0.06
        * np.sin(
            0.11 * y
        )
    )

    warped_y = (
        y
        + 0.06
        * np.sin(
            0.09 * x
        )
    )

    px = np.mod(
        warped_x,
        pitch,
    )

    py = np.mod(
        warped_y,
        pitch,
    )

    width = (
        BASE_LINE_WIDTH
        * (
            1.0
            + 0.15
            * np.sin(
                0.07 * x
            )
        )
    )

    vertical = (
        px < width
    )

    horizontal = (
        py < width
    )

    structure = np.where(
        vertical | horizontal,
        235.0,
        45.0,
    )

    modulation = (
        10.0
        * np.sin(
            0.031 * x
            + 0.017 * y
        )
    )

    return structure + modulation


# ============================================================
# TARGET
# ============================================================

def target_mask(
    x,
    y,
    tx,
    ty,
):

    dx = x - tx
    dy = y - ty

    vertical = (
        (
            np.abs(
                dx + 0.20
            )
            < 0.075
        )
        &
        (
            np.abs(dy)
            < 0.38
        )
    )

    horizontal = (
        (
            np.abs(
                dx - 0.10
            )
            < 0.075
        )
        &
        (
            np.abs(
                dy + 0.25
            )
            < 0.075
        )
    )

    diagonal = (
        np.abs(
            dy
            - 0.8 * dx
            - 0.18
        )
        < 0.045
    ) & (
        np.abs(dx)
        < 0.35
    )

    circle = (
        dx * dx
        + dy * dy
        < 0.055 ** 2
    )

    return (
        vertical
        | horizontal
        | diagonal
        | circle
    )


# ============================================================
# RENDER
# ============================================================

def render_sensor(
    width,
    height,
    ppu,
    origin_x,
    origin_y,
    tx,
    ty,
    scene_type,
    seed,
):

    xs = (
        origin_x
        + (
            np.arange(width)
            + 0.5
        )
        / ppu
    )

    ys = (
        origin_y
        + (
            np.arange(height)
            + 0.5
        )
        / ppu
    )

    X, Y = np.meshgrid(
        xs,
        ys,
    )

    if scene_type == "periodic":

        image = periodic_scene(
            X,
            Y,
        )

    else:

        image = quasiperiodic_scene(
            X,
            Y,
            seed,
        )

    image = np.where(
        target_mask(
            X,
            Y,
            tx,
            ty,
        ),
        255.0,
        image,
    )

    return np.clip(
        image,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# REFERENCE
# ============================================================

def create_reference(
    tx,
    ty,
    scene_type,
    seed,
):

    physical_fov = (
        TEMPLATE_SIZE
        / SEARCH_PPU
    )

    origin_x = (
        tx
        - physical_fov / 2
    )

    origin_y = (
        ty
        - physical_fov / 2
    )

    reference = render_sensor(
        TEMPLATE_SIZE,
        TEMPLATE_SIZE,
        REFERENCE_PPU,
        origin_x,
        origin_y,
        tx,
        ty,
        scene_type,
        seed,
    )

    return reference


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_channel(
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

    if std < 1e-6:
        return np.zeros_like(
            image
        )

    return (
        image - mean
    ) / std


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    image,
):

    image_f = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Raw
    # --------------------------------------------------------

    raw = normalize_channel(
        image_f
    )

    # --------------------------------------------------------
    # Gaussian residual / DOG
    # --------------------------------------------------------

    g_small = cv2.GaussianBlur(
        image_f,
        (0, 0),
        0.8,
    )

    g_large = cv2.GaussianBlur(
        image_f,
        (0, 0),
        2.5,
    )

    dog = normalize_channel(
        g_small - g_large
    )

    # --------------------------------------------------------
    # Sobel gradients
    # --------------------------------------------------------

    gx = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = np.sqrt(
        gx * gx
        + gy * gy
    )

    # Orientation encoded using sin/cos.
    orientation = np.arctan2(
        gy,
        gx,
    )

    orientation_sin = np.sin(
        orientation
    )

    orientation_cos = np.cos(
        orientation
    )

    gx = normalize_channel(
        gx
    )

    gy = normalize_channel(
        gy
    )

    magnitude = normalize_channel(
        magnitude
    )

    orientation_sin = normalize_channel(
        orientation_sin
    )

    orientation_cos = normalize_channel(
        orientation_cos
    )

    # --------------------------------------------------------
    # Laplacian
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(
        image_f,
        cv2.CV_32F,
    )

    laplacian = normalize_channel(
        laplacian
    )

    return {
        "raw": raw,
        "dog": dog,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "orientation_sin": orientation_sin,
        "orientation_cos": orientation_cos,
        "laplacian": laplacian,
    }


# ============================================================
# FEATURE CORRELATION
# ============================================================

def feature_similarity(
    search_feature,
    reference_feature,
):

    a = search_feature.ravel()
    b = reference_feature.ravel()

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom < 1e-8:
        return 0.0

    return float(
        np.dot(a, b)
        / denom
    )


# ============================================================
# MATCH ONE FEATURE
# ============================================================

def match_single_feature(
    search,
    reference,
):

    result = cv2.matchTemplate(
        search.astype(
            np.float32
        ),
        reference.astype(
            np.float32
        ),
        cv2.TM_CCOEFF_NORMED,
    )

    return result


# ============================================================
# MULTIFEATURE RESPONSE
# ============================================================

def multifeature_response(
    search_features,
    reference_features,
):

    height = (
        search_features[
            "raw"
        ].shape[0]
        - TEMPLATE_SIZE
        + 1
    )

    width = (
        search_features[
            "raw"
        ].shape[1]
        - TEMPLATE_SIZE
        + 1
    )

    response = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Equal weights deliberately.
    #
    # We are testing whether the representation itself works,
    # not tuning the benchmark.
    # --------------------------------------------------------

    weights = {
        "raw": 1.0,
        "dog": 1.0,
        "gx": 0.75,
        "gy": 0.75,
        "magnitude": 0.75,
        "orientation_sin": 0.50,
        "orientation_cos": 0.50,
        "laplacian": 0.75,
    }

    total_weight = sum(
        weights.values()
    )

    for name, weight in weights.items():

        result = cv2.matchTemplate(
            search_features[name],
            reference_features[name],
            cv2.TM_CCOEFF_NORMED,
        )

        response += (
            weight
            * result
        )

    response /= (
        total_weight
    )

    return response


# ============================================================
# RANK
# ============================================================

def rank_response(
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
            best_loc[0]
            - gt_x,
            best_loc[1]
            - gt_y,
        )
    )

    return (
        gt_score,
        rank,
        best_score,
        error,
    )


# ============================================================
# ONE SCENE
# ============================================================

def evaluate_scene(
    scene_type,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    tx = (
        BASE_TARGET_X
        + rng.uniform(
            -8,
            8,
        )
    )

    ty = (
        BASE_TARGET_Y
        + rng.uniform(
            -8,
            8,
        )
    )

    search = render_sensor(
        SEARCH_SIZE,
        SEARCH_SIZE,
        SEARCH_PPU,
        0,
        0,
        tx,
        ty,
        scene_type,
        seed,
    )

    reference = create_reference(
        tx,
        ty,
        scene_type,
        seed,
    )

    gt_x = int(
        round(
            tx
            * SEARCH_PPU
            - TEMPLATE_SIZE / 2
        )
    )

    gt_y = int(
        round(
            ty
            * SEARCH_PPU
            - TEMPLATE_SIZE / 2
        )
    )

    search_features = extract_features(
        search
    )

    reference_features = extract_features(
        reference
    )

    results = {}

    # ========================================================
    # Individual representations
    # ========================================================

    for name in [
        "raw",
        "dog",
        "magnitude",
        "laplacian",
    ]:

        response = match_single_feature(
            search_features[name],
            reference_features[name],
        )

        (
            gt_score,
            rank,
            best_score,
            error,
        ) = rank_response(
            response,
            gt_x,
            gt_y,
        )

        results[
            f"{name}_score"
        ] = gt_score

        results[
            f"{name}_rank"
        ] = rank

        results[
            f"{name}_error"
        ] = error

    # ========================================================
    # Multi-feature
    # ========================================================

    response = multifeature_response(
        search_features,
        reference_features,
    )

    (
        gt_score,
        rank,
        best_score,
        error,
    ) = rank_response(
        response,
        gt_x,
        gt_y,
    )

    results[
        "multifeature_score"
    ] = gt_score

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

    print("-" * 76)

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
        "MICRONYX STEP 21"
    )
    print(
        "MULTI-FEATURE DESCRIPTOR BENCHMARK"
    )
    print("=" * 76)

    print()
    print(
        f"Scenes: "
        f"{len(SEEDS) * len(SCENE_TYPES)}"
    )

    print(
        "Representations:"
    )

    print(
        "  raw"
    )

    print(
        "  DOG"
    )

    print(
        "  gradient magnitude"
    )

    print(
        "  Laplacian"
    )

    print(
        "  multi-feature"
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

        print("-" * 76)

        for seed in SEEDS:

            counter += 1

            result = evaluate_scene(
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

    fieldnames = [
        key
        for key in results[0].keys()
    ]

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