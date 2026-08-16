from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 19
# RESIDUAL / LOCAL-CONTRAST MATCHING
#
# Compare:
#
# 1. Raw intensity
# 2. High-pass residual
# 3. Difference of Gaussians
# 4. Local normalized residual
#
# Goal:
#
# Determine whether removing the slowly varying periodic
# background makes the target fingerprint easier to recover.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "residual_matching"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "residual_matching_results.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

SEARCH_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

SUPERSAMPLE = 2

TEMPLATE_SIZE = 10

TOP_K = [
    1,
    5,
    10,
    50,
    100,
    250,
    500,
    1000,
]

SEEDS = range(
    20260850,
    20260880,
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]


BASE_TARGET_X = 75.25
BASE_TARGET_Y = 113.75

BASE_PITCH = 0.50
BASE_LINE_WIDTH = 0.20


# ============================================================
# PERIODIC SCENE
# ============================================================

def periodic_scene(
    x,
    y,
):
    px = np.mod(
        x,
        BASE_PITCH,
    )

    py = np.mod(
        y,
        BASE_PITCH,
    )

    vertical = (
        px
        < BASE_LINE_WIDTH
    )

    horizontal = (
        py
        < BASE_LINE_WIDTH
    )

    return np.where(
        vertical | horizontal,
        235.0,
        45.0,
    )


# ============================================================
# QUASI-PERIODIC SCENE
# ============================================================

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
        px
        < width
    )

    horizontal = (
        py
        < width
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
# TARGET FINGERPRINT
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
# SENSOR RENDER
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
    high_width = (
        width
        * SUPERSAMPLE
    )

    high_height = (
        height
        * SUPERSAMPLE
    )

    high_ppu = (
        ppu
        * SUPERSAMPLE
    )

    step = (
        1.0
        / high_ppu
    )

    xs = (
        origin_x
        + (
            np.arange(
                high_width
            )
            + 0.5
        )
        * step
    )

    ys = (
        origin_y
        + (
            np.arange(
                high_height
            )
            + 0.5
        )
        * step
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

    target = target_mask(
        X,
        Y,
        tx,
        ty,
    )

    image = np.where(
        target,
        255.0,
        image,
    )

    image = np.clip(
        image,
        0,
        255,
    ).astype(
        np.uint8
    )

    return cv2.resize(
        image,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
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

    ref_size = int(
        round(
            physical_fov
            * REFERENCE_PPU
        )
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
        ref_size,
        ref_size,
        REFERENCE_PPU,
        origin_x,
        origin_y,
        tx,
        ty,
        scene_type,
        seed,
    )

    return cv2.resize(
        reference,
        (
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# REPRESENTATIONS
# ============================================================

def raw_representation(
    image,
):
    return image.astype(
        np.float32
    )


def highpass_representation(
    image,
):
    image = image.astype(
        np.float32
    )

    blur = cv2.GaussianBlur(
        image,
        (
            0,
            0,
        ),
        sigmaX=1.2,
    )

    return (
        image
        - blur
    )


def dog_representation(
    image,
):
    image = image.astype(
        np.float32
    )

    g1 = cv2.GaussianBlur(
        image,
        (
            0,
            0,
        ),
        sigmaX=0.8,
    )

    g2 = cv2.GaussianBlur(
        image,
        (
            0,
            0,
        ),
        sigmaX=2.5,
    )

    return (
        g1
        - g2
    )


def local_normalized_representation(
    image,
):
    image = image.astype(
        np.float32
    )

    mean = cv2.GaussianBlur(
        image,
        (
            0,
            0,
        ),
        sigmaX=2.0,
    )

    squared = (
        image
        * image
    )

    mean_squared = (
        cv2.GaussianBlur(
            squared,
            (
                0,
                0,
            ),
            sigmaX=2.0,
        )
    )

    variance = (
        mean_squared
        - mean * mean
    )

    variance = np.maximum(
        variance,
        1e-6,
    )

    std = np.sqrt(
        variance
    )

    return (
        image
        - mean
    ) / (
        std
        + 1e-6
    )


# ============================================================
# NORMALIZE FOR TEMPLATE MATCHING
# ============================================================

def normalize_for_matching(
    image,
):
    image = image.astype(
        np.float32
    )

    finite = np.isfinite(
        image
    )

    if not np.any(
        finite
    ):
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    low = np.percentile(
        image[finite],
        1,
    )

    high = np.percentile(
        image[finite],
        99,
    )

    if high <= low:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    image = (
        image
        - low
    ) / (
        high
        - low
    )

    return np.clip(
        image,
        0,
        1,
    ).astype(
        np.float32
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
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

    representations = {
        "raw": (
            raw_representation
        ),
        "highpass": (
            highpass_representation
        ),
        "dog": (
            dog_representation
        ),
        "localnorm": (
            local_normalized_representation
        ),
    }

    gt_x = (
        tx
        * SEARCH_PPU
    )

    gt_y = (
        ty
        * SEARCH_PPU
    )

    gt_left = int(
        round(
            gt_x
            - TEMPLATE_SIZE / 2
        )
    )

    gt_top = int(
        round(
            gt_y
            - TEMPLATE_SIZE / 2
        )
    )

    output = {}

    for name, transform in (
        representations.items()
    ):

        search_r = transform(
            search
        )

        reference_r = transform(
            reference
        )

        search_r = (
            normalize_for_matching(
                search_r
            )
        )

        reference_r = (
            normalize_for_matching(
                reference_r
            )
        )

        response = cv2.matchTemplate(
            search_r,
            reference_r,
            cv2.TM_CCOEFF_NORMED,
        )

        gt_score = float(
            response[
                gt_top,
                gt_left,
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

        best_x, best_y = best_loc

        error = float(
            np.hypot(
                best_x
                - gt_left,
                best_y
                - gt_top,
            )
        )

        output[
            f"{name}_gt_score"
        ] = gt_score

        output[
            f"{name}_rank"
        ] = rank

        output[
            f"{name}_error"
        ] = error

    return output


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
        "MICRONYX STEP 19"
    )
    print(
        "RESIDUAL / LOCAL-CONTRAST MATCHING"
    )
    print("=" * 76)
    print()

    print(
        f"Scenes:       "
        f"{len(SEEDS) * len(SCENE_TYPES)}"
    )

    print(
        "Representations:"
    )

    print(
        "  raw"
    )

    print(
        "  highpass"
    )

    print(
        "  difference-of-Gaussians"
    )

    print(
        "  local-normalized"
    )

    print()

    results = []

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    completed = 0

    for scene_type in SCENE_TYPES:

        print()
        print(
            f"SCENE: {scene_type.upper()}"
        )

        print("-" * 76)

        for seed in SEEDS:

            result = evaluate(
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

            completed += 1

            print(
                f"Scene "
                f"{completed:02d}/{total}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    keys = [
        key
        for key in results[0]
        if key not in [
            "scene",
            "seed",
        ]
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "scene,seed,"
            + ",".join(keys)
            + "\n"
        )

        for result in results:

            f.write(
                f"{result['scene']},"
                f"{result['seed']},"
                + ",".join(
                    f"{result[key]:.8f}"
                    if "score"
                    in key
                    else str(
                        result[key]
                    )
                    for key in keys
                )
                + "\n"
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

    methods = [
        "raw",
        "highpass",
        "dog",
        "localnorm",
    ]

    for scene_type in SCENE_TYPES:

        subset = [
            r
            for r in results
            if r["scene"]
            == scene_type
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
                        f"{method}_gt_score"
                    ]
                    for r in subset
                ]
            )

            print(
                f"{method:<12}"
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