from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 17
# CANDIDATE RECALL BENCHMARK
#
# Question:
#
# How large must the candidate set be before the true
# inspection location is reliably retained?
#
# This separates:
#
# 1. Candidate generation failure
# 2. Candidate verification failure
#
# We intentionally do NOT perform sophisticated ranking here.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "candidate_recall"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "candidate_recall_results.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

SEARCH_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

SUPERSAMPLE = 2

TEMPLATE_SIZE = 10

NUM_TOP_K = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
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

    image = cv2.resize(
        image,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return image


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

    reference_size = int(
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
        reference_size,
        reference_size,
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
# ONE EXPERIMENT
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

    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED,
    )

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

    gt_score = float(
        result[
            gt_top,
            gt_left,
        ]
    )

    flat = result.ravel()

    rank = (
        1
        + int(
            np.sum(
                flat
                > gt_score
            )
        )
    )

    return {
        "scene": scene_type,
        "seed": seed,
        "gt_score": gt_score,
        "gt_rank": rank,
    }


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
        "MICRONYX STEP 17"
    )
    print(
        "CANDIDATE RECALL BENCHMARK"
    )
    print("=" * 76)
    print()

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    print(
        f"Seeds:          {len(SEEDS)}"
    )

    print(
        f"Scene types:    {SCENE_TYPES}"
    )

    print(
        f"Total scenes:   {total}"
    )

    print(
        f"K values:       {NUM_TOP_K}"
    )

    print()

    raw = []

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

            raw.append(
                result
            )

            completed += 1

            print(
                f"Scene "
                f"{completed:02d}/{total} "
                f"GT rank="
                f"{result['gt_rank']}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "scene,"
            "seed,"
            "gt_score,"
            "gt_rank\n"
        )

        for r in raw:

            f.write(
                f"{r['scene']},"
                f"{r['seed']},"
                f"{r['gt_score']:.8f},"
                f"{r['gt_rank']}\n"
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

        subset = [
            r
            for r in raw
            if r["scene"]
            == scene_type
        ]

        ranks = np.array(
            [
                r["gt_rank"]
                for r in subset
            ]
        )

        print()
        print(
            scene_type.upper()
        )

        print("-" * 76)

        for k in NUM_TOP_K:

            recall = (
                np.mean(
                    ranks <= k
                )
                * 100
            )

            absent = (
                np.mean(
                    ranks > k
                )
                * 100
            )

            print(
                f"Recall@{k:<5d}: "
                f"{recall:6.2f}%   "
                f"GT absent: "
                f"{absent:6.2f}%"
            )

        print(
            f"Median GT rank: "
            f"{np.median(ranks):.1f}"
        )

        print(
            f"Mean GT rank:   "
            f"{np.mean(ranks):.1f}"
        )

        print(
            f"Worst GT rank:  "
            f"{np.max(ranks)}"
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