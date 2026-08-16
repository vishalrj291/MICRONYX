from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 15
# TEMPLATE / CONTEXT BENCHMARK
#
# PURPOSE
# -------
# Determine whether increasing spatial context reduces
# periodic localization ambiguity.
#
# IMPORTANT:
# We do NOT resize a fixed 100x100 reference to arbitrary
# sizes.
#
# Instead:
#
# Search template 10x10
#       <-> Reference 100x100
#
# Search template 20x20
#       <-> Reference 200x200
#
# This preserves the physical 10x magnification relationship.
#
# 10x10 is the PS02-equivalent baseline.
# Larger templates are research experiments.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "template_context"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "template_context_results.csv"
)


# ============================================================
# SENSOR GEOMETRY
# ============================================================

SEARCH_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

MAGNIFICATION = 10.0

SUPERSAMPLE = 2


# ============================================================
# BENCHMARK
# ============================================================

SEEDS = range(
    20260850,
    20260865,
)

TEMPLATE_SIZES = [
    5,
    10,
    15,
    20,
    25,
    30,
]


# ============================================================
# TARGET
# ============================================================

BASE_TARGET_X = 75.25
BASE_TARGET_Y = 113.75


# ============================================================
# PHYSICAL SCENE
# ============================================================

PHYSICAL_WIDTH = 200.0
PHYSICAL_HEIGHT = 200.0

BASE_PITCH = 0.50

BASE_LINE_WIDTH = 0.20


# ============================================================
# SCENE TYPES
# ============================================================

SCENE_TYPES = [
    "aperiodic",
    "periodic",
    "quasiperiodic",
]


# ============================================================
# CONTINUOUS APERIODIC SCENE
# ============================================================

def aperiodic_scene(
    x,
    y,
):
    """
    Deterministic non-periodic texture.
    """

    value = (
        128.0
        + 32.0
        * np.sin(
            0.71 * x
            + 0.33 * y
        )
        + 22.0
        * np.sin(
            1.17 * x
            - 0.52 * y
        )
        + 16.0
        * np.sin(
            0.37 * x
            + 1.41 * y
        )
        + 9.0
        * np.sin(
            2.31 * x
            + 0.83 * y
        )
    )

    return value


# ============================================================
# CONTINUOUS PERIODIC SCENE
# ============================================================

def periodic_scene(
    x,
    y,
):
    """
    Strongly periodic semiconductor-like lattice.
    """

    pitch = BASE_PITCH

    px = np.mod(
        x,
        pitch,
    )

    py = np.mod(
        y,
        pitch,
    )

    vertical = (
        px
        < BASE_LINE_WIDTH
    )

    horizontal = (
        py
        < BASE_LINE_WIDTH
    )

    image = np.where(
        vertical | horizontal,
        235.0,
        45.0,
    )

    return image


# ============================================================
# CONTINUOUS QUASI-PERIODIC SCENE
# ============================================================

def quasiperiodic_scene(
    x,
    y,
    seed,
):
    """
    Periodic structure with spatially varying pitch,
    orientation and local intensity.

    This is deliberately harder than the clean periodic case.
    """

    rng = np.random.default_rng(
        seed
    )

    # Deterministic seed-dependent parameters.
    pitch_variation = (
        rng.uniform(
            0.03,
            0.08,
        )
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
            + pitch_variation
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

    # Local coordinate warp.
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

    # Slightly variable line width.
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

    image = np.where(
        vertical | horizontal,
        235.0,
        45.0,
    )

    # Slowly varying intensity.
    intensity_modulation = (
        10.0
        * np.sin(
            0.031 * x
            + 0.017 * y
        )
    )

    image = (
        image
        + intensity_modulation
    )

    return image


# ============================================================
# UNIQUE TARGET FINGERPRINT
# ============================================================

def target_mask(
    x,
    y,
    target_x,
    target_y,
):
    """
    Unique asymmetric fingerprint.

    Same target is used in both search and reference.
    """

    dx = x - target_x
    dy = y - target_y

    # Vertical arm.
    vertical = (
        (
            np.abs(
                dx + 0.20
            )
            < 0.075
        )
        &
        (
            np.abs(
                dy
            )
            < 0.38
        )
    )

    # Horizontal arm.
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

    # Diagonal marker.
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

    # Small circular marker.
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
# BASE SCENE
# ============================================================

def render_continuous_scene(
    x,
    y,
    scene_type,
    seed,
):
    if scene_type == "aperiodic":

        return aperiodic_scene(
            x,
            y,
        )

    if scene_type == "periodic":

        return periodic_scene(
            x,
            y,
        )

    if scene_type == "quasiperiodic":

        return quasiperiodic_scene(
            x,
            y,
            seed,
        )

    raise ValueError(
        f"Unknown scene type: {scene_type}"
    )


# ============================================================
# ADD TARGET
# ============================================================

def add_target(
    image,
    x,
    y,
    target_x,
    target_y,
):
    mask = target_mask(
        x,
        y,
        target_x,
        target_y,
    )

    image = np.where(
        mask,
        255.0,
        image,
    )

    return image


# ============================================================
# RENDER SENSOR
# ============================================================

def render_sensor(
    width,
    height,
    ppu,
    origin_x,
    origin_y,
    target_x,
    target_y,
    scene_type,
    seed,
):
    """
    Render same physical scene through a specified sensor.
    """

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

    image = render_continuous_scene(
        X,
        Y,
        scene_type,
        seed,
    )

    image = add_target(
        image,
        X,
        Y,
        target_x,
        target_y,
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
# ONE EXPERIMENT
# ============================================================

def evaluate(
    scene_type,
    seed,
    template_size,
):
    """
    Evaluate one scene/template combination.
    """

    rng = np.random.default_rng(
        seed
    )

    # Slightly vary target location.
    target_x = (
        BASE_TARGET_X
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    target_y = (
        BASE_TARGET_Y
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = render_sensor(
        SEARCH_SIZE,
        SEARCH_SIZE,
        SEARCH_PPU,
        0.0,
        0.0,
        target_x,
        target_y,
        scene_type,
        seed,
    )

    # --------------------------------------------------------
    # Reference FOV
    #
    # Physical search footprint:
    #
    # template_size / SEARCH_PPU
    #
    # Reference pixels:
    #
    # physical_size * REFERENCE_PPU
    #
    # = template_size * 10
    # --------------------------------------------------------

    physical_fov = (
        template_size
        / SEARCH_PPU
    )

    reference_size = int(
        round(
            physical_fov
            * REFERENCE_PPU
        )
    )

    origin_x = (
        target_x
        - physical_fov / 2.0
    )

    origin_y = (
        target_y
        - physical_fov / 2.0
    )

    reference = render_sensor(
        reference_size,
        reference_size,
        REFERENCE_PPU,
        origin_x,
        origin_y,
        target_x,
        target_y,
        scene_type,
        seed,
    )

    # --------------------------------------------------------
    # Convert reference to search scale
    # --------------------------------------------------------

    template = cv2.resize(
        reference,
        (
            template_size,
            template_size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # Ground truth location
    # --------------------------------------------------------

    target_search_x = (
        target_x
        * SEARCH_PPU
    )

    target_search_y = (
        target_y
        * SEARCH_PPU
    )

    gt_left = int(
        round(
            target_search_x
            - template_size / 2.0
        )
    )

    gt_top = int(
        round(
            target_search_y
            - template_size / 2.0
        )
    )

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED,
    )

    gt_score = float(
        result[
            gt_top,
            gt_left,
        ]
    )

    _, best_score, _, location = (
        cv2.minMaxLoc(
            result
        )
    )

    predicted_x = (
        location[0]
        + template_size / 2.0
    )

    predicted_y = (
        location[1]
        + template_size / 2.0
    )

    error = float(
        np.hypot(
            predicted_x
            - target_search_x,
            predicted_y
            - target_search_y,
        )
    )

    # --------------------------------------------------------
    # GT rank
    # --------------------------------------------------------

    flat = result.ravel()

    gt_index = (
        gt_top
        * result.shape[1]
        + gt_left
    )

    gt_value = flat[
        gt_index
    ]

    gt_rank = (
        1
        + int(
            np.sum(
                flat
                > gt_value
            )
        )
    )

    # --------------------------------------------------------
    # Strongest false candidate
    # --------------------------------------------------------

    suppressed = result.copy()

    radius = max(
        2,
        template_size,
    )

    x0 = max(
        0,
        gt_left - radius,
    )

    x1 = min(
        result.shape[1],
        gt_left + radius + 1,
    )

    y0 = max(
        0,
        gt_top - radius,
    )

    y1 = min(
        result.shape[0],
        gt_top + radius + 1,
    )

    suppressed[
        y0:y1,
        x0:x1
    ] = -np.inf

    _, strongest_negative, _, _ = (
        cv2.minMaxLoc(
            suppressed
        )
    )

    margin = float(
        gt_score
        - strongest_negative
    )

    return {
        "scene": scene_type,
        "seed": seed,
        "template_size": template_size,
        "reference_size": reference_size,
        "gt_score": gt_score,
        "best_score": float(
            best_score
        ),
        "strongest_negative": float(
            strongest_negative
        ),
        "margin": margin,
        "error": error,
        "gt_rank": gt_rank,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
        * len(TEMPLATE_SIZES)
    )

    print()
    print("=" * 76)
    print(
        "MICRONYX STEP 15"
    )
    print(
        "TEMPLATE / CONTEXT BENCHMARK"
    )
    print("=" * 76)
    print()

    print(
        f"Scenes:             {len(SEEDS)}"
    )

    print(
        f"Scene types:        {len(SCENE_TYPES)}"
    )

    print(
        f"Template sizes:     {TEMPLATE_SIZES}"
    )

    print(
        f"Total experiments:  {total}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "10x10 is the PS02-equivalent baseline."
    )

    print(
        "Larger templates represent additional"
    )

    print(
        "physical context and are R&D experiments."
    )

    print()

    results = []

    completed = 0

    for scene_type in SCENE_TYPES:

        print()
        print(
            f"SCENE TYPE: {scene_type.upper()}"
        )

        print("-" * 76)

        for seed in SEEDS:

            for template_size in TEMPLATE_SIZES:

                result = evaluate(
                    scene_type,
                    seed,
                    template_size,
                )

                results.append(
                    result
                )

                completed += 1

                if (
                    completed % 20
                    == 0
                ):
                    print(
                        f"Progress: "
                        f"{completed}/{total}"
                    )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "scene,"
            "seed,"
            "template_size,"
            "reference_size,"
            "gt_score,"
            "best_score,"
            "strongest_negative,"
            "margin,"
            "error,"
            "gt_rank\n"
        )

        for r in results:

            f.write(
                f"{r['scene']},"
                f"{r['seed']},"
                f"{r['template_size']},"
                f"{r['reference_size']},"
                f"{r['gt_score']:.8f},"
                f"{r['best_score']:.8f},"
                f"{r['strongest_negative']:.8f},"
                f"{r['margin']:.8f},"
                f"{r['error']:.8f},"
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

        print()
        print(
            f"{scene_type.upper()}"
        )

        print("-" * 76)

        scene_results = [
            r
            for r in results
            if r["scene"]
            == scene_type
        ]

        for template_size in TEMPLATE_SIZES:

            subset = [
                r
                for r in scene_results
                if r["template_size"]
                == template_size
            ]

            errors = np.array(
                [
                    r["error"]
                    for r in subset
                ]
            )

            ranks = np.array(
                [
                    r["gt_rank"]
                    for r in subset
                ]
            )

            margins = np.array(
                [
                    r["margin"]
                    for r in subset
                ]
            )

            top1 = (
                np.mean(
                    ranks == 1
                )
                * 100.0
            )

            top5 = (
                np.mean(
                    ranks <= 5
                )
                * 100.0
            )

            within5 = (
                np.mean(
                    errors <= 5.0
                )
                * 100.0
            )

            print(
                f"{template_size:2d}x"
                f"{template_size:<2d} "
                f"Top1={top1:6.2f}% "
                f"Top5={top5:6.2f}% "
                f"<=5px={within5:6.2f}% "
                f"MedErr={np.median(errors):7.3f}px "
                f"MedRank={np.median(ranks):6.1f} "
                f"MedMargin={np.median(margins):8.5f}"
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