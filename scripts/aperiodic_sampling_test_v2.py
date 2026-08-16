from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 14
# CORRECTED APERIODIC SUB-PIXEL SAMPLING TEST
#
# SAME CONTINUOUS SCENE -> TWO DIFFERENT SAMPLING SYSTEMS
#
# Search:
#   1000 x 1000
#   5 px/unit
#
# Reference:
#   100 x 100
#   50 px/unit
#
# The underlying physical image is IDENTICAL.
# Only acquisition resolution/FOV changes.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "aperiodic_sampling_v2"
)

OUTPUT_RESULTS = (
    OUTPUT_DIR
    / "aperiodic_sampling_v2_results.csv"
)


# ============================================================
# GEOMETRY
# ============================================================

SEARCH_SIZE = 1000
REFERENCE_SIZE = 100

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

TEMPLATE_SIZE = 10

SUPERSAMPLE = 4


# ============================================================
# TARGET
# ============================================================

BASE_TARGET_X = 75.25
BASE_TARGET_Y = 113.75


# ============================================================
# PHASE GRID
# ============================================================

PHASES = np.arange(
    0.0,
    1.0,
    0.05,
)


# ============================================================
# CONTINUOUS APERIODIC SCENE
# ============================================================

def scene_intensity(
    x,
    y,
):
    """
    Deterministic aperiodic continuous image.

    This EXACT function is used for both search and reference.
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

    return np.clip(
        value,
        0,
        255,
    )


# ============================================================
# TARGET STRUCTURE
# ============================================================

def target_mask(
    x,
    y,
    target_x,
    target_y,
):
    """
    Unique asymmetric target.

    This exact mathematical target is used for both
    search and reference.
    """

    dx = x - target_x
    dy = y - target_y

    # Vertical arm
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

    # Horizontal arm
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

    # Diagonal asymmetric marker
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

    # Small circular marker
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
# CONTINUOUS SCENE
# ============================================================

def continuous_scene(
    x,
    y,
    target_x,
    target_y,
):
    """
    Same physical scene for both sensors.
    """

    image = scene_intensity(
        x,
        y,
    )

    target = target_mask(
        x,
        y,
        target_x,
        target_y,
    )

    # Target is bright.
    image = np.where(
        target,
        255.0,
        image,
    )

    return image


# ============================================================
# RENDER SENSOR IMAGE
# ============================================================

def render_sensor(
    width,
    height,
    pixels_per_unit,
    origin_x,
    origin_y,
    target_x,
    target_y,
):
    """
    Render the same continuous scene at a specified
    sampling density.
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
        pixels_per_unit
        * SUPERSAMPLE
    )

    step = (
        1.0
        / high_ppu
    )

    x = (
        origin_x
        + (
            np.arange(
                high_width
            )
            + 0.5
        )
        * step
    )

    y = (
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
        x,
        y,
    )

    image = continuous_scene(
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
# ONE PHASE EXPERIMENT
# ============================================================

def run_phase(
    phase_x,
    phase_y,
):
    # --------------------------------------------------------
    # Search coordinate
    # --------------------------------------------------------

    base_search_x = (
        BASE_TARGET_X
        * SEARCH_PPU
    )

    base_search_y = (
        BASE_TARGET_Y
        * SEARCH_PPU
    )

    search_x = (
        np.floor(
            base_search_x
        )
        + phase_x
    )

    search_y = (
        np.floor(
            base_search_y
        )
        + phase_y
    )

    # Physical target coordinate
    target_x = (
        search_x
        / SEARCH_PPU
    )

    target_y = (
        search_y
        / SEARCH_PPU
    )

    # --------------------------------------------------------
    # Search image
    # --------------------------------------------------------

    search = render_sensor(
        SEARCH_SIZE,
        SEARCH_SIZE,
        SEARCH_PPU,
        0.0,
        0.0,
        target_x,
        target_y,
    )

    # --------------------------------------------------------
    # Reference FOV
    # --------------------------------------------------------

    fov_width = (
        REFERENCE_SIZE
        / REFERENCE_PPU
    )

    fov_height = (
        REFERENCE_SIZE
        / REFERENCE_PPU
    )

    origin_x = (
        target_x
        - fov_width / 2.0
    )

    origin_y = (
        target_y
        - fov_height / 2.0
    )

    reference = render_sensor(
        REFERENCE_SIZE,
        REFERENCE_SIZE,
        REFERENCE_PPU,
        origin_x,
        origin_y,
        target_x,
        target_y,
    )

    # --------------------------------------------------------
    # Convert reference to search scale
    # --------------------------------------------------------

    template = cv2.resize(
        reference,
        (
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_left = int(
        round(
            search_x
            - TEMPLATE_SIZE / 2.0
        )
    )

    gt_top = int(
        round(
            search_y
            - TEMPLATE_SIZE / 2.0
        )
    )

    # --------------------------------------------------------
    # Template matching
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
        + TEMPLATE_SIZE / 2.0
    )

    predicted_y = (
        location[1]
        + TEMPLATE_SIZE / 2.0
    )

    error = float(
        np.sqrt(
            (
                predicted_x
                - search_x
            ) ** 2
            +
            (
                predicted_y
                - search_y
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # Exact GT rank
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

    return {
        "phase_x": phase_x,
        "phase_y": phase_y,
        "target_x": target_x,
        "target_y": target_y,
        "search_x": search_x,
        "search_y": search_y,
        "gt_score": gt_score,
        "best_score": float(
            best_score
        ),
        "predicted_x": predicted_x,
        "predicted_y": predicted_y,
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

    print()
    print("=" * 76)
    print(
        "MICRONYX STEP 14"
    )
    print(
        "CORRECTED APERIODIC SAMPLING TEST"
    )
    print("=" * 76)
    print()

    print(
        "Underlying scene:"
    )

    print(
        "  ONE continuous mathematical image"
    )

    print(
        "  SAME target"
    )

    print(
        "  SAME physical coordinates"
    )

    print(
        "  Different sensor sampling only"
    )

    print()

    total = (
        len(PHASES)
        ** 2
    )

    results = []

    completed = 0

    print(
        f"Testing {total} phase combinations..."
    )

    print()

    for phase_x in PHASES:

        for phase_y in PHASES:

            result = run_phase(
                phase_x,
                phase_y,
            )

            results.append(
                result
            )

            completed += 1

            if (
                completed % 25
                == 0
            ):
                print(
                    f"Progress: "
                    f"{completed}/{total}"
                )

    # ========================================================
    # STATISTICS
    # ========================================================

    errors = np.array(
        [
            r["error"]
            for r in results
        ]
    )

    scores = np.array(
        [
            r["gt_score"]
            for r in results
        ]
    )

    ranks = np.array(
        [
            r["gt_rank"]
            for r in results
        ]
    )

    top1 = (
        np.mean(
            ranks == 1
        )
        * 100.0
    )

    within_1 = (
        np.mean(
            errors <= 1.0
        )
        * 100.0
    )

    within_5 = (
        np.mean(
            errors <= 5.0
        )
        * 100.0
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
    print()

    print(
        f"Phase combinations:        "
        f"{len(results)}"
    )

    print(
        f"GT Top-1 rate:              "
        f"{top1:.2f}%"
    )

    print(
        f"Localization <= 1 px:       "
        f"{within_1:.2f}%"
    )

    print(
        f"Localization <= 5 px:       "
        f"{within_5:.2f}%"
    )

    print(
        f"Median GT score:             "
        f"{np.median(scores):.6f}"
    )

    print(
        f"Minimum GT score:            "
        f"{np.min(scores):.6f}"
    )

    print(
        f"Maximum GT score:            "
        f"{np.max(scores):.6f}"
    )

    print(
        f"Median error:               "
        f"{np.median(errors):.4f}px"
    )

    print(
        f"Maximum error:              "
        f"{np.max(errors):.4f}px"
    )

    print(
        f"Median GT rank:             "
        f"{np.median(ranks):.1f}"
    )

    print(
        f"Worst GT rank:              "
        f"{np.max(ranks)}"
    )

    # ========================================================
    # WORST CASES
    # ========================================================

    print()
    print(
        "WORST 10 CASES"
    )

    print("-" * 76)

    worst = sorted(
        results,
        key=lambda r:
            r["error"],
        reverse=True,
    )

    for r in worst[:10]:

        print(
            f"phase=({r['phase_x']:.2f},"
            f"{r['phase_y']:.2f}) "
            f"GT_score={r['gt_score']:.5f} "
            f"error={r['error']:.3f}px "
            f"rank={r['gt_rank']}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "phase_x,"
            "phase_y,"
            "target_x,"
            "target_y,"
            "search_x,"
            "search_y,"
            "gt_score,"
            "best_score,"
            "predicted_x,"
            "predicted_y,"
            "error,"
            "gt_rank\n"
        )

        for r in results:

            f.write(
                f"{r['phase_x']:.4f},"
                f"{r['phase_y']:.4f},"
                f"{r['target_x']:.8f},"
                f"{r['target_y']:.8f},"
                f"{r['search_x']:.8f},"
                f"{r['search_y']:.8f},"
                f"{r['gt_score']:.8f},"
                f"{r['best_score']:.8f},"
                f"{r['predicted_x']:.8f},"
                f"{r['predicted_y']:.8f},"
                f"{r['error']:.8f},"
                f"{r['gt_rank']}\n"
            )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_RESULTS
    )

    print()
    print("=" * 76)


if __name__ == "__main__":
    main()