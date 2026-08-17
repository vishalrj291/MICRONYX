from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 13
# APERIODIC SUB-PIXEL SAMPLING TEST
#
# PURPOSE
# -------
# Separate:
#
#   1. Sampling/acquisition errors
#   2. Periodic ambiguity
#
# We deliberately REMOVE the periodic semiconductor lattice.
#
# The scene contains deterministic random texture plus a unique
# asymmetric target fingerprint.
#
# If localization remains stable across sub-pixel phases,
# sampling is not the primary problem.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "aperiodic_sampling"
)

OUTPUT_RESULTS = (
    OUTPUT_DIR
    / "aperiodic_sampling_results.csv"
)


# ============================================================
# GEOMETRY
# ============================================================

SEARCH_SIZE = 1000
REFERENCE_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

TEMPLATE_SIZE = 100

SUPERSAMPLE = 4


# ============================================================
# SCENE
# ============================================================

PHYSICAL_WIDTH = 200.0
PHYSICAL_HEIGHT = 200.0

TARGET_X_BASE = 75.25
TARGET_Y_BASE = 113.75


# ============================================================
# DETERMINISTIC Aperiodic Texture
# ============================================================

TEXTURE_SEED = 20260816


# ============================================================
# UNIQUE TARGET
# ============================================================

TARGET_SIZE = 1.0


# ============================================================
# PHASES
# ============================================================

PHASES = np.arange(
    0.0,
    1.0,
    0.05,
)


# ============================================================
# CREATE APERIODIC PHYSICAL TEXTURE
# ============================================================

def create_texture():

    rng = np.random.default_rng(
        TEXTURE_SEED
    )

    # Work at search resolution first.
    texture = rng.normal(
        loc=127.0,
        scale=35.0,
        size=(
            SEARCH_SIZE,
            SEARCH_SIZE,
        ),
    )

    texture = np.clip(
        texture,
        0,
        255,
    ).astype(
        np.uint8
    )

    # Smooth deterministic texture.
    texture = cv2.GaussianBlur(
        texture,
        (
            0,
            0,
        ),
        2.0,
    )

    return texture


# ============================================================
# TARGET MASK
# ============================================================

def create_target(
    target_x,
    target_y,
    width,
    height,
    ppu,
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

    # FOV origin
    origin_x = (
        target_x
        - width / 2.0
    )

    origin_y = (
        target_y
        - height / 2.0
    )

    x = (
        origin_x
        + (
            np.arange(high_width)
            + 0.5
        )
        * step
    )

    y = (
        origin_y
        + (
            np.arange(high_height)
            + 0.5
        )
        * step
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    dx = X - target_x
    dy = Y - target_y

    # --------------------------------------------------------
    # Unique asymmetric "L" shape
    # --------------------------------------------------------

    vertical = (
        (
            np.abs(
                dx + 0.20
            )
            < 0.09
        )
        &
        (
            np.abs(
                dy
            )
            < 0.38
        )
    )

    horizontal = (
        (
            np.abs(
                dx - 0.10
            )
            < 0.09
        )
        &
        (
            np.abs(
                dy + 0.25
            )
            < 0.09
        )
    )

    # --------------------------------------------------------
    # Small diagonal marker
    # --------------------------------------------------------

    diagonal = (
        np.abs(
            dy
            - (
                0.8 * dx
            )
            - 0.18
        )
        < 0.055
    ) & (
        np.abs(dx)
        < 0.35
    )

    mask = (
        vertical
        | horizontal
        | diagonal
    )

    mask = (
        mask.astype(
            np.uint8
        )
        * 255
    )

    mask = cv2.resize(
        mask,
        (
            int(width * ppu),
            int(height * ppu),
        ),
        interpolation=cv2.INTER_AREA,
    )

    return mask


# ============================================================
# CREATE SEARCH IMAGE
# ============================================================

def create_search(
    target_x,
    target_y,
):

    rng = np.random.default_rng(
        TEXTURE_SEED
    )

    search = rng.normal(
        127.0,
        35.0,
        (
            SEARCH_SIZE,
            SEARCH_SIZE,
        ),
    )

    search = np.clip(
        search,
        0,
        255,
    ).astype(
        np.uint8
    )

    search = cv2.GaussianBlur(
        search,
        (
            0,
            0,
        ),
        2.0,
    )

    # --------------------------------------------------------
    # Create unique target directly at search resolution
    # --------------------------------------------------------

    target_x_px = (
        target_x
        * SEARCH_PPU
    )

    target_y_px = (
        target_y
        * SEARCH_PPU
    )

    # Asymmetric marker
    cx = target_x_px
    cy = target_y_px

    cv2.line(
        search,
        (
            int(round(cx - 1)),
            int(round(cy - 2)),
        ),
        (
            int(round(cx - 1)),
            int(round(cy + 2)),
        ),
        255,
        1,
    )

    cv2.line(
        search,
        (
            int(round(cx - 1)),
            int(round(cy + 2)),
        ),
        (
            int(round(cx + 1)),
            int(round(cy + 2)),
        ),
        255,
        1,
    )

    cv2.line(
        search,
        (
            int(round(cx)),
            int(round(cy - 1)),
        ),
        (
            int(round(cx + 2)),
            int(round(cy)),
        ),
        255,
        1,
    )

    return search


# ============================================================
# CREATE REFERENCE
# ============================================================

def create_reference(
    target_x,
    target_y,
):

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

    high_size = (
        REFERENCE_SIZE
        * SUPERSAMPLE
    )

    high_ppu = (
        REFERENCE_PPU
        * SUPERSAMPLE
    )

    step = (
        1.0
        / high_ppu
    )

    x = (
        origin_x
        + (
            np.arange(high_size)
            + 0.5
        )
        * step
    )

    y = (
        origin_y
        + (
            np.arange(high_size)
            + 0.5
        )
        * step
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    # Deterministic local texture
    rng = np.random.default_rng(
        TEXTURE_SEED
    )

    # A smooth analytic aperiodic texture
    texture = (
        127.0
        + 25.0
        * np.sin(
            0.71 * X
            + 0.33 * Y
        )
        + 18.0
        * np.sin(
            1.17 * X
            - 0.52 * Y
        )
        + 12.0
        * np.sin(
            0.37 * X
            + 1.41 * Y
        )
    )

    texture = np.clip(
        texture,
        0,
        255,
    )

    image = texture.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Unique asymmetric target
    # --------------------------------------------------------

    dx = X - target_x
    dy = Y - target_y

    vertical = (
        (
            np.abs(
                dx + 0.20
            )
            < 0.09
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
            < 0.09
        )
        &
        (
            np.abs(
                dy + 0.25
            )
            < 0.09
        )
    )

    diagonal = (
        np.abs(
            dy
            - 0.8 * dx
            - 0.18
        )
        < 0.055
    ) & (
        np.abs(dx)
        < 0.35
    )

    target = (
        vertical
        | horizontal
        | diagonal
    )

    image[
        target
    ] = 255

    image = cv2.resize(
        image,
        (
            REFERENCE_SIZE,
            REFERENCE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return image


# ============================================================
# PHASE TEST
# ============================================================

def run_phase(
    phase_x,
    phase_y,
):

    base_x = (
        TARGET_X_BASE
        * SEARCH_PPU
    )

    base_y = (
        TARGET_Y_BASE
        * SEARCH_PPU
    )

    search_x = (
        np.floor(base_x)
        + phase_x
    )

    search_y = (
        np.floor(base_y)
        + phase_y
    )

    target_x = (
        search_x
        / SEARCH_PPU
    )

    target_y = (
        search_y
        / SEARCH_PPU
    )

    search = create_search(
        target_x,
        target_y,
    )

    reference = create_reference(
        target_x,
        target_y,
    )

    template = cv2.resize(
        reference,
        (
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

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
        "gt_score": gt_score,
        "best_score": float(best_score),
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
        "MICRONYX STEP 13"
    )
    print(
        "APERIODIC SUB-PIXEL SAMPLING TEST"
    )
    print("=" * 76)
    print()

    results = []

    total = (
        len(PHASES)
        ** 2
    )

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

    print()
    print("=" * 76)
    print(
        "SUMMARY"
    )
    print("=" * 76)
    print()

    print(
        f"GT Top-1 rate:             "
        f"{np.mean(ranks == 1) * 100:.2f}%"
    )

    print(
        f"Localization <= 1 px:      "
        f"{np.mean(errors <= 1) * 100:.2f}%"
    )

    print(
        f"Localization <= 5 px:      "
        f"{np.mean(errors <= 5) * 100:.2f}%"
    )

    print(
        f"Median GT score:            "
        f"{np.median(scores):.6f}"
    )

    print(
        f"Minimum GT score:           "
        f"{np.min(scores):.6f}"
    )

    print(
        f"Maximum GT score:           "
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
            "gt_score,"
            "best_score,"
            "error,"
            "gt_rank\n"
        )

        for r in results:

            f.write(
                f"{r['phase_x']:.4f},"
                f"{r['phase_y']:.4f},"
                f"{r['gt_score']:.8f},"
                f"{r['best_score']:.8f},"
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