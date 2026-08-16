from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX — STEP 12 FINAL
# Sub-Pixel Sampling Alignment Test
#
# Correct acquisition model:
#
# SAME physical scene
#       |
#       +---- Search image
#       |       + target fingerprint
#       |
#       +---- Reference image
#               + same target fingerprint
#
# The periodic background is rendered once.
# The target is then inserted into the search at each phase.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "sampling_alignment"
)

OUTPUT_RESULTS = (
    OUTPUT_DIR
    / "sampling_alignment_results.csv"
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
# PHYSICAL SCENE
# ============================================================

PHYSICAL_WIDTH = 200.0
PHYSICAL_HEIGHT = 200.0

BASE_PITCH = 0.5
BASE_LINE_WIDTH = 0.2

PITCH_VARIATION = 0.06

PITCH_WAVELENGTH_X = 37.0
PITCH_WAVELENGTH_Y = 43.0

PHASE_AMPLITUDE = 0.08

PHASE_WAVELENGTH_X = 29.0
PHASE_WAVELENGTH_Y = 31.0


# ============================================================
# TARGET FINGERPRINT
# ============================================================

FINGERPRINT_WIDTH = 1.0
FINGERPRINT_HEIGHT = 1.0

DEFECT_WIDTH_INCREASE = 0.12
DEFECT_LENGTH = 0.38


# ============================================================
# BASE TARGET
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
# STRUCTURE
# ============================================================

def local_pitch(x, y):

    variation = (
        PITCH_VARIATION
        * np.sin(
            2.0
            * np.pi
            * x
            / PITCH_WAVELENGTH_X
        )
        * np.sin(
            2.0
            * np.pi
            * y
            / PITCH_WAVELENGTH_Y
        )
    )

    return (
        BASE_PITCH
        * (
            1.0
            + variation
        )
    )


def phase_warp_x(x, y):

    return (
        PHASE_AMPLITUDE
        * np.sin(
            2.0
            * np.pi
            * x
            / PHASE_WAVELENGTH_X
        )
        * np.cos(
            2.0
            * np.pi
            * y
            / PHASE_WAVELENGTH_Y
        )
    )


def phase_warp_y(x, y):

    return (
        PHASE_AMPLITUDE
        * np.cos(
            2.0
            * np.pi
            * x
            / PHASE_WAVELENGTH_X
        )
        * np.sin(
            2.0
            * np.pi
            * y
            / PHASE_WAVELENGTH_Y
        )
    )


def base_structure_mask(x, y):

    pitch = local_pitch(
        x,
        y,
    )

    warped_x = (
        x
        + phase_warp_x(
            x,
            y,
        )
    )

    warped_y = (
        y
        + phase_warp_y(
            x,
            y,
        )
    )

    x_phase = np.mod(
        warped_x,
        pitch,
    )

    y_phase = np.mod(
        warped_y,
        pitch,
    )

    vertical = (
        x_phase
        < BASE_LINE_WIDTH
    )

    horizontal = (
        y_phase
        < BASE_LINE_WIDTH
    )

    return (
        vertical
        | horizontal
    )


# ============================================================
# FINGERPRINT
# ============================================================

def fingerprint_mask(
    x,
    y,
    target_x,
    target_y,
):

    inside = (
        (
            np.abs(
                x - target_x
            )
            <= FINGERPRINT_WIDTH / 2.0
        )
        &
        (
            np.abs(
                y - target_y
            )
            <= FINGERPRINT_HEIGHT / 2.0
        )
    )

    defect_1 = (
        (
            np.abs(
                x
                - (
                    target_x
                    - 0.22
                )
            )
            <
            (
                BASE_LINE_WIDTH
                + DEFECT_WIDTH_INCREASE
            )
            / 2.0
        )
        &
        (
            np.abs(
                y
                - (
                    target_y
                    - 0.18
                )
            )
            <
            DEFECT_LENGTH / 2.0
        )
    )

    defect_2 = (
        (
            np.abs(
                x
                - (
                    target_x
                    + 0.21
                )
            )
            <
            DEFECT_LENGTH / 2.0
        )
        &
        (
            np.abs(
                y
                - (
                    target_y
                    + 0.17
                )
            )
            <
            (
                BASE_LINE_WIDTH
                + DEFECT_WIDTH_INCREASE
            )
            / 2.0
        )
    )

    return (
        inside
        &
        (
            defect_1
            | defect_2
        )
    )


# ============================================================
# RENDER BACKGROUND ONCE
# ============================================================

def render_background():

    high_size = (
        SEARCH_SIZE
        * SUPERSAMPLE
    )

    high_ppu = (
        SEARCH_PPU
        * SUPERSAMPLE
    )

    step = (
        1.0
        / high_ppu
    )

    x = (
        (
            np.arange(high_size)
            + 0.5
        )
        * step
    )

    y = (
        (
            np.arange(high_size)
            + 0.5
        )
        * step
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    background = base_structure_mask(
        X,
        Y,
    )

    background = (
        background.astype(
            np.uint8
        )
        * 255
    )

    background = cv2.resize(
        background,
        (
            SEARCH_SIZE,
            SEARCH_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return background


# ============================================================
# RENDER SEARCH WITH TARGET
# ============================================================

def render_search_with_target(
    target_x,
    target_y,
):

    high_size = (
        SEARCH_SIZE
        * SUPERSAMPLE
    )

    high_ppu = (
        SEARCH_PPU
        * SUPERSAMPLE
    )

    step = (
        1.0
        / high_ppu
    )

    x = (
        (
            np.arange(high_size)
            + 0.5
        )
        * step
    )

    y = (
        (
            np.arange(high_size)
            + 0.5
        )
        * step
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    target = fingerprint_mask(
        X,
        Y,
        target_x,
        target_y,
    )

    target = (
        target.astype(
            np.uint8
        )
        * 255
    )

    target = cv2.resize(
        target,
        (
            SEARCH_SIZE,
            SEARCH_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return target


# ============================================================
# RENDER REFERENCE
# ============================================================

def render_reference(
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

    structure = base_structure_mask(
        X,
        Y,
    )

    fingerprint = fingerprint_mask(
        X,
        Y,
        target_x,
        target_y,
    )

    image = (
        structure
        | fingerprint
    )

    image = (
        image.astype(
            np.uint8
        )
        * 255
    )

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
# ONE PHASE
# ============================================================

def run_phase(
    phase_x,
    phase_y,
    background,
):

    # --------------------------------------------------------
    # Desired search pixel coordinate
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
        np.floor(base_search_x)
        + phase_x
    )

    search_y = (
        np.floor(base_search_y)
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

    # --------------------------------------------------------
    # Create target-only image
    # --------------------------------------------------------

    target = render_search_with_target(
        target_x,
        target_y,
    )

    # Combine background + target.
    #
    # Target is white structure on black background.
    #
    search = np.maximum(
        background,
        target,
    )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = render_reference(
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

    # --------------------------------------------------------
    # Ground truth top-left
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
    # Match
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
    # FULL RESULT-MAP GT RANK
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
        "MICRONYX STEP 12 — CORRECTED"
    )
    print(
        "SUB-PIXEL SAMPLING ALIGNMENT TEST"
    )
    print("=" * 76)
    print()

    print(
        "Rendering periodic background once..."
    )

    background = render_background()

    print(
        "Background ready."
    )

    print()

    total = (
        len(PHASES)
        ** 2
    )

    print(
        f"Testing {total} phase combinations..."
    )

    print()

    results = []

    completed = 0

    for phase_x in PHASES:

        for phase_y in PHASES:

            result = run_phase(
                phase_x,
                phase_y,
                background,
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
        f"Median localization error:  "
        f"{np.median(errors):.4f}px"
    )

    print(
        f"Maximum localization error: "
        f"{np.max(errors):.4f}px"
    )

    print(
        f"Median GT rank:              "
        f"{np.median(ranks):.1f}"
    )

    print(
        f"Worst GT rank:               "
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
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
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

            file.write(
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