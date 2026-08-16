from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX — STEP 9
# Natural Quasi-Periodic Hard-Negative Experiment
#
# IMPORTANT:
# This is an EXPERIMENT.
# It does NOT modify generator_v02.py.
#
# No target patch is copied into other locations.
#
# Search and reference are rendered from the SAME continuous
# spatially varying physical geometry.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

V02_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
)

OUTPUT_SEARCH = (
    V02_DIR
    / "natural_periodic_search.png"
)

OUTPUT_REFERENCE = (
    V02_DIR
    / "natural_periodic_reference.png"
)

OUTPUT_VISUAL = (
    V02_DIR
    / "natural_periodic_candidates.png"
)

OUTPUT_RESULTS = (
    V02_DIR
    / "natural_periodic_results.txt"
)


# ============================================================
# IMAGE GEOMETRY
# ============================================================

SEARCH_WIDTH = 1000
SEARCH_HEIGHT = 1000

REFERENCE_WIDTH = 100
REFERENCE_HEIGHT = 100

MAGNIFICATION = 10.0


# ============================================================
# PHYSICAL SCENE
# ============================================================

PHYSICAL_WIDTH = 200.0
PHYSICAL_HEIGHT = 200.0

SEARCH_PIXELS_PER_UNIT = (
    SEARCH_WIDTH
    / PHYSICAL_WIDTH
)

REFERENCE_PIXELS_PER_UNIT = (
    SEARCH_PIXELS_PER_UNIT
    * MAGNIFICATION
)


# ============================================================
# BASE STRUCTURE
# ============================================================

BASE_PITCH = 0.5
BASE_LINE_WIDTH = 0.2


# ============================================================
# QUASI-PERIODICITY
# ============================================================
#
# The pitch is NOT constant everywhere.
#
# Instead, a smooth spatial modulation changes the local
# structure gradually.
#
# This produces naturally similar-but-not-identical regions.
# ============================================================

PITCH_VARIATION = 0.06

PITCH_WAVELENGTH_X = 37.0
PITCH_WAVELENGTH_Y = 43.0

PHASE_AMPLITUDE = 0.08

PHASE_WAVELENGTH_X = 29.0
PHASE_WAVELENGTH_Y = 31.0


# ============================================================
# TARGET
# ============================================================

TARGET_X = 75.25
TARGET_Y = 113.75


# ============================================================
# LOCAL FINGERPRINT
# ============================================================
#
# Same fingerprint concept as generator_v02.
# It is generated from the physical coordinate, NOT copied.
# ============================================================

FINGERPRINT_WIDTH = 1.0
FINGERPRINT_HEIGHT = 1.0

DEFECT_1_OFFSET_X = -0.22
DEFECT_1_OFFSET_Y = -0.18

DEFECT_2_OFFSET_X = 0.21
DEFECT_2_OFFSET_Y = 0.17

DEFECT_WIDTH_INCREASE = 0.12
DEFECT_LENGTH = 0.38


# ============================================================
# RENDERING
# ============================================================

SUPERSAMPLE = 4


# ============================================================
# REFERENCE FOV
# ============================================================

REFERENCE_FOV_WIDTH = (
    REFERENCE_WIDTH
    / REFERENCE_PIXELS_PER_UNIT
)

REFERENCE_FOV_HEIGHT = (
    REFERENCE_HEIGHT
    / REFERENCE_PIXELS_PER_UNIT
)


# ============================================================
# LOCAL PITCH
# ============================================================

def local_pitch(
    x,
    y,
):
    """
    Smooth spatial variation of the base pitch.

    The pitch remains close to BASE_PITCH but changes
    continuously over the physical scene.
    """

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


# ============================================================
# SPATIAL PHASE WARP
# ============================================================

def phase_warp_x(
    x,
    y,
):
    """
    Smooth coordinate warp.

    This prevents the scene from being merely a collection
    of independently generated blocks.
    """

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


def phase_warp_y(
    x,
    y,
):
    """
    Smooth Y-direction coordinate warp.
    """

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


# ============================================================
# BASE QUASI-PERIODIC STRUCTURE
# ============================================================

def base_structure_mask(
    x,
    y,
):
    """
    Generate a continuous quasi-periodic grid.

    The same function is used for search and reference.

    This is deliberately deterministic.
    """

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
# LOCAL FINGERPRINT
# ============================================================

def fingerprint_mask(
    x,
    y,
):
    """
    Same asymmetric target fingerprint used in the clean
    experiment.
    """

    dx = (
        x
        - TARGET_X
    )

    dy = (
        y
        - TARGET_Y
    )

    inside_region = (
        (np.abs(dx) <= FINGERPRINT_WIDTH / 2.0)
        &
        (np.abs(dy) <= FINGERPRINT_HEIGHT / 2.0)
    )

    # --------------------------------------------------------
    # Defect 1: short vertical thickening
    # --------------------------------------------------------

    defect_1_x = (
        TARGET_X
        + DEFECT_1_OFFSET_X
    )

    defect_1_y = (
        TARGET_Y
        + DEFECT_1_OFFSET_Y
    )

    defect_1 = (
        (np.abs(x - defect_1_x)
         < (
             BASE_LINE_WIDTH
             + DEFECT_WIDTH_INCREASE
         ) / 2.0)
        &
        (np.abs(y - defect_1_y)
         < DEFECT_LENGTH / 2.0)
    )

    # --------------------------------------------------------
    # Defect 2: short horizontal thickening
    # --------------------------------------------------------

    defect_2_x = (
        TARGET_X
        + DEFECT_2_OFFSET_X
    )

    defect_2_y = (
        TARGET_Y
        + DEFECT_2_OFFSET_Y
    )

    defect_2 = (
        (np.abs(x - defect_2_x)
         < DEFECT_LENGTH / 2.0)
        &
        (np.abs(y - defect_2_y)
         < (
             BASE_LINE_WIDTH
             + DEFECT_WIDTH_INCREASE
         ) / 2.0)
    )

    return (
        inside_region
        &
        (
            defect_1
            | defect_2
        )
    )


# ============================================================
# COMPLETE SCENE
# ============================================================

def render_scene(
    width_px,
    height_px,
    pixels_per_unit,
    origin_x,
    origin_y,
):
    """
    Render one physical FOV from the SAME continuous scene.
    """

    high_width = (
        width_px
        * SUPERSAMPLE
    )

    high_height = (
        height_px
        * SUPERSAMPLE
    )

    high_ppu = (
        pixels_per_unit
        * SUPERSAMPLE
    )

    dx = (
        1.0
        / high_ppu
    )

    dy = (
        1.0
        / high_ppu
    )

    x = (
        origin_x
        + (
            np.arange(high_width)
            + 0.5
        )
        * dx
    )

    y = (
        origin_y
        + (
            np.arange(high_height)
            + 0.5
        )
        * dy
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    # --------------------------------------------------------
    # Base geometry
    # --------------------------------------------------------

    base = base_structure_mask(
        X,
        Y,
    )

    # --------------------------------------------------------
    # Target fingerprint
    # --------------------------------------------------------

    fingerprint = fingerprint_mask(
        X,
        Y,
    )

    combined = (
        base
        | fingerprint
    )

    image_high = (
        combined.astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    image = cv2.resize(
        image_high,
        (
            width_px,
            height_px,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return image


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 72)
    print(
        "MICRONYX STEP 9"
    )
    print(
        "NATURAL QUASI-PERIODICITY EXPERIMENT"
    )
    print("=" * 72)
    print()

    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print("SCENE")
    print("-" * 72)

    print(
        f"Physical scene:       "
        f"{PHYSICAL_WIDTH:.1f} × "
        f"{PHYSICAL_HEIGHT:.1f}"
    )

    print(
        f"Base pitch:            "
        f"{BASE_PITCH:.4f}"
    )

    print(
        f"Pitch variation:       "
        f"±{PITCH_VARIATION * 100:.2f}%"
    )

    print(
        f"Fingerprint size:      "
        f"{FINGERPRINT_WIDTH:.2f} × "
        f"{FINGERPRINT_HEIGHT:.2f}"
    )

    print()

    print("TARGET")
    print("-" * 72)

    print(
        f"Physical target:       "
        f"({TARGET_X:.2f}, {TARGET_Y:.2f})"
    )

    target_search_x = (
        TARGET_X
        * SEARCH_PIXELS_PER_UNIT
    )

    target_search_y = (
        TARGET_Y
        * SEARCH_PIXELS_PER_UNIT
    )

    print(
        f"Search target:         "
        f"({target_search_x:.2f}, "
        f"{target_search_y:.2f})"
    )

    print()

    # ========================================================
    # REFERENCE FOV
    # ========================================================

    half_fov_x = (
        REFERENCE_FOV_WIDTH
        / 2.0
    )

    half_fov_y = (
        REFERENCE_FOV_HEIGHT
        / 2.0
    )

    reference_origin_x = (
        TARGET_X
        - half_fov_x
    )

    reference_origin_y = (
        TARGET_Y
        - half_fov_y
    )

    print("REFERENCE")
    print("-" * 72)

    print(
        f"FOV:                  "
        f"{REFERENCE_FOV_WIDTH:.4f} × "
        f"{REFERENCE_FOV_HEIGHT:.4f}"
    )

    print(
        f"Origin:               "
        f"({reference_origin_x:.4f}, "
        f"{reference_origin_y:.4f})"
    )

    print()

    # ========================================================
    # RENDER
    # ========================================================

    print(
        "Rendering natural quasi-periodic search..."
    )

    search = render_scene(
        SEARCH_WIDTH,
        SEARCH_HEIGHT,
        SEARCH_PIXELS_PER_UNIT,
        0.0,
        0.0,
    )

    print(
        "Rendering matching high-magnification reference..."
    )

    reference = render_scene(
        REFERENCE_WIDTH,
        REFERENCE_HEIGHT,
        REFERENCE_PIXELS_PER_UNIT,
        reference_origin_x,
        reference_origin_y,
    )

    # ========================================================
    # SAVE
    # ========================================================

    if not cv2.imwrite(
        str(OUTPUT_SEARCH),
        search,
    ):
        raise RuntimeError(
            "Failed to save search."
        )

    if not cv2.imwrite(
        str(OUTPUT_REFERENCE),
        reference,
    ):
        raise RuntimeError(
            "Failed to save reference."
        )

    # ========================================================
    # SCALE REFERENCE
    # ========================================================

    template_size = int(
        round(
            REFERENCE_WIDTH
            / MAGNIFICATION
        )
    )

    template = cv2.resize(
        reference,
        (
            template_size,
            template_size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # ========================================================
    # GROUND TRUTH LOCATION
    # ========================================================

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

    # ========================================================
    # TEMPLATE MATCHING
    # ========================================================

    print()
    print(
        "Running template matching..."
    )

    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED,
    )

    # ========================================================
    # GT SCORE
    # ========================================================

    gt_score = float(
        result[
            gt_top,
            gt_left,
        ]
    )

    # ========================================================
    # TOP CANDIDATES
    # ========================================================

    suppressed = result.copy()

    candidates = []

    TOP_K = 30

    for _ in range(TOP_K):

        _, score, _, location = (
            cv2.minMaxLoc(
                suppressed
            )
        )

        left = location[0]
        top = location[1]

        cx = (
            left
            + template_size / 2.0
        )

        cy = (
            top
            + template_size / 2.0
        )

        distance = float(
            np.sqrt(
                (
                    cx
                    - target_search_x
                ) ** 2
                +
                (
                    cy
                    - target_search_y
                ) ** 2
            )
        )

        candidates.append(
            {
                "rank": len(candidates) + 1,
                "x": cx,
                "y": cy,
                "score": float(score),
                "distance": distance,
            }
        )

        suppression_radius = (
            max(
                5,
                template_size,
            )
        )

        x1 = max(
            0,
            left
            - suppression_radius,
        )

        y1 = max(
            0,
            top
            - suppression_radius,
        )

        x2 = min(
            suppressed.shape[1],
            left
            + suppression_radius
            + 1,
        )

        y2 = min(
            suppressed.shape[0],
            top
            + suppression_radius
            + 1,
        )

        suppressed[
            y1:y2,
            x1:x2
        ] = -1.0

    # ========================================================
    # GT RANK
    # ========================================================

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

    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    threshold_1 = (
        gt_score
        * 0.99
    )

    threshold_2 = (
        gt_score
        * 0.98
    )

    threshold_5 = (
        gt_score
        * 0.95
    )

    count_1 = int(
        np.sum(
            flat
            >= threshold_1
        )
    )

    count_2 = int(
        np.sum(
            flat
            >= threshold_2
        )
    )

    count_5 = int(
        np.sum(
            flat
            >= threshold_5
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print(
        "RESULTS"
    )
    print("=" * 72)

    print()

    print(
        f"GT score:              "
        f"{gt_score:.6f}"
    )

    print(
        f"GT rank:               "
        f"{gt_rank:,}"
    )

    print()

    print(
        "Top candidates:"
    )

    print("-" * 72)

    for candidate in candidates[:15]:

        print(
            f"{candidate['rank']:2d}. "
            f"({candidate['x']:7.2f}, "
            f"{candidate['y']:7.2f}) "
            f"score={candidate['score']:.6f} "
            f"distance={candidate['distance']:.2f}px"
        )

    print()

    print(
        "AMBIGUITY COUNTS"
    )

    print("-" * 72)

    print(
        f"Candidates >= 99% of GT: "
        f"{count_1:,}"
    )

    print(
        f"Candidates >= 98% of GT: "
        f"{count_2:,}"
    )

    print(
        f"Candidates >= 95% of GT: "
        f"{count_5:,}"
    )

    print()

    # ========================================================
    # BEST MATCH
    # ========================================================

    best = candidates[0]

    best_error = best[
        "distance"
    ]

    print(
        f"Best predicted center: "
        f"({best['x']:.2f}, "
        f"{best['y']:.2f})"
    )

    print(
        f"Best error:            "
        f"{best_error:.4f}px"
    )

    print()

    # ========================================================
    # VISUALIZATION
    # ========================================================

    visual = cv2.cvtColor(
        search,
        cv2.COLOR_GRAY2BGR,
    )

    # Ground truth

    gt_x_int = int(
        round(
            target_search_x
        )
    )

    gt_y_int = int(
        round(
            target_search_y
        )
    )

    cv2.drawMarker(
        visual,
        (
            gt_x_int,
            gt_y_int,
        ),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        30,
        3,
    )

    cv2.putText(
        visual,
        "GT",
        (
            gt_x_int + 15,
            gt_y_int - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Top candidates

    for candidate in candidates[:15]:

        x = int(
            round(
                candidate["x"]
            )
        )

        y = int(
            round(
                candidate["y"]
            )
        )

        is_gt = (
            candidate["distance"]
            < 1.0
        )

        if is_gt:
            continue

        cv2.circle(
            visual,
            (x, y),
            7,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            visual,
            str(candidate["rank"]),
            (
                x + 8,
                y + 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    if not cv2.imwrite(
        str(OUTPUT_VISUAL),
        visual,
    ):
        raise RuntimeError(
            "Failed to save visualization."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "MICRONYX STEP 9\n"
        )

        file.write(
            "Natural Quasi-Periodic Experiment\n\n"
        )

        file.write(
            f"GT physical: "
            f"({TARGET_X}, {TARGET_Y})\n"
        )

        file.write(
            f"GT search: "
            f"({target_search_x}, "
            f"{target_search_y})\n"
        )

        file.write(
            f"GT score: "
            f"{gt_score}\n"
        )

        file.write(
            f"GT rank: "
            f"{gt_rank}\n"
        )

        file.write(
            f"Best error: "
            f"{best_error}\n\n"
        )

        file.write(
            "Ambiguity:\n"
        )

        file.write(
            f">=99%: {count_1}\n"
        )

        file.write(
            f">=98%: {count_2}\n"
        )

        file.write(
            f">=95%: {count_5}\n\n"
        )

        file.write(
            "Top candidates:\n"
        )

        for candidate in candidates:

            file.write(
                f"{candidate['rank']},"
                f"{candidate['x']:.4f},"
                f"{candidate['y']:.4f},"
                f"{candidate['score']:.8f},"
                f"{candidate['distance']:.4f}\n"
            )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 72)

    if best_error <= 5.0:
        print(
            "RESULT: NATURAL QUASI-PERIODIC "
            "SCENE RECOVERED BY TEMPLATE MATCHING"
        )
    else:
        print(
            "RESULT: NATURAL QUASI-PERIODIC "
            "SCENE CAUSED LOCALIZATION FAILURE"
        )

    print("=" * 72)

    print()
    print(
        f"Search:\n{OUTPUT_SEARCH}"
    )

    print(
        f"Reference:\n{OUTPUT_REFERENCE}"
    )

    print(
        f"Visualization:\n{OUTPUT_VISUAL}"
    )

    print(
        f"Results:\n{OUTPUT_RESULTS}"
    )

    print()