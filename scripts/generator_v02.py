from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX Synthetic Dataset Generator v0.2
#
# STEP 5
# ------------------------------------------------------------
# Controlled periodic geometry
# + local structural fingerprint
# + exact target coordinate
# + exact reference FOV
#
# IMPORTANT:
# This is NOT a physically accurate SEM simulator.
# It is a controlled semiconductor-inspired synthetic model.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

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
# PHYSICAL SCENE
# ============================================================

PHYSICAL_WIDTH = 200.0
PHYSICAL_HEIGHT = 200.0


# ============================================================
# SEARCH
# ============================================================

SEARCH_WIDTH = 1000
SEARCH_HEIGHT = 1000

SEARCH_PIXELS_PER_UNIT_X = (
    SEARCH_WIDTH / PHYSICAL_WIDTH
)

SEARCH_PIXELS_PER_UNIT_Y = (
    SEARCH_HEIGHT / PHYSICAL_HEIGHT
)


# ============================================================
# REFERENCE
# ============================================================

REFERENCE_WIDTH = 100
REFERENCE_HEIGHT = 100

MAGNIFICATION = 10.0

REFERENCE_PIXELS_PER_UNIT_X = (
    SEARCH_PIXELS_PER_UNIT_X
    * MAGNIFICATION
)

REFERENCE_PIXELS_PER_UNIT_Y = (
    SEARCH_PIXELS_PER_UNIT_Y
    * MAGNIFICATION
)


# ============================================================
# REFERENCE FIELD OF VIEW
# ============================================================

REFERENCE_FOV_WIDTH = (
    REFERENCE_WIDTH
    / REFERENCE_PIXELS_PER_UNIT_X
)

REFERENCE_FOV_HEIGHT = (
    REFERENCE_HEIGHT
    / REFERENCE_PIXELS_PER_UNIT_Y
)


# ============================================================
# BASE SEMICONDUCTOR GEOMETRY
# ============================================================

PITCH = 0.5

LINE_WIDTH = 0.2


# ============================================================
# LOCAL FINGERPRINT
# ============================================================

# Size of the region in which we introduce controlled
# structural variation.

FINGERPRINT_WIDTH = 1.0
FINGERPRINT_HEIGHT = 1.0


# The two modifications are deliberately asymmetric.
#
# They are positioned relative to the target center.

# Modification 1:
# local vertical feature
DEFECT_1_OFFSET_X = -0.22
DEFECT_1_OFFSET_Y = -0.18

# Modification 2:
# local horizontal feature
DEFECT_2_OFFSET_X = 0.21
DEFECT_2_OFFSET_Y = 0.17


# Additional width added to the corresponding structures.

DEFECT_WIDTH_INCREASE = 0.12


# Length of each local modification.

DEFECT_LENGTH = 0.38


# ============================================================
# TARGET
# ============================================================

TARGET_PHYSICAL_X = 75.25
TARGET_PHYSICAL_Y = 113.75


# ============================================================
# RENDERING
# ============================================================

SUPERSAMPLE = 4


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def physical_to_search_x(x):
    return (
        x
        * SEARCH_PIXELS_PER_UNIT_X
    )


def physical_to_search_y(y):
    return (
        y
        * SEARCH_PIXELS_PER_UNIT_Y
    )


# ============================================================
# BASE GEOMETRY
# ============================================================

def base_structure(
    x,
    y,
):
    """
    Evaluate the clean periodic semiconductor-inspired
    structure at physical coordinates x,y.

    Returns True if the point belongs to the base structure.
    """

    x_phase = (
        x % PITCH
    )

    y_phase = (
        y % PITCH
    )

    vertical = (
        x_phase
        < LINE_WIDTH
    )

    horizontal = (
        y_phase
        < LINE_WIDTH
    )

    return (
        vertical
        or horizontal
    )


# ============================================================
# LOCAL FINGERPRINT GEOMETRY
# ============================================================

def fingerprint_structure(
    x,
    y,
):
    """
    Add a controlled local structural fingerprint around
    the known target.

    The fingerprint is intentionally deterministic and
    asymmetric.

    Returns True if the point belongs to the added
    fingerprint.
    """

    dx = (
        x
        - TARGET_PHYSICAL_X
    )

    dy = (
        y
        - TARGET_PHYSICAL_Y
    )

    # --------------------------------------------------------
    # Reject points far outside fingerprint neighborhood.
    # --------------------------------------------------------

    if (
        abs(dx)
        > FINGERPRINT_WIDTH / 2.0
    ):
        return False

    if (
        abs(dy)
        > FINGERPRINT_HEIGHT / 2.0
    ):
        return False

    # --------------------------------------------------------
    # DEFECT 1
    #
    # A short vertical thickening.
    # --------------------------------------------------------

    defect_1_x = (
        TARGET_PHYSICAL_X
        + DEFECT_1_OFFSET_X
    )

    defect_1_y = (
        TARGET_PHYSICAL_Y
        + DEFECT_1_OFFSET_Y
    )

    vertical_width = (
        LINE_WIDTH
        + DEFECT_WIDTH_INCREASE
    )

    defect_1_vertical = (
        abs(x - defect_1_x)
        < vertical_width / 2.0
        and
        abs(y - defect_1_y)
        < DEFECT_LENGTH / 2.0
    )

    # --------------------------------------------------------
    # DEFECT 2
    #
    # A short horizontal thickening.
    # --------------------------------------------------------

    defect_2_x = (
        TARGET_PHYSICAL_X
        + DEFECT_2_OFFSET_X
    )

    defect_2_y = (
        TARGET_PHYSICAL_Y
        + DEFECT_2_OFFSET_Y
    )

    horizontal_width = (
        LINE_WIDTH
        + DEFECT_WIDTH_INCREASE
    )

    defect_2_horizontal = (
        abs(x - defect_2_x)
        < DEFECT_LENGTH / 2.0
        and
        abs(y - defect_2_y)
        < horizontal_width / 2.0
    )

    return (
        defect_1_vertical
        or defect_2_horizontal
    )


# ============================================================
# COMPLETE GEOMETRY
# ============================================================

def structure_value_at(
    x,
    y,
):
    """
    Evaluate complete synthetic geometry.

    Base periodic structure +
    local fingerprint.
    """

    base = base_structure(
        x,
        y,
    )

    fingerprint = fingerprint_structure(
        x,
        y,
    )

    if (
        base
        or fingerprint
    ):
        return 255

    return 0


# ============================================================
# SCENE RENDERER
# ============================================================

def create_physical_scene(
    width_px,
    height_px,
    pixels_per_unit_x,
    pixels_per_unit_y,
    origin_x,
    origin_y,
):
    """
    Render a physical field of view.

    origin_x / origin_y define the physical coordinate
    corresponding to the top-left of the image.

    The SAME geometry function is used for both search
    and reference observations.
    """

    high_width = (
        width_px
        * SUPERSAMPLE
    )

    high_height = (
        height_px
        * SUPERSAMPLE
    )

    high_ppu_x = (
        pixels_per_unit_x
        * SUPERSAMPLE
    )

    high_ppu_y = (
        pixels_per_unit_y
        * SUPERSAMPLE
    )

    # --------------------------------------------------------
    # Physical coordinate of every high-resolution pixel.
    # --------------------------------------------------------

    dx = (
        1.0
        / high_ppu_x
    )

    dy = (
        1.0
        / high_ppu_y
    )

    x_values = (
        origin_x
        + (
            np.arange(high_width)
            + 0.5
        )
        * dx
    )

    y_values = (
        origin_y
        + (
            np.arange(high_height)
            + 0.5
        )
        * dy
    )

    # --------------------------------------------------------
    # Base periodic structure.
    #
    # This is vectorized rather than looping over every pixel.
    # --------------------------------------------------------

    x_phase = np.mod(
        x_values,
        PITCH,
    )

    y_phase = np.mod(
        y_values,
        PITCH,
    )

    vertical_mask = (
        x_phase
        < LINE_WIDTH
    )

    horizontal_mask = (
        y_phase
        < LINE_WIDTH
    )

    scene_high = np.zeros(
        (
            high_height,
            high_width,
        ),
        dtype=np.uint8,
    )

    scene_high[
        :,
        vertical_mask
    ] = 255

    scene_high[
        horizontal_mask,
        :
    ] = 255

    # --------------------------------------------------------
    # Fingerprint 1
    # --------------------------------------------------------

    defect_1_x = (
        TARGET_PHYSICAL_X
        + DEFECT_1_OFFSET_X
    )

    defect_1_y = (
        TARGET_PHYSICAL_Y
        + DEFECT_1_OFFSET_Y
    )

    defect_1_width = (
        LINE_WIDTH
        + DEFECT_WIDTH_INCREASE
    )

    x1_min = (
        defect_1_x
        - defect_1_width / 2.0
    )

    x1_max = (
        defect_1_x
        + defect_1_width / 2.0
    )

    y1_min = (
        defect_1_y
        - DEFECT_LENGTH / 2.0
    )

    y1_max = (
        defect_1_y
        + DEFECT_LENGTH / 2.0
    )

    defect_1_x_mask = (
        (x_values >= x1_min)
        &
        (x_values <= x1_max)
    )

    defect_1_y_mask = (
        (y_values >= y1_min)
        &
        (y_values <= y1_max)
    )

    scene_high[
        np.ix_(
            defect_1_y_mask,
            defect_1_x_mask,
        )
    ] = 255

    # --------------------------------------------------------
    # Fingerprint 2
    # --------------------------------------------------------

    defect_2_x = (
        TARGET_PHYSICAL_X
        + DEFECT_2_OFFSET_X
    )

    defect_2_y = (
        TARGET_PHYSICAL_Y
        + DEFECT_2_OFFSET_Y
    )

    defect_2_height = (
        LINE_WIDTH
        + DEFECT_WIDTH_INCREASE
    )

    x2_min = (
        defect_2_x
        - DEFECT_LENGTH / 2.0
    )

    x2_max = (
        defect_2_x
        + DEFECT_LENGTH / 2.0
    )

    y2_min = (
        defect_2_y
        - defect_2_height / 2.0
    )

    y2_max = (
        defect_2_y
        + defect_2_height / 2.0
    )

    defect_2_x_mask = (
        (x_values >= x2_min)
        &
        (x_values <= x2_max)
    )

    defect_2_y_mask = (
        (y_values >= y2_min)
        &
        (y_values <= y2_max)
    )

    scene_high[
        np.ix_(
            defect_2_y_mask,
            defect_2_x_mask,
        )
    ] = 255

    # --------------------------------------------------------
    # Downsample.
    # --------------------------------------------------------

    scene = cv2.resize(
        scene_high,
        (
            width_px,
            height_px,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return scene


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 68)
    print("MICRONYX Synthetic Dataset Generator v0.2")
    print("STEP 5 — Controlled Local Fingerprint")
    print("=" * 68)
    print()

    # ========================================================
    # PHYSICAL MODEL
    # ========================================================

    print("PHYSICAL MODEL")
    print("-" * 68)

    print(
        f"Scene:                  "
        f"{PHYSICAL_WIDTH:.2f} × "
        f"{PHYSICAL_HEIGHT:.2f} units"
    )

    print(
        f"Base pitch:             "
        f"{PITCH:.3f} units"
    )

    print(
        f"Base line width:        "
        f"{LINE_WIDTH:.3f} units"
    )

    print(
        f"Fingerprint size:       "
        f"{FINGERPRINT_WIDTH:.2f} × "
        f"{FINGERPRINT_HEIGHT:.2f} units"
    )

    print()

    # ========================================================
    # SEARCH
    # ========================================================

    print("SEARCH")
    print("-" * 68)

    print(
        f"Resolution:             "
        f"{SEARCH_WIDTH} × "
        f"{SEARCH_HEIGHT}"
    )

    print(
        f"Sampling:               "
        f"{SEARCH_PIXELS_PER_UNIT_X:.2f} px/unit"
    )

    print(
        f"Pitch:                  "
        f"{PITCH * SEARCH_PIXELS_PER_UNIT_X:.2f} px"
    )

    print()

    # ========================================================
    # REFERENCE
    # ========================================================

    print("REFERENCE")
    print("-" * 68)

    print(
        f"Resolution:             "
        f"{REFERENCE_WIDTH} × "
        f"{REFERENCE_HEIGHT}"
    )

    print(
        f"Sampling:               "
        f"{REFERENCE_PIXELS_PER_UNIT_X:.2f} px/unit"
    )

    print(
        f"Magnification:          "
        f"{MAGNIFICATION:.2f}×"
    )

    print(
        f"FOV:                    "
        f"{REFERENCE_FOV_WIDTH:.3f} × "
        f"{REFERENCE_FOV_HEIGHT:.3f} units"
    )

    print()

    # ========================================================
    # TARGET
    # ========================================================

    print("TARGET")
    print("-" * 68)

    print(
        f"Physical:               "
        f"({TARGET_PHYSICAL_X:.4f}, "
        f"{TARGET_PHYSICAL_Y:.4f})"
    )

    target_search_x = (
        physical_to_search_x(
            TARGET_PHYSICAL_X
        )
    )

    target_search_y = (
        physical_to_search_y(
            TARGET_PHYSICAL_Y
        )
    )

    print(
        f"Search pixels:          "
        f"({target_search_x:.4f}, "
        f"{target_search_y:.4f})"
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
        TARGET_PHYSICAL_X
        - half_fov_x
    )

    reference_origin_y = (
        TARGET_PHYSICAL_Y
        - half_fov_y
    )

    print("REFERENCE FOV")
    print("-" * 68)

    print(
        f"Origin:                 "
        f"({reference_origin_x:.4f}, "
        f"{reference_origin_y:.4f})"
    )

    print(
        f"Center:                 "
        f"({TARGET_PHYSICAL_X:.4f}, "
        f"{TARGET_PHYSICAL_Y:.4f})"
    )

    print()

    # ========================================================
    # BOUNDARY CHECK
    # ========================================================

    if (
        reference_origin_x < 0
        or
        reference_origin_y < 0
        or
        reference_origin_x
        + REFERENCE_FOV_WIDTH
        > PHYSICAL_WIDTH
        or
        reference_origin_y
        + REFERENCE_FOV_HEIGHT
        > PHYSICAL_HEIGHT
    ):
        raise ValueError(
            "Reference FOV is outside physical scene."
        )

    print(
        "Reference FOV boundary check: PASS"
    )

    print()

    # ========================================================
    # RENDER SEARCH
    # ========================================================

    print(
        "Rendering search with fingerprint..."
    )

    search = create_physical_scene(
        width_px=SEARCH_WIDTH,
        height_px=SEARCH_HEIGHT,
        pixels_per_unit_x=SEARCH_PIXELS_PER_UNIT_X,
        pixels_per_unit_y=SEARCH_PIXELS_PER_UNIT_Y,
        origin_x=0.0,
        origin_y=0.0,
    )

    # ========================================================
    # RENDER REFERENCE
    # ========================================================

    print(
        "Rendering reference with fingerprint..."
    )

    reference = create_physical_scene(
        width_px=REFERENCE_WIDTH,
        height_px=REFERENCE_HEIGHT,
        pixels_per_unit_x=REFERENCE_PIXELS_PER_UNIT_X,
        pixels_per_unit_y=REFERENCE_PIXELS_PER_UNIT_Y,
        origin_x=reference_origin_x,
        origin_y=reference_origin_y,
    )

    # ========================================================
    # SAVE
    # ========================================================

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
            "Failed to save search image."
        )

    if not cv2.imwrite(
        str(reference_path),
        reference,
    ):
        raise RuntimeError(
            "Failed to save reference image."
        )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    reference_gt_x = (
        REFERENCE_WIDTH
        / 2.0
    )

    reference_gt_y = (
        REFERENCE_HEIGHT
        / 2.0
    )

    # ========================================================
    # MATHEMATICAL VERIFICATION
    # ========================================================

    reconstructed_target_x = (
        reference_origin_x
        + (
            reference_gt_x
            / REFERENCE_PIXELS_PER_UNIT_X
        )
    )

    reconstructed_target_y = (
        reference_origin_y
        + (
            reference_gt_y
            / REFERENCE_PIXELS_PER_UNIT_Y
        )
    )

    error_x = abs(
        reconstructed_target_x
        - TARGET_PHYSICAL_X
    )

    error_y = abs(
        reconstructed_target_y
        - TARGET_PHYSICAL_Y
    )

    # ========================================================
    # PRINT VERIFICATION
    # ========================================================

    print()
    print("GROUND-TRUTH VERIFICATION")
    print("-" * 68)

    print(
        f"Target physical:       "
        f"({TARGET_PHYSICAL_X:.4f}, "
        f"{TARGET_PHYSICAL_Y:.4f})"
    )

    print(
        f"Target search:         "
        f"({target_search_x:.4f}, "
        f"{target_search_y:.4f})"
    )

    print(
        f"Reference GT pixel:    "
        f"({reference_gt_x:.4f}, "
        f"{reference_gt_y:.4f})"
    )

    print(
        f"Reconstructed target:  "
        f"({reconstructed_target_x:.8f}, "
        f"{reconstructed_target_y:.8f})"
    )

    print(
        f"Physical error:        "
        f"({error_x:.10f}, "
        f"{error_y:.10f})"
    )

    if (
        error_x > 1e-9
        or
        error_y > 1e-9
    ):
        raise RuntimeError(
            "Ground-truth verification FAILED."
        )

    print(
        "Ground-truth verification: PASS"
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_path = (
        OUTPUT_DIR
        / "ground_truth.txt"
    )

    metadata = f"""
MICRONYX Dataset Generator v0.2
Step 5 — Controlled Local Fingerprint

Physical scene:
  width  = {PHYSICAL_WIDTH}
  height = {PHYSICAL_HEIGHT}

Base geometry:
  pitch      = {PITCH}
  line_width = {LINE_WIDTH}

Fingerprint:
  width  = {FINGERPRINT_WIDTH}
  height = {FINGERPRINT_HEIGHT}

  defect_1_offset =
    ({DEFECT_1_OFFSET_X}, {DEFECT_1_OFFSET_Y})

  defect_2_offset =
    ({DEFECT_2_OFFSET_X}, {DEFECT_2_OFFSET_Y})

  defect_width_increase =
    {DEFECT_WIDTH_INCREASE}

  defect_length =
    {DEFECT_LENGTH}

Search:
  width  = {SEARCH_WIDTH}
  height = {SEARCH_HEIGHT}
  pixels_per_unit =
    {SEARCH_PIXELS_PER_UNIT_X}

Reference:
  width  = {REFERENCE_WIDTH}
  height = {REFERENCE_HEIGHT}
  magnification =
    {MAGNIFICATION}

Target physical:
  ({TARGET_PHYSICAL_X}, {TARGET_PHYSICAL_Y})

Target search:
  ({target_search_x}, {target_search_y})

Reference FOV origin:
  ({reference_origin_x}, {reference_origin_y})

Reference GT:
  ({reference_gt_x}, {reference_gt_y})

Ground-truth error:
  ({error_x}, {error_y})
"""

    metadata_path.write_text(
        metadata.strip(),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 68)
    print("OUTPUT")
    print("=" * 68)

    print(
        f"Search:       {search_path}"
    )

    print(
        f"Reference:    {reference_path}"
    )

    print(
        f"Metadata:     {metadata_path}"
    )

    print()
    print("=" * 68)
    print("STEP 5 COMPLETE")
    print("=" * 68)
    print()