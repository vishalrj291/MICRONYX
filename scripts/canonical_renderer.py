"""
MICRONYX — Canonical Renderer
=============================

Canonical continuous physical scene + multi-resolution acquisition model.

FINAL PS02 ACQUISITION MODEL
-----------------------------

Search image:
    1000 x 1000 pixels
    5 pixels / canonical physical unit

Reference image:
    1000 x 1000 pixels
    50 pixels / canonical physical unit

Therefore:

    REFERENCE_PIXELS_PER_UNIT
    ------------------------- = 10x
     SEARCH_PIXELS_PER_UNIT

The reference is sampled at 10x finer spatial resolution than
the search image while retaining the same 1000 x 1000 pixel
image dimensions.

Physical field of view:

    Search:
        1000 / 5 = 200 x 200 physical units

    Reference:
        1000 / 50 = 20 x 20 physical units

The reference field of view is centered on the same physical
target observed in the search image.

SEARCH-EQUIVALENT TEMPLATE
---------------------------

The reference covers a 20 x 20 physical-unit region.

At the search sampling density:

    20 physical units x 5 pixels/unit
        = 100 x 100 pixels

Therefore:

    1000 x 1000 high-resolution reference
                    ↓
        search-equivalent physical crop
                    ↓
             100 x 100 template

The template therefore represents the same physical field of
view as the high-resolution reference, but at the search
image's spatial sampling density.

IMPORTANT
---------

This renderer provides a single canonical acquisition model.

No:
    - target fingerprint injection
    - alternate renderer
    - new ground truth
    - scene-specific manual selection

The search and reference observations are generated from the
same continuous physical scene. The difference between them is
the spatial sampling density and corresponding physical field
of view.
"""


from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# CANONICAL SCENE CONFIGURATION
# ============================================================================

SCENE_WIDTH = 200.0
SCENE_HEIGHT = 200.0

BASE_PITCH = 0.50


# ============================================================================
# CANONICAL ACQUISITION MODEL
# ============================================================================

# Search observation:
# 1000 x 1000 pixels at 5 pixels per canonical physical unit.
SEARCH_PIXELS_PER_UNIT = 5.0

# Reference observation:
# 100 x 100 pixels at 50 pixels per canonical physical unit.
REFERENCE_PIXELS_PER_UNIT = 50.0

# Reference sampling is 10x finer than search sampling.
SAMPLING_RATIO = (
    REFERENCE_PIXELS_PER_UNIT
    / SEARCH_PIXELS_PER_UNIT
)

EXPECTED_SAMPLING_RATIO = 10.0


# ============================================================================
# IMAGE DIMENSIONS
# ============================================================================

SEARCH_WIDTH = 1000
SEARCH_HEIGHT = 1000

REFERENCE_WIDTH = 1000
REFERENCE_HEIGHT = 1000

TEMPLATE_WIDTH = 100
TEMPLATE_HEIGHT = 100


# ============================================================================
# TARGET
# ============================================================================

# Canonical target used by the existing MICRONYX experiments.
#
# These coordinates are retained deliberately so that existing
# validation experiments remain reproducible.

DEFAULT_TX = 75.25
DEFAULT_TY = 113.75


# ============================================================================
# RANDOM / SENSOR CONFIGURATION
# ============================================================================

SUPERSAMPLE = 2

SENSOR_NOISE_STD = 0.0


# ============================================================================
# BASIC VALIDATION
# ============================================================================

if abs(SAMPLING_RATIO - EXPECTED_SAMPLING_RATIO) > 1e-12:
    raise RuntimeError(
        "Canonical sampling ratio is invalid: "
        f"{SAMPLING_RATIO} != {EXPECTED_SAMPLING_RATIO}"
    )


# ============================================================================
# SCENE FUNCTIONS
# ============================================================================

def periodic_scene(x, y):
    """
    Canonical periodic semiconductor-like structural field.

    Parameters
    ----------
    x, y : ndarray or scalar
        Canonical physical scene coordinates.

    Returns
    -------
    ndarray or scalar
        Continuous intensity field.
    """

    # Primary periodic structure.
    px = 2.0 * np.pi * x / BASE_PITCH
    py = 2.0 * np.pi * y / BASE_PITCH

    horizontal = np.sin(px)
    vertical = np.sin(py)

    # Interacting periodic structure.
    interaction = (
        0.30
        * np.sin(px + py)
    )

    return (
        horizontal
        + vertical
        + interaction
    )


def quasiperiodic_scene(x, y, seed):
    """
    Canonical quasiperiodic scene.

    The seed controls the deterministic perturbation while
    preserving reproducibility.
    """

    rng = np.random.default_rng(seed)

    phase_x = rng.uniform(
        -np.pi,
        np.pi,
    )

    phase_y = rng.uniform(
        -np.pi,
        np.pi,
    )

    scale_x = rng.uniform(
        0.92,
        1.08,
    )

    scale_y = rng.uniform(
        0.92,
        1.08,
    )

    p1 = (
        2.0
        * np.pi
        * x
        / (BASE_PITCH * scale_x)
    )

    p2 = (
        2.0
        * np.pi
        * y
        / (BASE_PITCH * scale_y)
    )

    # Incommensurate secondary components.
    q1 = (
        2.0
        * np.pi
        * x
        / (BASE_PITCH * np.sqrt(2.0))
    )

    q2 = (
        2.0
        * np.pi
        * y
        / (BASE_PITCH * np.sqrt(3.0))
    )

    return (
        np.sin(p1 + phase_x)
        + np.sin(p2 + phase_y)
        + 0.35 * np.sin(q1)
        + 0.35 * np.sin(q2)
    )


def continuous_scene(x, y, scene_type="periodic", seed=0):
    """
    Dispatch to the canonical continuous scene.
    """

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
        f"Unknown scene_type: {scene_type}"
    )


# ============================================================================
# TARGET MASK
# ============================================================================

def target_mask(x, y, tx, ty):
    """
    Canonical localized target structure.

    The target is intentionally deterministic and independent
    of the candidate-generation algorithms.

    Parameters
    ----------
    x, y : ndarray
        Physical scene coordinates.

    tx, ty : float
        Target center in canonical physical coordinates.
    """

    dx = x - tx
    dy = y - ty

    # Local vertical structure.
    vertical = (
        np.exp(
            -(
                (dx / 0.07) ** 2
                + (dy / 0.35) ** 2
            )
        )
    )

    # Local horizontal structure.
    horizontal = (
        np.exp(
            -(
                (dx / 0.35) ** 2
                + (dy / 0.07) ** 2
            )
        )
    )

    # Local diagonal structure.
    diagonal = (
        np.exp(
            -(
                ((dx - dy) / 0.10) ** 2
                + ((dx + dy) / 0.30) ** 2
            )
        )
    )

    # Compact contact-like structure.
    radial = (
        np.exp(
            -(
                (dx * dx + dy * dy)
                / (2.0 * 0.12 ** 2)
            )
        )
    )

    return (
        vertical
        + horizontal
        + 0.75 * diagonal
        + 0.50 * radial
    )


# ============================================================================
# IMAGE NORMALIZATION
# ============================================================================

def normalize_uint8(image):
    """
    Normalize a floating-point image to uint8 [0,255].
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    minimum = float(
        np.min(image)
    )

    maximum = float(
        np.max(image)
    )

    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
    ):
        raise ValueError(
            "Image contains non-finite values."
        )

    if maximum - minimum < 1e-12:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    normalized = (
        (image - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return np.clip(
        normalized,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================================
# SENSOR RENDERING
# ============================================================================

def render_sensor(
    width,
    height,
    pixels_per_unit,
    origin_x,
    origin_y,
    tx,
    ty,
    scene_type,
    seed,
):
    """
    Render a canonical sensor observation from the continuous physical scene.

    The same physical scene is sampled at different spatial sampling
    densities to model the PS02 search/reference acquisition process.
    """

    render_width = width * SUPERSAMPLE
    render_height = height * SUPERSAMPLE

    effective_ppu = pixels_per_unit * SUPERSAMPLE

    # Pixel-center sampling.
    xs = (
        origin_x
        + (np.arange(render_width) + 0.5)
        / effective_ppu
    )

    ys = (
        origin_y
        + (np.arange(render_height) + 0.5)
        / effective_ppu
    )

    xx, yy = np.meshgrid(xs, ys)

    # Continuous physical scene.
    scene = continuous_scene(
        xx,
        yy,
        scene_type,
        seed,
    )

    # Localized target structure.
    target = target_mask(
        xx,
        yy,
        tx,
        ty,
    )

    image = scene + target

    # Supersampled acquisition → sensor resolution.
    image = cv2.resize(
        image.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    # Optional sensor noise.
    if SENSOR_NOISE_STD > 0.0:

        rng = np.random.default_rng(
            seed + 991
        )

        noise = rng.normal(
            0.0,
            SENSOR_NOISE_STD,
            size=image.shape,
        )

        image = image + noise

    return normalize_uint8(image)


# ============================================================================
# SEARCH RENDERING
# ============================================================================

def render_search(
    tx,
    ty,
    scene_type,
    seed,
):
    """
    Render the canonical PS02 search image.

    Dimensions:
        1000 x 1000 pixels

    Sampling:
        5 pixels / physical unit

    Physical FOV:
        200 x 200 physical units
    """

    return render_sensor(
        width=SEARCH_WIDTH,
        height=SEARCH_HEIGHT,
        pixels_per_unit=SEARCH_PIXELS_PER_UNIT,
        origin_x=0.0,
        origin_y=0.0,
        tx=tx,
        ty=ty,
        scene_type=scene_type,
        seed=seed,
    )


# ============================================================================
# REFERENCE RENDERING
# ============================================================================
def render_reference(
    tx,
    ty,
    scene_type,
    seed,
):
    """
    Render the canonical PS02 reference image.

    Dimensions:
        1000 x 1000 pixels

    Sampling:
        50 pixels / physical unit

    Relative sampling:
        10x finer than the search image

    Physical field of view:
        20 x 20 physical units

    The reference FOV is centered on the same physical target
    observed by the search image.
    """

    reference_fov_width = (
        REFERENCE_WIDTH
        / REFERENCE_PIXELS_PER_UNIT
    )

    reference_fov_height = (
        REFERENCE_HEIGHT
        / REFERENCE_PIXELS_PER_UNIT
    )

    origin_x = (
        tx
        - reference_fov_width / 2.0
    )

    origin_y = (
        ty
        - reference_fov_height / 2.0
    )

    return render_sensor(
        width=REFERENCE_WIDTH,
        height=REFERENCE_HEIGHT,
        pixels_per_unit=REFERENCE_PIXELS_PER_UNIT,
        origin_x=origin_x,
        origin_y=origin_y,
        tx=tx,
        ty=ty,
        scene_type=scene_type,
        seed=seed,
    )

    
# ============================================================================
# SEARCH-EQUIVALENT TEMPLATE
# ============================================================================

def create_ps02_template(reference):
    """
    Convert the 100 x 100 high-resolution reference into
    the 10 x 10 search-equivalent template.

    This preserves the existing canonical MICRONYX
    acquisition/matching model.
    """

    reference = np.asarray(
        reference,
        dtype=np.uint8,
    )

    if reference.shape != (
        REFERENCE_HEIGHT,
        REFERENCE_WIDTH,
    ):
        raise ValueError(
            "Expected canonical reference shape "
            f"{REFERENCE_HEIGHT}x{REFERENCE_WIDTH}, "
            f"got {reference.shape}"
        )

    template = cv2.resize(
        reference,
        (
            TEMPLATE_WIDTH,
            TEMPLATE_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return template


# ============================================================================
# TARGET COORDINATE
# ============================================================================

def template_top_left(
    tx,
    ty,
):
    """
    Return the canonical search-image top-left coordinate
    corresponding to the physical target center.

    Search coordinates use SEARCH_PIXELS_PER_UNIT.
    """

    search_x = (
        tx
        * SEARCH_PIXELS_PER_UNIT
    )

    search_y = (
        ty
        * SEARCH_PIXELS_PER_UNIT
    )

    template_width = TEMPLATE_WIDTH
    template_height = TEMPLATE_HEIGHT

    top_left_x = (
        search_x
        - template_width / 2.0
    )

    top_left_y = (
        search_y
        - template_height / 2.0
    )

    return (
        int(round(top_left_x)),
        int(round(top_left_y)),
    )


# ============================================================================
# COMPLETE OBSERVATION
# ============================================================================

def generate_observation(
    tx,
    ty,
    scene_type,
    seed,
):
    """
    Generate the complete canonical PS02 observation.

    Returns
    -------
    dict
        search
        reference
        template
        gt_x
        gt_y
        target_x
        target_y
        scene_type
        seed
    """

    search = render_search(
        tx,
        ty,
        scene_type,
        seed,
    )

    reference = render_reference(
        tx,
        ty,
        scene_type,
        seed,
    )

    template = create_ps02_template(
        reference
    )

    gt_x, gt_y = template_top_left(
        tx,
        ty,
    )

    return {
        "search": search,
        "reference": reference,
        "template": template,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "target_x": tx,
        "target_y": ty,
        "scene_type": scene_type,
        "seed": seed,
    }


# ============================================================================
# CANONICAL SELF TEST
# ============================================================================

def self_test():

    print()
    print("=" * 70)
    print("MICRONYX CANONICAL RENDERER SELF TEST")
    print("=" * 70)
    print()

    print("Acquisition model:")
    print(
        f"  Search PPU:     {SEARCH_PIXELS_PER_UNIT}"
    )
    print(
        f"  Reference PPU:  {REFERENCE_PIXELS_PER_UNIT}"
    )
    print(
        f"  Sampling ratio: {SAMPLING_RATIO:.2f}x"
    )

    print()

    if abs(
        SAMPLING_RATIO
        - EXPECTED_SAMPLING_RATIO
    ) > 1e-12:

        raise AssertionError(
            "Sampling ratio is not 10x."
        )

    for scene_type in [
        "periodic",
        "quasiperiodic",
    ]:

        observation = generate_observation(
            DEFAULT_TX,
            DEFAULT_TY,
            scene_type,
            20260816,
        )

        search = observation[
            "search"
        ]

        reference = observation[
            "reference"
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

        print(
            f"Scene: {scene_type}"
        )

        print(
            "Search:",
            search.shape,
            search.dtype,
        )

        print(
            "Reference:",
            reference.shape,
            reference.dtype,
        )

        print(
            "Template:",
            template.shape,
            template.dtype,
        )

        print(
            "GT top-left:",
            (
                gt_x,
                gt_y,
            ),
        )

        # Shape checks.
        assert search.shape == (
            SEARCH_HEIGHT,
            SEARCH_WIDTH,
        )

        assert reference.shape == (
            REFERENCE_HEIGHT,
            REFERENCE_WIDTH,
        )

        assert template.shape == (
            TEMPLATE_HEIGHT,
            TEMPLATE_WIDTH,
        )

        # Type checks.
        assert search.dtype == np.uint8
        assert reference.dtype == np.uint8
        assert template.dtype == np.uint8

        # Numerical checks.
        assert np.isfinite(
            search
        ).all()

        assert np.isfinite(
            reference
        ).all()

        assert np.isfinite(
            template
        ).all()

        # GT bounds.
        assert (
            0
            <= gt_x
            <= SEARCH_WIDTH
            - TEMPLATE_WIDTH
        )

        assert (
            0
            <= gt_y
            <= SEARCH_HEIGHT
            - TEMPLATE_HEIGHT
        )

        print()

    print(
        "Canonical renderer: PASS"
    )

    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    self_test()