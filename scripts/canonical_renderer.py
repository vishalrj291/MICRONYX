from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX CANONICAL PHYSICAL RENDERER
#
# IMPORTANT:
# Both search and reference images originate from the SAME
# continuous mathematical scene.
#
# Physical scene
#       ↓
# continuous structure
#       ↓
# high-resolution rendering
#       ↓
# sensor integration / area downsampling
#       ↓
# observed image
#
# This avoids artificial sampling inconsistencies.
# ============================================================


# ============================================================
# GLOBAL PHYSICAL MODEL
# ============================================================

SCENE_WIDTH = 200.0
SCENE_HEIGHT = 200.0

BASE_PITCH = 0.50
BASE_LINE_WIDTH = 0.20

SUPERSAMPLE = 4


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
        (np.abs(dx + 0.20) < 0.075)
        &
        (np.abs(dy) < 0.38)
    )

    horizontal = (
        (np.abs(dx - 0.10) < 0.075)
        &
        (np.abs(dy + 0.25) < 0.075)
    )

    diagonal = (
        np.abs(
            dy
            - 0.8 * dx
            - 0.18
        ) < 0.045
    ) & (
        np.abs(dx) < 0.35
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
        px < BASE_LINE_WIDTH
    )

    horizontal = (
        py < BASE_LINE_WIDTH
    )

    structure = np.where(
        vertical | horizontal,
        235.0,
        45.0,
    )

    return structure.astype(
        np.float32
    )


# ============================================================
# QUASIPERIODIC SCENE
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

    return (
        structure
        + modulation
    ).astype(
        np.float32
    )


# ============================================================
# CONTINUOUS SCENE
# ============================================================

def continuous_scene(
    x,
    y,
    tx,
    ty,
    scene_type,
    seed,
):
    if scene_type == "periodic":

        image = periodic_scene(
            x,
            y,
        )

    elif scene_type == "quasiperiodic":

        image = quasiperiodic_scene(
            x,
            y,
            seed,
        )

    else:

        raise ValueError(
            f"Unknown scene type: "
            f"{scene_type}"
        )

    target = target_mask(
        x,
        y,
        tx,
        ty,
    )

    image = np.where(
        target,
        255.0,
        image,
    )

    return np.clip(
        image,
        0,
        255,
    ).astype(
        np.float32
    )


# ============================================================
# SENSOR RENDERER
# ============================================================

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
    Render one sensor observation from the same continuous
    physical scene.

    The high-resolution image is area-downsampled to model
    pixel integration.
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

    # Pixel-center coordinates at the supersampled level.
    xs = (
        origin_x
        + (
            np.arange(
                high_width,
                dtype=np.float32,
            )
            + 0.5
        )
        / high_ppu
    )

    ys = (
        origin_y
        + (
            np.arange(
                high_height,
                dtype=np.float32,
            )
            + 0.5
        )
        / high_ppu
    )

    X, Y = np.meshgrid(
        xs,
        ys,
    )

    high_image = continuous_scene(
        X,
        Y,
        tx,
        ty,
        scene_type,
        seed,
    )

    # Area averaging approximates sensor pixel integration.
    image = cv2.resize(
        high_image,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return np.clip(
        image,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# SEARCH IMAGE
# ============================================================

def render_search(
    tx,
    ty,
    scene_type,
    seed,
):
    return render_sensor(
        width=1000,
        height=1000,
        pixels_per_unit=5.0,
        origin_x=0.0,
        origin_y=0.0,
        tx=tx,
        ty=ty,
        scene_type=scene_type,
        seed=seed,
    )


# ============================================================
# REFERENCE IMAGE
# ============================================================

def render_reference(
    tx,
    ty,
    scene_type,
    seed,
):
    """
    Reference is a 2 × 2 physical-unit FOV.

    At 50 px/unit:
        2 × 2 units
        = 100 × 100 pixels
    """

    fov = (
        100
        / 50.0
    )

    origin_x = (
        tx
        - fov / 2.0
    )

    origin_y = (
        ty
        - fov / 2.0
    )

    return render_sensor(
        width=100,
        height=100,
        pixels_per_unit=50.0,
        origin_x=origin_x,
        origin_y=origin_y,
        tx=tx,
        ty=ty,
        scene_type=scene_type,
        seed=seed,
    )


# ============================================================
# PS02 TEMPLATE
# ============================================================

def create_ps02_template(
    reference,
):
    """
    Convert the 100 × 100 high-magnification reference into
    the equivalent 10 × 10 search-scale template.

    This is a separate observation-scale transformation.
    """

    return cv2.resize(
        reference,
        (
            10,
            10,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def template_top_left(
    tx,
    ty,
):
    """
    Search coordinates are:

        x_search = physical_x × 5
        y_search = physical_y × 5

    The 10 × 10 template is centered on the physical target.
    """

    center_x = (
        tx
        * 5.0
    )

    center_y = (
        ty
        * 5.0
    )

    top_left_x = (
        center_x
        - 5.0
    )

    top_left_y = (
        center_y
        - 5.0
    )

    return (
        int(round(top_left_x)),
        int(round(top_left_y)),
    )


# ============================================================
# COMPLETE OBSERVATION
# ============================================================

def generate_observation(
    tx,
    ty,
    scene_type,
    seed,
):
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
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "MICRONYX CANONICAL RENDERER SELF TEST"
    )
    print("=" * 70)

    tx = 75.25
    ty = 113.75
    seed = 20260816

    for scene_type in [
        "periodic",
        "quasiperiodic",
    ]:

        data = generate_observation(
            tx,
            ty,
            scene_type,
            seed,
        )

        print()
        print(
            f"Scene: {scene_type}"
        )

        print(
            "Search:",
            data["search"].shape,
            data["search"].dtype,
        )

        print(
            "Reference:",
            data["reference"].shape,
            data["reference"].dtype,
        )

        print(
            "Template:",
            data["template"].shape,
            data["template"].dtype,
        )

        print(
            "GT top-left:",
            (
                data["gt_x"],
                data["gt_y"],
            ),
        )

    print()
    print(
        "Canonical renderer: PASS"
    )