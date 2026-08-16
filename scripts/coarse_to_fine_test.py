from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 20
# COARSE-TO-FINE LOCALIZATION
#
# Methods:
#
# 1. Raw full-resolution NCC
# 2. DOG full-resolution NCC
# 3. Coarse-to-fine RAW
# 4. Coarse-to-fine DOG
# 5. Hybrid RAW + DOG
#
# Goal:
# Determine whether multi-resolution localization improves
# robustness against periodic / quasi-periodic ambiguity.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "coarse_to_fine"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "coarse_to_fine_results.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

SEARCH_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

SUPERSAMPLE = 2

TEMPLATE_SIZE = 10

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

def periodic_scene(x, y):

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

    return structure + modulation


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

    image = np.where(
        target_mask(
            X,
            Y,
            tx,
            ty,
        ),
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
# DOG
# ============================================================

def dog(
    image,
):

    image = image.astype(
        np.float32
    )

    g1 = cv2.GaussianBlur(
        image,
        (0, 0),
        0.8,
    )

    g2 = cv2.GaussianBlur(
        image,
        (0, 0),
        2.5,
    )

    return (
        g1 - g2
    ).astype(
        np.float32
    )


# ============================================================
# MATCH
# ============================================================

def match(
    search,
    reference,
):

    response = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED,
    )

    _, score, _, loc = (
        cv2.minMaxLoc(
            response
        )
    )

    return (
        float(score),
        loc,
        response,
    )


# ============================================================
# RANK
# ============================================================

def rank_at_gt(
    response,
    gt_x,
    gt_y,
):

    score = float(
        response[
            gt_y,
            gt_x,
        ]
    )

    rank = (
        1
        + int(
            np.sum(
                response
                > score
            )
        )
    )

    return (
        score,
        rank,
    )


# ============================================================
# COARSE-TO-FINE
# ============================================================

def coarse_to_fine(
    search,
    reference,
    use_dog=False,
):

    if use_dog:

        search = dog(
            search
        )

        reference = dog(
            reference
        )

    # --------------------------------------------------------
    # Level 1
    #
    # Downsample by 4
    # --------------------------------------------------------

    search_coarse = cv2.resize(
        search,
        (
            SEARCH_SIZE // 4,
            SEARCH_SIZE // 4,
        ),
        interpolation=cv2.INTER_AREA,
    )

    reference_coarse = cv2.resize(
        reference,
        (
            TEMPLATE_SIZE // 4,
            TEMPLATE_SIZE // 4,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # Minimum 2x2 template
    reference_coarse = cv2.resize(
        reference_coarse,
        (
            max(
                3,
                TEMPLATE_SIZE // 4,
            ),
            max(
                3,
                TEMPLATE_SIZE // 4,
            ),
        ),
        interpolation=cv2.INTER_AREA,
    )

    _, _, response_coarse = match(
        search_coarse,
        reference_coarse,
    )

    # --------------------------------------------------------
    # Keep multiple coarse candidates.
    # --------------------------------------------------------

    flat = response_coarse.ravel()

    K = min(
        50,
        flat.size,
    )

    indices = np.argpartition(
        flat,
        -K,
    )[-K:]

    candidates = []

    for index in indices:

        y, x = np.unravel_index(
            index,
            response_coarse.shape,
        )

        candidates.append(
            (
                x,
                y,
                float(
                    response_coarse[
                        y,
                        x
                    ]
                ),
            )
        )

    # --------------------------------------------------------
    # Fine refinement.
    #
    # Each coarse point corresponds to roughly 4x4 pixels
    # in the original search.
    # --------------------------------------------------------

    best_score = -np.inf
    best_loc = (
        0,
        0,
    )

    fine_radius = 12

    for cx, cy, _ in candidates:

        center_x = (
            cx
            * 4
        )

        center_y = (
            cy
            * 4
        )

        x0 = max(
            0,
            center_x
            - fine_radius,
        )

        y0 = max(
            0,
            center_y
            - fine_radius,
        )

        x1 = min(
            SEARCH_SIZE
            - TEMPLATE_SIZE
            + 1,
            center_x
            + fine_radius,
        )

        y1 = min(
            SEARCH_SIZE
            - TEMPLATE_SIZE
            + 1,
            center_y
            + fine_radius,
        )

        if x1 < x0 or y1 < y0:
            continue

        region = search[
            y0:y1 + TEMPLATE_SIZE,
            x0:x1 + TEMPLATE_SIZE,
        ]

        if (
            region.shape[0]
            < TEMPLATE_SIZE
            or
            region.shape[1]
            < TEMPLATE_SIZE
        ):
            continue

        response = cv2.matchTemplate(
            region,
            reference,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, loc = (
            cv2.minMaxLoc(
                response
            )
        )

        px = (
            x0
            + loc[0]
        )

        py = (
            y0
            + loc[1]
        )

        if score > best_score:

            best_score = float(
                score
            )

            best_loc = (
                px,
                py,
            )

    return (
        best_score,
        best_loc,
    )


# ============================================================
# HYBRID
# ============================================================

def hybrid_method(
    search,
    reference,
):

    raw_score, raw_loc, _ = (
        match(
            search,
            reference,
        )
    )

    dog_search = dog(
        search
    )

    dog_reference = dog(
        reference
    )

    dog_score, dog_loc, _ = (
        match(
            dog_search,
            dog_reference,
        )
    )

    # --------------------------------------------------------
    # Generate candidate points from both modalities.
    # --------------------------------------------------------

    candidates = [
        (
            raw_loc[0],
            raw_loc[1],
        ),
        (
            dog_loc[0],
            dog_loc[1],
        ),
    ]

    # --------------------------------------------------------
    # Local refinement around both candidates using a joint
    # score.
    # --------------------------------------------------------

    best_score = -np.inf
    best_loc = (
        0,
        0,
    )

    radius = 20

    for cx, cy in candidates:

        x0 = max(
            0,
            cx - radius,
        )

        y0 = max(
            0,
            cy - radius,
        )

        x1 = min(
            SEARCH_SIZE
            - TEMPLATE_SIZE
            + 1,
            cx + radius,
        )

        y1 = min(
            SEARCH_SIZE
            - TEMPLATE_SIZE
            + 1,
            cy + radius,
        )

        for y in range(
            y0,
            y1 + 1,
        ):

            for x in range(
                x0,
                x1 + 1,
            ):

                patch = search[
                    y:y + TEMPLATE_SIZE,
                    x:x + TEMPLATE_SIZE,
                ]

                if patch.shape != (
                    TEMPLATE_SIZE,
                    TEMPLATE_SIZE,
                ):
                    continue

                raw = cv2.matchTemplate(
                    patch,
                    reference,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]

                dpatch = dog(
                    patch
                )

                dref = dog(
                    reference
                )

                dscore = cv2.matchTemplate(
                    dpatch,
                    dref,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]

                score = (
                    0.5 * raw
                    + 0.5 * dscore
                )

                if score > best_score:

                    best_score = float(
                        score
                    )

                    best_loc = (
                        x,
                        y,
                    )

    return (
        best_score,
        best_loc,
    )


# ============================================================
# ONE SCENE
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

    gt_x = int(
        round(
            tx
            * SEARCH_PPU
            - TEMPLATE_SIZE / 2
        )
    )

    gt_y = int(
        round(
            ty
            * SEARCH_PPU
            - TEMPLATE_SIZE / 2
        )
    )

    results = {}

    # ========================================================
    # RAW
    # ========================================================

    raw_score, raw_loc, raw_response = (
        match(
            search,
            reference,
        )
    )

    gt_score, gt_rank = rank_at_gt(
        raw_response,
        gt_x,
        gt_y,
    )

    results[
        "raw_error"
    ] = float(
        np.hypot(
            raw_loc[0] - gt_x,
            raw_loc[1] - gt_y,
        )
    )

    results[
        "raw_rank"
    ] = gt_rank

    # ========================================================
    # DOG
    # ========================================================

    dsearch = dog(
        search
    )

    dereference = dog(
        reference
    )

    dog_score, dog_loc, dog_response = (
        match(
            dsearch,
            dereference,
        )
    )

    _, dog_rank = rank_at_gt(
        dog_response,
        gt_x,
        gt_y,
    )

    results[
        "dog_error"
    ] = float(
        np.hypot(
            dog_loc[0] - gt_x,
            dog_loc[1] - gt_y,
        )
    )

    results[
        "dog_rank"
    ] = dog_rank

    # ========================================================
    # COARSE RAW
    # ========================================================

    _, coarse_raw_loc = (
        coarse_to_fine(
            search,
            reference,
            False,
        )
    )

    results[
        "coarse_raw_error"
    ] = float(
        np.hypot(
            coarse_raw_loc[0]
            - gt_x,
            coarse_raw_loc[1]
            - gt_y,
        )
    )

    # ========================================================
    # COARSE DOG
    # ========================================================

    _, coarse_dog_loc = (
        coarse_to_fine(
            search,
            reference,
            True,
        )
    )

    results[
        "coarse_dog_error"
    ] = float(
        np.hypot(
            coarse_dog_loc[0]
            - gt_x,
            coarse_dog_loc[1]
            - gt_y,
        )
    )

    # ========================================================
    # HYBRID
    # ========================================================

    _, hybrid_loc = hybrid_method(
        search,
        reference,
    )

    results[
        "hybrid_error"
    ] = float(
        np.hypot(
            hybrid_loc[0]
            - gt_x,
            hybrid_loc[1]
            - gt_y,
        )
    )

    return results


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
        "MICRONYX STEP 20"
    )
    print(
        "COARSE-TO-FINE LOCALIZATION"
    )
    print("=" * 76)
    print()

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    results = []

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
    # SAVE
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

        for r in results:

            f.write(
                f"{r['scene']},"
                f"{r['seed']},"
                + ",".join(
                    f"{r[key]:.6f}"
                    if "error"
                    in key
                    else str(
                        r[key]
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
        "dog",
        "coarse_raw",
        "coarse_dog",
        "hybrid",
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

            errors = np.array(
                [
                    r[
                        f"{method}_error"
                    ]
                    for r in subset
                ]
            )

            if method in [
                "raw",
                "dog",
            ]:

                ranks = np.array(
                    [
                        r[
                            f"{method}_rank"
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
                    f"{np.median(ranks):8.1f}"
                )

            else:

                print(
                    f"{method:<12}"
                    f"<=1px="
                    f"{np.mean(errors <= 1) * 100:6.2f}% "
                    f"<=5px="
                    f"{np.mean(errors <= 5) * 100:6.2f}% "
                    f"MedErr="
                    f"{np.median(errors):8.3f}px "
                    f"MaxErr="
                    f"{np.max(errors):8.3f}px"
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