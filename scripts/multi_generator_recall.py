from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 18
# MULTI-GENERATOR CANDIDATE RECALL
#
# Goal:
#
# Test whether independent candidate generators provide
# better recall than NCC alone.
#
# Generators:
#   1. Intensity / NCC
#   2. Gradient magnitude
#   3. Edge structure
#   4. Frequency-domain local similarity
#
# We measure candidate recall before any verification.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "multi_generator"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "multi_generator_results.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

SEARCH_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

SUPERSAMPLE = 2

TEMPLATE_SIZE = 10

PER_GENERATOR_K = 250

UNION_K_VALUES = [
    50,
    100,
    250,
    500,
    1000,
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
# PERIODIC
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
# QUASI-PERIODIC
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
# NON-MAXIMUM SUPPRESSION
# ============================================================

def extract_top_candidates(
    response,
    k,
):
    work = response.copy()

    candidates = []

    suppression_radius = (
        TEMPLATE_SIZE * 2
    )

    for _ in range(k):

        _, score, _, loc = (
            cv2.minMaxLoc(
                work
            )
        )

        if not np.isfinite(
            score
        ):
            break

        x, y = loc

        candidates.append(
            (
                x,
                y,
                float(score),
            )
        )

        x0 = max(
            0,
            x - suppression_radius,
        )

        x1 = min(
            work.shape[1],
            x + suppression_radius + 1,
        )

        y0 = max(
            0,
            y - suppression_radius,
        )

        y1 = min(
            work.shape[0],
            y + suppression_radius + 1,
        )

        work[
            y0:y1,
            x0:x1
        ] = -np.inf

    return candidates


# ============================================================
# GRADIENT IMAGE
# ============================================================

def gradient_magnitude(
    image,
):
    image = image.astype(
        np.float32
    )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    return cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )


# ============================================================
# EDGE IMAGE
# ============================================================

def edge_image(
    image,
):
    return cv2.Canny(
        image,
        40,
        120,
    )


# ============================================================
# FREQUENCY RESPONSE
# ============================================================

def frequency_response(
    search,
    reference,
):
    """
    Local Fourier correlation.

    The reference is transformed into a spectral signature.
    Each search patch receives a corresponding frequency
    similarity score.
    """

    h, w = search.shape

    t = reference.astype(
        np.float32
    )

    t -= np.mean(t)

    t_fft = np.fft.fftshift(
        np.fft.fft2(
            t
        )
    )

    t_mag = np.log1p(
        np.abs(
            t_fft
        )
    )

    t_mag /= (
        np.linalg.norm(
            t_mag
        )
        + 1e-8
    )

    response = np.zeros(
        (
            h - TEMPLATE_SIZE + 1,
            w - TEMPLATE_SIZE + 1,
        ),
        dtype=np.float32,
    )

    for y in range(
        response.shape[0]
    ):

        for x in range(
            response.shape[1]
        ):

            patch = search[
                y:y + TEMPLATE_SIZE,
                x:x + TEMPLATE_SIZE,
            ].astype(
                np.float32
            )

            patch -= np.mean(
                patch
            )

            fft = np.fft.fftshift(
                np.fft.fft2(
                    patch
                )
            )

            mag = np.log1p(
                np.abs(
                    fft
                )
            )

            mag /= (
                np.linalg.norm(
                    mag
                )
                + 1e-8
            )

            response[
                y,
                x
            ] = np.sum(
                t_mag
                * mag
            )

    return response


# ============================================================
# ONE SCENE
# ============================================================

def evaluate_scene(
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

    # --------------------------------------------------------
    # Generator 1: NCC
    # --------------------------------------------------------

    ncc_response = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED,
    )

    ncc_candidates = (
        extract_top_candidates(
            ncc_response,
            PER_GENERATOR_K,
        )
    )

    # --------------------------------------------------------
    # Generator 2: Gradient
    # --------------------------------------------------------

    search_gradient = (
        gradient_magnitude(
            search
        )
    )

    reference_gradient = (
        gradient_magnitude(
            reference
        )
    )

    gradient_response = (
        cv2.matchTemplate(
            search_gradient,
            reference_gradient,
            cv2.TM_CCOEFF_NORMED,
        )
    )

    gradient_candidates = (
        extract_top_candidates(
            gradient_response,
            PER_GENERATOR_K,
        )
    )

    # --------------------------------------------------------
    # Generator 3: Edge
    # --------------------------------------------------------

    search_edges = edge_image(
        search
    )

    reference_edges = edge_image(
        reference
    )

    edge_response = cv2.matchTemplate(
        search_edges,
        reference_edges,
        cv2.TM_CCOEFF_NORMED,
    )

    edge_candidates = (
        extract_top_candidates(
            edge_response,
            PER_GENERATOR_K,
        )
    )

    # --------------------------------------------------------
    # Generator 4: Frequency
    # --------------------------------------------------------

    frequency_response_map = (
        frequency_response(
            search,
            reference,
        )
    )

    frequency_candidates = (
        extract_top_candidates(
            frequency_response_map,
            PER_GENERATOR_K,
        )
    )

    # --------------------------------------------------------
    # Candidate coordinate sets
    # --------------------------------------------------------

    generators = {
        "ncc": ncc_candidates,
        "gradient": gradient_candidates,
        "edge": edge_candidates,
        "frequency": frequency_candidates,
    }

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Recall helper
    # --------------------------------------------------------

    def recall(
        candidates,
        k,
    ):
        selected = candidates[
            :k
        ]

        for x, y, _ in selected:

            if np.hypot(
                x - gt_left,
                y - gt_top,
            ) <= 2.0:

                return True

        return False

    # --------------------------------------------------------
    # Individual generator recall
    # --------------------------------------------------------

    individual = {}

    for name, candidates in (
        generators.items()
    ):

        for k in [
            50,
            100,
            250,
        ]:

            individual[
                f"{name}@{k}"
            ] = recall(
                candidates,
                k,
            )

    # --------------------------------------------------------
    # Union candidates
    #
    # Preserve generator diversity.
    # --------------------------------------------------------

    union = {}

    for name, candidates in (
        generators.items()
    ):

        for x, y, score in candidates:

            key = (
                int(x),
                int(y),
            )

            if key not in union:

                union[key] = {
                    "x": x,
                    "y": y,
                    "sources": [],
                }

            union[key][
                "sources"
            ].append(
                name
            )

    union_candidates = list(
        union.values()
    )

    # --------------------------------------------------------
    # Rank union by number of independent generators
    # supporting the candidate.
    #
    # This is NOT the final verifier.
    # It only creates a diversity-aware candidate set.
    # --------------------------------------------------------

    union_candidates.sort(
        key=lambda c: (
            -len(
                set(
                    c["sources"]
                )
            ),
        )
    )

    # Add candidates not covered by support count.
    #
    # Stable ordering is preserved.
    #
    # Then evaluate recall for several K values.
    # --------------------------------------------------------

    union_results = {}

    for k in UNION_K_VALUES:

        union_results[
            f"union@{k}"
        ] = False

        selected = (
            union_candidates[
                :k
            ]
        )

        for candidate in selected:

            if np.hypot(
                candidate["x"]
                - gt_left,
                candidate["y"]
                - gt_top,
            ) <= 2.0:

                union_results[
                    f"union@{k}"
                ] = True

                break

    return {
        **individual,
        **union_results,
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
        "MICRONYX STEP 18"
    )
    print(
        "MULTI-GENERATOR CANDIDATE RECALL"
    )
    print("=" * 76)
    print()

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    print(
        f"Seeds:              {len(SEEDS)}"
    )

    print(
        f"Scene types:        {SCENE_TYPES}"
    )

    print(
        f"Total scenes:       {total}"
    )

    print(
        f"Per-generator K:    {PER_GENERATOR_K}"
    )

    print(
        f"Union K:             {UNION_K_VALUES}"
    )

    print()

    results = []

    completed = 0

    for scene_type in SCENE_TYPES:

        print()
        print(
            f"SCENE: {scene_type.upper()}"
        )

        print("-" * 76)

        for seed in SEEDS:

            result = evaluate_scene(
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

        for result in results:

            f.write(
                f"{result['scene']},"
                f"{result['seed']},"
                + ",".join(
                    str(
                        int(
                            result[key]
                        )
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

        # ----------------------------------------------------
        # Individual generators
        # ----------------------------------------------------

        for generator in [
            "ncc",
            "gradient",
            "edge",
            "frequency",
        ]:

            values = []

            for k in [
                50,
                100,
                250,
            ]:

                key = (
                    f"{generator}@{k}"
                )

                recall = (
                    np.mean(
                        [
                            r[key]
                            for r in subset
                        ]
                    )
                    * 100
                )

                print(
                    f"{key:<18}"
                    f"Recall="
                    f"{recall:6.2f}%"
                )

        print()

        # ----------------------------------------------------
        # Union
        # ----------------------------------------------------

        for k in UNION_K_VALUES:

            key = (
                f"union@{k}"
            )

            recall = (
                np.mean(
                    [
                        r[key]
                        for r in subset
                    ]
                )
                * 100
            )

            print(
                f"{key:<18}"
                f"Recall="
                f"{recall:6.2f}%"
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