from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 16
# CANDIDATE VERIFICATION BENCHMARK
#
# Goal:
# Determine whether independent evidence can disambiguate
# periodic / quasi-periodic candidates.
#
# Tested evidence:
#   1. Local appearance
#   2. Gradient structure
#   3. Orientation structure
#   4. Frequency signature
#   5. Multi-scale context
#
# IMPORTANT:
# We do NOT assume weights.
# Multiple weight configurations are experimentally tested.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "candidate_verification"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "candidate_verification_results.csv"
)


# ============================================================
# GEOMETRY
# ============================================================

SEARCH_SIZE = 1000

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

MAGNIFICATION = 10.0

SUPERSAMPLE = 2


# ============================================================
# EXPERIMENT
# ============================================================

SEEDS = range(
    20260850,
    20260865,
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

TEMPLATE_SIZE = 10

NUM_CANDIDATES = 50


# ============================================================
# TARGET
# ============================================================

BASE_TARGET_X = 75.25
BASE_TARGET_Y = 113.75


# ============================================================
# PHYSICAL SCENE
# ============================================================

BASE_PITCH = 0.50
BASE_LINE_WIDTH = 0.20


# ============================================================
# APERIODIC COMPONENT
# ============================================================

def aperiodic_component(
    x,
    y,
):
    return (
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


# ============================================================
# PERIODIC SCENE
# ============================================================

def periodic_scene(
    x,
    y,
):
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

    pitch_variation = rng.uniform(
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

    return (
        structure
        + modulation
    )


# ============================================================
# TARGET FINGERPRINT
# ============================================================

def target_mask(
    x,
    y,
    target_x,
    target_y,
):
    dx = x - target_x
    dy = y - target_y

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
# CONTINUOUS SCENE
# ============================================================

def render_continuous_scene(
    x,
    y,
    scene_type,
    seed,
    target_x,
    target_y,
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
            scene_type
        )

    target = target_mask(
        x,
        y,
        target_x,
        target_y,
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
    )


# ============================================================
# SENSOR RENDERING
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
        target_x,
        target_y,
    )

    image = image.astype(
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
# REFERENCE
# ============================================================

def create_reference(
    target_x,
    target_y,
    scene_type,
    seed,
):
    physical_fov = (
        TEMPLATE_SIZE
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

    template = cv2.resize(
        reference,
        (
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return template


# ============================================================
# CANDIDATE EXTRACTION
# ============================================================

def extract_candidates(
    result,
    num_candidates,
):
    """
    Extract spatially separated candidates.
    """

    work = result.copy()

    candidates = []

    min_distance = (
        TEMPLATE_SIZE * 2
    )

    for _ in range(
        num_candidates
    ):

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
            x - min_distance,
        )

        x1 = min(
            work.shape[1],
            x + min_distance + 1,
        )

        y0 = max(
            0,
            y - min_distance,
        )

        y1 = min(
            work.shape[0],
            y + min_distance + 1,
        )

        work[
            y0:y1,
            x0:x1
        ] = -np.inf

    return candidates


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patch(
    image,
    center_x,
    center_y,
    size,
):
    half = size // 2

    x = int(
        round(center_x)
    )

    y = int(
        round(center_y)
    )

    x0 = max(
        0,
        x - half,
    )

    y0 = max(
        0,
        y - half,
    )

    x1 = min(
        image.shape[1],
        x + half + 1,
    )

    y1 = min(
        image.shape[0],
        y + half + 1,
    )

    patch = image[
        y0:y1,
        x0:x1
    ]

    if (
        patch.shape[0] != size
        or patch.shape[1] != size
    ):
        patch = cv2.copyMakeBorder(
            patch,
            0,
            max(
                0,
                size - patch.shape[0],
            ),
            0,
            max(
                0,
                size - patch.shape[1],
            ),
            cv2.BORDER_REFLECT,
        )

    return patch[
        :size,
        :size
    ]


# ============================================================
# GRADIENT SCORE
# ============================================================

def gradient_score(
    reference,
    patch,
):
    ref = reference.astype(
        np.float32
    )

    pat = patch.astype(
        np.float32
    )

    ref_gx = cv2.Sobel(
        ref,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    ref_gy = cv2.Sobel(
        ref,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    pat_gx = cv2.Sobel(
        pat,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    pat_gy = cv2.Sobel(
        pat,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    ref_mag = cv2.magnitude(
        ref_gx,
        ref_gy,
    )

    pat_mag = cv2.magnitude(
        pat_gx,
        pat_gy,
    )

    ref_mag = cv2.normalize(
        ref_mag,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    pat_mag = cv2.normalize(
        pat_mag,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return float(
        np.corrcoef(
            ref_mag.ravel(),
            pat_mag.ravel(),
        )[0, 1]
    )


# ============================================================
# ORIENTATION SCORE
# ============================================================

def orientation_score(
    reference,
    patch,
):
    ref = reference.astype(
        np.float32
    )

    pat = patch.astype(
        np.float32
    )

    ref_gx = cv2.Sobel(
        ref,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    ref_gy = cv2.Sobel(
        ref,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    pat_gx = cv2.Sobel(
        pat,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    pat_gy = cv2.Sobel(
        pat,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    ref_angle = np.arctan2(
        ref_gy,
        ref_gx,
    )

    pat_angle = np.arctan2(
        pat_gy,
        pat_gx,
    )

    ref_weight = cv2.magnitude(
        ref_gx,
        ref_gy,
    )

    pat_weight = cv2.magnitude(
        pat_gx,
        pat_gy,
    )

    similarity = np.cos(
        ref_angle
        - pat_angle
    )

    weight = (
        ref_weight
        * pat_weight
    )

    denominator = (
        np.sum(weight)
        + 1e-8
    )

    score = (
        np.sum(
            similarity
            * weight
        )
        / denominator
    )

    return float(
        score
    )


# ============================================================
# FREQUENCY SCORE
# ============================================================

def frequency_score(
    reference,
    patch,
):
    ref = (
        reference.astype(
            np.float32
        )
        - np.mean(reference)
    )

    pat = (
        patch.astype(
            np.float32
        )
        - np.mean(patch)
    )

    ref_fft = np.fft.fftshift(
        np.fft.fft2(
            ref
        )
    )

    pat_fft = np.fft.fftshift(
        np.fft.fft2(
            pat
        )
    )

    ref_mag = np.log1p(
        np.abs(
            ref_fft
        )
    )

    pat_mag = np.log1p(
        np.abs(
            pat_fft
        )
    )

    ref_mag = (
        ref_mag
        / (
            np.linalg.norm(
                ref_mag
            )
            + 1e-8
        )
    )

    pat_mag = (
        pat_mag
        / (
            np.linalg.norm(
                pat_mag
            )
            + 1e-8
        )
    )

    return float(
        np.sum(
            ref_mag
            * pat_mag
        )
    )


# ============================================================
# MULTI-SCALE SCORE
# ============================================================

def multiscale_score(
    search,
    reference,
    center_x,
    center_y,
):
    scores = []

    for scale in [
        1,
        2,
        3,
    ]:

        size = (
            TEMPLATE_SIZE
            * scale
        )

        patch = extract_patch(
            search,
            center_x,
            center_y,
            size,
        )

        ref_scaled = cv2.resize(
            reference,
            (
                size,
                size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        score = cv2.matchTemplate(
            patch,
            ref_scaled,
            cv2.TM_CCOEFF_NORMED,
        )[0, 0]

        scores.append(
            float(score)
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# NORMALIZATION
# ============================================================

def minmax(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    vmin = np.min(
        values
    )

    vmax = np.max(
        values
    )

    if (
        vmax - vmin
        < 1e-12
    ):
        return np.full_like(
            values,
            0.5,
        )

    return (
        values
        - vmin
    ) / (
        vmax
        - vmin
    )


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

    reference = create_reference(
        target_x,
        target_y,
        scene_type,
        seed,
    )

    # --------------------------------------------------------
    # Local appearance candidate generation
    # --------------------------------------------------------

    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED,
    )

    candidates = extract_candidates(
        result,
        NUM_CANDIDATES,
    )

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
            - TEMPLATE_SIZE / 2
        )
    )

    gt_top = int(
        round(
            target_search_y
            - TEMPLATE_SIZE / 2
        )
    )

    # --------------------------------------------------------
    # Guarantee GT candidate is evaluated.
    # --------------------------------------------------------

    gt_center = (
        gt_left
        + TEMPLATE_SIZE / 2.0,
        gt_top
        + TEMPLATE_SIZE / 2.0,
    )

    candidate_centers = [
        (
            x + TEMPLATE_SIZE / 2.0,
            y + TEMPLATE_SIZE / 2.0,
            score,
        )
        for x, y, score
        in candidates
    ]

    # Add GT explicitly.
    candidate_centers.append(
        (
            gt_center[0],
            gt_center[1],
            float(
                result[
                    gt_top,
                    gt_left,
                ]
            ),
        )
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    local_scores = []
    gradient_scores = []
    orientation_scores = []
    frequency_scores = []
    multiscale_scores = []

    candidate_data = []

    for (
        cx,
        cy,
        local,
    ) in candidate_centers:

        patch = extract_patch(
            search,
            cx,
            cy,
            TEMPLATE_SIZE,
        )

        grad = gradient_score(
            reference,
            patch,
        )

        orient = orientation_score(
            reference,
            patch,
        )

        freq = frequency_score(
            reference,
            patch,
        )

        multi = multiscale_score(
            search,
            reference,
            cx,
            cy,
        )

        local_scores.append(
            local
        )

        gradient_scores.append(
            grad
        )

        orientation_scores.append(
            orient
        )

        frequency_scores.append(
            freq
        )

        multiscale_scores.append(
            multi
        )

        distance = float(
            np.hypot(
                cx
                - target_search_x,
                cy
                - target_search_y,
            )
        )

        candidate_data.append(
            {
                "x": cx,
                "y": cy,
                "local": local,
                "gradient": grad,
                "orientation": orient,
                "frequency": freq,
                "multiscale": multi,
                "distance": distance,
            }
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    local_n = minmax(
        local_scores
    )

    gradient_n = minmax(
        gradient_scores
    )

    orientation_n = minmax(
        orientation_scores
    )

    frequency_n = minmax(
        frequency_scores
    )

    multiscale_n = minmax(
        multiscale_scores
    )

    # --------------------------------------------------------
    # Attach normalized features
    # --------------------------------------------------------

    for i, candidate in enumerate(
        candidate_data
    ):
        candidate[
            "local_n"
        ] = local_n[i]

        candidate[
            "gradient_n"
        ] = gradient_n[i]

        candidate[
            "orientation_n"
        ] = orientation_n[i]

        candidate[
            "frequency_n"
        ] = frequency_n[i]

        candidate[
            "multiscale_n"
        ] = multiscale_n[i]

    # --------------------------------------------------------
    # Locate GT candidate
    # --------------------------------------------------------

    gt_index = int(
        np.argmin(
            [
                c["distance"]
                for c in candidate_data
            ]
        )
    )

    # ========================================================
    # SCORE CONFIGURATIONS
    # ========================================================

    configurations = {
        "local": {
            "local": 1.0,
        },

        "gradient": {
            "gradient": 1.0,
        },

        "frequency": {
            "frequency": 1.0,
        },

        "local_gradient": {
            "local": 0.5,
            "gradient": 0.5,
        },

        "local_frequency": {
            "local": 0.5,
            "frequency": 0.5,
        },

        "gradient_frequency": {
            "gradient": 0.5,
            "frequency": 0.5,
        },

        "all_equal": {
            "local": 0.20,
            "gradient": 0.20,
            "orientation": 0.20,
            "frequency": 0.20,
            "multiscale": 0.20,
        },

        "local_heavy": {
            "local": 0.40,
            "gradient": 0.15,
            "orientation": 0.10,
            "frequency": 0.15,
            "multiscale": 0.20,
        },

        "structure_heavy": {
            "local": 0.15,
            "gradient": 0.30,
            "orientation": 0.20,
            "frequency": 0.15,
            "multiscale": 0.20,
        },

        "frequency_heavy": {
            "local": 0.15,
            "gradient": 0.15,
            "orientation": 0.10,
            "frequency": 0.40,
            "multiscale": 0.20,
        },
    }

    outputs = []

    for name, weights in (
        configurations.items()
    ):

        scores = []

        for c in candidate_data:

            score = 0.0

            score += (
                weights.get(
                    "local",
                    0.0,
                )
                * c["local_n"]
            )

            score += (
                weights.get(
                    "gradient",
                    0.0,
                )
                * c["gradient_n"]
            )

            score += (
                weights.get(
                    "orientation",
                    0.0,
                )
                * c["orientation_n"]
            )

            score += (
                weights.get(
                    "frequency",
                    0.0,
                )
                * c["frequency_n"]
            )

            score += (
                weights.get(
                    "multiscale",
                    0.0,
                )
                * c["multiscale_n"]
            )

            scores.append(
                score
            )

        scores = np.asarray(
            scores
        )

        ranking = np.argsort(
            -scores
        )

        predicted_index = int(
            ranking[0]
        )

        predicted = (
            candidate_data[
                predicted_index
            ]
        )

        gt_rank = (
            1
            + int(
                np.where(
                    ranking
                    == gt_index
                )[0][0]
            )
        )

        outputs.append(
            {
                "configuration": name,
                "top1": (
                    predicted_index
                    == gt_index
                ),
                "top5": (
                    gt_rank
                    <= 5
                ),
                "error": predicted[
                    "distance"
                ],
                "gt_rank": gt_rank,
            }
        )

    return outputs


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
        "MICRONYX STEP 16"
    )
    print(
        "CANDIDATE VERIFICATION BENCHMARK"
    )
    print("=" * 76)
    print()

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    print(
        f"Seeds:          {len(SEEDS)}"
    )

    print(
        f"Scene types:    {SCENE_TYPES}"
    )

    print(
        f"Candidates:     {NUM_CANDIDATES}"
    )

    print(
        f"Total scenes:   {total}"
    )

    print()

    all_results = []

    completed = 0

    for scene_type in SCENE_TYPES:

        print()
        print(
            f"SCENE: {scene_type.upper()}"
        )

        print("-" * 76)

        for seed in SEEDS:

            outputs = evaluate_scene(
                scene_type,
                seed,
            )

            for output in outputs:

                output[
                    "scene"
                ] = scene_type

                output[
                    "seed"
                ] = seed

                all_results.append(
                    output
                )

            completed += 1

            print(
                f"Scene {completed:02d}/{total}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "scene,"
            "seed,"
            "configuration,"
            "top1,"
            "top5,"
            "error,"
            "gt_rank\n"
        )

        for r in all_results:

            f.write(
                f"{r['scene']},"
                f"{r['seed']},"
                f"{r['configuration']},"
                f"{int(r['top1'])},"
                f"{int(r['top5'])},"
                f"{r['error']:.6f},"
                f"{r['gt_rank']}\n"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    configurations = [
        "local",
        "gradient",
        "frequency",
        "local_gradient",
        "local_frequency",
        "gradient_frequency",
        "all_equal",
        "local_heavy",
        "structure_heavy",
        "frequency_heavy",
    ]

    print()
    print("=" * 76)
    print(
        "SUMMARY"
    )
    print("=" * 76)

    for scene_type in SCENE_TYPES:

        print()
        print(
            scene_type.upper()
        )

        print("-" * 76)

        for config in configurations:

            subset = [
                r
                for r in all_results
                if r["scene"]
                == scene_type
                and r[
                    "configuration"
                ]
                == config
            ]

            top1 = (
                np.mean(
                    [
                        r["top1"]
                        for r in subset
                    ]
                )
                * 100
            )

            top5 = (
                np.mean(
                    [
                        r["top5"]
                        for r in subset
                    ]
                )
                * 100
            )

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

            within5 = (
                np.mean(
                    errors <= 5
                )
                * 100
            )

            print(
                f"{config:<20} "
                f"Top1={top1:6.2f}% "
                f"Top5={top5:6.2f}% "
                f"<=5px={within5:6.2f}% "
                f"MedErr={np.median(errors):7.3f}px "
                f"MedRank={np.median(ranks):7.1f}"
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