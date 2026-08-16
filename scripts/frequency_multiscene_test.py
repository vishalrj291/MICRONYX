from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 11
# Multi-Scene Frequency Benchmark
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "frequency_multiscene_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SCENES = 30

SEED_BASE = 20260816

SEARCH_SIZE = 1000
REFERENCE_SIZE = 100

MAGNIFICATION = 10.0

PHYSICAL_WIDTH = 200.0
PHYSICAL_HEIGHT = 200.0

SEARCH_PPU = (
    SEARCH_SIZE
    / PHYSICAL_WIDTH
)

REFERENCE_PPU = (
    SEARCH_PPU
    * MAGNIFICATION
)

BASE_PITCH = 0.5
BASE_LINE_WIDTH = 0.2

PITCH_VARIATION = 0.06

PITCH_WAVELENGTH_X = 37.0
PITCH_WAVELENGTH_Y = 43.0

PHASE_AMPLITUDE = 0.08

PHASE_WAVELENGTH_X = 29.0
PHASE_WAVELENGTH_Y = 31.0

FINGERPRINT_WIDTH = 1.0
FINGERPRINT_HEIGHT = 1.0

DEFECT_WIDTH_INCREASE = 0.12
DEFECT_LENGTH = 0.38

SUPERSAMPLE = 2

TOP_K = 30

ALPHAS = [
    0.90,
    0.75,
    0.50,
    0.25,
]


# ============================================================
# RANDOM TARGET
# ============================================================

RNG_MARGIN = 10.0


# ============================================================
# PHYSICAL SCENE FUNCTIONS
# ============================================================

def local_pitch(
    x,
    y,
    phase_x,
    phase_y,
):
    variation = (
        PITCH_VARIATION
        * np.sin(
            2.0
            * np.pi
            * x
            / PITCH_WAVELENGTH_X
            + phase_x
        )
        * np.sin(
            2.0
            * np.pi
            * y
            / PITCH_WAVELENGTH_Y
            + phase_y
        )
    )

    return (
        BASE_PITCH
        * (
            1.0
            + variation
        )
    )


def phase_warp_x(
    x,
    y,
    phase_x,
):
    return (
        PHASE_AMPLITUDE
        * np.sin(
            2.0
            * np.pi
            * x
            / PHASE_WAVELENGTH_X
            + phase_x
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
    phase_y,
):
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
            + phase_y
        )
    )


def base_structure_mask(
    x,
    y,
    phase_x,
    phase_y,
):
    pitch = local_pitch(
        x,
        y,
        phase_x,
        phase_y,
    )

    warped_x = (
        x
        + phase_warp_x(
            x,
            y,
            phase_x,
        )
    )

    warped_y = (
        y
        + phase_warp_y(
            x,
            y,
            phase_y,
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
# TARGET FINGERPRINT
# ============================================================

def fingerprint_mask(
    x,
    y,
    target_x,
    target_y,
):
    inside = (
        (np.abs(x - target_x)
         <= FINGERPRINT_WIDTH / 2.0)
        &
        (np.abs(y - target_y)
         <= FINGERPRINT_HEIGHT / 2.0)
    )

    # Deterministic asymmetric fingerprint.
    defect_1 = (
        (np.abs(
            x
            - (
                target_x
                - 0.22
            )
        )
        < (
            BASE_LINE_WIDTH
            + DEFECT_WIDTH_INCREASE
        ) / 2.0)
        &
        (np.abs(
            y
            - (
                target_y
                - 0.18
            )
        )
        < DEFECT_LENGTH / 2.0)
    )

    defect_2 = (
        (np.abs(
            x
            - (
                target_x
                + 0.21
            )
        )
        < DEFECT_LENGTH / 2.0)
        &
        (np.abs(
            y
            - (
                target_y
                + 0.17
            )
        )
        < (
            BASE_LINE_WIDTH
            + DEFECT_WIDTH_INCREASE
        ) / 2.0)
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
# RENDER
# ============================================================

def render_scene(
    width_px,
    height_px,
    pixels_per_unit,
    origin_x,
    origin_y,
    target_x,
    target_y,
    phase_x,
    phase_y,
):
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

    dx = 1.0 / high_ppu

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
        * dx
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    base = base_structure_mask(
        X,
        Y,
        phase_x,
        phase_y,
    )

    fingerprint = fingerprint_mask(
        X,
        Y,
        target_x,
        target_y,
    )

    image = (
        base
        | fingerprint
    )

    image = (
        image.astype(np.uint8)
        * 255
    )

    image = cv2.resize(
        image,
        (
            width_px,
            height_px,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return image


# ============================================================
# FFT
# ============================================================

def fft_magnitude(image):

    image = image.astype(
        np.float32
    )

    image = (
        image
        - image.mean()
    )

    spectrum = np.fft.fft2(
        image
    )

    spectrum = np.fft.fftshift(
        spectrum
    )

    magnitude = np.abs(
        spectrum
    )

    magnitude = np.log1p(
        magnitude
    )

    norm = np.linalg.norm(
        magnitude
    )

    if norm < 1e-8:
        return magnitude

    return magnitude / norm


def frequency_similarity(
    a,
    b,
):
    fa = fft_magnitude(a)
    fb = fft_magnitude(b)

    a = fa.ravel()
    b = fb.ravel()

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if (
        na < 1e-8
        or nb < 1e-8
    ):
        return 0.0

    return float(
        np.dot(a, b)
        /
        (
            na * nb
        )
    )


# ============================================================
# MIN-MAX
# ============================================================

def minmax(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    lo = values.min()
    hi = values.max()

    if (
        hi - lo
        < 1e-12
    ):
        return np.ones_like(
            values
        )

    return (
        values - lo
    ) / (
        hi - lo
    )


# ============================================================
# RUN ONE SCENE
# ============================================================

def run_scene(
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    phase_x = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    phase_y = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    # Keep target away from boundaries.
    target_x = rng.uniform(
        RNG_MARGIN,
        PHYSICAL_WIDTH - RNG_MARGIN,
    )

    target_y = rng.uniform(
        RNG_MARGIN,
        PHYSICAL_HEIGHT - RNG_MARGIN,
    )

    reference_fov_width = (
        REFERENCE_SIZE
        / REFERENCE_PPU
    )

    reference_fov_height = (
        REFERENCE_SIZE
        / REFERENCE_PPU
    )

    reference_origin_x = (
        target_x
        - reference_fov_width / 2.0
    )

    reference_origin_y = (
        target_y
        - reference_fov_height / 2.0
    )

    search = render_scene(
        SEARCH_SIZE,
        SEARCH_SIZE,
        SEARCH_PPU,
        0.0,
        0.0,
        target_x,
        target_y,
        phase_x,
        phase_y,
    )

    reference = render_scene(
        REFERENCE_SIZE,
        REFERENCE_SIZE,
        REFERENCE_PPU,
        reference_origin_x,
        reference_origin_y,
        target_x,
        target_y,
        phase_x,
        phase_y,
    )

    template_size = int(
        round(
            REFERENCE_SIZE
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
            - template_size / 2.0
        )
    )

    gt_top = int(
        round(
            target_search_y
            - template_size / 2.0
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

    suppressed = result.copy()

    candidates = []

    for _ in range(TOP_K):

        _, score, _, location = (
            cv2.minMaxLoc(
                suppressed
            )
        )

        left = int(
            location[0]
        )

        top = int(
            location[1]
        )

        x = (
            left
            + template_size / 2.0
        )

        y = (
            top
            + template_size / 2.0
        )

        distance = float(
            np.sqrt(
                (
                    x
                    - target_search_x
                ) ** 2
                +
                (
                    y
                    - target_search_y
                ) ** 2
            )
        )

        patch = search[
            top:top + template_size,
            left:left + template_size,
        ]

        if patch.shape != (
            template_size,
            template_size,
        ):
            patch = cv2.copyMakeBorder(
                patch,
                0,
                max(
                    0,
                    template_size
                    - patch.shape[0],
                ),
                0,
                max(
                    0,
                    template_size
                    - patch.shape[1],
                ),
                cv2.BORDER_REFLECT,
            )

            patch = patch[
                :template_size,
                :template_size,
            ]

        frequency = (
            frequency_similarity(
                template,
                patch,
            )
        )

        candidates.append(
            {
                "x": x,
                "y": y,
                "spatial": float(score),
                "frequency": frequency,
                "distance": distance,
            }
        )

        radius = max(
            5,
            template_size,
        )

        x1 = max(
            0,
            left - radius,
        )

        y1 = max(
            0,
            top - radius,
        )

        x2 = min(
            suppressed.shape[1],
            left + radius + 1,
        )

        y2 = min(
            suppressed.shape[0],
            top + radius + 1,
        )

        suppressed[
            y1:y2,
            x1:x2
        ] = -1.0

    # --------------------------------------------------------
    # GT rank using spatial score
    # --------------------------------------------------------

    spatial_rank = (
        1
        + sum(
            candidate["spatial"]
            > gt_score
            for candidate in candidates
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    spatial_norm = minmax(
        [
            c["spatial"]
            for c in candidates
        ]
    )

    frequency_norm = minmax(
        [
            c["frequency"]
            for c in candidates
        ]
    )

    for index, candidate in enumerate(
        candidates
    ):

        candidate[
            "spatial_norm"
        ] = float(
            spatial_norm[index]
        )

        candidate[
            "frequency_norm"
        ] = float(
            frequency_norm[index]
        )

    # --------------------------------------------------------
    # Alpha experiments
    # --------------------------------------------------------

    alpha_ranks = {}

    alpha_errors = {}

    for alpha in ALPHAS:

        for candidate in candidates:

            candidate[
                "combined"
            ] = (
                alpha
                * candidate[
                    "spatial_norm"
                ]
                +
                (1.0 - alpha)
                * candidate[
                    "frequency_norm"
                ]
            )

        ranked = sorted(
            candidates,
            key=lambda c:
                c["combined"],
            reverse=True,
        )

        gt_rank = (
            min(
                range(
                    len(ranked)
                ),
                key=lambda i:
                    ranked[i]["distance"],
            )
            + 1
        )

        best_error = (
            ranked[0]["distance"]
        )

        alpha_ranks[
            alpha
        ] = gt_rank

        alpha_errors[
            alpha
        ] = best_error

    return {
        "seed": seed,
        "target_x": target_x,
        "target_y": target_y,
        "gt_score": gt_score,
        "spatial_rank": spatial_rank,
        "alpha_ranks": alpha_ranks,
        "alpha_errors": alpha_errors,
    }


# ============================================================
# MAIN BENCHMARK
# ============================================================

print()
print("=" * 76)
print(
    "MICRONYX STEP 11"
)
print(
    "MULTI-SCENE FREQUENCY BENCHMARK"
)
print("=" * 76)
print()

print(
    f"Scenes:       {NUM_SCENES}"
)

print(
    f"Seeds:        {SEED_BASE} ... "
    f"{SEED_BASE + NUM_SCENES - 1}"
)

print()

results = []

for scene_index in range(
    NUM_SCENES
):

    seed = (
        SEED_BASE
        + scene_index
    )

    result = run_scene(
        seed
    )

    results.append(
        result
    )

    alpha_text = " | ".join(
        [
            f"a={alpha:.2f}:"
            f"r{result['alpha_ranks'][alpha]}"
            for alpha in ALPHAS
        ]
    )

    print(
        f"Scene {scene_index + 1:02d} "
        f"seed={seed} "
        f"GTscore={result['gt_score']:.4f} "
        f"spatial_rank={result['spatial_rank']} "
        f"| {alpha_text}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 76)
print(
    "SUMMARY"
)
print("=" * 76)
print()

print(
    "Spatial-only:"
)

spatial_ranks = np.array(
    [
        result["spatial_rank"]
        for result in results
    ]
)

print(
    f"  Top-1: "
    f"{np.mean(spatial_ranks == 1) * 100:.2f}%"
)

print(
    f"  Top-5: "
    f"{np.mean(spatial_ranks <= 5) * 100:.2f}%"
)

print(
    f"  Median rank: "
    f"{np.median(spatial_ranks):.1f}"
)


# ============================================================
# ALPHA SUMMARY
# ============================================================

for alpha in ALPHAS:

    ranks = np.array(
        [
            result[
                "alpha_ranks"
            ][alpha]
            for result in results
        ]
    )

    errors = np.array(
        [
            result[
                "alpha_errors"
            ][alpha]
            for result in results
        ]
    )

    print()

    print(
        f"alpha={alpha:.2f}"
    )

    print(
        f"  Top-1: "
        f"{np.mean(ranks == 1) * 100:.2f}%"
    )

    print(
        f"  Top-5: "
        f"{np.mean(ranks <= 5) * 100:.2f}%"
    )

    print(
        f"  Median rank: "
        f"{np.median(ranks):.1f}"
    )

    print(
        f"  Median best error: "
        f"{np.median(errors):.2f}px"
    )


# ============================================================
# WRITE CSV
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    header = [
        "seed",
        "target_x",
        "target_y",
        "gt_score",
        "spatial_rank",
    ]

    for alpha in ALPHAS:
        header.append(
            f"rank_alpha_{alpha:.2f}"
        )

    for alpha in ALPHAS:
        header.append(
            f"error_alpha_{alpha:.2f}"
        )

    file.write(
        ",".join(header)
        + "\n"
    )

    for result in results:

        row = [
            str(
                result["seed"]
            ),
            f"{result['target_x']:.6f}",
            f"{result['target_y']:.6f}",
            f"{result['gt_score']:.8f}",
            str(
                result["spatial_rank"]
            ),
        ]

        for alpha in ALPHAS:
            row.append(
                str(
                    result[
                        "alpha_ranks"
                    ][alpha]
                )
            )

        for alpha in ALPHAS:
            row.append(
                f"{result['alpha_errors'][alpha]:.6f}"
            )

        file.write(
            ",".join(row)
            + "\n"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 76)
print(
    f"Saved:\n{OUTPUT_PATH}"
)
print("=" * 76)
print()