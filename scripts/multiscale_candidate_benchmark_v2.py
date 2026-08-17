"""
MICRONYX — STEP 32A
OPTIMIZED MULTI-SCALE PHYSICAL-CONTEXT CANDIDATE BENCHMARK

Canonical PS02 acquisition model
---------------------------------

Search:
    1000 x 1000 pixels
    5 pixels / physical unit
    200 x 200 physical-unit FOV

Reference:
    1000 x 1000 pixels
    50 pixels / physical unit
    20 x 20 physical-unit FOV

Sampling ratio:
    10x

Physical contexts:

    2 units:
        reference crop = 100 x 100
        search equivalent = 10 x 10

    4 units:
        reference crop = 200 x 200
        search equivalent = 20 x 20

    8 units:
        reference crop = 400 x 400
        search equivalent = 40 x 40

The reference crop is converted to the search-equivalent observation
using deterministic non-overlapping 10x10 block averaging.

IMPORTANT:
    No target fingerprint.
    No new ground truth.
    No alternate renderer.
    No arbitrary interpolation.
    No manual scene selection.

OPTIMIZATION:
    The original implementation repeatedly:
        - calculated Sobel gradients
        - sorted ~1M score locations completely
        - performed expensive Python NMS

    This version:
        - calculates search gradient once per scene
        - calculates template gradient once per context
        - uses np.argpartition instead of full sorting
        - uses grid-based spatial NMS

The physical benchmark methodology remains unchanged.
"""

from pathlib import Path
import sys
import time
import json

import cv2
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = ROOT / "scripts"

OUT = (
    ROOT
    / "validation"
    / "v02"
    / "multiscale_candidate_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(
    0,
    str(SCRIPTS)
)

import canonical_renderer as cr


# ============================================================================
# CANONICAL ACQUISITION MODEL
# ============================================================================

SEARCH_W = 1000
SEARCH_H = 1000

REFERENCE_W = 1000
REFERENCE_H = 1000

SEARCH_PIXELS_PER_UNIT = 5.0
REFERENCE_PIXELS_PER_UNIT = 50.0

SAMPLING_RATIO = (
    REFERENCE_PIXELS_PER_UNIT
    / SEARCH_PIXELS_PER_UNIT
)

DOWNSAMPLE_FACTOR = int(
    round(SAMPLING_RATIO)
)

if not np.isclose(
    SAMPLING_RATIO,
    10.0
):
    raise RuntimeError(
        "PS02 requires a 10x sampling ratio."
    )


# ============================================================================
# TARGET
# ============================================================================

TARGET_X = 75.25
TARGET_Y = 113.75

TARGET_SEARCH_X = (
    TARGET_X
    * SEARCH_PIXELS_PER_UNIT
)

TARGET_SEARCH_Y = (
    TARGET_Y
    * SEARCH_PIXELS_PER_UNIT
)


# ============================================================================
# PHYSICAL CONTEXTS
# ============================================================================

CONTEXT_PHYSICAL_SIZES = [
    2.0,
    4.0,
    8.0
]

REFERENCE_CONTEXT_SIZES = [
    int(
        round(
            x
            * REFERENCE_PIXELS_PER_UNIT
        )
    )
    for x in CONTEXT_PHYSICAL_SIZES
]

SEARCH_CONTEXT_SIZES = [
    int(
        round(
            x
            * SEARCH_PIXELS_PER_UNIT
        )
    )
    for x in CONTEXT_PHYSICAL_SIZES
]


# ============================================================================
# BENCHMARK SETTINGS
# ============================================================================

K_VALUES = [
    10,
    25,
    50,
    100,
    250,
    500
]

PRIMARY_K = 250

TOLERANCE_PX = 5.0

SCENE_COUNT = 60

PERIODIC_COUNT = 30

QUASIPERIODIC_COUNT = 30

SEED_START = 20260845


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Number of raw score locations retained before spatial NMS.
#
# This avoids sorting ~1 million locations completely.
#
# 20,000 is deliberately generous relative to K<=500.
TOP_RAW_CANDIDATES = 20000

NMS_DISTANCE = 2.0


# ============================================================================
# UTILITIES
# ============================================================================

def to_gray(image):
    """
    Convert renderer output to float32 grayscale [0,1].
    """

    if image is None:
        raise RuntimeError(
            "Canonical renderer returned None."
        )

    image = np.asarray(image)

    if image.ndim == 3:

        if image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2GRAY
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

    image = image.astype(
        np.float32
    )

    if image.size == 0:

        raise RuntimeError(
            "Renderer returned an empty image."
        )

    if image.max() > 1.0:

        image /= 255.0

    return image


def normalize01(x):
    """
    Normalize score map to [0,1].
    """

    x = np.asarray(
        x,
        dtype=np.float32
    )

    mn = float(
        np.min(x)
    )

    mx = float(
        np.max(x)
    )

    if mx - mn < 1e-12:

        return np.zeros_like(
            x
        )

    return (
        x - mn
    ) / (
        mx - mn
    )


def gradient_magnitude(image):
    """
    Sobel gradient magnitude.
    """

    image = image.astype(
        np.float32,
        copy=False
    )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    return np.sqrt(
        gx * gx
        +
        gy * gy
    )


def crop_center(
    image,
    width,
    height
):
    """
    Exact center crop.

    No interpolation.
    """

    h, w = image.shape[:2]

    cx = w // 2
    cy = h // 2

    x0 = int(
        round(
            cx
            - width / 2.0
        )
    )

    y0 = int(
        round(
            cy
            - height / 2.0
        )
    )

    x1 = x0 + width
    y1 = y0 + height

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):

        raise RuntimeError(
            f"Cannot extract "
            f"{width}x{height} crop "
            f"from {w}x{h} reference."
        )

    return image[
        y0:y1,
        x0:x1
    ].copy()


# ============================================================================
# PHYSICAL DOWNSAMPLING
# ============================================================================

def reference_to_search_equivalent(
    reference_crop
):
    """
    Deterministic 10x block averaging.

    Every 10x10 high-resolution reference block
    becomes one search-resolution pixel.
    """

    reference_crop = np.asarray(
        reference_crop,
        dtype=np.float32
    )

    h, w = reference_crop.shape[:2]

    factor = DOWNSAMPLE_FACTOR

    if (
        h % factor != 0
        or w % factor != 0
    ):

        raise RuntimeError(
            "Reference crop dimensions "
            "must be divisible by 10."
        )

    output_h = h // factor
    output_w = w // factor

    reshaped = reference_crop.reshape(
        output_h,
        factor,
        output_w,
        factor
    )

    result = reshaped.mean(
        axis=(1, 3)
    )

    return result.astype(
        np.float32
    )


# ============================================================================
# MATCHING
# ============================================================================

def match_ncc(
    search,
    template
):
    """
    Normalized cross correlation.
    """

    return cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )


def match_gradient(
    search_gradient,
    template_gradient
):
    """
    Gradient-domain NCC.

    Search gradient is precomputed once per scene.
    """

    return cv2.matchTemplate(
        search_gradient,
        template_gradient,
        cv2.TM_CCOEFF_NORMED
    )


# ============================================================================
# FAST CANDIDATE EXTRACTION
# ============================================================================

def score_map_to_candidates(
    score_map,
    template_shape,
    max_candidates=TOP_RAW_CANDIDATES
):
    """
    Extract high-scoring candidate centers without performing
    a complete sort over the entire ~1M-element score map.

    np.argpartition gives the top M elements in O(N) average
    selection time, after which only those M elements are sorted.
    """

    th, tw = template_shape

    flat = score_map.reshape(-1)

    n = flat.size

    count = min(
        int(max_candidates),
        n
    )

    if count <= 0:

        return (
            np.empty(0),
            np.empty(0),
            np.empty(0)
        )

    if count == n:

        indices = np.argsort(
            flat
        )[::-1]

    else:

        partition = np.argpartition(
            flat,
            -count
        )[-count:]

        indices = partition[
            np.argsort(
                flat[partition]
            )[::-1]
        ]

    scores = flat[
        indices
    ].astype(
        np.float64
    )

    map_width = score_map.shape[1]

    ys = (
        indices
        // map_width
    )

    xs = (
        indices
        % map_width
    )

    centers_x = (
        xs.astype(
            np.float64
        )
        +
        tw / 2.0
    )

    centers_y = (
        ys.astype(
            np.float64
        )
        +
        th / 2.0
    )

    return (
        scores,
        centers_x,
        centers_y
    )


# ============================================================================
# FAST GRID-BASED NMS
# ============================================================================

def suppress_nearby_fast(
    scores,
    xs,
    ys,
    min_distance
):
    """
    Greedy spatial NMS using a grid.

    This preserves score-descending selection while avoiding
    the O(N^2) pairwise distance calculation of the original
    implementation.
    """

    if len(scores) == 0:

        return (
            np.empty(0),
            np.empty(0),
            np.empty(0)
        )

    cell_size = float(
        min_distance
    )

    min_d2 = (
        float(min_distance)
        *
        float(min_distance)
    )

    selected_scores = []
    selected_x = []
    selected_y = []

    grid = {}

    for score, x, y in zip(
        scores,
        xs,
        ys
    ):

        gx = int(
            np.floor(
                x / cell_size
            )
        )

        gy = int(
            np.floor(
                y / cell_size
            )
        )

        accepted = True

        for nx in range(
            gx - 1,
            gx + 2
        ):

            for ny in range(
                gy - 1,
                gy + 2
            ):

                bucket = grid.get(
                    (nx, ny)
                )

                if bucket is None:
                    continue

                for idx in bucket:

                    dx = (
                        selected_x[idx]
                        -
                        x
                    )

                    dy = (
                        selected_y[idx]
                        -
                        y
                    )

                    if (
                        dx * dx
                        +
                        dy * dy
                        <
                        min_d2
                    ):

                        accepted = False
                        break

                if not accepted:
                    break

            if not accepted:
                break

        if not accepted:
            continue

        idx = len(
            selected_x
        )

        selected_scores.append(
            score
        )

        selected_x.append(
            x
        )

        selected_y.append(
            y
        )

        grid.setdefault(
            (gx, gy),
            []
        ).append(
            idx
        )

    return (
        np.asarray(
            selected_scores,
            dtype=np.float64
        ),
        np.asarray(
            selected_x,
            dtype=np.float64
        ),
        np.asarray(
            selected_y,
            dtype=np.float64
        )
    )


# ============================================================================
# CANDIDATE GENERATION
# ============================================================================

def generate_candidates(
    search,
    search_gradient,
    template,
    method
):
    """
    Generate ranked candidates.

    Methods:
        ncc
        gradient
        fusion
    """

    if method == "ncc":

        score_map = match_ncc(
            search,
            template
        )

    elif method == "gradient":

        template_gradient = (
            gradient_magnitude(
                template
            )
        )

        score_map = match_gradient(
            search_gradient,
            template_gradient
        )

    elif method == "fusion":

        ncc_map = match_ncc(
            search,
            template
        )

        template_gradient = (
            gradient_magnitude(
                template
            )
        )

        gradient_map = match_gradient(
            search_gradient,
            template_gradient
        )

        ncc_norm = normalize01(
            ncc_map
        )

        gradient_norm = normalize01(
            gradient_map
        )

        score_map = (
            0.5 * ncc_norm
            +
            0.5 * gradient_norm
        )

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    scores, xs, ys = (
        score_map_to_candidates(
            score_map,
            template.shape
        )
    )

    return suppress_nearby_fast(
        scores,
        xs,
        ys,
        NMS_DISTANCE
    )


# ============================================================================
# METRICS
# ============================================================================

def distances_to_target(
    xs,
    ys
):

    dx = (
        xs
        -
        TARGET_SEARCH_X
    )

    dy = (
        ys
        -
        TARGET_SEARCH_Y
    )

    return np.sqrt(
        dx * dx
        +
        dy * dy
    )


def recall_at_k(
    xs,
    ys,
    k
):

    if len(xs) == 0:
        return 0.0

    k = min(
        int(k),
        len(xs)
    )

    distances = distances_to_target(
        xs[:k],
        ys[:k]
    )

    return float(
        np.any(
            distances
            <= TOLERANCE_PX
        )
    )


def gt_rank(
    xs,
    ys
):

    if len(xs) == 0:
        return -1

    distances = distances_to_target(
        xs,
        ys
    )

    valid = np.where(
        distances
        <= TOLERANCE_PX
    )[0]

    if len(valid) == 0:
        return -1

    return int(
        valid[0] + 1
    )


def top_error(
    xs,
    ys
):

    if len(xs) == 0:
        return float("inf")

    return float(
        distances_to_target(
            xs[:1],
            ys[:1]
        )[0]
    )


# ============================================================================
# CANONICAL RENDERING
# ============================================================================

def render_scene(
    scene_type,
    seed
):

    search = cr.render_search(
        TARGET_X,
        TARGET_Y,
        scene_type,
        seed
    )

    reference = cr.render_reference(
        TARGET_X,
        TARGET_Y,
        scene_type,
        seed
    )

    search = to_gray(
        search
    )

    reference = to_gray(
        reference
    )

    if search.shape != (
        SEARCH_H,
        SEARCH_W
    ):

        raise RuntimeError(
            f"Unexpected search shape: "
            f"{search.shape}"
        )

    if reference.shape != (
        REFERENCE_H,
        REFERENCE_W
    ):

        raise RuntimeError(
            f"Unexpected reference shape: "
            f"{reference.shape}"
        )

    return (
        search,
        reference
    )


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def build_context_templates(
    reference
):

    templates = {}

    for physical_size, reference_size, search_size in zip(
        CONTEXT_PHYSICAL_SIZES,
        REFERENCE_CONTEXT_SIZES,
        SEARCH_CONTEXT_SIZES
    ):

        crop = crop_center(
            reference,
            reference_size,
            reference_size
        )

        template = (
            reference_to_search_equivalent(
                crop
            )
        )

        expected = (
            search_size,
            search_size
        )

        if template.shape != expected:

            raise RuntimeError(
                f"Template shape mismatch: "
                f"{template.shape} != {expected}"
            )

        templates[
            search_size
        ] = template

    return templates


# ============================================================================
# MAIN
# ============================================================================

def main():

    total_start = time.perf_counter()

    print("=" * 76)
    print(
        "MICRONYX STEP 32A"
    )
    print(
        "OPTIMIZED MULTI-SCALE "
        "PHYSICAL-CONTEXT CANDIDATE BENCHMARK"
    )
    print("=" * 76)

    print()
    print(
        "Canonical renderer:"
    )

    print(
        cr.__file__
    )

    print()
    print(
        "Canonical acquisition:"
    )

    print(
        f"  Search:    "
        f"{SEARCH_W} × {SEARCH_H}"
    )

    print(
        f"  Reference: "
        f"{REFERENCE_W} × {REFERENCE_H}"
    )

    print(
        f"  Search sampling:    "
        f"{SEARCH_PIXELS_PER_UNIT:.1f} px/unit"
    )

    print(
        f"  Reference sampling: "
        f"{REFERENCE_PIXELS_PER_UNIT:.1f} px/unit"
    )

    print(
        f"  Sampling ratio:     "
        f"{SAMPLING_RATIO:.1f}x"
    )

    print()
    print(
        "Target physical:"
    )

    print(
        f"  ({TARGET_X:.4f}, "
        f"{TARGET_Y:.4f})"
    )

    print()
    print(
        "Target search CENTER:"
    )

    print(
        f"  ({TARGET_SEARCH_X:.4f}, "
        f"{TARGET_SEARCH_Y:.4f})"
    )

    print()
    print(
        "Reference physical FOV:"
    )

    print(
        f"  "
        f"{REFERENCE_W / REFERENCE_PIXELS_PER_UNIT:.2f}"
        f" × "
        f"{REFERENCE_H / REFERENCE_PIXELS_PER_UNIT:.2f}"
        f" physical units"
    )

    print()
    print(
        "PHYSICAL CONTEXTS"
    )

    print("-" * 76)

    for physical_size, reference_size, search_size in zip(
        CONTEXT_PHYSICAL_SIZES,
        REFERENCE_CONTEXT_SIZES,
        SEARCH_CONTEXT_SIZES
    ):

        print(
            f"  {physical_size:.1f} units:"
            f"  reference crop "
            f"{reference_size}×{reference_size}"
            f"  -> search template "
            f"{search_size}×{search_size}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Reference contexts are canonical crops."
    )

    print(
        "  Search-equivalent templates use "
        "10x deterministic block averaging."
    )

    print(
        "  No arbitrary interpolation."
    )

    print(
        "  No target fingerprint."
    )

    print(
        "  No new ground truth."
    )

    print(
        "  No alternate renderer."
    )

    print()
    print(
        "PERFORMANCE:"
    )

    print(
        f"  Raw candidates retained: "
        f"{TOP_RAW_CANDIDATES}"
    )

    print(
        f"  Spatial NMS distance: "
        f"{NMS_DISTANCE:.1f}px"
    )

    methods = [
        "ncc",
        "gradient",
        "fusion"
    ]

    rows = []

    # ========================================================================
    # SCENES
    # ========================================================================

    for scene_index in range(
        SCENE_COUNT
    ):

        scene_start = time.perf_counter()

        if scene_index < PERIODIC_COUNT:

            scene_type = "periodic"

        else:

            scene_type = "quasiperiodic"

        seed = (
            SEED_START
            +
            scene_index
        )

        print()
        print(
            "=" * 76
        )

        print(
            f"SCENE "
            f"{scene_index + 1:02d}/"
            f"{SCENE_COUNT} "
            f"{scene_type.upper()}"
        )

        print(
            "=" * 76
        )

        # --------------------------------------------------------------------
        # CANONICAL OBSERVATION
        # --------------------------------------------------------------------

        search, reference = (
            render_scene(
                scene_type,
                seed
            )
        )

        # --------------------------------------------------------------------
        # IMPORTANT OPTIMIZATION:
        # Search gradient is identical for all three contexts and methods.
        # Calculate it ONCE.
        # --------------------------------------------------------------------

        search_gradient = (
            gradient_magnitude(
                search
            )
        )

        # --------------------------------------------------------------------
        # Build physically correct contexts.
        # --------------------------------------------------------------------

        templates = (
            build_context_templates(
                reference
            )
        )

        # --------------------------------------------------------------------
        # Matching
        # --------------------------------------------------------------------

        for physical_size, search_size in zip(
            CONTEXT_PHYSICAL_SIZES,
            SEARCH_CONTEXT_SIZES
        ):

            template = templates[
                search_size
            ]

            for method in methods:

                method_start = (
                    time.perf_counter()
                )

                scores, xs, ys = (
                    generate_candidates(
                        search,
                        search_gradient,
                        template,
                        method
                    )
                )

                rank = gt_rank(
                    xs,
                    ys
                )

                error = top_error(
                    xs,
                    ys
                )

                elapsed = (
                    time.perf_counter()
                    -
                    method_start
                )

                for k in K_VALUES:

                    recall = recall_at_k(
                        xs,
                        ys,
                        k
                    )

                    rows.append({

                        "scene_type":
                            scene_type,

                        "seed":
                            seed,

                        "scene_index":
                            scene_index + 1,

                        "context_physical_units":
                            physical_size,

                        "reference_context_px":
                            int(
                                round(
                                    physical_size
                                    * REFERENCE_PIXELS_PER_UNIT
                                )
                            ),

                        "search_template_px":
                            search_size,

                        "method":
                            method,

                        "K":
                            k,

                        "recall":
                            recall,

                        "recall_percent":
                            recall * 100.0,

                        "top1_error_px":
                            error,

                        "gt_rank":
                            rank,

                        "candidate_count":
                            len(xs),

                        "method_runtime_seconds":
                            elapsed
                    })

        scene_elapsed = (
            time.perf_counter()
            -
            scene_start
        )

        print(
            f"Scene completed in "
            f"{scene_elapsed:.2f}s"
        )

    # ========================================================================
    # DATAFRAME
    # ========================================================================

    df = pd.DataFrame(
        rows
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    for scene_type in [
        "periodic",
        "quasiperiodic"
    ]:

        print()
        print(
            scene_type.upper()
        )

        print("-" * 76)

        subset = df[
            df["scene_type"]
            ==
            scene_type
        ]

        for physical_size, search_size in zip(
            CONTEXT_PHYSICAL_SIZES,
            SEARCH_CONTEXT_SIZES
        ):

            for method in methods:

                vals = subset[
                    (
                        subset[
                            "context_physical_units"
                        ]
                        ==
                        physical_size
                    )
                    &
                    (
                        subset[
                            "method"
                        ]
                        ==
                        method
                    )
                    &
                    (
                        subset[
                            "K"
                        ]
                        ==
                        PRIMARY_K
                    )
                ]

                if len(vals) == 0:
                    continue

                recall = float(
                    vals[
                        "recall_percent"
                    ].mean()
                )

                median_error = float(
                    vals[
                        "top1_error_px"
                    ].median()
                )

                print(
                    f"{method:<10}"
                    f"Context={physical_size:>4.1f}u "
                    f"Template={search_size:>3}px "
                    f"Recall@{PRIMARY_K}="
                    f"{recall:7.2f}% "
                    f"MedianTop1Err="
                    f"{median_error:8.3f}px"
                )

    # ========================================================================
    # BEST STRATEGY
    # ========================================================================

    print()
    print("=" * 76)
    print(
        f"BEST STRATEGY AT K={PRIMARY_K}"
    )
    print("=" * 76)

    best_rows = []

    for scene_type in [
        "periodic",
        "quasiperiodic"
    ]:

        subset = df[
            (
                df["scene_type"]
                ==
                scene_type
            )
            &
            (
                df["K"]
                ==
                PRIMARY_K
            )
        ]

        grouped = (
            subset
            .groupby(
                [
                    "context_physical_units",
                    "search_template_px",
                    "method"
                ],
                as_index=False
            )[
                "recall_percent"
            ]
            .mean()
        )

        best = (
            grouped
            .sort_values(
                "recall_percent",
                ascending=False
            )
            .iloc[0]
        )

        print()
        print(
            scene_type.upper()
        )

        print(
            f"Best physical context: "
            f"{best['context_physical_units']:.1f} units"
        )

        print(
            f"Search-equivalent template: "
            f"{int(best['search_template_px'])}×"
            f"{int(best['search_template_px'])}"
        )

        print(
            f"Best method: "
            f"{best['method']}"
        )

        print(
            f"Recall@{PRIMARY_K}: "
            f"{best['recall_percent']:.2f}%"
        )

        best_rows.append({

            "scene_type":
                scene_type,

            "context_physical_units":
                float(
                    best[
                        "context_physical_units"
                    ]
                ),

            "search_template_px":
                int(
                    best[
                        "search_template_px"
                    ]
                ),

            "method":
                best[
                    "method"
                ],

            "recall_percent":
                float(
                    best[
                        "recall_percent"
                    ]
                )
        })

    # ========================================================================
    # RUNTIME
    # ========================================================================

    runtime = (
        time.perf_counter()
        -
        total_start
    )

    # ========================================================================
    # OUTPUT
    # ========================================================================

    result_csv = (
        OUT
        /
        "multiscale_candidate_v2_results.csv"
    )

    summary_csv = (
        OUT
        /
        "multiscale_candidate_v2_summary.csv"
    )

    summary_json = (
        OUT
        /
        "multiscale_candidate_v2_summary.json"
    )

    df.to_csv(
        result_csv,
        index=False
    )

    pd.DataFrame(
        best_rows
    ).to_csv(
        summary_csv,
        index=False
    )

    summary = {

        "step":
            "32A",

        "canonical_renderer":
            True,

        "search_shape":
            [
                SEARCH_H,
                SEARCH_W
            ],

        "reference_shape":
            [
                REFERENCE_H,
                REFERENCE_W
            ],

        "search_pixels_per_unit":
            SEARCH_PIXELS_PER_UNIT,

        "reference_pixels_per_unit":
            REFERENCE_PIXELS_PER_UNIT,

        "sampling_ratio":
            SAMPLING_RATIO,

        "context_physical_units":
            CONTEXT_PHYSICAL_SIZES,

        "reference_context_sizes_px":
            REFERENCE_CONTEXT_SIZES,

        "search_equivalent_template_sizes_px":
            SEARCH_CONTEXT_SIZES,

        "downsample_factor":
            DOWNSAMPLE_FACTOR,

        "downsampling_method":
            "deterministic_non_overlapping_block_mean",

        "arbitrary_interpolation":
            False,

        "template_resizing":
            False,

        "target_fingerprint":
            False,

        "new_ground_truth":
            False,

        "alternate_renderer":
            False,

        "methods":
            methods,

        "K_values":
            K_VALUES,

        "primary_K":
            PRIMARY_K,

        "candidate_tolerance_px":
            TOLERANCE_PX,

        "scene_count":
            SCENE_COUNT,

        "top_raw_candidates":
            TOP_RAW_CANDIDATES,

        "nms_distance_px":
            NMS_DISTANCE,

        "best_strategies":
            best_rows,

        "runtime_seconds":
            runtime
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================================
    # FINAL REPORT
    # ========================================================================

    print()
    print("=" * 76)
    print("FINAL REPORT")
    print("=" * 76)

    print(
        "Canonical renderer:              YES"
    )

    print(
        "Search observation:              1000 × 1000"
    )

    print(
        "Reference observation:           1000 × 1000"
    )

    print(
        f"Sampling ratio:                   "
        f"{SAMPLING_RATIO:.1f}x"
    )

    print(
        "Physical context crops:          YES"
    )

    print(
        "Block averaging:                 YES"
    )

    print(
        "Arbitrary interpolation:         NO"
    )

    print(
        "Template resizing:               NO"
    )

    print(
        "Target fingerprint:              NO"
    )

    print(
        "New ground truth:                NO"
    )

    print(
        "Alternate renderer:              NO"
    )

    print(
        f"Scenes:                           "
        f"{SCENE_COUNT}"
    )

    print(
        f"Primary K:                        "
        f"{PRIMARY_K}"
    )

    print(
        f"Runtime:                          "
        f"{runtime:.2f} seconds"
    )

    print()
    print("=" * 76)
    print("SAVED")
    print("=" * 76)

    print(
        result_csv
    )

    print(
        summary_csv
    )

    print(
        summary_json
    )

    print()
    print("=" * 76)
    print("STEP 32A COMPLETE")
    print("=" * 76)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()