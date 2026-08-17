"""
MICRONYX STEP 31
GRADIENT + NCC CANDIDATE GENERATION BENCHMARK

Purpose
-------
Evaluate physically meaningful gradient/NCC candidate generation against
the canonical MICRONYX observation model.

Representations:
    1. NCC
    2. Gradient magnitude correlation
    3. Gradient orientation consistency
    4. Gradient + NCC hybrid

Important methodology:
    - canonical_renderer.py is the ONLY renderer
    - canonical search/reference observations are used
    - canonical PS02 template is used
    - no alternate synthetic scene generator
    - no target fingerprint
    - no new ground truth
    - target coordinates are used ONLY for evaluation
    - candidate generation itself does not know the target

Metrics:
    Recall@K within 5 px
    Top-1 within 5 px
    Median localization error
    95th percentile error
    Candidate rank of ground truth
"""

from pathlib import Path
import json
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

import canonical_renderer as cr


OUT_DIR = (
    ROOT
    / "validation"
    / "v02"
    / "gradient_ncc_candidate"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# BENCHMARK CONFIG
# ============================================================================

SEARCH_W = 1000
SEARCH_H = 1000

REFERENCE_W = 100
REFERENCE_H = 100

TEMPLATE_W = 10
TEMPLATE_H = 10

PIXELS_PER_UNIT = 5.0

TARGET_X = 75.25
TARGET_Y = 113.75

TARGET_SEARCH_X = TARGET_X * PIXELS_PER_UNIT
TARGET_SEARCH_Y = TARGET_Y * PIXELS_PER_UNIT

# Template top-left is target center minus half template size.
GT_X = TARGET_SEARCH_X - TEMPLATE_W / 2.0
GT_Y = TARGET_SEARCH_Y - TEMPLATE_H / 2.0

SEEDS = list(
    range(
        20260875,
        20260905,
    )
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

K_VALUES = [
    10,
    25,
    50,
    100,
    250,
    500,
]

TOLERANCE_PX = 5.0


# ============================================================================
# UTILITY
# ============================================================================

def normalize01(x):
    x = np.asarray(x, dtype=np.float32)

    mn = float(np.min(x))
    mx = float(np.max(x))

    if mx - mn < 1e-12:
        return np.zeros_like(x)

    return (x - mn) / (mx - mn)


def robust_normalize(x):
    """
    Robust normalization for combining heterogeneous score maps.
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    p01 = np.percentile(x, 1)
    p99 = np.percentile(x, 99)

    if p99 - p01 < 1e-12:
        return normalize01(x)

    y = (
        x - p01
    ) / (
        p99 - p01
    )

    return np.clip(
        y,
        0.0,
        1.0,
    )


def rank_normalize(x):
    """
    Convert a score map into [0,1] rank scores.

    Higher value = better candidate.
    """

    flat = np.asarray(
        x,
        dtype=np.float32,
    ).reshape(-1)

    order = np.argsort(
        np.argsort(flat)
    )

    ranked = (
        order.astype(np.float32)
        / max(
            1,
            len(flat) - 1,
        )
    )

    return ranked.reshape(
        x.shape
    )


def extract_topk(score_map, K):
    """
    Extract top-K spatial candidates.

    Simple non-maximum suppression prevents selecting many adjacent
    pixels from the same local peak.
    """

    h, w = score_map.shape

    flat = score_map.reshape(-1)

    # Get more than K raw candidates so spatial suppression has room.
    raw_k = min(
        flat.size,
        max(
            K * 20,
            100,
        ),
    )

    idx = np.argpartition(
        flat,
        -raw_k,
    )[-raw_k:]

    idx = idx[
        np.argsort(
            flat[idx]
        )[::-1]
    ]

    selected = []

    # Candidate suppression radius.
    radius = max(
        2,
        TEMPLATE_W // 2,
    )

    radius2 = radius * radius

    for i in idx:

        y, x = np.unravel_index(
            int(i),
            score_map.shape,
        )

        good = True

        for sx, sy in selected:

            dx = x - sx
            dy = y - sy

            if (
                dx * dx
                +
                dy * dy
                <= radius2
            ):
                good = False
                break

        if good:

            selected.append(
                (
                    int(x),
                    int(y),
                )
            )

            if len(selected) >= K:
                break

    return selected


def candidate_error(
    candidates,
    gt_x,
    gt_y,
):
    if not candidates:
        return np.inf, -1

    errors = []

    for x, y in candidates:

        dx = (
            float(x)
            - gt_x
        )

        dy = (
            float(y)
            - gt_y
        )

        errors.append(
            np.sqrt(
                dx * dx
                +
                dy * dy
            )
        )

    errors = np.asarray(
        errors
    )

    best_idx = int(
        np.argmin(errors)
    )

    return (
        float(errors[best_idx]),
        best_idx + 1,
    )


def compute_error_from_best_map(
    score_map,
    gt_x,
    gt_y,
):
    y, x = np.unravel_index(
        np.argmax(score_map),
        score_map.shape,
    )

    error = np.sqrt(
        (
            float(x)
            - gt_x
        ) ** 2
        +
        (
            float(y)
            - gt_y
        ) ** 2
    )

    return (
        float(error),
        int(x),
        int(y),
    )


# ============================================================================
# GRADIENT REPRESENTATIONS
# ============================================================================

def gradient_fields(image):
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

    magnitude = np.sqrt(
        gx * gx
        +
        gy * gy
    )

    orientation = np.arctan2(
        gy,
        gx,
    )

    return (
        gx,
        gy,
        magnitude,
        orientation,
    )


def gradient_magnitude_match(
    search,
    template,
):
    """
    NCC on gradient magnitude.
    """

    _, _, search_mag, _ = (
        gradient_fields(
            search
        )
    )

    _, _, template_mag, _ = (
        gradient_fields(
            template
        )
    )

    response = cv2.matchTemplate(
        search_mag,
        template_mag,
        cv2.TM_CCOEFF_NORMED,
    )

    return response


def gradient_orientation_match(
    search,
    template,
):
    """
    Orientation consistency.

    Instead of matching raw orientation angles directly, represent
    orientation using cos(theta) and sin(theta), then correlate the
    two vector components.

    This avoids the artificial discontinuity between -pi and +pi.
    """

    (
        search_gx,
        search_gy,
        search_mag,
        _,
    ) = gradient_fields(
        search
    )

    (
        template_gx,
        template_gy,
        template_mag,
        _,
    ) = gradient_fields(
        template
    )

    eps = 1e-6

    search_cos = (
        search_gx
        /
        (
            search_mag
            +
            eps
        )
    )

    search_sin = (
        search_gy
        /
        (
            search_mag
            +
            eps
        )
    )

    template_cos = (
        template_gx
        /
        (
            template_mag
            +
            eps
        )
    )

    template_sin = (
        template_gy
        /
        (
            template_mag
            +
            eps
        )
    )

    response_cos = cv2.matchTemplate(
        search_cos,
        template_cos,
        cv2.TM_CCOEFF_NORMED,
    )

    response_sin = cv2.matchTemplate(
        search_sin,
        template_sin,
        cv2.TM_CCOEFF_NORMED,
    )

    response = (
        response_cos
        +
        response_sin
    ) / 2.0

    return response


# ============================================================================
# HYBRID
# ============================================================================

def hybrid_score(
    ncc,
    grad_mag,
    grad_orientation,
):
    """
    Rank-based fusion.

    Each representation is converted to percentile/rank space before
    fusion so that one representation cannot dominate simply because
    its numerical score has a different scale.

    Equal weighting is deliberately used for the benchmark.
    This is not the final learned fusion model.
    """

    n = rank_normalize(
        ncc
    )

    g = rank_normalize(
        grad_mag
    )

    o = rank_normalize(
        grad_orientation
    )

    return (
        n
        +
        g
        +
        o
    ) / 3.0


# ============================================================================
# CANONICAL OBSERVATION
# ============================================================================

def load_scene(
    scene_type,
    seed,
):
    """
    Load observations strictly through canonical_renderer.py.
    """

    search = cr.render_search(
        TARGET_X,
        TARGET_Y,
        scene_type,
        seed,
    )

    reference = cr.render_reference(
        TARGET_X,
        TARGET_Y,
        scene_type,
        seed,
    )

    template = cr.create_ps02_template(
        reference
    )

    search = np.asarray(
        search
    )

    reference = np.asarray(
        reference
    )

    template = np.asarray(
        template
    )

    return (
        search,
        reference,
        template,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    start = time.time()

    print("=" * 76)
    print("MICRONYX STEP 31")
    print("GRADIENT + NCC CANDIDATE GENERATION BENCHMARK")
    print("=" * 76)

    print()
    print("Canonical renderer:")
    print(
        Path(
            cr.__file__
        ).resolve()
    )

    print()
    print("Target physical:")
    print(
        f"  ({TARGET_X:.4f}, {TARGET_Y:.4f})"
    )

    print()
    print("Target search:")
    print(
        f"  ({TARGET_SEARCH_X:.4f}, "
        f"{TARGET_SEARCH_Y:.4f})"
    )

    print()
    print("Evaluation GT template top-left:")
    print(
        f"  ({GT_X:.4f}, {GT_Y:.4f})"
    )

    print()
    print("Search:")
    print(
        f"  {SEARCH_W} × {SEARCH_H}"
    )

    print()
    print("Reference:")
    print(
        f"  {REFERENCE_W} × {REFERENCE_H}"
    )

    print()
    print("Template:")
    print(
        f"  {TEMPLATE_W} × {TEMPLATE_H}"
    )

    print()
    print("Representations:")
    print("  ncc")
    print("  gradient_magnitude")
    print("  gradient_orientation")
    print("  gradient_ncc_hybrid")

    print()
    print("K values:")
    print(
        K_VALUES
    )

    print()
    print("Candidate tolerance:")
    print(
        f"  {TOLERANCE_PX} px"
    )

    # ---------------------------------------------------------------------
    # RESULT STORAGE
    # ---------------------------------------------------------------------

    rows = []

    methods = [
        "ncc",
        "gradient_magnitude",
        "gradient_orientation",
        "gradient_ncc_hybrid",
    ]

    # ---------------------------------------------------------------------
    # SCENES
    # ---------------------------------------------------------------------

    scene_number = 0

    for scene_type in SCENE_TYPES:

        print()
        print("=" * 76)
        print(
            f"SCENE TYPE: "
            f"{scene_type.upper()}"
        )
        print("=" * 76)

        for seed in SEEDS:

            scene_number += 1

            print(
                f"Scene "
                f"{scene_number:02d}/60"
            )

            # -------------------------------------------------------------
            # Canonical observations
            # -------------------------------------------------------------

            (
                search,
                reference,
                template,
            ) = load_scene(
                scene_type,
                seed,
            )

            # Ensure grayscale single-channel.
            if search.ndim == 3:
                search_gray = cv2.cvtColor(
                    search,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                search_gray = search

            if template.ndim == 3:
                template_gray = cv2.cvtColor(
                    template,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                template_gray = template

            search_gray = (
                search_gray
                .astype(np.float32)
            )

            template_gray = (
                template_gray
                .astype(np.float32)
            )

            # -------------------------------------------------------------
            # Candidate maps
            # -------------------------------------------------------------

            ncc = cv2.matchTemplate(
                search_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED,
            )

            grad_mag = (
                gradient_magnitude_match(
                    search_gray,
                    template_gray,
                )
            )

            grad_orientation = (
                gradient_orientation_match(
                    search_gray,
                    template_gray,
                )
            )

            hybrid = hybrid_score(
                ncc,
                grad_mag,
                grad_orientation,
            )

            maps = {
                "ncc": ncc,
                "gradient_magnitude": grad_mag,
                "gradient_orientation": grad_orientation,
                "gradient_ncc_hybrid": hybrid,
            }

            # -------------------------------------------------------------
            # Evaluate every method
            # -------------------------------------------------------------

            for method in methods:

                score_map = maps[
                    method
                ]

                # Best raw match.
                (
                    top1_error,
                    top1_x,
                    top1_y,
                ) = compute_error_from_best_map(
                    score_map,
                    GT_X,
                    GT_Y,
                )

                candidates_by_k = {}

                for K in K_VALUES:

                    candidates = (
                        extract_topk(
                            score_map,
                            K,
                        )
                    )

                    (
                        best_error,
                        gt_rank,
                    ) = candidate_error(
                        candidates,
                        GT_X,
                        GT_Y,
                    )

                    recall = (
                        1.0
                        if best_error
                        <= TOLERANCE_PX
                        else 0.0
                    )

                    candidates_by_k[K] = (
                        best_error,
                        gt_rank,
                        recall,
                    )

                    rows.append(
                        {
                            "scene_type":
                                scene_type,

                            "seed":
                                seed,

                            "method":
                                method,

                            "K":
                                K,

                            "best_error_px":
                                best_error,

                            "gt_rank":
                                gt_rank,

                            "recall":
                                recall,

                            "top1_error_px":
                                top1_error,

                            "top1_x":
                                top1_x,

                            "top1_y":
                                top1_y,
                        }
                    )

    # ---------------------------------------------------------------------
    # DATAFRAME
    # ---------------------------------------------------------------------

    results = pd.DataFrame(
        rows
    )

    csv_path = (
        OUT_DIR
        / "gradient_ncc_candidate_results.csv"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------------

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    summary_rows = []

    for scene_type in SCENE_TYPES:

        print()
        print(
            scene_type.upper()
        )
        print("-" * 76)

        subset = results[
            results["scene_type"]
            == scene_type
        ]

        for method in methods:

            method_subset = subset[
                subset["method"]
                == method
            ]

            recall_values = []

            median_errors = []
            top1_errors = []
            ranks = []

            for K in K_VALUES:

                ksub = method_subset[
                    method_subset["K"]
                    == K
                ]

                recall_pct = (
                    ksub[
                        "recall"
                    ].mean()
                    * 100.0
                )

                med_error = (
                    ksub[
                        "best_error_px"
                    ].median()
                )

                p95_error = (
                    ksub[
                        "best_error_px"
                    ].quantile(
                        0.95
                    )
                )

                mean_rank = (
                    ksub[
                        "gt_rank"
                    ]
                    .replace(
                        -1,
                        np.nan,
                    )
                    .mean()
                )

                print(
                    f"{method:24s} "
                    f"K={K:<4d} "
                    f"Recall@5px="
                    f"{recall_pct:6.2f}% "
                    f"MedErr="
                    f"{med_error:8.3f}px "
                    f"P95="
                    f"{p95_error:8.3f}px "
                    f"MeanRank="
                    f"{mean_rank:8.2f}"
                )

                summary_rows.append(
                    {
                        "scene_type":
                            scene_type,

                        "method":
                            method,

                        "K":
                            K,

                        "recall_percent":
                            recall_pct,

                        "median_error_px":
                            float(
                                med_error
                            ),

                        "p95_error_px":
                            float(
                                p95_error
                            ),

                        "mean_gt_rank":
                            float(
                                mean_rank
                            )
                            if not np.isnan(
                                mean_rank
                            )
                            else None,
                    }
                )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        OUT_DIR
        / "gradient_ncc_candidate_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    # ---------------------------------------------------------------------
    # GLOBAL COMPARISON AT PRIMARY K
    # ---------------------------------------------------------------------

    print()
    print("=" * 76)
    print("PRIMARY COMPARISON — K=250")
    print("=" * 76)

    primary = results[
        results["K"]
        == 250
    ]

    comparison = (
        primary
        .groupby(
            [
                "scene_type",
                "method",
            ]
        )
        .agg(
            recall_percent=(
                "recall",
                lambda x:
                    float(
                        x.mean()
                        * 100
                    ),
            ),

            median_error_px=(
                "best_error_px",
                "median",
            ),

            p95_error_px=(
                "best_error_px",
                lambda x:
                    float(
                        x.quantile(
                            0.95
                        )
                    ),
            ),

            mean_rank=(
                "gt_rank",
                lambda x:
                    float(
                        x.replace(
                            -1,
                            np.nan,
                        ).mean()
                    ),
            ),
        )
        .reset_index()
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------------
    # HYBRID VS INDIVIDUAL
    # ---------------------------------------------------------------------

    print()
    print("=" * 76)
    print("HYBRID GAIN ANALYSIS")
    print("=" * 76)

    for scene_type in SCENE_TYPES:

        sub = comparison[
            comparison["scene_type"]
            == scene_type
        ]

        print()
        print(
            scene_type.upper()
        )

        for method in methods:

            row = sub[
                sub["method"]
                == method
            ]

            if len(row) == 0:
                continue

            print(
                f"{method:24s} "
                f"{row['recall_percent'].iloc[0]:6.2f}%"
            )

    # ---------------------------------------------------------------------
    # METHODOLOGY SUMMARY
    # ---------------------------------------------------------------------

    runtime = (
        time.time()
        - start
    )

    methodology = {
        "step": 31,
        "name":
            "Gradient + NCC Candidate Generation Benchmark",

        "canonical_renderer":
            str(
                Path(
                    cr.__file__
                ).resolve()
            ),

        "canonical_observation":
            True,

        "target_fingerprint":
            False,

        "new_ground_truth":
            False,

        "alternate_renderer":
            False,

        "manual_generator_selection":
            False,

        "candidate_tolerance_px":
            TOLERANCE_PX,

        "primary_k":
            250,

        "methods":
            methods,

        "K_values":
            K_VALUES,

        "scene_count":
            60,

        "scene_types":
            SCENE_TYPES,

        "runtime_seconds":
            runtime,
    }

    json_path = (
        OUT_DIR
        / "gradient_ncc_candidate_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            methodology,
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print("METHODOLOGY")
    print("=" * 76)

    print(
        "Canonical renderer:              YES"
    )

    print(
        "Canonical observations:          YES"
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
        "Manual representation selection: NO"
    )

    print(
        "Candidate tolerance:             "
        f"{TOLERANCE_PX} px"
    )

    print(
        "Primary K:                        250"
    )

    print(
        "Runtime:                          "
        f"{runtime:.2f} seconds"
    )

    print()
    print("=" * 76)
    print("SAVED")
    print("=" * 76)

    print(csv_path)
    print(summary_csv)
    print(json_path)

    print()
    print("=" * 76)
    print("STEP 31 COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()