"""
============================================================================
MICRONYX STEP 32A
CORRECTED MULTI-SCALE PHYSICAL-CONTEXT CANDIDATE BENCHMARK
============================================================================

Purpose
-------
Benchmark REAL physical context sizes extracted from the canonical reference
observation.

IMPORTANT:
- No resizing of the 10x10 template.
- No alternate renderer.
- No new ground truth.
- No target fingerprint.
- Canonical renderer only.
- Context sizes are actual spatial crops from the canonical reference.

Context sizes:
    10 px
    20 px
    40 px

Representations:
    NCC
    Gradient magnitude
    NCC + Gradient fusion

Primary metric:
    Recall@K within 5 px of canonical GT center.
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
OUT = ROOT / "validation" / "v02" / "multiscale_candidate_v2"

OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS))

import canonical_renderer as cr


# ============================================================================
# CONFIGURATION
# ============================================================================

SEARCH_W = 1000
SEARCH_H = 1000

REFERENCE_W = 100
REFERENCE_H = 100

PIXELS_PER_UNIT = 5.0

TARGET_X = 75.25
TARGET_Y = 113.75

TARGET_SEARCH_X = TARGET_X * PIXELS_PER_UNIT
TARGET_SEARCH_Y = TARGET_Y * PIXELS_PER_UNIT

CONTEXT_SIZES = [10, 20, 40]

K_VALUES = [10, 25, 50, 100, 250, 500]

TOLERANCE_PX = 5.0

SCENE_COUNT = 60
PERIODIC_COUNT = 30
QUASIPERIODIC_COUNT = 30

SEED_START = 20260845


# ============================================================================
# UTILITIES
# ============================================================================

def to_gray(image):
    if image is None:
        raise RuntimeError("Renderer returned None.")

    image = np.asarray(image)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image.astype(np.float32)

    if image.max() > 1.0:
        image /= 255.0

    return image


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)

    mn = np.min(x)
    mx = np.max(x)

    if mx - mn < 1e-12:
        return np.zeros_like(x)

    return (x - mn) / (mx - mn)


def gradient_magnitude(image):
    image = image.astype(np.float32)

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

    return np.sqrt(gx * gx + gy * gy)


def crop_center(image, width, height):
    h, w = image.shape[:2]

    cx = w // 2
    cy = h // 2

    x0 = int(round(cx - width / 2))
    y0 = int(round(cy - height / 2))

    x1 = x0 + width
    y1 = y0 + height

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise RuntimeError(
            f"Cannot extract {width}x{height} center crop "
            f"from {w}x{h} reference."
        )

    return image[y0:y1, x0:x1].copy()


def match_ncc(search, template):
    result = cv2.matchTemplate(
        search.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED
    )

    return result


def match_gradient(search, template):
    search_g = gradient_magnitude(search)
    template_g = gradient_magnitude(template)

    result = cv2.matchTemplate(
        search_g.astype(np.float32),
        template_g.astype(np.float32),
        cv2.TM_CCOEFF_NORMED
    )

    return result


def result_to_candidates(score_map, template_shape):
    th, tw = template_shape

    ys, xs = np.indices(score_map.shape)

    centers_x = xs.astype(np.float64) + tw / 2.0
    centers_y = ys.astype(np.float64) + th / 2.0

    scores = score_map.astype(np.float64)

    flat_scores = scores.ravel()
    flat_x = centers_x.ravel()
    flat_y = centers_y.ravel()

    order = np.argsort(flat_scores)[::-1]

    return (
        flat_scores[order],
        flat_x[order],
        flat_y[order]
    )


def suppress_nearby(scores, xs, ys, min_distance):
    """
    Greedy spatial NMS.

    This prevents a single strong match from filling the entire
    candidate pool with nearly identical neighboring locations.
    """

    selected_scores = []
    selected_x = []
    selected_y = []

    min_d2 = float(min_distance) ** 2

    for score, x, y in zip(scores, xs, ys):

        if len(selected_x) == 0:
            selected_scores.append(score)
            selected_x.append(x)
            selected_y.append(y)
            continue

        dx = np.asarray(selected_x) - x
        dy = np.asarray(selected_y) - y

        d2 = dx * dx + dy * dy

        if np.all(d2 >= min_d2):
            selected_scores.append(score)
            selected_x.append(x)
            selected_y.append(y)

    return (
        np.asarray(selected_scores),
        np.asarray(selected_x),
        np.asarray(selected_y)
    )


def recall_at_k(xs, ys, k):
    if len(xs) == 0:
        return 0.0

    k = min(k, len(xs))

    dx = xs[:k] - TARGET_SEARCH_X
    dy = ys[:k] - TARGET_SEARCH_Y

    distances = np.sqrt(dx * dx + dy * dy)

    return float(np.any(distances <= TOLERANCE_PX))


def gt_rank(xs, ys):
    if len(xs) == 0:
        return -1

    dx = xs - TARGET_SEARCH_X
    dy = ys - TARGET_SEARCH_Y

    distances = np.sqrt(dx * dx + dy * dy)

    valid = np.where(distances <= TOLERANCE_PX)[0]

    if len(valid) == 0:
        return -1

    return int(valid[0] + 1)


def top_error(xs, ys):
    if len(xs) == 0:
        return float("inf")

    distances = np.sqrt(
        (xs - TARGET_SEARCH_X) ** 2 +
        (ys - TARGET_SEARCH_Y) ** 2
    )

    return float(distances[0])


# ============================================================================
# CANONICAL SCENE
# ============================================================================

def render_scene(scene_type, seed):

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

    search = to_gray(search)
    reference = to_gray(reference)

    if search.shape != (SEARCH_H, SEARCH_W):
        raise RuntimeError(
            f"Unexpected search shape: {search.shape}"
        )

    if reference.shape != (REFERENCE_H, REFERENCE_W):
        raise RuntimeError(
            f"Unexpected reference shape: {reference.shape}"
        )

    return search, reference


# ============================================================================
# ACTUAL PHYSICAL CONTEXT
# ============================================================================

def build_context_templates(reference):

    """
    IMPORTANT:

    These are REAL crops from the canonical reference observation.

    10x10 = local context
    20x20 = larger physical context
    40x40 = larger physical context

    There is NO interpolation / resizing.
    """

    templates = {}

    for size in CONTEXT_SIZES:

        template = crop_center(
            reference,
            size,
            size
        )

        templates[size] = template

    return templates


# ============================================================================
# SCORE STRATEGIES
# ============================================================================

def generate_candidates(search, template, method):

    if method == "ncc":

        score = match_ncc(
            search,
            template
        )

    elif method == "gradient":

        score = match_gradient(
            search,
            template
        )

    elif method == "fusion":

        ncc = match_ncc(
            search,
            template
        )

        grad = match_gradient(
            search,
            template
        )

        ncc = normalize01(ncc)
        grad = normalize01(grad)

        score = 0.5 * ncc + 0.5 * grad

    else:
        raise ValueError(f"Unknown method: {method}")

    scores, xs, ys = result_to_candidates(
        score,
        template.shape
    )

    # Spatial NMS.
    #
    # This is intentionally modest. We don't want the candidate pool
    # dominated by adjacent pixels representing the same local maximum.
    scores, xs, ys = suppress_nearby(
        scores,
        xs,
        ys,
        min_distance=2.0
    )

    return scores, xs, ys


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    print("=" * 76)
    print("MICRONYX STEP 32A")
    print("CORRECTED MULTI-SCALE PHYSICAL-CONTEXT CANDIDATE BENCHMARK")
    print("=" * 76)

    print()
    print("Canonical renderer:")
    print(cr.__file__)

    print()
    print("Target physical:")
    print(f"  ({TARGET_X:.4f}, {TARGET_Y:.4f})")

    print()
    print("Target search CENTER:")
    print(
        f"  ({TARGET_SEARCH_X:.4f}, "
        f"{TARGET_SEARCH_Y:.4f})"
    )

    print()
    print("Reference:")
    print(
        f"  {REFERENCE_W} × {REFERENCE_H}"
    )

    print()
    print("Actual physical contexts:")
    for size in CONTEXT_SIZES:
        print(
            f"  {size}px × {size}px"
            f"  = {size / PIXELS_PER_UNIT:.2f}"
            f" physical units"
        )

    print()
    print("IMPORTANT:")
    print("  Contexts are canonical crops.")
    print("  No template resizing.")
    print("  No target fingerprint.")
    print("  No new ground truth.")
    print("  No alternate renderer.")

    methods = [
        "ncc",
        "gradient",
        "fusion"
    ]

    rows = []

    scene_records = []

    for scene_index in range(SCENE_COUNT):

        if scene_index < PERIODIC_COUNT:
            scene_type = "periodic"
        else:
            scene_type = "quasiperiodic"

        seed = SEED_START + scene_index

        print()
        print("=" * 76)
        print(
            f"SCENE {scene_index + 1:02d}/{SCENE_COUNT} "
            f"{scene_type.upper()}"
        )
        print("=" * 76)

        search, reference = render_scene(
            scene_type,
            seed
        )

        templates = build_context_templates(
            reference
        )

        for context_size in CONTEXT_SIZES:

            template = templates[context_size]

            for method in methods:

                scores, xs, ys = generate_candidates(
                    search,
                    template,
                    method
                )

                rank = gt_rank(xs, ys)
                error = top_error(xs, ys)

                for k in K_VALUES:

                    recall = recall_at_k(
                        xs,
                        ys,
                        k
                    )

                    rows.append({
                        "scene_type": scene_type,
                        "seed": seed,
                        "scene_index": scene_index + 1,
                        "context_px": context_size,
                        "context_physical_units":
                            context_size / PIXELS_PER_UNIT,
                        "method": method,
                        "K": k,
                        "recall": recall,
                        "recall_percent": recall * 100.0,
                        "top1_error_px": error,
                        "gt_rank": rank
                    })

        scene_records.append({
            "scene_type": scene_type,
            "seed": seed
        })

    df = pd.DataFrame(rows)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    for scene_type in ["periodic", "quasiperiodic"]:

        print()
        print(scene_type.upper())
        print("-" * 76)

        subset = df[
            df["scene_type"] == scene_type
        ]

        for context_size in CONTEXT_SIZES:

            for method in methods:

                vals = subset[
                    (subset["context_px"] == context_size)
                    &
                    (subset["method"] == method)
                ]

                row = vals[
                    vals["K"] == 250
                ]

                if len(row) == 0:
                    continue

                recall = row["recall_percent"].mean()

                print(
                    f"{method:<10}"
                    f"context={context_size:>2}px "
                    f"Recall@250={recall:7.2f}%"
                )

    # ========================================================================
    # BEST STRATEGIES
    # ========================================================================

    print()
    print("=" * 76)
    print("BEST STRATEGY AT K=250")
    print("=" * 76)

    best_rows = []

    for scene_type in ["periodic", "quasiperiodic"]:

        subset = df[
            (df["scene_type"] == scene_type)
            &
            (df["K"] == 250)
        ]

        grouped = (
            subset
            .groupby(
                ["context_px", "method"],
                as_index=False
            )["recall_percent"]
            .mean()
        )

        best = grouped.sort_values(
            "recall_percent",
            ascending=False
        ).iloc[0]

        print()
        print(scene_type.upper())
        print(
            f"Best context: "
            f"{int(best['context_px'])} px"
        )
        print(
            f"Best method: "
            f"{best['method']}"
        )
        print(
            f"Recall@250: "
            f"{best['recall_percent']:.2f}%"
        )

        best_rows.append({
            "scene_type": scene_type,
            "context_px": int(best["context_px"]),
            "method": best["method"],
            "recall_percent":
                float(best["recall_percent"])
        })

    # ========================================================================
    # METHODOLOGY
    # ========================================================================

    runtime = time.time() - start_time

    summary = {
        "step": "32A",
        "name":
            "Corrected Multi-Scale Physical-Context "
            "Candidate Benchmark",

        "canonical_renderer": True,
        "canonical_observations": True,

        "context_sizes_px": CONTEXT_SIZES,

        "pixels_per_physical_unit":
            PIXELS_PER_UNIT,

        "actual_physical_context": True,

        "template_resizing": False,

        "target_fingerprint": False,
        "new_ground_truth": False,
        "alternate_renderer": False,

        "methods": methods,

        "K_values": K_VALUES,

        "candidate_tolerance_px":
            TOLERANCE_PX,

        "primary_K": 250,

        "scene_count": SCENE_COUNT,

        "best_strategies":
            best_rows,

        "runtime_seconds": runtime
    }

    # ========================================================================
    # SAVE
    # ========================================================================

    result_csv = (
        OUT /
        "multiscale_candidate_v2_results.csv"
    )

    summary_csv = (
        OUT /
        "multiscale_candidate_v2_summary.csv"
    )

    summary_json = (
        OUT /
        "multiscale_candidate_v2_summary.json"
    )

    df.to_csv(
        result_csv,
        index=False
    )

    summary_df = pd.DataFrame(
        best_rows
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

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
        "Actual physical context crops:   YES"
    )

    print(
        "Template resizing:               NO"
    )

    print(
        "Target fingerprint:              NO"
    )

    print(
        "New ground truth:                 NO"
    )

    print(
        "Alternate renderer:               NO"
    )

    print(
        f"Candidate tolerance:              "
        f"{TOLERANCE_PX:.1f} px"
    )

    print(
        "Primary K:                         250"
    )

    print(
        f"Runtime:                           "
        f"{runtime:.2f} seconds"
    )

    print()
    print("=" * 76)
    print("SAVED")
    print("=" * 76)

    print(result_csv)
    print(summary_csv)
    print(summary_json)

    print()
    print("=" * 76)
    print("STEP 32A COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()