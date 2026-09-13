"""
============================================================================
MICRONYX STEP 25 — CANONICAL ADAPTIVE CANDIDATE GENERATION
============================================================================

Purpose
-------
Evaluate whether multiple physically different candidate-generation
representations improve target candidate recall.

Canonical observation model
----------------------------

Continuous physical scene
        ↓
Canonical renderer
        ↓
Sensor observation
        ↓
Search image + reference image
        ↓
Canonical PS02 template
        ↓
Candidate generators
        ↓
Candidate union
        ↓
Recall@K

IMPORTANT
---------
This script does NOT:
    - create another renderer
    - create another physical scene
    - alter the target
    - inject a target fingerprint
    - train a verifier
    - perform final localization

This is candidate GENERATION research.

The canonical renderer used is:

    scripts/canonical_renderer.py

Known canonical API:

    periodic_scene(x, y)
    quasiperiodic_scene(x, y, seed)
    continuous_scene(x, y)
    generate_observation(tx, ty, scene_type, seed)
    render_search(tx, ty, scene_type, seed)
    render_reference(tx, ty, scene_type, seed)
    render_sensor(...)
    create_ps02_template(reference)
    target_mask(x, y, tx, ty)

============================================================================
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = ROOT / "scripts"

OUTPUT_DIR = (
    ROOT
    / "validation"
    / "v02"
    / "adaptive_candidate_generation_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "adaptive_candidate_generation_v2_results.csv"
)


# ============================================================================
# IMPORT CANONICAL RENDERER
# ============================================================================

sys.path.insert(
    0,
    str(SCRIPTS_DIR),
)

import canonical_renderer as canonical


# ============================================================================
# CONFIGURATION
# ============================================================================

SEARCH_WIDTH = 1000
SEARCH_HEIGHT = 1000

REFERENCE_WIDTH = 100
REFERENCE_HEIGHT = 100

TEMPLATE_WIDTH = 10
TEMPLATE_HEIGHT = 10

TARGET_PHYSICAL_X = 75.25
TARGET_PHYSICAL_Y = 113.75

SEARCH_PIXELS_PER_UNIT = 5.0

TARGET_SEARCH_X = (
    TARGET_PHYSICAL_X
    * SEARCH_PIXELS_PER_UNIT
)

TARGET_SEARCH_Y = (
    TARGET_PHYSICAL_Y
    * SEARCH_PIXELS_PER_UNIT
)


# ---------------------------------------------------------------------------
# Benchmark seeds
# ---------------------------------------------------------------------------

SEEDS = list(
    range(
        20260875,
        20260905,
    )
)


# ---------------------------------------------------------------------------
# Candidate K values
# ---------------------------------------------------------------------------

K_VALUES = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]


# ---------------------------------------------------------------------------
# Generator bank
# ---------------------------------------------------------------------------

GENERATORS = [
    "ncc",
    "dog",
    "gradient",
    "edge",
    "frequency",
]


# ---------------------------------------------------------------------------
# Multi-generator combinations
# ---------------------------------------------------------------------------

COMBINATIONS = {

    "ncc_dog": [
        "ncc",
        "dog",
    ],

    "dog_gradient": [
        "dog",
        "gradient",
    ],

    "dog_frequency": [
        "dog",
        "frequency",
    ],

    "dog_edge": [
        "dog",
        "edge",
    ],

    "dog_gradient_frequency": [
        "dog",
        "gradient",
        "frequency",
    ],

    "dog_gradient_edge": [
        "dog",
        "gradient",
        "edge",
    ],

    "all": [
        "ncc",
        "dog",
        "gradient",
        "edge",
        "frequency",
    ],
}


# ============================================================================
# DISPLAY
# ============================================================================

print()
print("=" * 76)
print("MICRONYX STEP 25")
print("CANONICAL ADAPTIVE CANDIDATE GENERATION")
print("=" * 76)
print()

print("Canonical renderer:")
print(
    canonical.__file__
)

print()

print("Canonical APIs:")
print(
    "  periodic_scene(x, y)"
)

print(
    "  quasiperiodic_scene(x, y, seed)"
)

print(
    "  continuous_scene(x, y)"
)

print(
    "  render_search(tx, ty, scene_type, seed)"
)

print(
    "  render_reference(tx, ty, scene_type, seed)"
)

print(
    "  create_ps02_template(reference)"
)

print()

print("Target physical:")
print(
    f"  ({TARGET_PHYSICAL_X:.4f}, "
    f"{TARGET_PHYSICAL_Y:.4f})"
)

print()

print("Target search:")
print(
    f"  ({TARGET_SEARCH_X:.4f}, "
    f"{TARGET_SEARCH_Y:.4f})"
)

print()

print("Search:")
print(
    f"  {SEARCH_WIDTH} × {SEARCH_HEIGHT}"
)

print("Reference:")
print(
    f"  {REFERENCE_WIDTH} × {REFERENCE_HEIGHT}"
)

print("Template:")
print(
    f"  {TEMPLATE_WIDTH} × {TEMPLATE_HEIGHT}"
)

print()

print("Generator bank:")
for generator in GENERATORS:
    print(
        f"  {generator}"
    )

print()


# ============================================================================
# NUMERICAL HELPERS
# ============================================================================

def as_float32(
    image,
):
    """
    Convert image to float32 grayscale.
    """

    arr = np.asarray(
        image
    )

    if arr.ndim == 3:

        if arr.shape[-1] == 3:

            arr = cv2.cvtColor(
                arr,
                cv2.COLOR_BGR2GRAY,
            )

        elif arr.shape[-1] == 4:

            arr = cv2.cvtColor(
                arr,
                cv2.COLOR_BGRA2GRAY,
            )

        else:

            arr = np.mean(
                arr,
                axis=-1,
            )

    return arr.astype(
        np.float32
    )


def safe_normalize(
    image,
):
    """
    Normalize an image to approximately [-1, 1].
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    mean = float(
        np.mean(image)
    )

    std = float(
        np.std(image)
    )

    if std < 1e-8:

        return np.zeros_like(
            image
        )

    return (
        image - mean
    ) / std


# ============================================================================
# IMAGE REPRESENTATIONS
# ============================================================================

def representation_raw(
    image,
):
    return as_float32(
        image
    )


def representation_dog(
    image,
):
    """
    Difference of Gaussians.

    DOG(x) = G_sigma1(x) - G_sigma2(x)
    """

    image = as_float32(
        image
    )

    blur_small = cv2.GaussianBlur(
        image,
        (0, 0),
        0.8,
    )

    blur_large = cv2.GaussianBlur(
        image,
        (0, 0),
        2.0,
    )

    return (
        blur_small
        - blur_large
    )


def representation_gradient(
    image,
):
    """
    Gradient magnitude:

        sqrt(Gx^2 + Gy^2)
    """

    image = as_float32(
        image
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

    return magnitude


def representation_edge(
    image,
):
    """
    Canny edge representation.
    """

    image = as_float32(
        image
    )

    minimum = float(
        np.min(image)
    )

    maximum = float(
        np.max(image)
    )

    if maximum - minimum > 1e-8:

        normalized = (
            image - minimum
        ) / (
            maximum - minimum
        )

        normalized *= 255.0

    else:

        normalized = np.zeros_like(
            image
        )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(
        np.uint8
    )

    edges = cv2.Canny(
        normalized,
        30,
        100,
    )

    return edges.astype(
        np.float32
    )


def representation_frequency(
    image,
):
    """
    Frequency-domain residual representation.

    Low-frequency/DC components are suppressed.
    """

    image = as_float32(
        image
    )

    height, width = image.shape

    wy = np.hanning(
        height
    )

    wx = np.hanning(
        width
    )

    window = np.outer(
        wy,
        wx,
    )

    centered = (
        image
        - np.mean(image)
    )

    spectrum = np.fft.fft2(
        centered * window
    )

    spectrum = np.fft.fftshift(
        spectrum
    )

    cy = height // 2
    cx = width // 2

    yy, xx = np.ogrid[
        :height,
        :width,
    ]

    radius = np.sqrt(
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    cutoff = max(
        2.0,
        0.03 * min(
            height,
            width,
        ),
    )

    high_frequency_mask = (
        radius >= cutoff
    )

    filtered = (
        spectrum
        * high_frequency_mask
    )

    result = np.fft.ifft2(
        np.fft.ifftshift(
            filtered
        )
    )

    return np.real(
        result
    ).astype(
        np.float32
    )


# ============================================================================
# REPRESENTATION DISPATCH
# ============================================================================

def build_representation(
    image,
    name,
):
    if name == "ncc":
        return representation_raw(
            image
        )

    if name == "dog":
        return representation_dog(
            image
        )

    if name == "gradient":
        return representation_gradient(
            image
        )

    if name == "edge":
        return representation_edge(
            image
        )

    if name == "frequency":
        return representation_frequency(
            image
        )

    raise ValueError(
        f"Unknown generator: {name}"
    )


# ============================================================================
# TEMPLATE MATCHING
# ============================================================================

def compute_response(
    search,
    template,
    generator,
):
    """
    Compute dense response map.
    """

    search_rep = build_representation(
        search,
        generator,
    )

    template_rep = build_representation(
        template,
        generator,
    )

    search_rep = np.asarray(
        search_rep,
        dtype=np.float32,
    )

    template_rep = np.asarray(
        template_rep,
        dtype=np.float32,
    )

    # OpenCV requires the template to be
    # smaller than the search image.
    if (
        template_rep.shape[0]
        > search_rep.shape[0]
        or
        template_rep.shape[1]
        > search_rep.shape[1]
    ):

        raise RuntimeError(
            "Template is larger than search image."
        )

    response = cv2.matchTemplate(
        search_rep,
        template_rep,
        cv2.TM_CCOEFF_NORMED,
    )

    return response


# ============================================================================
# TOP-K EXTRACTION
# ============================================================================

def extract_top_k(
    response,
    k,
    suppression_radius=4,
):
    """
    Extract spatially non-duplicate top candidates.

    Returned coordinates are candidate CENTER coordinates,
    not template top-left coordinates.
    """

    response = np.asarray(
        response,
        dtype=np.float32,
    ).copy()

    candidates = []

    for _ in range(k):

        if response.size == 0:
            break

        _, max_value, _, max_location = (
            cv2.minMaxLoc(
                response
            )
        )

        if not np.isfinite(
            max_value
        ):
            break

        x, y = max_location

        center_x = (
            x
            + TEMPLATE_WIDTH / 2.0
        )

        center_y = (
            y
            + TEMPLATE_HEIGHT / 2.0
        )

        candidates.append(
            {
                "x": float(
                    center_x
                ),
                "y": float(
                    center_y
                ),
                "score": float(
                    max_value
                ),
            }
        )

        x0 = max(
            0,
            x - suppression_radius,
        )

        x1 = min(
            response.shape[1],
            x + suppression_radius + 1,
        )

        y0 = max(
            0,
            y - suppression_radius,
        )

        y1 = min(
            response.shape[0],
            y + suppression_radius + 1,
        )

        response[
            y0:y1,
            x0:x1
        ] = -np.inf

    return candidates


# ============================================================================
# DISTANCE
# ============================================================================

def candidate_distance(
    candidate,
    gt_x,
    gt_y,
):
    return math.hypot(
        candidate["x"] - gt_x,
        candidate["y"] - gt_y,
    )


# ============================================================================
# RECALL
# ============================================================================

def candidate_recall(
    candidates,
    gt_x,
    gt_y,
    k,
    tolerance=5.0,
):
    """
    Candidate recall means:

    Is at least one candidate within tolerance
    pixels of the true target?
    """

    for candidate in candidates[:k]:

        distance = candidate_distance(
            candidate,
            gt_x,
            gt_y,
        )

        if distance <= tolerance:

            return True

    return False


# ============================================================================
# UNION
# ============================================================================

def union_candidates(
    candidate_bank,
    generators,
    k,
    merge_radius=5.0,
):
    """
    Merge candidate lists from multiple generators.

    Candidates within merge_radius pixels are treated as the same
    spatial candidate.

    The strongest score is retained.
    """

    pool = []

    for generator in generators:

        candidates = candidate_bank[
            generator
        ]

        for candidate in candidates[:k]:

            pool.append(
                {
                    "x": candidate["x"],
                    "y": candidate["y"],
                    "score": candidate["score"],
                    "generator": generator,
                }
            )

    # Highest response first.
    pool.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    merged = []

    for candidate in pool:

        duplicate = False

        for selected in merged:

            distance = math.hypot(
                candidate["x"]
                - selected["x"],

                candidate["y"]
                - selected["y"],
            )

            if distance <= merge_radius:

                duplicate = True
                break

        if not duplicate:

            merged.append(
                candidate
            )

    return merged


# ============================================================================
# CANONICAL SCENE
# ============================================================================

def render_canonical_scene(
    scene_type,
    seed,
):
    """
    Render exactly one canonical scene.

    This function deliberately uses only the validated canonical
    renderer API.

    Target:
        physical (75.25, 113.75)

    Search:
        1000 × 1000

    Reference:
        100 × 100
    """

    tx = TARGET_PHYSICAL_X
    ty = TARGET_PHYSICAL_Y

    # ------------------------------------------------------------------
    # Search observation
    # ------------------------------------------------------------------

    search = canonical.render_search(
        tx,
        ty,
        scene_type,
        seed,
    )

    # ------------------------------------------------------------------
    # Reference observation
    # ------------------------------------------------------------------

    reference = canonical.render_reference(
        tx,
        ty,
        scene_type,
        seed,
    )

    # ------------------------------------------------------------------
    # Canonical PS02 template
    # ------------------------------------------------------------------

    template = canonical.create_ps02_template(
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

    # ------------------------------------------------------------------
    # Hard validation
    # ------------------------------------------------------------------

    if search.shape != (
        SEARCH_HEIGHT,
        SEARCH_WIDTH,
    ):

        raise RuntimeError(
            "Canonical search shape mismatch: "
            f"{search.shape}; expected "
            f"{SEARCH_HEIGHT, SEARCH_WIDTH}"
        )

    if reference.shape != (
        REFERENCE_HEIGHT,
        REFERENCE_WIDTH,
    ):

        raise RuntimeError(
            "Canonical reference shape mismatch: "
            f"{reference.shape}; expected "
            f"{REFERENCE_HEIGHT, REFERENCE_WIDTH}"
        )

    if template.shape != (
        TEMPLATE_HEIGHT,
        TEMPLATE_WIDTH,
    ):

        raise RuntimeError(
            "Canonical template shape mismatch: "
            f"{template.shape}; expected "
            f"{TEMPLATE_HEIGHT, TEMPLATE_WIDTH}"
        )

    # ------------------------------------------------------------------
    # Ground truth in search coordinates
    # ------------------------------------------------------------------

    gt_x = (
        tx
        * SEARCH_PIXELS_PER_UNIT
    )

    gt_y = (
        ty
        * SEARCH_PIXELS_PER_UNIT
    )

    return (
        as_float32(search),
        as_float32(reference),
        as_float32(template),
        gt_x,
        gt_y,
    )


# ============================================================================
# PROCESS ONE SCENE
# ============================================================================

def process_scene(
    scene_type,
    seed,
):
    """
    Process one canonical scene.
    """

    (
        search,
        reference,
        template,
        gt_x,
        gt_y,
    ) = render_canonical_scene(
        scene_type,
        seed,
    )

    candidate_bank = {}

    # ------------------------------------------------------------------
    # Generate all candidate pools
    # ------------------------------------------------------------------

    for generator in GENERATORS:

        response = compute_response(
            search,
            template,
            generator,
        )

        candidate_bank[
            generator
        ] = extract_top_k(
            response,
            max(K_VALUES),
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    scene_results = []

    # ------------------------------------------------------------------
    # Individual generators
    # ------------------------------------------------------------------

    for generator in GENERATORS:

        candidates = candidate_bank[
            generator
        ]

        for k in K_VALUES:

            found = candidate_recall(
                candidates,
                gt_x,
                gt_y,
                k,
                tolerance=5.0,
            )

            scene_results.append(
                {
                    "scene_type": scene_type,
                    "seed": seed,
                    "method": generator,
                    "K": k,
                    "recall": int(found),
                }
            )

    # ------------------------------------------------------------------
    # Generator combinations
    # ------------------------------------------------------------------

    for combination_name, generators in (
        COMBINATIONS.items()
    ):

        for k in K_VALUES:

            merged = union_candidates(
                candidate_bank,
                generators,
                k,
                merge_radius=5.0,
            )

            found = candidate_recall(
                merged,
                gt_x,
                gt_y,
                k,
                tolerance=5.0,
            )

            scene_results.append(
                {
                    "scene_type": scene_type,
                    "seed": seed,
                    "method": combination_name,
                    "K": k,
                    "recall": int(found),
                }
            )

    return scene_results


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(
    rows,
):
    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    methods = (
        GENERATORS
        + list(
            COMBINATIONS.keys()
        )
    )

    for scene_type in [
        "periodic",
        "quasiperiodic",
    ]:

        print()
        print(
            scene_type.upper()
        )

        print(
            "-" * 76
        )

        for method in methods:

            output = []

            for k in K_VALUES:

                selected = [
                    row
                    for row in rows
                    if row["scene_type"]
                    == scene_type
                    and row["method"]
                    == method
                    and row["K"]
                    == k
                ]

                if not selected:
                    continue

                recall_value = (
                    100.0
                    * sum(
                        row["recall"]
                        for row in selected
                    )
                    / len(selected)
                )

                output.append(
                    f"R@{k}="
                    f"{recall_value:6.2f}%"
                )

            print(
                f"{method:<25}"
                + "  ".join(
                    output
                )
            )


# ============================================================================
# SAVE CSV
# ============================================================================

def save_results(
    rows,
):
    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scene_type",
                "seed",
                "method",
                "K",
                "recall",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    rows = []

    total_scenes = (
        len(SEEDS)
        * 2
    )

    scene_counter = 0

    # ========================================================================
    # PERIODIC
    # ========================================================================

    print()
    print("=" * 76)
    print("SCENE TYPE: PERIODIC")
    print("=" * 76)

    for seed in SEEDS:

        scene_counter += 1

        print(
            f"Scene "
            f"{scene_counter:02d}/"
            f"{total_scenes}"
        )

        result = process_scene(
            "periodic",
            seed,
        )

        rows.extend(
            result
        )

    # ========================================================================
    # QUASIPERIODIC
    # ========================================================================

    print()
    print("=" * 76)
    print("SCENE TYPE: QUASIPERIODIC")
    print("=" * 76)

    for seed in SEEDS:

        scene_counter += 1

        print(
            f"Scene "
            f"{scene_counter:02d}/"
            f"{total_scenes}"
        )

        result = process_scene(
            "quasiperiodic",
            seed,
        )

        rows.extend(
            result
        )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print_summary(
        rows
    )

    # ========================================================================
    # VALIDITY INFORMATION
    # ========================================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 76)
    print("VALIDITY / METHODOLOGY")
    print("=" * 76)

    print(
        "Renderer:              canonical_renderer.py"
    )

    print(
        "Search observation:    canonical"
    )

    print(
        "Reference observation: canonical"
    )

    print(
        "Template:              canonical PS02"
    )

    print(
        "Target injection:      NONE"
    )

    print(
        "Alternate renderer:    NO"
    )

    print(
        "Candidate tolerance:   5 px"
    )

    print(
        "Metric:                Recall@K"
    )

    print(
        "Scenes:                "
        f"{total_scenes}"
    )

    print(
        "K values:              "
        f"{K_VALUES}"
    )

    print()

    # ========================================================================
    # SAVE
    # ========================================================================

    save_results(
        rows
    )

    print(
        "Runtime: "
        f"{elapsed:.2f} seconds"
    )

    print()

    print(
        "Saved:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print("=" * 76)
    print("STEP 25 COMPLETE")
    print("=" * 76)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()