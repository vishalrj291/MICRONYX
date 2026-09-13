"""
============================================================================
MICRONYX STEP 25
ADAPTIVE CANDIDATE GENERATION
============================================================================

Goal:
    Improve candidate recall before learned candidate ranking.

Architecture:

    Continuous physical scene
            |
            v
    Search / Reference observations
            |
            v
    Generator bank
        - NCC
        - DOG
        - Gradient
        - Edge
        - Frequency
            |
            v
    Candidate union
            |
            v
    Deduplication
            |
            v
    Recall@K evaluation

Important:
    This experiment evaluates candidate GENERATION only.
    It does not train the Step 24 XGBoost ranker.

The purpose is to answer:

    "Can we get the true target into the candidate pool
     more reliably than DOG alone?"
"""

from __future__ import annotations

import os
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

SEARCH_W = 1000
SEARCH_H = 1000

REFERENCE_W = 100
REFERENCE_H = 100

PHYSICAL_W = 200.0
PHYSICAL_H = 200.0

SEARCH_PPU = 5.0
REFERENCE_PPU = 50.0

TARGET_X = 75.25
TARGET_Y = 113.75

REFERENCE_FOV_W = 2.0
REFERENCE_FOV_H = 2.0

TARGET_SEARCH_X = TARGET_X * SEARCH_PPU
TARGET_SEARCH_Y = TARGET_Y * SEARCH_PPU

TEMPLATE_W = 10
TEMPLATE_H = 10

SEEDS = list(range(20260950, 20260980))

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

K_VALUES = [10, 25, 50, 100, 250, 500, 1000]

# Generator combinations.
COMBINATIONS = {
    "ncc": ["ncc"],
    "dog": ["dog"],
    "gradient": ["gradient"],
    "edge": ["edge"],
    "frequency": ["frequency"],

    "ncc_dog": ["ncc", "dog"],
    "dog_gradient": ["dog", "gradient"],
    "dog_frequency": ["dog", "frequency"],
    "dog_edge": ["dog", "edge"],

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

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "v02"
    / "adaptive_candidate_generation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "adaptive_candidate_generation_results.csv"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Scene:
    scene_type: str
    seed: int

    search: np.ndarray
    reference: np.ndarray

    target_x: float
    target_y: float


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    mn = float(np.min(img))
    mx = float(np.max(img))

    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    out = (img - mn) / (mx - mn)
    out = np.clip(out * 255.0, 0, 255)

    return out.astype(np.uint8)


def zscore(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    mean = float(np.mean(img))
    std = float(np.std(img))

    if std < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    return (img - mean) / std


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    mn = np.min(x)
    mx = np.max(x)

    if mx - mn < 1e-8:
        return np.zeros_like(x)

    return (x - mn) / (mx - mn)


def center_distance(x: float, y: float) -> float:
    return math.sqrt(
        (x - TARGET_SEARCH_X) ** 2
        + (y - TARGET_SEARCH_Y) ** 2
    )


# ============================================================================
# CONTINUOUS PHYSICAL SCENE
# ============================================================================

def create_continuous_scene(
    scene_type: str,
    seed: int,
    resolution: int = 2000,
) -> np.ndarray:

    rng = np.random.default_rng(seed)

    xs = np.linspace(
        0.0,
        PHYSICAL_W,
        resolution,
        endpoint=False,
        dtype=np.float32,
    )

    ys = np.linspace(
        0.0,
        PHYSICAL_H,
        resolution,
        endpoint=False,
        dtype=np.float32,
    )

    X, Y = np.meshgrid(xs, ys)

    # ------------------------------------------------------------
    # Base semiconductor-like line structure
    # ------------------------------------------------------------

    pitch = 0.5

    if scene_type == "periodic":

        phase_x = 0.0
        phase_y = 0.0

        px = np.mod(X + phase_x, pitch)
        py = np.mod(Y + phase_y, pitch)

        line_x = np.exp(
            -(px / 0.075) ** 2
        )

        line_y = np.exp(
            -(py / 0.075) ** 2
        )

        structure = 0.5 * line_x + 0.5 * line_y

    else:

        # Quasiperiodic structure.
        phase_x = rng.uniform(0, pitch)
        phase_y = rng.uniform(0, pitch)

        pitch_variation_x = (
            1.0
            + 0.06 * np.sin(Y * 0.11)
            + 0.025 * np.sin(Y * 0.37)
        )

        pitch_variation_y = (
            1.0
            + 0.05 * np.sin(X * 0.09)
            + 0.02 * np.sin(X * 0.31)
        )

        local_x = np.mod(
            X + phase_x + 0.18 * np.sin(Y * 0.07),
            pitch * pitch_variation_x,
        )

        local_y = np.mod(
            Y + phase_y + 0.16 * np.sin(X * 0.08),
            pitch * pitch_variation_y,
        )

        line_x = np.exp(
            -(local_x / 0.075) ** 2
        )

        line_y = np.exp(
            -(local_y / 0.075) ** 2
        )

        structure = 0.5 * line_x + 0.5 * line_y

    # ------------------------------------------------------------
    # Low-frequency physical illumination variation
    # ------------------------------------------------------------

    illumination = (
        0.10 * np.sin(X * 0.021)
        + 0.08 * np.cos(Y * 0.017)
        + 0.04 * np.sin((X + Y) * 0.013)
    )

    scene = structure + illumination

    # ------------------------------------------------------------
    # Deterministic target fingerprint
    #
    # This represents a local structural feature that makes the
    # target region distinguishable from an arbitrary repeated cell.
    # ------------------------------------------------------------

    dx = X - TARGET_X
    dy = Y - TARGET_Y

    target_mask = np.exp(
        -(
            (dx / 0.34) ** 2
            + (dy / 0.34) ** 2
        )
    )

    target_feature = (
        0.30 * np.exp(
            -(
                ((dx - 0.12) / 0.08) ** 2
                + ((dy + 0.05) / 0.18) ** 2
            )
        )
        +
        0.22 * np.exp(
            -(
                ((dx + 0.11) / 0.06) ** 2
                + ((dy - 0.10) / 0.07) ** 2
            )
        )
    )

    scene += target_mask * target_feature

    # ------------------------------------------------------------
    # Small structural perturbations
    # ------------------------------------------------------------

    scene += (
        0.015
        * np.sin(X * 0.91 + Y * 0.37)
    )

    return scene.astype(np.float32)


# ============================================================================
# OBSERVATION MODEL
# ============================================================================

def area_sample(
    continuous: np.ndarray,
    output_h: int,
    output_w: int,
) -> np.ndarray:

    """
    Approximate sensor-area integration.

    INTER_AREA is used instead of nearest-neighbor resizing.
    """

    return cv2.resize(
        continuous,
        (output_w, output_h),
        interpolation=cv2.INTER_AREA,
    )


def extract_reference(
    continuous: np.ndarray,
) -> np.ndarray:

    """
    Extract the 2x2 physical-unit reference FOV centered on the
    physical target.

    The continuous scene has 2000x2000 samples over 200x200 units,
    therefore 10 samples/unit.

    Reference:
        2x2 physical units
        100x100 sensor pixels
    """

    samples_per_unit = 10.0

    center_x = int(round(TARGET_X * samples_per_unit))
    center_y = int(round(TARGET_Y * samples_per_unit))

    half_w = int(
        round(
            REFERENCE_FOV_W
            * samples_per_unit
            / 2.0
        )
    )

    half_h = int(
        round(
            REFERENCE_FOV_H
            * samples_per_unit
            / 2.0
        )
    )

    x0 = center_x - half_w
    x1 = center_x + half_w

    y0 = center_y - half_h
    y1 = center_y + half_h

    crop = continuous[y0:y1, x0:x1]

    reference = cv2.resize(
        crop,
        (REFERENCE_W, REFERENCE_H),
        interpolation=cv2.INTER_AREA,
    )

    return normalize_uint8(reference)


def build_scene(
    scene_type: str,
    seed: int,
) -> Scene:

    continuous = create_continuous_scene(
        scene_type,
        seed,
    )

    search = area_sample(
        continuous,
        SEARCH_H,
        SEARCH_W,
    )

    reference = extract_reference(
        continuous,
    )

    search = normalize_uint8(search)

    return Scene(
        scene_type=scene_type,
        seed=seed,
        search=search,
        reference=reference,
        target_x=TARGET_SEARCH_X,
        target_y=TARGET_SEARCH_Y,
    )


# ============================================================================
# TEMPLATE PREPARATION
# ============================================================================

def make_template(reference: np.ndarray) -> np.ndarray:

    """
    Convert 100x100 high-magnification reference to the
    PS02-equivalent 10x10 search-space template.
    """

    return cv2.resize(
        reference,
        (TEMPLATE_W, TEMPLATE_H),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================================
# REPRESENTATIONS
# ============================================================================

def dog_representation(
    img: np.ndarray,
) -> np.ndarray:

    x = img.astype(np.float32) / 255.0

    g1 = cv2.GaussianBlur(
        x,
        (0, 0),
        sigmaX=0.8,
    )

    g2 = cv2.GaussianBlur(
        x,
        (0, 0),
        sigmaX=2.0,
    )

    dog = g1 - g2

    return dog.astype(np.float32)


def gradient_representation(
    img: np.ndarray,
) -> np.ndarray:

    x = img.astype(np.float32) / 255.0

    gx = cv2.Sobel(
        x,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        x,
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


def edge_representation(
    img: np.ndarray,
) -> np.ndarray:

    return cv2.Canny(
        img,
        threshold1=30,
        threshold2=100,
    ).astype(np.float32)


def frequency_representation(
    img: np.ndarray,
) -> np.ndarray:

    """
    Frequency-domain representation.

    Removes the dominant low-frequency component using
    an FFT high-pass mask, then transforms back to spatial
    representation.

    This is deliberately lightweight because this step is
    evaluating candidate generation rather than final
    frequency-domain verification.
    """

    x = img.astype(np.float32)

    h, w = x.shape

    window_y = np.hanning(h).astype(np.float32)
    window_x = np.hanning(w).astype(np.float32)

    window = np.outer(
        window_y,
        window_x,
    )

    xw = x * window

    F = np.fft.fftshift(
        np.fft.fft2(xw)
    )

    cy = h // 2
    cx = w // 2

    yy, xx = np.ogrid[:h, :w]

    radius = np.sqrt(
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    cutoff = max(
        3.0,
        min(h, w) * 0.035,
    )

    highpass = radius > cutoff

    F_filtered = F * highpass

    result = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(F_filtered)
        )
    )

    return result.astype(np.float32)


# ============================================================================
# MATCHING
# ============================================================================

def match_response(
    search: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:

    return cv2.matchTemplate(
        search.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )


def get_generator_response(
    search: np.ndarray,
    template: np.ndarray,
    generator: str,
) -> np.ndarray:

    if generator == "ncc":

        return match_response(
            search,
            template,
        )

    if generator == "dog":

        search_r = dog_representation(search)

        template_r = dog_representation(template)

        return match_response(
            search_r,
            template_r,
        )

    if generator == "gradient":

        search_r = gradient_representation(search)

        template_r = gradient_representation(template)

        return match_response(
            search_r,
            template_r,
        )

    if generator == "edge":

        search_r = edge_representation(search)

        template_r = edge_representation(template)

        return match_response(
            search_r,
            template_r,
        )

    if generator == "frequency":

        search_r = frequency_representation(search)

        template_r = frequency_representation(template)

        return match_response(
            search_r,
            template_r,
        )

    raise ValueError(
        f"Unknown generator: {generator}"
    )


# ============================================================================
# CANDIDATE EXTRACTION
# ============================================================================

def response_to_candidates(
    response: np.ndarray,
    k: int,
) -> list[tuple[float, float, float]]:

    """
    Extract top-k candidate centers.

    Non-maximum suppression prevents a cluster of adjacent
    pixels around the same candidate from occupying the
    entire candidate list.
    """

    work = response.copy()

    candidates = []

    radius = 4

    for _ in range(k):

        _, max_val, _, max_loc = cv2.minMaxLoc(
            work
        )

        if not np.isfinite(max_val):
            break

        x, y = max_loc

        # Convert template top-left to center.
        cx = x + TEMPLATE_W / 2.0
        cy = y + TEMPLATE_H / 2.0

        candidates.append(
            (
                float(cx),
                float(cy),
                float(max_val),
            )
        )

        x0 = max(0, x - radius)
        x1 = min(
            work.shape[1],
            x + radius + 1,
        )

        y0 = max(0, y - radius)
        y1 = min(
            work.shape[0],
            y + radius + 1,
        )

        work[y0:y1, x0:x1] = -np.inf

    return candidates


# ============================================================================
# CANDIDATE UNION
# ============================================================================

def deduplicate_candidates(
    candidates: list[tuple[float, float, float, str]],
    distance_threshold: float = 5.0,
) -> list[tuple[float, float, float, str]]:

    """
    Merge candidates from different generators.

    If two candidates are spatially close, retain the one
    with the stronger score.
    """

    candidates = sorted(
        candidates,
        key=lambda x: x[2],
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        x, y, score, source = candidate

        too_close = False

        for sx, sy, _, _ in selected:

            distance = math.sqrt(
                (x - sx) ** 2
                + (y - sy) ** 2
            )

            if distance <= distance_threshold:

                too_close = True
                break

        if not too_close:

            selected.append(candidate)

    return selected


# ============================================================================
# RECALL
# ============================================================================

def contains_ground_truth(
    candidates,
    target_x,
    target_y,
    tolerance,
) -> bool:

    for x, y, *_ in candidates:

        distance = math.sqrt(
            (x - target_x) ** 2
            + (y - target_y) ** 2
        )

        if distance <= tolerance:
            return True

    return False


def recall_at_k(
    candidates,
    target_x,
    target_y,
    k,
    tolerance=5.0,
) -> bool:

    return contains_ground_truth(
        candidates[:k],
        target_x,
        target_y,
        tolerance,
    )


# ============================================================================
# DATASET PROFILE
# ============================================================================

def estimate_periodicity(
    image: np.ndarray,
) -> float:

    """
    Simple autocorrelation-based periodicity indicator.

    This is NOT a final scientific periodicity estimator.
    It is only used for Step 25 adaptive policy analysis.
    """

    x = image.astype(np.float32)

    x -= np.mean(x)

    if np.std(x) < 1e-8:
        return 0.0

    # Horizontal autocorrelation.
    max_shift = min(
        100,
        x.shape[1] // 3,
    )

    scores = []

    for shift in range(5, max_shift):

        a = x[:, :-shift]
        b = x[:, shift:]

        denom = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            + 1e-8
        )

        score = float(
            np.sum(a * b) / denom
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(
        np.clip(
            max(scores),
            0.0,
            1.0,
        )
    )


def dataset_profile(
    scene: Scene,
) -> dict:

    img = scene.search.astype(np.float32)

    dynamic_range = float(
        np.max(img) - np.min(img)
    )

    contrast = float(
        np.std(img) / 255.0
    )

    periodicity = estimate_periodicity(
        scene.search
    )

    gradient = gradient_representation(
        scene.search
    )

    edge_density = float(
        np.mean(
            scene.search > np.mean(scene.search)
        )
    )

    return {
        "dynamic_range": dynamic_range,
        "contrast": contrast,
        "periodicity": periodicity,
        "edge_density": edge_density,
    }


# ============================================================================
# ADAPTIVE POLICY
# ============================================================================

def choose_adaptive_generators(
    profile: dict,
) -> list[str]:

    """
    Preliminary rule-based policy.

    IMPORTANT:
    This is intentionally NOT the final AutoML policy.

    Step 25 first establishes whether dataset-adaptive
    generator selection is useful.

    Later this policy can itself be learned.
    """

    periodicity = profile["periodicity"]
    contrast = profile["contrast"]
    edge_density = profile["edge_density"]

    generators = [
        "dog"
    ]

    # Strong periodicity:
    # gradient/frequency representations provide complementary
    # information to DOG.
    if periodicity > 0.45:

        generators.append(
            "gradient"
        )

        generators.append(
            "frequency"
        )

    # High structural edge content.
    if edge_density > 0.45:

        generators.append(
            "edge"
        )

    # Low contrast.
    if contrast < 0.18:

        generators.append(
            "ncc"
        )

    # Remove duplicates while preserving order.
    generators = list(
        dict.fromkeys(generators)
    )

    return generators


# ============================================================================
# SINGLE SCENE EVALUATION
# ============================================================================

def evaluate_scene(
    scene: Scene,
) -> dict:

    template = make_template(
        scene.reference
    )

    responses = {}

    candidate_sets = {}

    # ------------------------------------------------------------
    # Individual generators
    # ------------------------------------------------------------

    for generator in [
        "ncc",
        "dog",
        "gradient",
        "edge",
        "frequency",
    ]:

        response = get_generator_response(
            scene.search,
            template,
            generator,
        )

        responses[generator] = response

        candidate_sets[generator] = (
            response_to_candidates(
                response,
                max(K_VALUES),
            )
        )

    # ------------------------------------------------------------
    # Generator combinations
    # ------------------------------------------------------------

    combination_candidates = {}

    for name, generators in COMBINATIONS.items():

        merged = []

        for generator in generators:

            for x, y, score in candidate_sets[
                generator
            ]:

                merged.append(
                    (
                        x,
                        y,
                        score,
                        generator,
                    )
                )

        merged = deduplicate_candidates(
            merged,
            distance_threshold=5.0,
        )

        combination_candidates[name] = merged

    # ------------------------------------------------------------
    # Adaptive policy
    # ------------------------------------------------------------

    profile = dataset_profile(
        scene
    )

    adaptive_generators = (
        choose_adaptive_generators(
            profile
        )
    )

    adaptive_pool = []

    for generator in adaptive_generators:

        for x, y, score in candidate_sets[
            generator
        ]:

            adaptive_pool.append(
                (
                    x,
                    y,
                    score,
                    generator,
                )
            )

    adaptive_pool = deduplicate_candidates(
        adaptive_pool,
        distance_threshold=5.0,
    )

    combination_candidates[
        "adaptive"
    ] = adaptive_pool

    return {
        "profile": profile,
        "adaptive_generators": adaptive_generators,
        "candidates": combination_candidates,
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():

    print()
    print("=" * 76)
    print("MICRONYX STEP 25")
    print("ADAPTIVE CANDIDATE GENERATION")
    print("=" * 76)
    print()

    print("Search resolution:       "
          f"{SEARCH_W} × {SEARCH_H}")

    print("Reference resolution:    "
          f"{REFERENCE_W} × {REFERENCE_H}")

    print("Template:                "
          f"{TEMPLATE_W} × {TEMPLATE_H}")

    print("Scenes:                  "
          f"{len(SEEDS) * len(SCENE_TYPES)}")

    print("K values:                "
          f"{K_VALUES}")

    print()

    print("Generator bank:")
    print("  NCC")
    print("  DOG")
    print("  Gradient")
    print("  Edge")
    print("  Frequency")

    print()

    all_rows = []

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    total_scenes = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    scene_index = 0

    start_time = time.time()

    for scene_type in SCENE_TYPES:

        print()
        print("=" * 76)
        print(
            f"SCENE TYPE: "
            f"{scene_type.upper()}"
        )
        print("=" * 76)

        for seed in SEEDS:

            scene_index += 1

            print(
                f"Scene "
                f"{scene_index:02d}/"
                f"{total_scenes}"
            )

            scene = build_scene(
                scene_type,
                seed,
            )

            result = evaluate_scene(
                scene
            )

            profile = result[
                "profile"
            ]

            adaptive_generators = result[
                "adaptive_generators"
            ]

            candidates = result[
                "candidates"
            ]

            # ----------------------------------------------------
            # Store every generator / combination
            # ----------------------------------------------------

            for name, candidate_list in candidates.items():

                for K in K_VALUES:

                    present = recall_at_k(
                        candidate_list,
                        scene.target_x,
                        scene.target_y,
                        K,
                        tolerance=5.0,
                    )

                    all_rows.append(
                        {
                            "scene_type": scene_type,
                            "seed": seed,
                            "method": name,
                            "K": K,
                            "recall": int(
                                present
                            ),
                            "periodicity": profile[
                                "periodicity"
                            ],
                            "contrast": profile[
                                "contrast"
                            ],
                            "edge_density": profile[
                                "edge_density"
                            ],
                            "adaptive_generators":
                                "+".join(
                                    adaptive_generators
                                ),
                        }
                    )

    elapsed = time.time() - start_time

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    for scene_type in SCENE_TYPES:

        print()
        print(
            scene_type.upper()
        )
        print("-" * 76)

        for method in COMBINATIONS.keys():

            rows = [
                r
                for r in all_rows
                if r["scene_type"] == scene_type
                and r["method"] == method
            ]

            if not rows:
                continue

            values = {}

            for K in K_VALUES:

                k_rows = [
                    r
                    for r in rows
                    if r["K"] == K
                ]

                recall = (
                    100.0
                    * sum(
                        r["recall"]
                        for r in k_rows
                    )
                    / len(k_rows)
                )

                values[K] = recall

            formatted = "  ".join(
                f"R@{K}={values[K]:6.2f}%"
                for K in K_VALUES
            )

            print(
                f"{method:<25}"
                f"{formatted}"
            )

        # Adaptive separately.
        rows = [
            r
            for r in all_rows
            if r["scene_type"] == scene_type
            and r["method"] == "adaptive"
        ]

        if rows:

            print()
            print(
                "ADAPTIVE POLICY"
            )
            print("-" * 76)

            for K in K_VALUES:

                k_rows = [
                    r
                    for r in rows
                    if r["K"] == K
                ]

                recall = (
                    100.0
                    * sum(
                        r["recall"]
                        for r in k_rows
                    )
                    / len(k_rows)
                )

                print(
                    f"Recall@{K:<4}: "
                    f"{recall:6.2f}%"
                )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    fieldnames = [
        "scene_type",
        "seed",
        "method",
        "K",
        "recall",
        "periodicity",
        "contrast",
        "edge_density",
        "adaptive_generators",
    ]

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print()
    print("=" * 76)
    print("INTERPRETATION")
    print("=" * 76)

    print()
    print(
        "The key metric is candidate Recall@K."
    )

    print(
        "A learned verifier cannot recover a target "
        "that never enters the candidate pool."
    )

    print()
    print(
        "Step 25 therefore evaluates whether combining "
        "multiple physically different representations "
        "improves candidate recall."
    )

    print()
    print(
        "Adaptive policy is currently a preliminary "
        "EDA-driven policy."
    )

    print(
        "It will NOT be treated as the final automated "
        "model-selection system."
    )

    print()
    print(
        f"Runtime: {elapsed:.2f} seconds"
    )

    print()
    print(
        "Saved:"
    )

    print(
        RESULTS_CSV
    )

    print()
    print("=" * 76)
    print("STEP 25 COMPLETE")
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()