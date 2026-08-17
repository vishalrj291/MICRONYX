"""
MICRONYX PS02
FINAL INFERENCE / LOCALIZATION SCRIPT

Input:
    1000x1000 search image
    1000x1000 reference image

Output:
    predicted center (x, y) in search-image pixels

Pipeline:
    Search + Reference
        ↓
    Physical-context extraction
        ↓
    DOG candidate generation
        ↓
    12-feature candidate representation
        ↓
    Existing XGBoost verifier
        ↓
    Candidate ranking
        ↓
    Final predicted center

Important:
    This script uses the already-trained MICRONYX XGBoost model.
    It does NOT retrain the model.
"""

from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "learned_ranker"
    / "xgboost_ranker.json"
)


# ============================================================
# CANONICAL ACQUISITION MODEL
# ============================================================

SEARCH_WIDTH = 1000
SEARCH_HEIGHT = 1000

REFERENCE_WIDTH = 1000
REFERENCE_HEIGHT = 1000

SEARCH_PIXELS_PER_UNIT = 5.0
REFERENCE_PIXELS_PER_UNIT = 50.0

SAMPLING_RATIO = (
    REFERENCE_PIXELS_PER_UNIT
    / SEARCH_PIXELS_PER_UNIT
)

DOWNSAMPLE_FACTOR = int(
    SAMPLING_RATIO
)


# ============================================================
# LOCALIZATION CONFIGURATION
# ============================================================

TOP_K = 250
MIN_DISTANCE = 6

CONTEXT_SIZES = [10, 20, 40]

FEATURE_NAMES = [
    "dog_score",
    "context_10",
    "context_20",
    "context_40",
    "gradient_score",
    "orientation_score",
    "contrast_score",
    "context_gain_20",
    "context_gain_40",
    "dog_context_gap",
    "context_consistency",
    "dog_rank_normalized",
]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    """
    Load an image as grayscale float32 in [0, 1].
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    if image.shape != (
        SEARCH_HEIGHT,
        SEARCH_WIDTH,
    ):
        raise RuntimeError(
            f"Search image must be "
            f"{SEARCH_WIDTH}x{SEARCH_HEIGHT}; "
            f"got {image.shape}"
        )

    return (
        image.astype(np.float32)
        / 255.0
    )


def load_reference(path):
    """
    Load the 1000x1000 high-resolution reference.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read reference image:\n{path}"
        )

    if image.shape != (
        REFERENCE_HEIGHT,
        REFERENCE_WIDTH,
    ):
        raise RuntimeError(
            f"Reference image must be "
            f"{REFERENCE_WIDTH}x{REFERENCE_HEIGHT}; "
            f"got {image.shape}"
        )

    return (
        image.astype(np.float32)
        / 255.0
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):

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
            image,
            dtype=np.float32,
        )

    return (
        image - mean
    ) / std


# ============================================================
# DIFFERENCE OF GAUSSIANS
# ============================================================

def dog(image):

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

    return normalize(
        g1 - g2
    )


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def template_match(
    search,
    template,
):

    return cv2.matchTemplate(
        search.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )


# ============================================================
# TOP-K CANDIDATES
# ============================================================

def top_k_candidates(
    response,
    k,
    min_distance=6,
):

    work = response.copy()

    candidates = []

    h, w = work.shape

    for _ in range(k):

        _, score, _, loc = (
            cv2.minMaxLoc(work)
        )

        x, y = loc

        if not np.isfinite(score):
            break

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
            w,
            x + min_distance + 1,
        )

        y0 = max(
            0,
            y - min_distance,
        )

        y1 = min(
            h,
            y + min_distance + 1,
        )

        work[
            y0:y1,
            x0:x1
        ] = -np.inf

    return candidates


# ============================================================
# REFERENCE PHYSICAL CONTEXTS
# ============================================================

def block_average(
    image,
    factor,
):
    """
    Deterministic physical downsampling.

    No arbitrary interpolation.
    """

    h, w = image.shape

    if (
        h % factor != 0
        or w % factor != 0
    ):
        raise RuntimeError(
            "Image dimensions must be "
            "divisible by downsampling factor."
        )

    output_h = h // factor
    output_w = w // factor

    reshaped = image.reshape(
        output_h,
        factor,
        output_w,
        factor,
    )

    return reshaped.mean(
        axis=(1, 3)
    ).astype(np.float32)


def center_crop(
    image,
    size,
):

    h, w = image.shape

    cx = w // 2
    cy = h // 2

    x0 = cx - size // 2
    y0 = cy - size // 2

    x1 = x0 + size
    y1 = y0 + size

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):
        raise RuntimeError(
            f"Cannot extract "
            f"{size}x{size} center crop "
            f"from {w}x{h} reference."
        )

    return image[
        y0:y1,
        x0:x1
    ].copy()


def build_reference_contexts(
    reference,
):
    """
    Convert canonical 1000x1000 reference
    into the three search-equivalent contexts.

    2 physical units:
        100x100 reference
        -> 10x10 search context

    4 physical units:
        200x200 reference
        -> 20x20 search context

    8 physical units:
        400x400 reference
        -> 40x40 search context
    """

    contexts = {}

    for search_size in CONTEXT_SIZES:

        physical_size = (
            search_size
            / SEARCH_PIXELS_PER_UNIT
        )

        reference_size = int(
            round(
                physical_size
                * REFERENCE_PIXELS_PER_UNIT
            )
        )

        crop = center_crop(
            reference,
            reference_size,
        )

        contexts[
            search_size
        ] = block_average(
            crop,
            DOWNSAMPLE_FACTOR,
        )

    return contexts


# ============================================================
# PATCH EXTRACTION
# ============================================================

def centered_patch(
    image,
    center_x,
    center_y,
    size,
):

    half = size / 2.0

    x0 = int(
        round(
            center_x - half
        )
    )

    y0 = int(
        round(
            center_y - half
        )
    )

    x1 = x0 + size
    y1 = y0 + size

    h, w = image.shape

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):
        return None

    return image[
        y0:y1,
        x0:x1
    ]


# ============================================================
# GRADIENT FEATURES
# ============================================================

def gradient_features(
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

    magnitude = np.sqrt(
        gx * gx
        + gy * gy
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


# ============================================================
# CONTEXT SCORE
# ============================================================

def context_score(
    search,
    reference,
    center_x,
    center_y,
):

    size = reference.shape[0]

    patch = centered_patch(
        search,
        center_x,
        center_y,
        size,
    )

    if patch is None:
        return -1.0

    return float(
        template_match(
            normalize(patch),
            normalize(reference),
        )[0, 0]
    )


# ============================================================
# CANDIDATE FEATURES
# ============================================================

def extract_candidate_features(
    search,
    references,
    candidate,
    dog_rank,
    max_candidates,
):

    x, y, dog_score = candidate

    center_x = x + 5.0
    center_y = y + 5.0

    context_10 = context_score(
        search,
        references[10],
        center_x,
        center_y,
    )

    context_20 = context_score(
        search,
        references[20],
        center_x,
        center_y,
    )

    context_40 = context_score(
        search,
        references[40],
        center_x,
        center_y,
    )

    (
        search_gx,
        search_gy,
        search_mag,
        search_ori,
    ) = gradient_features(
        search
    )

    (
        temp_gx,
        temp_gy,
        temp_mag,
        temp_ori,
    ) = gradient_features(
        references[10]
    )

    patch_gx = centered_patch(
        search_gx,
        center_x,
        center_y,
        10,
    )

    patch_gy = centered_patch(
        search_gy,
        center_x,
        center_y,
        10,
    )

    patch_mag = centered_patch(
        search_mag,
        center_x,
        center_y,
        10,
    )

    patch_ori = centered_patch(
        search_ori,
        center_x,
        center_y,
        10,
    )

    if (
        patch_gx is None
        or patch_gy is None
        or patch_mag is None
        or patch_ori is None
    ):

        gradient_score = 0.0
        orientation_score = 0.0

    else:

        gradient_score = float(
            np.mean(
                normalize(patch_mag)
                * normalize(temp_mag)
            )
        )

        delta = (
            patch_ori
            - temp_ori
        )

        orientation_similarity = (
            np.cos(delta)
        )

        weights = (
            patch_mag
            + temp_mag
            + 1e-6
        )

        orientation_score = float(
            np.sum(
                orientation_similarity
                * weights
            )
            / np.sum(weights)
        )

    gradient_score = float(
        np.clip(
            gradient_score,
            -1.0,
            1.0,
        )
    )

    orientation_score = float(
        np.clip(
            orientation_score,
            -1.0,
            1.0,
        )
    )

    local_patch = centered_patch(
        search,
        center_x,
        center_y,
        10,
    )

    if local_patch is None:

        contrast_score = 0.0

    else:

        contrast_score = float(
            np.mean(
                normalize(local_patch)
                * normalize(references[10])
            )
        )

    contrast_score = float(
        np.clip(
            contrast_score,
            -1.0,
            1.0,
        )
    )

    context_gain_20 = (
        context_20
        - context_10
    )

    context_gain_40 = (
        context_40
        - context_10
    )

    dog_context_gap = (
        dog_score
        - context_10
    )

    context_consistency = (
        context_10
        + context_20
        + context_40
    ) / 3.0

    dog_rank_normalized = (
        dog_rank
        / max_candidates
    )

    return np.array(
        [
            dog_score,
            context_10,
            context_20,
            context_40,
            gradient_score,
            orientation_score,
            contrast_score,
            context_gain_20,
            context_gain_40,
            dog_context_gap,
            context_consistency,
            dog_rank_normalized,
        ],
        dtype=np.float32,
    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained XGBoost model not found:\n"
            f"{MODEL_PATH}"
        )

    model = XGBClassifier()

    model.load_model(
        str(MODEL_PATH)
    )

    return model


# ============================================================
# LOCALIZATION
# ============================================================

def localize(
    search,
    reference,
):

    references = (
        build_reference_contexts(
            reference
        )
    )

    # 10x search-equivalent reference.
    reference_10 = references[10]

    search_dog = dog(search)
    reference_dog = dog(reference_10)

    response = template_match(
        search_dog,
        reference_dog,
    )

    candidates = top_k_candidates(
        response,
        TOP_K,
        MIN_DISTANCE,
    )

    if not candidates:
        raise RuntimeError(
            "No candidates generated."
        )

    (
        search_gx,
        search_gy,
        search_mag,
        search_ori,
    ) = gradient_features(search)

    (
        temp_gx,
        temp_gy,
        temp_mag,
        temp_ori,
    ) = gradient_features(
        reference_10
    )

    # Precomputed gradients are intentionally
    # calculated once for efficiency.
    feature_rows = []

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):

        x, y, dog_score = candidate

        center_x = x + 5.0
        center_y = y + 5.0

        context_10 = context_score(
            search,
            references[10],
            center_x,
            center_y,
        )

        context_20 = context_score(
            search,
            references[20],
            center_x,
            center_y,
        )

        context_40 = context_score(
            search,
            references[40],
            center_x,
            center_y,
        )

        patch_mag = centered_patch(
            search_mag,
            center_x,
            center_y,
            10,
        )

        patch_ori = centered_patch(
            search_ori,
            center_x,
            center_y,
            10,
        )

        if (
            patch_mag is None
            or patch_ori is None
        ):

            gradient_score = 0.0
            orientation_score = 0.0

        else:

            gradient_score = float(
                np.mean(
                    normalize(patch_mag)
                    * normalize(temp_mag)
                )
            )

            delta = (
                patch_ori
                - temp_ori
            )

            orientation_similarity = (
                np.cos(delta)
            )

            weights = (
                patch_mag
                + temp_mag
                + 1e-6
            )

            orientation_score = float(
                np.sum(
                    orientation_similarity
                    * weights
                )
                / np.sum(weights)
            )

        gradient_score = float(
            np.clip(
                gradient_score,
                -1.0,
                1.0,
            )
        )

        orientation_score = float(
            np.clip(
                orientation_score,
                -1.0,
                1.0,
            )
        )

        local_patch = centered_patch(
            search,
            center_x,
            center_y,
            10,
        )

        if local_patch is None:

            contrast_score = 0.0

        else:

            contrast_score = float(
                np.mean(
                    normalize(local_patch)
                    * normalize(reference_10)
                )
            )

        contrast_score = float(
            np.clip(
                contrast_score,
                -1.0,
                1.0,
            )
        )

        context_gain_20 = (
            context_20
            - context_10
        )

        context_gain_40 = (
            context_40
            - context_10
        )

        dog_context_gap = (
            dog_score
            - context_10
        )

        context_consistency = (
            context_10
            + context_20
            + context_40
        ) / 3.0

        dog_rank_normalized = (
            rank / TOP_K
        )

        feature_rows.append(
            [
                dog_score,
                context_10,
                context_20,
                context_40,
                gradient_score,
                orientation_score,
                contrast_score,
                context_gain_20,
                context_gain_40,
                dog_context_gap,
                context_consistency,
                dog_rank_normalized,
            ]
        )

    X = np.asarray(
        feature_rows,
        dtype=np.float32,
    )

    model = load_model()

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    best_index = int(
        np.argmax(probabilities)
    )

    best_candidate = candidates[
        best_index
    ]

    x, y, dog_score = best_candidate

    predicted_x = (
        x + 5.0
    )

    predicted_y = (
        y + 5.0
    )

    ranking = []

    for index, candidate in enumerate(
        candidates
    ):

        cx = candidate[0] + 5.0
        cy = candidate[1] + 5.0

        ranking.append(
            {
                "rank": index + 1,
                "x": float(cx),
                "y": float(cy),
                "dog_score": float(
                    candidate[2]
                ),
                "probability": float(
                    probabilities[index]
                ),
            }
        )

    ranking.sort(
        key=lambda item:
            item["probability"],
        reverse=True,
    )

    for index, item in enumerate(
        ranking,
        start=1,
    ):
        item["final_rank"] = index

    return {
        "predicted_x": float(
            predicted_x
        ),
        "predicted_y": float(
            predicted_y
        ),
        "confidence": float(
            probabilities[best_index]
        ),
        "candidate_count": len(
            candidates
        ),
        "top_candidates": ranking[:10],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MICRONYX PS02 final "
            "1000x1000 image localization"
        )
    )

    parser.add_argument(
        "--search",
        required=True,
        help="Path to 1000x1000 search image",
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to 1000x1000 reference image",
    )

    args = parser.parse_args()

    print("=" * 76)
    print(
        "MICRONYX PS02 FINAL LOCALIZATION"
    )
    print("=" * 76)

    print()
    print(
        "Search:",
        args.search,
    )

    print(
        "Reference:",
        args.reference,
    )

    print()
    print(
        "Canonical acquisition:"
    )

    print(
        "  Search:       1000 x 1000"
    )

    print(
        "  Reference:    1000 x 1000"
    )

    print(
        "  Sampling:     10x"
    )

    print()
    print(
        "Loading images..."
    )

    search = load_image(
        args.search
    )

    reference = load_reference(
        args.reference
    )

    print(
        "Search shape:",
        search.shape,
    )

    print(
        "Reference shape:",
        reference.shape,
    )

    start = time.perf_counter()

    result = localize(
        search,
        reference,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("=" * 76)
    print(
        "RESULT"
    )
    print("=" * 76)

    print(
        f"Predicted center: "
        f"({result['predicted_x']:.3f}, "
        f"{result['predicted_y']:.3f}) px"
    )

    print(
        f"Model confidence: "
        f"{result['confidence']:.6f}"
    )

    print(
        f"Candidates: "
        f"{result['candidate_count']}"
    )

    print(
        f"Localization runtime: "
        f"{elapsed * 1000.0:.2f} ms"
    )

    print()
    print(
        "TOP 10 CANDIDATES"
    )

    print("-" * 76)

    for candidate in result[
        "top_candidates"
    ]:

        print(
            f"{candidate['final_rank']:3d} "
            f"x={candidate['x']:8.3f} "
            f"y={candidate['y']:8.3f} "
            f"prob={candidate['probability']:.6f} "
            f"DOG={candidate['dog_score']:.6f}"
        )

    print()
    print("=" * 76)


if __name__ == "__main__":
    main()