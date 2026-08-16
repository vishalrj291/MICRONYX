from pathlib import Path
import csv

import cv2
import numpy as np

from canonical_renderer import generate_observation


# ============================================================
# MICRONYX STEP 22
# DOG CANDIDATE GENERATION + CANDIDATE VERIFICATION
#
# Pipeline:
#
#   Search
#      ↓
#   DOG response
#      ↓
#   top-K candidates
#      ↓
#   candidate verification
#      ├── DOG
#      ├── gradient orientation
#      ├── local contrast
#      └── spatial consistency
#      ↓
#   final ranking
#
# IMPORTANT:
# Candidate generation and candidate verification are
# deliberately separated.
# ============================================================


PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "dog_candidate_verification"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "dog_candidate_verification_results.csv"
)


SEEDS = range(
    20260850,
    20260880,
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

TOP_K_VALUES = [
    10,
    25,
    50,
    100,
    250,
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):

    image = image.astype(
        np.float32
    )

    mean = np.mean(image)
    std = np.std(image)

    if std < 1e-8:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    return (
        image - mean
    ) / std


# ============================================================
# DOG
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
# GRADIENT
# ============================================================

def gradient_features(image):

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
# NCC
# ============================================================

def template_match(
    search,
    template,
):

    return cv2.matchTemplate(
        search.astype(
            np.float32
        ),
        template.astype(
            np.float32
        ),
        cv2.TM_CCOEFF_NORMED,
    )


# ============================================================
# TOP K LOCAL MAXIMA
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
            cv2.minMaxLoc(
                work
            )
        )

        x, y = loc

        candidates.append(
            (
                x,
                y,
                float(score),
            )
        )

        # Suppress a neighbourhood so
        # candidates represent distinct
        # hypotheses rather than adjacent
        # pixels from the same peak.

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
# PATCH EXTRACTION
# ============================================================

def safe_patch(
    image,
    x,
    y,
    width,
    height,
):

    h, w = image.shape[:2]

    x0 = int(
        round(
            x - width / 2
        )
    )

    y0 = int(
        round(
            y - height / 2
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
        return None

    return image[
        y0:y1,
        x0:x1
    ]


# ============================================================
# CANDIDATE VERIFICATION
# ============================================================

def verify_candidate(
    search,
    template,
    candidate,
):

    x, y, dog_score = candidate

    # --------------------------------------------------------
    # DOG local score
    # --------------------------------------------------------

    search_dog = dog(
        search
    )

    template_dog = dog(
        template
    )

    dog_response = template_match(
        search_dog,
        template_dog,
    )

    local_dog = float(
        dog_response[y, x]
    )

    # --------------------------------------------------------
    # Gradient information
    # --------------------------------------------------------

    search_gx, search_gy, search_mag, search_ori = (
        gradient_features(
            search
        )
    )

    temp_gx, temp_gy, temp_mag, temp_ori = (
        gradient_features(
            template
        )
    )

    th, tw = template.shape

    patch_gx = safe_patch(
        search_gx,
        x + tw / 2,
        y + th / 2,
        tw,
        th,
    )

    patch_gy = safe_patch(
        search_gy,
        x + tw / 2,
        y + th / 2,
        tw,
        th,
    )

    patch_mag = safe_patch(
        search_mag,
        x + tw / 2,
        y + th / 2,
        tw,
        th,
    )

    patch_ori = safe_patch(
        search_ori,
        x + tw / 2,
        y + th / 2,
        tw,
        th,
    )

    if (
        patch_gx is None
        or patch_gy is None
        or patch_mag is None
        or patch_ori is None
    ):
        return {
            "dog": local_dog,
            "orientation": 0.0,
            "gradient": 0.0,
            "contrast": 0.0,
            "final": local_dog,
        }

    # --------------------------------------------------------
    # Gradient magnitude similarity
    # --------------------------------------------------------

    a = normalize(
        patch_mag
    )

    b = normalize(
        temp_mag
    )

    gradient_score = float(
        np.mean(
            a * b
        )
    )

    gradient_score = np.clip(
        gradient_score,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Orientation consistency
    #
    # cos(theta_search - theta_template)
    # weighted by gradient strength.
    # --------------------------------------------------------

    delta = (
        patch_ori
        - temp_ori
    )

    orientation_similarity = (
        np.cos(
            delta
        )
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
        / np.sum(
            weights
        )
    )

    orientation_score = np.clip(
        orientation_score,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Local contrast similarity
    # --------------------------------------------------------

    patch_local = normalize(
        safe_patch(
            search,
            x + tw / 2,
            y + th / 2,
            tw,
            th,
        )
    )

    template_local = normalize(
        template
    )

    contrast_score = float(
        np.mean(
            patch_local
            * template_local
        )
    )

    contrast_score = np.clip(
        contrast_score,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # FINAL VERIFICATION SCORE
    #
    # DOG gets the highest weight because Step 21 showed
    # that DOG is our strongest current representation.
    #
    # Other features only verify the candidate.
    # --------------------------------------------------------

    final_score = (
        0.60 * local_dog
        + 0.20 * gradient_score
        + 0.15 * orientation_score
        + 0.05 * contrast_score
    )

    return {
        "dog": local_dog,
        "orientation": orientation_score,
        "gradient": gradient_score,
        "contrast": contrast_score,
        "final": float(
            final_score
        ),
    }


# ============================================================
# EVALUATE ONE SCENE
# ============================================================

def run_scene(
    scene_type,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    tx = (
        75.25
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    ty = (
        113.75
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    observation = (
        generate_observation(
            tx,
            ty,
            scene_type,
            seed,
        )
    )

    search = observation[
        "search"
    ]

    template = observation[
        "template"
    ]

    gt_x = observation[
        "gt_x"
    ]

    gt_y = observation[
        "gt_y"
    ]

    # --------------------------------------------------------
    # DOG candidate generation
    # --------------------------------------------------------

    search_dog = dog(
        search
    )

    template_dog = dog(
        template
    )

    response = template_match(
        search_dog,
        template_dog,
    )

    # Need enough candidates for the largest K.
    candidates = top_k_candidates(
        response,
        max(TOP_K_VALUES),
    )

    # --------------------------------------------------------
    # GT candidate rank under DOG
    # --------------------------------------------------------

    gt_score = float(
        response[
            gt_y,
            gt_x
        ]
    )

    dog_rank = (
        1
        + int(
            np.sum(
                response
                > gt_score
            )
        )
    )

    # --------------------------------------------------------
    # Verify candidates
    # --------------------------------------------------------

    verified = []

    for candidate in candidates:

        scores = verify_candidate(
            search,
            template,
            candidate,
        )

        x, y, dog_score = (
            candidate
        )

        distance = float(
            np.hypot(
                x - gt_x,
                y - gt_y,
            )
        )

        verified.append(
            {
                "x": x,
                "y": y,
                "dog_score": dog_score,
                "verify_dog":
                    scores["dog"],
                "gradient":
                    scores["gradient"],
                "orientation":
                    scores["orientation"],
                "contrast":
                    scores["contrast"],
                "final":
                    scores["final"],
                "distance":
                    distance,
            }
        )

    # --------------------------------------------------------
    # Evaluate multiple K
    # --------------------------------------------------------

    output = {
        "scene": scene_type,
        "seed": seed,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "dog_gt_rank": dog_rank,
        "dog_gt_score": gt_score,
    }

    for k in TOP_K_VALUES:

        pool = verified[:k]

        # ----------------------------------------------------
        # Candidate recall
        # ----------------------------------------------------

        recalled = any(
            item["distance"]
            <= 5.0
            for item in pool
        )

        output[
            f"recall_{k}"
        ] = int(
            recalled
        )

        # ----------------------------------------------------
        # Final verification ranking
        # ----------------------------------------------------

        best = max(
            pool,
            key=lambda item:
                item["final"],
        )

        output[
            f"pred_x_{k}"
        ] = best["x"]

        output[
            f"pred_y_{k}"
        ] = best["y"]

        output[
            f"error_{k}"
        ] = best["distance"]

        output[
            f"score_{k}"
        ] = best["final"]

    return output


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results,
    scene_type,
):

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

    print(
        "-" * 76
    )

    for k in TOP_K_VALUES:

        recall = np.mean(
            [
                r[
                    f"recall_{k}"
                ]
                for r in subset
            ]
        ) * 100.0

        errors = np.array(
            [
                r[
                    f"error_{k}"
                ]
                for r in subset
            ]
        )

        top1 = np.mean(
            errors == 0
        ) * 100.0

        within5 = np.mean(
            errors <= 5
        ) * 100.0

        print(
            f"K={k:<5}"
            f"Recall@5px="
            f"{recall:6.2f}% "
            f"Top1="
            f"{top1:6.2f}% "
            f"<=5px="
            f"{within5:6.2f}% "
            f"MedErr="
            f"{np.median(errors):8.3f}px"
        )


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
        "MICRONYX STEP 22"
    )
    print(
        "DOG CANDIDATE GENERATION + VERIFICATION"
    )
    print("=" * 76)

    print()
    print(
        "Candidate generator: DOG"
    )

    print(
        "Candidate verifier:"
    )

    print(
        "  DOG"
    )

    print(
        "  gradient magnitude"
    )

    print(
        "  gradient orientation"
    )

    print(
        "  local contrast"
    )

    print()
    print(
        "K values:",
        TOP_K_VALUES,
    )

    results = []

    total = (
        len(SEEDS)
        * len(SCENE_TYPES)
    )

    counter = 0

    for scene_type in SCENE_TYPES:

        print()
        print(
            f"SCENE: "
            f"{scene_type.upper()}"
        )

        print(
            "-" * 76
        )

        for seed in SEEDS:

            counter += 1

            print(
                f"Scene "
                f"{counter:02d}/{total}"
            )

            result = run_scene(
                scene_type,
                seed,
            )

            results.append(
                result
            )

    # ========================================================
    # SAVE
    # ========================================================

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_CSV,
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
            results
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

        print_summary(
            results,
            scene_type,
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