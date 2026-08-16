from pathlib import Path
import csv

import cv2
import numpy as np

from canonical_renderer import (
    render_search,
    render_sensor,
)


# ============================================================
# MICRONYX STEP 23
# MULTI-SCALE PHYSICAL CONTEXT VERIFICATION
#
# Candidate generation:
#       DOG
#
# Verification:
#       10 px  = local PS02 template
#       20 px  = 4 × 4 physical-unit context
#       40 px  = 8 × 8 physical-unit context
#
# IMPORTANT:
# Larger references are NOT obtained by enlarging the
# original 100×100 reference.
#
# They are rendered from the SAME continuous physical scene
# using a larger high-magnification FOV.
# ============================================================


PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "multiscale_context"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "multiscale_context_results.csv"
)


SEEDS = range(
    20260850,
    20260880,
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

# Search-scale context sizes.
CONTEXT_SIZES = [
    10,
    20,
    40,
]

# Number of DOG candidates to verify.
K_VALUES = [
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
# TEMPLATE MATCH
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
# TOP-K DOG CANDIDATES
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
# PHYSICAL CONTEXT REFERENCE
# ============================================================

def render_context_reference(
    tx,
    ty,
    scene_type,
    seed,
    context_size,
):
    """
    Create a high-magnification reference whose physical
    FOV corresponds exactly to the requested search-scale
    context.

    Search:
        5 px / physical unit

    Therefore:

        10 search px = 2 physical units
        20 search px = 4 physical units
        40 search px = 8 physical units

    Reference sensor:
        50 px / physical unit

    Thus reference dimensions are:

        10 search px -> 100 reference px
        20 search px -> 200 reference px
        40 search px -> 400 reference px
    """

    physical_size = (
        context_size
        / 5.0
    )

    reference_size = int(
        round(
            physical_size
            * 50.0
        )
    )

    origin_x = (
        tx
        - physical_size / 2.0
    )

    origin_y = (
        ty
        - physical_size / 2.0
    )

    reference = render_sensor(
        width=reference_size,
        height=reference_size,
        pixels_per_unit=50.0,
        origin_x=origin_x,
        origin_y=origin_y,
        tx=tx,
        ty=ty,
        scene_type=scene_type,
        seed=seed,
    )

    # Convert high-magnification reference into the
    # equivalent search-scale representation.
    context = cv2.resize(
        reference,
        (
            context_size,
            context_size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return context


# ============================================================
# CENTERED PATCH FROM SEARCH
# ============================================================

def centered_patch(
    search,
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

    x1 = (
        x0
        + size
    )

    y1 = (
        y0
        + size
    )

    h, w = search.shape

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):
        return None

    return search[
        y0:y1,
        x0:x1
    ]


# ============================================================
# CONTEXT SCORE
# ============================================================

def context_score(
    search,
    reference,
    center_x,
    center_y,
):
    """
    Compare a physically matched search context and
    reference context at identical search-scale dimensions.
    """

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
# MULTI-SCALE VERIFIER
# ============================================================

def verify_candidate(
    search,
    references,
    center_x,
    center_y,
):
    """
    Calculate independent context scores.

    10 px:
        local structural evidence

    20 px:
        medium-range neighborhood

    40 px:
        larger periodic/quasiperiodic context
    """

    scores = {}

    for size in CONTEXT_SIZES:

        score = context_score(
            search,
            references[size],
            center_x,
            center_y,
        )

        scores[
            f"context_{size}"
        ] = score

    # --------------------------------------------------------
    # DOG local score
    # --------------------------------------------------------

    search_dog = dog(
        search
    )

    dog_reference = dog(
        references[10]
    )

    dog_response = template_match(
        search_dog,
        dog_reference,
    )

    # Candidate corresponds to top-left for 10×10.
    dog_x = int(
        round(
            center_x - 5
        )
    )

    dog_y = int(
        round(
            center_y - 5
        )
    )

    h, w = dog_response.shape

    if (
        0 <= dog_x < w
        and 0 <= dog_y < h
    ):
        local_dog = float(
            dog_response[
                dog_y,
                dog_x
            ]
        )
    else:
        local_dog = -1.0

    scores[
        "dog"
    ] = local_dog

    # --------------------------------------------------------
    # Final multi-scale score
    #
    # DOG + local context + medium context + large context
    # --------------------------------------------------------

    final = (
        0.40
        * scores["dog"]
        + 0.30
        * scores["context_20"]
        + 0.20
        * scores["context_40"]
        + 0.10
        * scores["context_10"]
    )

    scores[
        "final"
    ] = float(
        final
    )

    return scores


# ============================================================
# ONE SCENE
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

    # --------------------------------------------------------
    # Canonical search
    # --------------------------------------------------------

    search = render_search(
        tx,
        ty,
        scene_type,
        seed,
    )

    # --------------------------------------------------------
    # Ground truth search coordinate
    # --------------------------------------------------------

    gt_x = (
        tx
        * 5.0
    )

    gt_y = (
        ty
        * 5.0
    )

    # --------------------------------------------------------
    # Render physically correct contexts
    # --------------------------------------------------------

    references = {}

    for size in CONTEXT_SIZES:

        references[size] = (
            render_context_reference(
                tx,
                ty,
                scene_type,
                seed,
                size,
            )
        )

    # --------------------------------------------------------
    # DOG candidate generator
    #
    # Use 10×10 equivalent reference.
    # --------------------------------------------------------

    search_dog = dog(
        search
    )

    reference_dog = dog(
        references[10]
    )

    response = template_match(
        search_dog,
        reference_dog,
    )

    candidates = top_k_candidates(
        response,
        max(K_VALUES),
    )

    # --------------------------------------------------------
    # Verify candidates once
    # --------------------------------------------------------

    verified = []

    for x, y, dog_score in candidates:

        # Candidate location is top-left of the
        # 10×10 template. Convert to center.

        center_x = (
            x + 5.0
        )

        center_y = (
            y + 5.0
        )

        scores = verify_candidate(
            search,
            references,
            center_x,
            center_y,
        )

        distance = float(
            np.hypot(
                center_x - gt_x,
                center_y - gt_y,
            )
        )

        verified.append(
            {
                "x": center_x,
                "y": center_y,
                "dog_generator":
                    dog_score,
                "dog":
                    scores["dog"],
                "context_10":
                    scores["context_10"],
                "context_20":
                    scores["context_20"],
                "context_40":
                    scores["context_40"],
                "final":
                    scores["final"],
                "distance":
                    distance,
            }
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    output = {
        "scene": scene_type,
        "seed": seed,
        "gt_x": gt_x,
        "gt_y": gt_y,
    }

    for k in K_VALUES:

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
        # Final verification
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

    for k in K_VALUES:

        recall = (
            np.mean(
                [
                    r[
                        f"recall_{k}"
                    ]
                    for r in subset
                ]
            )
            * 100.0
        )

        errors = np.array(
            [
                r[
                    f"error_{k}"
                ]
                for r in subset
            ]
        )

        print(
            f"K={k:<5}"
            f"Recall@5px="
            f"{recall:6.2f}% "
            f"Top1="
            f"{np.mean(errors < 1.0) * 100:6.2f}% "
            f"<=5px="
            f"{np.mean(errors <= 5.0) * 100:6.2f}% "
            f"MedErr="
            f"{np.median(errors):8.3f}px "
            f"MaxErr="
            f"{np.max(errors):8.3f}px"
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
        "MICRONYX STEP 23"
    )
    print(
        "MULTI-SCALE PHYSICAL CONTEXT VERIFICATION"
    )
    print("=" * 76)

    print()
    print(
        "Context sizes:"
    )

    print(
        "10 px = 2 physical units"
    )

    print(
        "20 px = 4 physical units"
    )

    print(
        "40 px = 8 physical units"
    )

    print()
    print(
        "Candidate generator: DOG"
    )

    print(
        "Verifier:"
    )

    print(
        "  10 px local context"
    )

    print(
        "  20 px physical context"
    )

    print(
        "  40 px physical context"
    )

    print()
    print(
        "K values:",
        K_VALUES,
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
    # SAVE CSV
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