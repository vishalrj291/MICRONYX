from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX — STEP 8
# Context-Aware Candidate Ranking
#
# Hypothesis:
# A local 10x10 match can be ambiguous because of periodicity.
# A larger surrounding context may distinguish candidates.
#
# IMPORTANT:
# This is an experiment only.
# It does NOT modify the dataset generator.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

V02_DIR = PROJECT_DIR / "validation" / "v02"

SEARCH_PATH = V02_DIR / "hard_negative_search.png"
REFERENCE_PATH = V02_DIR / "clean_reference.png"

OUTPUT_PATH = V02_DIR / "context_ranking_results.txt"
VISUAL_PATH = V02_DIR / "context_ranking_visualization.png"


# ============================================================
# GROUND TRUTH
# ============================================================

GT_X = 376.25
GT_Y = 568.75

MAGNIFICATION = 10.0


# ============================================================
# CONTEXT SIZES
# ============================================================

# These are search-image dimensions.

CONTEXT_SIZES = [
    10,
    20,
    40,
    80,
    120,
]


# ============================================================
# NUMBER OF CANDIDATES
# ============================================================

TOP_K = 20


# ============================================================
# LOAD IMAGES
# ============================================================

search = cv2.imread(
    str(SEARCH_PATH),
    cv2.IMREAD_GRAYSCALE,
)

reference = cv2.imread(
    str(REFERENCE_PATH),
    cv2.IMREAD_GRAYSCALE,
)

if search is None:
    raise FileNotFoundError(
        f"Could not load:\n{SEARCH_PATH}"
    )

if reference is None:
    raise FileNotFoundError(
        f"Could not load:\n{REFERENCE_PATH}"
    )


# ============================================================
# DOWN-SCALE REFERENCE TO SEARCH RESOLUTION
# ============================================================

template_size = int(
    round(
        reference.shape[0]
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


# ============================================================
# BASE TEMPLATE MATCHING
# ============================================================

result = cv2.matchTemplate(
    search,
    template,
    cv2.TM_CCOEFF_NORMED,
)


# ============================================================
# GET TOP CANDIDATES
# ============================================================

result_copy = result.copy()

candidate_locations = []

for _ in range(TOP_K):

    _, score, _, location = cv2.minMaxLoc(
        result_copy
    )

    x = (
        location[0]
        + template_size / 2.0
    )

    y = (
        location[1]
        + template_size / 2.0
    )

    candidate_locations.append(
        {
            "x": float(x),
            "y": float(y),
            "local_score": float(score),
        }
    )

    # Suppress neighborhood around selected candidate.
    radius = max(
        5,
        template_size,
    )

    x1 = max(
        0,
        location[0] - radius,
    )

    y1 = max(
        0,
        location[1] - radius,
    )

    x2 = min(
        result_copy.shape[1],
        location[0] + radius + 1,
    )

    y2 = min(
        result_copy.shape[0],
        location[1] + radius + 1,
    )

    result_copy[
        y1:y2,
        x1:x2
    ] = -1.0


# ============================================================
# DISTANCE TO GT
# ============================================================

for candidate in candidate_locations:

    dx = (
        candidate["x"]
        - GT_X
    )

    dy = (
        candidate["y"]
        - GT_Y
    )

    candidate["distance"] = float(
        np.sqrt(
            dx * dx
            +
            dy * dy
        )
    )


# ============================================================
# EXTRACT NORMALIZED CONTEXT
# ============================================================

def extract_patch(
    image,
    center_x,
    center_y,
    size,
):
    """
    Extract a square patch centered at x,y.

    If the patch touches an image boundary, pad using
    reflection.
    """

    half = size // 2

    cx = int(
        round(center_x)
    )

    cy = int(
        round(center_y)
    )

    x1 = cx - half
    y1 = cy - half

    x2 = x1 + size
    y2 = y1 + size

    pad_left = max(
        0,
        -x1,
    )

    pad_top = max(
        0,
        -y1,
    )

    pad_right = max(
        0,
        x2 - image.shape[1],
    )

    pad_bottom = max(
        0,
        y2 - image.shape[0],
    )

    if (
        pad_left
        or pad_top
        or pad_right
        or pad_bottom
    ):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT,
        )

        x1 += pad_left
        x2 += pad_left

        y1 += pad_top
        y2 += pad_top

    patch = image[
        y1:y2,
        x1:x2,
    ]

    return patch


# ============================================================
# REFERENCE CONTEXT GENERATION
# ============================================================

def create_reference_context(
    reference,
    size,
):
    """
    The reference is 100x100.

    For a search context larger than 10x10, the reference
    itself cannot magically contain additional physical context.

    Therefore we first represent the reference at its native
    high-magnification scale and resize it to the corresponding
    search-scale context.

    This keeps the comparison geometrically consistent.
    """

    # Physical extent represented by search context.
    #
    # At 10x magnification, reference image contains
    # 10 search pixels worth of spatial extent.

    reference_pixels_needed = int(
        round(
            size
            * MAGNIFICATION
        )
    )

    # The current reference only covers 100x100 pixels.
    #
    # If requested context is larger than 10 search pixels,
    # the available reference does not contain that much
    # surrounding physical context.

    if (
        reference_pixels_needed
        <= reference.shape[0]
    ):

        center = (
            reference.shape[0] // 2
        )

        half = (
            reference_pixels_needed // 2
        )

        crop = reference[
            center - half:
            center + half,
            center - half:
            center + half,
        ]

        context = cv2.resize(
            crop,
            (
                size,
                size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        return context

    # --------------------------------------------------------
    # IMPORTANT:
    # For context > 10 search pixels, the current reference
    # does not contain enough field of view.
    #
    # We therefore return None instead of inventing context.
    # --------------------------------------------------------

    return None


# ============================================================
# CONTEXT COMPARISON
# ============================================================

def context_score(
    search_patch,
    reference_context,
):
    """
    Compare two context patches using normalized
    correlation after mean/std normalization.
    """

    search_float = (
        search_patch.astype(
            np.float32
        )
    )

    reference_float = (
        reference_context.astype(
            np.float32
        )
    )

    search_std = float(
        search_float.std()
    )

    reference_std = float(
        reference_float.std()
    )

    if (
        search_std < 1e-8
        or
        reference_std < 1e-8
    ):
        return 0.0

    search_norm = (
        search_float
        - search_float.mean()
    ) / search_std

    reference_norm = (
        reference_float
        - reference_float.mean()
    ) / reference_std

    score = float(
        np.mean(
            search_norm
            * reference_norm
        )
    )

    return score


# ============================================================
# RUN CONTEXT EXPERIMENT
# ============================================================

print()
print("=" * 72)
print("MICRONYX CONTEXT-AWARE CANDIDATE RANKING")
print("=" * 72)
print()

print(
    f"GT center: ({GT_X:.2f}, {GT_Y:.2f})"
)

print(
    f"Template size: {template_size} × {template_size}"
)

print(
    f"Initial candidates: {TOP_K}"
)

print()


results = []


for context_size in CONTEXT_SIZES:

    print()
    print(
        f"CONTEXT = {context_size} × {context_size}"
    )

    print("-" * 72)

    reference_context = (
        create_reference_context(
            reference,
            context_size,
        )
    )

    if reference_context is None:

        print(
            "SKIPPED"
        )

        print(
            "Reason: reference FOV is too small "
            "for this context size."
        )

        results.append(
            {
                "context": context_size,
                "status": "SKIPPED",
            }
        )

        continue

    scored = []

    for candidate in candidate_locations:

        patch = extract_patch(
            search,
            candidate["x"],
            candidate["y"],
            context_size,
        )

        score = context_score(
            patch,
            reference_context,
        )

        item = candidate.copy()

        item["context_score"] = score

        scored.append(
            item
        )

    scored.sort(
        key=lambda item:
            item["context_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Find GT candidate rank.
    # --------------------------------------------------------

    gt_index = None

    for index, candidate in enumerate(
        scored
    ):

        if (
            candidate["distance"]
            < 1.0
        ):
            gt_index = index
            break

    if gt_index is None:

        gt_rank = None

    else:

        gt_rank = (
            gt_index + 1
        )

    # --------------------------------------------------------
    # Print top candidates.
    # --------------------------------------------------------

    for index, candidate in enumerate(
        scored[:10],
        start=1,
    ):

        print(
            f"{index:2d}. "
            f"({candidate['x']:7.2f}, "
            f"{candidate['y']:7.2f}) "
            f"local={candidate['local_score']:.6f} "
            f"context={candidate['context_score']:.6f} "
            f"dist={candidate['distance']:.2f}"
        )

    print()

    print(
        f"GT rank using context: "
        f"{gt_rank}"
    )

    results.append(
        {
            "context": context_size,
            "status": "OK",
            "gt_rank": gt_rank,
            "top_candidate": scored[0],
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "MICRONYX CONTEXT-AWARE RANKING\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        f"Ground truth: "
        f"({GT_X}, {GT_Y})\n"
    )

    file.write(
        f"Top K: {TOP_K}\n\n"
    )

    for result_item in results:

        file.write(
            f"Context: "
            f"{result_item['context']}x"
            f"{result_item['context']}\n"
        )

        file.write(
            f"Status: "
            f"{result_item['status']}\n"
        )

        if (
            result_item["status"]
            == "OK"
        ):

            file.write(
                f"GT rank: "
                f"{result_item['gt_rank']}\n"
            )

            top = (
                result_item[
                    "top_candidate"
                ]
            )

            file.write(
                f"Top candidate: "
                f"({top['x']:.2f}, "
                f"{top['y']:.2f})\n"
            )

            file.write(
                f"Local score: "
                f"{top['local_score']:.6f}\n"
            )

            file.write(
                f"Context score: "
                f"{top['context_score']:.6f}\n"
            )

        file.write("\n")


# ============================================================
# VISUALIZATION
# ============================================================

visual = cv2.cvtColor(
    search,
    cv2.COLOR_GRAY2BGR,
)


# Ground truth

cv2.drawMarker(
    visual,
    (
        int(round(GT_X)),
        int(round(GT_Y)),
    ),
    (0, 255, 0),
    cv2.MARKER_CROSS,
    30,
    3,
)

cv2.putText(
    visual,
    "GT",
    (
        int(round(GT_X)) + 15,
        int(round(GT_Y)) - 15,
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2,
)


# Draw top candidates from local matching

for index, candidate in enumerate(
    candidate_locations[:10],
    start=1,
):

    x = int(
        round(candidate["x"])
    )

    y = int(
        round(candidate["y"])
    )

    if (
        abs(
            candidate["x"]
            - GT_X
        )
        < 1.0
    ):
        continue

    cv2.circle(
        visual,
        (x, y),
        8,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        visual,
        str(index),
        (
            x + 10,
            y + 5,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )


cv2.imwrite(
    str(VISUAL_PATH),
    visual,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 72)

print(
    f"Results saved:\n{OUTPUT_PATH}"
)

print(
    f"Visualization saved:\n{VISUAL_PATH}"
)

print("=" * 72)
print()