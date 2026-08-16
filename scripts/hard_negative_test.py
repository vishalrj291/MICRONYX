from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX — STEP 7
# Periodic Hard-Negative Experiment
#
# Purpose:
#   Test whether a local appearance matcher can distinguish
#   the true target from highly similar nearby candidates.
#
# IMPORTANT:
#   This is an EXPERIMENT, not part of the final generator.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

V02_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
)

SEARCH_PATH = (
    V02_DIR
    / "clean_search.png"
)

REFERENCE_PATH = (
    V02_DIR
    / "clean_reference.png"
)

OUTPUT_PATH = (
    V02_DIR
    / "hard_negative_search.png"
)

VISUAL_PATH = (
    V02_DIR
    / "hard_negative_candidates.png"
)


# ============================================================
# GROUND TRUTH
# ============================================================

GT_X = 376.25
GT_Y = 568.75

MAGNIFICATION = 10.0


# ============================================================
# HARD NEGATIVE LOCATIONS
# ============================================================
#
# These locations are deliberately far enough from the target
# that they represent independent candidate regions.
#
# We will insert near-identical local structures there.
#
# Format:
#     (x, y)
#
# Coordinates are SEARCH IMAGE coordinates.
# ============================================================

HARD_NEGATIVES = [
    (180, 180),
    (780, 180),
    (180, 780),
    (780, 780),
]


# ============================================================
# Load images
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
# Convert reference to search scale
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
# Ground-truth template location
# ============================================================

gt_left = int(
    round(
        GT_X
        - template_size / 2.0
    )
)

gt_top = int(
    round(
        GT_Y
        - template_size / 2.0
    )
)


# ============================================================
# Create experiment image
# ============================================================

hard_search = search.copy()


# ============================================================
# Insert near-identical hard negatives
# ============================================================

print()
print("=" * 68)
print("MICRONYX PERIODIC HARD-NEGATIVE EXPERIMENT")
print("=" * 68)
print()

print(
    f"Template size: {template_size} × {template_size}"
)

print(
    f"Ground truth:  ({GT_X:.2f}, {GT_Y:.2f})"
)

print()

print("Creating hard negatives...")
print("-" * 68)


for index, (cx, cy) in enumerate(
    HARD_NEGATIVES,
    start=1,
):

    left = int(
        round(
            cx
            - template_size / 2.0
        )
    )

    top = int(
        round(
            cy
            - template_size / 2.0
        )
    )

    # --------------------------------------------------------
    # Extract the original target-scale template.
    # --------------------------------------------------------

    candidate = template.copy()

    # --------------------------------------------------------
    # Apply a VERY small geometric perturbation.
    #
    # This prevents the decoy from being an exact duplicate.
    #
    # The perturbation is intentionally tiny.
    # --------------------------------------------------------

    shift_x = (
        0.35
        if index % 2 == 0
        else -0.35
    )

    shift_y = (
        -0.25
        if index % 2 == 0
        else 0.25
    )

    matrix = np.float32(
        [
            [1.0, 0.0, shift_x],
            [0.0, 1.0, shift_y],
        ]
    )

    candidate = cv2.warpAffine(
        candidate,
        matrix,
        (
            template_size,
            template_size,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # --------------------------------------------------------
    # Very small intensity perturbation.
    #
    # This makes it a near-match rather than a clone.
    # --------------------------------------------------------

    candidate_float = (
        candidate.astype(np.float32)
        * (
            0.97
            + 0.01 * index
        )
    )

    candidate = np.clip(
        candidate_float,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Insert into search.
    # --------------------------------------------------------

    y2 = top + template_size
    x2 = left + template_size

    if (
        left < 0
        or top < 0
        or x2 > hard_search.shape[1]
        or y2 > hard_search.shape[0]
    ):
        raise RuntimeError(
            f"Hard negative {index} "
            f"falls outside image."
        )

    hard_search[
        top:y2,
        left:x2,
    ] = candidate

    print(
        f"{index}. "
        f"center=({cx},{cy}) "
        f"box=({left},{top})→({x2},{y2})"
    )


# ============================================================
# Save hard-negative search image
# ============================================================

if not cv2.imwrite(
    str(OUTPUT_PATH),
    hard_search,
):
    raise RuntimeError(
        "Failed to save hard-negative search."
    )


# ============================================================
# MATCH REFERENCE AGAINST HARD-NEGATIVE SEARCH
# ============================================================

print()
print("TEMPLATE MATCHING")
print("-" * 68)

result = cv2.matchTemplate(
    hard_search,
    template,
    cv2.TM_CCOEFF_NORMED,
)


# ============================================================
# Score at ground truth
# ============================================================

gt_score = float(
    result[
        gt_top,
        gt_left,
    ]
)


# ============================================================
# Score at each hard negative
# ============================================================

candidate_scores = []

for cx, cy in HARD_NEGATIVES:

    left = int(
        round(
            cx
            - template_size / 2.0
        )
    )

    top = int(
        round(
            cy
            - template_size / 2.0
        )
    )

    score = float(
        result[
            top,
            left,
        ]
    )

    candidate_scores.append(
        (
            score,
            cx,
            cy,
        )
    )


# ============================================================
# Sort hard negatives
# ============================================================

candidate_scores.sort(
    reverse=True
)


# ============================================================
# Global best match
# ============================================================

_, best_score, _, best_location = (
    cv2.minMaxLoc(result)
)

best_left = best_location[0]
best_top = best_location[1]

best_x = (
    best_left
    + template_size / 2.0
)

best_y = (
    best_top
    + template_size / 2.0
)

best_error = float(
    np.sqrt(
        (
            best_x - GT_X
        ) ** 2
        +
        (
            best_y - GT_Y
        ) ** 2
    )
)


# ============================================================
# Rank ground truth
# ============================================================

flat_result = result.ravel()

gt_flat_index = (
    gt_top
    * result.shape[1]
    +
    gt_left
)

gt_value = flat_result[
    gt_flat_index
]

gt_rank = (
    1
    + int(
        np.sum(
            flat_result > gt_value
        )
    )
)


# ============================================================
# Print results
# ============================================================

print()

print(
    f"Ground-truth score: "
    f"{gt_score:.6f}"
)

print()

print("Hard-negative scores:")
print()

for rank, (
    score,
    cx,
    cy,
) in enumerate(
    candidate_scores,
    start=1,
):

    print(
        f"{rank}. "
        f"({cx:4d},{cy:4d}) "
        f"score={score:.6f}"
    )


print()

print(
    f"Global best score:   "
    f"{best_score:.6f}"
)

print(
    f"Global best center:  "
    f"({best_x:.2f}, {best_y:.2f})"
)

print(
    f"Global best error:   "
    f"{best_error:.4f} px"
)

print(
    f"Ground-truth rank:   "
    f"{gt_rank:,}"
)


# ============================================================
# Compare GT vs strongest hard negative
# ============================================================

strongest_negative_score = (
    candidate_scores[0][0]
)

score_gap = (
    gt_score
    - strongest_negative_score
)


print()

print(
    f"GT - strongest negative: "
    f"{score_gap:.6f}"
)


# ============================================================
# Create visualization
# ============================================================

visual = cv2.cvtColor(
    hard_search,
    cv2.COLOR_GRAY2BGR,
)


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

gt_cx = int(
    round(GT_X)
)

gt_cy = int(
    round(GT_Y)
)

cv2.drawMarker(
    visual,
    (
        gt_cx,
        gt_cy,
    ),
    (0, 255, 0),
    cv2.MARKER_CROSS,
    20,
    2,
)

cv2.putText(
    visual,
    "GT",
    (
        gt_cx + 12,
        gt_cy - 12,
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 0),
    2,
)


# ------------------------------------------------------------
# Hard negatives
# ------------------------------------------------------------

for index, (
    cx,
    cy,
) in enumerate(
    HARD_NEGATIVES,
    start=1,
):

    cx_int = int(cx)
    cy_int = int(cy)

    half = template_size // 2

    cv2.rectangle(
        visual,
        (
            cx_int - half,
            cy_int - half,
        ),
        (
            cx_int + half,
            cy_int + half,
        ),
        (0, 0, 255),
        2,
    )

    cv2.putText(
        visual,
        f"HN{index}",
        (
            cx_int + 8,
            cy_int - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )


# ============================================================
# Save visualization
# ============================================================

if not cv2.imwrite(
    str(VISUAL_PATH),
    visual,
):
    raise RuntimeError(
        "Failed to save hard-negative visualization."
    )


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 68)

if (
    best_error
    <= 5.0
):

    print(
        "RESULT: TEMPLATE MATCHING SURVIVED "
        "THE HARD-NEGATIVE TEST."
    )

else:

    print(
        "RESULT: TEMPLATE MATCHING FAILED "
        "THE HARD-NEGATIVE TEST."
    )

print("=" * 68)

print()
print(
    f"Search image:\n{OUTPUT_PATH}"
)

print(
    f"Visualization:\n{VISUAL_PATH}"
)

print()