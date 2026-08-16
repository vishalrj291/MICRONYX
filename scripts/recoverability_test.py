from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX STEP 6
# Clean-image recoverability test
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


# ============================================================
# Ground truth
# ============================================================

GT_X = 376.25
GT_Y = 568.75


# ============================================================
# Magnification
# ============================================================

MAGNIFICATION = 10.0


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
        f"Could not load search image:\n{SEARCH_PATH}"
    )

if reference is None:
    raise FileNotFoundError(
        f"Could not load reference image:\n{REFERENCE_PATH}"
    )


# ============================================================
# Validate dimensions
# ============================================================

if search.shape != (1000, 1000):
    raise ValueError(
        f"Unexpected search shape: {search.shape}"
    )

if reference.shape != (100, 100):
    raise ValueError(
        f"Unexpected reference shape: {reference.shape}"
    )


# ============================================================
# Convert reference to search-image scale
# ============================================================

template_width = round(
    reference.shape[1]
    / MAGNIFICATION
)

template_height = round(
    reference.shape[0]
    / MAGNIFICATION
)

template = cv2.resize(
    reference,
    (
        template_width,
        template_height,
    ),
    interpolation=cv2.INTER_AREA,
)


# ============================================================
# Ground-truth template top-left
# ============================================================

gt_left = (
    GT_X
    - template_width / 2.0
)

gt_top = (
    GT_Y
    - template_height / 2.0
)


# ============================================================
# Integer ground-truth location
# ============================================================

gt_left_int = int(
    round(gt_left)
)

gt_top_int = int(
    round(gt_top)
)


# ============================================================
# Extract GT search patch
# ============================================================

gt_patch = search[
    gt_top_int:
        gt_top_int + template_height,
    gt_left_int:
        gt_left_int + template_width,
]


# ============================================================
# Check dimensions
# ============================================================

if (
    gt_patch.shape
    != template.shape
):
    raise RuntimeError(
        "Ground-truth patch size does not "
        "match template size."
    )


# ============================================================
# Template matching
# ============================================================

result = cv2.matchTemplate(
    search,
    template,
    cv2.TM_CCOEFF_NORMED,
)


# ============================================================
# Best match
# ============================================================

_, best_score, _, best_location = (
    cv2.minMaxLoc(result)
)


best_left = best_location[0]
best_top = best_location[1]


best_x = (
    best_left
    + template_width / 2.0
)

best_y = (
    best_top
    + template_height / 2.0
)


# ============================================================
# GT score
# ============================================================

gt_score = result[
    gt_top_int,
    gt_left_int,
]


# ============================================================
# Localization error
# ============================================================

error = np.sqrt(
    (
        best_x - GT_X
    ) ** 2
    +
    (
        best_y - GT_Y
    ) ** 2
)


# ============================================================
# Ground-truth rank
# ============================================================

flat_result = result.ravel()

gt_flat_index = (
    gt_top_int
    * result.shape[1]
    +
    gt_left_int
)

gt_value = flat_result[
    gt_flat_index
]

rank = (
    1
    +
    np.sum(
        flat_result > gt_value
    )
)


# ============================================================
# Local score statistics
# ============================================================

local_radius = 10

y1 = max(
    0,
    gt_top_int - local_radius,
)

y2 = min(
    result.shape[0],
    gt_top_int + local_radius + 1,
)

x1 = max(
    0,
    gt_left_int - local_radius,
)

x2 = min(
    result.shape[1],
    gt_left_int + local_radius + 1,
)

local_region = result[
    y1:y2,
    x1:x2,
]

local_min = float(
    local_region.min()
)

local_max = float(
    local_region.max()
)

local_mean = float(
    local_region.mean()
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 64)
print("MICRONYX CLEAN-IMAGE RECOVERABILITY TEST")
print("=" * 64)

print()

print("INPUTS")
print("-" * 64)

print(
    f"Search:                {search.shape}"
)

print(
    f"Reference:             {reference.shape}"
)

print(
    f"Downsampled template:  {template.shape}"
)

print()

print("GROUND TRUTH")
print("-" * 64)

print(
    f"GT center:             "
    f"({GT_X:.2f}, {GT_Y:.2f})"
)

print(
    f"GT template top-left:  "
    f"({gt_left:.2f}, {gt_top:.2f})"
)

print(
    f"Integer GT top-left:   "
    f"({gt_left_int}, {gt_top_int})"
)

print()

print("MATCHING")
print("-" * 64)

print(
    f"GT score:              "
    f"{gt_score:.6f}"
)

print(
    f"Best score:            "
    f"{best_score:.6f}"
)

print(
    f"Best template location:"
    f" ({best_left}, {best_top})"
)

print(
    f"Predicted center:      "
    f"({best_x:.2f}, {best_y:.2f})"
)

print()

print("LOCALIZATION")
print("-" * 64)

print(
    f"Euclidean error:       "
    f"{error:.4f} px"
)

print(
    f"Ground-truth rank:     "
    f"{rank:,}"
)

print()

print("LOCAL SCORE REGION")
print("-" * 64)

print(
    f"Min:                   "
    f"{local_min:.6f}"
)

print(
    f"Mean:                  "
    f"{local_mean:.6f}"
)

print(
    f"Max:                   "
    f"{local_max:.6f}"
)

print()

print("=" * 64)

if error <= 5:
    print("RESULT: RECOVERABLE @ 5 px")
else:
    print("RESULT: NOT RECOVERABLE @ 5 px")

print("=" * 64)
print()