from pathlib import Path
import json

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

sample_dir = (
    PROJECT_DIR
    / "dataset_v0.1"
    / "train"
    / "DRAM"
    / "DRAM_00001"
)

search = cv2.imread(
    str(sample_dir / "search.png"),
    cv2.IMREAD_GRAYSCALE,
)

reference = cv2.imread(
    str(sample_dir / "reference.png"),
    cv2.IMREAD_GRAYSCALE,
)

with open(sample_dir / "metadata.json", "r") as f:
    metadata = json.load(f)

gt_x, gt_y = metadata["target_center_xy"]

gt_x = int(gt_x)
gt_y = int(gt_y)


# -------------------------------------------------
# Convert the 100x100 reference to the nominal
# 10x10 search-space representation.
# -------------------------------------------------

reference_small = cv2.resize(
    reference,
    (10, 10),
    interpolation=cv2.INTER_AREA,
)


# -------------------------------------------------
# Compute gradient magnitude.
# -------------------------------------------------

reference_x = cv2.Sobel(
    reference_small,
    cv2.CV_32F,
    1,
    0,
    ksize=3,
)

reference_y = cv2.Sobel(
    reference_small,
    cv2.CV_32F,
    0,
    1,
    ksize=3,
)

reference_gradient = cv2.magnitude(
    reference_x,
    reference_y,
)


# Normalize reference gradient.
reference_gradient -= reference_gradient.mean()

reference_norm = np.linalg.norm(reference_gradient)

if reference_norm > 0:
    reference_gradient /= reference_norm


# -------------------------------------------------
# Search image gradient.
# -------------------------------------------------

search_x = cv2.Sobel(
    search,
    cv2.CV_32F,
    1,
    0,
    ksize=3,
)

search_y = cv2.Sobel(
    search,
    cv2.CV_32F,
    0,
    1,
    ksize=3,
)

search_gradient = cv2.magnitude(
    search_x,
    search_y,
)


# -------------------------------------------------
# Match gradient template across the search image.
# -------------------------------------------------

result = cv2.matchTemplate(
    search_gradient,
    reference_gradient,
    cv2.TM_CCOEFF_NORMED,
)


_, best_score, _, best_location = cv2.minMaxLoc(
    result
)


best_x = best_location[0] + 5
best_y = best_location[1] + 5


# Ground-truth score.
gt_location = (
    gt_x - 5,
    gt_y - 5,
)

gt_score = result[
    gt_location[1],
    gt_location[0],
]


print("Gradient-based matching")
print("-----------------------")

print(
    f"Ground truth: "
    f"({gt_x}, {gt_y})"
)

print(
    f"Ground-truth structural score: "
    f"{gt_score:.4f}"
)

print(
    f"Best structural location: "
    f"({best_x}, {best_y})"
)

print(
    f"Best structural score: "
    f"{best_score:.4f}"
)