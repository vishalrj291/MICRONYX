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


# -----------------------------------------
# Nominal 10x representation
# -----------------------------------------

reference_small = cv2.resize(
    reference,
    (10, 10),
    interpolation=cv2.INTER_AREA,
)


# -----------------------------------------
# Gradient magnitude
# -----------------------------------------

ref_x = cv2.Sobel(
    reference_small,
    cv2.CV_32F,
    1,
    0,
    ksize=3,
)

ref_y = cv2.Sobel(
    reference_small,
    cv2.CV_32F,
    0,
    1,
    ksize=3,
)

reference_gradient = cv2.magnitude(
    ref_x,
    ref_y,
)


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


# -----------------------------------------
# Full matching map
# -----------------------------------------

result = cv2.matchTemplate(
    search_gradient,
    reference_gradient,
    cv2.TM_CCOEFF_NORMED,
)


# -----------------------------------------
# Exact GT score
# -----------------------------------------

gt_top_left_x = gt_x - 5
gt_top_left_y = gt_y - 5

gt_score = result[
    gt_top_left_y,
    gt_top_left_x,
]


# -----------------------------------------
# Search only within ±100 pixels of GT
# -----------------------------------------

radius = 100

x1 = max(0, gt_top_left_x - radius)
y1 = max(0, gt_top_left_y - radius)

x2 = min(
    result.shape[1],
    gt_top_left_x + radius + 1,
)

y2 = min(
    result.shape[0],
    gt_top_left_y + radius + 1,
)

local_region = result[y1:y2, x1:x2]

_, local_best_score, _, local_location = cv2.minMaxLoc(
    local_region
)

local_x = x1 + local_location[0] + 5
local_y = y1 + local_location[1] + 5

local_distance = np.sqrt(
    (local_x - gt_x) ** 2
    + (local_y - gt_y) ** 2
)


print("Local ground-truth analysis")
print("---------------------------")

print(
    f"Ground truth: ({gt_x}, {gt_y})"
)

print(
    f"GT score: {gt_score:.4f}"
)

print(
    f"Best candidate within ±{radius}px: "
    f"({local_x}, {local_y})"
)

print(
    f"Local best score: {local_best_score:.4f}"
)

print(
    f"Distance from GT: {local_distance:.2f}px"
)

print(
    f"Local improvement over GT: "
    f"{local_best_score - gt_score:.4f}"
)