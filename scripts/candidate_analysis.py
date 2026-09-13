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


# --------------------------------------------------
# Reference -> nominal 10x search-space representation
# --------------------------------------------------

reference_small = cv2.resize(
    reference,
    (10, 10),
    interpolation=cv2.INTER_AREA,
)


# --------------------------------------------------
# Gradient magnitude
# --------------------------------------------------

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


# --------------------------------------------------
# Template matching
# --------------------------------------------------

result = cv2.matchTemplate(
    search_gradient,
    reference_gradient,
    cv2.TM_CCOEFF_NORMED,
)


# --------------------------------------------------
# Ground-truth score
# --------------------------------------------------

gt_top_left_x = gt_x - 5
gt_top_left_y = gt_y - 5

gt_score = result[
    gt_top_left_y,
    gt_top_left_x,
]


# --------------------------------------------------
# Find distinct local maxima
# --------------------------------------------------

work = result.copy()

candidates = []

suppression_radius = 25

for _ in range(20):

    _, score, _, location = cv2.minMaxLoc(work)

    if score <= -1:
        break

    x = location[0] + 5
    y = location[1] + 5

    candidates.append(
        {
            "x": x,
            "y": y,
            "score": float(score),
        }
    )

    x1 = max(
        0,
        location[0] - suppression_radius,
    )

    y1 = max(
        0,
        location[1] - suppression_radius,
    )

    x2 = min(
        work.shape[1],
        location[0] + suppression_radius + 1,
    )

    y2 = min(
        work.shape[0],
        location[1] + suppression_radius + 1,
    )

    work[y1:y2, x1:x2] = -1


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("Top structural candidates")
print("-------------------------")

for i, candidate in enumerate(candidates, start=1):

    dx = candidate["x"] - gt_x
    dy = candidate["y"] - gt_y

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    print(
        f"{i:2d}. "
        f"({candidate['x']:3d}, {candidate['y']:3d}) "
        f"score={candidate['score']:.4f} "
        f"distance_from_GT={distance:7.2f}px"
    )


# --------------------------------------------------
# GT rank
# --------------------------------------------------

rank = 1 + int(
    np.sum(result > gt_score)
)

print("\nGround truth")
print("------------")
print(f"GT location: ({gt_x}, {gt_y})")
print(f"GT score:    {gt_score:.4f}")
print(f"GT raw rank: {rank}")

print("\nBest-vs-GT gap")
print("-------------")

best_score = candidates[0]["score"]

print(
    f"Best score: {best_score:.4f}"
)

print(
    f"GT score:   {gt_score:.4f}"
)

print(
    f"Score gap:  {best_score - gt_score:.4f}"
)