from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "dataset_v0.1"
)

DRAM_DIR = DATASET_DIR / "train" / "DRAM"


def localization_error(pred_x, pred_y, gt_x, gt_y):
    return np.sqrt(
        (pred_x - gt_x) ** 2
        + (pred_y - gt_y) ** 2
    )


samples = sorted(DRAM_DIR.iterdir())[:10]

correct = 0

print(f"Testing {len(samples)} DRAM samples\n")

for sample_dir in samples:

    search_path = sample_dir / "search.png"
    reference_path = sample_dir / "reference.png"
    metadata_path = sample_dir / "metadata.json"

    search = cv2.imread(
        str(search_path),
        cv2.IMREAD_GRAYSCALE
    )

    reference = cv2.imread(
        str(reference_path),
        cv2.IMREAD_GRAYSCALE
    )

    import json

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    gt_x, gt_y = metadata["target_center_xy"]

    reference_small = cv2.resize(
        reference,
        (10, 10),
        interpolation=cv2.INTER_AREA
    )

    result = cv2.matchTemplate(
        search,
        reference_small,
        cv2.TM_CCOEFF_NORMED
    )

    _, best_score, _, best_location = cv2.minMaxLoc(result)

    pred_x = best_location[0] + 5
    pred_y = best_location[1] + 5

    error = localization_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    gt_score = result[
        int(gt_y - 5),
        int(gt_x - 5)
    ]

    is_correct = error <= 5

    if is_correct:
        correct += 1

    print(
        f"{sample_dir.name:12s} "
        f"GT=({gt_x},{gt_y}) "
        f"Pred=({pred_x},{pred_y}) "
        f"Error={error:7.2f}px "
        f"GT_score={gt_score:.4f} "
        f"Best={best_score:.4f} "
        f"{'✓' if is_correct else '✗'}"
    )


accuracy = correct / len(samples) * 100

print("\n--------------------------------")
print(f"Accuracy @ 5px: {accuracy:.2f}%")
print("--------------------------------")