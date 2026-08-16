from pathlib import Path
import json

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent

sample_dir = (
    PROJECT_DIR
    / "dataset_v0.1"
    / "train"
    / "DRAM"
    / "DRAM_00001"
)

output_dir = PROJECT_DIR / "validation"
output_dir.mkdir(exist_ok=True)

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

gt_x, gt_y = map(
    int,
    metadata["target_center_xy"],
)

candidate_x = 812
candidate_y = 296


def extract_patch(image, x, y, size=10):
    half = size // 2

    return image[
        y - half:y + half,
        x - half:x + half,
    ]


gt_patch = extract_patch(
    search,
    gt_x,
    gt_y,
)

candidate_patch = extract_patch(
    search,
    candidate_x,
    candidate_y,
)

reference_small = cv2.resize(
    reference,
    (10, 10),
    interpolation=cv2.INTER_AREA,
)


def enlarge(image):
    return cv2.resize(
        image,
        (300, 300),
        interpolation=cv2.INTER_NEAREST,
    )


cv2.imwrite(
    str(output_dir / "compare_reference.png"),
    enlarge(reference_small),
)

cv2.imwrite(
    str(output_dir / "compare_gt.png"),
    enlarge(gt_patch),
)

cv2.imwrite(
    str(output_dir / "compare_candidate.png"),
    enlarge(candidate_patch),
)

print("Saved:")
print("  validation/compare_reference.png")
print("  validation/compare_gt.png")
print("  validation/compare_candidate.png")