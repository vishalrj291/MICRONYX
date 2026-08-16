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

x, y = metadata["target_center_xy"]

x = int(x)
y = int(y)

# Approximate search-space equivalent of the 100x100
# reference under a nominal 10x magnification.
half_size = 5

x1 = x - half_size
y1 = y - half_size
x2 = x + half_size
y2 = y + half_size

target_patch = search[y1:y2, x1:x2]

reference_small = cv2.resize(
    reference,
    (10, 10),
    interpolation=cv2.INTER_AREA,
)

# Enlarge both for human inspection.
target_large = cv2.resize(
    target_patch,
    (300, 300),
    interpolation=cv2.INTER_NEAREST,
)

reference_large = cv2.resize(
    reference_small,
    (300, 300),
    interpolation=cv2.INTER_NEAREST,
)

output_dir = PROJECT_DIR / "validation"
output_dir.mkdir(exist_ok=True)

cv2.imwrite(
    str(output_dir / "gt_search_patch.png"),
    target_large,
)

cv2.imwrite(
    str(output_dir / "reference_downscaled.png"),
    reference_large,
)

print("Ground truth:", (x, y))
print("Target patch:", target_patch.shape)
print("Saved:")
print("  validation/gt_search_patch.png")
print("  validation/reference_downscaled.png")