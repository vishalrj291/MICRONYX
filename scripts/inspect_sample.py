from pathlib import Path

import cv2


sample_dir = (
    Path(__file__).resolve().parent.parent
    / "dataset_v0.1"
    / "train"
    / "DRAM"
    / "DRAM_00001"
)

search_path = sample_dir / "search.png"
reference_path = sample_dir / "reference.png"

search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)

print("Search shape:", search.shape)
print("Reference shape:", reference.shape)

print("\nSearch:")
print("  min:", search.min())
print("  max:", search.max())
print("  mean:", search.mean())
print("  std:", search.std())

print("\nReference:")
print("  min:", reference.min())
print("  max:", reference.max())
print("  mean:", reference.mean())
print("  std:", reference.std())
search_black = (search == 0).mean() * 100
search_white = (search == 255).mean() * 100

reference_black = (reference == 0).mean() * 100
reference_white = (reference == 255).mean() * 100

print("\nSaturation:")
print(f"Search pixels at 0:   {search_black:.2f}%")
print(f"Search pixels at 255: {search_white:.2f}%")

print(f"Reference pixels at 0:   {reference_black:.2f}%")
print(f"Reference pixels at 255: {reference_white:.2f}%")
# -----------------------------------------
# Simple 10x scale template-matching test
# -----------------------------------------

target_x, target_y = 759, 231

# High-magnification reference is approximately
# 10x larger than its low-magnification equivalent.
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

best_x = best_location[0] + 5
best_y = best_location[1] + 5

# Score at ground-truth location
gt_x = target_x - 5
gt_y = target_y - 5

gt_score = result[gt_y, gt_x]

print("\nTemplate Matching:")
print("  Reference after 10x reduction:", reference_small.shape)
print(f"  Ground truth: ({target_x}, {target_y})")
print(f"  Ground-truth score: {gt_score:.4f}")
print(f"  Best predicted location: ({best_x}, {best_y})")
print(f"  Best score: {best_score:.4f}")