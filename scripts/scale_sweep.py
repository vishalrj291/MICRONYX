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

target_x, target_y = metadata["target_center_xy"]

target_x = int(target_x)
target_y = int(target_y)


candidate_sizes = [
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    75,
    100,
]


print("Scale sweep at ground-truth location")
print("--------------------------------------")

results = []


for size in candidate_sizes:

    half = size // 2

    # Make sure crop stays inside image.
    if (
        target_x - half < 0
        or target_y - half < 0
        or target_x + half > search.shape[1]
        or target_y + half > search.shape[0]
    ):
        continue

    search_patch = search[
        target_y - half: target_y + half,
        target_x - half: target_x + half,
    ]

    # Resize reference to candidate search-space size.
    reference_resized = cv2.resize(
        reference,
        (search_patch.shape[1], search_patch.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    # Raw intensity correlation.
    raw_a = search_patch.astype(np.float32).flatten()
    raw_b = reference_resized.astype(np.float32).flatten()

    raw_corr = np.corrcoef(raw_a, raw_b)[0, 1]

    # Structural correlation using gradients.
    search_edges = cv2.Laplacian(
        search_patch,
        cv2.CV_32F,
    )

    reference_edges = cv2.Laplacian(
        reference_resized,
        cv2.CV_32F,
    )

    edge_a = search_edges.flatten()
    edge_b = reference_edges.flatten()

    edge_corr = np.corrcoef(edge_a, edge_b)[0, 1]

    results.append(
        (size, raw_corr, edge_corr)
    )

    print(
        f"Size={size:3d}x{size:<3d} "
        f"Raw={raw_corr: .4f} "
        f"Structural={edge_corr: .4f}"
    )


print("\nBest raw correlation:")
best_raw = max(results, key=lambda x: x[1])
print(
    f"Size={best_raw[0]} "
    f"Raw={best_raw[1]:.4f}"
)

print("\nBest structural correlation:")
best_edge = max(results, key=lambda x: x[2])
print(
    f"Size={best_edge[0]} "
    f"Structural={best_edge[2]:.4f}"
)