from pathlib import Path
import cv2
import numpy as np
import json


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

# Approximate 10x relationship
target_patch = search[y - 5:y + 5, x - 5:x + 5]

reference_small = cv2.resize(
    reference,
    (10, 10),
    interpolation=cv2.INTER_AREA,
)

# Remove low-frequency intensity information.
target_edges = cv2.Laplacian(
    target_patch,
    cv2.CV_32F
)

reference_edges = cv2.Laplacian(
    reference_small,
    cv2.CV_32F
)

target_edges = target_edges.flatten()
reference_edges = reference_edges.flatten()

correlation = np.corrcoef(
    target_edges,
    reference_edges
)[0, 1]

print("Structural correlation:", correlation)