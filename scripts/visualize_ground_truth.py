from pathlib import Path

import cv2


# ============================================================
# MICRONYX Ground-Truth Visualization
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

OUTPUT_PATH = (
    V02_DIR
    / "target_overlay.png"
)


# ============================================================
# Ground truth from generator_v02.py
# ============================================================

TARGET_X = 376.25
TARGET_Y = 568.75

REFERENCE_WIDTH_IN_SEARCH = 10.0
REFERENCE_HEIGHT_IN_SEARCH = 10.0


# ============================================================
# Load image
# ============================================================

search = cv2.imread(
    str(SEARCH_PATH),
    cv2.IMREAD_GRAYSCALE,
)

if search is None:
    raise FileNotFoundError(
        f"Could not load:\n{SEARCH_PATH}"
    )


# Convert grayscale → BGR
overlay = cv2.cvtColor(
    search,
    cv2.COLOR_GRAY2BGR,
)


# ============================================================
# Calculate reference footprint
# ============================================================

half_width = (
    REFERENCE_WIDTH_IN_SEARCH
    / 2.0
)

half_height = (
    REFERENCE_HEIGHT_IN_SEARCH
    / 2.0
)

x1 = int(round(TARGET_X - half_width))
y1 = int(round(TARGET_Y - half_height))

x2 = int(round(TARGET_X + half_width))
y2 = int(round(TARGET_Y + half_height))


# ============================================================
# Draw reference footprint
# ============================================================

cv2.rectangle(
    overlay,
    (x1, y1),
    (x2, y2),
    (0, 0, 255),
    2,
)


# ============================================================
# Draw exact target center
# ============================================================

cx = int(round(TARGET_X))
cy = int(round(TARGET_Y))

cv2.drawMarker(
    overlay,
    (cx, cy),
    (0, 255, 0),
    markerType=cv2.MARKER_CROSS,
    markerSize=20,
    thickness=2,
)


# ============================================================
# Add coordinate label
# ============================================================

label = (
    f"GT ({TARGET_X:.2f}, {TARGET_Y:.2f})"
)

cv2.putText(
    overlay,
    label,
    (cx + 15, cy - 15),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2,
    cv2.LINE_AA,
)


# ============================================================
# Save
# ============================================================

success = cv2.imwrite(
    str(OUTPUT_PATH),
    overlay,
)

if not success:
    raise RuntimeError(
        "Failed to save overlay."
    )


# ============================================================
# Output
# ============================================================

print()
print("MICRONYX Ground-Truth Visualization")
print("------------------------------------")

print(
    f"Target center: "
    f"({TARGET_X:.2f}, {TARGET_Y:.2f})"
)

print(
    f"Reference footprint: "
    f"{REFERENCE_WIDTH_IN_SEARCH:.1f} × "
    f"{REFERENCE_HEIGHT_IN_SEARCH:.1f} search pixels"
)

print(
    f"Bounding box: "
    f"({x1}, {y1}) → ({x2}, {y2})"
)

print()
print(
    f"Saved:\n{OUTPUT_PATH}"
)
# ============================================================
# Target zoom
# ============================================================

ZOOM_SIZE = 100

zx1 = max(
    0,
    cx - ZOOM_SIZE // 2,
)

zy1 = max(
    0,
    cy - ZOOM_SIZE // 2,
)

zx2 = min(
    search.shape[1],
    cx + ZOOM_SIZE // 2,
)

zy2 = min(
    search.shape[0],
    cy + ZOOM_SIZE // 2,
)

target_zoom = search[
    zy1:zy2,
    zx1:zx2,
]

target_zoom = cv2.resize(
    target_zoom,
    (500, 500),
    interpolation=cv2.INTER_NEAREST,
)

zoom_path = (
    V02_DIR
    / "target_zoom.png"
)

cv2.imwrite(
    str(zoom_path),
    target_zoom,
)

print(
    f"Target zoom:\n{zoom_path}"
)