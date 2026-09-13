from pathlib import Path

import cv2
import numpy as np


# ============================================================
# MICRONYX — STEP 10
# Frequency-Domain Candidate Analysis
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

V02_DIR = PROJECT_DIR / "validation" / "v02"

SEARCH_PATH = V02_DIR / "natural_periodic_search.png"
REFERENCE_PATH = V02_DIR / "natural_periodic_reference.png"

OUTPUT_PATH = V02_DIR / "frequency_candidate_results.txt"


# ============================================================
# CONFIGURATION
# ============================================================

MAGNIFICATION = 10.0

GT_X = 376.25
GT_Y = 568.75

TOP_K = 30

ALPHAS = [
    0.90,
    0.75,
    0.50,
    0.25,
]


# ============================================================
# LOAD IMAGES
# ============================================================

search = cv2.imread(
    str(SEARCH_PATH),
    cv2.IMREAD_GRAYSCALE,
)

reference = cv2.imread(
    str(REFERENCE_PATH),
    cv2.IMREAD_GRAYSCALE,
)

if search is None:
    raise FileNotFoundError(
        f"Could not load search image:\n{SEARCH_PATH}"
    )

if reference is None:
    raise FileNotFoundError(
        f"Could not load reference image:\n{REFERENCE_PATH}"
    )


# ============================================================
# VALIDATE DIMENSIONS
# ============================================================

if search.shape != (1000, 1000):
    raise ValueError(
        f"Unexpected search shape: {search.shape}"
    )

if reference.shape != (100, 100):
    raise ValueError(
        f"Unexpected reference shape: {reference.shape}"
    )


# ============================================================
# DOWN-SCALE REFERENCE TO SEARCH SCALE
# ============================================================

template_size = int(
    round(
        reference.shape[0] / MAGNIFICATION
    )
)

template = cv2.resize(
    reference,
    (
        template_size,
        template_size,
    ),
    interpolation=cv2.INTER_AREA,
)


# ============================================================
# TEMPLATE MATCHING
# ============================================================

result = cv2.matchTemplate(
    search,
    template,
    cv2.TM_CCOEFF_NORMED,
)


# ============================================================
# GET TOP CANDIDATES
# ============================================================

suppressed = result.copy()

candidates = []

for _ in range(TOP_K):

    _, score, _, location = cv2.minMaxLoc(
        suppressed
    )

    left = int(location[0])
    top = int(location[1])

    x = (
        left
        + template_size / 2.0
    )

    y = (
        top
        + template_size / 2.0
    )

    distance = float(
        np.sqrt(
            (x - GT_X) ** 2
            +
            (y - GT_Y) ** 2
        )
    )

    candidates.append(
        {
            "x": x,
            "y": y,
            "left": left,
            "top": top,
            "spatial": float(score),
            "distance": distance,
        }
    )

    # Suppress a neighborhood so that we get
    # spatially distinct candidates.
    radius = max(
        5,
        template_size,
    )

    x1 = max(
        0,
        left - radius,
    )

    y1 = max(
        0,
        top - radius,
    )

    x2 = min(
        suppressed.shape[1],
        left + radius + 1,
    )

    y2 = min(
        suppressed.shape[0],
        top + radius + 1,
    )

    suppressed[
        y1:y2,
        x1:x2
    ] = -1.0


# ============================================================
# FFT MAGNITUDE
# ============================================================

def fft_magnitude(image):
    """
    Compute normalized log-magnitude FFT.
    """

    image = image.astype(
        np.float32
    )

    # Remove DC component.
    image = (
        image
        - image.mean()
    )

    spectrum = np.fft.fft2(
        image
    )

    spectrum = np.fft.fftshift(
        spectrum
    )

    magnitude = np.abs(
        spectrum
    )

    magnitude = np.log1p(
        magnitude
    )

    norm = np.linalg.norm(
        magnitude
    )

    if norm < 1e-8:
        return magnitude

    return magnitude / norm


# ============================================================
# FREQUENCY SIMILARITY
# ============================================================

def frequency_similarity(
    image_a,
    image_b,
):
    """
    Cosine similarity between FFT magnitude spectra.
    """

    fft_a = fft_magnitude(
        image_a
    )

    fft_b = fft_magnitude(
        image_b
    )

    a = fft_a.ravel()
    b = fft_b.ravel()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if (
        norm_a < 1e-8
        or norm_b < 1e-8
    ):
        return 0.0

    return float(
        np.dot(a, b)
        /
        (
            norm_a
            * norm_b
        )
    )


# ============================================================
# EXTRACT PATCH
# ============================================================

def extract_patch(
    image,
    left,
    top,
    size,
):
    """
    Extract size × size search patch.
    """

    patch = image[
        top:top + size,
        left:left + size,
    ]

    if patch.shape == (
        size,
        size,
    ):
        return patch

    # Boundary handling.
    pad_bottom = max(
        0,
        size - patch.shape[0],
    )

    pad_right = max(
        0,
        size - patch.shape[1],
    )

    patch = cv2.copyMakeBorder(
        patch,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_REFLECT,
    )

    return patch[
        :size,
        :size
    ]


# ============================================================
# REFERENCE FREQUENCY REPRESENTATION
# ============================================================

reference_fft = fft_magnitude(
    template
)


# ============================================================
# CALCULATE FREQUENCY SCORE
# ============================================================

for candidate in candidates:

    patch = extract_patch(
        search,
        candidate["left"],
        candidate["top"],
        template_size,
    )

    candidate["frequency"] = (
        frequency_similarity(
            template,
            patch,
        )
    )


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def minmax(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    minimum = values.min()
    maximum = values.max()

    if (
        maximum - minimum
        < 1e-12
    ):
        return np.ones_like(
            values
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


spatial_values = [
    candidate["spatial"]
    for candidate in candidates
]

frequency_values = [
    candidate["frequency"]
    for candidate in candidates
]

spatial_normalized = minmax(
    spatial_values
)

frequency_normalized = minmax(
    frequency_values
)


for index, candidate in enumerate(
    candidates
):

    candidate["spatial_norm"] = float(
        spatial_normalized[index]
    )

    candidate["frequency_norm"] = float(
        frequency_normalized[index]
    )


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 72)
print(
    "MICRONYX STEP 10"
)
print(
    "FREQUENCY-DOMAIN CANDIDATE ANALYSIS"
)
print("=" * 72)
print()

print(
    f"Search shape:          {search.shape}"
)

print(
    f"Reference shape:       {reference.shape}"
)

print(
    f"Template size:         "
    f"{template_size} × {template_size}"
)

print(
    f"Ground truth:          "
    f"({GT_X:.2f}, {GT_Y:.2f})"
)

print()


# ============================================================
# PRINT CANDIDATES
# ============================================================

print(
    "TOP CANDIDATES"
)

print("-" * 72)

for index, candidate in enumerate(
    candidates[:15],
    start=1,
):

    print(
        f"{index:2d}. "
        f"({candidate['x']:7.2f}, "
        f"{candidate['y']:7.2f}) "
        f"spatial="
        f"{candidate['spatial']:.6f} "
        f"freq="
        f"{candidate['frequency']:.6f} "
        f"dist="
        f"{candidate['distance']:.2f}"
    )


# ============================================================
# GROUND-TRUTH INFORMATION
# ============================================================

gt_candidate_index = min(
    range(len(candidates)),
    key=lambda i:
        candidates[i]["distance"],
)

gt_candidate = candidates[
    gt_candidate_index
]

print()

print(
    "GROUND TRUTH CANDIDATE"
)

print("-" * 72)

print(
    f"Closest candidate:    "
    f"({gt_candidate['x']:.2f}, "
    f"{gt_candidate['y']:.2f})"
)

print(
    f"Distance:             "
    f"{gt_candidate['distance']:.4f}px"
)

print(
    f"Spatial score:        "
    f"{gt_candidate['spatial']:.6f}"
)

print(
    f"Frequency score:      "
    f"{gt_candidate['frequency']:.6f}"
)


# ============================================================
# COMBINED SCORE EXPERIMENT
# ============================================================

print()
print(
    "COMBINED SCORE EXPERIMENT"
)

print("-" * 72)

alpha_results = []

for alpha in ALPHAS:

    combined_key = (
        f"combined_{alpha:.2f}"
    )

    for candidate in candidates:

        candidate[combined_key] = (
            alpha
            * candidate["spatial_norm"]
            +
            (1.0 - alpha)
            * candidate["frequency_norm"]
        )

    ranked = sorted(
        candidates,
        key=lambda candidate:
            candidate[combined_key],
        reverse=True,
    )

    # Find rank of the candidate closest to GT.
    gt_rank = None

    for rank, candidate in enumerate(
        ranked,
        start=1,
    ):

        if (
            candidate["distance"]
            < 1.0
        ):
            gt_rank = rank
            break

    if gt_rank is None:

        # If the exact GT isn't inside top-K,
        # report its nearest candidate rank.
        gt_rank = (
            min(
                range(len(ranked)),
                key=lambda i:
                    ranked[i]["distance"],
            )
            + 1
        )

    best = ranked[0]

    print()

    print(
        f"alpha = {alpha:.2f}"
    )

    print(
        f"  Best candidate: "
        f"({best['x']:.2f}, "
        f"{best['y']:.2f})"
    )

    print(
        f"  Combined score: "
        f"{best[combined_key]:.6f}"
    )

    print(
        f"  Best error:     "
        f"{best['distance']:.4f}px"
    )

    print(
        f"  GT rank:        "
        f"{gt_rank}"
    )

    alpha_results.append(
        {
            "alpha": alpha,
            "gt_rank": gt_rank,
            "best_error": best["distance"],
            "best_x": best["x"],
            "best_y": best["y"],
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "MICRONYX STEP 10\n"
    )

    file.write(
        "Frequency-Domain Candidate Analysis\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        f"Ground truth: "
        f"({GT_X}, {GT_Y})\n"
    )

    file.write(
        f"Template size: "
        f"{template_size}x{template_size}\n\n"
    )

    file.write(
        "CANDIDATES\n"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        file.write(
            f"{index}, "
            f"x={candidate['x']:.4f}, "
            f"y={candidate['y']:.4f}, "
            f"spatial={candidate['spatial']:.8f}, "
            f"frequency={candidate['frequency']:.8f}, "
            f"distance={candidate['distance']:.4f}\n"
        )

    file.write(
        "\nCOMBINED SCORE RESULTS\n"
    )

    for item in alpha_results:

        file.write(
            f"alpha={item['alpha']:.2f}, "
            f"GT_rank={item['gt_rank']}, "
            f"best_error={item['best_error']:.4f}, "
            f"best=({item['best_x']:.2f}, "
            f"{item['best_y']:.2f})\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 72)

print(
    f"Results saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 72)
print()