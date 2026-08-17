from pathlib import Path
import sys
import json
import csv

import numpy as np
import cv2


# ============================================================
# MICRONYX STEP 26
# AUTOMATED EDA & SCENE CHARACTERIZATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
OUT_DIR = ROOT / "validation" / "v02" / "automated_eda"

OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

import canonical_renderer as cr


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

SEEDS = list(range(20260875, 20260905))

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

EPS = 1e-12


# ------------------------------------------------------------
# BASIC STATISTICS
# ------------------------------------------------------------

def safe_float(x):
    return float(np.asarray(x).item())


def intensity_features(img):
    x = img.astype(np.float64) / 255.0

    return {
        "mean": safe_float(np.mean(x)),
        "std": safe_float(np.std(x)),
        "min": safe_float(np.min(x)),
        "max": safe_float(np.max(x)),
        "p01": safe_float(np.percentile(x, 1)),
        "p05": safe_float(np.percentile(x, 5)),
        "p25": safe_float(np.percentile(x, 25)),
        "median": safe_float(np.median(x)),
        "p75": safe_float(np.percentile(x, 75)),
        "p95": safe_float(np.percentile(x, 95)),
        "p99": safe_float(np.percentile(x, 99)),
    }


# ------------------------------------------------------------
# CONTRAST
# ------------------------------------------------------------

def contrast_features(img):
    x = img.astype(np.float64) / 255.0

    mean = np.mean(x)
    std = np.std(x)

    return {
        "global_contrast": safe_float(std),
        "rms_contrast": safe_float(
            np.sqrt(np.mean((x - mean) ** 2))
        ),
        "dynamic_range": safe_float(
            np.max(x) - np.min(x)
        ),
    }


# ------------------------------------------------------------
# GRADIENT FEATURES
# ------------------------------------------------------------

def gradient_features(img):
    x = img.astype(np.float32) / 255.0

    gx = cv2.Sobel(
        x,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        x,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = np.sqrt(gx * gx + gy * gy)

    return {
        "gradient_mean": safe_float(np.mean(magnitude)),
        "gradient_std": safe_float(np.std(magnitude)),
        "gradient_max": safe_float(np.max(magnitude)),
        "gradient_p95": safe_float(
            np.percentile(magnitude, 95)
        ),
    }


# ------------------------------------------------------------
# EDGE DENSITY
# ------------------------------------------------------------

def edge_features(img):
    edges = cv2.Canny(
        img,
        threshold1=50,
        threshold2=150,
    )

    density = np.mean(edges > 0)

    return {
        "edge_density": safe_float(density),
        "edge_pixels": int(np.sum(edges > 0)),
    }


# ------------------------------------------------------------
# LAPLACIAN / HIGH FREQUENCY STRUCTURE
# ------------------------------------------------------------

def laplacian_features(img):
    x = img.astype(np.float32)

    lap = cv2.Laplacian(
        x,
        cv2.CV_32F,
        ksize=3,
    )

    return {
        "laplacian_mean": safe_float(np.mean(lap)),
        "laplacian_std": safe_float(np.std(lap)),
        "laplacian_energy": safe_float(
            np.mean(lap * lap)
        ),
    }


# ------------------------------------------------------------
# FOURIER FEATURES
# ------------------------------------------------------------

def frequency_features(img):
    x = img.astype(np.float64)

    x = x - np.mean(x)

    spectrum = np.fft.fftshift(
        np.fft.fft2(x)
    )

    power = np.abs(spectrum) ** 2

    total_energy = np.sum(power) + EPS

    # Remove DC component
    h, w = power.shape
    cy = h // 2
    cx = w // 2

    power_no_dc = power.copy()
    power_no_dc[
        max(0, cy - 1):cy + 2,
        max(0, cx - 1):cx + 2
    ] = 0

    non_dc_energy = np.sum(power_no_dc)

    # Radial frequency energy
    yy, xx = np.indices(power.shape)

    radius = np.sqrt(
        (yy - cy) ** 2 +
        (xx - cx) ** 2
    )

    max_radius = np.max(radius)

    low_mask = radius <= 0.15 * max_radius
    mid_mask = (
        (radius > 0.15 * max_radius) &
        (radius <= 0.40 * max_radius)
    )
    high_mask = radius > 0.40 * max_radius

    low_energy = np.sum(power[low_mask])
    mid_energy = np.sum(power[mid_mask])
    high_energy = np.sum(power[high_mask])

    # Spectral entropy
    p = power_no_dc / (
        np.sum(power_no_dc) + EPS
    )

    p_nonzero = p[p > EPS]

    spectral_entropy = -np.sum(
        p_nonzero * np.log2(p_nonzero)
    )

    spectral_entropy /= np.log2(
        len(p_nonzero) + EPS
    )

    return {
        "fft_total_energy": safe_float(total_energy),
        "fft_non_dc_energy": safe_float(non_dc_energy),
        "fft_low_ratio": safe_float(
            low_energy / total_energy
        ),
        "fft_mid_ratio": safe_float(
            mid_energy / total_energy
        ),
        "fft_high_ratio": safe_float(
            high_energy / total_energy
        ),
        "spectral_entropy": safe_float(
            spectral_entropy
        ),
    }


# ------------------------------------------------------------
# AUTOCORRELATION / PERIODICITY
# ------------------------------------------------------------

def autocorrelation_features(img):
    x = img.astype(np.float64)

    x -= np.mean(x)

    norm = np.sum(x * x) + EPS

    # Horizontal autocorrelation
    corr_x = []

    max_shift = min(100, x.shape[1] // 2)

    for shift in range(1, max_shift + 1):
        a = x[:, :-shift]
        b = x[:, shift:]

        corr = np.sum(a * b) / norm
        corr_x.append(corr)

    # Vertical autocorrelation
    corr_y = []

    max_shift_y = min(100, x.shape[0] // 2)

    for shift in range(1, max_shift_y + 1):
        a = x[:-shift, :]
        b = x[shift:, :]

        corr = np.sum(a * b) / norm
        corr_y.append(corr)

    corr_x = np.asarray(corr_x)
    corr_y = np.asarray(corr_y)

    return {
        "autocorr_x_max": safe_float(
            np.max(corr_x)
        ),
        "autocorr_y_max": safe_float(
            np.max(corr_y)
        ),
        "autocorr_x_mean": safe_float(
            np.mean(corr_x)
        ),
        "autocorr_y_mean": safe_float(
            np.mean(corr_y)
        ),
        "periodicity_indicator": safe_float(
            max(
                np.max(corr_x),
                np.max(corr_y),
            )
        ),
    }


# ------------------------------------------------------------
# LOCAL TEXTURE COMPLEXITY
# ------------------------------------------------------------

def texture_features(img):
    x = img.astype(np.float32) / 255.0

    mean = cv2.GaussianBlur(
        x,
        (0, 0),
        3.0,
    )

    sq_mean = cv2.GaussianBlur(
        x * x,
        (0, 0),
        3.0,
    )

    local_variance = np.maximum(
        sq_mean - mean * mean,
        0,
    )

    local_std = np.sqrt(
        local_variance
    )

    return {
        "local_std_mean": safe_float(
            np.mean(local_std)
        ),
        "local_std_std": safe_float(
            np.std(local_std)
        ),
        "local_std_p95": safe_float(
            np.percentile(local_std, 95)
        ),
    }


# ------------------------------------------------------------
# CHARACTERIZATION
# ------------------------------------------------------------

def characterize(features):
    periodicity = features[
        "periodicity_indicator"
    ]

    entropy = features[
        "spectral_entropy"
    ]

    edge = features[
        "edge_density"
    ]

    high_freq = features[
        "fft_high_ratio"
    ]

    contrast = features[
        "global_contrast"
    ]

    # IMPORTANT:
    # These are EDA descriptors, NOT final model-selection rules.
    # Step 27 will learn/validate automated selection.

    if periodicity > 0.65:
        structure_class = "strongly_periodic"
    elif periodicity > 0.35:
        structure_class = "structured"
    else:
        structure_class = "weakly_periodic_or_aperiodic"

    if entropy > 0.75:
        spectral_class = "broadband"
    elif entropy > 0.45:
        spectral_class = "mixed_frequency"
    else:
        spectral_class = "concentrated_frequency"

    if edge > 0.15:
        edge_class = "edge_dense"
    elif edge > 0.05:
        edge_class = "moderate_edges"
    else:
        edge_class = "edge_sparse"

    if high_freq > 0.25:
        texture_class = "high_frequency"
    else:
        texture_class = "low_or_mid_frequency"

    if contrast > 0.20:
        contrast_class = "high_contrast"
    elif contrast > 0.08:
        contrast_class = "moderate_contrast"
    else:
        contrast_class = "low_contrast"

    return {
        "structure_class": structure_class,
        "spectral_class": spectral_class,
        "edge_class": edge_class,
        "texture_class": texture_class,
        "contrast_class": contrast_class,
    }


# ------------------------------------------------------------
# SINGLE IMAGE EDA
# ------------------------------------------------------------

def analyze_image(img):
    features = {}

    features.update(
        intensity_features(img)
    )

    features.update(
        contrast_features(img)
    )

    features.update(
        gradient_features(img)
    )

    features.update(
        edge_features(img)
    )

    features.update(
        laplacian_features(img)
    )

    features.update(
        frequency_features(img)
    )

    features.update(
        autocorrelation_features(img)
    )

    features.update(
        texture_features(img)
    )

    features.update(
        characterize(features)
    )

    return features


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 76)
    print("MICRONYX STEP 26")
    print("AUTOMATED EDA & SCENE CHARACTERIZATION")
    print("=" * 76)

    print()
    print("Canonical renderer:")
    print(cr.__file__)

    print()
    print("Scenes:", len(SEEDS) * len(SCENE_TYPES))
    print("Scene types:", SCENE_TYPES)

    rows = []

    total = len(SEEDS) * len(SCENE_TYPES)
    counter = 0

    for scene_type in SCENE_TYPES:

        print()
        print("=" * 76)
        print("SCENE TYPE:", scene_type.upper())
        print("=" * 76)

        for seed in SEEDS:

            counter += 1

            print(
                f"Scene {counter:02d}/{total}"
            )

            tx = 75.25
            ty = 113.75

            search = cr.render_search(
                tx,
                ty,
                scene_type,
                seed,
            )

            reference = cr.render_reference(
                tx,
                ty,
                scene_type,
                seed,
            )

            template = cr.create_ps02_template(
                reference
            )

            search_features = analyze_image(
                search
            )

            reference_features = analyze_image(
                template
            )

            row = {
                "scene_type": scene_type,
                "seed": seed,

                "search_height": search.shape[0],
                "search_width": search.shape[1],

                "reference_height": reference.shape[0],
                "reference_width": reference.shape[1],

                "template_height": template.shape[0],
                "template_width": template.shape[1],
            }

            for key, value in search_features.items():
                row[
                    "search_" + key
                ] = value

            for key, value in reference_features.items():
                row[
                    "template_" + key
                ] = value

            rows.append(row)

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = (
        OUT_DIR /
        "automated_eda_results.csv"
    )

    fieldnames = list(rows[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {}

    for scene_type in SCENE_TYPES:

        subset = [
            r for r in rows
            if r["scene_type"] == scene_type
        ]

        summary[scene_type] = {
            "scenes": len(subset),

            "mean_contrast": float(
                np.mean([
                    r["search_global_contrast"]
                    for r in subset
                ])
            ),

            "mean_gradient": float(
                np.mean([
                    r["search_gradient_mean"]
                    for r in subset
                ])
            ),

            "mean_edge_density": float(
                np.mean([
                    r["search_edge_density"]
                    for r in subset
                ])
            ),

            "mean_spectral_entropy": float(
                np.mean([
                    r["search_spectral_entropy"]
                    for r in subset
                ])
            ),

            "mean_periodicity": float(
                np.mean([
                    r["search_periodicity_indicator"]
                    for r in subset
                ])
            ),

            "mean_high_frequency_ratio": float(
                np.mean([
                    r["search_fft_high_ratio"]
                    for r in subset
                ])
            ),

            "structure_classes": {},
        }

        classes = {}

        for r in subset:

            c = r[
                "search_structure_class"
            ]

            classes[c] = (
                classes.get(c, 0) + 1
            )

        summary[scene_type][
            "structure_classes"
        ] = classes

    json_path = (
        OUT_DIR /
        "automated_eda_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    for scene_type in SCENE_TYPES:

        s = summary[scene_type]

        print()
        print(scene_type.upper())
        print("-" * 76)

        print(
            f"Mean contrast:          "
            f"{s['mean_contrast']:.6f}"
        )

        print(
            f"Mean gradient:          "
            f"{s['mean_gradient']:.6f}"
        )

        print(
            f"Mean edge density:      "
            f"{s['mean_edge_density']:.6f}"
        )

        print(
            f"Mean spectral entropy:  "
            f"{s['mean_spectral_entropy']:.6f}"
        )

        print(
            f"Mean periodicity:       "
            f"{s['mean_periodicity']:.6f}"
        )

        print(
            f"Mean high-frequency:    "
            f"{s['mean_high_frequency_ratio']:.6f}"
        )

        print(
            "Structure classes:"
        )

        for cls, count in (
            s["structure_classes"].items()
        ):

            print(
                f"  {cls:<32}"
                f"{count}"
            )

    print()
    print("=" * 76)
    print("SAVED")
    print("=" * 76)

    print(csv_path)
    print(json_path)

    print()
    print("STEP 26 COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()