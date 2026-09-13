"""
MICRONYX Phase 4A
Automated EDA and Scene Characterization

Usage
-----
From the repository root:

    python scripts/automated_eda.py --dataset dataset_v1

For a canonical-renderer smoke test:

    python scripts/automated_eda.py --canonical-synthetic

Outputs
-------
validation/phase4/automated_eda/
    automated_eda_results.csv
    automated_eda_summary.json
    automated_eda_report.md

Design rules
------------
- EDA never reads target_x/target_y for image features.
- The canonical renderer is used for the synthetic smoke test.
- Thresholds below are descriptive labels only, not model-selection rules.
- All numeric descriptors are deterministic for deterministic input images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
DEFAULT_OUT = ROOT / "validation" / "phase4" / "automated_eda"
EPS = 1e-12

sys.path.insert(0, str(SCRIPT_DIR))


def finite_float(value: float) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"Non-finite EDA value: {value}")

    return value


# ---------------------------------------------------------------------------
# Intensity
# ---------------------------------------------------------------------------

def intensity_features(image: np.ndarray) -> dict:
    x = image.astype(np.float64) / 255.0

    return {
        "mean": finite_float(np.mean(x)),
        "std": finite_float(np.std(x)),
        "min": finite_float(np.min(x)),
        "max": finite_float(np.max(x)),
        "p01": finite_float(np.percentile(x, 1)),
        "p05": finite_float(np.percentile(x, 5)),
        "p25": finite_float(np.percentile(x, 25)),
        "median": finite_float(np.median(x)),
        "p75": finite_float(np.percentile(x, 75)),
        "p95": finite_float(np.percentile(x, 95)),
        "p99": finite_float(np.percentile(x, 99)),
    }


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------

def contrast_features(image: np.ndarray) -> dict:
    x = image.astype(np.float64) / 255.0
    mean = np.mean(x)

    return {
        "global_contrast": finite_float(np.std(x)),
        "dynamic_range": finite_float(np.max(x) - np.min(x)),
        "coefficient_of_variation": finite_float(
            np.std(x) / (abs(mean) + EPS)
        ),
    }


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def gradient_features(image: np.ndarray) -> dict:
    x = image.astype(np.float32) / 255.0

    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.hypot(gx, gy)

    return {
        "gradient_mean": finite_float(np.mean(magnitude)),
        "gradient_std": finite_float(np.std(magnitude)),
        "gradient_p95": finite_float(np.percentile(magnitude, 95)),
        "gradient_max": finite_float(np.max(magnitude)),
    }


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def edge_features(image: np.ndarray) -> dict:
    edges = cv2.Canny(image, 50, 150)

    count = int(np.count_nonzero(edges))

    return {
        "edge_density": finite_float(count / edges.size),
        "edge_pixels": count,
    }


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def laplacian_features(image: np.ndarray) -> dict:
    x = image.astype(np.float32) / 255.0

    lap = cv2.Laplacian(
        x,
        cv2.CV_32F,
        ksize=3,
    )

    return {
        "laplacian_std": finite_float(np.std(lap)),
        "laplacian_energy": finite_float(np.mean(lap * lap)),
    }


# ---------------------------------------------------------------------------
# Frequency domain
# ---------------------------------------------------------------------------

def frequency_features(image: np.ndarray) -> dict:
    x = image.astype(np.float64)
    x -= np.mean(x)

    spectrum = np.fft.fftshift(
        np.fft.fft2(x)
    )

    power = np.abs(spectrum) ** 2

    h, w = power.shape

    cy = h // 2
    cx = w // 2

    dc_free = power.copy()

    dc_free[
        max(0, cy - 1):cy + 2,
        max(0, cx - 1):cx + 2,
    ] = 0.0

    total = float(np.sum(power)) + EPS
    non_dc = float(np.sum(dc_free)) + EPS

    yy, xx = np.indices(power.shape)

    radius = np.hypot(
        yy - cy,
        xx - cx,
    )

    rmax = float(np.max(radius))

    low = radius <= 0.15 * rmax

    mid = (
        (radius > 0.15 * rmax)
        & (radius <= 0.40 * rmax)
    )

    high = radius > 0.40 * rmax

    p = dc_free / non_dc
    p = p[p > EPS]

    entropy = -np.sum(
        p * np.log2(p)
    )

    entropy /= max(
        np.log2(len(p)),
        1.0,
    )

    return {
        "fft_non_dc_energy": finite_float(non_dc),
        "fft_low_ratio": finite_float(
            np.sum(power[low]) / total
        ),
        "fft_mid_ratio": finite_float(
            np.sum(power[mid]) / total
        ),
        "fft_high_ratio": finite_float(
            np.sum(power[high]) / total
        ),
        "spectral_entropy": finite_float(entropy),
    }


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _autocorrelation_1d(
    signal: np.ndarray,
) -> np.ndarray:

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    signal -= np.mean(signal)

    n = signal.size

    padded = np.zeros(
        2 * n,
        dtype=np.float64,
    )

    padded[:n] = signal

    fft = np.fft.rfft(padded)

    ac = np.fft.irfft(
        fft * np.conjugate(fft),
        n=2 * n,
    )[:n]

    if ac[0] <= EPS:
        return np.zeros(
            n,
            dtype=np.float64,
        )

    return ac / ac[0]


def autocorrelation_features(
    image: np.ndarray,
) -> dict:

    x = image.astype(np.float64)

    x -= np.mean(x)

    row_indices = np.linspace(
        0,
        x.shape[0] - 1,
        min(64, x.shape[0]),
        dtype=int,
    )

    col_indices = np.linspace(
        0,
        x.shape[1] - 1,
        min(64, x.shape[1]),
        dtype=int,
    )

    row_acs = [
        _autocorrelation_1d(x[i, :])
        for i in row_indices
    ]

    col_acs = [
        _autocorrelation_1d(x[:, j])
        for j in col_indices
    ]

    ac_x = np.mean(
        row_acs,
        axis=0,
    )

    ac_y = np.mean(
        col_acs,
        axis=0,
    )

    limit_x = min(
        200,
        len(ac_x) - 1,
    )

    limit_y = min(
        200,
        len(ac_y) - 1,
    )

    ac_x = ac_x[
        1:limit_x + 1
    ]

    ac_y = ac_y[
        1:limit_y + 1
    ]

    def first_below_half(
        ac: np.ndarray,
    ) -> float:

        hits = np.flatnonzero(
            ac < 0.5
        )

        if len(hits):
            return float(hits[0] + 1)

        return float(len(ac))

    def peak_lag(
        ac: np.ndarray,
    ) -> float:

        if len(ac) == 0:
            return 0.0

        return float(
            np.argmax(ac) + 1
        )

    return {
        "autocorr_x_max": finite_float(
            np.max(ac_x)
        ),
        "autocorr_y_max": finite_float(
            np.max(ac_y)
        ),
        "autocorr_x_mean": finite_float(
            np.mean(ac_x)
        ),
        "autocorr_y_mean": finite_float(
            np.mean(ac_y)
        ),
        "correlation_length_x_px": finite_float(
            first_below_half(ac_x)
        ),
        "correlation_length_y_px": finite_float(
            first_below_half(ac_y)
        ),
        "dominant_period_x_px": finite_float(
            peak_lag(ac_x)
        ),
        "dominant_period_y_px": finite_float(
            peak_lag(ac_y)
        ),
        "periodicity_indicator": finite_float(
            max(
                np.max(ac_x),
                np.max(ac_y),
            )
        ),
    }


# ---------------------------------------------------------------------------
# Local texture
# ---------------------------------------------------------------------------

def texture_features(image: np.ndarray) -> dict:
    x = image.astype(np.float32) / 255.0

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

    local_var = np.maximum(
        sq_mean - mean * mean,
        0.0,
    )

    local_std = np.sqrt(
        local_var
    )

    return {
        "local_std_mean": finite_float(
            np.mean(local_std)
        ),
        "local_std_std": finite_float(
            np.std(local_std)
        ),
        "local_std_p95": finite_float(
            np.percentile(local_std, 95)
        ),
    }


# ---------------------------------------------------------------------------
# Noise proxy
# ---------------------------------------------------------------------------

def noise_proxy_features(
    image: np.ndarray,
) -> dict:

    """
    High-frequency residual after Gaussian smoothing.

    This is explicitly a proxy.

    It is NOT automatically sensor noise because edges, texture,
    aliasing and other structures also contribute to the residual.
    """

    x = image.astype(
        np.float32
    ) / 255.0

    smooth = cv2.GaussianBlur(
        x,
        (0, 0),
        1.0,
    )

    residual = x - smooth

    return {
        "highpass_residual_std": finite_float(
            np.std(residual)
        ),
        "highpass_residual_mad": finite_float(
            np.median(
                np.abs(
                    residual
                    - np.median(residual)
                )
            )
        ),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(features: dict) -> dict:
    """
    Descriptive labels only.

    These labels must not become Phase 5 model-selection rules without
    held-out validation.
    """

    periodicity = features[
        "periodicity_indicator"
    ]

    entropy = features[
        "spectral_entropy"
    ]

    edge_density = features[
        "edge_density"
    ]

    high_frequency = features[
        "fft_high_ratio"
    ]

    contrast = features[
        "global_contrast"
    ]

    if periodicity >= 0.65:
        structure = "strongly_periodic"
    elif periodicity >= 0.35:
        structure = "structured"
    else:
        structure = "weakly_periodic_or_aperiodic"

    if entropy >= 0.75:
        spectrum = "broadband"
    elif entropy >= 0.45:
        spectrum = "mixed_frequency"
    else:
        spectrum = "concentrated_frequency"

    if edge_density >= 0.15:
        edge_class = "edge_dense"
    elif edge_density >= 0.05:
        edge_class = "moderate_edges"
    else:
        edge_class = "edge_sparse"

    texture = (
        "high_frequency"
        if high_frequency >= 0.25
        else "low_or_mid_frequency"
    )

    if contrast >= 0.20:
        contrast_class = "high_contrast"
    elif contrast >= 0.08:
        contrast_class = "moderate_contrast"
    else:
        contrast_class = "low_contrast"

    return {
        "structure_class": structure,
        "spectral_class": spectrum,
        "edge_class": edge_class,
        "texture_class": texture,
        "contrast_class": contrast_class,
    }


# ---------------------------------------------------------------------------
# Main image analysis
# ---------------------------------------------------------------------------

def analyze_image(
    image: np.ndarray,
) -> dict:

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image, got {image.shape}"
        )

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    features = {}

    features.update(
        intensity_features(image)
    )

    features.update(
        contrast_features(image)
    )

    features.update(
        gradient_features(image)
    )

    features.update(
        edge_features(image)
    )

    features.update(
        laplacian_features(image)
    )

    features.update(
        frequency_features(image)
    )

    features.update(
        autocorrelation_features(image)
    )

    features.update(
        texture_features(image)
    )

    features.update(
        noise_proxy_features(image)
    )

    features.update(
        classify(features)
    )

    return features


# ---------------------------------------------------------------------------
# Sampling metadata
# ---------------------------------------------------------------------------

def estimate_sampling(
    search_shape,
    reference_shape,
    metadata=None,
):

    metadata = metadata or {}

    for key in (
        "sampling_ratio",
        "scale_relationship_nominal",
    ):

        if key in metadata:

            try:
                value = float(
                    metadata[key]
                )

                if value > 0:
                    return value, "metadata"

            except (
                TypeError,
                ValueError,
            ):
                pass

    if (
        search_shape == reference_shape
    ):
        return (
            None,
            "not_inferable_from_dimensions",
        )

    return (
        None,
        "unknown",
    )


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def load_manifest(
    dataset_root: Path,
):

    manifest = (
        dataset_root
        / "manifest.csv"
    )

    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest}"
        )

    with manifest.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        raise ValueError(
            "manifest.csv is empty"
        )

    return rows


def locate_pair_dir(
    dataset_root: Path,
    pair_id: str,
):

    direct = (
        dataset_root
        / pair_id
    )

    if direct.is_dir():
        return direct

    matches = [
        p
        for p in dataset_root.rglob(pair_id)
        if p.is_dir()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Could not locate pair: {pair_id}"
        )

    return matches[0]


def find_image(
    pair_dir: Path,
    name: str,
):

    candidates = [
        pair_dir / name,
        pair_dir / "images" / name,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(
        pair_dir.rglob(name)
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {name} under {pair_dir}"
    )


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(
    dataset_root: Path,
):

    rows = []

    manifest_rows = load_manifest(
        dataset_root
    )

    for entry in manifest_rows:

        pair_id = entry["pair_id"]

        pair_dir = locate_pair_dir(
            dataset_root,
            pair_id,
        )

        search_path = find_image(
            pair_dir,
            "search.png",
        )

        reference_path = find_image(
            pair_dir,
            "reference.png",
        )

        search = cv2.imread(
            str(search_path),
            cv2.IMREAD_GRAYSCALE,
        )

        reference = cv2.imread(
            str(reference_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if search is None:
            raise ValueError(
                f"Failed to read {search_path}"
            )

        if reference is None:
            raise ValueError(
                f"Failed to read {reference_path}"
            )

        row = {
            "source": str(dataset_root),
            "pair_id": pair_id,
            "split": entry.get(
                "split",
                "",
            ),
            "architecture": entry.get(
                "architecture",
                "",
            ),
            "seed": entry.get(
                "seed",
                "",
            ),
            "search_height": search.shape[0],
            "search_width": search.shape[1],
            "reference_height": reference.shape[0],
            "reference_width": reference.shape[1],
        }

        ratio, source = estimate_sampling(
            search.shape,
            reference.shape,
            entry,
        )

        row["sampling_ratio"] = (
            ratio
            if ratio is not None
            else ""
        )

        row["sampling_ratio_source"] = source

        search_features = analyze_image(
            search
        )

        reference_features = analyze_image(
            reference
        )

        for key, value in search_features.items():

            row[
                f"search_{key}"
            ] = value

        for key, value in reference_features.items():

            row[
                f"reference_{key}"
            ] = value

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Canonical renderer smoke test
# ---------------------------------------------------------------------------

def canonical_rows():

    import canonical_renderer as cr

    rows = []

    seeds = list(
        range(
            20260875,
            20260905,
        )
    )

    for scene_type in (
        "periodic",
        "quasiperiodic",
    ):

        for seed in seeds:

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

            template = (
                cr.create_ps02_template(
                    reference
                )
            )

            search_features = analyze_image(
                search
            )

            template_features = analyze_image(
                template
            )

            row = {
                "source": "canonical_renderer",
                "pair_id": (
                    f"{scene_type}_{seed}"
                ),
                "split": "synthetic_smoke",
                "architecture": scene_type,
                "seed": seed,
                "search_height": search.shape[0],
                "search_width": search.shape[1],
                "reference_height": reference.shape[0],
                "reference_width": reference.shape[1],
                "template_height": template.shape[0],
                "template_width": template.shape[1],
                "sampling_ratio": float(
                    cr.SAMPLING_RATIO
                ),
                "sampling_ratio_source":
                    "canonical_renderer",
            }

            for key, value in search_features.items():

                row[
                    f"search_{key}"
                ] = value

            for key, value in template_features.items():

                row[
                    f"template_{key}"
                ] = value

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(
    rows,
    path: Path,
):

    if not rows:
        raise ValueError(
            "No EDA rows to write"
        )

    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def numeric_columns(rows):

    keys = rows[0].keys()

    return [
        key
        for key in keys
        if isinstance(
            rows[0][key],
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    ]


def build_summary(rows):

    summary = {
        "n_observations": len(rows),
        "architectures_or_scene_types": {},
        "numeric_means": {},
        "sampling_ratio": {},
    }

    groups = {}

    for row in rows:

        group = str(
            row.get(
                "architecture"
            )
            or row.get(
                "scene_type"
            )
            or "unknown"
        )

        groups.setdefault(
            group,
            [],
        ).append(row)

    for group, group_rows in groups.items():

        structure_classes = {}

        for row in group_rows:

            cls = row.get(
                "search_structure_class",
                "unknown",
            )

            structure_classes[cls] = (
                structure_classes.get(
                    cls,
                    0,
                )
                + 1
            )

        spectral_classes = {}

        for row in group_rows:

            cls = row.get(
                "search_spectral_class",
                "unknown",
            )

            spectral_classes[cls] = (
                spectral_classes.get(
                    cls,
                    0,
                )
                + 1
            )

        summary[
            "architectures_or_scene_types"
        ][group] = {
            "count": len(group_rows),
            "structure_classes":
                structure_classes,
            "spectral_classes":
                spectral_classes,
        }

    for key in numeric_columns(rows):

        values = []

        for row in rows:

            value = row[key]

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):

                values.append(
                    float(value)
                )

        summary[
            "numeric_means"
        ][key] = (
            float(np.mean(values))
            if values
            else None
        )

    ratios = [
        r.get("sampling_ratio")
        for r in rows
        if isinstance(
            r.get("sampling_ratio"),
            (
                int,
                float,
            ),
        )
    ]

    if ratios:

        summary[
            "sampling_ratio"
        ] = {
            "mean":
                float(np.mean(ratios)),
            "min":
                float(np.min(ratios)),
            "max":
                float(np.max(ratios)),
        }

    return summary


def write_report(
    summary,
    rows,
    path: Path,
):

    classes = {}

    for row in rows:

        cls = row.get(
            "search_structure_class",
            "unknown",
        )

        classes[cls] = (
            classes.get(
                cls,
                0,
            )
            + 1
        )

    report = f"""# MICRONYX Phase 4A — Automated EDA Report

## Scope

This report is generated automatically from the Phase 4 EDA profiler.

The profiler measures:

- intensity statistics
- contrast
- spatial correlation
- frequency content
- edge and gradient structure
- local texture
- high-frequency residual noise proxy

The structural labels are descriptive EDA descriptors.
They are not model-selection decisions.

## Dataset

- Observations: {summary["n_observations"]}
- Structure classes: {json.dumps(classes, sort_keys=True)}
- Sampling ratio: {json.dumps(summary["sampling_ratio"], sort_keys=True)}

## Mean search-image descriptors

- Mean intensity: {summary["numeric_means"].get("search_mean")}
- Contrast: {summary["numeric_means"].get("search_global_contrast")}
- Gradient mean: {summary["numeric_means"].get("search_gradient_mean")}
- Edge density: {summary["numeric_means"].get("search_edge_density")}
- Spectral entropy: {summary["numeric_means"].get("search_spectral_entropy")}
- Periodicity indicator: {summary["numeric_means"].get("search_periodicity_indicator")}
- Dominant period X (px): {summary["numeric_means"].get("search_dominant_period_x_px")}
- Dominant period Y (px): {summary["numeric_means"].get("search_dominant_period_y_px")}
- Correlation length X (px): {summary["numeric_means"].get("search_correlation_length_x_px")}
- Correlation length Y (px): {summary["numeric_means"].get("search_correlation_length_y_px")}
- High-frequency ratio: {summary["numeric_means"].get("search_fft_high_ratio")}
- High-pass residual std: {summary["numeric_means"].get("search_highpass_residual_std")}

## Physical context

Correlation lengths and dominant periods are reported in image pixels.

Conversion to physical units requires validated acquisition metadata.

Target coordinates are not used as image descriptors.

## Reproducibility

- CSV contains one row per analyzed observation.
- JSON contains aggregate statistics.
- Canonical synthetic mode uses `canonical_renderer.py`.
- Results should be regenerated when dataset or profiler code changes.

## Interpretation boundary

EDA identifies measurable properties of observations.

Phase 5 is responsible for validating whether these properties improve
model selection on held-out data.
"""

    path.write_text(
        report,
        encoding="utf-8",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
    )

    parser.add_argument(
        "--canonical-synthetic",
        action="store_true",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    if (
        args.dataset is None
        and not args.canonical_synthetic
    ):
        parser.error(
            "Use --dataset DATASET_ROOT "
            "or --canonical-synthetic"
        )

    if (
        args.dataset is not None
        and args.canonical_synthetic
    ):
        parser.error(
            "Choose one input mode"
        )

    if args.dataset is not None:

        rows = analyze_dataset(
            args.dataset
        )

    else:

        rows = canonical_rows()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        args.out
        / "automated_eda_results.csv"
    )

    json_path = (
        args.out
        / "automated_eda_summary.json"
    )

    report_path = (
        args.out
        / "automated_eda_report.md"
    )

    write_csv(
        rows,
        csv_path,
    )

    summary = build_summary(
        rows
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    write_report(
        summary,
        rows,
        report_path,
    )

    print(
        f"EDA complete: {len(rows)} observations"
    )

    print(
        f"CSV:    {csv_path}"
    )

    print(
        f"JSON:   {json_path}"
    )

    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()