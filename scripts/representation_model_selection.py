"""
MICRONYX - Phase B
Automated Representation & Model Selection

Phase-B responsibilities
------------------------
1. Consume Phase-A EDA descriptors.
2. Benchmark candidate representations:
      - NCC
      - DoG
      - Gradient
      - Edge
      - Frequency
3. Compare three selection policies:
      - rule-based
      - optimization-based
      - learned
4. Evaluate selection on held-out scenes.
5. Produce generator weights rather than permanently hard-coding
   one representation as universally best.
6. Support an expert override with validation.
7. Save machine-readable and human-readable reports.

Important boundary
------------------
This module evaluates representations for MODEL SELECTION.

It does not become the final production candidate-generation engine.
That belongs to Phase C.

Target coordinates are used ONLY as evaluation labels.
They are never used as EDA/model-selection input features.

Usage
-----

    python scripts/representation_model_selection.py \
        --dataset dataset_v1

Optional:

    python scripts/representation_model_selection.py \
        --dataset dataset_v1 \
        --eda validation/phase4/automated_eda/automated_eda_results.csv

Expert override:

    python scripts/representation_model_selection.py \
        --dataset dataset_v1 \
        --override dog:0.5,ncc:0.2,gradient:0.1,edge:0.1,frequency:0.1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

import automated_eda


GENERATORS = (
    "ncc",
    "dog",
    "gradient",
    "edge",
    "frequency",
)

EPS = 1e-12

DEFAULT_OUTPUT = (
    ROOT
    / "validation"
    / "phase5"
    / "representation_selection"
)

DEFAULT_SCALE = 10.0

TARGET_ALIASES_X = (
    "target_x",
    "gt_x",
    "target_center_x",
    "gt_center_x",
    "center_x",
)

TARGET_ALIASES_Y = (
    "target_y",
    "gt_y",
    "target_center_y",
    "gt_center_y",
    "center_y",
)


# ============================================================================
# Utilities
# ============================================================================

def finite(value: float) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"Non-finite value encountered: {value}"
        )

    return value


def normalize_weights(
    weights: dict[str, float],
) -> dict[str, float]:

    clean = {}

    for generator in GENERATORS:

        value = float(
            weights.get(
                generator,
                0.0,
            )
        )

        if not np.isfinite(value):
            raise ValueError(
                f"Invalid weight for {generator}: {value}"
            )

        if value < 0:
            raise ValueError(
                f"Negative weight for {generator}: {value}"
            )

        clean[generator] = value

    total = sum(
        clean.values()
    )

    if total <= EPS:
        raise ValueError(
            "Generator weights must contain "
            "at least one positive value."
        )

    return {
        key: finite(
            value / total
        )
        for key, value in clean.items()
    }


def softmax(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "softmax expects a one-dimensional array."
        )

    if values.size == 0:
        raise ValueError(
            "softmax cannot operate on an empty array."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "softmax received non-finite values."
        )

    shifted = values - np.max(values)

    exp_values = np.exp(
        shifted
    )

    total = np.sum(
        exp_values
    )

    if total <= EPS:
        raise ValueError(
            "softmax normalization failed."
        )

    result = (
        exp_values / total
    )

    # Remove tiny accumulated floating-point error so the returned
    # probability vector sums exactly to one in normal cases.
    result[
        np.argmax(result)
    ] += (
        1.0 - np.sum(result)
    )

    return result

def load_image(
    path: Path,
) -> np.ndarray:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return image


def locate_image(
    pair_dir: Path,
    filename: str,
) -> Path:

    candidates = (
        pair_dir / filename,
        pair_dir / "images" / filename,
    )

    for path in candidates:

        if path.exists():
            return path

    matches = list(
        pair_dir.rglob(
            filename
        )
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"{filename} not found under {pair_dir}"
        )

    raise ValueError(
        f"Multiple {filename} files found under {pair_dir}: "
        f"{matches}"
    )


def locate_pair(
    dataset_root: Path,
    pair_id: str,
) -> Path:

    direct = (
        dataset_root
        / pair_id
    )

    if direct.is_dir():
        return direct

    matches = [
        p
        for p in dataset_root.rglob(
            pair_id
        )
        if p.is_dir()
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Cannot locate pair directory for {pair_id}"
        )

    raise ValueError(
        f"Multiple directories found for {pair_id}: {matches}"
    )


# ============================================================================
# Manifest
# ============================================================================

def load_manifest(
    dataset_root: Path,
) -> list[dict[str, str]]:

    path = (
        dataset_root
        / "manifest.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        raise ValueError(
            "manifest.csv contains no rows."
        )

    if "pair_id" not in rows[0]:
        raise ValueError(
            "manifest.csv must contain pair_id."
        )

    return rows


def get_float_column(
    row: dict[str, str],
    aliases: tuple[str, ...],
) -> float | None:

    for key in aliases:

        if key not in row:
            continue

        raw = row[key]

        if raw is None or raw == "":
            continue

        try:

            value = float(raw)

            if np.isfinite(value):
                return value

        except (
            TypeError,
            ValueError,
        ):
            pass

    return None


def get_target(
    row: dict[str, str],
) -> tuple[float, float]:

    x = get_float_column(
        row,
        TARGET_ALIASES_X,
    )

    y = get_float_column(
        row,
        TARGET_ALIASES_Y,
    )

    if x is None or y is None:

        raise ValueError(
            "Representation benchmarking requires target coordinates "
            "in the manifest. Supported X aliases: "
            f"{TARGET_ALIASES_X}; Y aliases: {TARGET_ALIASES_Y}"
        )

    return (
        finite(x),
        finite(y),
    )


# ============================================================================
# Image representation construction
# ============================================================================

def resize_reference(
    reference: np.ndarray,
    search_shape: tuple[int, int],
    scale: float,
) -> np.ndarray:

    """
    Convert high-resolution reference into expected search-scale template.

    For the challenge's nominal 10x relationship:

        1000x1000 reference
        ->
        100x100 search-scale template
    """

    expected_h = max(
        1,
        int(
            round(
                reference.shape[0]
                / scale
            )
        ),
    )

    expected_w = max(
        1,
        int(
            round(
                reference.shape[1]
                / scale
            )
        ),
    )

    return cv2.resize(
        reference,
        (
            expected_w,
            expected_h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def gradient_image(
    image: np.ndarray,
) -> np.ndarray:

    x = image.astype(
        np.float32
    )

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

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    return magnitude


def edge_image(
    image: np.ndarray,
) -> np.ndarray:

    return cv2.Canny(
        image,
        50,
        150,
    ).astype(
        np.float32
    )


def dog_image(
    image: np.ndarray,
) -> np.ndarray:

    image_float = image.astype(
        np.float32
    )

    blur_small = cv2.GaussianBlur(
        image_float,
        (0, 0),
        1.0,
    )

    blur_large = cv2.GaussianBlur(
        image_float,
        (0, 0),
        2.0,
    )

    return (
        blur_small
        - blur_large
    )


# ============================================================================
# Representation matching
# ============================================================================

def normalized_template_match(
    search: np.ndarray,
    template: np.ndarray,
    method: str = "ccoeff",
) -> tuple[float, tuple[int, int]]:

    if (
        template.shape[0]
        > search.shape[0]
        or template.shape[1]
        > search.shape[1]
    ):

        raise ValueError(
            "Template cannot be larger than search image."
        )

    if method == "ccoeff":

        score_method = cv2.TM_CCOEFF_NORMED

    elif method == "ccorr":

        score_method = cv2.TM_CCORR_NORMED

    else:

        raise ValueError(
            f"Unsupported matching method: {method}"
        )

    result = cv2.matchTemplate(
        search.astype(np.float32),
        template.astype(np.float32),
        score_method,
    )

    min_value, max_value, min_loc, max_loc = (
        cv2.minMaxLoc(result)
    )

    del min_value, min_loc

    return (
        finite(max_value),
        (
            int(max_loc[0]),
            int(max_loc[1]),
        ),
    )


def frequency_representation(
    image: np.ndarray,
) -> np.ndarray:

    x = image.astype(
        np.float32
    )

    x -= np.mean(x)

    fft = np.fft.fft2(
        x
    )

    magnitude = np.abs(
        fft
    )

    magnitude = np.fft.fftshift(
        magnitude
    )

    magnitude = np.log1p(
        magnitude
    )

    return magnitude.astype(
        np.float32
    )


def frequency_match(
    search: np.ndarray,
    template: np.ndarray,
) -> tuple[float, tuple[int, int]]:

    """
    Frequency-domain-inspired local matcher.

    The image and template are represented by gradient-like spectral
    magnitudes and matched spatially.

    This is deliberately kept as a representation benchmark.
    Phase C will provide the production candidate-generation interface.
    """

    search_freq = frequency_representation(
        search
    )

    template_freq = frequency_representation(
        template
    )

    # Match template spectral content after resizing it to the
    # corresponding search-space size.
    template_freq = cv2.resize(
        template_freq,
        (
            template.shape[1],
            template.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return normalized_template_match(
        search_freq,
        template_freq,
        "ccoeff",
    )


def run_representation(
    search: np.ndarray,
    reference: np.ndarray,
    scale: float,
    generator: str,
) -> tuple[float, tuple[int, int]]:

    template = resize_reference(
        reference,
        search.shape,
        scale,
    )

    if generator == "ncc":

        return normalized_template_match(
            search,
            template,
            "ccoeff",
        )

    if generator == "dog":

        return normalized_template_match(
            dog_image(search),
            dog_image(template),
            "ccoeff",
        )

    if generator == "gradient":

        return normalized_template_match(
            gradient_image(search),
            gradient_image(template),
            "ccorr",
        )

    if generator == "edge":

        return normalized_template_match(
            edge_image(search),
            edge_image(template),
            "ccorr",
        )

    if generator == "frequency":

        return frequency_match(
            search,
            template,
        )

    raise ValueError(
        f"Unknown generator: {generator}"
    )


# ============================================================================
# Evaluation
# ============================================================================

@dataclass
class GeneratorResult:

    generator: str
    score: float
    predicted_x: float
    predicted_y: float
    error_px: float
    runtime_ms: float


def evaluate_generator(
    search: np.ndarray,
    reference: np.ndarray,
    target_x: float,
    target_y: float,
    scale: float,
    generator: str,
) -> GeneratorResult:

    start = time.perf_counter()

    score, location = run_representation(
        search,
        reference,
        scale,
        generator,
    )

    runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    template = resize_reference(
        reference,
        search.shape,
        scale,
    )

    predicted_x = (
        location[0]
        + template.shape[1] / 2.0
    )

    predicted_y = (
        location[1]
        + template.shape[0] / 2.0
    )

    error = math.hypot(
        predicted_x - target_x,
        predicted_y - target_y,
    )

    return GeneratorResult(
        generator=generator,
        score=finite(score),
        predicted_x=finite(predicted_x),
        predicted_y=finite(predicted_y),
        error_px=finite(error),
        runtime_ms=finite(runtime_ms),
    )


def best_generator(
    results: list[GeneratorResult],
) -> str:

    return min(
        results,
        key=lambda item: (
            item.error_px,
            -item.score,
        ),
    ).generator


# ============================================================================
# EDA feature extraction
# ============================================================================

EDA_FEATURES = (
    "search_global_contrast",
    "search_edge_density",
    "search_highpass_residual_std",
    "search_spectral_entropy",
    "search_periodicity_score",
    "search_autocorr_x_max",
    "search_autocorr_y_max",
    "search_correlation_length_x_px",
    "search_correlation_length_y_px",
    "search_dominant_period_x_px",
    "search_dominant_period_y_px",
    "search_effective_scale_px",
    "search_admissible_context_min_px",
)


def numeric_eda_features(
    row: dict[str, Any],
) -> np.ndarray:

    values = []

    for key in EDA_FEATURES:

        value = row.get(
            key,
            0.0,
        )

        try:

            value = float(value)

            if not np.isfinite(value):
                value = 0.0

        except (
            TypeError,
            ValueError,
        ):

            value = 0.0

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def standardize_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    mean = np.mean(
        train_x,
        axis=0,
    )

    std = np.std(
        train_x,
        axis=0,
    )

    std[
        std < EPS
    ] = 1.0

    return (
        (train_x - mean) / std,
        (test_x - mean) / std,
    )


# ============================================================================
# Rule-based policy
# ============================================================================

def rule_policy(
    eda_row: dict[str, Any],
) -> dict[str, float]:

    """
    Transparent baseline.

    This is a baseline policy, not the final hard-coded policy.

    It uses only Phase-A EDA descriptors.
    """

    structure = str(
        eda_row.get(
            "search_scene_structure_class",
            "structured_mixed",
        )
    )

    periodicity = float(
        eda_row.get(
            "search_periodicity_score",
            0.0,
        )
    )

    spectral_entropy = float(
        eda_row.get(
            "search_spectral_entropy",
            1.0,
        )
    )

    edge_density = float(
        eda_row.get(
            "search_edge_density",
            0.0,
        )
    )

    weights = {
        generator: 0.10
        for generator in GENERATORS
    }

    if structure == "periodic":

        weights.update({
            "dog": 0.28,
            "frequency": 0.28,
            "ncc": 0.18,
            "gradient": 0.16,
            "edge": 0.10,
        })

    elif structure == "quasi_periodic":

        weights.update({
            "dog": 0.27,
            "gradient": 0.23,
            "frequency": 0.22,
            "ncc": 0.16,
            "edge": 0.12,
        })

    elif structure == "aperiodic":

        weights.update({
            "ncc": 0.30,
            "gradient": 0.25,
            "dog": 0.20,
            "edge": 0.15,
            "frequency": 0.10,
        })

    else:

        weights.update({
            "dog": 0.24,
            "gradient": 0.22,
            "ncc": 0.20,
            "edge": 0.18,
            "frequency": 0.16,
        })

    if periodicity > 0.65:

        weights["dog"] += 0.05
        weights["frequency"] += 0.05
        weights["ncc"] -= 0.05
        weights["edge"] -= 0.05

    if spectral_entropy > 0.70:

        weights["gradient"] += 0.05
        weights["edge"] += 0.05
        weights["frequency"] -= 0.05
        weights["dog"] -= 0.05

    if edge_density > 0.20:

        weights["gradient"] += 0.04
        weights["edge"] += 0.04
        weights["frequency"] -= 0.04
        weights["ncc"] -= 0.04

    return normalize_weights(
        weights
    )


# ============================================================================
# Optimization-based policy
# ============================================================================

def utility_from_results(
    results: list[GeneratorResult],
) -> np.ndarray:

    errors = np.asarray(
        [
            item.error_px
            for item in results
        ],
        dtype=np.float64,
    )

    scores = np.asarray(
        [
            item.score
            for item in results
        ],
        dtype=np.float64,
    )

    runtimes = np.asarray(
        [
            item.runtime_ms
            for item in results
        ],
        dtype=np.float64,
    )

    error_term = (
        1.0
        / (
            errors
            + 1.0
        )
    )

    score_term = (
        scores
        - np.min(scores)
    ) / (
        np.ptp(scores)
        + EPS
    )

    runtime_term = (
        1.0
        / (
            runtimes
            + 1.0
        )
    )

    utility = (
        0.60
        * error_term
        + 0.30
        * score_term
        + 0.10
        * runtime_term
    )

    return utility


def optimize_weights(
    train_results: list[
        list[GeneratorResult]
    ],
    seed: int = 20260913,
) -> dict[str, float]:

    """
    Deterministic simplex optimization.

    Objective:
        maximize average representation utility.

    The optimization chooses generator weights from held-out training
    scenes. Test scenes are never used here.
    """

    rng = np.random.default_rng(
        seed
    )

    if not train_results:
        raise ValueError(
            "Optimization requires training results."
        )

    utilities = np.vstack([
        utility_from_results(
            results
        )
        for results in train_results
    ])

    mean_utility = np.mean(
        utilities,
        axis=0,
    )

    best = np.zeros(
        len(GENERATORS),
        dtype=np.float64,
    )

    best[
        int(
            np.argmax(
                mean_utility
            )
        )
    ] = 1.0

    best_objective = float(
        np.dot(
            best,
            mean_utility,
        )
    )

    # Random simplex proposals.
    for _ in range(500):

        candidate = rng.dirichlet(
            np.ones(
                len(GENERATORS)
            )
        )

        objective = float(
            np.dot(
                candidate,
                mean_utility,
            )
        )

        if objective > best_objective:

            best = candidate
            best_objective = objective

    return normalize_weights({
        generator: float(
            best[index]
        )
        for index, generator
        in enumerate(GENERATORS)
    })


# ============================================================================
# Learned policy
# ============================================================================

def train_learned_policy(
    train_rows: list[dict[str, Any]],
) -> Any:

    """
    Train a scene-level strategy selector.

    Label:
        best representation on the training observation.

    Features:
        Phase-A EDA descriptors only.

    Ground-truth coordinates are NOT included as features.
    """

    try:

        from sklearn.ensemble import (
            RandomForestClassifier,
        )

    except ImportError as exc:

        raise RuntimeError(
            "Phase B learned policy requires scikit-learn."
        ) from exc

    x = np.vstack([
        numeric_eda_features(
            row
        )
        for row in train_rows
    ])

    labels = np.asarray([
        row[
            "best_generator"
        ]
        for row in train_rows
    ])

    if len(
        np.unique(labels)
    ) < 2:

        raise RuntimeError(
            "Learned policy needs at least two different "
            "best-generator classes in the training set. "
            "The dataset does not contain enough representation diversity."
        )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=20260913,
        class_weight="balanced",
        n_jobs=-1,
    )

    classifier.fit(
        x,
        labels,
    )

    return classifier


def learned_weights(
    classifier: Any,
    eda_row: dict[str, Any],
) -> dict[str, float]:

    x = numeric_eda_features(
        eda_row
    ).reshape(
        1,
        -1,
    )

    probabilities = classifier.predict_proba(
        x
    )[0]

    weights = {
        generator: 0.0
        for generator in GENERATORS
    }

    for class_name, probability in zip(
        classifier.classes_,
        probabilities,
    ):

        weights[
            str(class_name)
        ] = float(
            probability
        )

    # Preserve every representation with a tiny floor.
    # This avoids a brittle zero-probability policy.
    for generator in GENERATORS:
        weights[generator] += 0.01

    return normalize_weights(
        weights
    )


# ============================================================================
# Combining generator scores
# ============================================================================

def combine_generator_results(
    results: list[GeneratorResult],
    weights: dict[str, float],
) -> GeneratorResult:

    by_name = {
        item.generator: item
        for item in results
    }

    scores = np.asarray([
        by_name[
            generator
        ].score
        for generator in GENERATORS
    ])

    errors = np.asarray([
        by_name[
            generator
        ].error_px
        for generator in GENERATORS
    ])

    runtimes = np.asarray([
        by_name[
            generator
        ].runtime_ms
        for generator in GENERATORS
    ])

    # Convert each representation score into a comparable rank signal.
    score_signal = softmax(
        scores
    )

    # Lower error is better, but error is only available for benchmark
    # evaluation. In the production selector, this signal is NOT used.
    #
    # Here it is used to evaluate whether the selected policy actually
    # chooses a good representation.
    error_signal = softmax(
        -errors
    )

    generator_weights = np.asarray([
        weights[
            generator
        ]
        for generator in GENERATORS
    ])

    combined = (
        0.70
        * generator_weights
        * score_signal
        + 0.30
        * generator_weights
        * error_signal
    )

    index = int(
        np.argmax(
            combined
        )
    )

    selected = by_name[
        GENERATORS[index]
    ]

    return selected


# ============================================================================
# Expert override
# ============================================================================

def parse_override(
    value: str | None,
) -> dict[str, float] | None:

    if value is None:
        return None

    weights = {}

    for token in value.split(","):

        token = token.strip()

        if not token:
            continue

        if ":" not in token:

            raise ValueError(
                "Override must use generator:weight syntax."
            )

        generator, raw_weight = (
            token.split(
                ":",
                1,
            )
        )

        generator = generator.strip().lower()

        if generator not in GENERATORS:

            raise ValueError(
                f"Unknown generator in override: {generator}"
            )

        try:

            weight = float(
                raw_weight
            )

        except ValueError as exc:

            raise ValueError(
                f"Invalid override weight: {raw_weight}"
            ) from exc

        weights[
            generator
        ] = weight

    return normalize_weights(
        weights
    )


def validate_override(
    weights: dict[str, float],
):

    weights = normalize_weights(
        weights
    )

    if any(
        value < 0
        for value in weights.values()
    ):

        raise ValueError(
            "Override contains negative weight."
        )

    return weights


# ============================================================================
# Dataset processing
# ============================================================================

def load_eda_rows(
    path: Path,
) -> dict[str, dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"EDA file not found: {path}"
        )

    rows = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        for row in csv.DictReader(f):

            pair_id = row.get(
                "pair_id"
            )

            if not pair_id:
                continue

            rows[
                pair_id
            ] = row

    return rows


def find_default_eda(
    dataset_root: Path,
) -> Path:

    candidates = (
        DEFAULT_OUTPUT.parent.parent
        / "phase4"
        / "automated_eda"
        / "automated_eda_results.csv",

        ROOT
        / "validation"
        / "phase4"
        / "automated_eda"
        / "automated_eda_results.csv",
    )

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Phase-4 EDA CSV. "
        "Run automated_eda.py first or provide --eda."
    )


def build_observation_rows(
    dataset_root: Path,
    eda_rows: dict[str, dict[str, Any]],
    scale: float,
) -> list[dict[str, Any]]:

    manifest = load_manifest(
        dataset_root
    )

    output = []

    for manifest_row in manifest:

        pair_id = manifest_row[
            "pair_id"
        ]

        if pair_id not in eda_rows:

            raise ValueError(
                f"Pair {pair_id} exists in manifest but not in Phase-4 EDA."
            )

        pair_dir = locate_pair(
            dataset_root,
            pair_id,
        )

        search_path = locate_image(
            pair_dir,
            "search.png",
        )

        reference_path = locate_image(
            pair_dir,
            "reference.png",
        )

        search = load_image(
            search_path
        )

        reference = load_image(
            reference_path
        )

        target_x, target_y = get_target(
            manifest_row
        )

        eda_row = eda_rows[
            pair_id
        ]

        generator_results = []

        for generator in GENERATORS:

            result = evaluate_generator(
                search,
                reference,
                target_x,
                target_y,
                scale,
                generator,
            )

            generator_results.append(
                result
            )

        best = best_generator(
            generator_results
        )

        row = dict(
            eda_row
        )

        row.update({
            "pair_id":
                pair_id,

            "split":
                manifest_row.get(
                    "split",
                    "",
                ),

            "architecture":
                manifest_row.get(
                    "architecture",
                    "",
                ),

            "target_x":
                target_x,

            "target_y":
                target_y,

            "best_generator":
                best,
        })

        for result in generator_results:

            prefix = (
                result.generator
            )

            row[
                f"{prefix}_score"
            ] = result.score

            row[
                f"{prefix}_predicted_x"
            ] = result.predicted_x

            row[
                f"{prefix}_predicted_y"
            ] = result.predicted_y

            row[
                f"{prefix}_error_px"
            ] = result.error_px

            row[
                f"{prefix}_runtime_ms"
            ] = result.runtime_ms

        output.append(
            row
        )

    return output


# ============================================================================
# Policy evaluation
# ============================================================================

def get_result_list(
    row: dict[str, Any],
) -> list[GeneratorResult]:

    results = []

    for generator in GENERATORS:

        results.append(
            GeneratorResult(
                generator=generator,
                score=float(
                    row[
                        f"{generator}_score"
                    ]
                ),
                predicted_x=float(
                    row[
                        f"{generator}_predicted_x"
                    ]
                ),
                predicted_y=float(
                    row[
                        f"{generator}_predicted_y"
                    ]
                ),
                error_px=float(
                    row[
                        f"{generator}_error_px"
                    ]
                ),
                runtime_ms=float(
                    row[
                        f"{generator}_runtime_ms"
                    ]
                ),
            )
        )

    return results


def evaluate_policy(
    rows: list[dict[str, Any]],
    policy_name: str,
    policy_function,
) -> dict[str, Any]:

    selected = []

    for row in rows:

        weights = policy_function(
            row
        )

        results = get_result_list(
            row
        )

        selected_result = combine_generator_results(
            results,
            weights,
        )

        selected.append({
            "pair_id":
                row["pair_id"],

            "selected_generator":
                selected_result.generator,

            "error_px":
                selected_result.error_px,

            "runtime_ms":
                selected_result.runtime_ms,

            "weights":
                weights,
        })

    errors = np.asarray([
        item["error_px"]
        for item in selected
    ])

    runtimes = np.asarray([
        item["runtime_ms"]
        for item in selected
    ])

    return {
        "policy":
            policy_name,

        "n":
            len(selected),

        "median_error_px":
            finite(np.median(errors)),

        "mean_error_px":
            finite(np.mean(errors)),

        "p95_error_px":
            finite(
                np.percentile(
                    errors,
                    95,
                )
            ),

        "worst_case_error_px":
            finite(np.max(errors)),

        "median_runtime_ms":
            finite(np.median(runtimes)),

        "selection_distribution":
            {
                generator:
                    sum(
                        item[
                            "selected_generator"
                        ]
                        == generator
                        for item in selected
                    )
                for generator in GENERATORS
            },

        "selected_rows":
            selected,
    }


# ============================================================================
# Results writer
# ============================================================================

def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
):

    if not rows:
        raise ValueError(
            "Cannot write empty CSV."
        )

    fields = []

    for row in rows:

        for key in row:

            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def write_json(
    data: dict[str, Any],
    path: Path,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            allow_nan=False,
        )


def write_report(
    summary: dict[str, Any],
    path: Path,
):

    report = f"""# MICRONYX Phase B — Automated Representation & Model Selection

## Scope

Phase B replaces permanently hard-coded representation selection with a
data-driven selection layer.

The candidate representations evaluated are:

- NCC
- DoG
- gradient
- edge
- frequency

The implementation compares:

1. rule-based selection
2. optimization-based selection
3. learned selection

Evaluation is performed on held-out scenes.

## Dataset

Observations evaluated: {summary["n_observations"]}

Train observations: {summary["train_observations"]}

Test observations: {summary["test_observations"]}

Scale used for the representation benchmark:
{summary["scale"]}x

## Representation benchmark

{json.dumps(
    summary["generator_benchmark"],
    indent=2,
)}

## Selection policies

{json.dumps(
    summary["policies"],
    indent=2,
)}

## Selected policy

{summary["selected_policy"]}

## Human override

{summary["expert_override"]}

## Leakage protection

EDA features are derived from the observed image.

Target coordinates are evaluation labels only.

Target coordinates are NOT included in:

- EDA features
- rule-policy inputs
- optimization-policy inputs
- learned-policy feature vectors

The learned policy is trained on training scenes and evaluated on
held-out scenes.

## Model-selection boundary

Phase B selects and weights representations.

It does not implement the final production candidate-generation
interface.

Candidate-pool construction, Recall@K, duplicate suppression,
spatial-diversity enforcement and hard-negative mining belong to Phase C.

## Decision principle

No representation is declared universally best.

The selected policy is the policy that provides the best held-out
validation evidence under the configured evaluation metric.

## Reproducibility

Dataset version:
{summary["dataset_version"]}

Phase-A EDA fingerprint:
{summary["eda_source"]}

Random seed:
{summary["seed"]}
"""

    path.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================================
# Dataset fingerprint
# ============================================================================

def file_hash(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def dataset_version(
    dataset_root: Path,
) -> str:

    manifest = (
        dataset_root
        / "manifest.csv"
    )

    return file_hash(
        manifest
    )[:16]


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MICRONYX Phase B representation/model selection"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--eda",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
    )

    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help=(
            "Expert weights, e.g. "
            "dog:0.5,ncc:0.2,gradient:0.1,edge:0.1,frequency:0.1"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError(
            "--scale must be > 0"
        )

    dataset_root = (
        args.dataset.resolve()
    )

    eda_path = (
        args.eda.resolve()
        if args.eda
        else find_default_eda(
            dataset_root
        )
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    eda_rows = load_eda_rows(
        eda_path
    )

    rows = build_observation_rows(
        dataset_root,
        eda_rows,
        args.scale,
    )

    if len(rows) < 4:

        raise RuntimeError(
            "Phase B requires at least four observations "
            "to create a meaningful held-out evaluation."
        )

    # ------------------------------------------------------------------------
    # Split
    #
    # Prefer explicit dataset splits.
    # Otherwise use deterministic 80/20 split.
    # ------------------------------------------------------------------------

    explicit_train = [
        row
        for row in rows
        if str(
            row.get(
                "split",
                ""
            )
        ).lower()
        in {
            "train",
            "training",
        }
    ]

    explicit_test = [
        row
        for row in rows
        if str(
            row.get(
                "split",
                ""
            )
        ).lower()
        in {
            "test",
            "testing",
            "validation",
            "val",
        }
    ]

    if (
        explicit_train
        and explicit_test
    ):

        train_rows = explicit_train
        test_rows = explicit_test

    else:

        rng = np.random.default_rng(
            20260913
        )

        indices = np.arange(
            len(rows)
        )

        rng.shuffle(
            indices
        )

        split_index = max(
            1,
            int(
                0.80
                * len(rows)
            ),
        )

        train_indices = indices[
            :split_index
        ]

        test_indices = indices[
            split_index:
        ]

        if len(test_indices) == 0:

            test_indices = train_indices[
                -1:
            ]

            train_indices = train_indices[
                :-1
            ]

        train_rows = [
            rows[int(i)]
            for i in train_indices
        ]

        test_rows = [
            rows[int(i)]
            for i in test_indices
        ]

    if not train_rows:
        raise RuntimeError(
            "Training split is empty."
        )

    if not test_rows:
        raise RuntimeError(
            "Held-out test split is empty."
        )

    # ------------------------------------------------------------------------
    # Policy 1: Rule-based
    # ------------------------------------------------------------------------

    rule_test = evaluate_policy(
        test_rows,
        "rule_based",
        rule_policy,
    )

    # ------------------------------------------------------------------------
    # Policy 2: Optimization-based
    # ------------------------------------------------------------------------

    train_generator_results = [
        get_result_list(
            row
        )
        for row in train_rows
    ]

    optimized_weights = optimize_weights(
        train_generator_results
    )

    optimization_test = evaluate_policy(
        test_rows,
        "optimization_based",
        lambda row: optimized_weights,
    )

    # ------------------------------------------------------------------------
    # Policy 3: Learned
    # ------------------------------------------------------------------------

    learned_available = True
    learned_test = None
    learned_classifier = None

    try:

        learned_classifier = train_learned_policy(
            train_rows
        )

        learned_test = evaluate_policy(
            test_rows,
            "learned",
            lambda row: learned_weights(
                learned_classifier,
                row,
            ),
        )

    except RuntimeError:

        learned_available = False

    # ------------------------------------------------------------------------
    # Select best policy using held-out median error.
    # ------------------------------------------------------------------------

    policy_results = [
        rule_test,
        optimization_test,
    ]

    if learned_available and learned_test:
        policy_results.append(
            learned_test
        )

    selected_policy_result = min(
        policy_results,
        key=lambda result: (
            result[
                "median_error_px"
            ],
            result[
                "p95_error_px"
            ],
            result[
                "worst_case_error_px"
            ],
        ),
    )

    selected_policy = (
        selected_policy_result[
            "policy"
        ]
    )

    # ------------------------------------------------------------------------
    # Expert override
    # ------------------------------------------------------------------------

    expert_override = None

    if args.override:

        expert_override = validate_override(
            parse_override(
                args.override
            )
        )

    # ------------------------------------------------------------------------
    # Generator-level benchmark
    # ------------------------------------------------------------------------

    generator_benchmark = {}

    for generator in GENERATORS:

        errors = np.asarray([
            float(
                row[
                    f"{generator}_error_px"
                ]
            )
            for row in test_rows
        ])

        runtimes = np.asarray([
            float(
                row[
                    f"{generator}_runtime_ms"
                ]
            )
            for row in test_rows
        ])

        generator_benchmark[
            generator
        ] = {
            "median_error_px":
                finite(
                    np.median(
                        errors
                    )
                ),

            "mean_error_px":
                finite(
                    np.mean(
                        errors
                    )
                ),

            "p95_error_px":
                finite(
                    np.percentile(
                        errors,
                        95,
                    )
                ),

            "worst_case_error_px":
                finite(
                    np.max(errors)
                ),

            "median_runtime_ms":
                finite(
                    np.median(
                        runtimes
                    )
                ),
        }

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary = {
        "phase":
            "Phase B",

        "name":
            "Automated Representation & Model Selection",

        "n_observations":
            len(rows),

        "train_observations":
            len(train_rows),

        "test_observations":
            len(test_rows),

        "scale":
            float(args.scale),

        "seed":
            20260913,

        "dataset_version":
            dataset_version(
                dataset_root
            ),

        "eda_source":
            str(eda_path),

        "generator_benchmark":
            generator_benchmark,

        "optimized_weights":
            optimized_weights,

        "policies":
            {
                "rule_based":
                    {
                        key: value
                        for key, value
                        in rule_test.items()
                        if key != "selected_rows"
                    },

                "optimization_based":
                    {
                        key: value
                        for key, value
                        in optimization_test.items()
                        if key != "selected_rows"
                    },

                "learned":
                    (
                        {
                            key: value
                            for key, value
                            in learned_test.items()
                            if key != "selected_rows"
                        }
                        if learned_test
                        else {
                            "available": False,
                            "reason":
                                "Insufficient class diversity "
                                "or scikit-learn unavailable.",
                        }
                    ),
            },

        "selected_policy":
            selected_policy,

        "expert_override":
            expert_override,
    }

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    result_csv = (
        args.out
        / "representation_selection_results.csv"
    )

    summary_json = (
        args.out
        / "representation_selection_summary.json"
    )

    report_md = (
        args.out
        / "representation_selection_report.md"
    )

    write_csv(
        rows,
        result_csv,
    )

    write_json(
        summary,
        summary_json,
    )

    write_report(
        summary,
        report_md,
    )

    print()
    print(
        "MICRONYX Phase B complete"
    )
    print(
        f"Observations      : {len(rows)}"
    )
    print(
        f"Training          : {len(train_rows)}"
    )
    print(
        f"Held-out          : {len(test_rows)}"
    )
    print(
        f"Selected policy   : {selected_policy}"
    )
    print()
    print(
        "Generator benchmark:"
    )

    for generator in GENERATORS:

        result = generator_benchmark[
            generator
        ]

        print(
            f"  {generator:10s} "
            f"median_error={result['median_error_px']:.4f}px "
            f"p95={result['p95_error_px']:.4f}px"
        )

    print()
    print(
        f"CSV    : {result_csv}"
    )
    print(
        f"JSON   : {summary_json}"
    )
    print(
        f"Report : {report_md}"
    )
    print()


if __name__ == "__main__":
    main()