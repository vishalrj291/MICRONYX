"""
MICRONYX STEP 29
POLICY-CONSTRAINED CANDIDATE GENERATION

Step 28 XGBoost policy models are used to automatically select
the candidate-generation policy from canonical EDA features.

IMPORTANT:
- Canonical renderer only.
- Exact 37 Step-26 EDA features.
- Step-28 XGBoost REGRESSORS, not classifiers.
- No target fingerprint.
- No target coordinates are supplied to the policy model.
- No new ground truth.
- No manual generator selection.
- Candidate generation is actually executed after policy selection.
- Ground truth is used only for evaluation.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
VALIDATION = ROOT / "validation" / "v02"

EDA_FILE = (
    VALIDATION
    / "automated_eda"
    / "automated_eda_results.csv"
)

POLICY_DIR = (
    VALIDATION
    / "eda_guided_multi_generator_policy"
)

MODEL_DIR = POLICY_DIR / "models"

CONFIG_FILE = POLICY_DIR / "policy_config.json"

OUTPUT_DIR = (
    VALIDATION
    / "policy_constrained_candidate_generation_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = (
    OUTPUT_DIR
    / "policy_constrained_v2_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "policy_constrained_v2_summary.json"
)


# ============================================================================
# EXPERIMENT CONFIG
# ============================================================================

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

SEEDS = list(
    range(
        20260875,
        20260905,
    )
)

K_VALUES = [
    10,
    25,
    50,
    100,
    250,
]

CANDIDATE_TOLERANCE = 5.0

GENERATORS = [
    "ncc",
    "dog",
    "gradient",
    "edge",
    "frequency",
]


# ============================================================================
# CANONICAL PHYSICAL TARGET
# ============================================================================
#
# These coordinates are used to generate the canonical observation and
# ONLY for evaluation of candidate recall.
#
# They are NEVER passed into the policy model.
# ============================================================================

TARGET_X = 75.25
TARGET_Y = 113.75

PIXELS_PER_UNIT = 5.0

TARGET_SEARCH_X = (
    TARGET_X * PIXELS_PER_UNIT
)

TARGET_SEARCH_Y = (
    TARGET_Y * PIXELS_PER_UNIT
)


# ============================================================================
# CANONICAL RENDERER
# ============================================================================

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_renderer as CR


# ============================================================================
# EXACT STEP-28 FEATURE SCHEMA
# ============================================================================

EXPECTED_FEATURES = [
    "search_mean",
    "search_std",
    "search_min",
    "search_max",
    "search_p01",
    "search_p05",
    "search_p25",
    "search_median",
    "search_p75",
    "search_p95",
    "search_p99",
    "search_global_contrast",
    "search_rms_contrast",
    "search_dynamic_range",
    "search_gradient_mean",
    "search_gradient_std",
    "search_gradient_max",
    "search_gradient_p95",
    "search_edge_density",
    "search_edge_pixels",
    "search_laplacian_mean",
    "search_laplacian_std",
    "search_laplacian_energy",
    "search_fft_total_energy",
    "search_fft_non_dc_energy",
    "search_fft_low_ratio",
    "search_fft_mid_ratio",
    "search_fft_high_ratio",
    "search_spectral_entropy",
    "search_autocorr_x_max",
    "search_autocorr_y_max",
    "search_autocorr_x_mean",
    "search_autocorr_y_mean",
    "search_periodicity_indicator",
    "search_local_std_mean",
    "search_local_std_std",
    "search_local_std_p95",
]


# ============================================================================
# IMAGE HELPERS
# ============================================================================

def ensure_gray(image):

    if image.ndim == 3:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    return image


def normalize(image):

    image = image.astype(
        np.float32
    )

    mn = float(
        np.min(image)
    )

    mx = float(
        np.max(image)
    )

    if mx - mn < 1e-12:

        return np.zeros_like(
            image
        )

    return (
        image - mn
    ) / (
        mx - mn
    )


# ============================================================================
# GENERATOR REPRESENTATIONS
# ============================================================================

def dog_representation(image):

    image = image.astype(
        np.float32
    )

    g1 = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0,
    )

    g2 = cv2.GaussianBlur(
        image,
        (0, 0),
        2.5,
    )

    return g1 - g2


def gradient_representation(image):

    image = image.astype(
        np.float32
    )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    return np.sqrt(
        gx * gx + gy * gy
    )


def edge_representation(image):

    image8 = np.uint8(
        np.clip(
            normalize(image)
            * 255.0,
            0,
            255,
        )
    )

    return cv2.Canny(
        image8,
        50,
        150,
    ).astype(
        np.float32
    )


def frequency_representation(image):

    image = image.astype(
        np.float32
    )

    wy = np.hanning(
        image.shape[0]
    )

    wx = np.hanning(
        image.shape[1]
    )

    window = np.outer(
        wy,
        wx,
    )

    weighted = (
        image * window
    )

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            weighted
        )
    )

    return np.log1p(
        np.abs(
            spectrum
        )
    ).astype(
        np.float32
    )


# ============================================================================
# TEMPLATE MATCHING
# ============================================================================

def match_template(
    search,
    template,
):

    search = normalize(
        search
    )

    template = normalize(
        template
    )

    return cv2.matchTemplate(
        search.astype(
            np.float32
        ),
        template.astype(
            np.float32
        ),
        cv2.TM_CCOEFF_NORMED,
    )


def extract_candidates(
    response,
    K,
):

    flat = response.reshape(
        -1
    )

    K = min(
        K,
        len(flat),
    )

    indices = np.argpartition(
        flat,
        -K,
    )[-K:]

    indices = indices[
        np.argsort(
            flat[indices]
        )[::-1]
    ]

    h, w = response.shape

    candidates = []

    for idx in indices:

        y, x = np.unravel_index(
            int(idx),
            (h, w),
        )

        candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "score": float(
                    flat[idx]
                ),
            }
        )

    return candidates


# ============================================================================
# SINGLE GENERATOR
# ============================================================================

def generate_candidates(
    search,
    template,
    generator,
    K,
):

    if generator == "ncc":

        s = search
        t = template

    elif generator == "dog":

        s = dog_representation(
            search
        )

        t = dog_representation(
            template
        )

    elif generator == "gradient":

        s = gradient_representation(
            search
        )

        t = gradient_representation(
            template
        )

    elif generator == "edge":

        s = edge_representation(
            search
        )

        t = edge_representation(
            template
        )

    elif generator == "frequency":

        s = frequency_representation(
            search
        )

        t = frequency_representation(
            template
        )

    else:

        raise ValueError(
            f"Unknown generator: {generator}"
        )

    response = match_template(
        s,
        t,
    )

    return extract_candidates(
        response,
        K,
    )


# ============================================================================
# CANDIDATE UNION / DEDUPLICATION
# ============================================================================

def distance(a, b):

    return math.hypot(
        a["x"] - b["x"],
        a["y"] - b["y"],
    )


def union_candidates(
    generator_outputs,
    K,
):

    merged = []

    for generator, candidates in (
        generator_outputs.items()
    ):

        for candidate in candidates:

            item = dict(
                candidate
            )

            item[
                "generator"
            ] = generator

            merged.append(
                item
            )

    merged.sort(
        key=lambda z: z["score"],
        reverse=True,
    )

    output = []

    for candidate in merged:

        duplicate = False

        for existing in output:

            if distance(
                candidate,
                existing,
            ) <= 1.0:

                duplicate = True
                break

        if not duplicate:

            output.append(
                candidate
            )

        if len(output) >= K:
            break

    return output


# ============================================================================
# EDA LOADING
# ============================================================================

def load_eda():

    if not EDA_FILE.exists():

        raise FileNotFoundError(
            f"EDA file missing:\n{EDA_FILE}"
        )

    df = pd.read_csv(
        EDA_FILE
    )

    required = {
        "scene_type",
        "seed",
        *EXPECTED_FEATURES,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Step-26 EDA is missing "
            f"required columns:\n{sorted(missing)}"
        )

    return df


# ============================================================================
# POLICY CONFIG
# ============================================================================

def load_config():

    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            f"Policy config missing:\n{CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    features = config.get(
        "feature_columns"
    )

    if features != EXPECTED_FEATURES:

        raise RuntimeError(
            "Step-28 feature schema does not "
            "match expected schema."
        )

    return config


# ============================================================================
# LOAD XGBOOST REGRESSION MODELS
# ============================================================================

def load_models(
    policies
):

    models = {}

    print()
    print(
        "Loading Step-28 XGBoost regression models..."
    )

    for policy in policies:

        model_path = (
            MODEL_DIR
            / f"{policy}_xgb.json"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Missing policy model:\n{model_path}"
            )

        model = xgb.XGBRegressor()

        model.load_model(
            str(model_path)
        )

        models[policy] = model

        print(
            f"  loaded: {policy}"
        )

    if len(models) != len(
        policies
    ):

        raise RuntimeError(
            "Not all Step-28 models loaded."
        )

    return models


# ============================================================================
# BUILD EXACT 37-FEATURE MATRIX
# ============================================================================

def feature_vector(
    eda_row
):

    values = []

    for feature in EXPECTED_FEATURES:

        value = eda_row[
            feature
        ]

        try:

            value = float(
                value
            )

        except Exception:

            value = 0.0

        if not np.isfinite(
            value
        ):

            value = 0.0

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )


# ============================================================================
# AUTOMATED POLICY PREDICTION
# ============================================================================

def predict_policy(
    models,
    X,
    policies,
):

    predictions = []

    for policy in policies:

        model = models[
            policy
        ]

        prediction = float(
            model.predict(X)[0]
        )

        predictions.append(
            (
                policy,
                prediction,
            )
        )

    # Step-28 decision rule:
    # highest predicted recall
    #
    # Tie-break:
    # fewer generators

    def tie_key(item):

        policy, prediction = item

        generator_count = len(
            policy.split("_")
        )

        if policy == "all":

            generator_count = len(
                GENERATORS
            )

        return (
            -prediction,
            generator_count,
        )

    predictions.sort(
        key=tie_key
    )

    return predictions[0], predictions


# ============================================================================
# POLICY -> GENERATORS
# ============================================================================

def policy_generators(
    policy
):

    if policy == "all":

        return list(
            GENERATORS
        )

    return policy.split(
        "_"
    )


# ============================================================================
# EVALUATION
# ============================================================================

def recall_at_tolerance(
    candidates,
    tolerance,
):

    for candidate in candidates:

        if math.hypot(
            candidate["x"]
            - TARGET_SEARCH_X,
            candidate["y"]
            - TARGET_SEARCH_Y,
        ) <= tolerance:

            return 1

    return 0


def top1_error(
    candidates
):

    if not candidates:

        return float(
            "inf"
        )

    candidate = candidates[
        0
    ]

    return math.hypot(
        candidate["x"]
        - TARGET_SEARCH_X,
        candidate["y"]
        - TARGET_SEARCH_Y,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 76)
    print(
        "MICRONYX STEP 29 — CORRECTED"
    )
    print(
        "POLICY-CONSTRAINED CANDIDATE GENERATION"
    )
    print("=" * 76)

    print()
    print(
        "Canonical renderer:"
    )

    print(
        CR.__file__
    )

    # ------------------------------------------------------------------------
    # LOAD DATA / CONFIG
    # ------------------------------------------------------------------------

    print()
    print(
        "Loading Step-26 EDA..."
    )

    eda = load_eda()

    print(
        f"EDA rows: {len(eda)}"
    )

    print()
    print(
        "Loading Step-28 policy configuration..."
    )

    config = load_config()

    policies = config[
        "policies"
    ]

    print(
        "Policies:"
    )

    for p in policies:

        print(
            f"  {p}"
        )

    models = load_models(
        policies
    )

    print()
    print(
        f"Policy models loaded: {len(models)}"
    )

    if len(models) != len(
        policies
    ):

        raise RuntimeError(
            "Model count mismatch."
        )

    # ------------------------------------------------------------------------
    # EDA LOOKUP
    # ------------------------------------------------------------------------

    eda_lookup = {}

    for _, row in eda.iterrows():

        key = (
            str(
                row["scene_type"]
            ),
            int(
                row["seed"]
            ),
        )

        eda_lookup[key] = row

    # ------------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------------

    results = []

    policy_counts = {}

    total_start = (
        time.perf_counter()
    )

    scene_number = 0

    for scene_type in SCENE_TYPES:

        print()
        print("=" * 76)
        print(
            f"SCENE TYPE: {scene_type.upper()}"
        )
        print("=" * 76)

        for seed in SEEDS:

            scene_number += 1

            print(
                f"Scene {scene_number}/60"
            )

            key = (
                scene_type,
                seed,
            )

            if key not in eda_lookup:

                raise RuntimeError(
                    f"Missing EDA row: {key}"
                )

            row = eda_lookup[
                key
            ]

            # --------------------------------------------------------------
            # CANONICAL OBSERVATION
            # --------------------------------------------------------------

            search = CR.render_search(
                TARGET_X,
                TARGET_Y,
                scene_type,
                seed,
            )

            reference = CR.render_reference(
                TARGET_X,
                TARGET_Y,
                scene_type,
                seed,
            )

            search = ensure_gray(
                search
            )

            reference = ensure_gray(
                reference
            )

            template = (
                CR.create_ps02_template(
                    reference
                )
            )

            # --------------------------------------------------------------
            # POLICY PREDICTION
            # --------------------------------------------------------------

            X = feature_vector(
                row
            )

            (
                selected,
                predictions,
            ) = predict_policy(
                models,
                X,
                policies,
            )

            selected_policy = (
                selected[0]
            )

            predicted_recall = (
                selected[1]
            )

            generators = (
                policy_generators(
                    selected_policy
                )
            )

            policy_counts[
                selected_policy
            ] = (
                policy_counts.get(
                    selected_policy,
                    0,
                )
                + 1
            )

            # --------------------------------------------------------------
            # ACTUAL GENERATOR EXECUTION
            # --------------------------------------------------------------

            generator_outputs = {}

            runtime_start = (
                time.perf_counter()
            )

            for generator in generators:

                generator_outputs[
                    generator
                ] = generate_candidates(
                    search,
                    template,
                    generator,
                    max(K_VALUES),
                )

            generator_runtime = (
                time.perf_counter()
                - runtime_start
            )

            # --------------------------------------------------------------
            # EVALUATION AT ALL K
            # --------------------------------------------------------------

            for K in K_VALUES:

                selected_candidates = (
                    union_candidates(
                        {
                            g: candidates[:K]
                            for g, candidates
                            in generator_outputs.items()
                        },
                        K,
                    )
                )

                recall = (
                    recall_at_tolerance(
                        selected_candidates,
                        CANDIDATE_TOLERANCE,
                    )
                )

                error = (
                    top1_error(
                        selected_candidates
                    )
                )

                results.append(
                    {
                        "scene_type":
                            scene_type,
                        "seed":
                            seed,
                        "selected_policy":
                            selected_policy,
                        "predicted_policy_recall":
                            predicted_recall,
                        "generators":
                            ",".join(
                                generators
                            ),
                        "generator_count":
                            len(
                                generators
                            ),
                        "K":
                            K,
                        "candidate_count":
                            len(
                                selected_candidates
                            ),
                        "recall_5px":
                            recall,
                        "top1_error_px":
                            error,
                        "runtime_sec":
                            generator_runtime,
                    }
                )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        RESULT_FILE,
        index=False,
    )

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "SUMMARY"
    )
    print("=" * 76)

    summary = {}

    for scene_type in SCENE_TYPES:

        sub = df[
            df["scene_type"]
            == scene_type
        ]

        print()
        print(
            scene_type.upper()
        )

        print("-" * 76)

        scene_summary = {}

        for K in K_VALUES:

            s = sub[
                sub["K"] == K
            ]

            recall = (
                s["recall_5px"]
                .mean()
                * 100.0
            )

            median_error = (
                s["top1_error_px"]
                .median()
            )

            print(
                f"K={K:<4} "
                f"Recall@5px={recall:6.2f}% "
                f"MedianError={median_error:8.3f}px"
            )

            scene_summary[
                str(K)
            ] = {
                "recall_percent":
                    float(
                        recall
                    ),
                "median_error_px":
                    float(
                        median_error
                    ),
            }

        summary[
            scene_type
        ] = scene_summary

    print()
    print("=" * 76)
    print(
        "AUTOMATED POLICY DISTRIBUTION"
    )
    print("=" * 76)

    for policy, count in sorted(
        policy_counts.items(),
        key=lambda x: -x[1],
    ):

        print(
            f"{policy:<25} "
            f"{count}"
        )

    print()
    print("=" * 76)
    print(
        "COMPUTATIONAL COST"
    )
    print("=" * 76)

    mean_generators = (
        df[
            df["K"]
            == max(K_VALUES)
        ]["generator_count"]
        .mean()
    )

    mean_runtime = (
        df[
            df["K"]
            == max(K_VALUES)
        ]["runtime_sec"]
        .mean()
    )

    print(
        f"Mean generators/scene: "
        f"{mean_generators:.2f}"
    )

    print(
        f"Mean generator runtime: "
        f"{mean_runtime:.4f} sec"
    )

    # ------------------------------------------------------------------------
    # METHODOLOGY
    # ------------------------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "VALIDITY / METHODOLOGY"
    )
    print("=" * 76)

    print(
        "Canonical renderer:       YES"
    )

    print(
        "Step-26 EDA features:     YES"
    )

    print(
        "Step-28 XGBoost regressors: YES"
    )

    print(
        "Target fingerprint:       NONE"
    )

    print(
        "New ground truth:         NO"
    )

    print(
        "Manual policy selection:  NO"
    )

    print(
        "Policy execution:         YES"
    )

    print(
        "Candidate tolerance:      5 px"
    )

    print(
        f"Runtime: {total_runtime:.2f} seconds"
    )

    # ------------------------------------------------------------------------
    # SUMMARY JSON
    # ------------------------------------------------------------------------

    payload = {
        "step": 29,
        "name":
            "Policy-Constrained Candidate Generation",
        "canonical_renderer": True,
        "step26_eda": True,
        "step28_xgboost_regressors": True,
        "target_fingerprint": False,
        "new_ground_truth": False,
        "manual_policy_selection": False,
        "policy_execution": True,
        "policies": policies,
        "policy_distribution":
            policy_counts,
        "performance":
            summary,
        "runtime_seconds":
            total_runtime,
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:"
    )

    print(
        RESULT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print("=" * 76)
    print(
        "STEP 29 COMPLETE"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()