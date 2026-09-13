from pathlib import Path
import csv
import json

import cv2
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit(
        "\nXGBoost is not installed.\n"
        "Run:\n"
        "    pip install xgboost\n"
        "\nThen run this script again.\n"
    )

from canonical_renderer import (
    render_search,
    render_sensor,
)


# ============================================================
# MICRONYX STEP 24
# LEARNED CANDIDATE RANKING
#
# Pipeline:
#
# Continuous physical scene
#          ↓
# Canonical renderer
#          ↓
# DOG candidate generation
#          ↓
# Candidate features
#          ↓
# XGBoost candidate classifier
#          ↓
# P(correct candidate)
#          ↓
# Final ranking
#
# IMPORTANT:
# Scene-level train/validation/test split.
#
# Candidates from the same scene NEVER appear in multiple
# splits.
# ============================================================


PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "validation"
    / "v02"
    / "learned_ranker"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "learned_ranker_results.csv"
)

MODEL_JSON = (
    OUTPUT_DIR
    / "xgboost_ranker.json"
)

FEATURE_IMPORTANCE_JSON = (
    OUTPUT_DIR
    / "feature_importance.json"
)


# ============================================================
# DATA SPLIT
# ============================================================

# 45 completely different physical scenes per architecture.
#
# Train:
#   25 seeds × 2 scene types = 50 scenes
#
# Validation:
#   10 seeds × 2 = 20 scenes
#
# Test:
#   10 seeds × 2 = 20 scenes
#
# No seed crosses a split.
# ============================================================

TRAIN_SEEDS = range(
    20260850,
    20260875,
)

VAL_SEEDS = range(
    20260875,
    20260885,
)

TEST_SEEDS = range(
    20260885,
    20260895,
)

SCENE_TYPES = [
    "periodic",
    "quasiperiodic",
]

TOP_K = 250

CONTEXT_SIZES = [
    10,
    20,
    40,
]


# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [
    "dog_score",
    "context_10",
    "context_20",
    "context_40",
    "gradient_score",
    "orientation_score",
    "contrast_score",
    "context_gain_20",
    "context_gain_40",
    "dog_context_gap",
    "context_consistency",
    "dog_rank_normalized",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):

    image = image.astype(
        np.float32
    )

    mean = float(
        np.mean(image)
    )

    std = float(
        np.std(image)
    )

    if std < 1e-8:

        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    return (
        image - mean
    ) / std


# ============================================================
# DOG
# ============================================================

def dog(image):

    image = image.astype(
        np.float32
    )

    g1 = cv2.GaussianBlur(
        image,
        (0, 0),
        0.8,
    )

    g2 = cv2.GaussianBlur(
        image,
        (0, 0),
        2.5,
    )

    return normalize(
        g1 - g2
    )


# ============================================================
# TEMPLATE MATCH
# ============================================================

def template_match(
    search,
    template,
):

    return cv2.matchTemplate(
        search.astype(
            np.float32
        ),
        template.astype(
            np.float32
        ),
        cv2.TM_CCOEFF_NORMED,
    )


# ============================================================
# TOP-K DOG CANDIDATES
# ============================================================

def top_k_candidates(
    response,
    k,
    min_distance=6,
):

    work = response.copy()

    candidates = []

    h, w = work.shape

    for _ in range(k):

        _, score, _, loc = (
            cv2.minMaxLoc(
                work
            )
        )

        x, y = loc

        candidates.append(
            (
                x,
                y,
                float(score),
            )
        )

        x0 = max(
            0,
            x - min_distance,
        )

        x1 = min(
            w,
            x + min_distance + 1,
        )

        y0 = max(
            0,
            y - min_distance,
        )

        y1 = min(
            h,
            y + min_distance + 1,
        )

        work[
            y0:y1,
            x0:x1
        ] = -np.inf

    return candidates


# ============================================================
# PHYSICAL CONTEXT REFERENCE
# ============================================================

def render_context_reference(
    tx,
    ty,
    scene_type,
    seed,
    context_size,
):

    physical_size = (
        context_size / 5.0
    )

    reference_size = int(
        round(
            physical_size
            * 50.0
        )
    )

    origin_x = (
        tx
        - physical_size / 2.0
    )

    origin_y = (
        ty
        - physical_size / 2.0
    )

    reference = render_sensor(
        width=reference_size,
        height=reference_size,
        pixels_per_unit=50.0,
        origin_x=origin_x,
        origin_y=origin_y,
        tx=tx,
        ty=ty,
        scene_type=scene_type,
        seed=seed,
    )

    return cv2.resize(
        reference,
        (
            context_size,
            context_size,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# CENTERED PATCH
# ============================================================

def centered_patch(
    image,
    center_x,
    center_y,
    size,
):

    half = size / 2.0

    x0 = int(
        round(
            center_x - half
        )
    )

    y0 = int(
        round(
            center_y - half
        )
    )

    x1 = (
        x0 + size
    )

    y1 = (
        y0 + size
    )

    h, w = image.shape

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):
        return None

    return image[
        y0:y1,
        x0:x1
    ]


# ============================================================
# GRADIENT FEATURES
# ============================================================

def gradient_features(
    image,
):

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

    magnitude = np.sqrt(
        gx * gx
        + gy * gy
    )

    orientation = np.arctan2(
        gy,
        gx,
    )

    return (
        gx,
        gy,
        magnitude,
        orientation,
    )


# ============================================================
# CONTEXT SCORE
# ============================================================

def context_score(
    search,
    reference,
    center_x,
    center_y,
):

    size = reference.shape[0]

    patch = centered_patch(
        search,
        center_x,
        center_y,
        size,
    )

    if patch is None:
        return -1.0

    return float(
        template_match(
            normalize(patch),
            normalize(reference),
        )[0, 0]
    )


# ============================================================
# CANDIDATE FEATURES
# ============================================================

def extract_candidate_features(
    search,
    references,
    candidate,
    dog_rank,
    max_candidates,
):

    x, y, dog_score = (
        candidate
    )

    # DOG candidate location is top-left.
    center_x = (
        x + 5.0
    )

    center_y = (
        y + 5.0
    )

    # --------------------------------------------------------
    # Context scores
    # --------------------------------------------------------

    context_10 = context_score(
        search,
        references[10],
        center_x,
        center_y,
    )

    context_20 = context_score(
        search,
        references[20],
        center_x,
        center_y,
    )

    context_40 = context_score(
        search,
        references[40],
        center_x,
        center_y,
    )

    # --------------------------------------------------------
    # Gradient features
    # --------------------------------------------------------

    search_gx, search_gy, search_mag, search_ori = (
        gradient_features(
            search
        )
    )

    temp_gx, temp_gy, temp_mag, temp_ori = (
        gradient_features(
            references[10]
        )
    )

    patch_gx = centered_patch(
        search_gx,
        center_x,
        center_y,
        10,
    )

    patch_gy = centered_patch(
        search_gy,
        center_x,
        center_y,
        10,
    )

    patch_mag = centered_patch(
        search_mag,
        center_x,
        center_y,
        10,
    )

    patch_ori = centered_patch(
        search_ori,
        center_x,
        center_y,
        10,
    )

    if (
        patch_gx is None
        or patch_gy is None
        or patch_mag is None
        or patch_ori is None
    ):

        gradient_score = 0.0
        orientation_score = 0.0

    else:

        gradient_score = float(
            np.mean(
                normalize(
                    patch_mag
                )
                * normalize(
                    temp_mag
                )
            )
        )

        delta = (
            patch_ori
            - temp_ori
        )

        orientation_similarity = (
            np.cos(
                delta
            )
        )

        weights = (
            patch_mag
            + temp_mag
            + 1e-6
        )

        orientation_score = float(
            np.sum(
                orientation_similarity
                * weights
            )
            / np.sum(
                weights
            )
        )

    gradient_score = float(
        np.clip(
            gradient_score,
            -1,
            1,
        )
    )

    orientation_score = float(
        np.clip(
            orientation_score,
            -1,
            1,
        )
    )

    # --------------------------------------------------------
    # Local contrast
    # --------------------------------------------------------

    local_patch = centered_patch(
        search,
        center_x,
        center_y,
        10,
    )

    if local_patch is None:

        contrast_score = 0.0

    else:

        contrast_score = float(
            np.mean(
                normalize(
                    local_patch
                )
                * normalize(
                    references[10]
                )
            )
        )

    contrast_score = float(
        np.clip(
            contrast_score,
            -1,
            1,
        )
    )

    # --------------------------------------------------------
    # Derived features
    # --------------------------------------------------------

    context_gain_20 = (
        context_20
        - context_10
    )

    context_gain_40 = (
        context_40
        - context_10
    )

    dog_context_gap = (
        dog_score
        - context_10
    )

    context_consistency = (
        (
            context_10
            + context_20
            + context_40
        )
        / 3.0
    )

    dog_rank_normalized = (
        dog_rank
        / max_candidates
    )

    return np.array(
        [
            dog_score,
            context_10,
            context_20,
            context_40,
            gradient_score,
            orientation_score,
            contrast_score,
            context_gain_20,
            context_gain_40,
            dog_context_gap,
            context_consistency,
            dog_rank_normalized,
        ],
        dtype=np.float32,
    )


# ============================================================
# CREATE ONE SCENE DATASET
# ============================================================

def generate_scene_candidates(
    scene_type,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    tx = (
        75.25
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    ty = (
        113.75
        + rng.uniform(
            -8.0,
            8.0,
        )
    )

    search = render_search(
        tx,
        ty,
        scene_type,
        seed,
    )

    references = {}

    for size in CONTEXT_SIZES:

        references[size] = (
            render_context_reference(
                tx,
                ty,
                scene_type,
                seed,
                size,
            )
        )

    # --------------------------------------------------------
    # DOG generator
    # --------------------------------------------------------

    search_dog = dog(
        search
    )

    reference_dog = dog(
        references[10]
    )

    response = template_match(
        search_dog,
        reference_dog,
    )

    candidates = top_k_candidates(
        response,
        TOP_K,
    )

    gt_x = (
        tx * 5.0
    )

    gt_y = (
        ty * 5.0
    )

    feature_rows = []

    candidate_records = []

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):

        x, y, dog_score = (
            candidate
        )

        center_x = (
            x + 5.0
        )

        center_y = (
            y + 5.0
        )

        distance = float(
            np.hypot(
                center_x - gt_x,
                center_y - gt_y,
            )
        )

        features = (
            extract_candidate_features(
                search,
                references,
                candidate,
                rank,
                TOP_K,
            )
        )

        # ----------------------------------------------------
        # Label
        #
        # Positive = within 5 px.
        # ----------------------------------------------------

        label = int(
            distance <= 5.0
        )

        feature_rows.append(
            features
        )

        candidate_records.append(
            {
                "rank": rank,
                "x": center_x,
                "y": center_y,
                "distance": distance,
                "label": label,
                "features": features,
            }
        )

    return (
        np.array(
            feature_rows,
            dtype=np.float32,
        ),
        np.array(
            [
                c["label"]
                for c in candidate_records
            ],
            dtype=np.int32,
        ),
        candidate_records,
        (
            gt_x,
            gt_y,
        ),
    )


# ============================================================
# BUILD SPLIT
# ============================================================

def build_split(
    seeds,
    split_name,
):

    X = []
    y = []
    metadata = []

    total = (
        len(seeds)
        * len(SCENE_TYPES)
    )

    counter = 0

    print()
    print(
        f"BUILDING {split_name.upper()} SET"
    )

    print(
        "-" * 76
    )

    for scene_type in SCENE_TYPES:

        for seed in seeds:

            counter += 1

            print(
                f"{split_name}: "
                f"{counter:02d}/{total}"
            )

            (
                features,
                labels,
                records,
                gt,
            ) = generate_scene_candidates(
                scene_type,
                seed,
            )

            X.append(
                features
            )

            y.append(
                labels
            )

            metadata.append(
                {
                    "scene":
                        scene_type,
                    "seed":
                        seed,
                    "records":
                        records,
                    "gt":
                        gt,
                }
            )

    X = np.vstack(
        X
    )

    y = np.concatenate(
        y
    )

    return (
        X,
        y,
        metadata,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):

    print()
    print(
        "TRAINING XGBOOST"
    )

    print(
        "-" * 76
    )

    positives = int(
        np.sum(
            y_train == 1
        )
    )

    negatives = int(
        np.sum(
            y_train == 0
        )
    )

    print(
        "Positive candidates:",
        positives,
    )

    print(
        "Negative candidates:",
        negatives,
    )

    scale_pos_weight = (
        negatives
        / max(
            positives,
            1,
        )
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# RANK ONE SCENE
# ============================================================

def rank_scene(
    model,
    scene_metadata,
):

    records = (
        scene_metadata["records"]
    )

    X = np.vstack(
        [
            record["features"]
            for record in records
        ]
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    for record, probability in zip(
        records,
        probabilities,
    ):

        record[
            "probability"
        ] = float(
            probability
        )

    ranked = sorted(
        records,
        key=lambda record:
            record["probability"],
        reverse=True,
    )

    gt_x, gt_y = (
        scene_metadata["gt"]
    )

    # --------------------------------------------------------
    # Find GT candidate rank
    # --------------------------------------------------------

    gt_records = [
        record
        for record in records
        if record["label"] == 1
    ]

    gt_recalled = (
        len(gt_records) > 0
    )

    if gt_recalled:

        best_gt_probability = max(
            record["probability"]
            for record in gt_records
        )

        predicted_gt_rank = (
            1
            + sum(
                record["probability"]
                > best_gt_probability
                for record in records
            )
        )

    else:

        predicted_gt_rank = -1

        best_gt_probability = 0.0

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    best = ranked[0]

    prediction_error = float(
        np.hypot(
            best["x"] - gt_x,
            best["y"] - gt_y,
        )
    )

    # --------------------------------------------------------
    # Top-5
    # --------------------------------------------------------

    top5 = ranked[:5]

    top5_success = any(
        record["distance"] <= 5.0
        for record in top5
    )

    return {
        "pred_x":
            best["x"],
        "pred_y":
            best["y"],
        "error":
            prediction_error,
        "gt_candidate_recalled":
            int(gt_recalled),
        "predicted_gt_rank":
            predicted_gt_rank,
        "best_gt_probability":
            best_gt_probability,
        "top5_success":
            int(top5_success),
        "pred_probability":
            best["probability"],
    }


# ============================================================
# EVALUATE SPLIT
# ============================================================

def evaluate_split(
    model,
    metadata,
    split_name,
):

    results = []

    print()
    print(
        f"EVALUATING {split_name.upper()}"
    )

    print(
        "-" * 76
    )

    for index, scene_metadata in enumerate(
        metadata,
        start=1,
    ):

        scene = scene_metadata[
            "scene"
        ]

        seed = scene_metadata[
            "seed"
        ]

        result = rank_scene(
            model,
            scene_metadata,
        )

        result[
            "scene"
        ] = scene

        result[
            "seed"
        ] = seed

        result[
            "split"
        ] = split_name

        results.append(
            result
        )

        print(
            f"{split_name}: "
            f"{index:02d}/"
            f"{len(metadata)} "
            f"{scene:<15} "
            f"seed={seed} "
            f"error="
            f"{result['error']:8.3f}px "
            f"GT_rank="
            f"{result['predicted_gt_rank']:4d}"
        )

    return results


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    results,
    split_name,
):

    print()
    print(
        "=" * 76
    )

    print(
        f"{split_name.upper()} SUMMARY"
    )

    print(
        "=" * 76
    )

    for scene_type in SCENE_TYPES:

        subset = [
            r
            for r in results
            if (
                r["split"]
                == split_name
                and r["scene"]
                == scene_type
            )
        ]

        errors = np.array(
            [
                r["error"]
                for r in subset
            ]
        )

        ranks = np.array(
            [
                r[
                    "predicted_gt_rank"
                ]
                for r in subset
                if r[
                    "predicted_gt_rank"
                ] > 0
            ]
        )

        recall = np.mean(
            [
                r[
                    "gt_candidate_recalled"
                ]
                for r in subset
            ]
        ) * 100.0

        top1 = np.mean(
            errors < 1.0
        ) * 100.0

        top5 = np.mean(
            errors <= 5.0
        ) * 100.0

        print()
        print(
            scene_type.upper()
        )

        print(
            "-" * 76
        )

        print(
            f"Candidate Recall@5px: "
            f"{recall:6.2f}%"
        )

        print(
            f"Final Top-1 (<1px):   "
            f"{top1:6.2f}%"
        )

        print(
            f"Final <=5px:          "
            f"{top5:6.2f}%"
        )

        print(
            f"Median Error:         "
            f"{np.median(errors):8.3f}px"
        )

        print(
            f"95th percentile:      "
            f"{np.percentile(errors, 95):8.3f}px"
        )

        if len(ranks):

            print(
                f"Median GT rank:       "
                f"{np.median(ranks):8.1f}"
            )

            print(
                f"Best GT rank:         "
                f"{np.min(ranks):8d}"
            )

            print(
                f"Worst GT rank:        "
                f"{np.max(ranks):8d}"
            )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
):

    importance = (
        model.feature_importances_
    )

    data = {
        name: float(value)
        for name, value
        in zip(
            FEATURE_NAMES,
            importance,
        )
    }

    data = dict(
        sorted(
            data.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )
    )

    with open(
        FEATURE_IMPORTANCE_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

    print()
    print(
        "FEATURE IMPORTANCE"
    )

    print(
        "-" * 76
    )

    for name, value in data.items():

        print(
            f"{name:<25}"
            f"{value:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 76)
    print(
        "MICRONYX STEP 24"
    )
    print(
        "LEARNED CANDIDATE RANKING"
    )
    print("=" * 76)

    print()
    print(
        "Model: XGBoost"
    )

    print(
        "Candidate generator: DOG"
    )

    print(
        "Candidate count:",
        TOP_K,
    )

    print()
    print(
        "Scene-level split:"
    )

    print(
        "Train:",
        len(TRAIN_SEEDS),
        "seeds"
    )

    print(
        "Validation:",
        len(VAL_SEEDS),
        "seeds"
    )

    print(
        "Test:",
        len(TEST_SEEDS),
        "seeds"
    )

    # ========================================================
    # BUILD DATA
    # ========================================================

    (
        X_train,
        y_train,
        train_metadata,
    ) = build_split(
        TRAIN_SEEDS,
        "train",
    )

    (
        X_val,
        y_val,
        val_metadata,
    ) = build_split(
        VAL_SEEDS,
        "validation",
    )

    (
        X_test,
        y_test,
        test_metadata,
    ) = build_split(
        TEST_SEEDS,
        "test",
    )

    print()
    print(
        "DATASET SHAPES"
    )

    print(
        "Train:",
        X_train.shape,
        y_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
        y_val.shape,
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    model = train_model(
        X_train,
        y_train,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save_model(
        str(
            MODEL_JSON
        )
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    save_feature_importance(
        model
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_results = (
        evaluate_split(
            model,
            val_metadata,
            "validation",
        )
    )

    summarize(
        validation_results,
        "validation",
    )

    # ========================================================
    # TEST
    # ========================================================

    test_results = (
        evaluate_split(
            model,
            test_metadata,
            "test",
        )
    )

    summarize(
        test_results,
        "test",
    )

    # ========================================================
    # SAVE TEST RESULTS
    # ========================================================

    all_results = (
        validation_results
        + test_results
    )

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print()
    print("=" * 76)

    print(
        "SAVED"
    )

    print(
        OUTPUT_CSV
    )

    print(
        MODEL_JSON
    )

    print(
        FEATURE_IMPORTANCE_JSON
    )

    print("=" * 76)


if __name__ == "__main__":
    main()