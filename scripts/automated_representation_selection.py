from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# MICRONYX STEP 27
# AUTOMATED REPRESENTATION SELECTION
# ============================================================
#
# Step 26:
#   Automated EDA / scene characterization
#
# Step 25:
#   Canonical adaptive candidate-generation recall
#
# Step 27:
#   Learn which representation is most suitable from EDA.
#
# IMPORTANT:
#   This is a research baseline for automated selection.
#
#   It does NOT:
#       - inject target fingerprints
#       - create new ground truth
#       - use an alternate renderer
#       - manually choose a representation per scene
#
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

EDA_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "automated_eda"
    / "automated_eda_results.csv"
)

RECALL_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "adaptive_candidate_generation_v2"
    / "adaptive_candidate_generation_v2_results.csv"
)

OUT_DIR = (
    ROOT
    / "validation"
    / "v02"
    / "automated_representation_selection"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


REPRESENTATIONS = [
    "ncc",
    "dog",
    "gradient",
    "edge",
    "frequency",
]

K_VALUES = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]

PRIMARY_K = 250


# ============================================================
# HELPERS
# ============================================================

def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def print_header(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ============================================================
# LOAD EDA
# ============================================================

def load_eda():

    if not EDA_PATH.exists():
        raise FileNotFoundError(
            f"\nStep 26 EDA file not found:\n{EDA_PATH}\n\n"
            "Run:\n"
            "python .\\scripts\\automated_eda.py"
        )

    df = pd.read_csv(EDA_PATH)

    required = {
        "scene_type",
        "seed",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Step 26 CSV missing columns: {missing}"
        )

    return df


# ============================================================
# LOAD RECALL
# ============================================================

def load_recall():

    if not RECALL_PATH.exists():
        raise FileNotFoundError(
            f"\nStep 25 recall file not found:\n{RECALL_PATH}\n\n"
            "Run:\n"
            "python .\\scripts\\adaptive_candidate_generation_v2.py"
        )

    df = pd.read_csv(RECALL_PATH)

    required = {
        "scene_type",
        "seed",
        "method",
        "K",
        "recall",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Step 25 CSV missing columns: {missing}"
        )

    df["method"] = (
        df["method"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["K"] = df["K"].astype(int)
    df["recall"] = df["recall"].astype(float)

    return df


# ============================================================
# EDA FEATURES
# ============================================================

def get_eda_features(eda):

    numeric = eda.select_dtypes(
        include=[np.number]
    ).copy()

    # Remove identifiers / dimensions.
    remove = [
        "seed",
        "search_height",
        "search_width",
        "reference_height",
        "reference_width",
        "template_height",
        "template_width",
    ]

    for c in remove:

        if c in numeric.columns:
            numeric.drop(
                columns=c,
                inplace=True,
            )

    # Only search observation characteristics.
    feature_columns = [
        c
        for c in numeric.columns
        if c.startswith("search_")
    ]

    return feature_columns


# ============================================================
# BUILD LONG EDA-PERFORMANCE TABLE
# ============================================================

def build_long_table(
    eda,
    recall,
):

    feature_columns = get_eda_features(
        eda
    )

    eda_small = eda[
        [
            "scene_type",
            "seed",
        ]
        + feature_columns
    ].copy()

    merged = recall.merge(
        eda_small,
        on=[
            "scene_type",
            "seed",
        ],
        how="inner",
    )

    return merged, feature_columns


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(
    merged,
    feature_columns,
):

    results = []

    subset = merged[
        merged["K"] == PRIMARY_K
    ]

    for method in REPRESENTATIONS:

        method_data = subset[
            subset["method"] == method
        ]

        for feature in feature_columns:

            x = method_data[
                feature
            ].astype(float)

            y = method_data[
                "recall"
            ].astype(float)

            valid = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            x = x[valid]
            y = y[valid]

            if (
                len(x) < 3
                or x.nunique() < 2
                or y.nunique() < 2
            ):
                corr = 0.0

            else:

                corr = float(
                    np.corrcoef(
                        x,
                        y,
                    )[0, 1]
                )

                if not np.isfinite(corr):
                    corr = 0.0

            results.append({
                "method": method,
                "feature": feature,
                "pearson_correlation": corr,
                "absolute_correlation": abs(corr),
                "samples": len(x),
            })

    return pd.DataFrame(results)


# ============================================================
# FIND BEST REPRESENTATION PER SCENE
# ============================================================

def create_scene_targets(
    recall,
):

    subset = recall[
        recall["K"] == PRIMARY_K
    ].copy()

    targets = []

    for (
        scene_type,
        seed,
    ), group in subset.groupby(
        [
            "scene_type",
            "seed",
        ]
    ):

        group = group[
            group["method"].isin(
                REPRESENTATIONS
            )
        ].sort_values(
            "recall",
            ascending=False,
        )

        if group.empty:
            continue

        best_recall = float(
            group.iloc[0]["recall"]
        )

        winners = group[
            np.isclose(
                group["recall"],
                best_recall,
                atol=1e-9,
            )
        ]["method"].tolist()

        targets.append({
            "scene_type": scene_type,
            "seed": int(seed),
            "best_method": winners[0],
            "best_recall": best_recall,
            "winner_count": len(winners),
        })

    return pd.DataFrame(targets)


# ============================================================
# BUILD SCENE-LEVEL TRAINING TABLE
# ============================================================

def build_scene_table(
    eda,
    targets,
):

    feature_columns = get_eda_features(
        eda
    )

    scene_table = eda[
        [
            "scene_type",
            "seed",
        ]
        + feature_columns
    ].copy()

    scene_table = scene_table.merge(
        targets,
        on=[
            "scene_type",
            "seed",
        ],
        how="inner",
    )

    return scene_table, feature_columns


# ============================================================
# STANDARDIZATION
# ============================================================

def fit_scaler(
    train,
    feature_columns,
):

    means = {}
    stds = {}

    for feature in feature_columns:

        values = train[
            feature
        ].astype(float)

        mean = float(
            values.mean()
        )

        std = float(
            values.std()
        )

        if not np.isfinite(std) or std < 1e-12:
            std = 1.0

        means[feature] = mean
        stds[feature] = std

    return means, stds


def scaled_distance(
    row,
    centroid,
    means,
    stds,
    feature_columns,
):

    distances = []

    for feature in feature_columns:

        value = float(
            row[feature]
        )

        if not np.isfinite(value):
            continue

        z = (
            value
            - means[feature]
        ) / stds[feature]

        center = centroid[
            feature
        ]

        distances.append(
            (z - center) ** 2
        )

    if not distances:
        return float("inf")

    return float(
        np.mean(distances)
    )


# ============================================================
# CENTROID SELECTOR
# ============================================================

def fit_centroids(
    train,
    feature_columns,
):

    means, stds = fit_scaler(
        train,
        feature_columns,
    )

    models = {}

    for method in REPRESENTATIONS:

        subset = train[
            train["best_method"]
            == method
        ]

        if subset.empty:
            continue

        centroid = {}

        for feature in feature_columns:

            values = (
                subset[feature]
                .astype(float)
                .to_numpy()
            )

            z = (
                values
                - means[feature]
            ) / stds[feature]

            centroid[feature] = float(
                np.mean(z)
            )

        models[method] = centroid

    return models, means, stds


# ============================================================
# PREDICT
# ============================================================

def predict_method(
    row,
    models,
    means,
    stds,
    feature_columns,
):

    scores = {}

    for method, centroid in models.items():

        scores[method] = scaled_distance(
            row,
            centroid,
            means,
            stds,
            feature_columns,
        )

    if not scores:
        return None, {}

    prediction = min(
        scores,
        key=scores.get,
    )

    return prediction, scores


# ============================================================
# LEAVE-ONE-SEED-OUT VALIDATION
# ============================================================

def cross_validate(
    table,
    feature_columns,
):

    predictions = []

    seeds = sorted(
        table["seed"].unique()
    )

    for seed in seeds:

        train = table[
            table["seed"] != seed
        ]

        test = table[
            table["seed"] == seed
        ]

        models, means, stds = fit_centroids(
            train,
            feature_columns,
        )

        for _, row in test.iterrows():

            predicted, distances = (
                predict_method(
                    row,
                    models,
                    means,
                    stds,
                    feature_columns,
                )
            )

            actual = row[
                "best_method"
            ]

            predictions.append({
                "scene_type": row[
                    "scene_type"
                ],
                "seed": int(seed),
                "actual_best": actual,
                "predicted_best": predicted,
                "correct": (
                    predicted == actual
                ),
                "best_recall": float(
                    row["best_recall"]
                ),
            })

    return pd.DataFrame(
        predictions
    )


# ============================================================
# POLICY PERFORMANCE
# ============================================================

def evaluate_policy(
    predictions,
    recall,
):

    lookup = recall[
        recall["K"] == PRIMARY_K
    ][
        [
            "scene_type",
            "seed",
            "method",
            "recall",
        ]
    ].copy()

    results = []

    for _, row in predictions.iterrows():

        selected_method = row[
            "predicted_best"
        ]

        matches = lookup[
            (
                lookup["scene_type"]
                == row["scene_type"]
            )
            &
            (
                lookup["seed"]
                == row["seed"]
            )
            &
            (
                lookup["method"]
                == selected_method
            )
        ]

        if matches.empty:

            selected_recall = np.nan

        else:

            selected_recall = float(
                matches.iloc[0]["recall"]
            )

        results.append({
            **row.to_dict(),
            "selected_recall": selected_recall,
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print(
        "MICRONYX STEP 27"
    )
    print(
        "AUTOMATED REPRESENTATION SELECTION"
    )
    print("=" * 76)

    print()
    print("EDA source:")
    print(EDA_PATH)

    print()
    print("Recall source:")
    print(RECALL_PATH)

    print()
    print(
        f"Primary K: {PRIMARY_K}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print_header(
        "LOADING STEP 26 EDA"
    )

    eda = load_eda()

    print(
        f"EDA rows: {len(eda)}"
    )

    print(
        f"EDA columns: {len(eda.columns)}"
    )

    print_header(
        "LOADING STEP 25 RECALL"
    )

    recall = load_recall()

    print(
        f"Recall rows: {len(recall)}"
    )

    print(
        "Recall columns:"
    )

    for column in recall.columns:
        print(
            f"  {column}"
        )

    # --------------------------------------------------------
    # LONG TABLE
    # --------------------------------------------------------

    print_header(
        "BUILDING EDA → PERFORMANCE TABLE"
    )

    long_table, feature_columns = (
        build_long_table(
            eda,
            recall,
        )
    )

    print(
        f"Performance rows: "
        f"{len(long_table)}"
    )

    print(
        f"EDA features: "
        f"{len(feature_columns)}"
    )

    print(
        "Representations:",
        REPRESENTATIONS,
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    print_header(
        "EDA / REPRESENTATION CORRELATION"
    )

    correlations = correlation_analysis(
        long_table,
        feature_columns,
    )

    top = (
        correlations
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .head(25)
    )

    for _, row in top.iterrows():

        print(
            f"{row['method']:<12}"
            f"{row['feature']:<42}"
            f"corr="
            f"{row['pearson_correlation']: .4f}"
        )

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    print_header(
        "BEST REPRESENTATION DISTRIBUTION"
    )

    targets = create_scene_targets(
        recall
    )

    counts = (
        targets[
            "best_method"
        ]
        .value_counts()
    )

    for method in REPRESENTATIONS:

        print(
            f"{method:<12}"
            f"{int(counts.get(method, 0)):>6}"
        )

    print()
    print(
        "Total scenes:",
        len(targets),
    )

    # --------------------------------------------------------
    # SCENE TABLE
    # --------------------------------------------------------

    scene_table, scene_features = (
        build_scene_table(
            eda,
            targets,
        )
    )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    print_header(
        "LEAVE-ONE-SEED-OUT SELECTION"
    )

    predictions = cross_validate(
        scene_table,
        scene_features,
    )

    selector_accuracy = (
        predictions["correct"]
        .mean()
        * 100.0
    )

    print(
        f"Selector accuracy: "
        f"{selector_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # POLICY EVALUATION
    # --------------------------------------------------------

    evaluated = evaluate_policy(
        predictions,
        recall,
    )

    selected_recall = (
        evaluated[
            "selected_recall"
        ]
        .dropna()
    )

    oracle_recall = (
        scene_table[
            "best_recall"
        ]
    )

    print()

    print(
        f"Oracle mean Recall@{PRIMARY_K}: "
        f"{oracle_recall.mean() * 100:.2f}%"
    )

    print(
        f"Automated mean Recall@{PRIMARY_K}: "
        f"{selected_recall.mean() * 100:.2f}%"
    )

    print(
        f"Automated median Recall@{PRIMARY_K}: "
        f"{selected_recall.median() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SCENE TYPE
    # --------------------------------------------------------

    print_header(
        "SCENE-TYPE PERFORMANCE"
    )

    for scene_type in sorted(
        evaluated[
            "scene_type"
        ].unique()
    ):

        subset = evaluated[
            evaluated[
                "scene_type"
            ] == scene_type
        ]

        accuracy = (
            subset[
                "correct"
            ].mean()
            * 100.0
        )

        selected = (
            subset[
                "selected_recall"
            ].mean()
            * 100.0
        )

        oracle = (
            subset[
                "best_recall"
            ].mean()
            * 100.0
        )

        print()
        print(
            scene_type.upper()
        )

        print(
            f"Scenes: "
            f"{len(subset)}"
        )

        print(
            f"Selector accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            f"Automated Recall@{PRIMARY_K}: "
            f"{selected:.2f}%"
        )

        print(
            f"Oracle Recall@{PRIMARY_K}: "
            f"{oracle:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    long_path = (
        OUT_DIR
        / "eda_representation_performance.csv"
    )

    long_table.to_csv(
        long_path,
        index=False,
    )

    targets_path = (
        OUT_DIR
        / "scene_representation_targets.csv"
    )

    targets.to_csv(
        targets_path,
        index=False,
    )

    predictions_path = (
        OUT_DIR
        / "representation_selection_predictions.csv"
    )

    evaluated.to_csv(
        predictions_path,
        index=False,
    )

    correlations_path = (
        OUT_DIR
        / "representation_correlations.csv"
    )

    correlations.to_csv(
        correlations_path,
        index=False,
    )

    # --------------------------------------------------------
    # POLICY JSON
    # --------------------------------------------------------

    policy = {
        "step": 27,
        "name": (
            "Automated Representation Selection"
        ),
        "status": "research_baseline",
        "primary_k": PRIMARY_K,
        "representations": REPRESENTATIONS,

        "selection_method": (
            "EDA centroid selector "
            "with leave-one-seed-out validation"
        ),

        "manual_representation_selection": False,
        "target_injection": False,
        "new_ground_truth": False,

        "eda_features": scene_features,

        "selector_accuracy_percent": (
            float(selector_accuracy)
        ),

        "oracle_mean_recall_at_primary_k": (
            float(
                oracle_recall.mean()
            )
        ),

        "automated_mean_recall_at_primary_k": (
            float(
                selected_recall.mean()
            )
        ),

        "automated_median_recall_at_primary_k": (
            float(
                selected_recall.median()
            )
        ),
    }

    policy_path = (
        OUT_DIR
        / "representation_selection_policy.json"
    )

    with open(
        policy_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            policy,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_header(
        "STEP 27 OUTPUTS"
    )

    print(long_path)
    print(targets_path)
    print(predictions_path)
    print(correlations_path)
    print(policy_path)

    print()
    print(
        "METHODOLOGY"
    )

    print(
        "Canonical Step 26 EDA used."
    )

    print(
        "Canonical Step 25 candidate recall used."
    )

    print(
        "No target fingerprint injected."
    )

    print(
        "No new ground truth generated."
    )

    print(
        "No alternate renderer used."
    )

    print(
        "Representation selection is automated."
    )

    print(
        "This is NOT yet the final production selector."
    )

    print()
    print(
        "STEP 27 COMPLETE"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()