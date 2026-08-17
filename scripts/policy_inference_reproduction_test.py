from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

EDA_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "automated_eda"
    / "automated_eda_results.csv"
)

POLICY_DATASET_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "eda_guided_multi_generator_policy"
    / "scene_policy_dataset.csv"
)

POLICY_CONFIG_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "eda_guided_multi_generator_policy"
    / "policy_config.json"
)

MODEL_DIR = (
    ROOT
    / "validation"
    / "v02"
    / "eda_guided_multi_generator_policy"
    / "models"
)

RECALL_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "adaptive_candidate_generation_v2"
    / "adaptive_candidate_generation_v2_results.csv"
)


OUTPUT_DIR = (
    ROOT
    / "validation"
    / "v02"
    / "policy_inference_reproduction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# FALLBACK FEATURE LIST
# ============================================================================

FEATURE_COLUMNS = [
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
# HEADER
# ============================================================================

print("=" * 76)
print("MICRONYX STEP 29C")
print("POLICY INFERENCE REPRODUCTION TEST")
print("=" * 76)

print("\nPurpose:")
print("Reproduce Step-28 policy inference independently.")
print("No candidate generation.")
print("No new renderer.")
print("No new ground truth.")
print("No target fingerprint.")
print("No manual policy selection.")


# ============================================================================
# LOAD CONFIG
# ============================================================================

print("\n" + "=" * 76)
print("LOADING STEP-28 CONFIGURATION")
print("=" * 76)

with open(
    POLICY_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)


policies = config["policies"]

feature_columns = config.get(
    "feature_columns",
    FEATURE_COLUMNS,
)

print("\nPolicies:")
for p in policies:
    print(" ", p)

print(
    "\nFeature count:",
    len(feature_columns),
)

print(
    "Selection method:",
    config.get(
        "selection_method",
        "unknown",
    ),
)

print(
    "Decision rule:",
    config.get(
        "decision_rule",
        "unknown",
    ),
)

print(
    "Tie break:",
    config.get(
        "tie_break",
        "unknown",
    ),
)


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 76)
print("LOADING STEP-28 DATASET")
print("=" * 76)

dataset = pd.read_csv(
    POLICY_DATASET_PATH
)

print(
    "Shape:",
    dataset.shape,
)

required = [
    "scene_type",
    "seed",
]

missing = [
    c for c in required + feature_columns
    if c not in dataset.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


print("\n" + "=" * 76)
print("FEATURE MATRIX")
print("=" * 76)

X = (
    dataset[
        feature_columns
    ]
    .copy()
)

X = X.astype(
    np.float32
)

print(
    "X shape:",
    X.shape,
)

print(
    "NaN:",
    int(
        X.isna().sum().sum()
    ),
)

print(
    "Inf:",
    int(
        np.isinf(
            X.to_numpy()
        ).sum()
    ),
)


# ============================================================================
# LOAD ACTUAL STEP-25 RECALL
# ============================================================================

print("\n" + "=" * 76)
print("LOADING STEP-25 RECALL")
print("=" * 76)

recall = pd.read_csv(
    RECALL_PATH
)

recall = recall[
    recall["K"] == 250
].copy()

recall["method"] = (
    recall["method"]
    .astype(str)
    .str.lower()
    .str.strip()
)

print(
    "K=250 rows:",
    len(recall),
)


# ============================================================================
# BUILD ACTUAL ORACLE
# ============================================================================

actual_wide = (
    recall
    .pivot_table(
        index=[
            "scene_type",
            "seed",
        ],
        columns="method",
        values="recall",
        aggfunc="first",
    )
)

actual_wide = actual_wide.reindex(
    columns=policies
)

actual_oracle_policy = (
    actual_wide
    .idxmax(axis=1)
)

actual_oracle_recall = (
    actual_wide
    .max(axis=1)
)


# ============================================================================
# LOAD ALL MODELS
# ============================================================================

print("\n" + "=" * 76)
print("LOADING STEP-28 XGBOOST MODELS")
print("=" * 76)

models = {}

for policy in policies:

    path = (
        MODEL_DIR
        / f"{policy}_xgb.json"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing model: {path}"
        )

    model = xgb.XGBRegressor()

    model.load_model(
        str(path)
    )

    models[policy] = model

    print(
        f"loaded: {policy:24s} "
        f"{path.stat().st_size:,} bytes"
    )

print(
    "\nModels loaded:",
    len(models),
)


# ============================================================================
# MODEL FEATURE COMPATIBILITY
# ============================================================================

print("\n" + "=" * 76)
print("MODEL FEATURE COMPATIBILITY")
print("=" * 76)

for policy, model in models.items():

    model_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_features is not None:

        same = list(
            model_features
        ) == list(
            feature_columns
        )

        print(
            f"{policy:24s} "
            f"features={len(model_features):3d} "
            f"exact_order_match={same}"
        )

    else:

        print(
            f"{policy:24s} "
            f"feature_names_in_=None"
        )


# ============================================================================
# PREDICT ALL POLICIES
# ============================================================================

print("\n" + "=" * 76)
print("GENERATING POLICY PREDICTIONS")
print("=" * 76)

predictions = pd.DataFrame(
    index=dataset.index
)

for policy in policies:

    model = models[policy]

    pred = model.predict(
        X
    )

    predictions[policy] = (
        np.asarray(pred)
        .reshape(-1)
    )

    print(
        f"{policy:24s} "
        f"min={predictions[policy].min(): .6f} "
        f"max={predictions[policy].max(): .6f} "
        f"mean={predictions[policy].mean(): .6f}"
    )


# ============================================================================
# PREDICTED POLICY
# ============================================================================

# Exact Step-28 decision rule:
# select policy with highest predicted Recall@K.
#
# For ties:
# prefer fewer generators.

GENERATOR_COUNT = {
    "ncc": 1,
    "dog": 1,
    "gradient": 1,
    "edge": 1,
    "frequency": 1,
    "ncc_dog": 2,
    "dog_gradient": 2,
    "dog_frequency": 2,
    "dog_edge": 2,
    "dog_gradient_frequency": 3,
    "dog_gradient_edge": 3,
    "all": 5,
}


def select_policy(row):

    best_score = max(
        row[p]
        for p in policies
    )

    candidates = [
        p
        for p in policies
        if np.isclose(
            row[p],
            best_score,
            rtol=1e-10,
            atol=1e-12,
        )
    ]

    candidates.sort(
        key=lambda p: (
            GENERATOR_COUNT[p],
            policies.index(p),
        )
    )

    return candidates[0]


predicted_policy = (
    predictions
    .apply(
        select_policy,
        axis=1,
    )
)

predicted_score = np.array(
    [
        predictions.iloc[i][
            predicted_policy.iloc[i]
        ]
        for i in range(
            len(predictions)
        )
    ]
)


# ============================================================================
# BUILD REPRODUCTION TABLE
# ============================================================================

result = dataset[
    [
        "scene_type",
        "seed",
    ]
].copy()

result[
    "actual_oracle_policy"
] = [
    actual_oracle_policy.loc[
        (
            row.scene_type,
            row.seed,
        )
    ]
    for row in dataset[
        [
            "scene_type",
            "seed",
        ]
    ].itertuples()
]

result[
    "actual_oracle_recall"
] = [
    actual_oracle_recall.loc[
        (
            row.scene_type,
            row.seed,
        )
    ]
    for row in dataset[
        [
            "scene_type",
            "seed",
        ]
    ].itertuples()
]

result[
    "predicted_policy"
] = predicted_policy.values

result[
    "predicted_score"
] = predicted_score


result[
    "policy_match"
] = (
    result[
        "actual_oracle_policy"
    ]
    ==
    result[
        "predicted_policy"
    ]
)


# ============================================================================
# POLICY SELECTION ACCURACY
# ============================================================================

print("\n" + "=" * 76)
print("REPRODUCED POLICY SELECTION")
print("=" * 76)

accuracy = (
    result[
        "policy_match"
    ].mean()
    * 100
)

print(
    f"Selection accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================================
# COMPUTE ACTUAL RECALL OF SELECTED POLICY
# ============================================================================

actual_recall_lookup = (
    recall
    .set_index(
        [
            "scene_type",
            "seed",
            "method",
        ]
    )["recall"]
)

selected_actual_recall = []

for row in result.itertuples():

    key = (
        row.scene_type,
        row.seed,
        row.predicted_policy,
    )

    selected_actual_recall.append(
        actual_recall_lookup.loc[key]
    )

result[
    "selected_actual_recall"
] = selected_actual_recall


automated_recall = (
    result[
        "selected_actual_recall"
    ].mean()
    * 100
)

oracle_recall = (
    result[
        "actual_oracle_recall"
    ].mean()
    * 100
)

regret = (
    result[
        "actual_oracle_recall"
    ]
    -
    result[
        "selected_actual_recall"
    ]
).mean() * 100


print(
    f"Automated Recall@250: "
    f"{automated_recall:.2f}%"
)

print(
    f"Oracle Recall@250:    "
    f"{oracle_recall:.2f}%"
)

print(
    f"Mean regret:          "
    f"{regret:.2f} pp"
)


# ============================================================================
# SCENE TYPE
# ============================================================================

print("\n" + "=" * 76)
print("SCENE-TYPE REPRODUCTION")
print("=" * 76)

for scene_type in [
    "periodic",
    "quasiperiodic",
]:

    sub = result[
        result["scene_type"]
        == scene_type
    ]

    acc = (
        sub["policy_match"]
        .mean()
        * 100
    )

    rec = (
        sub[
            "selected_actual_recall"
        ]
        .mean()
        * 100
    )

    oracle = (
        sub[
            "actual_oracle_recall"
        ]
        .mean()
        * 100
    )

    print(
        f"\n{scene_type.upper()}"
    )

    print(
        f"Scenes:              {len(sub)}"
    )

    print(
        f"Selection accuracy:  {acc:.2f}%"
    )

    print(
        f"Automated recall:    {rec:.2f}%"
    )

    print(
        f"Oracle recall:       {oracle:.2f}%"
    )


# ============================================================================
# POLICY DISTRIBUTION
# ============================================================================

print("\n" + "=" * 76)
print("REPRODUCED POLICY DISTRIBUTION")
print("=" * 76)

print(
    result[
        "predicted_policy"
    ]
    .value_counts()
    .to_string()
)


# ============================================================================
# IMPORTANT MODEL SCORE DIAGNOSTIC
# ============================================================================

print("\n" + "=" * 76)
print("MODEL SCORE DIAGNOSTIC")
print("=" * 76)

for scene_type in [
    "periodic",
    "quasiperiodic",
]:

    sub_indices = dataset.index[
        dataset["scene_type"]
        == scene_type
    ]

    print(
        f"\n{scene_type.upper()}"
    )

    for policy in policies:

        values = predictions.loc[
            sub_indices,
            policy,
        ]

        print(
            f"{policy:24s} "
            f"mean={values.mean(): .6f} "
            f"std={values.std(): .6f}"
        )


# ============================================================================
# FIRST 20 DETAILED ROWS
# ============================================================================

print("\n" + "=" * 76)
print("FIRST 20 SCENE DECISIONS")
print("=" * 76)

print(
    result.head(20).to_string(
        index=False
    )
)


# ============================================================================
# SAVE
# ============================================================================

result_path = (
    OUTPUT_DIR
    / "policy_inference_reproduction.csv"
)

result.to_csv(
    result_path,
    index=False,
)

predictions_path = (
    OUTPUT_DIR
    / "policy_model_predictions.csv"
)

predictions_out = dataset[
    [
        "scene_type",
        "seed",
    ]
].copy()

for policy in policies:
    predictions_out[
        f"pred_{policy}"
    ] = predictions[policy]

predictions_out.to_csv(
    predictions_path,
    index=False,
)


summary = {
    "step": "29C",
    "models_loaded": len(models),
    "feature_count": len(feature_columns),
    "selection_accuracy_percent": float(
        accuracy
    ),
    "automated_recall_percent": float(
        automated_recall
    ),
    "oracle_recall_percent": float(
        oracle_recall
    ),
    "mean_regret_percentage_points": float(
        regret
    ),
    "policy_distribution":
        result[
            "predicted_policy"
        ]
        .value_counts()
        .to_dict(),
}


summary_path = (
    OUTPUT_DIR
    / "policy_inference_reproduction_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================================
# FINAL
# ============================================================================

print("\n" + "=" * 76)
print("STEP 29C COMPLETE")
print("=" * 76)

print("\nSaved:")
print(result_path)
print(predictions_path)
print(summary_path)

print("\nInterpretation:")
print(
    "If Step-29C reproduces approximately "
    "the Step-28 automated result (~76.67%), "
    "the XGBoost models and inference are valid "
    "and the bug is downstream in Step-29 execution."
)

print(
    "If Step-29C instead selects one policy for "
    "almost every scene, the problem is in model "
    "inference, feature preprocessing, or model "
    "selection logic."
)