from pathlib import Path
import json
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    raise RuntimeError(
        "\nXGBoost is required for Step 28.\n"
        "Install it with:\n"
        "pip install xgboost\n"
    )

warnings.filterwarnings("ignore")


# ============================================================================
# MICRONYX STEP 28
# EDA-GUIDED MULTI-GENERATOR POLICY
# ============================================================================
#
# Objective:
#
#   Learn a scene-dependent generator policy from automated EDA.
#
# Instead of:
#
#       EDA -> choose one representation
#
# we learn:
#
#       EDA
#         |
#         +--> predicted Recall of NCC
#         +--> predicted Recall of DOG
#         +--> predicted Recall of Gradient
#         +--> predicted Recall of Edge
#         +--> predicted Recall of Frequency
#         +--> predicted Recall of combinations
#                     |
#                     v
#              choose best policy
#
#
# Important methodological constraints:
#
#   - Canonical Step 26 EDA only
#   - Canonical Step 25 recall only
#   - No target fingerprint
#   - No alternate renderer
#   - No new GT
#   - No manual scene-specific generator choice
#   - Scene-level cross-validation
#
# ============================================================================


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
    / "eda_guided_multi_generator_policy"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PRIMARY_K = 250


BASE_GENERATORS = [
    "ncc",
    "dog",
    "gradient",
    "edge",
    "frequency",
]


POLICIES = [
    "ncc",
    "dog",
    "gradient",
    "edge",
    "frequency",

    "ncc_dog",

    "dog_gradient",
    "dog_frequency",
    "dog_edge",

    "dog_gradient_frequency",
    "dog_gradient_edge",

    "all",
]


# ============================================================================
# DISPLAY
# ============================================================================

def header(title):

    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ============================================================================
# LOAD EDA
# ============================================================================

def load_eda():

    if not EDA_PATH.exists():

        raise FileNotFoundError(
            f"\nStep 26 EDA not found:\n{EDA_PATH}\n\n"
            "Run:\n"
            "python .\\scripts\\automated_eda.py"
        )

    df = pd.read_csv(
        EDA_PATH
    )

    required = {
        "scene_type",
        "seed",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"EDA missing columns: {missing}"
        )

    return df


# ============================================================================
# LOAD RECALL
# ============================================================================

def load_recall():

    if not RECALL_PATH.exists():

        raise FileNotFoundError(
            f"\nStep 25 recall not found:\n{RECALL_PATH}\n\n"
            "Run:\n"
            "python .\\scripts\\adaptive_candidate_generation_v2.py"
        )

    df = pd.read_csv(
        RECALL_PATH
    )

    required = {
        "scene_type",
        "seed",
        "method",
        "K",
        "recall",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"Recall CSV missing columns: {missing}"
        )

    df["method"] = (
        df["method"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["K"] = (
        df["K"]
        .astype(int)
    )

    df["recall"] = (
        df["recall"]
        .astype(float)
    )

    return df


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def get_features(
    eda
):

    numeric = eda.select_dtypes(
        include=[np.number]
    ).copy()

    ignored = {
        "seed",
        "search_height",
        "search_width",
        "reference_height",
        "reference_width",
        "template_height",
        "template_width",
    }

    feature_columns = [
        c
        for c in numeric.columns
        if c not in ignored
        and c.startswith("search_")
    ]

    if not feature_columns:

        raise RuntimeError(
            "No search EDA features found."
        )

    return feature_columns


# ============================================================================
# BUILD SCENE TABLE
# ============================================================================

def build_scene_table(
    eda,
    recall,
    feature_columns,
):

    scene_features = eda[
        [
            "scene_type",
            "seed",
        ]
        + feature_columns
    ].copy()

    performance = recall[
        recall["K"] == PRIMARY_K
    ].copy()

    performance = performance[
        performance["method"].isin(
            POLICIES
        )
    ]

    # Pivot:
    #
    # scene_type seed
    #       |
    #       +--> ncc recall
    #       +--> dog recall
    #       +--> ...
    #
    pivot = performance.pivot_table(
        index=[
            "scene_type",
            "seed",
        ],
        columns="method",
        values="recall",
        aggfunc="mean",
    ).reset_index()

    pivot.columns.name = None

    scene = scene_features.merge(
        pivot,
        on=[
            "scene_type",
            "seed",
        ],
        how="inner",
    )

    return scene


# ============================================================================
# POLICY SIZE
# ============================================================================

def policy_size(
    policy
):

    if policy == "all":
        return len(BASE_GENERATORS)

    return len(
        policy.split("_")
    )


# ============================================================================
# ORACLE POLICY
# ============================================================================

def get_oracle_policy(
    row
):

    values = {}

    for policy in POLICIES:

        value = row.get(
            policy,
            np.nan
        )

        if np.isfinite(value):

            values[policy] = float(
                value
            )

    if not values:

        return None, np.nan

    # Primary objective:
    #
    #   maximize recall
    #
    # Tie-break:
    #
    #   prefer fewer generators
    #
    best = sorted(
        values.items(),
        key=lambda item: (
            -item[1],
            policy_size(item[0]),
        ),
    )[0]

    return best[0], best[1]


# ============================================================================
# PREPARE FEATURES
# ============================================================================

def prepare_X(
    df,
    feature_columns,
    scaler=None,
    fit=False,
):

    X = (
        df[
            feature_columns
        ]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # Median imputation.
    if fit:

        medians = X.median()

    else:

        medians = None

    if fit:

        X = X.fillna(
            medians
        )

    else:

        X = X.fillna(
            X.median()
        )

    if scaler is None:

        scaler = StandardScaler()

    if fit:

        X_scaled = scaler.fit_transform(
            X
        )

    else:

        X_scaled = scaler.transform(
            X
        )

    return (
        X_scaled,
        scaler,
    )


# ============================================================================
# MODEL
# ============================================================================

def create_model():

    return XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=20260817,
        n_jobs=-1,
    )


# ============================================================================
# SELECT POLICY
# ============================================================================

def choose_policy(
    predictions
):

    valid = {
        p: float(v)
        for p, v in predictions.items()
        if np.isfinite(v)
    }

    if not valid:

        return None

    # Highest predicted recall.
    #
    # If predictions are essentially tied,
    # choose the smaller generator set.
    best = sorted(
        valid.items(),
        key=lambda item: (
            -item[1],
            policy_size(item[0]),
        ),
    )[0]

    return best[0]


# ============================================================================
# LEAVE-ONE-SEED-OUT POLICY LEARNING
# ============================================================================

def cross_validate(
    scene,
    feature_columns,
):

    seeds = sorted(
        scene["seed"].unique()
    )

    predictions = []

    for seed in seeds:

        train = scene[
            scene["seed"] != seed
        ].copy()

        test = scene[
            scene["seed"] == seed
        ].copy()

        if train.empty or test.empty:
            continue

        X_train, scaler = prepare_X(
            train,
            feature_columns,
            fit=True,
        )

        X_test, _ = prepare_X(
            test,
            feature_columns,
            scaler=scaler,
            fit=False,
        )

        models = {}

        # ----------------------------------------------------
        # Train one regression model per generator policy.
        # ----------------------------------------------------

        for policy in POLICIES:

            y = (
                train[policy]
                .astype(float)
                .to_numpy()
            )

            if not np.isfinite(y).any():
                continue

            # Replace invalid target values.
            finite = np.isfinite(y)

            if finite.sum() < 5:
                continue

            X_fit = X_train[
                finite
            ]

            y_fit = y[
                finite
            ]

            model = create_model()

            model.fit(
                X_fit,
                y_fit,
                verbose=False,
            )

            models[policy] = model

        # ----------------------------------------------------
        # Predict recall of every policy.
        # ----------------------------------------------------

        for idx, (_, row) in enumerate(
            test.iterrows()
        ):

            predicted_recall = {}

            actual_recall = {}

            for policy, model in models.items():

                prediction = float(
                    model.predict(
                        X_test[idx:idx + 1]
                    )[0]
                )

                # Recall must be bounded.
                prediction = float(
                    np.clip(
                        prediction,
                        0.0,
                        1.0,
                    )
                )

                predicted_recall[
                    policy
                ] = prediction

                actual_recall[
                    policy
                ] = float(
                    row[policy]
                )

            selected_policy = choose_policy(
                predicted_recall
            )

            if selected_policy is None:
                continue

            actual_selected_recall = (
                actual_recall[
                    selected_policy
                ]
            )

            oracle_policy, oracle_recall = (
                get_oracle_policy(
                    row
                )
            )

            predictions.append({
                "scene_type": row[
                    "scene_type"
                ],
                "seed": int(seed),

                "selected_policy":
                    selected_policy,

                "selected_recall":
                    actual_selected_recall,

                "oracle_policy":
                    oracle_policy,

                "oracle_recall":
                    oracle_recall,

                "regret":
                    oracle_recall
                    - actual_selected_recall,

                "selection_success":
                    selected_policy
                    == oracle_policy,
            })

    return pd.DataFrame(
        predictions
    )


# ============================================================================
# TRAIN FINAL POLICY MODELS
# ============================================================================

def train_final_models(
    scene,
    feature_columns,
):

    X, scaler = prepare_X(
        scene,
        feature_columns,
        fit=True,
    )

    models = {}

    for policy in POLICIES:

        y = (
            scene[policy]
            .astype(float)
            .to_numpy()
        )

        finite = np.isfinite(y)

        if finite.sum() < 5:
            continue

        model = create_model()

        model.fit(
            X[finite],
            y[finite],
            verbose=False,
        )

        models[policy] = model

    return models, scaler


# ============================================================================
# MAIN
# ============================================================================

def main():

    header(
        "MICRONYX STEP 28"
    )

    print(
        "EDA-GUIDED MULTI-GENERATOR POLICY"
    )

    print()
    print(
        "Primary K:",
        PRIMARY_K,
    )

    print()
    print(
        "Objective:"
    )

    print(
        "Automatically predict generator-policy performance "
        "from EDA and select the best policy."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    header(
        "LOADING STEP 26 EDA"
    )

    eda = load_eda()

    print(
        "EDA rows:",
        len(eda),
    )

    print(
        "EDA columns:",
        len(eda.columns),
    )

    header(
        "LOADING STEP 25 RECALL"
    )

    recall = load_recall()

    print(
        "Recall rows:",
        len(recall),
    )

    print(
        "Policies:",
        POLICIES,
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    header(
        "BUILDING SCENE-LEVEL DATASET"
    )

    feature_columns = get_features(
        eda
    )

    scene = build_scene_table(
        eda,
        recall,
        feature_columns,
    )

    print(
        "Scenes:",
        len(scene),
    )

    print(
        "EDA features:",
        len(feature_columns),
    )

    print(
        "Policy columns:",
        len(POLICIES),
    )

    # --------------------------------------------------------
    # ORACLE
    # --------------------------------------------------------

    header(
        "ORACLE POLICY ANALYSIS"
    )

    oracle_rows = []

    for _, row in scene.iterrows():

        policy, value = (
            get_oracle_policy(
                row
            )
        )

        oracle_rows.append({
            "scene_type":
                row["scene_type"],

            "seed":
                int(row["seed"]),

            "oracle_policy":
                policy,

            "oracle_recall":
                value,
        })

    oracle_df = pd.DataFrame(
        oracle_rows
    )

    print()

    print(
        oracle_df[
            "oracle_policy"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Oracle mean Recall@{PRIMARY_K}: "
        f"{oracle_df['oracle_recall'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    header(
        "LEAVE-ONE-SEED-OUT POLICY LEARNING"
    )

    cv = cross_validate(
        scene,
        feature_columns,
    )

    if cv.empty:

        raise RuntimeError(
            "Cross-validation produced no predictions."
        )

    selection_accuracy = (
        cv[
            "selection_success"
        ]
        .mean()
        * 100.0
    )

    automated_recall = (
        cv[
            "selected_recall"
        ]
        .mean()
        * 100.0
    )

    oracle_recall = (
        cv[
            "oracle_recall"
        ]
        .mean()
        * 100.0
    )

    regret = (
        cv[
            "regret"
        ]
        .mean()
        * 100.0
    )

    print(
        f"Policy selection accuracy: "
        f"{selection_accuracy:.2f}%"
    )

    print(
        f"Automated Recall@{PRIMARY_K}: "
        f"{automated_recall:.2f}%"
    )

    print(
        f"Oracle Recall@{PRIMARY_K}: "
        f"{oracle_recall:.2f}%"
    )

    print(
        f"Mean recall regret: "
        f"{regret:.2f} percentage points"
    )

    # --------------------------------------------------------
    # SCENE TYPE
    # --------------------------------------------------------

    header(
        "SCENE-TYPE PERFORMANCE"
    )

    for scene_type in sorted(
        cv[
            "scene_type"
        ].unique()
    ):

        subset = cv[
            cv[
                "scene_type"
            ] == scene_type
        ]

        accuracy = (
            subset[
                "selection_success"
            ].mean()
            * 100
        )

        selected = (
            subset[
                "selected_recall"
            ].mean()
            * 100
        )

        oracle = (
            subset[
                "oracle_recall"
            ].mean()
            * 100
        )

        regret_type = (
            subset[
                "regret"
            ].mean()
            * 100
        )

        print()
        print(
            scene_type.upper()
        )

        print(
            f"Scenes: {len(subset)}"
        )

        print(
            f"Policy accuracy: "
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

        print(
            f"Mean regret: "
            f"{regret_type:.2f} pp"
        )

    # --------------------------------------------------------
    # SELECTED POLICY DISTRIBUTION
    # --------------------------------------------------------

    header(
        "AUTOMATED POLICY DISTRIBUTION"
    )

    print(
        cv[
            "selected_policy"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # REGRET ANALYSIS
    # --------------------------------------------------------

    header(
        "POLICY REGRET"
    )

    print(
        cv[
            [
                "scene_type",
                "seed",
                "selected_policy",
                "oracle_policy",
                "selected_recall",
                "oracle_recall",
                "regret",
            ]
        ]
        .sort_values(
            "regret",
            ascending=False,
        )
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TRAIN FINAL MODELS
    # --------------------------------------------------------

    header(
        "TRAINING FINAL POLICY MODELS"
    )

    models, scaler = train_final_models(
        scene,
        feature_columns,
    )

    print(
        "Models trained:",
        len(models),
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_rows = []

    for policy, model in models.items():

        importance = model.feature_importances_

        for feature, value in zip(
            feature_columns,
            importance,
        ):

            importance_rows.append({
                "policy":
                    policy,

                "feature":
                    feature,

                "importance":
                    float(value),
            })

    importance_df = pd.DataFrame(
        importance_rows
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    scene_path = (
        OUT_DIR
        / "scene_policy_dataset.csv"
    )

    scene.to_csv(
        scene_path,
        index=False,
    )

    cv_path = (
        OUT_DIR
        / "policy_cross_validation.csv"
    )

    cv.to_csv(
        cv_path,
        index=False,
    )

    oracle_path = (
        OUT_DIR
        / "oracle_policy_results.csv"
    )

    oracle_df.to_csv(
        oracle_path,
        index=False,
    )

    importance_path = (
        OUT_DIR
        / "policy_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # --------------------------------------------------------
    # MODEL FILES
    # --------------------------------------------------------

    model_dir = (
        OUT_DIR
        / "models"
    )

    model_dir.mkdir(
        exist_ok=True
    )

    for policy, model in models.items():

        safe_name = (
            policy
            .replace(
                "+",
                "_",
            )
        )

        model.save_model(
            str(
                model_dir
                / f"{safe_name}_xgb.json"
            )
        )

    # --------------------------------------------------------
    # POLICY CONFIG
    # --------------------------------------------------------

    config = {

        "step": 28,

        "name":
            "EDA-Guided Multi-Generator Policy",

        "status":
            "research_baseline",

        "primary_k":
            PRIMARY_K,

        "base_generators":
            BASE_GENERATORS,

        "policies":
            POLICIES,

        "selection_method":
            "XGBoost regression per generator policy",

        "decision_rule":
            "select policy with highest predicted Recall@K",

        "tie_break":
            "prefer fewer generators",

        "cross_validation":
            "leave-one-seed-out",

        "target_injection":
            False,

        "new_ground_truth":
            False,

        "alternate_renderer":
            False,

        "manual_generator_selection":
            False,

        "feature_count":
            len(feature_columns),

        "feature_columns":
            feature_columns,

        "policy_selection_accuracy_percent":
            float(selection_accuracy),

        "automated_recall_percent":
            float(automated_recall),

        "oracle_recall_percent":
            float(oracle_recall),

        "mean_regret_percentage_points":
            float(regret),
    }

    config_path = (
        OUT_DIR
        / "policy_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    header(
        "STEP 28 OUTPUTS"
    )

    print(
        scene_path
    )

    print(
        cv_path
    )

    print(
        oracle_path
    )

    print(
        importance_path
    )

    print(
        config_path
    )

    print(
        model_dir
    )

    print()

    print(
        "METHODOLOGY"
    )

    print(
        "Canonical Step 26 EDA used."
    )

    print(
        "Canonical Step 25 recall used."
    )

    print(
        "No target fingerprint injected."
    )

    print(
        "No alternate renderer used."
    )

    print(
        "No scene-specific manual generator selection."
    )

    print(
        "Generator policy selected automatically."
    )

    print(
        "Policy selection evaluated with "
        "leave-one-seed-out validation."
    )

    print()

    print(
        "STEP 28 COMPLETE"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()