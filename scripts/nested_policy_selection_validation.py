"""
MICRONYX STEP 30
NESTED / GROUPED POLICY SELECTION VALIDATION

Purpose
-------
Validate automated generator-policy selection without seed leakage.

Important:
- Canonical Step-26 EDA is used.
- Canonical Step-25 recall is used.
- No target fingerprint is used.
- No new ground truth is generated.
- No manual policy selection.
- Each seed is held out completely.
- Both periodic and quasiperiodic scenes belonging to the held-out seed
  are excluded from training.
- A fresh XGBoost model is trained for every policy inside every fold.
- The held-out scene is used ONLY for final evaluation.

This is the correct experiment for determining whether the automated
policy selector generalizes to unseen scenes.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise RuntimeError(
        "XGBoost is required. Install with:\n"
        "pip install xgboost"
    ) from exc


# ============================================================================
# CONFIGURATION
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
    / "nested_policy_selection"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_K = 250

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
# XGBOOST CONFIG
# ============================================================================

XGB_PARAMS = dict(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.035,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.10,
    reg_lambda=2.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


# ============================================================================
# FEATURE DEFINITIONS
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
# HELPERS
# ============================================================================

def banner(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def load_inputs():
    banner("LOADING STEP-26 EDA")

    if not EDA_PATH.exists():
        raise FileNotFoundError(f"EDA file not found:\n{EDA_PATH}")

    eda = pd.read_csv(EDA_PATH)

    print(f"EDA rows: {len(eda)}")
    print(f"EDA columns: {len(eda.columns)}")

    banner("LOADING STEP-25 RECALL")

    if not RECALL_PATH.exists():
        raise FileNotFoundError(f"Recall file not found:\n{RECALL_PATH}")

    recall = pd.read_csv(RECALL_PATH)

    print(f"Recall rows: {len(recall)}")
    print("Recall columns:")
    for c in recall.columns:
        print(f"  {c}")

    return eda, recall


def validate_eda(eda):
    banner("VALIDATING EDA")

    required = {"scene_type", "seed", *EXPECTED_FEATURES}

    missing = sorted(required - set(eda.columns))

    if missing:
        raise RuntimeError(
            "Step-26 EDA is missing required columns:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    if eda[EXPECTED_FEATURES].isna().any().any():
        raise RuntimeError("EDA contains NaN values.")

    if np.isinf(eda[EXPECTED_FEATURES].to_numpy(dtype=float)).any():
        raise RuntimeError("EDA contains Inf values.")

    print(f"Feature count: {len(EXPECTED_FEATURES)}")
    print(f"NaN count: {int(eda[EXPECTED_FEATURES].isna().sum().sum())}")
    print(
        f"Inf count: "
        f"{int(np.isinf(eda[EXPECTED_FEATURES].to_numpy(dtype=float)).sum())}"
    )

    print()
    print("Scene distribution:")
    print(eda["scene_type"].value_counts())

    print()
    print("Unique seeds:", eda["seed"].nunique())

    # Every seed should have both scene types.
    seed_scene_counts = (
        eda.groupby("seed")["scene_type"]
        .nunique()
    )

    incomplete = seed_scene_counts[seed_scene_counts != 2]

    if len(incomplete):
        print()
        print("WARNING:")
        print("Some seeds do not contain both scene types:")
        print(incomplete)

    return eda


def build_recall_table(recall):
    banner("BUILDING STEP-25 POLICY TARGET TABLE")

    required = {"scene_type", "seed", "method", "K", "recall"}

    missing = sorted(required - set(recall.columns))

    if missing:
        raise RuntimeError(
            "Recall CSV is missing required columns:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    r = recall[recall["K"] == PRIMARY_K].copy()

    print(f"K={PRIMARY_K} rows: {len(r)}")

    methods = sorted(r["method"].unique())

    print("Methods found:")
    for m in methods:
        print(f"  {m}")

    missing_policies = sorted(set(POLICIES) - set(methods))

    if missing_policies:
        raise RuntimeError(
            "Missing policies in Step-25 recall:\n"
            + "\n".join(f"  {x}" for x in missing_policies)
        )

    # There must be exactly one recall value for each
    # scene_type / seed / policy combination.
    dup = (
        r.groupby(["scene_type", "seed", "method"])
        .size()
    )

    bad_dup = dup[dup != 1]

    if len(bad_dup):
        raise RuntimeError(
            "Duplicate or missing scene-policy observations detected."
        )

    table = (
        r.pivot_table(
            index=["scene_type", "seed"],
            columns="method",
            values="recall",
            aggfunc="first",
        )
        .reset_index()
    )

    table.columns.name = None

    missing_after_pivot = [
        p for p in POLICIES if p not in table.columns
    ]

    if missing_after_pivot:
        raise RuntimeError(
            "Policies missing after pivot:\n"
            + "\n".join(missing_after_pivot)
        )

    # Force binary / numeric recall.
    for p in POLICIES:
        table[p] = pd.to_numeric(table[p], errors="raise")

    print(f"Scene-policy rows: {len(table)}")
    print(f"Policy columns: {len(POLICIES)}")

    return table


def merge_eda_recall(eda, recall_table):
    banner("MERGING EDA + POLICY TARGETS")

    key = ["scene_type", "seed"]

    # Ensure EDA has exactly one row per scene.
    if eda.duplicated(key).any():
        raise RuntimeError(
            "EDA contains duplicate scene_type/seed rows."
        )

    merged = eda[key + EXPECTED_FEATURES].merge(
        recall_table[key + POLICIES],
        on=key,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(eda):
        raise RuntimeError(
            f"Merge lost rows: EDA={len(eda)}, merged={len(merged)}"
        )

    print(f"Final scene count: {len(merged)}")
    print(f"Feature count: {len(EXPECTED_FEATURES)}")
    print(f"Policy count: {len(POLICIES)}")

    return merged


def select_policy(predictions):
    """
    Select highest predicted recall.

    Deterministic tie-break:
    fewer generators preferred.
    """

    generator_count = {
        p: len(p.split("_"))
        for p in POLICIES
    }

    ranked = sorted(
        POLICIES,
        key=lambda p: (
            -float(predictions[p]),
            generator_count[p],
            p,
        ),
    )

    return ranked[0]


def train_model(X, y, seed_offset):
    params = dict(XGB_PARAMS)
    params["random_state"] = 1000 + seed_offset

    model = XGBRegressor(**params)

    model.fit(X, y)

    return model


# ============================================================================
# MAIN NESTED VALIDATION
# ============================================================================

def main():

    banner("MICRONYX STEP 30")
    print("NESTED / GROUPED POLICY SELECTION VALIDATION")
    print()
    print("Purpose:")
    print("Validate automated policy selection on unseen seeds.")
    print()
    print("Leakage protection:")
    print("  Held-out seed excluded from training")
    print("  Both scene types held out together")
    print("  Fresh model trained inside every fold")
    print("  Held-out recall NEVER used for policy prediction")
    print()
    print(f"Primary K: {PRIMARY_K}")

    # ---------------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------------

    eda, recall = load_inputs()

    eda = validate_eda(eda)

    recall_table = build_recall_table(recall)

    data = merge_eda_recall(
        eda,
        recall_table,
    )

    X_all = data[EXPECTED_FEATURES].to_numpy(dtype=np.float32)

    # ---------------------------------------------------------------------
    # GROUPS
    # ---------------------------------------------------------------------

    seeds = sorted(data["seed"].unique())

    print()
    print("Grouped validation seeds:")
    print(f"  Number of unique seeds: {len(seeds)}")
    print("  Group definition: seed")
    print("  Each fold holds out both periodic and quasiperiodic scenes.")

    if len(seeds) < 5:
        raise RuntimeError(
            "Too few unique seeds for meaningful grouped validation."
        )

    # ---------------------------------------------------------------------
    # RESULT STORAGE
    # ---------------------------------------------------------------------

    results = []

    policy_predictions = []

    # ---------------------------------------------------------------------
    # GROUPED LEAVE-ONE-SEED-OUT
    # ---------------------------------------------------------------------

    banner("GROUPED LEAVE-ONE-SEED-OUT VALIDATION")

    for fold_idx, held_seed in enumerate(seeds, start=1):

        train_mask = data["seed"] != held_seed
        test_mask = data["seed"] == held_seed

        train_df = data.loc[train_mask].copy()
        test_df = data.loc[test_mask].copy()

        X_train = train_df[EXPECTED_FEATURES].to_numpy(
            dtype=np.float32
        )

        X_test = test_df[EXPECTED_FEATURES].to_numpy(
            dtype=np.float32
        )

        print(
            f"Fold {fold_idx:02d}/{len(seeds)} "
            f"held_seed={held_seed} "
            f"train={len(train_df)} "
            f"test={len(test_df)}"
        )

        fold_models = {}

        # -------------------------------------------------------------
        # Train one independent model per policy
        # -------------------------------------------------------------

        for policy_idx, policy in enumerate(POLICIES):

            y_train = train_df[policy].to_numpy(
                dtype=np.float32
            )

            model = train_model(
                X_train,
                y_train,
                seed_offset=fold_idx * 100 + policy_idx,
            )

            fold_models[policy] = model

        # -------------------------------------------------------------
        # Predict held-out scenes
        # -------------------------------------------------------------

        for row_idx in range(len(test_df)):

            row = test_df.iloc[row_idx]

            predictions = {}

            for policy in POLICIES:
                pred = float(
                    fold_models[policy].predict(
                        X_test[row_idx:row_idx + 1]
                    )[0]
                )

                predictions[policy] = pred

            predicted_policy = select_policy(predictions)

            actual_recalls = {
                p: float(row[p])
                for p in POLICIES
            }

            oracle_policy = max(
                POLICIES,
                key=lambda p: (
                    actual_recalls[p],
                    -len(p.split("_")),
                    p,
                ),
            )

            selected_recall = actual_recalls[predicted_policy]
            oracle_recall = actual_recalls[oracle_policy]

            policy_match = (
                predicted_policy == oracle_policy
            )

            regret = oracle_recall - selected_recall

            record = {
                "fold": fold_idx,
                "held_out_seed": held_seed,
                "scene_type": row["scene_type"],
                "predicted_policy": predicted_policy,
                "oracle_policy": oracle_policy,
                "selected_recall": selected_recall,
                "oracle_recall": oracle_recall,
                "regret": regret,
                "policy_match": int(policy_match),
            }

            for p in POLICIES:
                record[f"pred_{p}"] = predictions[p]

            results.append(record)

            policy_predictions.append(
                {
                    "fold": fold_idx,
                    "held_out_seed": held_seed,
                    "scene_type": row["scene_type"],
                    **{
                        f"pred_{p}": predictions[p]
                        for p in POLICIES
                    },
                }
            )

    results_df = pd.DataFrame(results)
    prediction_df = pd.DataFrame(policy_predictions)

    # ---------------------------------------------------------------------
    # GLOBAL METRICS
    # ---------------------------------------------------------------------

    banner("GLOBAL RESULTS")

    selection_accuracy = (
        results_df["policy_match"].mean() * 100.0
    )

    automated_recall = (
        results_df["selected_recall"].mean() * 100.0
    )

    oracle_recall = (
        results_df["oracle_recall"].mean() * 100.0
    )

    regret = (
        results_df["regret"].mean() * 100.0
    )

    print(f"Scenes evaluated:             {len(results_df)}")
    print(f"Selection accuracy:           {selection_accuracy:6.2f}%")
    print(f"Automated Recall@{PRIMARY_K}:       {automated_recall:6.2f}%")
    print(f"Oracle Recall@{PRIMARY_K}:          {oracle_recall:6.2f}%")
    print(f"Mean regret:                  {regret:6.2f} pp")

    # ---------------------------------------------------------------------
    # SCENE TYPE
    # ---------------------------------------------------------------------

    banner("SCENE-TYPE PERFORMANCE")

    scene_rows = []

    for scene_type in sorted(
        results_df["scene_type"].unique()
    ):

        subset = results_df[
            results_df["scene_type"] == scene_type
        ]

        acc = subset["policy_match"].mean() * 100.0
        auto = subset["selected_recall"].mean() * 100.0
        oracle = subset["oracle_recall"].mean() * 100.0
        reg = subset["regret"].mean() * 100.0

        print()
        print(scene_type.upper())
        print("-" * 76)
        print(f"Scenes:              {len(subset)}")
        print(f"Selection accuracy:  {acc:6.2f}%")
        print(f"Automated Recall:    {auto:6.2f}%")
        print(f"Oracle Recall:       {oracle:6.2f}%")
        print(f"Mean regret:         {reg:6.2f} pp")

        scene_rows.append(
            {
                "scene_type": scene_type,
                "scenes": len(subset),
                "selection_accuracy_percent": acc,
                "automated_recall_percent": auto,
                "oracle_recall_percent": oracle,
                "mean_regret_percentage_points": reg,
            }
        )

    scene_summary_df = pd.DataFrame(scene_rows)

    # ---------------------------------------------------------------------
    # POLICY DISTRIBUTION
    # ---------------------------------------------------------------------

    banner("AUTOMATED POLICY DISTRIBUTION")

    distribution = (
        results_df["predicted_policy"]
        .value_counts()
        .rename_axis("predicted_policy")
        .reset_index(name="count")
    )

    print(distribution.to_string(index=False))

    # ---------------------------------------------------------------------
    # ORACLE DISTRIBUTION
    # ---------------------------------------------------------------------

    banner("ORACLE POLICY DISTRIBUTION")

    oracle_distribution = (
        results_df["oracle_policy"]
        .value_counts()
        .rename_axis("oracle_policy")
        .reset_index(name="count")
    )

    print(oracle_distribution.to_string(index=False))

    # ---------------------------------------------------------------------
    # REGRET
    # ---------------------------------------------------------------------

    banner("WORST POLICY REGRETS")

    worst = (
        results_df[
            [
                "scene_type",
                "held_out_seed",
                "predicted_policy",
                "oracle_policy",
                "selected_recall",
                "oracle_recall",
                "regret",
            ]
        ]
        .sort_values(
            ["regret", "scene_type"],
            ascending=[False, True],
        )
        .head(20)
    )

    print(worst.to_string(index=False))

    # ---------------------------------------------------------------------
    # PER-POLICY PREDICTION QUALITY
    # ---------------------------------------------------------------------

    banner("POLICY PREDICTION QUALITY")

    prediction_quality = []

    for policy in POLICIES:

        pred_col = f"pred_{policy}"

        y_true = results_df[policy] if policy in results_df else None

        # Actual recall is not currently copied into results for every
        # policy, so reconstruct from the merged scene table.

        actual_map = data.set_index(
            ["scene_type", "seed"]
        )[policy]

        actual_values = []

        for _, r in results_df.iterrows():
            actual_values.append(
                float(
                    actual_map.loc[
                        (r["scene_type"], r["held_out_seed"])
                    ]
                )
            )

        actual_values = np.asarray(actual_values)
        predicted_values = results_df[pred_col].to_numpy()

        rmse = float(
            np.sqrt(
                np.mean(
                    (predicted_values - actual_values) ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    predicted_values - actual_values
                )
            )
        )

        prediction_quality.append(
            {
                "policy": policy,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

    quality_df = pd.DataFrame(prediction_quality)

    print(
        quality_df.sort_values("RMSE").to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------------

    banner("SAVING STEP-30 OUTPUTS")

    results_path = (
        OUT_DIR
        / "nested_policy_selection_results.csv"
    )

    predictions_path = (
        OUT_DIR
        / "nested_policy_predictions.csv"
    )

    scene_summary_path = (
        OUT_DIR
        / "nested_policy_scene_summary.csv"
    )

    distribution_path = (
        OUT_DIR
        / "policy_distribution.csv"
    )

    quality_path = (
        OUT_DIR
        / "policy_prediction_quality.csv"
    )

    summary_path = (
        OUT_DIR
        / "nested_policy_selection_summary.json"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    scene_summary_df.to_csv(
        scene_summary_path,
        index=False,
    )

    distribution.to_csv(
        distribution_path,
        index=False,
    )

    quality_df.to_csv(
        quality_path,
        index=False,
    )

    summary = {
        "step": 30,
        "name": "Nested / Grouped Policy Selection Validation",
        "status": "research_validation",
        "primary_k": PRIMARY_K,

        "canonical_step26_eda": True,
        "canonical_step25_recall": True,

        "target_fingerprint": False,
        "new_ground_truth": False,
        "alternate_renderer": False,
        "manual_policy_selection": False,

        "feature_count": len(EXPECTED_FEATURES),
        "policy_count": len(POLICIES),

        "validation": {
            "method": "grouped leave-one-seed-out",
            "group_column": "seed",
            "unique_seeds": len(seeds),
            "both_scene_types_held_out_together": True,
            "fresh_models_per_fold": True,
            "held_out_targets_used_for_selection": False,
        },

        "global": {
            "scenes_evaluated": int(len(results_df)),
            "selection_accuracy_percent": selection_accuracy,
            "automated_recall_percent": automated_recall,
            "oracle_recall_percent": oracle_recall,
            "mean_regret_percentage_points": regret,
        },

        "scene_type": scene_rows,

        "outputs": {
            "results": str(results_path),
            "predictions": str(predictions_path),
            "scene_summary": str(scene_summary_path),
            "policy_distribution": str(distribution_path),
            "prediction_quality": str(quality_path),
        },
    }

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

    print()
    print("Saved:")
    print(results_path)
    print(predictions_path)
    print(scene_summary_path)
    print(distribution_path)
    print(quality_path)
    print(summary_path)

    # ---------------------------------------------------------------------
    # METHODOLOGY
    # ---------------------------------------------------------------------

    banner("METHODOLOGY")

    print("Canonical Step-26 EDA:              YES")
    print("Canonical Step-25 recall:           YES")
    print("Target fingerprint:                 NO")
    print("New ground truth:                   NO")
    print("Alternate renderer:                 NO")
    print("Manual policy selection:            NO")
    print("Grouped seed isolation:             YES")
    print("Fresh model per fold:               YES")
    print("Held-out target leakage:            NO")
    print("Primary metric:                     Recall@250")
    print("Validation:                         Leave-one-seed-out")

    print()
    print("=" * 76)
    print("STEP 30 COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()