from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]

EDA_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "automated_eda"
    / "automated_eda_results.csv"
)

CONFIG_PATH = (
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


def load_model(path):
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def main():

    print("=" * 76)
    print("MICRONYX STEP 29A")
    print("POLICY INFERENCE DIAGNOSTIC")
    print("=" * 76)

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    policies = config["policies"]
    feature_columns = config["feature_columns"]

    print("\nPolicies:")
    for p in policies:
        print(" ", p)

    print("\nExpected feature count:")
    print(" ", len(feature_columns))

    # ---------------------------------------------------------
    # Load EDA
    # ---------------------------------------------------------

    eda = pd.read_csv(EDA_PATH)

    print("\nEDA shape:")
    print(" ", eda.shape)

    print("\nEDA columns:")
    print(" ", len(eda.columns))

    required_scene_columns = [
        "scene_type",
        "seed",
    ]

    for c in required_scene_columns:
        if c not in eda.columns:
            raise RuntimeError(f"Missing required EDA column: {c}")

    missing = [
        c for c in feature_columns
        if c not in eda.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing feature columns:\n" +
            "\n".join(missing)
        )

    # ---------------------------------------------------------
    # Feature matrix
    # ---------------------------------------------------------

    X = eda[feature_columns].copy()

    print("\nFeature matrix:")
    print(" ", X.shape)

    print("\nFeature dtypes:")
    print(X.dtypes.value_counts())

    # ---------------------------------------------------------
    # NaN / Inf diagnostics
    # ---------------------------------------------------------

    nan_count = int(X.isna().sum().sum())

    inf_count = int(
        np.isinf(
            X.select_dtypes(include=[np.number]).to_numpy()
        ).sum()
    )

    print("\nNaN count:", nan_count)
    print("Inf count:", inf_count)

    if nan_count:
        print("\nWARNING: NaNs detected.")

    if inf_count:
        print("\nWARNING: Inf values detected.")

    # ---------------------------------------------------------
    # Load all models
    # ---------------------------------------------------------

    print("\n" + "=" * 76)
    print("LOADING MODELS")
    print("=" * 76)

    models = {}

    for policy in policies:

        path = MODEL_DIR / f"{policy}_xgb.json"

        if not path.exists():
            print(f"WARNING: missing {policy}")
            continue

        model = load_model(path)
        models[policy] = model

        print(
            f"{policy:24s} "
            f"loaded"
        )

    print("\nModels loaded:", len(models))

    if len(models) != len(policies):
        raise RuntimeError(
            "Not all policy models were loaded."
        )

    # ---------------------------------------------------------
    # Prediction matrix
    # ---------------------------------------------------------

    print("\n" + "=" * 76)
    print("PREDICTING POLICY PERFORMANCE")
    print("=" * 76)

    predictions = {}

    for policy in policies:

        model = models[policy]

        pred = model.predict(X)

        pred = np.asarray(pred, dtype=float)

        predictions[policy] = pred

        print(
            f"{policy:24s} "
            f"min={pred.min(): .6f} "
            f"mean={pred.mean(): .6f} "
            f"max={pred.max(): .6f}"
        )

    pred_df = pd.DataFrame(predictions)

    # ---------------------------------------------------------
    # Selected policy
    # ---------------------------------------------------------

    selected = pred_df.idxmax(axis=1)

    print("\n" + "=" * 76)
    print("PREDICTED POLICY DISTRIBUTION")
    print("=" * 76)

    print(
        selected.value_counts()
    )

    # ---------------------------------------------------------
    # Top-3 policy diagnostics
    # ---------------------------------------------------------

    print("\n" + "=" * 76)
    print("SCENE-LEVEL POLICY PREDICTIONS")
    print("=" * 76)

    rows = []

    for i in range(len(eda)):

        row = pred_df.iloc[i]

        ranked = row.sort_values(
            ascending=False
        )

        scene_type = eda.iloc[i]["scene_type"]
        seed = eda.iloc[i]["seed"]

        top1 = ranked.index[0]
        top2 = ranked.index[1]
        top3 = ranked.index[2]

        print(
            f"\nScene {i+1:02d} "
            f"{scene_type:14s} "
            f"seed={seed}"
        )

        print(
            f"  TOP1 "
            f"{top1:24s} "
            f"{ranked.iloc[0]:.6f}"
        )

        print(
            f"  TOP2 "
            f"{top2:24s} "
            f"{ranked.iloc[1]:.6f}"
        )

        print(
            f"  TOP3 "
            f"{top3:24s} "
            f"{ranked.iloc[2]:.6f}"
        )

        rows.append({
            "scene_type": scene_type,
            "seed": seed,
            "selected_policy": top1,
            "top1_score": ranked.iloc[0],
            "top2_policy": top2,
            "top2_score": ranked.iloc[1],
            "top3_policy": top3,
            "top3_score": ranked.iloc[2],
        })

    diagnostic = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_dir = (
        ROOT
        / "validation"
        / "v02"
        / "policy_inference_diagnostic"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pred_df.insert(
        0,
        "seed",
        eda["seed"].values
    )

    pred_df.insert(
        0,
        "scene_type",
        eda["scene_type"].values
    )

    pred_path = (
        output_dir
        / "policy_predictions.csv"
    )

    diag_path = (
        output_dir
        / "policy_top3_diagnostic.csv"
    )

    pred_df.to_csv(
        pred_path,
        index=False
    )

    diagnostic.to_csv(
        diag_path,
        index=False
    )

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)

    print("\nSelected policy distribution:")
    print(
        selected.value_counts()
    )

    print("\nMean predicted score:")
    print(
        pred_df[policies]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\nOutput:")
    print(pred_path)
    print(diag_path)

    print("\n" + "=" * 76)
    print("STEP 29A COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()