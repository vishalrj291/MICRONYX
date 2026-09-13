from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "eda_guided_multi_generator_policy"
    / "scene_policy_dataset.csv"
)

RECALL_PATH = (
    ROOT
    / "validation"
    / "v02"
    / "adaptive_candidate_generation_v2"
    / "adaptive_candidate_generation_v2_results.csv"
)


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


def stats(series):

    s = pd.Series(series).astype(float)

    return {
        "count": len(s),
        "unique": int(s.nunique()),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def main():

    print("=" * 76)
    print("MICRONYX STEP 29B")
    print("POLICY TRAINING DATA AUDIT — CORRECTED")
    print("=" * 76)

    # =========================================================
    # LOAD STEP 28 DATASET
    # =========================================================

    print("\nLoading Step-28 dataset:")
    print(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nScene types:")
    print(
        df["scene_type"]
        .value_counts()
        .to_string()
    )

    print("\nUnique seeds:")
    print(df["seed"].nunique())

    # =========================================================
    # CHECK POLICY COLUMNS
    # =========================================================

    print("\n" + "=" * 76)
    print("POLICY TARGET COLUMNS")
    print("=" * 76)

    missing = [
        p for p in POLICIES
        if p not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing policy columns:\n" +
            "\n".join(missing)
        )

    for policy in POLICIES:

        s = df[policy].astype(float)

        print(
            f"{policy:24s} "
            f"count={len(s):3d} "
            f"unique={s.nunique():3d} "
            f"min={s.min():.6f} "
            f"max={s.max():.6f} "
            f"mean={s.mean():.6f} "
            f"std={s.std():.6f}"
        )

    # =========================================================
    # CHECK FEATURES
    # =========================================================

    print("\n" + "=" * 76)
    print("FEATURE CHECK")
    print("=" * 76)

    missing_features = [
        f for f in FEATURE_COLUMNS
        if f not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Missing EDA feature columns:\n" +
            "\n".join(missing_features)
        )

    X = df[FEATURE_COLUMNS]

    print(
        "Feature matrix:",
        X.shape
    )

    print(
        "NaN:",
        int(X.isna().sum().sum())
    )

    numeric = X.select_dtypes(
        include=[np.number]
    )

    print(
        "Inf:",
        int(
            np.isinf(
                numeric.to_numpy()
            ).sum()
        )
    )

    # =========================================================
    # LOAD STEP 25
    # =========================================================

    print("\n" + "=" * 76)
    print("LOADING STEP 25 RECALL")
    print("=" * 76)

    recall = pd.read_csv(
        RECALL_PATH
    )

    print(
        "Step-25 rows:",
        len(recall)
    )

    print(
        "K values:",
        sorted(
            recall["K"]
            .unique()
            .tolist()
        )
    )

    k250 = recall[
        recall["K"] == 250
    ].copy()

    print(
        "K=250 rows:",
        len(k250)
    )

    # =========================================================
    # NORMALIZE STEP 25 METHOD NAMES
    # =========================================================

    k250["method"] = (
        k250["method"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # =========================================================
    # BASIC STEP 25 SUMMARY
    # =========================================================

    print("\n" + "=" * 76)
    print("STEP 25 ACTUAL RECALL @250")
    print("=" * 76)

    summary = (
        k250
        .groupby(
            ["scene_type", "method"]
        )["recall"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
                "nunique",
            ]
        )
    )

    print(
        summary.to_string()
    )

    # =========================================================
    # BUILD LONG VERSION OF STEP 28
    # =========================================================

    print("\n" + "=" * 76)
    print("RESHAPING STEP 28")
    print("=" * 76)

    id_columns = [
        "scene_type",
        "seed",
    ]

    long28 = df[
        id_columns + POLICIES
    ].melt(
        id_vars=id_columns,
        value_vars=POLICIES,
        var_name="policy",
        value_name="step28_target",
    )

    print(
        "Step-28 long rows:",
        len(long28)
    )

    # =========================================================
    # PREPARE STEP 25
    # =========================================================

    step25 = k250[
        [
            "scene_type",
            "seed",
            "method",
            "recall",
        ]
    ].rename(
        columns={
            "method": "policy",
            "recall": "step25_recall",
        }
    )

    # =========================================================
    # CROSS-CHECK
    # =========================================================

    print("\n" + "=" * 76)
    print("STEP 28 VS STEP 25")
    print("=" * 76)

    merged = long28.merge(
        step25,
        on=[
            "scene_type",
            "seed",
            "policy",
        ],
        how="outer",
        indicator=True,
    )

    print("\nMerge status:")

    print(
        merged["_merge"]
        .value_counts()
        .to_string()
    )

    matched = merged[
        merged["_merge"] == "both"
    ].copy()

    if len(matched) == 0:
        raise RuntimeError(
            "No Step-28 / Step-25 rows matched."
        )

    # =========================================================
    # TARGET DIFFERENCE
    # =========================================================

    matched["absolute_difference"] = (
        matched["step28_target"]
        - matched["step25_recall"]
    ).abs()

    matched["signed_difference"] = (
        matched["step28_target"]
        - matched["step25_recall"]
    )

    print(
        "\nMatched rows:",
        len(matched)
    )

    print(
        "Mean absolute difference:",
        matched["absolute_difference"].mean()
    )

    print(
        "Maximum absolute difference:",
        matched["absolute_difference"].max()
    )

    print(
        "Exact matches:",
        (
            matched["absolute_difference"] < 1e-12
        ).sum()
    )

    # =========================================================
    # POLICY-BY-POLICY COMPARISON
    # =========================================================

    print("\n" + "=" * 76)
    print("POLICY TARGET ACCURACY")
    print("=" * 76)

    policy_comparison = (
        matched
        .groupby("policy")
        .agg(
            rows=("policy", "size"),
            step28_mean=("step28_target", "mean"),
            step25_mean=("step25_recall", "mean"),
            mae=("absolute_difference", "mean"),
            max_error=("absolute_difference", "max"),
            exact_matches=(
                "absolute_difference",
                lambda x: int(
                    (x < 1e-12).sum()
                ),
            ),
        )
        .sort_values(
            "mae",
            ascending=False,
        )
    )

    print(
        policy_comparison.to_string()
    )

    # =========================================================
    # SCENE TYPE COMPARISON
    # =========================================================

    print("\n" + "=" * 76)
    print("SCENE TYPE TARGET COMPARISON")
    print("=" * 76)

    scene_comparison = (
        matched
        .groupby("scene_type")
        .agg(
            rows=("policy", "size"),
            step28_mean=("step28_target", "mean"),
            step25_mean=("step25_recall", "mean"),
            mae=("absolute_difference", "mean"),
            max_error=("absolute_difference", "max"),
        )
    )

    print(
        scene_comparison.to_string()
    )

    # =========================================================
    # SHOW LARGEST MISMATCHES
    # =========================================================

    print("\n" + "=" * 76)
    print("LARGEST TARGET MISMATCHES")
    print("=" * 76)

    largest = (
        matched
        .sort_values(
            "absolute_difference",
            ascending=False,
        )
        .head(40)
    )

    print(
        largest[
            [
                "scene_type",
                "seed",
                "policy",
                "step28_target",
                "step25_recall",
                "absolute_difference",
            ]
        ]
        .to_string(index=False)
    )

    # =========================================================
    # ORACLE COMPARISON
    # =========================================================

    print("\n" + "=" * 76)
    print("ORACLE COMPARISON")
    print("=" * 76)

    actual = (
        step25
        .pivot_table(
            index=[
                "scene_type",
                "seed",
            ],
            columns="policy",
            values="step25_recall",
            aggfunc="first",
        )
    )

    actual_oracle = actual.max(
        axis=1
    )

    actual_oracle_policy = actual.idxmax(
        axis=1
    )

    predicted = (
        long28
        .pivot_table(
            index=[
                "scene_type",
                "seed",
            ],
            columns="policy",
            values="step28_target",
            aggfunc="first",
        )
    )

    predicted_oracle = predicted.max(
        axis=1
    )

    predicted_oracle_policy = predicted.idxmax(
        axis=1
    )

    oracle_compare = pd.DataFrame({
        "actual_oracle_recall":
            actual_oracle,

        "actual_oracle_policy":
            actual_oracle_policy,

        "step28_oracle_target":
            predicted_oracle,

        "step28_oracle_policy":
            predicted_oracle_policy,
    })

    oracle_compare[
        "policy_match"
    ] = (
        oracle_compare[
            "actual_oracle_policy"
        ]
        ==
        oracle_compare[
            "step28_oracle_policy"
        ]
    )

    print(
        oracle_compare
        .groupby(
            oracle_compare.index
            .get_level_values(
                "scene_type"
            )
        )["policy_match"]
        .mean()
    )

    print(
        "\nOverall oracle-policy agreement:",
        oracle_compare[
            "policy_match"
        ].mean()
    )

    # =========================================================
    # SAVE
    # =========================================================

    output_dir = (
        ROOT
        / "validation"
        / "v02"
        / "policy_training_audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched.to_csv(
        output_dir
        / "step28_vs_step25.csv",
        index=False,
    )

    policy_comparison.to_csv(
        output_dir
        / "policy_target_statistics.csv"
    )

    scene_comparison.to_csv(
        output_dir
        / "scene_target_statistics.csv"
    )

    oracle_compare.to_csv(
        output_dir
        / "oracle_comparison.csv"
    )

    print("\n" + "=" * 76)
    print("AUDIT COMPLETE")
    print("=" * 76)

    print("\nSaved:")

    print(
        output_dir
        / "step28_vs_step25.csv"
    )

    print(
        output_dir
        / "policy_target_statistics.csv"
    )

    print(
        output_dir
        / "scene_target_statistics.csv"
    )

    print(
        output_dir
        / "oracle_comparison.csv"
    )


if __name__ == "__main__":
    main()