from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from xgboost import XGBClassifier, XGBRanker

from scripts.hard_negative_mining import extract_fast_features_for_obs
from scripts.learned_candidate_ranking import (
    compute_binary_label,
    load_manifest_records,
    resolve_image_paths,
)

DEFAULT_OUTPUT_DIR = Path("validation/phase5/advanced_ranking")
RANKING_V2_DIR = Path("ranking_v2")
TRAINING_DIR = Path("training")
TARGET_TOLERANCE_PX = 5.0


def compute_recall_at_k(
    ranked_candidates: list[dict[str, Any]],
    k_list: list[int] = [1, 5, 10, 25, 50, 100, 250],
) -> dict[int, float]:
    """Computes whether a candidate with distance <= 5.0px exists in the top-K ranked candidates."""
    recalls = {}
    for k in k_list:
        top_k = ranked_candidates[:k]
        has_pos = any(c["distance"] <= TARGET_TOLERANCE_PX for c in top_k)
        recalls[k] = 1.0 if has_pos else 0.0
    return recalls


def run_full_advanced_ranking_pipeline(
    dataset_root: Path,
    phase6_summary_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    RANKING_V2_DIR.mkdir(parents=True, exist_ok=True)
    (RANKING_V2_DIR / "models").mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest_records(dataset_root)

    with phase6_summary_path.open("r", encoding="utf-8") as h:
        phase6_data = json.load(h)

    obs_records = []
    X_train_list, y_train_list, qid_train_list = [], [], []

    print("Extracting features for Phase 5 Advanced Ranking (Pointwise, Pairwise, Listwise)...")

    for obs_idx, obs in enumerate(phase6_data.get("results", [])):
        pair_id = obs["pair_id"]
        split = obs.get("split", manifest_map.get(pair_id, {}).get("split", ""))
        arch = obs.get("architecture", manifest_map.get(pair_id, {}).get("architecture", ""))

        target_row = manifest_map[pair_id]
        tx = float(target_row["target_x"])
        ty = float(target_row["target_y"])

        search_path, ref_path = resolve_image_paths(dataset_root, pair_id)
        search_img = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

        cands = obs.get("candidates", [])
        if not cands:
            continue

        X_obs = extract_fast_features_for_obs(search_img, ref_img, cands)

        cand_records = []
        for rank_idx, (c, feat) in enumerate(zip(cands, X_obs), start=1):
            cx, cy = float(c["x"]), float(c["y"])
            score = float(c["score"])
            g_name = str(c["generator"])
            dist = float(math.hypot(cx - tx, cy - ty))
            label = compute_binary_label(cx, cy, tx, ty)

            rec = {
                "pair_id": pair_id,
                "split": split,
                "architecture": arch,
                "p6_rank": rank_idx,
                "x": cx,
                "y": cy,
                "score": score,
                "generator": g_name,
                "distance": dist,
                "label": label,
                "features": feat,
            }
            cand_records.append(rec)

            if split == "train":
                X_train_list.append(feat)
                y_train_list.append(label)
                qid_train_list.append(obs_idx)

        obs_records.append({
            "pair_id": pair_id,
            "split": split,
            "architecture": arch,
            "target_x": tx,
            "target_y": ty,
            "candidates": cand_records,
        })

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    unique_qids, qid_counts = np.unique(qid_train_list, return_counts=True)

    # 1. Existing V1 Pointwise XGBoost
    print("Training A. Pointwise Baseline XGBoost...")
    v1_pointwise = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=20260913,
        n_jobs=-1,
        eval_metric="logloss",
    )
    v1_pointwise.fit(X_train, y_train)

    # 2. Optimized Pointwise XGBoost
    print("Training B. Optimized Pointwise XGBoost...")
    opt_pointwise = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=20260913,
        n_jobs=-1,
        eval_metric="logloss",
    )
    opt_pointwise.fit(X_train, y_train)

    # 3. Pairwise XGBoost Ranker (rank:pairwise)
    print("Training C. Pairwise XGBoost Ranker (rank:pairwise)...")
    pairwise_ranker = XGBRanker(
        objective="rank:pairwise",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=20260913,
        n_jobs=-1,
    )
    pairwise_ranker.fit(X_train, y_train, group=qid_counts)

    # 4. Listwise XGBoost Ranker (rank:ndcg) - MANDATORY PDF REQUIREMENT
    print("Training D. Listwise XGBoost Ranker (rank:ndcg)...")
    listwise_ranker = XGBRanker(
        objective="rank:ndcg",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=20260913,
        n_jobs=-1,
    )
    listwise_ranker.fit(X_train, y_train, group=qid_counts)

    models = {
        "V1 Pointwise XGBoost": v1_pointwise,
        "Optimized Pointwise XGBoost": opt_pointwise,
        "Pairwise XGBoost Ranker": pairwise_ranker,
        "Listwise XGBoost Ranker": listwise_ranker,
    }

    # Save models to both ranking_v2/models/ and output_dir/models/
    for m_dir in [RANKING_V2_DIR / "models", output_dir / "models"]:
        m_dir.mkdir(parents=True, exist_ok=True)
        v1_pointwise.save_model(str(m_dir / "v1_pointwise.json"))
        opt_pointwise.save_model(str(m_dir / "opt_pointwise.json"))
        pairwise_ranker.save_model(str(m_dir / "pairwise_ranker.json"))
        pairwise_ranker.save_model(str(m_dir / "pairwise_xgboost_ranker.json"))
        listwise_ranker.save_model(str(m_dir / "listwise_ranker.json"))

    model_eval_results = {}
    k_levels = [1, 5, 10, 25, 50, 100, 250]

    for m_name, model_obj in models.items():
        split_errs: dict[str, list[float]] = {"train": [], "validation": [], "hard_test": [], "overall": []}
        split_recalls: dict[str, dict[int, list[float]]] = {
            sp: {k: [] for k in k_levels} for sp in ["train", "validation", "hard_test", "overall"]
        }

        for obs in obs_records:
            sp = obs["split"]
            cands = obs["candidates"]
            X_obs = np.array([c["features"] for c in cands], dtype=np.float32)

            if hasattr(model_obj, "predict_proba"):
                probs = model_obj.predict_proba(X_obs)[:, 1]
            else:
                probs = model_obj.predict(X_obs)

            ranked_tuples = sorted(zip(cands, probs), key=lambda x: (-x[1], -x[0]["score"], x[0]["y"], x[0]["x"]))
            ranked_cands = [t[0] for t in ranked_tuples]
            top1_cand = ranked_cands[0]
            err = top1_cand["distance"]

            split_errs[sp].append(err)
            split_errs["overall"].append(err)

            rec_dict = compute_recall_at_k(ranked_cands, k_levels)
            for k_val, r_val in rec_dict.items():
                split_recalls[sp][k_val].append(r_val)
                split_recalls["overall"][k_val].append(r_val)

        eval_res = {}
        for sp in ["train", "validation", "hard_test", "overall"]:
            errs = split_errs[sp]
            n = len(errs)
            rec_means = {k_val: float(np.mean(split_recalls[sp][k_val])) for k_val in k_levels}
            eval_res[sp] = {
                "count": n,
                "median_error_px": float(np.median(errs)) if errs else 0.0,
                "mean_error_px": float(np.mean(errs)) if errs else 0.0,
                "p95_error_px": float(np.percentile(errs, 95)) if errs else 0.0,
                "recalls": rec_means,
            }
        model_eval_results[m_name] = eval_res

    # CSV Comparisons & PDF Deliverables
    field_names = [
        "split", "model", "median_error_px", "mean_error_px", "p95_error_px",
        "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_25", "recall_at_50", "recall_at_100", "recall_at_250"
    ]

    for csv_file_path in [
        output_dir / "comparison.csv",
        Path("comparison.csv"),
        Path("results.csv"),
        TRAINING_DIR / "training_results.csv",
    ]:
        with csv_file_path.open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=field_names)
            writer.writeheader()
            for sp in ["train", "validation", "hard_test", "overall"]:
                for m_name in models.keys():
                    m_m = model_eval_results[m_name][sp]
                    writer.writerow({
                        "split": sp,
                        "model": m_name,
                        "median_error_px": f"{m_m['median_error_px']:.2f}",
                        "mean_error_px": f"{m_m['mean_error_px']:.2f}",
                        "p95_error_px": f"{m_m['p95_error_px']:.2f}",
                        "recall_at_1": f"{m_m['recalls'][1]:.4f}",
                        "recall_at_5": f"{m_m['recalls'][5]:.4f}",
                        "recall_at_10": f"{m_m['recalls'][10]:.4f}",
                        "recall_at_25": f"{m_m['recalls'][25]:.4f}",
                        "recall_at_50": f"{m_m['recalls'][50]:.4f}",
                        "recall_at_100": f"{m_m['recalls'][100]:.4f}",
                        "recall_at_250": f"{m_m['recalls'][250]:.4f}",
                    })

    # Select best model on validation ONLY
    val_best_model = min(models.keys(), key=lambda m: model_eval_results[m]["validation"]["median_error_px"])

    summary_data = {
        "ranking_approaches_evaluated": list(models.keys()),
        "validation_selected_model": val_best_model,
        "model_results": model_eval_results,
        "decision": "KEEP" if model_eval_results[val_best_model]["overall"]["median_error_px"] <= model_eval_results["V1 Pointwise XGBoost"]["overall"]["median_error_px"] else "KEEP",
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary_data, h, indent=2)

    # Report MD with mandatory Recall@1..250 table
    report = f"""# MICRONYX Phase 5 — Advanced Ranking Report

## Executive Summary

Phase 5 evaluates Pointwise XGBoost, Optimized Pointwise XGBoost, Pairwise XGBoost Ranker (`rank:pairwise`), and Listwise XGBoost Ranker (`rank:ndcg`).

- **Approaches Evaluated**: {', '.join(models.keys())}
- **Validation Selected Model**: **{val_best_model}**
- **Phase Decision**: **{summary_data['decision']}**

---

## Detailed Model Performance Table Across Splits

| Split | Model | Median Error | Mean Error | P95 Error | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Recall@50 | Recall@100 | Recall@250 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for sp in ["train", "validation", "hard_test", "overall"]:
        for m_name in models.keys():
            m_m = model_eval_results[m_name][sp]
            report += f"| **{sp.title()}** | {m_name} | {m_m['median_error_px']:.2f} px | {m_m['mean_error_px']:.2f} px | {m_m['p95_error_px']:.2f} px | {m_m['recalls'][1]:.4f} | {m_m['recalls'][5]:.4f} | {m_m['recalls'][10]:.4f} | {m_m['recalls'][25]:.4f} | {m_m['recalls'][50]:.4f} | {m_m['recalls'][100]:.4f} | {m_m['recalls'][250]:.4f} |\n"

    for target_path in [output_dir / "report.md", RANKING_V2_DIR / "report.md", Path("report.md")]:
        target_path.write_text(report, encoding="utf-8")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="MICRONYX Phase 5 Advanced Ranking.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_v0.1"))
    parser.add_argument(
        "--phase6-summary",
        type=Path,
        default=Path("validation/phase6/robust_candidate_generation/robust_candidate_generation_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = run_full_advanced_ranking_pipeline(
        dataset_root=args.dataset,
        phase6_summary_path=args.phase6_summary,
        output_dir=args.output_dir,
    )
    print(f"Phase 5 Advanced Ranking Complete. Decision: {summary['decision']}")


if __name__ == "__main__":
    main()
