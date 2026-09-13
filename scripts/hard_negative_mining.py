from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from xgboost import XGBClassifier

from scripts.learned_candidate_ranking import (
    compute_binary_label,
    extract_candidate_features,
    load_manifest_records,
    resolve_image_paths,
)

DEFAULT_OUTPUT_DIR = Path("validation/phase3/hard_negative_mining")
TARGET_TOLERANCE_PX = 5.0


def extract_fast_features_for_obs(
    search_img: np.ndarray,
    reference_img: np.ndarray,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    """Precomputes gradient maps once per observation image for fast feature extraction."""
    from scripts.learned_candidate_ranking import (
        centered_patch,
        compute_gradient_features,
        normalize_patch,
    )

    if search_img.ndim == 3:
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    if reference_img.ndim == 3:
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    ref_10 = cv2.resize(reference_img, (10, 10), interpolation=cv2.INTER_AREA)
    ref_20 = cv2.resize(reference_img, (20, 20), interpolation=cv2.INTER_AREA)
    ref_40 = cv2.resize(reference_img, (40, 40), interpolation=cv2.INTER_AREA)

    _, _, search_mag, search_ori = compute_gradient_features(search_img)
    _, _, ref_mag, ref_ori = compute_gradient_features(ref_10)

    norm_ref_10 = normalize_patch(ref_10)
    norm_ref_20 = normalize_patch(ref_20)
    norm_ref_40 = normalize_patch(ref_40)
    norm_ref_mag = normalize_patch(ref_mag)

    max_cands = len(candidates)
    feat_matrix = []

    for rank_idx, c in enumerate(candidates, start=1):
        cx = float(c["x"])
        cy = float(c["y"])
        gen_score = float(c["score"])
        g_name = str(c["generator"]).lower()
        gen_rank_norm = float(rank_idx / max(1, max_cands))

        patch_10 = centered_patch(search_img, cx, cy, 10)
        patch_20 = centered_patch(search_img, cx, cy, 20)
        patch_40 = centered_patch(search_img, cx, cy, 40)

        context_10 = float(cv2.matchTemplate(normalize_patch(patch_10), norm_ref_10, cv2.TM_CCOEFF_NORMED)[0, 0]) if patch_10 is not None else -1.0
        context_20 = float(cv2.matchTemplate(normalize_patch(patch_20), norm_ref_20, cv2.TM_CCOEFF_NORMED)[0, 0]) if patch_20 is not None else -1.0
        context_40 = float(cv2.matchTemplate(normalize_patch(patch_40), norm_ref_40, cv2.TM_CCOEFF_NORMED)[0, 0]) if patch_40 is not None else -1.0

        patch_mag = centered_patch(search_mag, cx, cy, 10)
        patch_ori = centered_patch(search_ori, cx, cy, 10)

        if patch_mag is None or patch_ori is None:
            gradient_score = 0.0
            orientation_score = 0.0
        else:
            g_norm_p = normalize_patch(patch_mag)
            gradient_score = float(np.mean(g_norm_p * norm_ref_mag))
            delta_ori = patch_ori - ref_ori
            ori_sim = np.cos(delta_ori)
            ori_weights = patch_mag + ref_mag + 1e-6
            orientation_score = float(np.sum(ori_sim * ori_weights) / np.sum(ori_weights))

        gradient_score = float(np.clip(gradient_score, -1.0, 1.0))
        orientation_score = float(np.clip(orientation_score, -1.0, 1.0))
        contrast_score = float(np.mean(normalize_patch(patch_10) * norm_ref_10)) if patch_10 is not None else 0.0
        contrast_score = float(np.clip(contrast_score, -1.0, 1.0))

        context_gain_20 = float(context_20 - context_10)
        context_gain_40 = float(context_40 - context_10)
        generator_context_gap = float(gen_score - context_10)
        context_consistency = float((context_10 + context_20 + context_40) / 3.0)

        gen_ncc = 1.0 if g_name == "ncc" else 0.0
        gen_dog = 1.0 if g_name == "dog" else 0.0
        gen_gradient = 1.0 if g_name == "gradient" else 0.0
        gen_edge = 1.0 if g_name == "edge" else 0.0
        gen_frequency = 1.0 if g_name == "frequency" else 0.0

        feat = np.array([
            gen_score,
            gen_rank_norm,
            context_10,
            context_20,
            context_40,
            gradient_score,
            orientation_score,
            contrast_score,
            context_gain_20,
            context_gain_40,
            generator_context_gap,
            context_consistency,
            gen_ncc,
            gen_dog,
            gen_gradient,
            gen_edge,
            gen_frequency,
        ], dtype=np.float32)

        if not np.isfinite(feat).all():
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

        feat_matrix.append(feat)

    return np.array(feat_matrix, dtype=np.float32)


def mine_hard_negatives(
    candidates: list[dict[str, Any]],
    raw_scores: np.ndarray,
    top_k_hard: int = 15,
) -> list[dict[str, Any]]:
    """
    Identifies high-scoring incorrect candidates (dist > 5px but top-ranked by generator score or raw XGB score).
    Returns list of mined hard negative records.
    """
    hard_negs = []
    for c, score in zip(candidates, raw_scores):
        if c["label"] == 0 and c["distance"] > TARGET_TOLERANCE_PX:
            record = dict(c)
            record["model_score"] = float(score)
            hard_negs.append(record)

    # Sort by model score descending to pick top hard negatives
    hard_negs.sort(key=lambda x: -x["model_score"])
    return hard_negs[:top_k_hard]


def run_hard_negative_mining_pipeline(
    dataset_root: Path,
    phase6_summary_path: Path,
    phase7_model_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest_records(dataset_root)

    with phase6_summary_path.open("r", encoding="utf-8") as h:
        phase6_data = json.load(h)

    baseline_clf = XGBClassifier()
    baseline_clf.load_model(str(phase7_model_path))

    obs_records = []
    all_hard_negatives = []

    X_train_list, y_train_list, sample_weights_list = [], [], []

    print("Mining hard negatives from Train split candidate pools...")

    for obs in phase6_data.get("results", []):
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
        raw_probs = baseline_clf.predict_proba(X_obs)[:, 1]

        cand_records = []
        for rank_idx, (c, p) in enumerate(zip(cands, raw_probs), start=1):
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
                "raw_score": float(p),
                "features": X_obs[rank_idx - 1],
            }
            cand_records.append(rec)

        # Mine hard negatives for train split
        if split == "train":
            mined = mine_hard_negatives(cand_records, raw_probs, top_k_hard=15)
            all_hard_negatives.extend(mined)

            mined_set = set(id(c) for c in mined)

            for c in cand_records:
                X_train_list.append(c["features"])
                y_train_list.append(c["label"])

                # Assign 3.0 weight to mined hard negatives vs 1.0 to standard negatives
                if c["label"] == 1:
                    sample_weights_list.append(10.0)
                elif id(c) in mined_set:
                    sample_weights_list.append(3.0)
                else:
                    sample_weights_list.append(1.0)

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
    weights_train = np.array(sample_weights_list, dtype=np.float32)

    print(f"Mined {len(all_hard_negatives)} hard negatives across Train split.")
    print("Training Hard-Negative Aware XGBoost model...")

    hn_clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=20260913,
        n_jobs=-1,
        eval_metric="logloss",
    )
    hn_clf.fit(X_train, y_train, sample_weight=weights_train)

    hn_model_path = model_dir / "hn_xgboost_ranker.json"
    hn_clf.save_model(str(hn_model_path))

    # Evaluate Baseline vs Hard-Negative Model across splits
    def evaluate_model(clf_model: XGBClassifier) -> dict[str, Any]:
        split_errs: dict[str, list[float]] = {"train": [], "validation": [], "hard_test": [], "overall": []}
        split_succ: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}

        for obs in obs_records:
            sp = obs["split"]
            cands = obs["candidates"]
            X_obs = np.array([c["features"] for c in cands], dtype=np.float32)
            probs = clf_model.predict_proba(X_obs)[:, 1]

            ranked = sorted(zip(cands, probs), key=lambda x: (-x[1], -x[0]["score"], x[0]["y"], x[0]["x"]))
            top1_cand, top1_prob = ranked[0]
            err = top1_cand["distance"]

            split_errs[sp].append(err)
            split_errs["overall"].append(err)

            if err <= TARGET_TOLERANCE_PX:
                split_succ[sp] += 1
                split_succ["overall"] += 1

        res = {}
        for sp in ["train", "validation", "hard_test", "overall"]:
            errs = split_errs[sp]
            n = len(errs)
            res[sp] = {
                "count": n,
                "median_error_px": float(np.median(errs)) if errs else 0.0,
                "mean_error_px": float(np.mean(errs)) if errs else 0.0,
                "p95_error_px": float(np.percentile(errs, 95)) if errs else 0.0,
                "max_error_px": float(np.max(errs)) if errs else 0.0,
                "recall_at_5px": float(split_succ[sp] / max(1, n)),
            }
        return res

    baseline_metrics = evaluate_model(baseline_clf)
    hn_metrics = evaluate_model(hn_clf)

    # Save CSVs
    hn_csv_path = output_dir / "hard_negatives.csv"
    with hn_csv_path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(
            h,
            fieldnames=["pair_id", "split", "architecture", "p6_rank", "x", "y", "score", "generator", "distance", "raw_score"],
        )
        writer.writeheader()
        for hn in all_hard_negatives:
            writer.writerow({k: hn.get(k, "") for k in writer.fieldnames})

    train_res_csv = output_dir / "training_results.csv"
    with train_res_csv.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(
            h,
            fieldnames=["split", "model_version", "median_error_px", "mean_error_px", "p95_error_px", "max_error_px", "recall_at_5px"],
        )
        writer.writeheader()
        for sp in ["train", "validation", "hard_test", "overall"]:
            writer.writerow({
                "split": sp,
                "model_version": "baseline_xgboost",
                "median_error_px": f"{baseline_metrics[sp]['median_error_px']:.2f}",
                "mean_error_px": f"{baseline_metrics[sp]['mean_error_px']:.2f}",
                "p95_error_px": f"{baseline_metrics[sp]['p95_error_px']:.2f}",
                "max_error_px": f"{baseline_metrics[sp]['max_error_px']:.2f}",
                "recall_at_5px": f"{baseline_metrics[sp]['recall_at_5px']:.4f}",
            })
            writer.writerow({
                "split": sp,
                "model_version": "hard_negative_xgboost",
                "median_error_px": f"{hn_metrics[sp]['median_error_px']:.2f}",
                "mean_error_px": f"{hn_metrics[sp]['mean_error_px']:.2f}",
                "p95_error_px": f"{hn_metrics[sp]['p95_error_px']:.2f}",
                "max_error_px": f"{hn_metrics[sp]['max_error_px']:.2f}",
                "recall_at_5px": f"{hn_metrics[sp]['recall_at_5px']:.4f}",
            })

    # Summary JSON
    summary_data = {
        "mined_hard_negatives_count": len(all_hard_negatives),
        "baseline_metrics": baseline_metrics,
        "hard_negative_metrics": hn_metrics,
        "decision": "KEEP" if hn_metrics["overall"]["median_error_px"] <= baseline_metrics["overall"]["median_error_px"] else "REJECT",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary_data, h, indent=2)

    # Report MD
    report = fr"""# MICRONYX Phase 3 — Hard-Negative Mining Report

## Executive Summary

Phase 3 evaluates hard-negative mining by collecting high-scoring incorrect candidates ($dist > 5.0\text{{ px}}$) from training candidate pools and weighting them during XGBoost retraining.

- **Mined Hard Negatives**: {len(all_hard_negatives)} candidates across Train split
- **Baseline Median Error (Overall)**: {baseline_metrics['overall']['median_error_px']:.2f} px
- **Hard-Negative XGBoost Median Error (Overall)**: {hn_metrics['overall']['median_error_px']:.2f} px
- **Phase Decision**: **{summary_data['decision']}**

## Baseline vs Hard-Negative Model Metric Comparison

| Split | Model Version | Median Error | Mean Error | P95 Error | Recall@5px |
|---|---|---:|---:|---:|---:|
| **Train** | Baseline | {baseline_metrics['train']['median_error_px']:.2f} px | {baseline_metrics['train']['mean_error_px']:.2f} px | {baseline_metrics['train']['p95_error_px']:.2f} px | {baseline_metrics['train']['recall_at_5px']:.4f} |
| **Train** | Hard-Negative | {hn_metrics['train']['median_error_px']:.2f} px | {hn_metrics['train']['mean_error_px']:.2f} px | {hn_metrics['train']['p95_error_px']:.2f} px | {hn_metrics['train']['recall_at_5px']:.4f} |
| **Validation** | Baseline | {baseline_metrics['validation']['median_error_px']:.2f} px | {baseline_metrics['validation']['mean_error_px']:.2f} px | {baseline_metrics['validation']['p95_error_px']:.2f} px | {baseline_metrics['validation']['recall_at_5px']:.4f} |
| **Validation** | Hard-Negative | {hn_metrics['validation']['median_error_px']:.2f} px | {hn_metrics['validation']['mean_error_px']:.2f} px | {hn_metrics['validation']['p95_error_px']:.2f} px | {hn_metrics['validation']['recall_at_5px']:.4f} |
| **Hard Test** | Baseline | {baseline_metrics['hard_test']['median_error_px']:.2f} px | {baseline_metrics['hard_test']['mean_error_px']:.2f} px | {baseline_metrics['hard_test']['p95_error_px']:.2f} px | {baseline_metrics['hard_test']['recall_at_5px']:.4f} |
| **Hard Test** | Hard-Negative | {hn_metrics['hard_test']['median_error_px']:.2f} px | {hn_metrics['hard_test']['mean_error_px']:.2f} px | {hn_metrics['hard_test']['p95_error_px']:.2f} px | {hn_metrics['hard_test']['recall_at_5px']:.4f} |
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    return summary_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MICRONYX Phase 3 Hard-Negative Mining.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_v0.1"))
    parser.add_argument(
        "--phase6-summary",
        type=Path,
        default=Path("validation/phase6/robust_candidate_generation/robust_candidate_generation_summary.json"),
    )
    parser.add_argument(
        "--phase7-model",
        type=Path,
        default=Path("validation/phase7/learned_candidate_ranking/xgboost_ranker.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_hard_negative_mining_pipeline(
        dataset_root=args.dataset,
        phase6_summary_path=args.phase6_summary,
        phase7_model_path=args.phase7_model,
        output_dir=args.output_dir,
    )
    print(f"MICRONYX Phase 3 complete. Decision: {summary['decision']}. Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
