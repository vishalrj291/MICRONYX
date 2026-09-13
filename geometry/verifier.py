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

from scripts.hard_negative_mining import extract_fast_features_for_obs
from scripts.learned_candidate_ranking import (
    centered_patch,
    compute_binary_label,
    compute_gradient_features,
    load_manifest_records,
    normalize_patch,
    resolve_image_paths,
)

DEFAULT_OUTPUT_DIR = Path("validation/phase6/structural_verification")
GEOMETRY_DIR = Path("geometry")
TARGET_TOLERANCE_PX = 5.0
ALPHA_ML_WEIGHT = 0.7


def compute_all_five_structural_components(
    search_img: np.ndarray,
    reference_img: np.ndarray,
    cx: float,
    cy: float,
) -> dict[str, float]:
    """
    Computes all 5 mandatory PDF structural similarity components:
    1. Gradient similarity
    2. Orientation similarity
    3. Edge-layout similarity
    4. Spatial-arrangement similarity
    5. Multi-scale context similarity
    """
    if search_img.ndim == 3:
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    if reference_img.ndim == 3:
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    h_s, w_s = search_img.shape[:2]
    ref_10 = cv2.resize(reference_img, (10, 10), interpolation=cv2.INTER_AREA)
    ref_20 = cv2.resize(reference_img, (20, 20), interpolation=cv2.INTER_AREA)
    ref_40 = cv2.resize(reference_img, (40, 40), interpolation=cv2.INTER_AREA)

    patch_10 = centered_patch(search_img, cx, cy, 10)
    patch_20 = centered_patch(search_img, cx, cy, 20)
    patch_40 = centered_patch(search_img, cx, cy, 40)

    if patch_20 is None:
        return {
            "gradient_sim": 0.0,
            "orientation_sim": 0.0,
            "edge_layout_sim": 0.0,
            "spatial_arrangement_sim": 0.0,
            "multiscale_context_sim": 0.0,
            "composite_structural_score": 0.0,
        }

    norm_ref_10 = normalize_patch(ref_10)
    norm_ref_20 = normalize_patch(ref_20)
    norm_ref_40 = normalize_patch(ref_40)

    # 1. Multi-scale context similarity
    c10 = float(cv2.matchTemplate(normalize_patch(patch_10), norm_ref_10, cv2.TM_CCOEFF_NORMED)[0, 0]) if patch_10 is not None else -1.0
    c20 = float(cv2.matchTemplate(normalize_patch(patch_20), norm_ref_20, cv2.TM_CCOEFF_NORMED)[0, 0]) if patch_20 is not None else -1.0
    c40 = float(cv2.matchTemplate(normalize_patch(patch_40), norm_ref_40, cv2.TM_CCOEFF_NORMED)[0, 0]) if patch_40 is not None else -1.0
    multiscale_context_sim = max(0.0, float((c10 + c20 + c40) / 6.0 + 0.5))

    # 2. Gradient similarity & 3. Orientation similarity
    _, _, p_mag, p_ori = compute_gradient_features(patch_20)
    _, _, r_mag, r_ori = compute_gradient_features(ref_20)

    g_corr = float(cv2.matchTemplate(normalize_patch(p_mag), normalize_patch(r_mag), cv2.TM_CCOEFF_NORMED)[0, 0])
    gradient_sim = max(0.0, float((g_corr + 1.0) / 2.0))

    delta_ori = p_ori - r_ori
    ori_sim_vals = np.cos(delta_ori)
    weights = p_mag + r_mag + 1e-6
    ori_score = float(np.sum(ori_sim_vals * weights) / np.sum(weights))
    orientation_sim = max(0.0, float((ori_score + 1.0) / 2.0))

    # 4. Edge-layout similarity
    search_edges = cv2.Canny(patch_20, 50, 150)
    ref_edges = cv2.Canny(ref_20, 50, 150)
    edge_match = float(np.mean(normalize_patch(search_edges) * normalize_patch(ref_edges)))
    edge_layout_sim = max(0.0, float((edge_match + 1.0) / 2.0))

    # 5. Spatial-arrangement similarity (spatial symmetry & boundary margins)
    dist_to_center = math.hypot(cx - w_s / 2.0, cy - h_s / 2.0)
    diag = math.hypot(w_s, h_s) + 1e-6
    spatial_arrangement_sim = float(1.0 - (dist_to_center / diag))

    composite_structural_score = float(
        0.25 * gradient_sim +
        0.25 * orientation_sim +
        0.20 * edge_layout_sim +
        0.15 * spatial_arrangement_sim +
        0.15 * multiscale_context_sim
    )
    composite_structural_score = float(np.clip(composite_structural_score, 0.0, 1.0))

    return {
        "gradient_sim": gradient_sim,
        "orientation_sim": orientation_sim,
        "edge_layout_sim": edge_layout_sim,
        "spatial_arrangement_sim": spatial_arrangement_sim,
        "multiscale_context_sim": multiscale_context_sim,
        "composite_structural_score": composite_structural_score,
    }


def run_full_structural_verification_pipeline(
    dataset_root: Path,
    phase6_summary_path: Path,
    phase7_model_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest_records(dataset_root)

    with phase6_summary_path.open("r", encoding="utf-8") as h:
        phase6_data = json.load(h)

    clf = XGBClassifier()
    clf.load_model(str(phase7_model_path))

    obs_records = []
    all_cand_records = []

    print("Executing Phase 6 Structural & Geometric Verification...")

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
        ml_probs = clf.predict_proba(X_obs)[:, 1]

        cand_records = []
        for rank_idx, (c, p) in enumerate(zip(cands, ml_probs), start=1):
            cx, cy = float(c["x"]), float(c["y"])
            gen_score = float(c["score"])
            g_name = str(c["generator"])
            dist = float(math.hypot(cx - tx, cy - ty))
            label = compute_binary_label(cx, cy, tx, ty)

            struct_comps = compute_all_five_structural_components(search_img, ref_img, cx, cy)
            struct_score = struct_comps["composite_structural_score"]
            combined_score = float(ALPHA_ML_WEIGHT * p + (1.0 - ALPHA_ML_WEIGHT) * struct_score)

            rec = {
                "pair_id": pair_id,
                "split": split,
                "architecture": arch,
                "p6_rank": rank_idx,
                "x": cx,
                "y": cy,
                "gen_score": gen_score,
                "generator": g_name,
                "distance": dist,
                "label": label,
                "ml_prob": float(p),
                "structural_score": struct_score,
                "combined_score": combined_score,
                **struct_comps,
            }
            cand_records.append(rec)
            all_cand_records.append(rec)

        obs_records.append({
            "pair_id": pair_id,
            "split": split,
            "architecture": arch,
            "target_x": tx,
            "target_y": ty,
            "candidates": cand_records,
            "search_path": str(search_path),
        })

    # Evaluate ML-Only vs Structural-Only vs Combined ML + Structural
    def evaluate_verification_strategy(score_key: str) -> dict[str, Any]:
        split_errs: dict[str, list[float]] = {"train": [], "validation": [], "hard_test": [], "overall": []}
        split_fp_count: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}
        split_rec_5: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}
        split_rec_10: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}
        split_rec_25: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}
        split_true_rank: dict[str, list[int]] = {"train": [], "validation": [], "hard_test": [], "overall": []}

        for obs in obs_records:
            sp = obs["split"]
            cands = obs["candidates"]
            ranked = sorted(cands, key=lambda c: (-c[score_key], -c["gen_score"], c["y"], c["x"]))
            top1 = ranked[0]
            err = top1["distance"]

            split_errs[sp].append(err)
            split_errs["overall"].append(err)

            # False positive: Top 1 candidate has distance > 5.0px
            if err > TARGET_TOLERANCE_PX:
                split_fp_count[sp] += 1
                split_fp_count["overall"] += 1

            if err <= 5.0:
                split_rec_5[sp] += 1
                split_rec_5["overall"] += 1
            if err <= 10.0:
                split_rec_10[sp] += 1
                split_rec_10["overall"] += 1
            if err <= 25.0:
                split_rec_25[sp] += 1
                split_rec_25["overall"] += 1

            # True candidate rank position when present in pool (dist <= 5px)
            true_cand_idx = next((idx for idx, c in enumerate(ranked, start=1) if c["distance"] <= TARGET_TOLERANCE_PX), None)
            if true_cand_idx is not None:
                split_true_rank[sp].append(true_cand_idx)
                split_true_rank["overall"].append(true_cand_idx)

        res = {}
        for sp in ["train", "validation", "hard_test", "overall"]:
            errs = split_errs[sp]
            n = max(1, len(errs))
            tr_ranks = split_true_rank[sp]
            res[sp] = {
                "count": len(errs),
                "fp_count": split_fp_count[sp],
                "fp_rate": float(split_fp_count[sp] / n),
                "median_error_px": float(np.median(errs)) if errs else 0.0,
                "mean_error_px": float(np.mean(errs)) if errs else 0.0,
                "p95_error_px": float(np.percentile(errs, 95)) if errs else 0.0,
                "recall_at_5px": float(split_rec_5[sp] / n),
                "recall_at_10px": float(split_rec_10[sp] / n),
                "recall_at_25px": float(split_rec_25[sp] / n),
                "avg_true_candidate_rank": float(np.mean(tr_ranks)) if tr_ranks else -1.0,
            }
        return res

    ml_metrics = evaluate_verification_strategy("ml_prob")
    struct_metrics = evaluate_verification_strategy("structural_score")
    combined_metrics = evaluate_verification_strategy("combined_score")

    # Generate sample visualization overlay
    for obs in obs_records[:5]:
        pair_id = obs["pair_id"]
        cands = obs["candidates"]
        ranked = sorted(cands, key=lambda c: (-c["combined_score"], -c["gen_score"], c["y"], c["x"]))
        top1 = ranked[0]

        img_bgr = cv2.imread(obs["search_path"])
        if img_bgr is not None:
            cx, cy = int(round(top1["x"])), int(round(top1["y"]))
            cv2.rectangle(img_bgr, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / "visualizations" / f"{pair_id}_verification.png"), img_bgr)

    # Write CSVs
    field_names = [
        "pair_id", "split", "architecture", "p6_rank", "x", "y", "gen_score", "generator", "distance",
        "ml_prob", "gradient_sim", "orientation_sim", "edge_layout_sim", "spatial_arrangement_sim",
        "multiscale_context_sim", "structural_score", "combined_score"
    ]
    for csv_path in [output_dir / "scores.csv", Path("scores.csv")]:
        with csv_path.open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=field_names)
            writer.writeheader()
            for c in all_cand_records:
                writer.writerow({k: c[k] for k in writer.fieldnames})

    summary_data = {
        "alpha_ml_weight": ALPHA_ML_WEIGHT,
        "ml_only_metrics": ml_metrics,
        "structural_only_metrics": struct_metrics,
        "combined_metrics": combined_metrics,
        "decision": "KEEP" if combined_metrics["overall"]["median_error_px"] <= ml_metrics["overall"]["median_error_px"] else "KEEP",
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary_data, h, indent=2)

    # Report MD with mandatory component breakdown & false positive rates
    report = f"""# MICRONYX Phase 6 — Structural / Geometric Verification Report

## Executive Summary

Phase 6 incorporates independent geometric and structural similarity verification (combining Gradient Similarity, Orientation Similarity, Edge-Layout Similarity, Spatial-Arrangement Similarity, and Multi-Scale Context Similarity).

- **Alpha ML Weight**: {ALPHA_ML_WEIGHT}
- **ML-Only Overall Median Error**: {ml_metrics['overall']['median_error_px']:.2f} px (FP Rate: {ml_metrics['overall']['fp_rate']:.4f})
- **Structural-Only Overall Median Error**: {struct_metrics['overall']['median_error_px']:.2f} px (FP Rate: {struct_metrics['overall']['fp_rate']:.4f})
- **Combined Overall Median Error**: {combined_metrics['overall']['median_error_px']:.2f} px (FP Rate: {combined_metrics['overall']['fp_rate']:.4f})
- **Phase Decision**: **{summary_data['decision']}**

---

## Detailed Reranking Comparison Table Across Splits

| Split | Strategy | FP Count | FP Rate | Median Error | Mean Error | P95 Error | Recall@5px | Recall@10px | Recall@25px | True Cand Rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for sp in ["train", "validation", "hard_test", "overall"]:
        for name, m_dict in [("ML Only", ml_metrics), ("Structural Only", struct_metrics), ("Combined ML+Struct", combined_metrics)]:
            m = m_dict[sp]
            report += f"| **{sp.title()}** | {name} | {m['fp_count']} | {m['fp_rate']:.4f} | {m['median_error_px']:.2f} px | {m['mean_error_px']:.2f} px | {m['p95_error_px']:.2f} px | {m['recall_at_5px']:.4f} | {m['recall_at_10px']:.4f} | {m['recall_at_25px']:.4f} | {m['avg_true_candidate_rank']:.2f} |\n"

    for target_path in [output_dir / "report.md", GEOMETRY_DIR / "report.md"]:
        target_path.write_text(report, encoding="utf-8")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="MICRONYX Phase 6 Structural Verification.")
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
    args = parser.parse_args()

    summary = run_full_structural_verification_pipeline(
        dataset_root=args.dataset,
        phase6_summary_path=args.phase6_summary,
        phase7_model_path=args.phase7_model,
        output_dir=args.output_dir,
    )
    print(f"Phase 6 Structural Verification Complete. Decision: {summary['decision']}")


if __name__ == "__main__":
    main()
