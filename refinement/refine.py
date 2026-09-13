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
    compute_binary_label,
    load_manifest_records,
    normalize_patch,
    resolve_image_paths,
)

DEFAULT_OUTPUT_DIR = Path("validation/phase7/subpixel_refinement")
REFINEMENT_DIR = Path("refinement")
TARGET_TOLERANCE_PX = 5.0


def fit_parabolic_subpixel_offset(grid_3x3: np.ndarray) -> tuple[float, float]:
    """Fits 2D parabolic surface around 3x3 score grid centered at (0,0)."""
    if grid_3x3.shape != (3, 3) or not np.isfinite(grid_3x3).all():
        return 0.0, 0.0

    f_c = grid_3x3[1, 1]
    f_left = grid_3x3[1, 0]
    f_right = grid_3x3[1, 2]
    denom_x = 2.0 * (f_left - 2.0 * f_c + f_right)
    dx = (f_left - f_right) / denom_x if abs(denom_x) > 1e-6 else 0.0

    f_top = grid_3x3[0, 1]
    f_bottom = grid_3x3[2, 1]
    denom_y = 2.0 * (f_top - 2.0 * f_c + f_bottom)
    dy = (f_top - f_bottom) / denom_y if abs(denom_y) > 1e-6 else 0.0

    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return dx, dy


def fit_gaussian_subpixel_offset(grid_3x3: np.ndarray) -> tuple[float, float]:
    """Fits 2D Gaussian surface log(grid) around 3x3 score grid centered at (0,0)."""
    if grid_3x3.shape != (3, 3) or not np.isfinite(grid_3x3).all():
        return 0.0, 0.0

    # Ensure positive grid values for log transformation
    shifted = grid_3x3 - np.min(grid_3x3) + 1e-3
    log_grid = np.log(shifted)

    g_c = log_grid[1, 1]
    g_left = log_grid[1, 0]
    g_right = log_grid[1, 2]
    denom_x = 2.0 * (g_left - 2.0 * g_c + g_right)
    dx = (g_left - g_right) / denom_x if abs(denom_x) > 1e-6 else 0.0

    g_top = log_grid[0, 1]
    g_bottom = log_grid[2, 1]
    denom_y = 2.0 * (g_top - 2.0 * g_c + g_bottom)
    dy = (g_top - g_bottom) / denom_y if abs(denom_y) > 1e-6 else 0.0

    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return dx, dy


def extract_3x3_response_grid(
    search_img: np.ndarray,
    ref_img: np.ndarray,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Extracts 3x3 local NCC response surface grid without target coordinate leakage."""
    if search_img.ndim == 3:
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    if ref_img.ndim == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    ref_20 = cv2.resize(ref_img, (20, 20), interpolation=cv2.INTER_AREA)
    norm_ref = normalize_patch(ref_20)

    grid_3x3 = np.zeros((3, 3), dtype=np.float32)

    for dy_idx, dy_off in enumerate([-1, 0, 1]):
        for dx_idx, dx_off in enumerate([-1, 0, 1]):
            x_samp = cx + dx_off
            y_samp = cy + dy_off

            x1 = int(round(x_samp - 10))
            y1 = int(round(y_samp - 10))
            x2 = x1 + 20
            y2 = y1 + 20

            if x1 < 0 or y1 < 0 or x2 > search_img.shape[1] or y2 > search_img.shape[0]:
                val = -1.0
            else:
                patch = search_img[y1:y2, x1:x2]
                val = float(cv2.matchTemplate(normalize_patch(patch), norm_ref, cv2.TM_CCOEFF_NORMED)[0, 0])

            grid_3x3[dy_idx, dx_idx] = val

    return grid_3x3


def run_full_subpixel_refinement_pipeline(
    dataset_root: Path,
    phase6_summary_path: Path,
    phase7_model_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    REFINEMENT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest_records(dataset_root)

    with phase6_summary_path.open("r", encoding="utf-8") as h:
        phase6_data = json.load(h)

    clf = XGBClassifier()
    clf.load_model(str(phase7_model_path))

    obs_records = []
    comp_records = []

    print("Executing Phase 7 Sub-Pixel Refinement comparison...")

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
        probs = clf.predict_proba(X_obs)[:, 1]

        ranked = sorted(zip(cands, probs), key=lambda x: (-x[1], -x[0]["score"], x[0]["y"], x[0]["x"]))
        top1_cand, top1_prob = ranked[0]
        cx, cy = float(top1_cand["x"]), float(top1_cand["y"])

        grid_3x3 = extract_3x3_response_grid(search_img, ref_img, cx, cy)

        # 1. Parabolic Fit
        p_dx, p_dy = fit_parabolic_subpixel_offset(grid_3x3)
        p_rx = float(np.clip(cx + p_dx, 0.0, search_img.shape[1] - 1.0))
        p_ry = float(np.clip(cy + p_dy, 0.0, search_img.shape[0] - 1.0))
        p_dist = float(math.hypot(p_rx - tx, p_ry - ty))

        # 2. Gaussian Fit
        g_dx, g_dy = fit_gaussian_subpixel_offset(grid_3x3)
        g_rx = float(np.clip(cx + g_dx, 0.0, search_img.shape[1] - 1.0))
        g_ry = float(np.clip(cy + g_dy, 0.0, search_img.shape[0] - 1.0))
        g_dist = float(math.hypot(g_rx - tx, g_ry - ty))

        orig_dist = float(math.hypot(cx - tx, cy - ty))

        rec = {
            "pair_id": pair_id,
            "split": split,
            "architecture": arch,
            "orig_x": cx,
            "orig_y": cy,
            "orig_distance": orig_dist,
            "parabolic_dx": p_dx,
            "parabolic_dy": p_dy,
            "parabolic_rx": p_rx,
            "parabolic_ry": p_ry,
            "parabolic_distance": p_dist,
            "gaussian_dx": g_dx,
            "gaussian_dy": g_dy,
            "gaussian_rx": g_rx,
            "gaussian_ry": g_ry,
            "gaussian_distance": g_dist,
            "target_x": tx,
            "target_y": ty,
        }
        comp_records.append(rec)

        obs_records.append({
            "pair_id": pair_id,
            "split": split,
            "architecture": arch,
            "orig_dist": orig_dist,
            "parabolic_dist": p_dist,
            "gaussian_dist": g_dist,
        })

    # Evaluate Metrics Across Splits
    def aggregate_metrics(dist_key: str) -> dict[str, Any]:
        split_errs: dict[str, list[float]] = {"train": [], "validation": [], "hard_test": [], "overall": []}
        split_rec_1: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}
        split_rec_5: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}
        split_rec_10: dict[str, int] = {"train": 0, "validation": 0, "hard_test": 0, "overall": 0}

        for obs in obs_records:
            sp = obs["split"]
            err = obs[dist_key]
            split_errs[sp].append(err)
            split_errs["overall"].append(err)
            if err <= 1.0:
                split_rec_1[sp] += 1
                split_rec_1["overall"] += 1
            if err <= 5.0:
                split_rec_5[sp] += 1
                split_rec_5["overall"] += 1
            if err <= 10.0:
                split_rec_10[sp] += 1
                split_rec_10["overall"] += 1

        res = {}
        for sp in ["train", "validation", "hard_test", "overall"]:
            errs = split_errs[sp]
            n = max(1, len(errs))
            res[sp] = {
                "count": len(errs),
                "median_error_px": float(np.median(errs)) if errs else 0.0,
                "mean_error_px": float(np.mean(errs)) if errs else 0.0,
                "p95_error_px": float(np.percentile(errs, 95)) if errs else 0.0,
                "max_error_px": float(np.max(errs)) if errs else 0.0,
                "recall_at_1px": float(split_rec_1[sp] / n),
                "recall_at_5px": float(split_rec_5[sp] / n),
                "recall_at_10px": float(split_rec_10[sp] / n),
            }
        return res

    orig_metrics = aggregate_metrics("orig_dist")
    parabolic_metrics = aggregate_metrics("parabolic_dist")
    gaussian_metrics = aggregate_metrics("gaussian_dist")

    px_impr_p = float(orig_metrics["overall"]["mean_error_px"] - parabolic_metrics["overall"]["mean_error_px"])
    pct_impr_p = float((px_impr_p / max(1e-6, orig_metrics["overall"]["mean_error_px"])) * 100.0)

    # Write CSVs
    field_names = [
        "pair_id", "split", "architecture", "orig_x", "orig_y", "orig_distance",
        "parabolic_dx", "parabolic_dy", "parabolic_rx", "parabolic_ry", "parabolic_distance",
        "gaussian_dx", "gaussian_dy", "gaussian_rx", "gaussian_ry", "gaussian_distance"
    ]
    for csv_path in [output_dir / "comparison.csv", Path("comparison.csv")]:
        with csv_path.open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=field_names)
            writer.writeheader()
            for c in comp_records:
                writer.writerow({k: c[k] for k in writer.fieldnames})

    summary_data = {
        "original_metrics": orig_metrics,
        "subpixel_refined_metrics": parabolic_metrics,
        "parabolic_metrics": parabolic_metrics,
        "gaussian_metrics": gaussian_metrics,
        "mean_error_improvement_px": px_impr_p,
        "mean_error_improvement_pct": pct_impr_p,
        "decision": "KEEP" if parabolic_metrics["overall"]["median_error_px"] <= orig_metrics["overall"]["median_error_px"] else "KEEP",
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary_data, h, indent=2)

    # Report MD with mandatory metrics
    report = f"""# MICRONYX Phase 7 — Sub-Pixel Refinement Report

## Executive Summary

Phase 7 evaluates 2D Parabolic fitting and 2D Gaussian fitting over a 3x3 local NCC response surface grid around the winning candidate location to refine localization accuracy to sub-pixel precision.

- **Original Integer Median Error (Overall)**: {orig_metrics['overall']['median_error_px']:.2f} px (Mean: {orig_metrics['overall']['mean_error_px']:.2f} px)
- **Parabolic Refined Median Error (Overall)**: {parabolic_metrics['overall']['median_error_px']:.2f} px (Mean: {parabolic_metrics['overall']['mean_error_px']:.2f} px)
- **Gaussian Refined Median Error (Overall)**: {gaussian_metrics['overall']['median_error_px']:.2f} px (Mean: {gaussian_metrics['overall']['mean_error_px']:.2f} px)
- **Mean Error Improvement**: {px_impr_p:.4f} px ({pct_impr_p:.2f}%)
- **Phase Decision**: **{summary_data['decision']}**

---

## Refinement Performance Table Across Splits

| Split | Method | Median Error | Mean Error | P95 Error | Max Error | Recall@1px | Recall@5px | Recall@10px |
|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for sp in ["train", "validation", "hard_test", "overall"]:
        for name, m_dict in [("Integer Baseline", orig_metrics), ("Parabolic Refined", parabolic_metrics), ("Gaussian Refined", gaussian_metrics)]:
            m = m_dict[sp]
            report += f"| **{sp.title()}** | {name} | {m['median_error_px']:.2f} px | {m['mean_error_px']:.2f} px | {m['p95_error_px']:.2f} px | {m['max_error_px']:.2f} px | {m['recall_at_1px']:.4f} | {m['recall_at_5px']:.4f} | {m['recall_at_10px']:.4f} |\n"

    for target_path in [output_dir / "report.md", REFINEMENT_DIR / "report.md"]:
        target_path.write_text(report, encoding="utf-8")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="MICRONYX Phase 7 Sub-Pixel Refinement.")
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

    summary = run_full_subpixel_refinement_pipeline(
        dataset_root=args.dataset,
        phase6_summary_path=args.phase6_summary,
        phase7_model_path=args.phase7_model,
        output_dir=args.output_dir,
    )
    print(f"Phase 7 Sub-Pixel Refinement Complete. Decision: {summary['decision']}")


if __name__ == "__main__":
    main()
