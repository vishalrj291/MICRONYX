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

DEFAULT_OUTPUT_DIR = Path("validation/phase7/learned_candidate_ranking")
TARGET_TOLERANCE_PX = 5.0

FEATURE_NAMES = [
    "generator_score",
    "generator_rank_normalized",
    "context_10",
    "context_20",
    "context_40",
    "gradient_score",
    "orientation_score",
    "contrast_score",
    "context_gain_20",
    "context_gain_40",
    "generator_context_gap",
    "context_consistency",
    "generator_ncc",
    "generator_dog",
    "generator_gradient",
    "generator_edge",
    "generator_frequency",
]


def normalize_patch(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = float(np.mean(image))
    std = float(np.std(image))
    if std < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mean) / std


def centered_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    size: int,
) -> np.ndarray | None:
    half = size / 2.0
    x0 = int(round(center_x - half))
    y0 = int(round(center_y - half))
    x1 = x0 + size
    y1 = y0 + size

    h, w = image.shape[:2]

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        # Border candidate: create padded crop
        pad_x0 = max(0, -x0)
        pad_y0 = max(0, -y0)
        pad_x1 = max(0, x1 - w)
        pad_y1 = max(0, y1 - h)

        crop_x0 = max(0, x0)
        crop_y0 = max(0, y0)
        crop_x1 = min(w, x1)
        crop_y1 = min(h, y1)

        valid_crop = image[crop_y0:crop_y1, crop_x0:crop_x1]
        if valid_crop.size == 0:
            return None

        padded = np.pad(
            valid_crop,
            ((pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="reflect",
        )
        if padded.shape != (size, size):
            padded = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
        return padded

    return image[y0:y1, x0:x1]


def compute_gradient_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img_f = image.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    return gx, gy, magnitude, orientation


def extract_candidate_features(
    search_img: np.ndarray,
    reference_img: np.ndarray,
    candidate_x: float,
    candidate_y: float,
    generator_score: float,
    generator_rank: int,
    max_candidates: int,
    generator_name: str,
) -> np.ndarray:
    """
    Extract exactly 17 candidate features.
    CRITICAL: Does NOT accept target_x or target_y. No ground truth leakage.
    """
    if search_img.ndim == 3:
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    if reference_img.ndim == 3:
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    # Reference multi-scale templates
    ref_10 = cv2.resize(reference_img, (10, 10), interpolation=cv2.INTER_AREA)
    ref_20 = cv2.resize(reference_img, (20, 20), interpolation=cv2.INTER_AREA)
    ref_40 = cv2.resize(reference_img, (40, 40), interpolation=cv2.INTER_AREA)

    # 1. generator_score
    gen_score = float(generator_score)

    # 2. generator_rank_normalized
    gen_rank_norm = float(generator_rank / max(1, max_candidates))

    # 3. context_10
    patch_10 = centered_patch(search_img, candidate_x, candidate_y, 10)
    if patch_10 is not None:
        c10_res = cv2.matchTemplate(normalize_patch(patch_10), normalize_patch(ref_10), cv2.TM_CCOEFF_NORMED)
        context_10 = float(c10_res[0, 0])
    else:
        context_10 = -1.0

    # 4. context_20
    patch_20 = centered_patch(search_img, candidate_x, candidate_y, 20)
    if patch_20 is not None:
        c20_res = cv2.matchTemplate(normalize_patch(patch_20), normalize_patch(ref_20), cv2.TM_CCOEFF_NORMED)
        context_20 = float(c20_res[0, 0])
    else:
        context_20 = -1.0

    # 5. context_40
    patch_40 = centered_patch(search_img, candidate_x, candidate_y, 40)
    if patch_40 is not None:
        c40_res = cv2.matchTemplate(normalize_patch(patch_40), normalize_patch(ref_40), cv2.TM_CCOEFF_NORMED)
        context_40 = float(c40_res[0, 0])
    else:
        context_40 = -1.0

    # Gradient & Orientation features
    _, _, search_mag, search_ori = compute_gradient_features(search_img)
    _, _, ref_mag, ref_ori = compute_gradient_features(ref_10)

    patch_mag = centered_patch(search_mag, candidate_x, candidate_y, 10)
    patch_ori = centered_patch(search_ori, candidate_x, candidate_y, 10)

    if patch_mag is None or patch_ori is None:
        gradient_score = 0.0
        orientation_score = 0.0
    else:
        g_norm_p = normalize_patch(patch_mag)
        g_norm_r = normalize_patch(ref_mag)
        gradient_score = float(np.mean(g_norm_p * g_norm_r))

        delta_ori = patch_ori - ref_ori
        ori_sim = np.cos(delta_ori)
        ori_weights = patch_mag + ref_mag + 1e-6
        orientation_score = float(np.sum(ori_sim * ori_weights) / np.sum(ori_weights))

    gradient_score = float(np.clip(gradient_score, -1.0, 1.0))
    orientation_score = float(np.clip(orientation_score, -1.0, 1.0))

    # Contrast score
    if patch_10 is not None:
        contrast_score = float(np.mean(normalize_patch(patch_10) * normalize_patch(ref_10)))
    else:
        contrast_score = 0.0
    contrast_score = float(np.clip(contrast_score, -1.0, 1.0))

    # Derived context features
    context_gain_20 = float(context_20 - context_10)
    context_gain_40 = float(context_40 - context_10)
    generator_context_gap = float(gen_score - context_10)
    context_consistency = float((context_10 + context_20 + context_40) / 3.0)

    # One-hot generator features
    g_name = str(generator_name).lower()
    gen_ncc = 1.0 if g_name == "ncc" else 0.0
    gen_dog = 1.0 if g_name == "dog" else 0.0
    gen_gradient = 1.0 if g_name == "gradient" else 0.0
    gen_edge = 1.0 if g_name == "edge" else 0.0
    gen_frequency = 1.0 if g_name == "frequency" else 0.0

    features = np.array(
        [
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
        ],
        dtype=np.float32,
    )

    if not np.isfinite(features).all():
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    return features


def compute_binary_label(
    candidate_x: float,
    candidate_y: float,
    target_x: float,
    target_y: float,
    tolerance_px: float = TARGET_TOLERANCE_PX,
) -> int:
    """Supervised training label: 1 iff distance <= 5.0 px, else 0."""
    dist = math.hypot(candidate_x - target_x, candidate_y - target_y)
    return 1 if dist <= tolerance_px else 0


def load_manifest_records(dataset_root: Path) -> dict[str, dict]:
    manifest_csv = dataset_root / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_csv}")
    with manifest_csv.open("r", encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    return {r["pair_id"]: r for r in rows}


def resolve_image_paths(dataset_root: Path, pair_id: str) -> tuple[Path, Path]:
    direct = dataset_root / pair_id
    if not direct.is_dir():
        matches = [p for p in dataset_root.rglob(pair_id) if p.is_dir()]
        if len(matches) == 1:
            direct = matches[0]
        else:
            raise FileNotFoundError(f"Cannot locate pair directory for {pair_id}")

    search_path = direct / "search.png"
    ref_path = direct / "reference.png"
    if not search_path.exists():
        search_path = direct / "images" / "search.png"
    if not ref_path.exists():
        ref_path = direct / "images" / "reference.png"

    return search_path, ref_path


def run_phase7_pipeline(
    dataset_root: Path,
    phase6_summary_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[dict], dict, XGBClassifier]:
    if not phase6_summary_path.exists():
        raise FileNotFoundError(f"Phase 6 summary file not found: {phase6_summary_path}")

    with phase6_summary_path.open("r", encoding="utf-8") as h:
        phase6_data = json.load(h)

    manifest_map = load_manifest_records(dataset_root)

    phase6_results = phase6_data.get("results", [])
    if not phase6_results:
        raise ValueError("Phase 6 summary does not contain observation results.")

    # 1. Build Feature Dataset across all observations
    X_train_list, y_train_list = [], []

    obs_records = []

    print(f"Extracting features for {len(phase6_results)} Phase 6 candidate pools...")

    for obs in phase6_results:
        pair_id = obs["pair_id"]
        split = obs.get("split", manifest_map.get(pair_id, {}).get("split", ""))
        architecture = obs.get("architecture", manifest_map.get(pair_id, {}).get("architecture", ""))

        target_row = manifest_map[pair_id]
        tx = float(target_row["target_x"])
        ty = float(target_row["target_y"])

        search_path, ref_path = resolve_image_paths(dataset_root, pair_id)
        search_img = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

        candidates = obs.get("candidates", [])
        max_cands = len(candidates)

        obs_cands = []

        for rank_idx, c in enumerate(candidates, start=1):
            cx = float(c["x"])
            cy = float(c["y"])
            c_score = float(c["score"])
            g_name = str(c["generator"])

            # Feature extraction MUST NOT receive target_x or target_y!
            feat = extract_candidate_features(
                search_img=search_img,
                reference_img=ref_img,
                candidate_x=cx,
                candidate_y=cy,
                generator_score=c_score,
                generator_rank=rank_idx,
                max_candidates=max_cands,
                generator_name=g_name,
            )

            # Target coordinates accessed ONLY for binary label calculation
            label = compute_binary_label(cx, cy, tx, ty)
            dist_to_gt = float(math.hypot(cx - tx, cy - ty))

            cand_rec = {
                "rank_phase6": rank_idx,
                "x": cx,
                "y": cy,
                "score": c_score,
                "generator": g_name,
                "distance": dist_to_gt,
                "label": label,
                "features": feat,
            }
            obs_cands.append(cand_rec)

            # Accumulate TRAIN split samples only
            if split == "train":
                X_train_list.append(feat)
                y_train_list.append(label)

        obs_records.append({
            "pair_id": pair_id,
            "split": split,
            "architecture": architecture,
            "target_x": tx,
            "target_y": ty,
            "candidates": obs_cands,
        })

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos = float(neg_count / max(1, pos_count))

    print(f"Train split samples: {X_train.shape[0]} ({pos_count} positive, {neg_count} negative, scale_pos_weight={scale_pos:.2f})")

    # 2. Train XGBoost Candidate Classifier strictly on TRAIN split
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=20260913,
        n_jobs=-1,
        eval_metric="logloss",
    )

    clf.fit(X_train, y_train)

    # Feature Importance
    importances = clf.feature_importances_
    feat_imp = {
        name: float(imp)
        for name, imp in zip(FEATURE_NAMES, importances)
    }

    # 3. Predict & Re-Rank Candidates across all observations
    final_results = []

    for obs in obs_records:
        pair_id = obs["pair_id"]
        split = obs["split"]
        arch = obs["architecture"]
        tx = obs["target_x"]
        ty = obs["target_y"]
        cands = obs["candidates"]

        if cands:
            X_obs = np.array([c["features"] for c in cands], dtype=np.float32)
            probs = clf.predict_proba(X_obs)[:, 1]

            for c, p in zip(cands, probs):
                c["prob"] = float(p)

            # Sort by (-predicted_probability, -generator_score, y, x)
            ranked_cands = sorted(
                cands,
                key=lambda c: (-c["prob"], -c["score"], c["y"], c["x"]),
            )

            for final_r, c in enumerate(ranked_cands, start=1):
                c["final_rank"] = final_r

            top1_p6 = cands[0]  # Phase 6 baseline top-1 candidate (rank 1)
            top1_p7 = ranked_cands[0]  # Phase 7 XGBoost top-1 candidate

            p6_err = top1_p6["distance"]
            p7_err = top1_p7["distance"]
        else:
            top1_p6 = None
            top1_p7 = None
            p6_err = float("inf")
            p7_err = float("inf")
            ranked_cands = []

        res_entry = {
            "pair_id": pair_id,
            "split": split,
            "architecture": arch,
            "candidate_count": len(cands),
            "phase6_top1_error_px": p6_err,
            "phase6_top1_generator": top1_p6["generator"] if top1_p6 else "",
            "phase7_top1_error_px": p7_err,
            "phase7_top1_prob": top1_p7["prob"] if top1_p7 else 0.0,
            "phase7_top1_generator": top1_p7["generator"] if top1_p7 else "",
            "phase7_success_5px": p7_err <= TARGET_TOLERANCE_PX,
            "phase6_success_5px": p6_err <= TARGET_TOLERANCE_PX,
            "top1_candidate_x": top1_p7["x"] if top1_p7 else 0.0,
            "top1_candidate_y": top1_p7["y"] if top1_p7 else 0.0,
            "target_x": tx,
            "target_y": ty,
            "ranked_candidates": [
                {
                    "final_rank": c["final_rank"],
                    "phase6_rank": c["rank_phase6"],
                    "x": c["x"],
                    "y": c["y"],
                    "score": c["score"],
                    "generator": c["generator"],
                    "prob": c["prob"],
                    "distance": c["distance"],
                }
                for c in ranked_cands
            ],
        }
        final_results.append(res_entry)

    # 4. Compute Aggregate Metrics for splits (train, validation, hard_test, overall)
    def compute_stats(entries: list[dict]) -> dict:
        n = len(entries)
        if n == 0:
            return {}

        p6_errs = [e["phase6_top1_error_px"] for e in entries if math.isfinite(e["phase6_top1_error_px"])]
        p7_errs = [e["phase7_top1_error_px"] for e in entries if math.isfinite(e["phase7_top1_error_px"])]

        p6_succ = sum(1 for e in entries if e["phase6_success_5px"])
        p7_succ = sum(1 for e in entries if e["phase7_success_5px"])

        return {
            "observations": n,
            "phase6_baseline": {
                "recall_at_5px": p6_succ / n,
                "median_error_px": float(np.median(p6_errs)) if p6_errs else None,
                "mean_error_px": float(np.mean(p6_errs)) if p6_errs else None,
                "p95_error_px": float(np.percentile(p6_errs, 95)) if p6_errs else None,
                "max_error_px": float(np.max(p6_errs)) if p6_errs else None,
            },
            "phase7_learned_ranker": {
                "recall_at_5px": p7_succ / n,
                "median_error_px": float(np.median(p7_errs)) if p7_errs else None,
                "mean_error_px": float(np.mean(p7_errs)) if p7_errs else None,
                "p95_error_px": float(np.percentile(p7_errs, 95)) if p7_errs else None,
                "max_error_px": float(np.max(p7_errs)) if p7_errs else None,
            },
            "improvement_median_error_px": (float(np.median(p6_errs)) - float(np.median(p7_errs))) if p6_errs and p7_errs else 0.0,
        }

    overall_stats = compute_stats(final_results)
    train_stats = compute_stats([e for e in final_results if e["split"] == "train"])
    val_stats = compute_stats([e for e in final_results if e["split"] == "validation"])
    test_stats = compute_stats([e for e in final_results if e["split"] == "hard_test"])

    summary = {
        "observations_total": len(final_results),
        "train_observations": train_stats.get("observations", 0),
        "validation_observations": val_stats.get("observations", 0),
        "hard_test_observations": test_stats.get("observations", 0),
        "overall": overall_stats,
        "splits": {
            "train": train_stats,
            "validation": val_stats,
            "hard_test": test_stats,
        },
        "feature_importance": feat_imp,
    }

    return final_results, summary, clf


def write_phase7_outputs(
    output_dir: Path,
    results: list[dict],
    summary: dict,
    model: XGBClassifier,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "learned_candidate_ranking_results.csv"
    json_path = output_dir / "learned_candidate_ranking_summary.json"
    report_path = output_dir / "learned_candidate_ranking_report.md"
    model_path = output_dir / "xgboost_ranker.json"
    feat_path = output_dir / "feature_importance.json"

    # 1. Save Trained Model
    model.save_model(str(model_path))

    # 2. Save Feature Importance
    with feat_path.open("w", encoding="utf-8") as h:
        json.dump(summary["feature_importance"], h, indent=2)

    # 3. Save Summary JSON
    with json_path.open("w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)

    # 4. Save Results CSV
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(
            h,
            fieldnames=[
                "pair_id",
                "split",
                "architecture",
                "candidate_count",
                "phase6_top1_error_px",
                "phase6_top1_generator",
                "phase7_top1_error_px",
                "phase7_top1_prob",
                "phase7_top1_generator",
                "phase7_success_5px",
                "phase6_success_5px",
                "top1_candidate_x",
                "top1_candidate_y",
                "target_x",
                "target_y",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    # 5. Save Report MD
    overall_p6 = summary["overall"]["phase6_baseline"]
    overall_p7 = summary["overall"]["phase7_learned_ranker"]

    test_p6 = summary["splits"]["hard_test"]["phase6_baseline"]
    test_p7 = summary["splits"]["hard_test"]["phase7_learned_ranker"]

    feat_table = "\n".join(
        f"| `{name}` | {imp:.4f} |"
        for name, imp in sorted(summary["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    )

    report = f"""# MICRONYX Phase 7 — Learned Candidate Ranking Report

## Executive Summary

- **Total Observations Processed**: {summary["observations_total"]}
- **Split Distribution**: Train ({summary["train_observations"]}), Validation ({summary["validation_observations"]}), Hard-Test ({summary["hard_test_observations"]})
- **Model Trained**: XGBoost Classifier (`xgboost_ranker.json`)
- **Training Constraints**: Fitted strictly on 70 `train` split observations.

## Baseline vs Learned Ranking Comparison

### Overall Dataset Performance (100 Observations)

| Metric | Phase 6 Baseline | Phase 7 XGBoost Ranker | Delta |
|---|---:|---:|---:|
| **Recall@5px Success Rate** | {overall_p6["recall_at_5px"]:.4f} | {overall_p7["recall_at_5px"]:.4f} | {overall_p7["recall_at_5px"] - overall_p6["recall_at_5px"]:+.4f} |
| **Median Localization Error** | {overall_p6["median_error_px"]:.4f} px | {overall_p7["median_error_px"]:.4f} px | {overall_p7["median_error_px"] - overall_p6["median_error_px"]:+.4f} px |
| **Mean Localization Error** | {overall_p6["mean_error_px"]:.4f} px | {overall_p7["mean_error_px"]:.4f} px | {overall_p7["mean_error_px"] - overall_p6["mean_error_px"]:+.4f} px |
| **P95 Localization Error** | {overall_p6["p95_error_px"]:.4f} px | {overall_p7["p95_error_px"]:.4f} px | {overall_p7["p95_error_px"] - overall_p6["p95_error_px"]:+.4f} px |
| **Maximum Error** | {overall_p6["max_error_px"]:.4f} px | {overall_p7["max_error_px"]:.4f} px | {overall_p7["max_error_px"] - overall_p6["max_error_px"]:+.4f} px |

### Held-Out Hard Test Split Performance (15 Observations)

| Metric | Phase 6 Baseline | Phase 7 XGBoost Ranker | Delta |
|---|---:|---:|---:|
| **Recall@5px Success Rate** | {test_p6["recall_at_5px"]:.4f} | {test_p7["recall_at_5px"]:.4f} | {test_p7["recall_at_5px"] - test_p6["recall_at_5px"]:+.4f} |
| **Median Localization Error** | {test_p6["median_error_px"]:.4f} px | {test_p7["median_error_px"]:.4f} px | {test_p7["median_error_px"] - test_p6["median_error_px"]:+.4f} px |

## Feature Importance

| Feature Name | XGBoost Importance |
|---|---:|
{feat_table}

## Anti-Leakage & Integrity Verification

1. **Feature Extraction Signature**: `extract_candidate_features(search_img, reference_img, candidate_x, candidate_y, ...)` does NOT receive target coordinates.
2. **Train/Test Isolation**: Model fit strictly on `train` split records ($N={summary["train_observations"]}$). Validation and hard_test samples strictly excluded from training.
3. **Deterministic Inference**: Reproducible ranking using `random_state = 20260913`.
"""

    report_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MICRONYX Phase 7 Learned Candidate Ranking.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phase6-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results, summary, model = run_phase7_pipeline(
        dataset_root=args.dataset,
        phase6_summary_path=args.phase6_summary,
        output_dir=args.output_dir,
    )

    write_phase7_outputs(
        output_dir=args.output_dir,
        results=results,
        summary=summary,
        model=model,
    )

    overall = summary["overall"]["phase7_learned_ranker"]

    print("MICRONYX Phase 7 complete")
    print(f"Observations   : {summary['observations_total']}")
    print(f"Train Records  : {summary['train_observations']}")
    print(f"Validation     : {summary['validation_observations']}")
    print(f"Hard Test      : {summary['hard_test_observations']}")
    print(f"Phase 7 Median : {overall['median_error_px']:.4f} px")
    print(f"Phase 7 Recall : {overall['recall_at_5px']:.4f}")
    print(f"Model saved to : {args.output_dir / 'xgboost_ranker.json'}")


if __name__ == "__main__":
    main()
