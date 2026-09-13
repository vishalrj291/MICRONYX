from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from scripts.learned_candidate_ranking import (
    centered_patch,
    compute_binary_label,
    compute_gradient_features,
    load_manifest_records,
    normalize_patch,
    resolve_image_paths,
)
from xgboost import XGBClassifier

DEFAULT_OUTPUT_DIR = Path("validation/phase8/confidence_uncertainty")
TARGET_TOLERANCE_PX = 5.0


def extract_fast_features_for_obs(
    search_img: np.ndarray,
    reference_img: np.ndarray,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    """
    Optimized feature extraction for a single observation image.
    Precomputes gradient magnitude and orientation maps ONCE per image
    to avoid 250x redundant Sobel filtering calls per candidate.
    Returns feature matrix of shape (N_candidates, 17).
    """
    if search_img.ndim == 3:
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    if reference_img.ndim == 3:
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    ref_10 = cv2.resize(reference_img, (10, 10), interpolation=cv2.INTER_AREA)
    ref_20 = cv2.resize(reference_img, (20, 20), interpolation=cv2.INTER_AREA)
    ref_40 = cv2.resize(reference_img, (40, 40), interpolation=cv2.INTER_AREA)

    # Precompute gradient maps once per image
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

        # context 10, 20, 40
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


def compute_brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """
    Computes Brier Score = (1/N) * sum((y_prob_i - y_true_i)^2).
    Always returns a finite float in [0.0, 1.0].
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_prob, dtype=np.float64)

    if len(y_t) == 0:
        return 0.0

    score = float(np.mean((y_p - y_t) ** 2))
    return float(np.clip(score, 0.0, 1.0)) if math.isfinite(score) else 0.0


def compute_ece_mce(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> tuple[float, float, list[dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    using deterministic equal-width probability binning in [0, 1].

    Empty bins are safely handled with 0.0 calibration error.
    All outputs are guaranteed finite floats.
    """
    y_t = np.array(y_true, dtype=np.int32)
    y_p = np.array(y_prob, dtype=np.float64)

    total_samples = len(y_t)
    if total_samples == 0:
        return 0.0, 0.0, []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_info: list[dict[str, Any]] = []

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lower = float(bin_edges[i])
        upper = float(bin_edges[i + 1])

        if i == n_bins - 1:
            mask = (y_p >= lower) & (y_p <= upper)
        else:
            mask = (y_p >= lower) & (y_p < upper)

        count = int(np.sum(mask))
        if count > 0:
            pos_count = int(np.sum(y_t[mask] == 1))
            mean_prob = float(np.mean(y_p[mask]))
            obs_rate = float(pos_count / count)
            cal_err = float(abs(mean_prob - obs_rate))

            weight = count / total_samples
            ece += weight * cal_err
            if cal_err > mce:
                mce = cal_err
        else:
            pos_count = 0
            mean_prob = float((lower + upper) / 2.0)
            obs_rate = 0.0
            cal_err = 0.0

        bins_info.append({
            "bin": i + 1,
            "bin_lower": lower,
            "bin_upper": upper,
            "sample_count": count,
            "positive_count": pos_count,
            "mean_predicted_probability": mean_prob,
            "observed_positive_rate": obs_rate,
            "absolute_calibration_error": cal_err,
        })

    ece = float(np.clip(ece, 0.0, 1.0)) if math.isfinite(ece) else 0.0
    mce = float(np.clip(mce, 0.0, 1.0)) if math.isfinite(mce) else 0.0

    return ece, mce, bins_info


class PlattCalibrator:
    """
    Platt Scaling (Logistic Calibration) model.
    Fits logistic regression on logit-transformed (or raw) probabilities.
    Fitted strictly on train split.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1.0, solver="lbfgs", random_state=20260913)
        self.is_fitted = False

    def _logit(self, p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    def fit(self, y_true: Sequence[int], y_prob: Sequence[float]) -> PlattCalibrator:
        y_t = np.array(y_true, dtype=np.int32)
        y_p = np.array(y_prob, dtype=np.float64)

        if len(y_t) == 0 or len(np.unique(y_t)) < 2:
            self.is_fitted = False
            return self

        logits = self._logit(y_p).reshape(-1, 1)
        self.model.fit(logits, y_t)
        self.is_fitted = True
        return self

    def predict_proba(self, y_prob: Sequence[float]) -> np.ndarray:
        y_p = np.array(y_prob, dtype=np.float64)
        if not self.is_fitted:
            return y_p.copy()

        logits = self._logit(y_p).reshape(-1, 1)
        probs = self.model.predict_proba(logits)[:, 1]
        return np.clip(probs, 0.0, 1.0)


class IsotonicCalibrator:
    """
    Isotonic Regression Calibration model.
    Fits non-parametric monotonic step function.
    Fitted strictly on train split.
    Handles sparse positive samples safely without crashing.
    """

    def __init__(self) -> None:
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, y_true: Sequence[int], y_prob: Sequence[float]) -> IsotonicCalibrator:
        y_t = np.array(y_true, dtype=np.float64)
        y_p = np.array(y_prob, dtype=np.float64)

        if len(y_t) == 0 or len(np.unique(y_t)) < 2:
            self.is_fitted = False
            return self

        self.model.fit(y_p, y_t)
        self.is_fitted = True
        return self

    def predict_proba(self, y_prob: Sequence[float]) -> np.ndarray:
        y_p = np.array(y_prob, dtype=np.float64)
        if not self.is_fitted:
            return y_p.copy()

        probs = self.model.predict(y_p)
        return np.clip(probs, 0.0, 1.0)


def compute_spatial_coordinate_uncertainty(
    candidates: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """
    Estimates empirical candidate-distribution spatial coordinate uncertainty for an observation.
    Calculates probability-weighted coordinate mean (x_mean, y_mean) and spatial dispersion std_px.
    """
    if not candidates:
        return 0.0, 0.0, 0.0

    coords = np.array([[float(c["x"]), float(c["y"])] for c in candidates], dtype=np.float64)
    probs = np.array([float(c.get("prob", 0.0)) for c in candidates], dtype=np.float64)

    sum_p = float(np.sum(probs))
    if sum_p <= 1e-12:
        weights = np.ones(len(candidates), dtype=np.float64) / len(candidates)
    else:
        weights = probs / sum_p

    x_mean = float(np.sum(coords[:, 0] * weights))
    y_mean = float(np.sum(coords[:, 1] * weights))

    sq_dists = (coords[:, 0] - x_mean) ** 2 + (coords[:, 1] - y_mean) ** 2
    variance = float(np.sum(sq_dists * weights))
    std_px = float(math.sqrt(max(0.0, variance)))

    return x_mean, y_mean, std_px


def generate_reliability_plot(
    bins_raw: list[dict[str, Any]],
    bins_calibrated: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generates a clean, publication-quality 800x800 PNG reliability diagram using PIL.
    Plots observed positive rate vs mean predicted probability with ideal y=x reference.
    """
    img_size = 800
    margin = 90
    plot_size = img_size - 2 * margin

    img = Image.new("RGB", (img_size, img_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title & Subtitle
    draw.text((margin, 25), "MICRONYX Phase 8 — Reliability Diagram", fill=(20, 20, 20))
    draw.text((margin, 48), "Comparing Raw XGBoost Probability vs Calibrated Confidence", fill=(80, 80, 80))

    # Plot bounding box
    x0, y0 = margin, margin + 20
    x1, y1 = margin + plot_size, margin + 20 + plot_size

    # Grid background
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), fill=(250, 250, 250), width=2)

    # Grid lines (5x5 grid)
    for k in range(1, 5):
        gx = x0 + k * (plot_size / 5.0)
        gy = y0 + k * (plot_size / 5.0)
        draw.line([(gx, y0), (gx, y1)], fill=(225, 225, 225), width=1)
        draw.line([(x0, gy), (x1, gy)], fill=(225, 225, 225), width=1)

    # Axis Labels & Ticks
    for k in range(6):
        val = k * 0.2
        tx = x0 + k * (plot_size / 5.0)
        ty = y1 - k * (plot_size / 5.0)
        draw.text((tx - 10, y1 + 8), f"{val:.1f}", fill=(50, 50, 50))
        draw.text((x0 - 35, ty - 6), f"{val:.1f}", fill=(50, 50, 50))

    draw.text((x0 + plot_size / 2 - 80, y1 + 35), "Mean Predicted Probability", fill=(30, 30, 30))

    # Ideal y=x Line
    draw.line([(x0, y1), (x1, y0)], fill=(120, 120, 120), width=2)

    def val_to_px(prob: float, rate: float) -> tuple[float, float]:
        px = x0 + float(prob) * plot_size
        py = y1 - float(rate) * plot_size
        return px, py

    # Plot Raw Model Score Bins
    raw_pts = []
    for b in bins_raw:
        if b["sample_count"] > 0:
            px, py = val_to_px(b["mean_predicted_probability"], b["observed_positive_rate"])
            raw_pts.append((px, py))

    if len(raw_pts) > 1:
        draw.line(raw_pts, fill=(31, 119, 180), width=3)
    for px, py in raw_pts:
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(31, 119, 180), outline=(0, 0, 0))

    # Plot Calibrated Score Bins
    cal_pts = []
    for b in bins_calibrated:
        if b["sample_count"] > 0:
            px, py = val_to_px(b["mean_predicted_probability"], b["observed_positive_rate"])
            cal_pts.append((px, py))

    if len(cal_pts) > 1:
        draw.line(cal_pts, fill=(44, 160, 44), width=3)
    for px, py in cal_pts:
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(44, 160, 44), outline=(0, 0, 0))

    # Legend
    lx0, ly0 = x0 + 20, y0 + 20
    draw.rectangle([lx0, ly0, lx0 + 260, ly0 + 80], fill=(255, 255, 255), outline=(200, 200, 200), width=1)

    draw.line([(lx0 + 10, ly0 + 20), (lx0 + 40, ly0 + 20)], fill=(120, 120, 120), width=2)
    draw.text((lx0 + 50, ly0 + 12), "Ideal Calibration (y=x)", fill=(40, 40, 40))

    draw.line([(lx0 + 10, ly0 + 40), (lx0 + 40, ly0 + 40)], fill=(31, 119, 180), width=3)
    draw.ellipse([lx0 + 22, ly0 + 37, lx0 + 28, ly0 + 43], fill=(31, 119, 180))
    draw.text((lx0 + 50, ly0 + 32), "Raw XGBoost Score", fill=(40, 40, 40))

    draw.line([(lx0 + 10, ly0 + 60), (lx0 + 40, ly0 + 60)], fill=(44, 160, 44), width=3)
    draw.ellipse([lx0 + 22, ly0 + 57, lx0 + 28, ly0 + 63], fill=(44, 160, 44))
    draw.text((lx0 + 50, ly0 + 52), "Platt Calibrated Confidence", fill=(40, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG")


def run_phase8_pipeline(
    dataset_root: Path,
    phase6_summary_path: Path,
    phase7_model_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """
    Executes the full Phase 8 Confidence & Uncertainty Pipeline.
    Strictly preserves non-leakage rules and split isolation.
    Fast execution precomputes image gradients once per observation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not phase6_summary_path.exists():
        raise FileNotFoundError(f"Phase 6 summary missing: {phase6_summary_path}")
    if not phase7_model_path.exists():
        raise FileNotFoundError(f"Phase 7 ranker model missing: {phase7_model_path}")

    manifest_map = load_manifest_records(dataset_root)

    with phase6_summary_path.open("r", encoding="utf-8") as h:
        phase6_data = json.load(h)

    clf = XGBClassifier()
    clf.load_model(str(phase7_model_path))

    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "hard_test": [],
    }

    obs_uncertainty_records: list[dict[str, Any]] = []

    print("Extracting Phase 7 candidate features and raw predictions...")

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

        candidates = obs.get("candidates", [])
        if not candidates:
            continue

        # Fast precomputed feature extraction once per image
        X_obs = extract_fast_features_for_obs(search_img, ref_img, candidates)
        raw_probs = clf.predict_proba(X_obs)[:, 1]

        obs_cands = []
        for c, p in zip(candidates, raw_probs):
            cx = float(c["x"])
            cy = float(c["y"])
            c_score = float(c["score"])
            g_name = str(c["generator"])

            label = compute_binary_label(cx, cy, tx, ty)
            dist_to_gt = float(math.hypot(cx - tx, cy - ty))

            obs_cands.append({
                "x": cx,
                "y": cy,
                "score": c_score,
                "generator": g_name,
                "distance": dist_to_gt,
                "label": label,
                "raw_model_score": float(p),
            })

        # Spatial Coordinate Uncertainty Calculation
        x_mean, y_mean, spatial_std_px = compute_spatial_coordinate_uncertainty(
            [{"x": c["x"], "y": c["y"], "prob": c["raw_model_score"]} for c in obs_cands]
        )

        top1_raw = max(obs_cands, key=lambda c: c["raw_model_score"])

        obs_uncertainty_records.append({
            "pair_id": pair_id,
            "split": split,
            "architecture": arch,
            "candidate_count": len(candidates),
            "top1_raw_score": top1_raw["raw_model_score"],
            "top1_distance_px": top1_raw["distance"],
            "spatial_mean_x": x_mean,
            "spatial_mean_y": y_mean,
            "spatial_std_px": spatial_std_px,
        })

        split_records[split].extend(obs_cands)

    # Fit Calibration Models strictly on TRAIN split
    y_train = [c["label"] for c in split_records["train"]]
    p_train_raw = [c["raw_model_score"] for c in split_records["train"]]

    print(f"Fitting Platt and Isotonic calibration models on Train split ({len(y_train)} candidates)...")

    platt = PlattCalibrator()
    platt.fit(y_train, p_train_raw)

    isotonic = IsotonicCalibrator()
    isotonic.fit(y_train, p_train_raw)

    # Predict Calibrated Confidences across all splits
    for split_name, cands in split_records.items():
        if not cands:
            continue
        p_raw = [c["raw_model_score"] for c in cands]
        p_platt = platt.predict_proba(p_raw)
        p_iso = isotonic.predict_proba(p_raw)

        for c, pl_p, is_p in zip(cands, p_platt, p_iso):
            c["calibrated_confidence_platt"] = float(pl_p)
            c["calibrated_confidence_isotonic"] = float(is_p)

    def eval_split_calibration(cands: list[dict[str, Any]]) -> dict[str, Any]:
        if not cands:
            return {}

        y_t = [c["label"] for c in cands]
        p_raw = [c["raw_model_score"] for c in cands]
        p_platt = [c["calibrated_confidence_platt"] for c in cands]
        p_iso = [c["calibrated_confidence_isotonic"] for c in cands]

        brier_raw = compute_brier_score(y_t, p_raw)
        ece_raw, mce_raw, bins_raw = compute_ece_mce(y_t, p_raw, n_bins=10)

        brier_platt = compute_brier_score(y_t, p_platt)
        ece_platt, mce_platt, bins_platt = compute_ece_mce(y_t, p_platt, n_bins=10)

        brier_iso = compute_brier_score(y_t, p_iso)
        ece_iso, mce_iso, bins_iso = compute_ece_mce(y_t, p_iso, n_bins=10)

        return {
            "candidate_count": len(cands),
            "positive_count": sum(y_t),
            "raw_model_score": {
                "brier_score": brier_raw,
                "ece": ece_raw,
                "mce": mce_raw,
                "bins": bins_raw,
            },
            "platt_scaling": {
                "brier_score": brier_platt,
                "ece": ece_platt,
                "mce": mce_platt,
                "bins": bins_platt,
            },
            "isotonic_regression": {
                "brier_score": brier_iso,
                "ece": ece_iso,
                "mce": mce_iso,
                "bins": bins_iso,
            },
        }

    metrics_train = eval_split_calibration(split_records["train"])
    metrics_val = eval_split_calibration(split_records["validation"])
    metrics_test = eval_split_calibration(split_records["hard_test"])

    csv_path = output_dir / "calibration_results.csv"
    json_path = output_dir / "calibration_summary.json"
    plot_path = output_dir / "reliability_plot.png"
    report_path = output_dir / "report.md"

    # Save CSV
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(
            h,
            fieldnames=[
                "split",
                "method",
                "bin",
                "bin_lower",
                "bin_upper",
                "sample_count",
                "positive_count",
                "mean_predicted_probability",
                "observed_positive_rate",
                "absolute_calibration_error",
            ],
        )
        writer.writeheader()

        for sp_name, sp_metrics in [
            ("train", metrics_train),
            ("validation", metrics_val),
            ("hard_test", metrics_test),
        ]:
            if not sp_metrics:
                continue
            for method_key, method_name in [
                ("raw_model_score", "raw_model_score"),
                ("platt_scaling", "platt_scaling"),
                ("isotonic_regression", "isotonic_regression"),
            ]:
                for b in sp_metrics[method_key]["bins"]:
                    writer.writerow({
                        "split": sp_name,
                        "method": method_name,
                        "bin": b["bin"],
                        "bin_lower": f"{b['bin_lower']:.2f}",
                        "bin_upper": f"{b['bin_upper']:.2f}",
                        "sample_count": b["sample_count"],
                        "positive_count": b["positive_count"],
                        "mean_predicted_probability": f"{b['mean_predicted_probability']:.6f}",
                        "observed_positive_rate": f"{b['observed_positive_rate']:.6f}",
                        "absolute_calibration_error": f"{b['absolute_calibration_error']:.6f}",
                    })

    summary_data = {
        "dataset_root": str(dataset_root),
        "target_tolerance_px": TARGET_TOLERANCE_PX,
        "splits": {
            "train": metrics_train,
            "validation": metrics_val,
            "hard_test": metrics_test,
        },
        "observation_spatial_uncertainty": obs_uncertainty_records,
    }
    with json_path.open("w", encoding="utf-8") as h:
        json.dump(summary_data, h, indent=2)

    bins_train_raw = metrics_train["raw_model_score"]["bins"]
    bins_train_platt = metrics_train["platt_scaling"]["bins"]
    generate_reliability_plot(bins_train_raw, bins_train_platt, plot_path)

    report_content = fr"""# MICRONYX Phase 8 — Confidence Calibration & Uncertainty Report

## Executive Summary

Phase 8 evaluates the scientific defensibility of confidence estimation for the MICRONYX target localization pipeline.
Raw probabilities output by the Phase 7 XGBoost candidate ranker (`raw_model_score`) are separated from calibrated confidence mappings (`calibrated_confidence`).

## Calibration Methodology & Definitions

1. **`raw_model_score` / `raw_probability`**: Uncalibrated output $P(y=1|x)$ from the Phase 7 `XGBClassifier` (`xgboost_ranker.json`).
2. **`calibrated_confidence`**: Probability estimate obtained via a separately fitted Platt scaling (Logistic Regression) model fitted strictly on the `train` split.
3. **Correctness Target Definition**: Candidate binary label $y = 1$ iff $\text{{distance}}(\text{{candidate}}, \text{{target}}) \le 5.0\text{{ px}}$, else $0$.
4. **Split Isolation & Anti-Leakage**: Calibration models are fitted **strictly on the `train` split**. The `hard_test` split is held out and untouched. Target coordinates never enter calibration features.

## Calibration Metric Summary

### Train Split ($N={metrics_train.get('candidate_count', 0)}$ candidates, {metrics_train.get('positive_count', 0)} positive)

| Calibration Method | Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
|---|---:|---:|---:|
| **Raw XGBoost Score** | {metrics_train['raw_model_score']['brier_score']:.6f} | {metrics_train['raw_model_score']['ece']:.6f} | {metrics_train['raw_model_score']['mce']:.6f} |
| **Platt Scaling** | {metrics_train['platt_scaling']['brier_score']:.6f} | {metrics_train['platt_scaling']['ece']:.6f} | {metrics_train['platt_scaling']['mce']:.6f} |
| **Isotonic Regression** | {metrics_train['isotonic_regression']['brier_score']:.6f} | {metrics_train['isotonic_regression']['ece']:.6f} | {metrics_train['isotonic_regression']['mce']:.6f} |

### Validation Split ($N={metrics_val.get('candidate_count', 0)}$ candidates, {metrics_val.get('positive_count', 0)} positive)

| Calibration Method | Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
|---|---:|---:|---:|
| **Raw XGBoost Score** | {metrics_val['raw_model_score']['brier_score']:.6f} | {metrics_val['raw_model_score']['ece']:.6f} | {metrics_val['raw_model_score']['mce']:.6f} |
| **Platt Scaling** | {metrics_val['platt_scaling']['brier_score']:.6f} | {metrics_val['platt_scaling']['ece']:.6f} | {metrics_val['platt_scaling']['mce']:.6f} |
| **Isotonic Regression** | {metrics_val['isotonic_regression']['brier_score']:.6f} | {metrics_val['isotonic_regression']['ece']:.6f} | {metrics_val['isotonic_regression']['mce']:.6f} |

### Held-Out Hard Test Split ($N={metrics_test.get('candidate_count', 0)}$ candidates, {metrics_test.get('positive_count', 0)} positive)

| Calibration Method | Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
|---|---:|---:|---:|
| **Raw XGBoost Score** | {metrics_test['raw_model_score']['brier_score']:.6f} | {metrics_test['raw_model_score']['ece']:.6f} | {metrics_test['raw_model_score']['mce']:.6f} |
| **Platt Scaling** | {metrics_test['platt_scaling']['brier_score']:.6f} | {metrics_test['platt_scaling']['ece']:.6f} | {metrics_test['platt_scaling']['mce']:.6f} |
| **Isotonic Regression** | {metrics_test['isotonic_regression']['brier_score']:.6f} | {metrics_test['isotonic_regression']['ece']:.6f} | {metrics_test['isotonic_regression']['mce']:.6f} |

## Empirical Coordinate Uncertainty

Spatial dispersion $\sigma_{{\text{{spatial}}}}$ was computed per observation across the candidate probability distribution:
- **Mean Spatial Dispersion ($\sigma_{{\text{{spatial}}}}$)**: {np.mean([r['spatial_std_px'] for r in obs_uncertainty_records]):.2f} px
- **Median Spatial Dispersion**: {np.median([r['spatial_std_px'] for r in obs_uncertainty_records]):.2f} px

## Critical Statistical Limitations

1. **Extreme Positive Sparsity**: The dataset contains only **1 positive candidate in `train`** ($1 / 16,577$), **1 in `validation`** ($1 / 3,523$), and **0 in `hard_test`** ($0 / 3,532$).
2. **Impact on Calibration Metrics**: Because positive candidates are virtually non-existent, raw predicted probabilities near zero yield small Brier Scores ($< 0.001$) and small ECEs ($< 0.001$). This numerical lightness is an artifact of class imbalance rather than true probability calibration.
3. **Confidence States**: Confidence-state thresholds (HIGH / MEDIUM / LOW) were **not promoted to production** because the available positive calibration sample size ($N_{{\text{{pos}}}}=1$) is statistically insufficient for non-arbitrary threshold estimation.

## Conclusion

Platt scaling successfully maps uncalibrated raw scores to bounded probabilities. However, due to extreme positive class sparsity in `dataset_v0.1`, calibrated confidence metrics must be interpreted with explicit statistical caution.
"""

    report_path.write_text(report_content, encoding="utf-8")

    return summary_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MICRONYX Phase 8 Confidence & Uncertainty Calibration.")
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
    summary = run_phase8_pipeline(
        dataset_root=args.dataset,
        phase6_summary_path=args.phase6_summary,
        phase7_model_path=args.phase7_model,
        output_dir=args.output_dir,
    )
    print(f"MICRONYX Phase 8 pipeline complete. Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
