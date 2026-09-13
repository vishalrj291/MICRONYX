from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


DEFAULT_GENERATORS = (
    "ncc",
    "dog",
    "gradient",
    "edge",
    "frequency",
)

DEFAULT_BUDGET = 250
DEFAULT_RECALL_TOLERANCE_PX = 5.0
DEFAULT_MIN_DISTANCE_PX = 4.0


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    score: float
    generator: str

    def as_dict(self) -> dict:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "score": float(self.score),
            "generator": self.generator,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_count: int
    recall_at_1: bool
    recall_at_5: bool
    recall_at_10: bool
    recall_at_25: bool
    recall_at_50: bool
    recall_at_100: bool
    recall_at_250: bool
    best_error_px: float
    best_generator: str | None


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    if image is None:
        raise ValueError(f"{name} could not be loaded.")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim != 2:
        raise ValueError(
            f"{name} must be a grayscale 2-D image."
        )

    if image.size == 0:
        raise ValueError(f"{name} is empty.")

    image = image.astype(np.float32)

    if not np.isfinite(image).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return image


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {path}"
        )

    return _validate_image(
        image,
        str(path),
    )


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    mean = float(np.mean(image))
    std = float(np.std(image))

    if std <= 1e-12:
        return np.zeros_like(image)

    return (image - mean) / std


def resize_reference(
    reference: np.ndarray,
    search: np.ndarray,
    scale: float,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError(
            "Scale must be positive."
        )

    # If reference is high-res (e.g. 1000x1000 for a 1000x1000 search image), scale down by `scale`.
    # If reference is already at search-scale template size (e.g. 100x100), preserve its dimensions.
    if reference.shape[0] > search.shape[0] / scale and reference.shape[1] > search.shape[1] / scale:
        height = max(
            1,
            int(round(reference.shape[0] / scale)),
        )
        width = max(
            1,
            int(round(reference.shape[1] / scale)),
        )
    else:
        height = reference.shape[0]
        width = reference.shape[1]

    if height > search.shape[0] or width > search.shape[1]:
        raise ValueError(
            "Scaled reference is larger than search image."
        )

    if (height, width) == (reference.shape[0], reference.shape[1]):
        return reference

    return cv2.resize(
        reference,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def _response_to_candidates(
    response: np.ndarray,
    template_shape: tuple[int, int],
    generator: str,
    count: int,
) -> list[Candidate]:
    if response.ndim != 2:
        raise ValueError(
            "Matching response must be 2-D."
        )

    h, w = template_shape

    if response.shape[0] == 0 or response.shape[1] == 0:
        return []

    work = response.copy()

    results: list[Candidate] = []

    for _ in range(min(count, work.size)):
        _, max_value, _, max_location = cv2.minMaxLoc(
            work
        )

        x0, y0 = max_location

        x = float(x0 + (w - 1) / 2.0)
        y = float(y0 + (h - 1) / 2.0)

        results.append(
            Candidate(
                x=x,
                y=y,
                score=float(max_value),
                generator=generator,
            )
        )

        # Non-maximum suppression region.
        radius = max(
            2,
            int(round(min(h, w) * 0.25)),
        )

        x1 = max(0, x0 - radius)
        x2 = min(work.shape[1], x0 + radius + 1)
        y1 = max(0, y0 - radius)
        y2 = min(work.shape[0], y0 + radius + 1)

        work[y1:y2, x1:x2] = -np.inf

        if not np.isfinite(work).any():
            break

    return results


def ncc_response(
    search: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    search_n = normalize_image(search)
    reference_n = normalize_image(reference)

    return cv2.matchTemplate(
        search_n,
        reference_n,
        cv2.TM_CCOEFF_NORMED,
    )


def dog_image(
    image: np.ndarray,
) -> np.ndarray:
    sigma_small = 1.0
    sigma_large = 2.0

    small = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma_small,
    )

    large = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma_large,
    )

    return small - large


def dog_response(
    search: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return ncc_response(
        dog_image(search),
        dog_image(reference),
    )


def gradient_image(
    image: np.ndarray,
) -> np.ndarray:
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    return cv2.magnitude(gx, gy)


def gradient_response(
    search: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return ncc_response(
        gradient_image(search),
        gradient_image(reference),
    )


def edge_image(
    image: np.ndarray,
) -> np.ndarray:
    image_8 = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    return cv2.Canny(
        image_8,
        50,
        150,
    ).astype(np.float32)


def edge_response(
    search: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return ncc_response(
        edge_image(search),
        edge_image(reference),
    )


def frequency_image(
    image: np.ndarray,
) -> np.ndarray:
    image_n = normalize_image(image)

    blurred = cv2.GaussianBlur(
        image_n,
        (0, 0),
        sigmaX=3.0,
    )

    return image_n - blurred


def frequency_response(
    search: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return ncc_response(
        frequency_image(search),
        frequency_image(reference),
    )


GENERATOR_FUNCTIONS: dict[
    str,
    Callable[
        [np.ndarray, np.ndarray],
        np.ndarray,
    ],
] = {
    "ncc": ncc_response,
    "dog": dog_response,
    "gradient": gradient_response,
    "edge": edge_response,
    "frequency": frequency_response,
}


def validate_generators(
    generators: Iterable[str],
) -> tuple[str, ...]:
    generators = tuple(generators)

    if not generators:
        raise ValueError(
            "At least one generator is required."
        )

    unknown = [
        name
        for name in generators
        if name not in GENERATOR_FUNCTIONS
    ]

    if unknown:
        raise ValueError(
            "Unknown generator(s): "
            + ", ".join(sorted(unknown))
        )

    return generators


def allocate_budget(
    generators: Iterable[str],
    total_budget: int,
    weights: dict[str, float] | None = None,
) -> dict[str, int]:
    generators = validate_generators(
        generators
    )

    if total_budget <= 0:
        raise ValueError(
            "Candidate budget must be positive."
        )

    if weights is None:
        weights = {
            name: 1.0
            for name in generators
        }

    clean_weights = {}

    for name in generators:
        value = float(
            weights.get(name, 0.0)
        )

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid generator weight for {name}."
            )

        clean_weights[name] = value

    total_weight = sum(
        clean_weights.values()
    )

    if total_weight <= 0:
        raise ValueError(
            "Generator weights must not all be zero."
        )

    allocations = {
        name: 1
        for name in generators
    }

    remaining = max(
        0,
        total_budget - len(generators),
    )

    if remaining:
        raw = {
            name: remaining
            * clean_weights[name]
            / total_weight
            for name in generators
        }

        floors = {
            name: int(math.floor(raw[name]))
            for name in generators
        }

        for name in generators:
            allocations[name] += floors[name]

        left = remaining - sum(
            floors.values()
        )

        ranked = sorted(
            generators,
            key=lambda name: (
                raw[name] - floors[name],
                clean_weights[name],
                name,
            ),
            reverse=True,
        )

        for name in ranked[:left]:
            allocations[name] += 1

    return allocations


def generate_candidates(
    search: np.ndarray,
    reference: np.ndarray,
    generators: Iterable[str],
    total_budget: int = DEFAULT_BUDGET,
    generator_weights: dict[str, float] | None = None,
) -> list[Candidate]:
    search = _validate_image(
        search,
        "search",
    )

    reference = _validate_image(
        reference,
        "reference",
    )

    generators = validate_generators(
        generators
    )

    allocations = allocate_budget(
        generators,
        total_budget,
        generator_weights,
    )

    all_candidates: list[Candidate] = []

    for name in generators:
        function = GENERATOR_FUNCTIONS[name]

        response = function(
            search,
            reference,
        )

        candidates = _response_to_candidates(
            response,
            reference.shape,
            name,
            allocations[name],
        )

        all_candidates.extend(
            candidates
        )

    return all_candidates


def spatial_deduplicate(
    candidates: Iterable[Candidate],
    min_distance_px: float = DEFAULT_MIN_DISTANCE_PX,
) -> list[Candidate]:
    if min_distance_px < 0:
        raise ValueError(
            "Minimum candidate distance cannot be negative."
        )

    ordered = sorted(
        candidates,
        key=lambda c: (
            -c.score,
            c.generator,
            c.y,
            c.x,
        ),
    )

    kept: list[Candidate] = []
    min_distance_sq = (
        min_distance_px
        * min_distance_px
    )

    for candidate in ordered:
        keep = True

        for existing in kept:
            dx = (
                candidate.x
                - existing.x
            )
            dy = (
                candidate.y
                - existing.y
            )

            if (
                dx * dx + dy * dy
                < min_distance_sq
            ):
                keep = False
                break

        if keep:
            kept.append(candidate)

    return kept


def rank_candidates(
    candidates: Iterable[Candidate],
) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (
            -c.score,
            c.generator,
            c.y,
            c.x,
        ),
    )


def candidate_error(
    candidate: Candidate,
    target_x: float,
    target_y: float,
) -> float:
    return float(
        math.hypot(
            candidate.x - target_x,
            candidate.y - target_y,
        )
    )


def evaluate_recall(
    candidates: Iterable[Candidate],
    target_x: float,
    target_y: float,
    tolerance_px: float = DEFAULT_RECALL_TOLERANCE_PX,
) -> CandidateEvaluation:
    if tolerance_px <= 0:
        raise ValueError(
            "Recall tolerance must be positive."
        )

    ranked = rank_candidates(
        candidates
    )

    if not ranked:
        return CandidateEvaluation(
            candidate_count=0,
            recall_at_1=False,
            recall_at_5=False,
            recall_at_10=False,
            recall_at_25=False,
            recall_at_50=False,
            recall_at_100=False,
            recall_at_250=False,
            best_error_px=float("inf"),
            best_generator=None,
        )

    errors = [
        candidate_error(
            candidate,
            target_x,
            target_y,
        )
        for candidate in ranked
    ]

    best_index = int(
        np.argmin(errors)
    )

    def recalled(k: int) -> bool:
        return any(
            error <= tolerance_px
            for error in errors[:k]
        )

    best_candidate = ranked[
        best_index
    ]

    return CandidateEvaluation(
        candidate_count=len(ranked),
        recall_at_1=recalled(1),
        recall_at_5=recalled(5),
        recall_at_10=recalled(10),
        recall_at_25=recalled(25),
        recall_at_50=recalled(50),
        recall_at_100=recalled(100),
        recall_at_250=recalled(250),
        best_error_px=float(
            errors[best_index]
        ),
        best_generator=best_candidate.generator,
    )


def load_manifest(
    dataset_root: Path,
) -> list[dict]:
    dataset_path = Path(dataset_root)

    manifest_csv = dataset_path / "manifest.csv"
    if manifest_csv.exists():
        with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("manifest.csv is empty")
        return rows

    manifest_json = dataset_path / "manifest.json"
    if manifest_json.exists():
        with manifest_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get(
                "observations",
                payload.get("records", payload.get("items", [])),
            )
        else:
            raise ValueError(
                "manifest.json must contain a list or an object containing observations/records/items."
            )
        if not isinstance(rows, list):
            raise ValueError("Manifest observation records must be a list.")
        return rows

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("Manifest CSV is empty")
        return rows

    raise FileNotFoundError(f"Missing manifest in: {dataset_path}")


def _extract_target(
    record: dict,
) -> tuple[float, float]:
    for key_x in ("target_x", "gt_x", "target_center_x", "gt_center_x", "center_x"):
        if key_x in record and record[key_x] not in (None, ""):
            try:
                val_x = float(record[key_x])
                break
            except (TypeError, ValueError):
                pass
    else:
        val_x = None

    for key_y in ("target_y", "gt_y", "target_center_y", "gt_center_y", "center_y"):
        if key_y in record and record[key_y] not in (None, ""):
            try:
                val_y = float(record[key_y])
                break
            except (TypeError, ValueError):
                pass
    else:
        val_y = None

    if val_x is not None and val_y is not None:
        if not (math.isfinite(val_x) and math.isfinite(val_y)):
            raise ValueError("Target coordinates must be finite floats.")
        return (val_x, val_y)

    target = record.get("target", {})
    if isinstance(target, dict):
        center = target.get("center_xy_search_px", target.get("center_xy"))
        if isinstance(center, (list, tuple)) and len(center) == 2:
            return (float(center[0]), float(center[1]))

    raise ValueError(f"Manifest record missing valid target coordinates: {record}")


def _locate_pair_dir(
    dataset_root: Path,
    pair_id: str,
) -> Path:
    direct = dataset_root / pair_id
    if direct.is_dir():
        return direct

    matches = [p for p in dataset_root.rglob(pair_id) if p.is_dir()]
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(f"Cannot locate pair directory for {pair_id} under {dataset_root}")

    raise ValueError(f"Multiple pair directories found for {pair_id} under {dataset_root}: {matches}")


def _find_image_in_pair(
    pair_dir: Path,
    name: str,
) -> Path:
    candidates = [
        pair_dir / name,
        pair_dir / "images" / name,
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list(pair_dir.rglob(name))
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {pair_dir}")

    raise ValueError(f"Multiple {name} files found under {pair_dir}: {matches}")


def resolve_image_path(
    dataset_root: Path,
    relative_path: str,
) -> Path:
    direct = dataset_root / relative_path
    if direct.exists():
        return direct

    matches = list(dataset_root.rglob(Path(relative_path).name))
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(f"Image not found: {relative_path}")

    raise FileNotFoundError(f"Ambiguous image path: {relative_path}")


def resolve_record_images(
    dataset_root: Path,
    record: dict,
) -> tuple[Path, Path]:
    pair_id = record.get("pair_id")
    if pair_id:
        pair_dir = _locate_pair_dir(dataset_root, str(pair_id))
        search_path = _find_image_in_pair(pair_dir, "search.png")
        reference_path = _find_image_in_pair(pair_dir, "reference.png")
        return (search_path, reference_path)

    files = record.get("files")
    if isinstance(files, dict):
        ref_rel = files.get("reference")
        search_rel = files.get("search")
        if ref_rel and search_rel:
            ref_path = resolve_image_path(dataset_root, str(ref_rel))
            search_path = resolve_image_path(dataset_root, str(search_rel))
            return (search_path, ref_path)

    raise ValueError(f"Record does not specify pair_id or files structure: {record}")


def process_record(
    dataset_root: Path,
    record: dict,
    scale: float,
    generators: tuple[str, ...],
    budget: int,
    min_distance_px: float,
    weights: dict[str, float] | None,
) -> tuple[list[Candidate], CandidateEvaluation, int, int]:
    search_path, reference_path = resolve_record_images(dataset_root, record)

    reference = load_grayscale(reference_path)
    search = load_grayscale(search_path)

    reference = resize_reference(
        reference,
        search,
        scale,
    )

    raw_candidates = generate_candidates(
        search,
        reference,
        generators,
        budget,
        weights,
    )
    raw_count = len(raw_candidates)

    candidates = spatial_deduplicate(
        raw_candidates,
        min_distance_px,
    )
    dedup_count = len(candidates)

    target_x, target_y = _extract_target(record)

    evaluation = evaluate_recall(
        candidates,
        target_x,
        target_y,
    )

    return candidates, evaluation, raw_count, dedup_count


def build_weights_from_summary(
    summary_path: Path,
) -> dict[str, float]:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Phase 5 summary not found: {summary_path}"
        )

    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    benchmark = payload.get(
        "generator_benchmark",
        payload.get("benchmark", {}),
    )

    if not isinstance(benchmark, dict):
        raise ValueError(
            "Phase 5 summary does not contain a valid generator benchmark."
        )

    weights: dict[str, float] = {}

    for name in DEFAULT_GENERATORS:
        item = benchmark.get(name)
        if not isinstance(item, dict):
            continue

        median_error = item.get("median_error_px", item.get("median_error"))
        if median_error is None:
            continue

        error = float(median_error)
        if math.isfinite(error) and error > 0:
            weights[name] = 1.0 / error

    if not weights:
        return {name: 1.0 for name in DEFAULT_GENERATORS}

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def run_dataset(
    dataset_root: Path,
    scale: float,
    generators: tuple[str, ...],
    budget: int,
    min_distance_px: float,
    weights: dict[str, float] | None,
) -> tuple[list[dict], dict]:
    records = load_manifest(dataset_root)

    results: list[dict] = []

    recall_counts = {
        1: 0,
        5: 0,
        10: 0,
        25: 0,
        50: 0,
        100: 0,
        250: 0,
    }

    errors: list[float] = []
    total_generated = 0
    total_remaining = 0
    generator_best_counts = {name: 0 for name in generators}
    failure_cases: list[dict] = []

    for index, record in enumerate(records):
        pair_id = record.get("pair_id", record.get("id", f"observation_{index:04d}"))
        split = record.get("split", "")
        architecture = record.get("architecture", "")

        candidates, evaluation, raw_count, dedup_count = process_record(
            dataset_root,
            record,
            scale,
            generators,
            budget,
            min_distance_px,
            weights,
        )

        total_generated += raw_count
        total_remaining += dedup_count

        for k, key in (
            (1, "recall_at_1"),
            (5, "recall_at_5"),
            (10, "recall_at_10"),
            (25, "recall_at_25"),
            (50, "recall_at_50"),
            (100, "recall_at_100"),
            (250, "recall_at_250"),
        ):
            if getattr(evaluation, key):
                recall_counts[k] += 1

        if math.isfinite(evaluation.best_error_px):
            errors.append(evaluation.best_error_px)

        if evaluation.best_generator and evaluation.best_generator in generator_best_counts:
            generator_best_counts[evaluation.best_generator] += 1

        is_failure = (not evaluation.recall_at_250) or (evaluation.best_error_px > DEFAULT_RECALL_TOLERANCE_PX)

        if is_failure:
            failure_cases.append({
                "pair_id": str(pair_id),
                "split": str(split),
                "architecture": str(architecture),
                "best_error_px": evaluation.best_error_px,
                "best_generator": evaluation.best_generator,
                "candidate_count": evaluation.candidate_count,
            })

        results.append({
            "pair_id": str(pair_id),
            "split": str(split),
            "architecture": str(architecture),
            "candidates_before_dedup": raw_count,
            "candidate_count": evaluation.candidate_count,
            "recall_at_1": evaluation.recall_at_1,
            "recall_at_5": evaluation.recall_at_5,
            "recall_at_10": evaluation.recall_at_10,
            "recall_at_25": evaluation.recall_at_25,
            "recall_at_50": evaluation.recall_at_50,
            "recall_at_100": evaluation.recall_at_100,
            "recall_at_250": evaluation.recall_at_250,
            "best_error_px": evaluation.best_error_px,
            "best_generator": evaluation.best_generator,
            "is_failure": is_failure,
            "candidates": [
                candidate.as_dict()
                for candidate in rank_candidates(candidates)
            ],
        })

    count = len(records)

    def recall_val(k: int) -> float:
        return (recall_counts[k] / count) if count > 0 else 0.0

    summary = {
        "observations": count,
        "candidate_budget": budget,
        "minimum_spatial_distance_px": min_distance_px,
        "scale": scale,
        "generators": list(generators),
        "generator_weights": weights,
        "candidates_generated_before_dedup": total_generated,
        "candidates_remaining_after_dedup": total_remaining,
        "recall_at_k": {
            f"recall_at_{k}": recall_val(k)
            for k in (1, 5, 10, 25, 50, 100, 250)
        },
        "best_candidate_error_px": {
            "median": float(np.median(errors)) if errors else None,
            "p95": float(np.percentile(errors, 95)) if errors else None,
            "max": float(np.max(errors)) if errors else None,
            "mean": float(np.mean(errors)) if errors else None,
        },
        "generator_contribution": {
            g: {
                "count": generator_best_counts[g],
                "percentage": (generator_best_counts[g] / count * 100.0) if count > 0 else 0.0,
            }
            for g in generators
        },
        "failure_count": len(failure_cases),
        "failure_cases": failure_cases,
        "results": results,
    }

    return results, summary


def write_outputs(
    output_dir: Path,
    results: list[dict],
    summary: dict,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "robust_candidate_generation_results.csv"
    json_path = output_dir / "robust_candidate_generation_summary.json"
    report_path = output_dir / "robust_candidate_generation_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "split",
                "architecture",
                "candidates_before_dedup",
                "candidate_count",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_25",
                "recall_at_50",
                "recall_at_100",
                "recall_at_250",
                "best_error_px",
                "best_generator",
                "is_failure",
            ],
        )

        writer.writeheader()

        for row in results:
            writer.writerow({
                key: row.get(key, "")
                for key in writer.fieldnames
            })

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    recall = summary["recall_at_k"]
    error = summary["best_candidate_error_px"]
    contrib = summary["generator_contribution"]

    contrib_table = "\n".join(
        f"| {g} | {data['count']} | {data['percentage']:.2f}% |"
        for g, data in contrib.items()
    )

    report = f"""# MICRONYX Phase 6 — Robust Candidate Generation Report

> [!IMPORTANT]
> **Candidate-Pool Recall Boundary**: All Recall@K metrics reported in this phase evaluate the raw spatial candidate pool **BEFORE** any learned verification or ranking in Phase 7.

## Executive Summary

- **Observations Processed**: {summary["observations"]}
- **Target Candidate Budget**: {summary["candidate_budget"]} candidates per observation
- **Spatial Separation Threshold**: {summary["minimum_spatial_distance_px"]} px NMS
- **Scale Factor**: {summary["scale"]}×
- **Active Generators**: {", ".join(summary["generators"])}
- **Total Candidates Generated (Before Deduplication)**: {summary["candidates_generated_before_dedup"]}
- **Total Candidates Remaining (After Deduplication)**: {summary["candidates_remaining_after_dedup"]}

## Candidate Recall Performance (Pre-Verification)

| Metric | Recall Rate |
|---|---:|
| Recall@1 | {recall["recall_at_1"]:.4f} |
| Recall@5 | {recall["recall_at_5"]:.4f} |
| Recall@10 | {recall["recall_at_10"]:.4f} |
| Recall@25 | {recall["recall_at_25"]:.4f} |
| Recall@50 | {recall["recall_at_50"]:.4f} |
| Recall@100 | {recall["recall_at_100"]:.4f} |
| Recall@250 | {recall["recall_at_250"]:.4f} |

## Localization Error Distribution

- **Median Best-Candidate Error**: {error["median"]:.4f} px
- **Mean Best-Candidate Error**: {error["mean"]:.4f} px
- **P95 Best-Candidate Error**: {error["p95"]:.4f} px
- **Maximum Best-Candidate Error**: {error["max"]:.4f} px

## Generator Contribution

| Generator | Best Candidate Count | Best Candidate Percentage |
|---|---:|---:|
{contrib_table}

## Hard-Negative Preservation & Failure Cases

- **Failure Count (Best Error > 5 px or Unrecalled)**: {summary["failure_count"]}

All candidate pools (including hard negatives and failure cases) are fully preserved in JSON format for consumption by the Phase 7 learned verifier.

## System Invariants & Guarantees

1. **Non-Leakage Guarantee**: Target coordinates are never passed into candidate generation functions.
2. **Deterministic Order**: Candidates are sorted strictly by `(-score, generator, y, x)`.
3. **Adaptive Budgeting**: Generator budgets are allocated dynamically based on Phase 5 representation selection.
4. **Spatial Non-Maximum Suppression**: Duplicate candidates within `{summary["minimum_spatial_distance_px"]}` px are removed while preserving top responses.
"""

    report_path.write_text(
        report,
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MICRONYX Phase 6 robust "
            "candidate generation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=DEFAULT_MIN_DISTANCE_PX,
    )

    parser.add_argument(
        "--generators",
        nargs="+",
        choices=DEFAULT_GENERATORS,
        default=list(DEFAULT_GENERATORS),
    )

    parser.add_argument(
        "--phase5-summary",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "validation/phase6/robust_candidate_generation"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generators = validate_generators(
        args.generators
    )

    weights = None

    if args.phase5_summary is not None:
        weights = build_weights_from_summary(
            args.phase5_summary
        )

    results, summary = run_dataset(
        dataset_root=args.dataset,
        scale=args.scale,
        generators=generators,
        budget=args.budget,
        min_distance_px=args.min_distance,
        weights=weights,
    )

    write_outputs(
        args.output_dir,
        results,
        summary,
    )

    recall = summary[
        "recall_at_k"
    ]

    print(
        "MICRONYX Phase 6 complete"
    )
    print(
        f"Observations : {summary['observations']}"
    )
    print(
        f"Recall@5    : {recall['recall_at_5']:.4f}"
    )
    print(
        f"Recall@25   : {recall['recall_at_25']:.4f}"
    )
    print(
        f"Recall@100  : {recall['recall_at_100']:.4f}"
    )
    print(
        f"Recall@250  : {recall['recall_at_250']:.4f}"
    )
    print(
        "CSV         : "
        f"{args.output_dir / 'robust_candidate_generation_results.csv'}"
    )
    print(
        "JSON        : "
        f"{args.output_dir / 'robust_candidate_generation_summary.json'}"
    )
    print(
        "Report      : "
        f"{args.output_dir / 'robust_candidate_generation_report.md'}"
    )


if __name__ == "__main__":
    main()