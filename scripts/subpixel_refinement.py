from __future__ import annotations

import argparse
from pathlib import Path
from refinement.refine import (
    run_full_subpixel_refinement_pipeline,
    fit_parabolic_subpixel_offset,
    fit_gaussian_subpixel_offset,
    extract_3x3_response_grid,
    DEFAULT_OUTPUT_DIR,
)

run_subpixel_refinement_pipeline = run_full_subpixel_refinement_pipeline
fit_subpixel_parabola = fit_parabolic_subpixel_offset

def compute_subpixel_refined_location(search_img, ref_img, cx, cy):
    grid_3x3 = extract_3x3_response_grid(search_img, ref_img, cx, cy)
    dx, dy = fit_parabolic_subpixel_offset(grid_3x3)
    return float(cx + dx), float(cy + dy), dx, dy

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
    print(f"MICRONYX Phase 7 complete. Decision: {summary['decision']}. Artifacts written to: {args.output_dir}")

if __name__ == "__main__":
    main()
