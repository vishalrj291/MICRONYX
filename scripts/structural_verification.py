from __future__ import annotations

import argparse
from pathlib import Path
from geometry.verifier import (
    run_full_structural_verification_pipeline,
    compute_all_five_structural_components,
    DEFAULT_OUTPUT_DIR,
)

run_structural_verification_pipeline = run_full_structural_verification_pipeline

def compute_structural_similarity_score(search_img, reference_img, cx, cy):
    comps = compute_all_five_structural_components(search_img, reference_img, cx, cy)
    return comps["composite_structural_score"]

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
    print(f"MICRONYX Phase 6 complete. Decision: {summary['decision']}. Artifacts written to: {args.output_dir}")

if __name__ == "__main__":
    main()
