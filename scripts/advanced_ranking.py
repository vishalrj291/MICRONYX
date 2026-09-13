from __future__ import annotations

import argparse
from pathlib import Path
from ranking_v2.ranker import run_full_advanced_ranking_pipeline, DEFAULT_OUTPUT_DIR

run_advanced_ranking_pipeline = run_full_advanced_ranking_pipeline

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
    print(f"MICRONYX Phase 5 complete. Decision: {summary['decision']}. Artifacts written to: {args.output_dir}")

if __name__ == "__main__":
    main()
