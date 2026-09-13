# MICRONYX Phase 6 — Robust Candidate Generation Report

> [!IMPORTANT]
> **Candidate-Pool Recall Boundary**: All Recall@K metrics reported in this phase evaluate the raw spatial candidate pool **BEFORE** any learned verification or ranking in Phase 7.

## Executive Summary

- **Observations Processed**: 100
- **Target Candidate Budget**: 250 candidates per observation
- **Spatial Separation Threshold**: 4.0 px NMS
- **Scale Factor**: 10.0×
- **Active Generators**: ncc, dog, gradient, edge, frequency
- **Total Candidates Generated (Before Deduplication)**: 25000
- **Total Candidates Remaining (After Deduplication)**: 23632

## Candidate Recall Performance (Pre-Verification)

| Metric | Recall Rate |
|---|---:|
| Recall@1 | 0.0000 |
| Recall@5 | 0.0000 |
| Recall@10 | 0.0000 |
| Recall@25 | 0.0000 |
| Recall@50 | 0.0000 |
| Recall@100 | 0.0000 |
| Recall@250 | 0.0200 |

## Localization Error Distribution

- **Median Best-Candidate Error**: 26.5236 px
- **Mean Best-Candidate Error**: 27.8072 px
- **P95 Best-Candidate Error**: 48.8913 px
- **Maximum Best-Candidate Error**: 74.7830 px

## Generator Contribution

| Generator | Best Candidate Count | Best Candidate Percentage |
|---|---:|---:|
| ncc | 11 | 11.00% |
| dog | 29 | 29.00% |
| gradient | 18 | 18.00% |
| edge | 13 | 13.00% |
| frequency | 29 | 29.00% |

## Hard-Negative Preservation & Failure Cases

- **Failure Count (Best Error > 5 px or Unrecalled)**: 98

All candidate pools (including hard negatives and failure cases) are fully preserved in JSON format for consumption by the Phase 7 learned verifier.

## System Invariants & Guarantees

1. **Non-Leakage Guarantee**: Target coordinates are never passed into candidate generation functions.
2. **Deterministic Order**: Candidates are sorted strictly by `(-score, generator, y, x)`.
3. **Adaptive Budgeting**: Generator budgets are allocated dynamically based on Phase 5 representation selection.
4. **Spatial Non-Maximum Suppression**: Duplicate candidates within `4.0` px are removed while preserving top responses.
