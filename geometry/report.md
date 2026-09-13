# MICRONYX Phase 6 — Structural / Geometric Verification Report

## Executive Summary

Phase 6 incorporates independent geometric and structural similarity verification (combining Gradient Similarity, Orientation Similarity, Edge-Layout Similarity, Spatial-Arrangement Similarity, and Multi-Scale Context Similarity).

- **Alpha ML Weight**: 0.7
- **ML-Only Overall Median Error**: 399.61 px (FP Rate: 0.9900)
- **Structural-Only Overall Median Error**: 293.31 px (FP Rate: 1.0000)
- **Combined Overall Median Error**: 305.94 px (FP Rate: 0.9900)
- **Phase Decision**: **KEEP**

---

## Detailed Reranking Comparison Table Across Splits

| Split | Strategy | FP Count | FP Rate | Median Error | Mean Error | P95 Error | Recall@5px | Recall@10px | Recall@25px | True Cand Rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | ML Only | 69 | 0.9857 | 392.96 px | 396.15 px | 670.37 px | 0.0143 | 0.0143 | 0.0143 | 1.00 |
| **Train** | Structural Only | 70 | 1.0000 | 271.84 px | 292.36 px | 530.10 px | 0.0000 | 0.0000 | 0.0000 | 103.00 |
| **Train** | Combined ML+Struct | 69 | 0.9857 | 276.68 px | 295.28 px | 538.98 px | 0.0143 | 0.0143 | 0.0143 | 1.00 |
| **Validation** | ML Only | 15 | 1.0000 | 337.32 px | 371.28 px | 713.07 px | 0.0000 | 0.0000 | 0.0000 | 93.00 |
| **Validation** | Structural Only | 15 | 1.0000 | 364.84 px | 369.83 px | 762.96 px | 0.0000 | 0.0000 | 0.0000 | 212.00 |
| **Validation** | Combined ML+Struct | 15 | 1.0000 | 364.84 px | 379.57 px | 762.96 px | 0.0000 | 0.0000 | 0.0000 | 212.00 |
| **Hard_Test** | ML Only | 15 | 1.0000 | 462.91 px | 486.57 px | 705.25 px | 0.0000 | 0.0000 | 0.0000 | -1.00 |
| **Hard_Test** | Structural Only | 15 | 1.0000 | 398.61 px | 352.07 px | 562.91 px | 0.0000 | 0.0000 | 0.0000 | -1.00 |
| **Hard_Test** | Combined ML+Struct | 15 | 1.0000 | 398.61 px | 352.07 px | 562.91 px | 0.0000 | 0.0000 | 0.0000 | -1.00 |
| **Overall** | ML Only | 99 | 0.9900 | 399.61 px | 405.98 px | 690.88 px | 0.0100 | 0.0100 | 0.0100 | 47.00 |
| **Overall** | Structural Only | 100 | 1.0000 | 293.31 px | 312.94 px | 634.70 px | 0.0000 | 0.0000 | 0.0000 | 157.50 |
| **Overall** | Combined ML+Struct | 99 | 0.9900 | 305.94 px | 316.44 px | 642.29 px | 0.0100 | 0.0100 | 0.0100 | 106.50 |
