# MICRONYX Phase 5 — Advanced Ranking Report

## Executive Summary

Phase 5 evaluates Pointwise XGBoost, Optimized Pointwise XGBoost, Pairwise XGBoost Ranker (`rank:pairwise`), and Listwise XGBoost Ranker (`rank:ndcg`).

- **Approaches Evaluated**: V1 Pointwise XGBoost, Optimized Pointwise XGBoost, Pairwise XGBoost Ranker, Listwise XGBoost Ranker
- **Validation Selected Model**: **Pairwise XGBoost Ranker**
- **Phase Decision**: **KEEP**

---

## Detailed Model Performance Table Across Splits

| Split | Model | Median Error | Mean Error | P95 Error | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Recall@50 | Recall@100 | Recall@250 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | V1 Pointwise XGBoost | 509.57 px | 503.49 px | 804.11 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0143 |
| **Train** | Optimized Pointwise XGBoost | 509.57 px | 503.49 px | 804.11 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0143 |
| **Train** | Pairwise XGBoost Ranker | 404.07 px | 394.17 px | 686.86 px | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 |
| **Train** | Listwise XGBoost Ranker | 404.07 px | 393.83 px | 694.47 px | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 | 0.0143 |
| **Validation** | V1 Pointwise XGBoost | 527.73 px | 548.45 px | 822.62 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| **Validation** | Optimized Pointwise XGBoost | 527.73 px | 548.45 px | 822.62 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| **Validation** | Pairwise XGBoost Ranker | 311.74 px | 327.09 px | 692.08 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| **Validation** | Listwise XGBoost Ranker | 311.74 px | 327.09 px | 692.08 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| **Hard_Test** | V1 Pointwise XGBoost | 469.49 px | 479.43 px | 759.29 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Hard_Test** | Optimized Pointwise XGBoost | 469.49 px | 479.43 px | 759.29 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Hard_Test** | Pairwise XGBoost Ranker | 482.35 px | 430.40 px | 671.91 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Hard_Test** | Listwise XGBoost Ranker | 482.35 px | 435.34 px | 671.91 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Overall** | V1 Pointwise XGBoost | 509.57 px | 506.63 px | 809.72 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0200 |
| **Overall** | Optimized Pointwise XGBoost | 509.57 px | 506.63 px | 809.72 px | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0200 |
| **Overall** | Pairwise XGBoost Ranker | 397.83 px | 389.54 px | 704.81 px | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0200 |
| **Overall** | Listwise XGBoost Ranker | 397.83 px | 390.05 px | 704.81 px | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0200 |
