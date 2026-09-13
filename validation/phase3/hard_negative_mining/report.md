# MICRONYX Phase 3 — Hard-Negative Mining Report

## Executive Summary

Phase 3 evaluates hard-negative mining by collecting high-scoring incorrect candidates ($dist > 5.0\text{ px}$) from training candidate pools and weighting them during XGBoost retraining.

- **Mined Hard Negatives**: 1050 candidates across Train split
- **Baseline Median Error (Overall)**: 399.61 px
- **Hard-Negative XGBoost Median Error (Overall)**: 372.00 px
- **Phase Decision**: **KEEP**

## Baseline vs Hard-Negative Model Metric Comparison

| Split | Model Version | Median Error | Mean Error | P95 Error | Recall@5px |
|---|---|---:|---:|---:|---:|
| **Train** | Baseline | 392.96 px | 396.15 px | 670.37 px | 0.0143 |
| **Train** | Hard-Negative | 391.02 px | 389.52 px | 695.03 px | 0.0143 |
| **Validation** | Baseline | 337.32 px | 371.28 px | 713.07 px | 0.0000 |
| **Validation** | Hard-Negative | 311.74 px | 310.23 px | 557.05 px | 0.0000 |
| **Hard Test** | Baseline | 462.91 px | 486.57 px | 705.25 px | 0.0000 |
| **Hard Test** | Hard-Negative | 389.50 px | 379.62 px | 683.73 px | 0.0000 |
