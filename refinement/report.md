# MICRONYX Phase 7 — Sub-Pixel Refinement Report

## Executive Summary

Phase 7 evaluates 2D Parabolic fitting and 2D Gaussian fitting over a 3x3 local NCC response surface grid around the winning candidate location to refine localization accuracy to sub-pixel precision.

- **Original Integer Median Error (Overall)**: 399.61 px (Mean: 405.98 px)
- **Parabolic Refined Median Error (Overall)**: 399.55 px (Mean: 405.98 px)
- **Gaussian Refined Median Error (Overall)**: 399.55 px (Mean: 405.98 px)
- **Mean Error Improvement**: 0.0068 px (0.00%)
- **Phase Decision**: **KEEP**

---

## Refinement Performance Table Across Splits

| Split | Method | Median Error | Mean Error | P95 Error | Max Error | Recall@1px | Recall@5px | Recall@10px |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | Integer Baseline | 392.96 px | 396.15 px | 670.37 px | 755.53 px | 0.0000 | 0.0143 | 0.0143 |
| **Train** | Parabolic Refined | 393.06 px | 396.14 px | 671.04 px | 755.15 px | 0.0000 | 0.0143 | 0.0143 |
| **Train** | Gaussian Refined | 393.06 px | 396.14 px | 671.04 px | 755.15 px | 0.0000 | 0.0143 | 0.0143 |
| **Validation** | Integer Baseline | 337.32 px | 371.28 px | 713.07 px | 821.79 px | 0.0000 | 0.0000 | 0.0000 |
| **Validation** | Parabolic Refined | 337.91 px | 371.30 px | 713.77 px | 822.50 px | 0.0000 | 0.0000 | 0.0000 |
| **Validation** | Gaussian Refined | 337.91 px | 371.30 px | 713.77 px | 822.50 px | 0.0000 | 0.0000 | 0.0000 |
| **Hard_Test** | Integer Baseline | 462.91 px | 486.57 px | 705.25 px | 740.74 px | 0.0000 | 0.0000 | 0.0000 |
| **Hard_Test** | Parabolic Refined | 462.51 px | 486.53 px | 704.58 px | 740.12 px | 0.0000 | 0.0000 | 0.0000 |
| **Hard_Test** | Gaussian Refined | 462.51 px | 486.53 px | 704.58 px | 740.12 px | 0.0000 | 0.0000 | 0.0000 |
| **Overall** | Integer Baseline | 399.61 px | 405.98 px | 690.88 px | 821.79 px | 0.0000 | 0.0100 | 0.0100 |
| **Overall** | Parabolic Refined | 399.55 px | 405.98 px | 690.20 px | 822.50 px | 0.0000 | 0.0100 | 0.0100 |
| **Overall** | Gaussian Refined | 399.55 px | 405.98 px | 690.20 px | 822.50 px | 0.0000 | 0.0100 | 0.0100 |
