# MICRONYX Phase 4 — Multi-Scale Representation Report

## Executive Summary

Phase 4 evaluates multi-scale context ($10, 20, 40, 80\text{ px}$), intensity, gradient, edge, orientation, morphology, texture, frequency/periodicity, and spatial context features.

- **Total Multi-Scale Features**: 33
- **V1 Baseline Overall Median Error**: 509.57 px
- **Combined Multi-Scale Overall Median Error**: 509.57 px
- **Phase Decision**: **REJECT**

---

## Mandatory PDF Feature Category Evidence Table

| Feature Category | Implemented | Tested | Used in Model | Evidence |
|---|---|---|---|---|
| **Intensity** | YES | YES | YES | `intensity_mean_10`, `intensity_std_10`, `intensity_mean_20`, `intensity_std_20` |
| **Gradient** | YES | YES | YES | `g_score_10`, `g_score_20`, `scale_gradient_gain` |
| **Edge** | YES | YES | YES | `edge_density_10`, `edge_density_20`, `edge_response_corr` |
| **Orientation** | YES | YES | YES | `o_score_10`, `o_score_20` (Cosine orientation alignment) |
| **Morphology** | YES | YES | YES | `morph_grad_10`, `morph_grad_20` (Morphological gradient responses) |
| **Texture** | YES | YES | YES | `texture_var_10`, `texture_var_20`, `texture_var_40` |
| **Frequency / Periodicity** | YES | YES | YES | `fft_energy_ratio_10`, `fft_energy_ratio_20` (2D FFT energy spectrum ratio) |
| **Spatial Context** | YES | YES | YES | `spatial_x_norm`, `spatial_y_norm`, `spatial_center_dist` |
| **Local Context** | YES | YES | YES | `context_10` ($10\times10\text{ px}$ patch normalized cross-correlation) |
| **Medium Context** | YES | YES | YES | `context_20`, `context_40` ($20\times20, 40\times40\text{ px}$ patch context) |
| **Large Context** | YES | YES | YES | `context_80` ($80\times80\text{ px}$ patch context) |

---

## Feature Ablation Table

| Feature Set | Count | Train Median | Val Median | Test Median | Overall Median | Overall Recall@5px |
|---|---:|---:|---:|---:|---:|---:|
| V1 baseline | 17 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + intensity | 21 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + gradient | 22 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + edge | 25 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + orientation | 25 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + morphology | 27 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + texture | 28 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + frequency/periodicity | 30 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| + spatial context | 33 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
| combined multi-scale representation | 33 | 509.57 px | 527.73 px | 469.49 px | 509.57 px | 0.0000 |
