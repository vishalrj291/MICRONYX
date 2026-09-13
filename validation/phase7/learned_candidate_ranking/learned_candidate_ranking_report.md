# MICRONYX Phase 7 — Learned Candidate Ranking Report

## Executive Summary

- **Total Observations Processed**: 100
- **Split Distribution**: Train (70), Validation (15), Hard-Test (15)
- **Model Trained**: XGBoost Classifier (`xgboost_ranker.json`)
- **Training Constraints**: Fitted strictly on 70 `train` split observations.

## Baseline vs Learned Ranking Comparison

### Overall Dataset Performance (100 Observations)

| Metric | Phase 6 Baseline | Phase 7 XGBoost Ranker | Delta |
|---|---:|---:|---:|
| **Recall@5px Success Rate** | 0.0000 | 0.0100 | +0.0100 |
| **Median Localization Error** | 509.5734 px | 399.6059 px | -109.9675 px |
| **Mean Localization Error** | 506.6272 px | 405.9820 px | -100.6453 px |
| **P95 Localization Error** | 809.7204 px | 690.8782 px | -118.8422 px |
| **Maximum Error** | 866.1365 px | 821.7922 px | -44.3443 px |

### Held-Out Hard Test Split Performance (15 Observations)

| Metric | Phase 6 Baseline | Phase 7 XGBoost Ranker | Delta |
|---|---:|---:|---:|
| **Recall@5px Success Rate** | 0.0000 | 0.0000 | +0.0000 |
| **Median Localization Error** | 469.4939 px | 462.9109 px | -6.5830 px |

## Feature Importance

| Feature Name | XGBoost Importance |
|---|---:|
| `generator_context_gap` | 0.5619 |
| `context_10` | 0.2121 |
| `contrast_score` | 0.1187 |
| `context_consistency` | 0.0968 |
| `generator_score` | 0.0025 |
| `orientation_score` | 0.0021 |
| `generator_rank_normalized` | 0.0020 |
| `context_gain_40` | 0.0016 |
| `context_gain_20` | 0.0008 |
| `context_20` | 0.0007 |
| `context_40` | 0.0005 |
| `gradient_score` | 0.0004 |
| `generator_ncc` | 0.0000 |
| `generator_dog` | 0.0000 |
| `generator_gradient` | 0.0000 |
| `generator_edge` | 0.0000 |
| `generator_frequency` | 0.0000 |

## Anti-Leakage & Integrity Verification

1. **Feature Extraction Signature**: `extract_candidate_features(search_img, reference_img, candidate_x, candidate_y, ...)` does NOT receive target coordinates.
2. **Train/Test Isolation**: Model fit strictly on `train` split records ($N=70$). Validation and hard_test samples strictly excluded from training.
3. **Deterministic Inference**: Reproducible ranking using `random_state = 20260913`.
