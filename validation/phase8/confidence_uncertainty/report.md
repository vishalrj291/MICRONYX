# MICRONYX Phase 8 — Confidence Calibration & Uncertainty Report

## Executive Summary

Phase 8 evaluates the scientific defensibility of confidence estimation for the MICRONYX target localization pipeline.
Raw probabilities output by the Phase 7 XGBoost candidate ranker (`raw_model_score`) are separated from calibrated confidence mappings (`calibrated_confidence`).

## Calibration Methodology & Definitions

1. **`raw_model_score` / `raw_probability`**: Uncalibrated output $P(y=1|x)$ from the Phase 7 `XGBClassifier` (`xgboost_ranker.json`).
2. **`calibrated_confidence`**: Probability estimate obtained via a separately fitted Platt scaling (Logistic Regression) model fitted strictly on the `train` split.
3. **Correctness Target Definition**: Candidate binary label $y = 1$ iff $\text{distance}(\text{candidate}, \text{target}) \le 5.0\text{ px}$, else $0$.
4. **Split Isolation & Anti-Leakage**: Calibration models are fitted **strictly on the `train` split**. The `hard_test` split is held out and untouched. Target coordinates never enter calibration features.

## Calibration Metric Summary

### Train Split ($N=16577$ candidates, 1 positive)

| Calibration Method | Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
|---|---:|---:|---:|
| **Raw XGBoost Score** | 0.000004 | 0.000144 | 0.240119 |
| **Platt Scaling** | 0.000002 | 0.000030 | 0.166252 |
| **Isotonic Regression** | 0.000000 | 0.000000 | 0.000000 |

### Validation Split ($N=3523$ candidates, 1 positive)

| Calibration Method | Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
|---|---:|---:|---:|
| **Raw XGBoost Score** | 0.000386 | 0.000320 | 0.598939 |
| **Platt Scaling** | 0.000380 | 0.000429 | 0.581727 |
| **Isotonic Regression** | 0.000347 | 0.000418 | 0.472335 |

### Held-Out Hard Test Split ($N=3532$ candidates, 0 positive)

| Calibration Method | Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
|---|---:|---:|---:|
| **Raw XGBoost Score** | 0.000001 | 0.000144 | 0.000144 |
| **Platt Scaling** | 0.000000 | 0.000022 | 0.000022 |
| **Isotonic Regression** | 0.000000 | 0.000000 | 0.000000 |

## Empirical Coordinate Uncertainty

Spatial dispersion $\sigma_{\text{spatial}}$ was computed per observation across the candidate probability distribution:
- **Mean Spatial Dispersion ($\sigma_{\text{spatial}}$)**: 352.93 px
- **Median Spatial Dispersion**: 364.58 px

## Critical Statistical Limitations

1. **Extreme Positive Sparsity**: The dataset contains only **1 positive candidate in `train`** ($1 / 16,577$), **1 in `validation`** ($1 / 3,523$), and **0 in `hard_test`** ($0 / 3,532$).
2. **Impact on Calibration Metrics**: Because positive candidates are virtually non-existent, raw predicted probabilities near zero yield small Brier Scores ($< 0.001$) and small ECEs ($< 0.001$). This numerical lightness is an artifact of class imbalance rather than true probability calibration.
3. **Confidence States**: Confidence-state thresholds (HIGH / MEDIUM / LOW) were **not promoted to production** because the available positive calibration sample size ($N_{\text{pos}}=1$) is statistically insufficient for non-arbitrary threshold estimation.

## Conclusion

Platt scaling successfully maps uncalibrated raw scores to bounded probabilities. However, due to extreme positive class sparsity in `dataset_v0.1`, calibrated confidence metrics must be interpreted with explicit statistical caution.
