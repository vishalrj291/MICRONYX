# MICRONYX Phase 9 — System-Level Localization Evaluation & Failure Analysis Report

## Executive Summary

Phase 9 provides an end-to-end evaluation of the MICRONYX target localization system across all 100 dataset observations ($70$ `train`, $15$ `validation`, $15$ `hard_test`).

- **Total Observations Evaluated**: 100
- **Phase 6 Baseline Median Error**: 509.57 px
- **Phase 7 XGBoost Ranker Median Error**: 399.61 px
- **Median Localization Error Reduction**: -109.97 px (65 / 100 observations improved)
- **Confidence Status**: `insufficient_evidence` (preserved non-overclaim status due to $N=2$ positive calibration target pools).

## End-to-End Metric Comparison by Split

| Split | Obs | Baseline Median Error | Phase 7 Median Error | Baseline Recall@5px | Phase 7 Recall@5px | Pool Coverage @5px |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 70 | 509.57 px | **392.96 px** | 0.0000 | **0.0143** | 0.0143 |
| **Validation** | 15 | 527.73 px | **337.32 px** | 0.0000 | **0.0000** | 0.0667 |
| **Hard Test** | 15 | 469.49 px | **462.91 px** | 0.0000 | **0.0000** | 0.0000 |
| **Overall** | 100 | 509.57 px | **399.61 px** | 0.0000 | **0.0100** | 0.0200 |

## Candidate Generation Quality vs. Learned Ranking Quality

The diagnostic separates upstream candidate generation recall from downstream ranking performance:
- **$P(\text{target in candidate pool} \le 5\text{px})$**: **2.0%** (2 / 100 observations).
- **$P(\text{Phase 7 ranks target \#1} \mid \text{target in pool})$**: **50.0%** (1 / 2 observations).
- **Mean Reciprocal Rank (MRR)**: **0.5054** across positive pools.

## Failure Taxonomy Breakdown

Across the 99 failure cases ($dist > 5.0\text{ px}$):
- **Candidate Not Generated ($dist > 5.0\text{px}$ in pool)**: **93** cases (93.9%)
- **Periodic Structure Ambiguity / Ranking Failure**: **1** cases (1.0%)

## Production Readiness Conclusion

The Phase 7 learned ranker delivers a verified **~110 px reduction in median localization error**. However, industrial-grade localization at $\le 5.0\text{ px}$ is constrained upstream by Phase 6 candidate generation recall (98% of target candidates absent from candidate pools). `confidence_status` remains `insufficient_evidence` until candidate pool recall is reformed.
