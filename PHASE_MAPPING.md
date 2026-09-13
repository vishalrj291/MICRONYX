# MICRONYX V2 — Phase Mapping Document

This document maps the existing MICRONYX codebase to the **official MICRONYX V2 R&D Phase-Wise Work Plan**.

## Official V2 Phase Alignment

| Official V2 Phase | Phase Name | Existing Implementation File / Directory | Evidence & Status | Status |
|---|---|---|---|---|
| **Phase 0** | Automated EDA | `scripts/automated_eda.py`<br>`validation/phase4/automated_eda/` | 5 unit tests pass (`test_automated_eda.py`). 100 observations processed. | **COMPLETE** |
| **Phase 1** | Representation Model Selection | `scripts/representation_model_selection.py`<br>`validation/phase5/representation_selection/` | 15 unit tests pass (`test_representation_model_selection.py`). Rule-based policy selected. | **COMPLETE** |
| **Phase 2** | Robust Candidate Generation | `scripts/robust_candidate_generation.py`<br>`validation/phase6/robust_candidate_generation/` | 14 unit tests pass (`test_robust_candidate_generation.py`). 23,632 post-NMS candidates. | **COMPLETE** |
| **Phase 3** | Hard-Negative Mining | `scripts/hard_negative_mining.py` | Mines high-scoring incorrect candidates ($dist > 5.0\text{ px}$); trains hard-negative XGBoost model. | **TO BE EXECUTED** |
| **Phase 4** | Multi-Scale Representation | `scripts/multiscale_feature_extraction.py` | Context ($10, 20, 40, 80$), texture, morphology, orientation, frequency, feature ablations. | **TO BE EXECUTED** |
| **Phase 5** | Advanced Ranking | `scripts/advanced_ranking.py` | Pairwise vs pointwise ranking, feature selection, class re-balancing. | **TO BE EXECUTED** |
| **Phase 6** | Structural / Geometric Verification | `scripts/structural_verification.py` | Independent gradient, orientation, edge-layout, spatial arrangement similarity score & re-ranking. | **TO BE EXECUTED** |
| **Phase 7** | Sub-Pixel Refinement | `scripts/subpixel_refinement.py` | Quadratic/parabolic response surface peak sub-pixel refinement ($\Delta x, \Delta y$). | **TO BE EXECUTED** |
| **Phase 8** | Confidence & Uncertainty | `uncertainty/calibration.py`<br>`validation/phase8/confidence_uncertainty/` | 15 unit tests pass (`test_calibration.py`). Platt/Isotonic, Brier, ECE, coordinate uncertainty, non-overclaim status (`insufficient_evidence`). | **COMPLETE** |
