# MICRONYX Phase 4–7 PDF Compliance Audit

This document presents the official compliance audit results for MICRONYX V2 Phases 4–7 against the authoritative **MICRONYX PS02 — V2 R&D Phase-Wise Work Plan**.

---

## Phase 4 — Multi-Scale Representation

| PDF Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| **Local context representation (10x10)** | `multiscale/feature_extractor.py` extracts 10x10 NCC context patch correlation `context_10`. | `multiscale/feature_extractor.py:L98-L103` | PASS |
| **Medium context representation (20x20, 40x40)** | `multiscale/feature_extractor.py` extracts 20x20 and 40x40 context patch correlations `context_20`, `context_40`. | `multiscale/feature_extractor.py:L99-L105` | PASS |
| **Large context representation (80x80)** | `multiscale/feature_extractor.py` extracts 80x80 context patch correlation `context_80`. | `multiscale/feature_extractor.py:L101-L106` | PASS |
| **Intensity features** | Extracts local mean & std intensity across scales (`intensity_mean_10`, `intensity_std_10`, etc.). | `multiscale/feature_extractor.py:L109-L112` | PASS |
| **Gradient features** | Extracts multi-scale gradient correlation & scale gradient gain. | `multiscale/feature_extractor.py:L115-L131` | PASS |
| **Edge features** | Extracts Canny edge density & edge response correlation across scales. | `multiscale/feature_extractor.py:L134-L138` | PASS |
| **Orientation features** | Extracts gradient orientation cosine similarity across 10x10 and 20x20 patches. | `multiscale/feature_extractor.py:L118-L130` | PASS |
| **Morphology features** | Extracts morphological gradient responses `morph_grad_10`, `morph_grad_20`. | `multiscale/feature_extractor.py:L141-L144` | PASS |
| **Texture features** | Extracts multi-scale patch variance across 10, 20, and 40 px patches. | `multiscale/feature_extractor.py:L147-L149` | PASS |
| **Frequency / periodicity features** | Extracts 2D FFT power spectrum high/low frequency energy ratio (`fft_energy_ratio_10`, `fft_energy_ratio_20`). | `multiscale/feature_extractor.py:L152-L162` | PASS |
| **Spatial-context features** | Extracts normalized spatial coordinates and distance to image center (`spatial_x_norm`, `spatial_y_norm`, `spatial_center_dist`). | `multiscale/feature_extractor.py:L165-L167` | PASS |
| **Compare V1 features vs multi-scale features & run ablations** | Evaluates V1 baseline vs 9 feature group additions (+intensity, +gradient, +edge, +orientation, +morphology, +texture, +frequency, +spatial, +combined). | `multiscale/feature_ablation.csv`, `multiscale/report.md` | PASS |
| **Learned embeddings (Optional)** | Optional extension — not required since engineered features did not hit a ceiling. | Work plan specifies learned embeddings only if engineered features hit a ceiling. | OPTIONAL-NOT-REQUIRED |

---

## Phase 5 — Advanced Ranking

| PDF Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| **Optimize current XGBoost (feature selection, hyperparameter opt, class balancing, hard-negative weighting)** | Evaluates hyperparameter-optimized XGBoost and class-balanced XGBoost (`scale_pos_weight`). | `ranking_v2/ranker.py:L148-L169` | PASS |
| **Compare pointwise, pairwise, and listwise ranking approaches** | Evaluates Pointwise XGBoost, Optimized Pointwise XGBoost, Pairwise XGBoost (`rank:pairwise`), and Listwise XGBoost (`rank:ndcg`). | `ranking_v2/ranker.py:L134-L188`, `comparison.csv`, `results.csv`, `training/training_results.csv` | PASS |
| **Recall@1, Recall@5, Recall@10, Recall@25, Recall@50, Recall@100, Recall@250 evaluation** | Calculates exact candidate recall at K levels across train, validation, hard_test, and overall splits. | `comparison.csv`, `ranking_v2/report.md` | PASS |
| **Evaluate learned matching/ranking architecture (Optional)** | Optional extension — evaluated Listwise/Pairwise XGBoost rankers without adding unnecessary neural complexity. | Work plan specifies optional for complex neural rankers. | OPTIONAL-NOT-REQUIRED |

---

## Phase 6 — Structural / Geometric Verification

| PDF Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| **Gradient similarity** | Calculates normalized gradient magnitude correlation between candidate patch and reference patch. | `geometry/verifier.py:L58-L61` | PASS |
| **Orientation similarity** | Calculates orientation cosine alignment weighted by gradient magnitudes. | `geometry/verifier.py:L63-L67` | PASS |
| **Edge-layout similarity** | Calculates Canny edge contour layout correlation between search patch and reference patch. | `geometry/verifier.py:L70-L73` | PASS |
| **Spatial-arrangement similarity** | Calculates spatial center offset margin & symmetry constraint score. | `geometry/verifier.py:L76-L78` | PASS |
| **Multi-scale context similarity** | Calculates multi-scale patch correlation across 10x10, 20x20, and 40x40 px context. | `geometry/verifier.py:L50-L55` | PASS |
| **Combine ML score and structural score & rerank candidates** | Combines ML probability and composite structural score using weighted sum ($Score_{final} = 0.7 \cdot P_{ML} + 0.3 \cdot S_{struct}$). | `geometry/verifier.py:L148`, `scores.csv`, `geometry/report.md` | PASS |
| **Report False Positives (FP Count, FP Rate, Recall@5/10/25px, True Candidate Rank)** | Measures false positive count ($dist > 5.0\text{ px}$ top-1), FP rate, and average true candidate rank across splits. | `geometry/report.md`, `validation/phase6/structural_verification/summary.json` | PASS |
| **Robust correspondence / transformation methods (Optional)** | Optional extension — failure analysis confirms candidate pool recall is primary bottleneck. | Work plan specifies optional if failure data justifies. | OPTIONAL-NOT-REQUIRED |

---

## Phase 7 — Sub-Pixel Refinement

| PDF Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| **Inspect response surface around winning candidate** | Extracts 3x3 local NCC response surface grid centered around winning candidate $(x_0, y_0)$. | `refinement/refine.py:L64-L96` | PASS |
| **Implement peak refinement (Quadratic / Gaussian fitting)** | Implements 2D Quadratic parabolic surface peak fitting and 2D Gaussian surface log fitting. | `refinement/refine.py:L26-L61` | PASS |
| **Compare integer-pixel error vs refined error** | Compares Discrete Integer Winner vs Parabolic Refined vs Gaussian Refined across train, validation, hard_test. | `refinement/refine.py:L148-L188`, `comparison.csv`, `refinement/report.md` | PASS |
| **Report metrics (Median, Mean, P95, Max error, Recall@1/5/10px, px & % improvement)** | Reports exact error statistics, Recall@1/5/10px, pixel improvement, and percentage improvement. | `refinement/report.md`, `validation/phase7/subpixel_refinement/summary.json` | PASS |
| **Non-leakage & Bounds validation** | Verifies zero target coordinate leakage, sub-pixel shift bounds $|\Delta x|, |\Delta y| \le 1.0$, image boundary clipping, and degenerate surface fallback. | `refinement/refine.py:L116-L125` | PASS |
