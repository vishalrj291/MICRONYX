# MICRONYX PS02

## Physics-Aware Multi-Resolution Semiconductor Target Localization

MICRONYX is a research-oriented computer vision and machine learning system for **localizing a high-resolution semiconductor reference region inside a lower-resolution search image** under multi-resolution acquisition conditions.

The system combines a canonical physical acquisition model, structural candidate generation, multi-scale physical-context analysis, and learned candidate ranking to identify the most probable target location.

---

## 🌐 Live Demo

**Frontend:**  
https://micronyx.vercel.app/

**Backend API:**  
https://micronyx.onrender.com/

**API Documentation:**  
https://micronyx.onrender.com/docs

**Source Code:**  
https://github.com/vishalrj291/MICRONYX

---

## 🎯 Problem

Semiconductor inspection and microscopy workflows may observe the same physical region at different spatial sampling rates.

Given:

- A large-area, lower-resolution **search image**
- A smaller-area, higher-resolution **reference image**

MICRONYX determines where the physical region represented by the reference is located inside the search image.

The problem becomes difficult when semiconductor structures are:

- Periodic
- Quasi-periodic
- Repetitive
- Locally ambiguous
- Observed at different resolutions
- Affected by sampling differences

A simple template-matching system can therefore return highly similar but physically incorrect locations.

MICRONYX addresses this using a **candidate-generation + contextual verification + learned-ranking architecture**.

---

# 🧠 System Overview

```text
             HIGH-RESOLUTION REFERENCE
                       │
                       ▼
             Canonical Acquisition Model
                       │
                       ▼
              Physical Context Extraction
                       │
                       ▼
                SEARCH IMAGE
                       │
                       ▼
              Structural Candidate
                 Generation (DoG)
                       │
                       ▼
                Candidate Pool
                       │
                       ▼
              Feature Extraction
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       DoG/Ranks    Gradients    Multi-Scale
                                  Context
          │            │            │
          └────────────┼────────────┘
                       ▼
                XGBoost Ranker
                       │
                       ▼
              Final Localization
                       │
                       ▼
              Target Coordinates
```

The final system therefore does not rely on a single raw template-matching score.

---

# 📐 Canonical Acquisition Model

MICRONYX explicitly models the difference between search and reference acquisition.

### Search

```text
Resolution:       1000 × 1000
Sampling:         5 pixels / physical unit
Physical FOV:     200 × 200 units
```

### Reference

```text
Resolution:       1000 × 1000
Sampling:         50 pixels / physical unit
Physical FOV:     20 × 20 units
```

### Sampling Ratio

```text
50 / 5 = 10×

Reference sampling is 10× finer than search sampling.
```

This relationship is explicitly represented rather than treating the images as identical-resolution observations.

---

# 🔬 Physical Context Modeling

MICRONYX evaluates candidates using multiple physical context scales.

| Physical Context | Reference Crop | Search Equivalent |
|---|---:|---:|
| 2 units | 100 × 100 | 10 × 10 |
| 4 units | 200 × 200 | 20 × 20 |
| 8 units | 400 × 400 | 40 × 40 |

The high-resolution reference contexts are converted into search-equivalent observations using deterministic block averaging based on the known 10× sampling ratio.

No arbitrary template resizing is required for this physical-context conversion.

---

# 🔎 Candidate Generation

The current production localization pipeline uses **Difference-of-Gaussians (DoG)** based structural candidate generation.

The purpose of candidate generation is to achieve high candidate recall rather than directly make the final localization decision.

```text
Search Image
     │
     ▼
DoG Representation
     │
     ▼
Candidate Response Map
     │
     ▼
Top Candidate Pool
```

The production pipeline currently evaluates up to:

```text
250 candidates
```

before learned ranking.

---

# 🧩 Candidate Features

Each candidate is represented using structural and contextual information.

The learned ranker uses features including:

```text
dog_rank_normalized
dog_context_gap
orientation_score
context_10
context_20
context_consistency
context_gain_20
context_gain_40
gradient_score
dog_score
contrast_score
context_40
```

This allows the model to consider both local structure and larger physical context.

---

# 🤖 Learned Candidate Ranking

MICRONYX uses an **XGBoost classifier as a candidate ranker**.

Candidate labels are defined using a localization tolerance:

```text
Positive candidate:
distance from ground truth ≤ 5 px

Negative candidate:
distance from ground truth > 5 px
```

The model estimates the probability that each candidate represents the correct target.

The candidate with the highest predicted probability becomes the final localization.

---

# 🧪 Validation Framework

MICRONYX includes a broad validation framework covering multiple components and failure modes.

Current validation experiments include:

```text
✓ Candidate recall benchmarking
✓ Candidate verification
✓ Sampling alignment
✓ Aperiodic sampling
✓ Multi-scale context verification
✓ Multi-feature descriptors
✓ Gradient/NCC matching
✓ Hard-negative analysis
✓ Natural periodicity
✓ Coarse-to-fine localization
✓ Learned candidate ranking
✓ Policy-constrained candidate generation
✓ Renderer compliance
✓ Automated EDA
✓ Automated representation selection
✓ Multi-generator recall
✓ Residual matching
```

---

# 📊 Sampling Alignment Validation

The corrected sampling-alignment experiment evaluated:

```text
400 phase combinations
```

Observed results:

```text
GT Top-1 rate:          92.25%
Localization ≤ 1 px:    98.75%
Localization ≤ 5 px:    98.75%

Median GT score:         0.699036
Median localization:     0.4031 px
Worst GT rank:            15
```

The experiment also identified large-error cases caused by periodic ambiguity.

These failure cases are retained as part of the validation rather than being removed.

---

# 📊 Multi-Scale Context Validation

The multi-scale context verification experiment evaluated:

```text
60 scenes

30 periodic
30 quasiperiodic
```

using physical contexts of:

```text
2 units
4 units
8 units
```

### Periodic scenes

```text
Recall@5px = 100%
```

### Quasiperiodic scenes

```text
Recall@5px = 83.33%
```

This demonstrates that quasiperiodic structures remain significantly more difficult because multiple structurally similar locations can exist.

---

# 🧠 Learned Ranker Validation

The learned ranker was evaluated on validation and test scenes.

A successful production-style inference example:

```text
Predicted center:
(376.000, 569.000) px

Model confidence:
0.999962

Candidates:
250

Localization runtime:
~150 ms
```

Top candidate:

```text
Rank:        1
X:           376.000
Y:           569.000
Probability: 0.999962
DoG score:   0.857543
```

The validation dataset also contains hard failures where:

- The correct candidate was not recalled
- A hard negative outranked the correct candidate
- Periodic ambiguity caused large localization errors

These cases are important because candidate ranking cannot recover a target that candidate generation fails to propose.

---

# 🧪 Canonical Renderer Compliance

MICRONYX includes a dedicated canonical acquisition compliance test.

The compliance test verifies the required renderer API:

```text
✓ periodic_scene()
✓ quasiperiodic_scene()
✓ continuous_scene()
✓ render_search()
✓ render_reference()
✓ render_sensor()
✓ create_ps02_template()
✓ template_top_left()
✓ generate_observation()
```

It also verifies:

```text
Search sampling:      5 px/unit
Reference sampling:  50 px/unit
Sampling ratio:       10×

Search:               1000 × 1000
Reference:            1000 × 1000
Template:             100 × 100
```

The canonical acquisition model therefore remains explicit and internally consistent.

---

# ⚙️ Production Architecture

The deployed backend wraps the existing canonical localization pipeline.

```text
User
 │
 ▼
React Frontend
 │
 ▼
FastAPI Backend
 │
 ▼
Image Validation
 │
 ▼
micronyx_localize.py
 │
 ▼
DoG Candidate Generation
 │
 ▼
Feature Extraction
 │
 ▼
XGBoost Candidate Ranking
 │
 ▼
Localization Result
 │
 ▼
React Interface
```

The API does not implement a separate ML algorithm.

It serves the existing localization engine.

---

# 🚀 Production API

## Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "localizer_available": true,
  "model_available": true
}
```

## Localization

```http
POST /api/localize
```

Multipart inputs:

```text
search
reference
```

Expected dimensions:

```text
Search:     1000 × 1000
Reference:  1000 × 1000
```

The API returns:

```text
Predicted coordinates
Confidence
Candidate count
Runtime
Top candidates
Acquisition metadata
```

---

# 🖥️ Frontend

MICRONYX provides a React-based technical inspection interface.

The interface includes:

```text
• System health
• Acquisition information
• Search image inspection
• Reference image inspection
• Localization controls
• Candidate ranking
• Confidence visualization
• Localization results
```

The frontend is designed as a technical inspection interface rather than a generic AI SaaS dashboard.

---

# 🛠️ Technology Stack

## Backend

```text
Python
FastAPI
Uvicorn
OpenCV
NumPy
Scikit-learn
XGBoost
```

## Machine Learning

```text
XGBoost
Candidate ranking
Feature engineering
Multi-scale contextual analysis
```

## Computer Vision

```text
OpenCV
Difference-of-Gaussians
Template matching
Gradient analysis
Multi-resolution sampling
Spatial candidate generation
```

## Frontend

```text
React
Vite
Tailwind CSS
Lucide React
```

## Deployment

```text
Frontend  → Vercel
Backend   → Render
```

---

# 📁 Repository Structure

```text
MICRONYX/
│
├── dataset_v0.1/
│
├── docs/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── App.css
│   ├── package.json
│   └── vite.config.js
│
├── scripts/
│   ├── api_server.py
│   ├── micronyx_localize.py
│   ├── canonical_renderer.py
│   ├── generate_final_demo.py
│   ├── learned_candidate_ranker.py
│   ├── dog_candidate_verification.py
│   ├── multiscale_context_verification.py
│   ├── sampling_alignment_test.py
│   ├── aperiodic_sampling_test.py
│   ├── renderer_compliance_test.py
│   └── ...
│
├── validation/
│   └── v02/
│       ├── learned_ranker/
│       ├── sampling_alignment/
│       ├── multiscale_context/
│       ├── aperiodic_sampling/
│       ├── candidate_recall/
│       └── ...
│
├── requirements.txt
├── .python-version
├── .gitignore
└── LICENSE
```

---

# 💻 Local Setup

## 1. Clone

```bash
git clone https://github.com/vishalrj291/MICRONYX.git
cd MICRONYX
```

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3. Start Backend

```bash
uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Health:

```text
http://127.0.0.1:8000/health
```

API docs:

```text
http://127.0.0.1:8000/docs
```

---

# 🌐 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

For production:

```text
VITE_API_BASE_URL=https://micronyx.onrender.com
```

---

# ☁️ Deployment

## Frontend

The frontend is deployed on Vercel:

https://micronyx.vercel.app/

## Backend

The FastAPI inference service is deployed on Render:

https://micronyx.onrender.com/

---

# 🔁 End-to-End Workflow

```text
1. User opens MICRONYX
          ↓
2. Frontend checks backend health
          ↓
3. User provides search + reference images
          ↓
4. Images are uploaded to FastAPI
          ↓
5. Backend validates dimensions
          ↓
6. Canonical localization engine runs
          ↓
7. DoG generates candidate locations
          ↓
8. Candidate features are extracted
          ↓
9. XGBoost ranks candidates
          ↓
10. Highest-probability candidate is selected
          ↓
11. Coordinates + confidence are returned
          ↓
12. Frontend visualizes the result
```

---

# 🔬 Reproducibility

Important canonical parameters:

```text
Search resolution:
1000 × 1000

Reference resolution:
1000 × 1000

Search sampling:
5 pixels/unit

Reference sampling:
50 pixels/unit

Sampling ratio:
10×

Search FOV:
200 × 200 physical units

Reference FOV:
20 × 20 physical units
```

Controlled scene generation is used for reproducible validation.

---

# ⚠️ Current Limitations

MICRONYX is currently a **controlled research prototype**, not a validated industrial semiconductor inspection product.

Current limitations include:

- Validation is primarily based on controlled/synthetic scenes.
- Periodic structures can create severe localization ambiguity.
- Quasiperiodic scenes are significantly harder.
- Candidate recall limits the final ranker's performance.
- Hard negatives can still fool the learned ranker.
- The current renderer is not a full physical SEM simulator.
- Real SEM noise, charging effects, process variation and instrument-specific distortions require further validation.
- Production deployment demonstrates the inference pipeline, but does not by itself establish industrial-grade accuracy.

---

# 🔮 Future Research

Potential future improvements include:

```text
• Learned multi-scale feature encoders
• Siamese / metric-learning architectures
• Transformer-based image matching
• Uncertainty-aware localization
• Improved hard-negative mining
• Adaptive candidate generation
• Real SEM dataset validation
• Sensor/noise domain adaptation
• Sub-pixel localization refinement
• Probabilistic localization maps
• Active learning
• Dataset-adaptive model selection
• Physics-informed representation learning
• GPU-accelerated inference
```

---

# 🧭 Design Principles

### 1. Explicit acquisition modeling

Search and reference images are treated as observations acquired at different spatial sampling rates.

### 2. Candidate generation and ranking are separated

Candidate generation focuses on recall.

Candidate ranking focuses on selecting the most probable target.

### 3. Physical context matters

A candidate is evaluated using multiple spatial contexts rather than relying exclusively on a small local patch.

### 4. Failure cases are preserved

Hard negatives and ambiguous scenes are retained during validation rather than being silently removed.

### 5. Reproducibility matters

The canonical renderer and acquisition parameters are explicitly defined and tested.

---

# 📌 Project Status

```text
Canonical renderer                  ✓
Multi-resolution acquisition        ✓
Physical context modeling           ✓
DoG candidate generation            ✓
Multi-scale verification            ✓
Feature extraction                  ✓
XGBoost candidate ranking           ✓
Validation framework                ✓
Sampling alignment validation       ✓
Renderer compliance                 ✓
FastAPI backend                     ✓
Trained XGBoost model               ✓
React frontend                      ✓
Vercel deployment                   ✓
Render deployment                   ✓
End-to-end inference                ✓
```

---

# 🎬 Demo Result

Example production inference:

```text
MICRONYX PS02 FINAL LOCALIZATION

Predicted center:
(376.000, 569.000) px

Model confidence:
0.999962

Candidates:
250

Localization runtime:
~150 ms
```

This demonstrates the complete path from image input to learned candidate ranking and final localization.

---

# 🏆 Project Summary

MICRONYX approaches semiconductor target localization as a **multi-resolution physical observation and candidate-ranking problem**.

Instead of relying on a single template match, the system combines:

```text
Physical acquisition modeling
        +
Multi-scale context
        +
Structural candidate generation
        +
Feature engineering
        +
Learned candidate ranking
        +
Deterministic inference
```

The resulting architecture provides a foundation for further research into robust semiconductor image localization under periodicity, sampling variation and structural ambiguity.

---

## MICRONYX PS02

**Physics-aware. Multi-resolution. Candidate-driven. Learned ranking.**

Built as a research-oriented semiconductor localization platform.
