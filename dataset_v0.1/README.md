# MICRONYX Synthetic Dataset v0.1

Synthetic semiconductor-inspired reference/search pairs for PS-02 research.

## Contents
- DRAM-style and FinFET-style procedural structures
- 1000x1000 search images
- 100x100 reference images
- Independent reference/search noise
- Blur
- Edge brightening
- Intensity variation
- Rotation variation
- Ground-truth target center coordinates
- Hard-test split with stronger structural variation

## Important
This is **v0.1 synthetic semiconductor-inspired data**, not a physically validated SEM simulator.
The generator is intended for R&D and benchmarking and will be refined with literature-backed imaging/structure models.

## Reproducibility
Every pair stores its random seed and generation parameters in `metadata.json`.
