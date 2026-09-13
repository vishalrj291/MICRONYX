# MICRONYX Phase 4A — Automated EDA Report

## Scope

This report is generated automatically from the Phase 4 EDA profiler.

The profiler measures:

- intensity statistics
- contrast
- spatial correlation
- frequency content
- edge and gradient structure
- local texture
- high-frequency residual noise proxy

The structural labels are descriptive EDA descriptors.
They are not model-selection decisions.

## Dataset

- Observations: 100
- Structure classes: {"strongly_periodic": 84, "structured": 16}
- Sampling ratio: {}

## Mean search-image descriptors

- Mean intensity: 0.4099821565882353
- Contrast: 0.24593190802826237
- Gradient mean: 0.7349992644786835
- Edge density: 0.3324834899999999
- Spectral entropy: 0.5505889257986485
- Periodicity indicator: 0.7320177226353066
- Dominant period X (px): 1.17
- Dominant period Y (px): 1.22
- Correlation length X (px): 3.59
- Correlation length Y (px): 4.02
- High-frequency ratio: 0.16545277736247269
- High-pass residual std: 0.10324023485183716

## Physical context

Correlation lengths and dominant periods are reported in image pixels.

Conversion to physical units requires validated acquisition metadata.

Target coordinates are not used as image descriptors.

## Reproducibility

- CSV contains one row per analyzed observation.
- JSON contains aggregate statistics.
- Canonical synthetic mode uses `canonical_renderer.py`.
- Results should be regenerated when dataset or profiler code changes.

## Interpretation boundary

EDA identifies measurable properties of observations.

Phase 5 is responsible for validating whether these properties improve
model selection on held-out data.
