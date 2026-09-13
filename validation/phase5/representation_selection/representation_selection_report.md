# MICRONYX Phase B — Automated Representation & Model Selection

## Scope

Phase B replaces permanently hard-coded representation selection with a
data-driven selection layer.

The candidate representations evaluated are:

- NCC
- DoG
- gradient
- edge
- frequency

The implementation compares:

1. rule-based selection
2. optimization-based selection
3. learned selection

Evaluation is performed on held-out scenes.

## Dataset

Observations evaluated: 100

Train observations: 70

Test observations: 15

Scale used for the representation benchmark:
10.0x

## Representation benchmark

{
  "ncc": {
    "median_error_px": 602.3163620556892,
    "mean_error_px": 532.4505637974544,
    "p95_error_px": 841.3487613997837,
    "worst_case_error_px": 859.4137536716527,
    "median_runtime_ms": 24.679500027559698
  },
  "dog": {
    "median_error_px": 370.06080581439585,
    "mean_error_px": 445.8921610271037,
    "p95_error_px": 818.9075099556363,
    "worst_case_error_px": 951.4452164996154,
    "median_runtime_ms": 34.81650003232062
  },
  "gradient": {
    "median_error_px": 386.55918046270745,
    "mean_error_px": 441.15075003524885,
    "p95_error_px": 778.4954599337354,
    "worst_case_error_px": 820.1585212628105,
    "median_runtime_ms": 19.91580007597804
  },
  "edge": {
    "median_error_px": 640.5388356688453,
    "mean_error_px": 580.1951846857916,
    "p95_error_px": 816.4199703364624,
    "worst_case_error_px": 881.7193431018738,
    "median_runtime_ms": 36.602400010451674
  },
  "frequency": {
    "median_error_px": 338.92329515688357,
    "mean_error_px": 320.583234134913,
    "p95_error_px": 397.4782529847152,
    "worst_case_error_px": 412.71055232450743,
    "median_runtime_ms": 119.9750000378117
  }
}

## Selection policies

{
  "rule_based": {
    "policy": "rule_based",
    "n": 15,
    "median_error_px": 201.63581031156147,
    "mean_error_px": 202.43346181776897,
    "p95_error_px": 318.9751279614062,
    "worst_case_error_px": 361.3668496140729,
    "median_runtime_ms": 29.821199947036803,
    "selection_distribution": {
      "ncc": 2,
      "dog": 3,
      "gradient": 6,
      "edge": 1,
      "frequency": 3
    }
  },
  "optimization_based": {
    "policy": "optimization_based",
    "n": 15,
    "median_error_px": 370.06080581439585,
    "mean_error_px": 445.8921610271037,
    "p95_error_px": 818.9075099556363,
    "worst_case_error_px": 951.4452164996154,
    "median_runtime_ms": 34.81650003232062,
    "selection_distribution": {
      "ncc": 0,
      "dog": 15,
      "gradient": 0,
      "edge": 0,
      "frequency": 0
    }
  },
  "learned": {
    "policy": "learned",
    "n": 15,
    "median_error_px": 271.41296947640507,
    "mean_error_px": 267.1951629179546,
    "p95_error_px": 479.8575752036527,
    "worst_case_error_px": 754.9423819073877,
    "median_runtime_ms": 26.30280004814267,
    "selection_distribution": {
      "ncc": 1,
      "dog": 2,
      "gradient": 8,
      "edge": 1,
      "frequency": 3
    }
  }
}

## Selected policy

rule_based

## Human override

None

## Leakage protection

EDA features are derived from the observed image.

Target coordinates are evaluation labels only.

Target coordinates are NOT included in:

- EDA features
- rule-policy inputs
- optimization-policy inputs
- learned-policy feature vectors

The learned policy is trained on training scenes and evaluated on
held-out scenes.

## Model-selection boundary

Phase B selects and weights representations.

It does not implement the final production candidate-generation
interface.

Candidate-pool construction, Recall@K, duplicate suppression,
spatial-diversity enforcement and hard-negative mining belong to Phase C.

## Decision principle

No representation is declared universally best.

The selected policy is the policy that provides the best held-out
validation evidence under the configured evaluation metric.

## Reproducibility

Dataset version:
bc5a88cc2243933a

Phase-A EDA fingerprint:
C:\Users\kirti\Downloads\MICRONYX-main (1)\MICRONYX-main\validation\phase4\automated_eda\automated_eda_results.csv

Random seed:
20260913
