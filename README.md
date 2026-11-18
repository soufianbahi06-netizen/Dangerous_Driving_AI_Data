# Dangerous Driving AI — Data Repository

A data-engineering and validation repository for a project that aims to analyze driving behaviour and develop machine-learning models for detecting potentially dangerous driving patterns.

> **Current status:** Phase 9.2 and Phase 10.1 data processing and validation have been completed and reviewed. Model training, independent evaluation, and deployment have not yet been completed. No production-ready AI model is claimed.

## 1. Dataset Overview

The current processed dataset contains:

| Measure | Value |
|---|---:|
| Vehicles | 320 |
| Vehicle-trip records | 2,571 |
| Recorded samples | 1,585,741 |
| Recorded duration | 1,101,721.1 seconds |
| Audited events | 25,744 |
| Behaviour episodes | 18,696 |
| Trips containing events | 2,489 |
| Trips containing zero events | 82 |

These are dataset statistics, not model-performance metrics.

## 2. Repository Contents

| File | Description |
|---|---|
| `Trip_Event_Summary.csv` | Source trip-level event summary |
| `Phase9_2_Event_Audit.csv` | Audited event-level records |
| `Phase9_2_Behavior_Episodes.csv` | Grouped behaviour episodes |
| `Phase9_2_Transitions.csv` | Event transition statistics |
| `Phase10_1_Feature_Engineering.py` | Feature engineering and validation script |
| `Phase10_1_Trip_Features.csv` | Trip-level engineered features |
| `Phase10_1_Vehicle_Features.csv` | Vehicle-level aggregated features |

## 3. Event Categories

| Event Type | Count |
|---|---:|
| HARD_ACCELERATION | 4,297 |
| MODERATE_ACCELERATION | 8,578 |
| HARD_BRAKING | 3,256 |
| MODERATE_BRAKING | 9,613 |
| **Total** | **25,744** |

The event thresholds and their original definitions should be documented from the upstream event-generation methodology. This README does not redefine them.

## 4. Phase 9.2 — Event and Behaviour Analysis

### Event Audit

- Total audited events: 25,744.
- Unique event identifiers: 25,744.
- PASS status: 25,443.
- ROUNDING_AMBIGUOUS status: 301.
- No duplicate event identifiers or missing required fields were found in the reviewed file.
- Event totals and categories reconciled with the Phase 10.1 trip features.

`ROUNDING_AMBIGUOUS` is an audit status, not an additional event category.

### Behaviour Episodes

| Behaviour Pattern | Episodes |
|---|---:|
| ACCELERATION_ONLY | 8,213 |
| BRAKING_ONLY | 8,379 |
| MIXED_ACCELERATION_AND_BRAKING | 2,104 |
| **Total** | **18,696** |

The reviewed episode output reconciled with all 25,744 audited event identifiers. Duration, event counts, event sequences, and event-type counts passed the documented consistency checks.

### Event Transitions

- Total transitions within the configured 0–60 second window: 11,796.
- Acceleration-to-braking: 2,754.
- Braking-to-acceleration: 2,608.
- Alternation transitions: 5,362.

The transition summary passed checks for non-negative gaps, internally consistent statistics, and the configured maximum gap.

## 5. Phase 10.1 — Feature Engineering

### Trip-Level Features

- 2,571 records.
- 38 columns.
- 2,571 unique `Vehicle_ID` + `Trip_ID` pairs.
- 82 trips with zero events.
- 2,489 trips with one or more events.

The table includes event counts, event rates, episode counts, transition counts, recording duration, sample coverage, and derived ratios and shares.

### Vehicle-Level Features

- 320 records.
- 30 columns.
- Aggregated trip counts, duration, samples, event counts, episode counts, and transition counts.

Vehicle-level aggregate counts were reconciled against the corresponding trip-level records.

**Important aggregation detail:** `Mean_Events_Per_Recorded_Minute` and `Mean_Hard_Events_Per_Recorded_Minute` are unweighted means of trip-level rates. They are not calculated as total vehicle events divided by total vehicle recording duration. This definition must be preserved when reproducing the pipeline.

Undefined ratios may be blank when their denominator is zero. Such values are intentional where applicable. Infinite values and unexpected missing values are not accepted by the Phase 10.1 validation.

## 6. Validation Status

The reviewed Phase 10.1 run reported:

- 54,353 source summary segment rows.
- 2,571 trips.
- 320 vehicles.
- 25,744 events reconciled across the source summary, event audit, trip features, episode totals, and vehicle features.
- 18,696 episodes reconciled across the feature outputs.
- 11,796 transitions.
- All reported reconciliation checks passed.
- Formula checks for event totals, acceleration/braking totals, event shares, episode totals, transition totals, event rates, and coverage ratios reported zero mismatches.

The reviewed trip-level and vehicle-level feature files also passed their additional consistency checks.

Re-run the validation script after changing any input or output file.

## 7. Data Interpretation and Limitations

- An event count alone is not a complete measure of driver risk.
- A trip with zero recorded events does not prove that the trip was entirely safe.
- Recording duration and sample coverage can differ between trips. Comparisons should account for exposure and data coverage.
- Vehicle-level and trip-level records are related. They must not be randomly split in a way that allows information from the same vehicle or trip to leak across training and test partitions.
- The current feature tables do not, by themselves, establish a validated supervised learning target.
- No accuracy, precision, recall, F1 score, or real-world detection capability is claimed at this stage.

## 8. Planned Next Phases

1. **Dataset readiness:** verify source provenance, field meanings, units, event thresholds, and missing-data policies.
2. **Target definition:** define precisely what the model should predict and how reliable ground-truth labels will be obtained.
3. **Leakage-safe splitting:** create training, validation, and independent test partitions grouped by vehicle and, where appropriate, trip or time.
4. **Baseline modelling:** train and evaluate simple baseline models before introducing more complex architectures.
5. **Evaluation:** report per-class precision, recall, F1, confusion matrix, and performance on genuinely unseen vehicles or trips.
6. **Robustness assessment:** evaluate recording failures, different operating conditions, false alarms, and deployment constraints.

A phase should only be marked complete after its inputs, outputs, and validation evidence have been reviewed.

## 9. Reproducibility

The feature engineering script included in this repository is:

`Phase10_1_Feature_Engineering.py`

Use a compatible Python environment and the script's documented input/output expectations to reproduce Phase 10.1.

For each new run, record:

- Python and dependency versions.
- Input file hashes.
- Output file hashes.
- Row counts and key aggregate totals.
- Validation results.

Do not overwrite validated source data without keeping a recoverable version and recording the change.

## 10. Data Provenance and Licensing

The complete upstream dataset provenance, collection protocol, and formal license are not yet documented here. These details should be added only after they are confirmed.

Public repository visibility does not automatically mean that all included data is unrestricted for reuse.

Contributions should explain the reason for the change, the input files and processing steps, before-and-after totals, validation results, and any changes to field definitions or event thresholds.

---

**Project status:** Phase 9.2 event, episode, and transition outputs and Phase 10.1 trip and vehicle feature outputs have passed the checks described above. The next milestone is a documented, leakage-safe training dataset and a clearly defined prediction target—not a claim that the AI model is already complete.
