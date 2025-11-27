# Phase 3: Initial Model Evaluation Summary

## Overview
Notebook 05 (`05_Comprehensive_Evaluation.ipynb`) re-ran the baseline-to-ML comparison using the refreshed `trajectory_ml_ready_advanced.csv` dataset (**12,054 rows**, **21 base features**) spanning 2016-2022. The goal was to quantify how much value the SMOTE-enabled XGBoost model adds over simple heuristics before moving to the excellent dataset.

## Model Performance (Advanced Dataset Split)

| Model | Accuracy | ROC-AUC | Macro F1 | Declining F1 | Stable F1 | Improving F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (SMOTE)** | **0.554** | **0.765** | **0.516** | 0.441 | 0.675 | 0.433 |
| Majority Class Baseline | 0.472 | – | 0.214 | 0.000 | 0.642 | 0.000 |
| Random Guess Baseline | 0.320 | – | 0.303 | 0.348 | 0.367 | 0.195 |

## Key Observations
1. **Clear Lift vs. Heuristics:** The ML pipeline adds +8.2 percentage points of accuracy and +0.20 macro-F1 over the best baseline, validating the modeling approach prior to advanced feature engineering.
2. **ROC-AUC Headroom:** 0.765 ROC-AUC indicates the advanced feature set already captures strong ranking signal even before the excellent dataset engineering pass.
3. **Minority Class Performance:** Improving-class F1 remains at 0.43, establishing a baseline to beat in the excellent modeling phase.

## Recommendations Heading Into Phase 4
1. Carry SMOTE forward; it remains the most reliable balancer from Notebook 03.
2. Focus on engineered efficiency/division features surfaced in Notebook 04 to push the improving-class recall.
3. Use these metrics as the checkpoint when reporting gains from the excellent dataset (see `10_Advanced_Model_Results.md`).
