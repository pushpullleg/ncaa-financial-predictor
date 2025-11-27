# CSCI 538 Final Project Report

**Title:** Forecasting NCAA Athletic Department Financial Trajectories
**Authors:** _TODO: Insert team members_
**Date:** November 27, 2025

---

## Abstract
We built a single, end-to-end pipeline that predicts whether an NCAA athletic department is on an **Improving, Stable, or Declining** two-year financial trajectory. Using a 10,332-row / 54-feature dataset derived from NCAA EADA filings, the final XGBoost + SMOTE model reaches **0.864 accuracy**, **0.965 ROC-AUC**, **0.817 macro F1**, and **0.647 Improving-class F1**. Temporal splits, feature explainability (SHAP), and a tested CLI make the solution ready for course submission and stakeholder demos. _TODO: keep abstract ≤10 lines once final wording is set._

## 1 Introduction and Background
**Aim:** give athletic administrators a reliable warning system so they know if finances are trending up, flat, or down before drastic cuts occur.

### 1.1 Problem Statement
Forecast the next two-year financial trajectory category for each institution using only historical (t-2 to t) structural, participation, and financial signals. This avoids circular metrics (e.g., predicting efficiency from its components) and enforces temporal causality.

### 1.2 Results from the Literature
Planned citations cover (i) NCAA revenue/expense landscape reports, (ii) prior ML work on education/finance forecasting, and (iii) class-imbalance / interpretability techniques (SMOTE, SHAP). _TODO: add IEEE-style references with short relevance notes._

### 1.3 Existing Tools and Programs
Current dashboards provide descriptive ratios but no forward-looking classification. Our contribution is a reproducible, notebook-driven pipeline that couples engineered lag features with explainable ML and a prediction CLI so non-technical stakeholders can run scenarios. _TODO: mention specific NCAA/DoE reports we build upon._

## 2 Overview of the Architecture
The project has four running components:
1. **Data engineering notebooks** (`archive/today/notebooks/04` & `08`): build the advanced/excellent datasets and enforce ≥10k rows / ≥10 features.
2. **Model training/evaluation notebooks** (`archive/analysis_and_improvements/05` & `09`): train XGBoost + Random Forest with SMOTE, log metrics, and export joblib artifacts.
3. **Documentation bundle** (files in this `final/` folder): concise descriptions, metrics, and references.
4. **Deployment scripts** (`final/assets/scripts/predict_trajectory.py` + regression test) that load the latest model/dataset combo automatically.

### 2.1 Finished Work
- Excellent dataset + artifacts regenerated on Nov 26–27 (see `evidence/dataset_overview.md`).
- Final XGBoost model saved at `final/assets/models/final_trajectory_model.joblib`.
- CLI + automated test verified (Appendix C / `evidence/deployment_testing_notes.md`).
- Reports updated: `evidence/baseline_results.md`, `evidence/excellent_results.md`, `evidence/optimization_notes.md`.

### 2.2 Work in Progress
- Wire the CLI regression script into CI.
- Automate annual EADA refresh jobs.
- Optional lightweight dashboard for administrators.

### 2.3 Future Work
- Enrich features with endowment and win/loss data.
- Experiment with sequence-aware models (e.g., temporal CNN/LSTM) once longer histories are curated.
- Formalize fairness monitoring across divisions and states.

## 3 Data Collection
- **Source:** NCAA EADA (2014–2023) for 1,722 institutions.
- **Processing:** filter for institutions with ≥5 consecutive years, align divisions, engineer forward trajectory labels, and compute lagged growth/ratio features.
- **Deliverable:** `final/assets/data/trajectory_ml_ready_excellent.csv` (10,332 rows / 54 columns) documented in `evidence/dataset_overview.md`.
- **Train/val/test strategy:** rolling windows so years t-2..t predict t+1..t+2; no same-year leakage. _TODO: cite the official EADA portal and include annotation details._

## 4 Methods and Implementation
1. **Preprocessing:** pandas + scikit-learn pipelines standardize numeric ranges, one-hot encode categorical fields, and apply SMOTE within each training fold to balance classes.
2. **Models:** Random Forest baseline for interpretability, XGBoost for best accuracy; hyperparameters tuned via randomized search (depth, learning rate, estimators, subsample, colsample).
3. **Evaluation:** accuracy, macro F1, class-specific precision/recall, ROC-AUC, and SHAP explanations (global + per-class).
4. **Deployment:** joblib artifacts referenced by `final/assets/scripts/predict_trajectory.py`, with regression test ensuring CLI output remains stable.

## 5 Results and Evaluation
| Model | Dataset | Accuracy | ROC-AUC | Macro F1 | Improving F1 |
| --- | --- | --- | --- | --- | --- |
| Persistence baseline | historical label only | 0.700 | — | 0.467 | 0.412 |
| XGBoost (advanced) | `archive/today/trajectory_ml_ready_advanced.csv` | 0.554 | 0.765 | 0.516 | 0.433 |
| **XGBoost (excellent)** | `final/assets/data/trajectory_ml_ready_excellent.csv` | **0.864** | **0.965** | **0.817** | **0.647** |

Supporting artifacts: confusion matrices, ROC curves, and SHAP plots cataloged in `supplemental_figures.md`. _TODO: embed figure/table references per template guidelines._

## 6 Achievements and Observations
- Dataset now exceeds coursework thresholds and captures realistic momentum signals.
- Final model beats the heuristic baseline by >16 accuracy points and raises macro F1 above 0.80.
- CLI + regression test prove the solution can be demonstrated quickly.
- SHAP highlights expense/revenue momentum, division, and participation trends as top drivers, matching domain expectations.
- _Appendix A will enumerate individual contributions once finalized._

## 7 Discussion and Conclusions
The solution delivers a forecasting tool that is accurate, interpretable, and reproducible. Remaining gaps relate to automating data refresh, incorporating exogenous signals (endowment, wins/losses), and embedding the regression test into CI so classroom demos stay reliable.

## 8 References
_TODO: Replace with IEEE-formatted citations (NCAA EADA portal, NCAA finance studies, SMOTE, SHAP, scikit-learn, XGBoost, etc.)._

---

## Appendices (Stubs)
- **Appendix A:** Individual contribution narratives (`contributions_outline.md`).
- **Appendix B:** Supplemental figures/tables (`supplemental_figures.md`).
- **Appendix C:** CLI regression test logs (`deployment_testing_notes.md`).
