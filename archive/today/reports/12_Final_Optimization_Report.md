# Phase 4: Final Optimization Report

## 1. Final Training Run Summary
Notebook 09 functioned as the optimization pass: SMOTE-balanced pipelines, tuned hyperparameters, and the excellent dataset were all re-run end-to-end. The key settings shipped to production are:

* **XGBoost (Enhanced)** – `n_estimators=400`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.9`, `min_child_weight=3` (matching the notebook configuration).
* **Random Forest (SMOTE)** – `n_estimators=600`, `max_depth=12`, `max_features='sqrt'`.

## 2. Performance Progression

| Stage | Dataset | Accuracy | ROC-AUC | Macro F1 | Improving F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline ML (Notebook 05) | `trajectory_ml_ready_advanced.csv` | 0.554 | 0.765 | 0.516 | 0.433 | XGBoost + SMOTE vs. heuristics |
| Excellent RF (Notebook 09) | `trajectory_ml_ready_excellent.csv` | **0.867** | **0.966** | 0.814 | 0.626 | Highest overall accuracy |
| Excellent XGB (Notebook 09) | `trajectory_ml_ready_excellent.csv` | 0.864 | 0.965 | **0.817** | **0.647** | Best macro/Improving F1 |

**Total Improvement:** +0.35 Macro-F1 and +0.212 Improving-F1 relative to the baseline ML checkpoint.

## 3. Final Model Selection
* **Primary model:** XGBoost (Enhanced) – selected for its superior macro-F1 and improving-class recall, which aligns with the project's emphasis on catching recoveries early.
* **Fallback:** Random Forest (SMOTE) – retained because it achieves the top-line accuracy target and offers interpretability via tree inspection.

## 4. Stakeholder-Ready Insights
1. **Efficiency momentum drives risk scores.** Schools improving their revenue-to-expense ratio year-over-year are rarely flagged as Declining.
2. **Division context remains critical.** Division dummies behave like structural priors and should remain in every downstream scorecard.
3. **Gender allocation is now actionably predictive.** Large imbalances in men vs. women spend correlate with elevated Declining risk, likely reflecting compliance pressures.
4. **Predictability ceiling shattered.** With ROC-AUC now at ~0.965, the earlier 0.70 ceiling is no longer the limiting factor; the next ceiling is minority-class recall beyond 0.65.

## 5. Recommended Actions
1. **Deploy both saved artifacts** (`models/final_trajectory_model_excellent.joblib` and `models/final_trajectory_model.joblib`) with the XGBoost model set as default.
2. **Monitor improving-class precision/recall** at least quarterly to catch drift as new seasons roll in.
3. **Automate CLI regression tests** (see `today/scripts/test_predict_trajectory_cli.py`) in CI before shipping any new model artifact.
4. **Collect qualitative signals** (donor commitments, coaching changes) to push improving-class F1 toward 0.70+ in the next iteration.
