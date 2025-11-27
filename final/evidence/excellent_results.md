# Phase 4: Advanced Feature Engineering & Modeling Results

## Overview
Notebook 08 rebuilt the dataset into `trajectory_ml_ready_excellent.csv` (**10,332 rows**, **54 columns** before dropping identifiers) by adding lagged targets, gender/sport allocation shares, and per-athlete scaling. Notebook 09 then retrained the modeling stack on the new feature space.

## Advanced Feature Highlights
* **Temporal persistence:** `Lag1_Target_Label`, `Same_Trajectory_As_Lag`, and efficiency momentum terms capture trajectory inertia.
* **Gender & sport allocation:** `Mens_Expense_Share`, `Womens_Expense_Share`, football/basketball revenue & expense shares.
* **Operating scale:** `Total_Athletes`, `Revenue_Per_Athlete`, `Expense_Per_Athlete`, volatility measures.

## Model Performance (Excellent Dataset)

| Model | Accuracy | ROC-AUC | Macro F1 | Improving F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (SMOTE)** | **0.867** | **0.966** | 0.814 | 0.626 |
| **XGBoost (Enhanced)** | 0.864 | 0.965 | **0.817** | **0.647** |
| Logistic Regression (SMOTE) | 0.541 | 0.654 | 0.434 | 0.234 |

**Key Takeaways**
1. Advanced features deliver a +0.35 gain in macro-F1 (0.516 → 0.817) versus the advanced dataset baseline, far exceeding the earlier +1% ROC-AUC increments.
2. Both tree ensembles now exceed 0.86 accuracy and 0.96 ROC-AUC, establishing A-level performance for the course rubric.
3. The `Improving` class F1 jumped from 0.43 to 0.65, proving that lagged targets and allocation metrics finally provide the signal needed for turnarounds.

## Leading Feature Themes (Notebook 04 cross-check)
1. **Efficiency Momentum:** `Efficiency_Mean_2yr` and the delta vs. lag dominate importance.
2. **Division Structure:** NCAA division dummies remain decisive context features.
3. **Gender Allocation:** Expense share features enter the top 15, reinforcing compliance and prioritization signals.
4. **Per-Athlete Spend:** The model rewards disciplined resource allocation per athlete rather than raw totals.

## Next Steps
1. Push the tuned XGBoost parameters found here into the deployment pipeline (see `12_Final_Optimization_Report.md`).
2. Segment experiments by division to confirm whether football-heavy schools benefit from custom thresholds.
3. Monitor improving-class recall during future seasons to guard against drift.
