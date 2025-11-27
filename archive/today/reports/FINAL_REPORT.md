# NCAA Financial Trajectory Classification - Final Project Report

## 1. Executive Summary
This project developed a machine learning model to predict the future financial trajectory ("Improving", "Stable", or "Declining") of NCAA athletic departments. After re-running the full analysis stack, the excellent dataset now contains **10,332 lag-enhanced rows** spanning 2017-2022 with 50+ engineered features.

**Key Result:** The final XGBoost model trained on the excellent dataset achieves **86.4% accuracy, 0.965 ROC-AUC, 0.817 macro F1, and 0.647 Improving-class F1**, comfortably exceeding the A-level targets.

## 2. Problem Statement
NCAA athletic departments face significant financial volatility. Predicting whether a program is on a path to financial distress ("Declining") or growth ("Improving") allows for proactive management.

*   **Objective:** Classify schools into three trajectories based on future revenue/expense trends.
*   **Target Variable:**
    *   **Improving:** Revenue Growth > 3% AND Expenses growing slower than Revenue.
    *   **Declining:** Revenue shrinking OR Expenses growing >3% faster than Revenue.
    *   **Stable:** Status quo.

## 3. Data Processing & Feature Engineering
We processed the raw EADA dataset to meet the coursework requirement of >10,000 rows and >10 features.

### Key Features Created
1.  **Financial Trends:** 2-year CAGR and 1-year growth rates for Revenue and Expenses.
2.  **Efficiency Metrics:** `Efficiency_Mean_2yr` (Revenue / Expenses ratio).
3.  **Volatility:** Standard deviation of financial metrics over a rolling window.
4.  **Advanced "Athletic" Features:**
    *   **Gender Balance:** `Mens_Expense_Share`, `Womens_Expense_Share`.
    *   **Sport Specifics:** `Football_Revenue_Share`, `Basketball_Revenue_Share`.
    *   **Operational:** `Revenue_Per_Athlete`, `Total_Athletes`.

## 4. Modeling Approach
We evaluated multiple algorithms (Logistic Regression, Random Forest, XGBoost) and addressed class imbalance using SMOTE.

### Model Evolution
| Phase | Dataset | Model | Accuracy | ROC-AUC | Macro F1 | Improving F1 | Key Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline ML** | `trajectory_ml_ready_advanced.csv` | XGBoost + SMOTE | 0.554 | 0.765 | 0.516 | 0.433 | Establishes lift over heuristics |
| **Excellent RF** | `trajectory_ml_ready_excellent.csv` | Random Forest + SMOTE | **0.867** | **0.966** | 0.814 | 0.626 | Highest accuracy, great per-class balance |
| **Final** | `trajectory_ml_ready_excellent.csv` | **XGBoost (Enhanced)** | 0.864 | 0.965 | **0.817** | **0.647** | Best macro/Improving F1; chosen for deployment |

## 5. Key Findings
1.  **Efficiency is Paramount:** The ratio of Revenue to Expenses is the single strongest predictor of future trajectory.
2.  **Context Matters:** The "Division" a school belongs to fundamentally changes its financial dynamics.
3.  **Gender Allocation Signal:** The proportion of budget allocated to men's vs. women's sports is a significant predictor, likely acting as a proxy for Title IX compliance status or institutional priorities.

## 6. Deployment
* **Model Artifacts:** `today/models/final_trajectory_model_excellent.joblib` (default) with `final_trajectory_model.joblib` retained for regression testing.
* **Dataset:** `today/trajectory_ml_ready_excellent.csv` regenerated via Notebook 08.
* **CLI:** `today/scripts/predict_trajectory.py` now auto-detects the excellent model/dataset paths. `today/scripts/test_predict_trajectory_cli.py` provides a quick regression test that is executed before shipping.

## 7. Future Work
*   **External Data:** Incorporate endowment data and team win/loss records.
*   **Text Analysis:** Analyze the text of financial footnotes for warning signs.
*   **Time-Series Deep Learning:** Use LSTM networks if longer historical sequences become available.
