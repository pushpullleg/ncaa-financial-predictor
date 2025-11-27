# Analysis & Improvements – Execution Checklist

Run these steps in order inside your own Python environment (>=3.10). Each step assumes the repo root is your working directory.

1. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost
   ```

2. **Rebuild enhanced dataset (optional if you already trust `trajectory_ml_ready_excellent.csv`)**
   - Open/run `analysis_and_improvements/08_Excellent_Dataset_Engineering.ipynb`
   - Output: `today/trajectory_ml_ready_excellent.csv`

3. **Train and evaluate A-grade models (preferred: run script)**
   ```bash
   python analysis_and_improvements/run_excellent_eval.py
   ```
   - Saves best model to `today/models/final_trajectory_model_excellent.joblib`
   - Logs accuracy, ROC-AUC, Macro F1, Improving F1 for each model

4. **(Optional) Inspect results in notebook form**
   - Open `analysis_and_improvements/09_Excellent_Models.ipynb`
   - Re-run to reproduce console outputs/plots

5. **Update summary/report with actual metrics**
   - Edit `analysis_and_improvements/SUMMARY_REPORT.md` “Performance Targets” section with the new numbers to confirm the A-grade thresholds are met.

6. **(Optional) Re-run interpretability / prediction tests**
   - `analysis_and_improvements/04_Feature_Importance_Analysis.ipynb`
   - `analysis_and_improvements/07_Test_Prediction_Script.ipynb`

7. **Share artifacts**
   - Export final confusion matrices/plots from `09_Excellent_Models.ipynb`
   - Include updated `SUMMARY_REPORT.md` in your submission

That’s it—each notebook/script is self-contained, so running them sequentially will reproduce the full “Excellent” pipeline in any environment with the listed dependencies installed.***

