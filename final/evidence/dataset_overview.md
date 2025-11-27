# Dataset Fact Sheet — NCAA Financial Trajectory Project

 **Canonical file:** `final/assets/data/trajectory_ml_ready_excellent.csv`
 **Generation source:** `archive/analysis_and_improvements/08_Excellent_Dataset_Engineering.ipynb`
  - Structural: Division, State, enrollment scale, sport offerings
  - Financial trend metrics: 1- and 2-year growth rates, CAGR, rolling means
  - Efficiency & ratios: revenue/expense ratios, surplus streaks, per-athlete allocations
  - Participation dynamics: total athletes, gender expenditure shares, recruiting intensity
  - Volatility indicators: rolling standard deviations for revenue/expense trajectories
 **Downstream consumers:** `archive/analysis_and_improvements/09_Excellent_Models.ipynb`, `archive/today/notebooks/09_Advanced_Models.ipynb`, and deployment script `final/assets/scripts/predict_trajectory.py`.

> The raw NCAA EADA extracts and intermediate parquet files remain in `archive/` for provenance while the submission-ready dataset now lives inside `final/assets/data/`.
