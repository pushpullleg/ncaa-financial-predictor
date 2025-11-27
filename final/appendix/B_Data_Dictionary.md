# Appendix B — Data Dictionary

## Dataset Overview

| Property | Value |
| :--- | :--- |
| **File** | `assets/data/trajectory_excellent.csv` |
| **Rows** | 10,332 (meets ≥10,000 requirement) |
| **Features** | 52 columns (meets ≥10 requirement) |
| **Time Span** | 2017–2022 (derived from 2014–2023 EADA filings) |
| **Source Notebook** | `assets/notebooks/03_Feature_Engineering_Advanced.ipynb` |

---

## Data Pipeline

| Step | Input | Output | Rows |
| :--- | :--- | :--- | :--- |
| 1. Raw Data | NCAA EADA | `raw_eada_10yrs.csv` | 17,220 |
| 2. Basic Features | Raw data | `trajectory_advanced.csv` | 12,054 |
| 3. Lag Features | Advanced data | `trajectory_excellent.csv` | 10,332 |

---

## Feature Families

### Structural
- Division (NCAA I, II, III)
- State
- Enrollment scale
- Sport offerings count

### Financial Trends
- 1-year and 2-year growth rates (Revenue, Expenses)
- CAGR (Compound Annual Growth Rate)
- Rolling means

### Efficiency & Ratios
- Revenue/Expense ratio (`Efficiency_Mean_2yr`)
- Surplus streaks
- Per-athlete allocations

### Participation Dynamics
- Total athletes
- Gender expenditure shares (`Mens_Expense_Share`, `Womens_Expense_Share`)
- Recruiting intensity proxies

### Volatility Indicators
- Rolling standard deviations for revenue/expense trajectories

---

## Label Definition

**Target Variable:** `Trajectory_Label`

| Class | Definition |
| :--- | :--- |
| **Improving** | Revenue Growth > 3% AND Expenses growing slower than Revenue |
| **Declining** | Revenue shrinking OR Expenses growing >3% faster than Revenue |
| **Stable** | Neither improving nor declining |

Labels are computed from forward two-year revenue/expense deltas, ensuring no leakage.

---

## Downstream Consumers
- `assets/notebooks/04_Model_Training.ipynb`
- `assets/scripts/predict.py`
