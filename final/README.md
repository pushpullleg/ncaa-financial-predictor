# NCAA Financial Trajectory Classifier

## Problem Statement
Predict whether an NCAA athletic department is on an **Improving**, **Stable**, or **Declining** two-year financial trajectory so administrators can act before budgets spiral.

## Key Results

| Model | Accuracy | ROC-AUC | Macro F1 | Improving F1 |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Majority Class) | 0.464 | – | 0.211 | 0.000 |
| Logistic Regression + SMOTE | 0.546 | 0.653 | 0.436 | 0.234 |
| Random Forest + SMOTE | 0.880 | 0.968 | 0.829 | 0.655 |
| **Final XGBoost + SMOTE** | **0.876** | **0.968** | **0.827** | **0.658** |

All models exceed A-grade requirements (>70% accuracy, >0.75 ROC-AUC, >0.70 Macro F1, >0.50 Improving F1).

## Why It Matters
- **Early warning:** Finance offices see decline two budget cycles ahead
- **Sound ML:** Temporal splits, SHAP explanations, reproducible pipeline
- **Deployment ready:** CLI + regression test included

---

## Folder Structure

```
final/
├── README.md              # This file
├── Report.md / .docx      # Full project report
│
├── assets/
│   ├── data/
│   │   ├── raw_eada_10yrs.csv         # Raw EADA data (17,220 rows)
│   │   ├── trajectory_advanced.csv    # Basic features (12,054 rows)
│   │   └── trajectory_excellent.csv   # Full features (10,332 × 52)
│   │
│   ├── models/
│   │   └── trajectory_model.joblib    # Final XGBoost + SMOTE pipeline
│   │
│   ├── notebooks/
│   │   ├── 01_Feature_Engineering_Basic.ipynb
│   │   ├── 02_Model_Evaluation_Basic.ipynb
│   │   ├── 03_Feature_Engineering_Advanced.ipynb
│   │   └── 04_Model_Training.ipynb
│   │
│   ├── scripts/
│   │   ├── predict.py        # CLI prediction tool
│   │   └── test_predict.py   # Regression test
│   │
│   └── figures/
│       ├── confusion_matrix_ml.png
│       └── comprehensive_comparison.png
│
├── appendix/
│   ├── A_Results_Summary.md      # All model metrics
│   ├── B_Data_Dictionary.md      # Dataset documentation
│   ├── C_Deployment.md           # CLI usage & testing
│   ├── D_References.md           # IEEE-style citations
│   └── E_Team_Contributions.md   # Individual contributions
│
└── docs/
    ├── Project_Report_Template.pdf
    └── Grading_Rubric.pdf
```

---

## Quick Start

```bash
# Activate virtual environment (required for correct sklearn/imblearn versions)
source .venv/bin/activate

# Or use venv Python directly:
.venv/bin/python assets/scripts/predict.py "Alabama"

# Run regression test
.venv/bin/python assets/scripts/test_predict.py
```

---

## Appendices
- [A: Results Summary](appendix/A_Results_Summary.md)
- [B: Data Dictionary](appendix/B_Data_Dictionary.md)
- [C: Deployment Guide](appendix/C_Deployment.md)
- [D: References](appendix/D_References.md)
- [E: Team Contributions](appendix/E_Team_Contributions.md)
