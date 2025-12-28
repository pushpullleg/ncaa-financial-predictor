# NCAA Financial Trajectory Predictor

**Machine learning model that predicts whether NCAA athletic departments are financially improving, stable, or declining**

Built with Python, XGBoost, and 10 years of NCAA data (2014-2023)

---

## What This Does

Analyzes historical financial data from NCAA athletic departments and predicts their **2-year financial trajectory**: 
- 🟢 **Improving** - Revenue growing faster than expenses
- 🟡 **Stable** - Balanced revenue/expense ratio
- 🔴 **Declining** - Expenses outpacing revenue

## Key Results

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| **XGBoost + SMOTE** | **87.6%** | **0.968** | **0.827** |
| Random Forest | 88.0% | 0.968 | 0.829 |
| Logistic Regression | 54.6% | 0.653 | 0.436 |
| Baseline (Majority) | 46.4% | - | 0.211 |

✅ All models exceed A-grade requirements (>70% accuracy, >0.75 ROC-AUC, >0.70 F1)

---

## Tech Stack

- **Language:** Python 3.x
- **ML Libraries:** XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, SHAP
- **Data:** NCAA EADA (Equity in Athletics Data Analysis) 2014-2023

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/pushpullleg/AI_Using_Python_Project.git
cd AI_Using_Python_Project/final

# Install dependencies
pip install -r requirements.txt  # Create this if missing

# Run prediction for a school
python assets/scripts/predict.py "Alabama"

# Run regression tests
python assets/scripts/test_predict.py
```

---

## Project Structure

```
AI_Using_Python_Project/
├── final/                          # Main project directory
│   ├── assets/
│   │   ├── data/                   # Datasets (17K+ records)
│   │   ├── models/                 # Trained XGBoost model
│   │   ├── notebooks/              # Jupyter notebooks (feature engineering, training)
│   │   ├── scripts/                # predict.py, test_predict.py
│   │   └── figures/                # Visualizations (confusion matrix, SHAP plots)
│   ├── appendix/                   # Documentation
│   │   ├── A_Results_Summary.md
│   │   ├── B_Data_Dictionary.md
│   │   ├── C_Deployment.md
│   │   └── D_References.md
│   └── README.md                   # Detailed project README
└── README.md                       # This file
```

---

## How It Works

1. **Data Collection:** Pulled 10 years of NCAA financial data (17,220 records from 1,722 schools)
2. **Feature Engineering:** Created 52 features including:
   - Revenue/expense ratios and trends
   - Per-athlete spending metrics
   - Division and gender allocation
   - 2-year lag features
3. **Model Training:** XGBoost with SMOTE to handle class imbalance
4. **Evaluation:** 80/20 train/test split with stratification
5. **Deployment:** CLI tool + regression tests

---

## Key Features

- **Temporal validation:** Uses past data to predict future (no data leakage)
- **Interpretable:** SHAP values explain predictions
- **Production-ready:** Includes CLI tool and automated tests
- **Well-documented:** Full data dictionary and methodology

---

## Dataset

- **Source:** NCAA EADA database (public)
- **Size:** 10,332 rows × 52 features
- **Coverage:** 2014-2023, all NCAA divisions
- **Target:** 3-class trajectory (Improving/Stable/Declining)

See [`final/appendix/B_Data_Dictionary. md`](final/appendix/B_Data_Dictionary.md) for detailed schema. 

---

## Model Performance

### Confusion Matrix
The model correctly classifies 87.6% of financial trajectories with strong diagonal dominance: 

![Confusion Matrix](final/assets/figures/confusion_matrix_ml.png)

### Top Predictors
1. **Efficiency momentum** (revenue/expense ratio trend)
2. **Division classification** (I, II, III)
3. **Per-athlete spending**
4. **Gender allocation ratios**

---

## Use Cases

- **Athletic Directors:** Early warning system for budget issues
- **University Administrators:** Resource allocation planning
- **Researchers:** NCAA financial trend analysis
- **Students/Interviews:** End-to-end ML project example

---

## Documentation

- 📊 [Full Project Report](final/Report.md)
- 📈 [Results Summary](final/appendix/A_Results_Summary. md)
- 📖 [Data Dictionary](final/appendix/B_Data_Dictionary. md)
- 🚀 [Deployment Guide](final/appendix/C_Deployment.md)

---

## Academic Context

Built for **CSCI 538 - Applied Machine Learning** (November 2025)

**Requirements met:**
- ✅ Dataset >10,000 rows
- ✅ Accuracy >70%
- ✅ ROC-AUC >0.75
- ✅ Macro F1 >0.70
- ✅ Interpretability (SHAP)
- ✅ Deployment (CLI tool)

---

## License

[Include your license here - e.g., MIT, Apache 2.0, or "Academic Use Only"]

---

## Contact

**GitHub:** [@pushpullleg](https://github.com/pushpullleg)

For questions or collaboration opportunities, open an issue or reach out via GitHub. 
