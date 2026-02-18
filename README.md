# ncaa-financial-predictor

An XGBoost pipeline that predicts whether an NCAA athletic department's finances are improving, stable, or declining over a 2-year horizon. Trained on 10 years of public EADA data (2014–2023) across all divisions.

**Results:** 87.6% accuracy · 0.97 ROC-AUC · 0.83 F1

The model uses SMOTE to handle class imbalance and SHAP to explain individual predictions. A CLI tool lets you run a prediction for any school by name.

## Quick start

```bash
git clone https://github.com/pushpullleg/ncaa-financial-predictor.git
cd ncaa-financial-predictor/final
pip install -r requirements.txt

python assets/scripts/predict.py "Alabama"
```

## How it works

1. **Data** — 17,220 records from 1,722 schools, joined and cleaned from the NCAA EADA database
2. **Features** — 52 engineered features: revenue/expense ratios, per-athlete spending, division type, 2-year lag trends
3. **Training** — XGBoost with SMOTE oversampling, 80/20 stratified split
4. **Output** — Improving / Stable / Declining with SHAP-based explanation

## Structure

```
final/
├── assets/
│   ├── data/          # raw and processed datasets
│   ├── models/        # trained XGBoost model
│   ├── notebooks/     # feature engineering and training
│   ├── scripts/       # predict.py, test_predict.py
│   └── figures/       # confusion matrix, SHAP plots
└── appendix/          # data dictionary, methodology, deployment notes
```

## License

Academic use only
