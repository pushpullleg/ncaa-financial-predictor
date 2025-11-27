# Appendix C — Deployment & Testing

## CLI Prediction Script

**Location:** `assets/scripts/predict.py`

### Usage
```bash
# From project root, using the virtual environment:
.venv/bin/python final/assets/scripts/predict.py "University Name"

# Or from the final folder:
cd final
../.venv/bin/python assets/scripts/predict.py "Alabama"
```

### Latest Test Run (Nov 27, 2025)
```
Input: Alabama
Output:
--- Prediction for Alabama A & M University (2022) ---
Predicted Trajectory: Stable
Confidence: Declining 0.01% | Stable 99.98% | Improving 0.02%
Exit Code: 0
```

---

## Regression Test

**Location:** `assets/scripts/test_predict.py`

Executes the CLI with a default query ("Alabama") and asserts a valid prediction block is returned.

### Running the Test
```bash
# From project root:
.venv/bin/python final/assets/scripts/test_predict.py

# Expected output:
{
  "query": "Alabama",
  "returncode": 0,
  "stdout": "--- Prediction for Alabama A & M University (2022) ---\nPredicted Trajectory: Stable\nConfidence: Declining 0.01% | Stable 99.98% | Improving 0.02%"
}
```

---

## Model Artifacts

| File | Description |
| :--- | :--- |
| `assets/models/trajectory_model.joblib` | XGBoost + SMOTE pipeline (2.8 MB) |
| `assets/data/trajectory_excellent.csv` | Final dataset (10,332 × 52) |

---

## Environment Requirements

**Important:** The model uses `imbalanced-learn` (SMOTE) which requires matching sklearn versions.

1. Use the project virtual environment:
   ```bash
   source .venv/bin/activate
   # Or directly: .venv/bin/python
   ```

2. Key dependencies (installed in .venv):
   - pandas==2.3.3
   - scikit-learn==1.7.2
   - xgboost==3.1.2
   - imbalanced-learn==0.14.0
   - joblib==1.5.2

3. Verify paths from project root:
   - Model: `final/assets/models/trajectory_model.joblib`
   - Data: `final/assets/data/trajectory_excellent.csv`
