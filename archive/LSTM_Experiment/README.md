# LSTM Experiment - Summary

## What We Built

We created a complete **Time-Series Deep Learning** experiment to compare against our XGBoost baseline (ROC-AUC: 0.70).

### Directory Structure
```
LSTM_Experiment/
├── 01_Data_Preparation.ipynb  ✓ Executed Successfully
├── 02_LSTM_Model.ipynb         (Ready to run with TensorFlow)
└── lstm_data.npz               ✓ Generated
```

## Data Preparation Results (01_Data_Preparation.ipynb)

**Successfully executed!** Here's what we found:

### Schools with 10 Years of Data
- **1,722 schools** have complete 10-year financial histories (2013-2022)
- This gives us a robust dataset for time-series analysis

### Sequence Structure
- **Input:** 5 years of financial history per school
- **Output:** Predict the 6th year's trajectory
- **Total Sequences Created:** ~8,600 training examples
- **Shape:** `(8600, 5, 6)` 
  - 8,600 sequences
  - 5 time steps (years)
  - 6 features per year

### Features Used
1. Revenue (normalized)
2. Expenses (normalized)
3. Efficiency Ratio (Revenue/Expenses)
4. Net Income
5. Revenue Growth (year-over-year)
6. Expense Growth (year-over-year)

## LSTM Model Architecture (02_LSTM_Model.ipynb)

The model is designed with:
- **2 LSTM layers** (64 → 32 units) to capture temporal patterns
- **Dropout layers** (30%) to prevent overfitting
- **Dense output layer** with softmax for 3-class classification

### Why This Might Beat XGBoost
1. **Temporal Memory:** LSTM can remember that "3 years of slow decline → crisis in year 4"
2. **Pattern Recognition:** Can detect complex sequences like "revenue spike followed by expense lag = improving"
3. **Non-linear Relationships:** Neural networks excel at finding hidden patterns

### Why XGBoost Might Still Win
1. **Limited Data:** 8,600 sequences is "small" for deep learning
2. **Simpler Patterns:** If trajectories are mostly determined by recent snapshots, XGBoost's feature engineering might be enough
3. **Overfitting Risk:** Neural networks need more data to generalize well

## Next Steps to Run the Experiment

### Option 1: Install TensorFlow Locally
```bash
pip install tensorflow
cd "/Users/mukeshravichandran/ML EDA/LSTM_Experiment"
jupyter notebook 02_LSTM_Model.ipynb
```

### Option 2: Run on Google Colab (Recommended)
1. Upload both notebooks to Google Colab
2. Upload `lstm_data.npz`
3. Run `02_LSTM_Model.ipynb` (Colab has TensorFlow pre-installed)
4. Training will take ~5-10 minutes on Colab's free GPU

### Option 3: Use the Prepared Data for Other Experiments
The `lstm_data.npz` file contains clean, sequenced data that can be used for:
- Other RNN architectures (GRU, Transformer)
- Ensemble methods combining LSTM + XGBoost
- Attention mechanisms

## Expected Outcome

Based on the data characteristics, I predict:
- **If LSTM wins:** ROC-AUC ~0.72-0.75 (modest improvement)
- **If XGBoost wins:** The patterns are too "snapshot-based" rather than truly sequential

Either way, this experiment demonstrates advanced ML techniques and provides valuable insights into whether financial trajectories have true temporal dependencies or are primarily driven by current-year metrics.

## Academic Value

This LSTM experiment adds significant depth to your project:
1. **Demonstrates understanding** of different ML paradigms (tree-based vs. neural networks)
2. **Shows initiative** in exploring cutting-edge techniques
3. **Provides comparison** between traditional ML and deep learning
4. **Addresses the question:** "Does history matter, or just the present?"
