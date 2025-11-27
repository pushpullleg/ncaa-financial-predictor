# ML Modeling Plan - NCAA EADA Surplus Forecasting

## Objective
Beat the **85.71% persistence baseline** by forecasting next year's athletic department surplus using current characteristics and lagged financial indicators.

---

## 📊 Data Preparation (DONE ✅)
- ✅ Dataset: `eada_ml_ready_temporal_forecasting.csv` (13,776 records)
- ✅ Features: 21 predictive features (lag, structural, growth)
- ✅ Target: `Target_Next_Year_Surplus` (binary)
- ✅ Splits: Train (2015-2019), Val (2020-2021), Test (2022-2023)

---

## 🤖 Models to Build

### 1. Baseline Models (Already Analyzed ✅)
- **Majority Class:** 62.14% accuracy
- **Persistence:** 85.71% accuracy ⭐ (THIS IS WHAT WE MUST BEAT)
- **Simple Rule:** 74.22% accuracy

### 2. Traditional ML Models
**Model 1: Logistic Regression**
- Purpose: Interpretable linear baseline
- Expected: 82-86% (may not beat persistence)
- Benefit: Feature importance via coefficients

**Model 2: Random Forest**
- Purpose: Non-linear patterns, feature importance
- Expected: 87-89% accuracy
- Benefit: Built-in feature importance

**Model 3: XGBoost** ⭐ (PRIMARY MODEL)
- Purpose: State-of-the-art gradient boosting
- Expected: 88-91% accuracy (GOAL: >90%)
- Benefit: SHAP values for interpretability

### 3. Advanced Approach (Optional)
**Ensemble Model:**
- Combine: Logistic + RF + XGBoost
- Stacking or weighted voting
- Expected: 90-92% accuracy

---

## 📈 Evaluation Metrics

### Primary Metrics:
1. **Accuracy:** Must beat 85.71%
2. **ROC-AUC:** Target >0.75 (good discrimination)
3. **F1-Score:** Target >0.85 (balanced performance)

### Secondary Metrics:
4. **Precision:** How many predicted surpluses are correct?
5. **Recall:** How many actual surpluses did we catch?
6. **Confusion Matrix:** Where are we making mistakes?

### Business Metrics:
7. **Cost-Benefit Analysis:** What's the cost of false positives/negatives?
8. **Early Warning:** Can we identify at-risk departments 1 year ahead?

---

## 🔬 Model Building Steps

### Step 1: Feature Selection
```python
# Already have 21 features, but check:
- Remove highly correlated features (>0.95)
- Check for multicollinearity (VIF)
- Test feature subsets (all vs top 10)
```

### Step 2: Handle Class Imbalance
```python
# Current: 60.5% No Surplus, 39.5% Has Surplus
# Imbalance ratio: 1.53:1 (MILD - no special handling needed)
# But can try:
- Class weights in XGBoost
- SMOTE (if needed)
```

### Step 3: Hyperparameter Tuning
```python
# XGBoost parameters to tune:
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [100, 200, 500]
- min_child_weight: [1, 3, 5]
- subsample: [0.7, 0.8, 1.0]

# Use validation set (2020-2021) for tuning
```

### Step 4: Training
```python
# Train on 2015-2019 data (8,610 records)
# Validate on 2020-2021 data (3,444 records)
# Final test on 2022-2023 data (1,722 records)
```

### Step 5: Evaluation
```python
# Compare to baseline:
if test_accuracy > 85.71 + 0.05:  # 5 point improvement
    print("✅ ML justified!")
else:
    print("❌ Stick with persistence baseline")
```

---

## 🎯 Expected Results

### Conservative Estimate:
- **Accuracy:** 87-88% (2-3 point improvement over baseline)
- **ROC-AUC:** 0.78-0.82
- **F1-Score:** 0.85-0.87

### Optimistic Estimate:
- **Accuracy:** 89-91% (4-6 point improvement)
- **ROC-AUC:** 0.83-0.88
- **F1-Score:** 0.88-0.91

### Key Insight:
**Lag1_Has_Surplus (0.69 correlation)** is so strong that the model might just learn:
```
"If surplus last year → predict surplus next year"
"If no surplus last year → predict no surplus next year"
```

**This is OK!** It's still useful for:
1. Institutions with no historical data (new programs)
2. Identifying when persistence breaks down
3. Understanding which OTHER factors matter (SHAP analysis)

---

## 📊 Interpretability (CRITICAL)

### SHAP Analysis:
1. **Global importance:** Which features matter most across all predictions?
2. **Local explanations:** Why did model predict surplus for Institution X?
3. **Feature interactions:** How do Division + Enrollment interact?

### Expected SHAP Insights:
- **Lag1_Has_Surplus:** Will dominate (explains 48% variance)
- **Division:** Independent schools → higher surplus probability
- **Lag1_Efficiency_Ratio:** Trending up → higher surplus probability
- **Total_Athletes:** Larger programs → more stable (higher surplus)

---

## ⚠️ Addressing the 60% Exactly 1.0 Issue

### Option A: Exclude Exact 1.0 Values
```python
# Only model institutions with "real" variation
df_clean = df[df['Efficiency_Ratio'] != 1.0]
# Reduces dataset to ~40% (5,500 records)
# BUT: More realistic financial behavior
```

### Option B: Two-Stage Model
```python
# Stage 1: Predict "Does institution manipulate books?" (1.0 vs not 1.0)
# Stage 2: For non-manipulators, predict surplus

# This actually has business value!
# "Which institutions cook the books?"
```

### Option C: Ignore It (CURRENT APPROACH)
```python
# Model all data, acknowledge limitation in discussion
# "60% report exactly 1.0 due to institutional support adjustments"
# "Model predicts based on available data, including manipulated values"
```

**Recommendation:** Start with Option C, discuss Options A/B as "future work"

---

## 🚀 Implementation Timeline

### Week 1: Model Building
- [ ] Day 1-2: Logistic Regression + Random Forest
- [ ] Day 3-4: XGBoost with hyperparameter tuning
- [ ] Day 5: Ensemble model (if time permits)

### Week 2: Evaluation & Interpretation
- [ ] Day 1-2: SHAP analysis, feature importance
- [ ] Day 3-4: Error analysis, confusion matrix
- [ ] Day 5: Business recommendations

### Week 3: Report Writing
- [ ] Day 1-2: Methods section
- [ ] Day 3-4: Results section
- [ ] Day 5: Discussion & conclusions

---

## 📝 Success Criteria

### ✅ MINIMUM (Pass):
- Build at least 2 models (Logistic + XGBoost)
- Beat persistence baseline by 2+ points (>87.71%)
- Provide SHAP analysis showing feature importance

### ✅ GOOD (B Grade):
- Build 3+ models with proper comparison
- Beat persistence baseline by 5+ points (>90.71%)
- ROC-AUC > 0.80
- Business-focused interpretation

### ✅ EXCELLENT (A Grade):
- Build ensemble model
- Beat persistence by 5-7 points
- ROC-AUC > 0.85
- Comprehensive SHAP analysis
- Address the "exactly 1.0" data quality issue
- Provide actionable business recommendations

---

## 💡 Key Insights to Communicate

1. **"Persistence is a strong baseline (85.71%)"**
   - Most institutions behave consistently year-over-year
   - ML adds value by identifying exceptions to this rule

2. **"Division matters more than you'd think"**
   - Independent schools: 66.7% surplus rate
   - NCCAA D1: 3.9% surplus rate
   - 62.8 percentage point spread!

3. **"Data quality is a major limitation"**
   - 60% report exactly 1.0 (accounting manipulation)
   - Real financial variation may be masked
   - Results should be interpreted cautiously

4. **"ML is justified because of temporal forecasting"**
   - We're predicting FUTURE (year t+1) using CURRENT data (year t)
   - Not circular reasoning like original approach
   - Enables proactive budget planning

---

## 📚 Next Steps After Modeling

1. **Write up results** in a professional report
2. **Create presentation** for professor/stakeholders
3. **Deploy model** (if approved) for 2024 predictions
4. **Monitor performance** on 2024 actual data
5. **Iterate** based on feedback

---

**Status:** Ready to proceed with modeling!
**Data:** ✅ Clean and prepared
**Baseline:** ✅ Established (85.71%)
**Target:** ✅ Defined (>90% accuracy)
**Tools:** Python, scikit-learn, XGBoost, SHAP

**Let's build some models! 🚀**
