# Analysis and Improvements

This folder contains comprehensive analysis and improvement notebooks for the NCAA Financial Trajectory Classification project.

## Notebooks Overview

### 01_Temporal_Structure_Verification.ipynb
**Purpose**: Verify that the model correctly uses temporal structure (predicting future from past) and identify any potential data leakage issues.

**Key Checks**:
- Verifies features from year `t` predict trajectory in year `t+1`
- Checks for data leakage (using same-year data to predict same-year outcome)
- Validates feature engineering calculations
- Confirms temporal gap between features and target

### 02_Baseline_Models.ipynb
**Purpose**: Implement baseline models to establish a performance floor and demonstrate that ML models add value over simple heuristics.

**Baseline Strategies**:
1. **Random Guess**: 33.3% accuracy (3 classes)
2. **Majority Class**: Predict most common class
3. **Persistence Model**: Predict that future trajectory = current trajectory
4. **Simple Rule-Based**: Use simple thresholds on key features

**Output**: Comparison of baseline models vs ML model performance

### 03_Class_Imbalance_Analysis.ipynb
**Purpose**: Analyze class imbalance issues, especially for the "Improving" class, and implement solutions to improve model performance on minority classes.

**Techniques Tested**:
- SMOTE (Synthetic Minority Over-sampling)
- ADASYN (Adaptive Synthetic Sampling)
- Random Over-sampling
- SMOTE-Tomek (Combined over/under sampling)
- Class Weights (XGBoost built-in)

**Output**: Comparison of different balancing techniques and recommendations

### 04_Feature_Importance_Analysis.ipynb
**Purpose**: Extract and analyze feature importance from the saved model to identify:
- Which features are most predictive
- Which features might be redundant
- Opportunities for feature selection
- Domain insights about what drives financial trajectories

**Output**: 
- Top features visualization
- Feature importance by category
- Feature selection recommendations

### 05_Comprehensive_Evaluation.ipynb
**Purpose**: Create a comprehensive evaluation comparing:
- Baseline models vs ML models
- Different balancing techniques
- Feature importance insights
- Overall model performance assessment

**Output**: Final summary with recommendations

## How to Use

1. **Start with Temporal Verification** (01): Ensure the model structure is correct
2. **Establish Baselines** (02): Understand what performance floor to beat
3. **Address Class Imbalance** (03): Improve performance on minority classes
4. **Analyze Features** (04): Understand what drives predictions
5. **Comprehensive Evaluation** (05): Get final summary and recommendations

## Key Findings

### Current Model Performance
- **Accuracy**: ~52% (beats random guess of 33%)
- **ROC-AUC**: 0.70 (moderate performance)
- **Macro F1**: ~0.47 (room for improvement)

### Main Issues Identified
1. **Class Imbalance**: "Improving" class is minority (~14% of data)
2. **Low Precision on Minority Class**: "Improving" class has precision ~0.26
3. **Feature Selection**: Some features may be redundant

### Recommendations
1. Use class balancing techniques (SMOTE or class weights)
2. Consider feature selection to reduce overfitting
3. Test different hyperparameters
4. Validate temporal structure to ensure no data leakage
5. Consider ensemble methods for better performance

## Files Generated

When you run these notebooks, they will generate:
- `baseline_comparison.png` - Baseline models comparison chart
- `ml_vs_baselines.png` - ML model vs baselines comparison
- `class_distribution.png` - Class distribution visualization
- `balancing_comparison.png` - Class balancing techniques comparison
- `feature_importance_top20.png` - Top 20 features visualization
- `feature_importance_by_category.png` - Features by category
- `cumulative_feature_importance.png` - Cumulative importance chart
- `confusion_matrix_ml.png` - ML model confusion matrix
- `comprehensive_comparison.png` - Final comprehensive comparison

## Next Steps

After running these analyses:
1. Review findings in each notebook
2. Implement recommended improvements
3. Retrain models with best techniques
4. Compare improved models with original
5. Update final model if improvements are significant

