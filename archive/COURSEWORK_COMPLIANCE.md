# COURSEWORK COMPLIANCE - LSTM Experiment

## ⚠️ Important Note on Coursework Requirements

### Current LSTM Data Structure:
- **Sequences:** 6,888 (❌ Less than 10,000 required)
- **Features per timestep:** 6 (❌ Less than 10 required)
- **Time steps:** 5 years

### Why This Doesn't Meet Requirements (Yet):
The LSTM data is in a **3D format** `(6888, 5, 6)` which is necessary for neural networks, but doesn't directly satisfy the "10,000 rows × 10 features" requirement.

## ✅ SOLUTION: Two Approaches

### Approach 1: Flatten the Sequences (Recommended for Coursework)
We can "unroll" the time-series data into a flat format:

**Before (3D):** Each school has 1 sequence of 5 years  
**After (2D):** Each school-year becomes a separate row

This would give us:
- **Rows:** 15,498 (1,722 schools × 9 years) ✓ **Exceeds 10,000**
- **Features:** We can add more derived features to reach 10+

### Approach 2: Use the XGBoost Dataset (Already Compliant)
Your **main project** (`/ML EDA/today/`) already has:
- `trajectory_ml_ready_advanced.csv`: **12,054 rows × 27 features** ✓✓

This dataset **fully meets** the coursework requirements and is what your XGBoost model (ROC-AUC 0.70) was trained on.

## Recommendation for Coursework Submission

### Primary Submission (Meets All Requirements):
**Use the XGBoost project** in `/ML EDA/today/`:
- ✓ 12,054 rows
- ✓ 27 features  
- ✓ Complete pipeline (feature engineering → modeling → deployment)
- ✓ Final model saved and deployable
- ✓ Comprehensive documentation

### LSTM Experiment (Bonus/Advanced Section):
Present the LSTM work as an **"Advanced Exploration"** or **"Future Work"** section:
- Shows initiative and understanding of deep learning
- Demonstrates knowledge of time-series modeling
- Provides comparison between traditional ML and neural networks
- **Note:** Mention that the LSTM uses a different data structure (sequences) optimized for temporal patterns

## Final Verdict

**For Coursework:** Submit the XGBoost project (`/ML EDA/today/`) as your primary work.  
**For Bonus Points:** Include the LSTM experiment as supplementary material showing advanced techniques.

The XGBoost project is production-ready, meets all requirements, and demonstrates a complete ML workflow from data to deployment.
