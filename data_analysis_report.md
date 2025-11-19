
# EPL PREDICTION MODEL - DATA ANALYSIS REPORT

## Dataset Overview
- Total matches: 3,800
- Target classes: H (Home Win), D (Draw), A (Away Win)
- Features available: 12 engineered features

## Issues Identified
- Data not chronological - potential data leakage

## Feature Statistics
- momentum: mean=0.276, std=0.555
- form: mean=0.455, std=0.258
- rest_days: mean=6.504, std=4.780
- matches_7d: mean=0.606, std=0.607
- impact: mean=0.061, std=0.856

## Recommended Next Steps
1. Install required ML packages (scikit-learn, pandas)
2. Implement proper preprocessing pipeline
3. Use temporal train-validation-test split
4. Train multiple models with cross-validation
5. Check for overfitting using validation set
6. Select best model and evaluate on test set

## Split Strategy
- Training: 70% (chronological)
- Validation: 15% (chronological)
- Testing: 15% (chronological)
