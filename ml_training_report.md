# EPL PREDICTION MODEL - COMPREHENSIVE ML REPORT

## Model Training Results

### Best Model: Logistic Regression

- Test Accuracy: 0.5228
- Test Precision: 0.3945
- Test Recall: 0.5228
- Test F1 Score: 0.4458

### All Models Performance (Validation Set)

#### Logistic Regression

- Accuracy: 0.5316
- Precision: 0.4156
- Recall: 0.5316
- F1 Score: 0.4652

#### Decision Tree

- Accuracy: 0.4386
- Precision: 0.4197
- Recall: 0.4386
- F1 Score: 0.4244

#### Random Forest

- Accuracy: 0.5123
- Precision: 0.4679
- Recall: 0.5123
- F1 Score: 0.4666

#### KNN

- Accuracy: 0.4719
- Precision: 0.4438
- Recall: 0.4719
- F1 Score: 0.4511

### Overfitting Analysis

#### Logistic Regression

- Training Accuracy: 0.5447
- Validation Accuracy: 0.5316
- Overfitting Gap: 0.0132

#### Decision Tree

- Training Accuracy: 0.7387
- Validation Accuracy: 0.4386
- Overfitting Gap: 0.3001

#### Random Forest

- Training Accuracy: 0.8774
- Validation Accuracy: 0.5123
- Overfitting Gap: 0.3652

#### KNN

- Training Accuracy: 0.6207
- Validation Accuracy: 0.4719
- Overfitting Gap: 0.1487

### Feature Importancex

#### Logistic Regression

Top 10 Features:

1. strength_difference: 0.1905
2. home_team_strength: 0.1707
3. away_team_strength: 0.1327
4. away_recent_form: 0.0557
5. home_matches_last_14d: 0.0400
6. form_difference: 0.0396
7. away_last_result_impact: 0.0393
8. away_matches_last_7d: 0.0368
9. impact_difference: 0.0366
10. away_rest_days: 0.0342

#### Decision Tree

Top 10 Features:

1. strength_difference: 0.2967
2. momentum_difference: 0.1010
3. away_team_encoded: 0.0597
4. home_rest_days: 0.0496
5. home_team_encoded: 0.0484
6. away_team_strength: 0.0480
7. home_momentum_score: 0.0403
8. rest_days_difference: 0.0391
9. form_difference: 0.0373
10. away_rest_days: 0.0373

#### Random Forest

Top 10 Features:

1. strength_difference: 0.1633
2. home_team_strength: 0.0845
3. away_team_strength: 0.0749
4. momentum_difference: 0.0730
5. form_difference: 0.0601
6. away_momentum_score: 0.0571
7. home_momentum_score: 0.0541
8. home_team_encoded: 0.0496
9. away_team_encoded: 0.0471
10. away_recent_form: 0.0464

### Recommendations

1. Use the best model for production predictions
2. Monitor performance over time
3. Consider ensemble methods for improvement
4. Retrain periodically with new data

### Files Generated

- Trained models: model\_\*.pkl
- Results: ml_results.pkl
- This report: ml_training_report.md
