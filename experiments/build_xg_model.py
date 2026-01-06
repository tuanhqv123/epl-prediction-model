import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import pickle

print("BUILD OWN XG MODEL FROM RAW DATA")
print("-" * 50)

df = pd.read_csv("understat_data/epl_shots_all.csv")
print(f"Total shots: {len(df):,}")

# Target
df['is_goal'] = (df['result'] == 'Goal').astype(int)

# Features from raw data
df['X'] = df['X'].astype(float)
df['Y'] = df['Y'].astype(float)

# Distance to goal center (goal at X=1, Y=0.5)
df['distance'] = np.sqrt((1 - df['X'])**2 + (0.5 - df['Y'])**2)

# Angle to goal
df['angle'] = np.abs(np.arctan2(df['Y'] - 0.5, 1 - df['X']))

# Distance squared (non-linear effect)
df['distance_sq'] = df['distance'] ** 2

# Central position (shots from center are better)
df['central'] = 1 - np.abs(df['Y'] - 0.5) * 2

# One-hot encode categorical features
shot_type_dummies = pd.get_dummies(df['shotType'], prefix='type')
situation_dummies = pd.get_dummies(df['situation'], prefix='sit')

# Combine features
feature_cols = ['X', 'Y', 'distance', 'angle', 'distance_sq', 'central']
X = pd.concat([df[feature_cols], shot_type_dummies, situation_dummies], axis=1)
y = df['is_goal']

print(f"Features: {list(X.columns)}")

# Split by season (use 2024/25 as test)
train_mask = df['season'] < 2024
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"\nTrain: {len(X_train):,} shots (2014-2024)")
print(f"Test: {len(X_test):,} shots (2024/25)")

# Train model
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict_proba(X_train)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]

print(f"\nMODEL PERFORMANCE")
print("-" * 50)
print(f"Train Log Loss: {log_loss(y_train, train_pred):.4f}")
print(f"Test Log Loss: {log_loss(y_test, test_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, test_pred):.4f}")
print(f"Test Brier: {brier_score_loss(y_test, test_pred):.4f}")

# Compare with Understat xG
understat_xg_test = df.loc[~train_mask, 'xG'].astype(float)
print(f"\nCOMPARISON WITH UNDERSTAT XG")
print("-" * 50)
print(f"Our xG - Log Loss: {log_loss(y_test, test_pred):.4f}")
print(f"Understat xG - Log Loss: {log_loss(y_test, understat_xg_test):.4f}")
print(f"Our xG - AUC: {roc_auc_score(y_test, test_pred):.4f}")
print(f"Understat xG - AUC: {roc_auc_score(y_test, understat_xg_test):.4f}")

# Calibration check
print(f"\nCALIBRATION (Our xG)")
print("-" * 50)
df_test = df[~train_mask].copy()
df_test['our_xg'] = test_pred
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
df_test['xg_bin'] = pd.cut(df_test['our_xg'], bins=bins)
cal = df_test.groupby('xg_bin', observed=True).agg({
    'our_xg': 'mean',
    'is_goal': ['mean', 'count']
})
cal.columns = ['pred_xg', 'actual_goal_rate', 'count']
print(cal.to_string())

# Feature importance
print(f"\nFEATURE IMPORTANCE")
print("-" * 50)
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coef': model.coef_[0]
}).sort_values('coef', key=abs, ascending=False)
print(coef_df.head(15).to_string(index=False))

# Save model
model_data = {
    'model': model,
    'feature_cols': list(X.columns),
    'train_seasons': list(range(2014, 2024)),
    'test_season': 2024
}
with open('experiments/xg_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print(f"\nModel saved to experiments/xg_model.pkl")
