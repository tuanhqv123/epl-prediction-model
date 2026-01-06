import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

print("BUILD XG MODEL FROM RAW DATA ONLY")
print("-" * 60)

# Load raw shot data
df = pd.read_csv("understat_data/epl_shots_all.csv")
print(f"Total shots: {len(df):,}")

# Check raw columns available
print(f"\nRaw columns: {df.columns.tolist()}")

# What is truly RAW data (not model output)?
# RAW: X, Y, shotType, situation, result, minute, h_a, lastAction
# NOT RAW (model output): xG (this is Understat's xG model output)

print(f"\n1. RAW FEATURES AVAILABLE:")
print("-" * 60)
print(f"X (position): {df['X'].min():.3f} - {df['X'].max():.3f}")
print(f"Y (position): {df['Y'].min():.3f} - {df['Y'].max():.3f}")
print(f"shotType: {df['shotType'].unique().tolist()}")
print(f"situation: {df['situation'].unique().tolist()}")
print(f"result: {df['result'].unique().tolist()}")
print(f"lastAction: {df['lastAction'].dropna().unique()[:10].tolist()}...")
print(f"minute: {df['minute'].min()} - {df['minute'].max()}")
print(f"h_a (home/away): {df['h_a'].unique().tolist()}")

# Target: is_goal (from result - this is REAL outcome, not model)
df['is_goal'] = (df['result'] == 'Goal').astype(int)
print(f"\nGoal rate: {df['is_goal'].mean()*100:.2f}%")

print(f"\n2. CREATE FEATURES FROM RAW DATA")
print("-" * 60)

# Position features
df['X'] = df['X'].astype(float)
df['Y'] = df['Y'].astype(float)

# Distance to goal center (goal at X=1.0, Y=0.5)
# Note: Understat uses 0-1 scale where X=1 is the goal line
df['distance_to_goal'] = np.sqrt((1.0 - df['X'])**2 + (0.5 - df['Y'])**2)

# Angle to goal (narrower angle = harder shot)
# Using arctan to calculate angle
df['angle'] = np.abs(np.arctan2(df['Y'] - 0.5, 1.0 - df['X']))

# How central is the shot (0.5 = center, 0 or 1 = sides)
df['central_position'] = 1 - 2 * np.abs(df['Y'] - 0.5)

# Distance squared (non-linear effect - further = much harder)
df['distance_sq'] = df['distance_to_goal'] ** 2

# Is shot from inside box? (roughly X > 0.83 based on pitch dimensions)
df['inside_box'] = (df['X'] > 0.83).astype(int)

# Is shot from very close? (X > 0.94 = 6 yard box area)
df['very_close'] = (df['X'] > 0.94).astype(int)

print("Created features:")
print(f"  distance_to_goal: {df['distance_to_goal'].mean():.3f} (avg)")
print(f"  angle: {df['angle'].mean():.3f} (avg)")
print(f"  central_position: {df['central_position'].mean():.3f} (avg)")
print(f"  inside_box: {df['inside_box'].mean()*100:.1f}% of shots")
print(f"  very_close: {df['very_close'].mean()*100:.1f}% of shots")

print(f"\n3. ENCODE CATEGORICAL FEATURES")
print("-" * 60)

# Shot type
shot_type_dummies = pd.get_dummies(df['shotType'], prefix='shot')
print(f"Shot types: {list(shot_type_dummies.columns)}")

# Situation
situation_dummies = pd.get_dummies(df['situation'], prefix='sit')
print(f"Situations: {list(situation_dummies.columns)}")

# Home/Away
df['is_home'] = (df['h_a'] == 'h').astype(int)

print(f"\n4. BUILD XG MODEL")
print("-" * 60)

# Combine all features
numeric_features = ['X', 'Y', 'distance_to_goal', 'angle', 'central_position', 
                    'distance_sq', 'inside_box', 'very_close', 'is_home']
X = pd.concat([df[numeric_features], shot_type_dummies, situation_dummies], axis=1)
y = df['is_goal']

print(f"Total features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")

# Split by season (train on 2014-2023, test on 2024)
train_mask = df['season'] < 2024
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"\nTrain: {len(X_train):,} shots (seasons 2014-2023)")
print(f"Test: {len(X_test):,} shots (season 2024)")

# Train logistic regression
model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
model.fit(X_train, y_train)

# Predict
train_pred = model.predict_proba(X_train)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]

print(f"\n5. MODEL PERFORMANCE")
print("-" * 60)
print(f"Train Log Loss: {log_loss(y_train, train_pred):.4f}")
print(f"Test Log Loss: {log_loss(y_test, test_pred):.4f}")
print(f"Test AUC-ROC: {roc_auc_score(y_test, test_pred):.4f}")
print(f"Test Brier Score: {brier_score_loss(y_test, test_pred):.4f}")

print(f"\n6. CALIBRATION CHECK")
print("-" * 60)
print("(Does predicted xG match actual goal rate?)")

df_test = df[~train_mask].copy()
df_test['our_xG'] = test_pred

# Bin by predicted xG
bins = [0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
df_test['xG_bin'] = pd.cut(df_test['our_xG'], bins=bins)
calibration = df_test.groupby('xG_bin', observed=True).agg({
    'our_xG': 'mean',
    'is_goal': ['mean', 'count']
})
calibration.columns = ['predicted_xG', 'actual_goal_rate', 'count']
calibration['diff'] = calibration['predicted_xG'] - calibration['actual_goal_rate']
print(calibration.to_string())

print(f"\n7. FEATURE IMPORTANCE")
print("-" * 60)
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df.head(15).to_string(index=False))

print(f"\n8. VERIFY WITH EXAMPLES")
print("-" * 60)
# Penalty should have high xG
penalty_mask = df_test['situation'] == 'Penalty'
print(f"Penalty avg xG: {df_test.loc[penalty_mask, 'our_xG'].mean():.3f} (actual goal rate: {df_test.loc[penalty_mask, 'is_goal'].mean():.3f})")

# Header from corner should have lower xG
header_corner = (df_test['shotType'] == 'Head') & (df_test['situation'] == 'FromCorner')
print(f"Header from corner avg xG: {df_test.loc[header_corner, 'our_xG'].mean():.3f} (actual: {df_test.loc[header_corner, 'is_goal'].mean():.3f})")

# Close range shot
close_shot = df_test['X'] > 0.94
print(f"Very close shot avg xG: {df_test.loc[close_shot, 'our_xG'].mean():.3f} (actual: {df_test.loc[close_shot, 'is_goal'].mean():.3f})")

# Long range shot
long_shot = df_test['X'] < 0.75
print(f"Long range shot avg xG: {df_test.loc[long_shot, 'our_xG'].mean():.3f} (actual: {df_test.loc[long_shot, 'is_goal'].mean():.3f})")

# Save model
import pickle
model_data = {
    'model': model,
    'feature_cols': list(X.columns),
    'numeric_features': numeric_features
}
with open('experiments/our_xg_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print(f"\nModel saved to experiments/our_xg_model.pkl")
