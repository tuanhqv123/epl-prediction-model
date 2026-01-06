import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

print("BUILD XG MODEL WITH LASTACTION FEATURE")
print("-" * 60)

df = pd.read_csv("understat_data/epl_shots_all.csv")
print(f"Total shots: {len(df):,}")

# Check lastAction
print(f"\nlastAction distribution:")
print(df['lastAction'].value_counts().head(15))

# Target
df['is_goal'] = (df['result'] == 'Goal').astype(int)

# Features
df['X'] = df['X'].astype(float)
df['Y'] = df['Y'].astype(float)
df['distance_to_goal'] = np.sqrt((1.0 - df['X'])**2 + (0.5 - df['Y'])**2)
df['angle'] = np.abs(np.arctan2(df['Y'] - 0.5, 1.0 - df['X']))
df['central_position'] = 1 - 2 * np.abs(df['Y'] - 0.5)
df['distance_sq'] = df['distance_to_goal'] ** 2
df['inside_box'] = (df['X'] > 0.83).astype(int)
df['very_close'] = (df['X'] > 0.94).astype(int)
df['is_home'] = (df['h_a'] == 'h').astype(int)

# Fill missing lastAction
df['lastAction'] = df['lastAction'].fillna('Unknown')

# Encode categoricals
shot_type_dummies = pd.get_dummies(df['shotType'], prefix='shot')
situation_dummies = pd.get_dummies(df['situation'], prefix='sit')
lastaction_dummies = pd.get_dummies(df['lastAction'], prefix='action')

numeric_features = ['X', 'Y', 'distance_to_goal', 'angle', 'central_position', 
                    'distance_sq', 'inside_box', 'very_close', 'is_home']

# Model 1: Without lastAction
X1 = pd.concat([df[numeric_features], shot_type_dummies, situation_dummies], axis=1)

# Model 2: With lastAction
X2 = pd.concat([df[numeric_features], shot_type_dummies, situation_dummies, lastaction_dummies], axis=1)

y = df['is_goal']

# Split
train_mask = df['season'] < 2024
X1_train, X1_test = X1[train_mask], X1[~train_mask]
X2_train, X2_test = X2[train_mask], X2[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"\nTrain: {len(X1_train):,}, Test: {len(X1_test):,}")
print(f"Features without lastAction: {X1.shape[1]}")
print(f"Features with lastAction: {X2.shape[1]}")

# Train models
model1 = LogisticRegression(max_iter=1000, C=1.0)
model1.fit(X1_train, y_train)
pred1 = model1.predict_proba(X1_test)[:, 1]

model2 = LogisticRegression(max_iter=1000, C=1.0)
model2.fit(X2_train, y_train)
pred2 = model2.predict_proba(X2_test)[:, 1]

print(f"\nRESULTS:")
print("-" * 60)
print(f"Without lastAction: Log Loss={log_loss(y_test, pred1):.4f}, AUC={roc_auc_score(y_test, pred1):.4f}")
print(f"With lastAction:    Log Loss={log_loss(y_test, pred2):.4f}, AUC={roc_auc_score(y_test, pred2):.4f}")

# Check lastAction feature importance
print(f"\nlastAction feature importance:")
action_cols = [c for c in X2.columns if c.startswith('action_')]
coef_df = pd.DataFrame({
    'feature': X2.columns,
    'coef': model2.coef_[0]
})
action_coef = coef_df[coef_df['feature'].isin(action_cols)].sort_values('coef', ascending=False)
print(action_coef.head(10).to_string(index=False))
print("...")
print(action_coef.tail(5).to_string(index=False))

# Calculate match xG with new model
print(f"\nCALCULATE MATCH XG WITH IMPROVED MODEL")
print("-" * 60)

df['our_xG_v2'] = model2.predict_proba(X2)[:, 1]

# Aggregate by match
match_xg = df.groupby(['match_id', 'h_goals', 'a_goals']).apply(
    lambda x: pd.Series({
        'h_xG_v2': x[x['h_a'] == 'h']['our_xG_v2'].sum(),
        'a_xG_v2': x[x['h_a'] == 'a']['our_xG_v2'].sum(),
    })
).reset_index()

print(f"Correlation with goals:")
print(f"  h_xG_v2 vs h_goals: {match_xg['h_xG_v2'].corr(match_xg['h_goals']):.4f}")
print(f"  a_xG_v2 vs a_goals: {match_xg['a_xG_v2'].corr(match_xg['a_goals']):.4f}")

# Save improved model
import pickle
model_data = {
    'model': model2,
    'feature_cols': list(X2.columns),
}
with open('experiments/our_xg_model_v2.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Save match xG
match_xg.to_csv('understat_data/our_match_xg_v2.csv', index=False)
print(f"\nSaved improved model and match xG")
