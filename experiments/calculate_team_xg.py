import pandas as pd
import numpy as np
import pickle

print("CALCULATE TEAM XG PER MATCH")
print("-" * 50)

# Load shot data and model
df = pd.read_csv("understat_data/epl_shots_all.csv")
with open('experiments/xg_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['feature_cols']

# Prepare features
df['X'] = df['X'].astype(float)
df['Y'] = df['Y'].astype(float)
df['distance'] = np.sqrt((1 - df['X'])**2 + (0.5 - df['Y'])**2)
df['angle'] = np.abs(np.arctan2(df['Y'] - 0.5, 1 - df['X']))
df['distance_sq'] = df['distance'] ** 2
df['central'] = 1 - np.abs(df['Y'] - 0.5) * 2

shot_type_dummies = pd.get_dummies(df['shotType'], prefix='type')
situation_dummies = pd.get_dummies(df['situation'], prefix='sit')

X = pd.concat([df[['X', 'Y', 'distance', 'angle', 'distance_sq', 'central']], 
               shot_type_dummies, situation_dummies], axis=1)

# Ensure all columns exist
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

# Calculate our xG
df['our_xg'] = model.predict_proba(X)[:, 1]

# Aggregate by match
df['is_home'] = df['h_a'] == 'h'
df['team'] = np.where(df['is_home'], df['h_team'], df['a_team'])

match_xg = df.groupby(['match_id', 'date', 'h_team', 'a_team', 'h_goals', 'a_goals', 'season']).agg({
    'our_xg': lambda x: x[df.loc[x.index, 'is_home']].sum(),
    'xG': lambda x: x[df.loc[x.index, 'is_home']].sum()
}).reset_index()
match_xg.columns = ['match_id', 'date', 'h_team', 'a_team', 'h_goals', 'a_goals', 'season', 'h_xg_our', 'h_xg_understat']

# Get away xG
away_xg = df.groupby('match_id').agg({
    'our_xg': lambda x: x[~df.loc[x.index, 'is_home']].sum(),
    'xG': lambda x: x[~df.loc[x.index, 'is_home']].sum()
}).reset_index()
away_xg.columns = ['match_id', 'a_xg_our', 'a_xg_understat']

match_xg = match_xg.merge(away_xg, on='match_id')

print(f"Total matches: {len(match_xg)}")
print(f"\nSample data:")
print(match_xg[['date', 'h_team', 'a_team', 'h_goals', 'a_goals', 'h_xg_our', 'a_xg_our']].head(10).to_string())

# Verify xG vs actual goals
print(f"\nXG VS ACTUAL GOALS CORRELATION")
print("-" * 50)
print(f"Home xG (our) vs Home Goals: {match_xg['h_xg_our'].corr(match_xg['h_goals']):.4f}")
print(f"Away xG (our) vs Away Goals: {match_xg['a_xg_our'].corr(match_xg['a_goals']):.4f}")
print(f"Home xG (Understat) vs Home Goals: {match_xg['h_xg_understat'].corr(match_xg['h_goals']):.4f}")
print(f"Away xG (Understat) vs Away Goals: {match_xg['a_xg_understat'].corr(match_xg['a_goals']):.4f}")

# Save
match_xg.to_csv('understat_data/match_xg.csv', index=False)
print(f"\nSaved to understat_data/match_xg.csv")

# Summary stats
print(f"\nSUMMARY STATS")
print("-" * 50)
print(f"Avg Home xG (our): {match_xg['h_xg_our'].mean():.3f}")
print(f"Avg Away xG (our): {match_xg['a_xg_our'].mean():.3f}")
print(f"Avg Home Goals: {match_xg['h_goals'].mean():.3f}")
print(f"Avg Away Goals: {match_xg['a_goals'].mean():.3f}")
