import pandas as pd
import numpy as np
import pickle

print("CALCULATE MATCH XG FROM OUR MODEL")
print("-" * 60)

# Load shot data
shots = pd.read_csv("understat_data/epl_shots_all.csv")
print(f"Total shots: {len(shots):,}")

# Load our xG model
with open('experiments/our_xg_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['feature_cols']

# Prepare features (same as training)
shots['X'] = shots['X'].astype(float)
shots['Y'] = shots['Y'].astype(float)
shots['distance_to_goal'] = np.sqrt((1.0 - shots['X'])**2 + (0.5 - shots['Y'])**2)
shots['angle'] = np.abs(np.arctan2(shots['Y'] - 0.5, 1.0 - shots['X']))
shots['central_position'] = 1 - 2 * np.abs(shots['Y'] - 0.5)
shots['distance_sq'] = shots['distance_to_goal'] ** 2
shots['inside_box'] = (shots['X'] > 0.83).astype(int)
shots['very_close'] = (shots['X'] > 0.94).astype(int)
shots['is_home'] = (shots['h_a'] == 'h').astype(int)

shot_type_dummies = pd.get_dummies(shots['shotType'], prefix='shot')
situation_dummies = pd.get_dummies(shots['situation'], prefix='sit')

numeric_features = ['X', 'Y', 'distance_to_goal', 'angle', 'central_position', 
                    'distance_sq', 'inside_box', 'very_close', 'is_home']
X = pd.concat([shots[numeric_features], shot_type_dummies, situation_dummies], axis=1)

# Ensure all columns exist
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

# Calculate xG for each shot
shots['our_xG'] = model.predict_proba(X)[:, 1]

print(f"Our xG calculated for all shots")
print(f"Our xG mean: {shots['our_xG'].mean():.4f}")
print(f"Actual goal rate: {(shots['result'] == 'Goal').mean():.4f}")

# Aggregate by match
print(f"\nAGGREGATE BY MATCH")
print("-" * 60)

# Home team xG = sum of home shots xG
# Away team xG = sum of away shots xG
match_xg = shots.groupby(['match_id', 'date', 'h_team', 'a_team', 'h_goals', 'a_goals', 'season']).apply(
    lambda x: pd.Series({
        'h_xG': x.loc[x['h_a'] == 'h', 'our_xG'].sum(),
        'a_xG': x.loc[x['h_a'] == 'a', 'our_xG'].sum(),
        'h_shots': (x['h_a'] == 'h').sum(),
        'a_shots': (x['h_a'] == 'a').sum(),
    })
).reset_index()

print(f"Total matches: {len(match_xg)}")
print(f"\nSample data:")
print(match_xg[['date', 'h_team', 'a_team', 'h_goals', 'a_goals', 'h_xG', 'a_xG', 'h_shots', 'a_shots']].head(10).to_string())

# Verify xG makes sense
print(f"\nVERIFY XG")
print("-" * 60)
print(f"Avg Home xG: {match_xg['h_xG'].mean():.3f}")
print(f"Avg Away xG: {match_xg['a_xG'].mean():.3f}")
print(f"Avg Home Goals: {match_xg['h_goals'].mean():.3f}")
print(f"Avg Away Goals: {match_xg['a_goals'].mean():.3f}")

print(f"\nCorrelation xG vs Goals:")
print(f"  Home: {match_xg['h_xG'].corr(match_xg['h_goals']):.4f}")
print(f"  Away: {match_xg['a_xG'].corr(match_xg['a_goals']):.4f}")

# xG difference vs result
match_xg['xG_diff'] = match_xg['h_xG'] - match_xg['a_xG']
match_xg['goal_diff'] = match_xg['h_goals'] - match_xg['a_goals']
match_xg['result'] = np.where(match_xg['h_goals'] > match_xg['a_goals'], 'H',
                     np.where(match_xg['h_goals'] < match_xg['a_goals'], 'A', 'D'))

print(f"  xG diff vs Goal diff: {match_xg['xG_diff'].corr(match_xg['goal_diff']):.4f}")

# Check by result
print(f"\nAvg xG diff by result:")
for result in ['H', 'D', 'A']:
    mask = match_xg['result'] == result
    print(f"  {result}: xG_diff = {match_xg.loc[mask, 'xG_diff'].mean():.3f}")

# Save
match_xg.to_csv('understat_data/our_match_xg.csv', index=False)
print(f"\nSaved to understat_data/our_match_xg.csv")

# Also check per season
print(f"\nPER SEASON CHECK")
print("-" * 60)
for season in sorted(match_xg['season'].unique()):
    s_df = match_xg[match_xg['season'] == season]
    print(f"{season}: {len(s_df)} matches, h_xG={s_df['h_xG'].mean():.2f}, a_xG={s_df['a_xG'].mean():.2f}")
