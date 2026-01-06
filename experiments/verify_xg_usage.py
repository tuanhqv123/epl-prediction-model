import pandas as pd
import numpy as np

print("VERIFY XG CALCULATION AND USAGE")
print("=" * 60)

# Load data
shots = pd.read_csv('understat_data/epl_shots_all.csv')
our_xg = pd.read_csv('understat_data/our_match_xg.csv')

print("1. CHECK OUR XG MODEL OUTPUT")
print("-" * 60)

# Our xG model predicts probability of goal for each shot
# Then we SUM all shot xG to get match xG
# Is this correct?

# Check a specific match
match_id = 4749  # First match in data
match_shots = shots[shots['match_id'] == match_id]
print(f"Match {match_id}: {match_shots['h_team'].iloc[0]} vs {match_shots['a_team'].iloc[0]}")
print(f"Score: {match_shots['h_goals'].iloc[0]}-{match_shots['a_goals'].iloc[0]}")
print(f"Total shots: {len(match_shots)}")
print(f"Home shots: {(match_shots['h_a'] == 'h').sum()}")
print(f"Away shots: {(match_shots['h_a'] == 'a').sum()}")

# Check Understat xG for this match
understat_h_xg = match_shots[match_shots['h_a'] == 'h']['xG'].sum()
understat_a_xg = match_shots[match_shots['h_a'] == 'a']['xG'].sum()
print(f"\nUnderstat xG (sum of shots): Home={understat_h_xg:.3f}, Away={understat_a_xg:.3f}")

# Check our xG
our_match = our_xg[our_xg['match_id'] == match_id]
if len(our_match) > 0:
    print(f"Our xG: Home={our_match['h_xG'].iloc[0]:.3f}, Away={our_match['a_xG'].iloc[0]:.3f}")

print("\n2. COMPARE UNDERSTAT XG VS OUR XG (all matches)")
print("-" * 60)

# Calculate Understat xG per match
understat_match_xg = shots.groupby('match_id').apply(
    lambda x: pd.Series({
        'understat_h_xG': x[x['h_a'] == 'h']['xG'].sum(),
        'understat_a_xG': x[x['h_a'] == 'a']['xG'].sum(),
    })
).reset_index()

# Merge with our xG
merged = our_xg.merge(understat_match_xg, on='match_id')

print(f"Correlation Our xG vs Understat xG:")
print(f"  Home: {merged['h_xG'].corr(merged['understat_h_xG']):.4f}")
print(f"  Away: {merged['a_xG'].corr(merged['understat_a_xG']):.4f}")

print(f"\nMean comparison:")
print(f"  Our Home xG: {merged['h_xG'].mean():.3f}")
print(f"  Understat Home xG: {merged['understat_h_xG'].mean():.3f}")
print(f"  Our Away xG: {merged['a_xG'].mean():.3f}")
print(f"  Understat Away xG: {merged['understat_a_xG'].mean():.3f}")

print("\n3. CHECK IF OUR XG PREDICTS GOALS WELL")
print("-" * 60)

# Correlation with actual goals
print(f"Correlation with actual goals:")
print(f"  Our h_xG vs h_goals: {merged['h_xG'].corr(merged['h_goals']):.4f}")
print(f"  Understat h_xG vs h_goals: {merged['understat_h_xG'].corr(merged['h_goals']):.4f}")
print(f"  Our a_xG vs a_goals: {merged['a_xG'].corr(merged['a_goals']):.4f}")
print(f"  Understat a_xG vs a_goals: {merged['understat_a_xG'].corr(merged['a_goals']):.4f}")

print("\n4. CHECK SAMPLE SHOTS FROM OUR XG MODEL")
print("-" * 60)

# Load our xG model and check predictions
import pickle
with open('experiments/our_xg_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['feature_cols']

# Prepare features for sample shots
sample = shots.head(20).copy()
sample['distance_to_goal'] = np.sqrt((1.0 - sample['X'])**2 + (0.5 - sample['Y'])**2)
sample['angle'] = np.abs(np.arctan2(sample['Y'] - 0.5, 1.0 - sample['X']))
sample['central_position'] = 1 - 2 * np.abs(sample['Y'] - 0.5)
sample['distance_sq'] = sample['distance_to_goal'] ** 2
sample['inside_box'] = (sample['X'] > 0.83).astype(int)
sample['very_close'] = (sample['X'] > 0.94).astype(int)
sample['is_home'] = (sample['h_a'] == 'h').astype(int)

shot_type_dummies = pd.get_dummies(sample['shotType'], prefix='shot')
situation_dummies = pd.get_dummies(sample['situation'], prefix='sit')

numeric_features = ['X', 'Y', 'distance_to_goal', 'angle', 'central_position', 
                    'distance_sq', 'inside_box', 'very_close', 'is_home']
X = pd.concat([sample[numeric_features], shot_type_dummies, situation_dummies], axis=1)

for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

sample['our_xG'] = model.predict_proba(X)[:, 1]

print("Sample shots comparison:")
print(sample[['X', 'Y', 'shotType', 'situation', 'result', 'xG', 'our_xG']].to_string())

print("\n5. KEY INSIGHT")
print("-" * 60)
print("""
Understat xG correlation với goals: ~0.62
Our xG correlation với goals: ~0.49

=> Understat xG TỐT HƠN vì họ có thêm features:
   - Defender positions
   - Goalkeeper position  
   - Game state (score at time of shot)
   - Player skill

=> Chúng ta chỉ có: X, Y, shotType, situation
=> Đây là limitation của raw data
""")
