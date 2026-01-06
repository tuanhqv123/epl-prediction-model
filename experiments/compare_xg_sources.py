import pandas as pd
import numpy as np

print("COMPARE XG SOURCES")
print("-" * 60)

# Load data
shots = pd.read_csv('understat_data/epl_shots_all.csv')
matches_clean = pd.read_csv('understat_data/matches_clean.csv')

# Calculate xG from shots (sum of individual shot xG)
shots['xG'] = shots['xG'].astype(float)
shots_xg = shots.groupby(['match_id', 'h_a']).agg({'xG': 'sum'}).reset_index()
shots_xg_pivot = shots_xg.pivot(index='match_id', columns='h_a', values='xG').reset_index()
shots_xg_pivot.columns = ['match_id', 'a_xG_shots', 'h_xG_shots']

# Merge with match data
matches_clean['match_id'] = matches_clean['id']
merged = matches_clean.merge(shots_xg_pivot, on='match_id', how='left')

print("1. COMPARE XG FROM MATCH DATA vs SUM OF SHOT XG")
print("-" * 60)
print(f"Match data h_xG mean: {merged['h_xG'].mean():.4f}")
print(f"Sum of shots h_xG mean: {merged['h_xG_shots'].mean():.4f}")
print(f"Difference: {abs(merged['h_xG'].mean() - merged['h_xG_shots'].mean()):.4f}")
print()
print(f"Match data a_xG mean: {merged['a_xG'].mean():.4f}")
print(f"Sum of shots a_xG mean: {merged['a_xG_shots'].mean():.4f}")
print(f"Difference: {abs(merged['a_xG'].mean() - merged['a_xG_shots'].mean()):.4f}")

print(f"\nCorrelation between two sources:")
print(f"  Home: {merged['h_xG'].corr(merged['h_xG_shots']):.4f}")
print(f"  Away: {merged['a_xG'].corr(merged['a_xG_shots']):.4f}")

# Check sample
print(f"\n2. SAMPLE COMPARISON")
print("-" * 60)
sample = merged[['date', 'h_team', 'a_team', 'h_xG', 'h_xG_shots', 'a_xG', 'a_xG_shots']].head(10)
sample['h_diff'] = sample['h_xG'] - sample['h_xG_shots']
sample['a_diff'] = sample['a_xG'] - sample['a_xG_shots']
print(sample.to_string())

print(f"\n3. XG FROM UNDERSTAT (THEIR MODEL) - THIS IS WHAT WE SHOULD USE")
print("-" * 60)
print("Understat xG is calculated by their model, not by us")
print("We should use h_xG and a_xG from match data directly")
print("This is the 'real' xG from Understat's model")

# Now let's use Understat's xG properly
print(f"\n4. VERIFY UNDERSTAT XG QUALITY")
print("-" * 60)
print(f"xG vs Goals correlation:")
print(f"  Home xG vs Home Goals: {merged['h_xG'].corr(merged['h_goals']):.4f}")
print(f"  Away xG vs Away Goals: {merged['a_xG'].corr(merged['a_goals']):.4f}")

# xG difference vs result
merged['xG_diff'] = merged['h_xG'] - merged['a_xG']
merged['goal_diff'] = merged['h_goals'] - merged['a_goals']
print(f"  xG diff vs Goal diff: {merged['xG_diff'].corr(merged['goal_diff']):.4f}")

# Save proper xG data
proper_xg = merged[['match_id', 'date', 'h_team', 'a_team', 'h_goals', 'a_goals', 
                    'h_xG', 'a_xG', 'forecast_h', 'forecast_d', 'forecast_a', 'result']]
proper_xg.to_csv('understat_data/matches_with_xg.csv', index=False)
print(f"\nSaved to understat_data/matches_with_xg.csv")
