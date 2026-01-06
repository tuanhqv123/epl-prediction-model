"""
Test more xG-based features
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load shots data
shots = pd.read_csv("understat_data/epl_shots_all.csv")
shots['xG'] = shots['xG'].astype(float)

print("ANALYZING XG DATA FOR NEW FEATURES")
print("="*60)

# Calculate per-match xG stats
match_stats = shots.groupby('match_id').apply(
    lambda x: pd.Series({
        'h_shots': len(x[x['h_a'] == 'h']),
        'a_shots': len(x[x['h_a'] == 'a']),
        'h_xG': x[x['h_a'] == 'h']['xG'].sum(),
        'a_xG': x[x['h_a'] == 'a']['xG'].sum(),
        'h_big_chances': len(x[(x['h_a'] == 'h') & (x['xG'] > 0.3)]),
        'a_big_chances': len(x[(x['h_a'] == 'a') & (x['xG'] > 0.3)]),
        'h_avg_xG': x[x['h_a'] == 'h']['xG'].mean() if len(x[x['h_a'] == 'h']) > 0 else 0,
        'a_avg_xG': x[x['h_a'] == 'a']['xG'].mean() if len(x[x['h_a'] == 'a']) > 0 else 0,
    })
).reset_index()

match_info = shots.groupby('match_id').first()[['date', 'h_team', 'a_team', 'season', 'h_goals', 'a_goals']].reset_index()
match_stats = match_stats.merge(match_info, on='match_id')
match_stats['date'] = pd.to_datetime(match_stats['date'])

print(f"Matches with xG data: {len(match_stats)}")
print(f"\nSample stats:")
print(match_stats[['h_team', 'a_team', 'h_xG', 'a_xG', 'h_big_chances', 'a_big_chances']].head())

# Calculate rolling xG stats per team
print("\n" + "="*60)
print("CALCULATING ROLLING XG FEATURES")
print("="*60)

team_xg_history = {}

for _, row in match_stats.sort_values('date').iterrows():
    h_team, a_team = row['h_team'], row['a_team']
    
    for team, is_home in [(h_team, True), (a_team, False)]:
        if team not in team_xg_history:
            team_xg_history[team] = {
                'xG': [], 'xGA': [], 'big_chances': [], 'shots': [], 'avg_xG_per_shot': []
            }
        
        if is_home:
            team_xg_history[team]['xG'].append(row['h_xG'])
            team_xg_history[team]['xGA'].append(row['a_xG'])
            team_xg_history[team]['big_chances'].append(row['h_big_chances'])
            team_xg_history[team]['shots'].append(row['h_shots'])
            team_xg_history[team]['avg_xG_per_shot'].append(row['h_avg_xG'])
        else:
            team_xg_history[team]['xG'].append(row['a_xG'])
            team_xg_history[team]['xGA'].append(row['h_xG'])
            team_xg_history[team]['big_chances'].append(row['a_big_chances'])
            team_xg_history[team]['shots'].append(row['a_shots'])
            team_xg_history[team]['avg_xG_per_shot'].append(row['a_avg_xG'])

# Show sample
print("\nSample team xG history (Man City):")
if 'Manchester City' in team_xg_history:
    hist = team_xg_history['Manchester City']
    print(f"  Last 5 xG: {hist['xG'][-5:]}")
    print(f"  Last 5 xGA: {hist['xGA'][-5:]}")
    print(f"  Last 5 big chances: {hist['big_chances'][-5:]}")

# Ideas for new features:
# 1. Rolling xG difference (xG - xGA)
# 2. Big chances created per game
# 3. xG per shot (shot quality)
# 4. xG overperformance (goals - xG)

print("\n" + "="*60)
print("NEW XG-BASED FEATURE IDEAS:")
print("="*60)
print("""
1. xg_rolling_diff: Rolling (xG - xGA) over last N games
2. big_chances_diff: Difference in big chances created
3. shot_quality_diff: Average xG per shot difference
4. xg_overperformance: Goals scored vs xG (luck/finishing)
""")
