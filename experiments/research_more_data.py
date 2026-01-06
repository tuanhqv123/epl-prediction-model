import pandas as pd
import numpy as np

print("RESEARCH: WHAT DATA DO WE HAVE BUT NOT USING?")
print("=" * 60)

# 1. Main match data
df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
print("\n1. MAIN MATCH DATA COLUMNS:")
print(df.columns.tolist())

# Check what we're NOT using
used_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'season']
unused_cols = [c for c in df.columns if c not in used_cols]
print(f"\nUNUSED columns: {unused_cols}")

for col in unused_cols:
    if df[col].dtype in ['int64', 'float64']:
        print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    else:
        print(f"  {col}: {df[col].nunique()} unique values")

# 2. Shot data
shots = pd.read_csv("understat_data/epl_shots_all.csv")
print("\n2. SHOT DATA COLUMNS:")
print(shots.columns.tolist())

# Check what we're NOT using from shots
used_shot_cols = ['X', 'Y', 'shotType', 'situation', 'lastAction', 'result', 'h_a', 'match_id', 'date', 'h_team', 'a_team', 'season']
unused_shot_cols = [c for c in shots.columns if c not in used_shot_cols]
print(f"\nUNUSED shot columns: {unused_shot_cols}")

for col in unused_shot_cols:
    if shots[col].dtype in ['int64', 'float64']:
        print(f"  {col}: mean={shots[col].mean():.2f}")
    else:
        print(f"  {col}: {shots[col].nunique()} unique values")

# 3. Check potential new features
print("\n3. POTENTIAL NEW FEATURES:")
print("-" * 60)

# Half-time data
print("\nA. HALF-TIME DATA (HTHG, HTAG, HTR):")
print(f"   Home HT goals avg: {df['HTHG'].mean():.2f}")
print(f"   Away HT goals avg: {df['HTAG'].mean():.2f}")
# Can we use HT result to predict FT? No - that's cheating (data from same match)
# But we can use HISTORICAL HT performance

# Referee data
print("\nB. REFEREE DATA:")
print(f"   Unique referees: {df['Referee'].nunique()}")
print(f"   Top referees: {df['Referee'].value_counts().head(5).to_dict()}")

# Cards data
print("\nC. CARDS DATA:")
print(f"   Home Yellow avg: {df['HY'].mean():.2f}")
print(f"   Away Yellow avg: {df['AY'].mean():.2f}")
print(f"   Home Red avg: {df['HR'].mean():.3f}")
print(f"   Away Red avg: {df['AR'].mean():.3f}")

# Corners
print("\nD. CORNERS DATA:")
print(f"   Home Corners avg: {df['HC'].mean():.2f}")
print(f"   Away Corners avg: {df['AC'].mean():.2f}")

# Fouls
print("\nE. FOULS DATA:")
print(f"   Home Fouls avg: {df['HF'].mean():.2f}")
print(f"   Away Fouls avg: {df['AF'].mean():.2f}")

# Shot minute data
print("\nF. SHOT TIMING (minute):")
print(f"   Shot minute range: {shots['minute'].min()} - {shots['minute'].max()}")
# Can calculate: late goals tendency, early pressure, etc.

# Player data
print("\nG. PLAYER DATA:")
print(f"   Unique players: {shots['player'].nunique()}")
print(f"   Unique assisters: {shots['player_assisted'].nunique()}")

# 4. Correlation analysis
print("\n4. CORRELATION WITH HOME WIN:")
print("-" * 60)
df['home_win'] = (df['FTR'] == 'H').astype(int)
for col in ['HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'HTHG', 'HTAG']:
    corr = df[col].corr(df['home_win'])
    print(f"  {col}: {corr:.4f}")

# 5. Ideas for new features
print("\n5. IDEAS FOR NEW FEATURES:")
print("-" * 60)
print("""
A. FROM MAIN DATA:
   - corners_rolling: team's corner-taking ability
   - fouls_rolling: team's discipline
   - cards_rolling: team's aggression/discipline
   - ht_goals_rolling: team's first-half performance
   - referee_home_bias: referee's historical home win rate

B. FROM SHOT DATA:
   - big_chance_rate: % of shots with xG > 0.3
   - shot_quality: average xG per shot
   - late_goal_tendency: % of goals in last 15 min
   - counter_attack_xg: xG from fast breaks (Rebound, BallRecovery)

C. DERIVED:
   - form_momentum: weighted recent results
   - goal_timing_pattern: when team typically scores
""")
