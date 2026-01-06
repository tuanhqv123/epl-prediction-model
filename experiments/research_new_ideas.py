import pandas as pd
import numpy as np

print("RESEARCH: WHAT DATA/FEATURES HAVEN'T WE TRIED?")
print("=" * 70)

# Load all data
df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
shots = pd.read_csv("understat_data/epl_shots_all.csv")

print("\n1. MAIN MATCH DATA - UNUSED COLUMNS:")
print("-" * 70)
used = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 
        'season', 'Referee', 'HF', 'AF']
unused = [c for c in df.columns if c not in used]
print(f"Unused: {unused}")

for col in ['HTHG', 'HTAG', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']:
    if col in df.columns and df[col].dtype in ['int64', 'float64']:
        print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
print(f"  HTR: {df['HTR'].value_counts().to_dict()}")

print("\n2. SHOT DATA - UNUSED COLUMNS:")
print("-" * 70)
shot_used = ['X', 'Y', 'shotType', 'situation', 'lastAction', 'result', 'h_a', 'match_id', 
             'date', 'h_team', 'a_team', 'season', 'minute', 'xG']
shot_unused = [c for c in shots.columns if c not in shot_used]
print(f"Unused: {shot_unused}")

for col in shot_unused:
    if shots[col].dtype == 'object':
        print(f"  {col}: {shots[col].nunique()} unique - {shots[col].value_counts().head(3).to_dict()}")
    else:
        print(f"  {col}: mean={shots[col].mean():.2f}")

print("\n3. POTENTIAL NEW FEATURES TO TRY:")
print("-" * 70)

# A. Goal timing patterns
print("\nA. GOAL TIMING PATTERNS:")
shots['is_goal'] = (shots['result'] == 'Goal').astype(int)
early_goals = shots[shots['minute'] <= 30].groupby('match_id')['is_goal'].sum()
late_goals = shots[shots['minute'] >= 75].groupby('match_id')['is_goal'].sum()
print(f"   Avg early goals (0-30 min): {early_goals.mean():.2f}")
print(f"   Avg late goals (75+ min): {late_goals.mean():.2f}")

# B. Shot location patterns
print("\nB. SHOT LOCATION PATTERNS:")
shots['distance'] = np.sqrt((shots['X'] - 100)**2 + (shots['Y'] - 50)**2)
close_shots = shots[shots['distance'] < 15]
far_shots = shots[shots['distance'] >= 25]
print(f"   Close shots (<15): {len(close_shots)} ({close_shots['is_goal'].mean()*100:.1f}% conversion)")
print(f"   Far shots (>=25): {len(far_shots)} ({far_shots['is_goal'].mean()*100:.1f}% conversion)")

# C. Player data
print("\nC. PLAYER DATA:")
print(f"   Unique players: {shots['player'].nunique()}")
print(f"   Top scorers: {shots[shots['is_goal']==1]['player'].value_counts().head(5).to_dict()}")

# D. Assist patterns
print("\nD. ASSIST PATTERNS:")
print(f"   Unique assisters: {shots['player_assisted'].nunique()}")

# E. Half-time patterns
print("\nE. HALF-TIME PATTERNS:")
df['ht_lead'] = df['HTHG'] - df['HTAG']
df['home_win'] = (df['FTR'] == 'H').astype(int)
print(f"   HT lead correlation with home win: {df['ht_lead'].corr(df['home_win']):.4f}")
ht_lead_wins = df[df['ht_lead'] > 0]['FTR'].value_counts(normalize=True)
print(f"   When home leads at HT: H={ht_lead_wins.get('H',0)*100:.1f}%, D={ht_lead_wins.get('D',0)*100:.1f}%, A={ht_lead_wins.get('A',0)*100:.1f}%")

# F. Cards patterns
print("\nF. CARDS PATTERNS:")
print(f"   Yellow cards correlation with home win: HY={df['HY'].corr(df['home_win']):.4f}, AY={df['AY'].corr(df['home_win']):.4f}")
print(f"   Red cards correlation with home win: HR={df['HR'].corr(df['home_win']):.4f}, AR={df['AR'].corr(df['home_win']):.4f}")

print("\n4. IDEAS FOR NEW FEATURES:")
print("-" * 70)
print("""
A. FROM SHOT DATA (not yet used):
   1. shot_conversion_rate: goals / shots (rolling)
   2. close_shot_rate: % of shots from close range
   3. counter_attack_xg: xG from fast breaks
   4. set_piece_efficiency: goals from set pieces / set piece shots
   5. player_form: top scorer's recent goals

B. FROM MATCH DATA (not yet used):
   1. ht_dominance: historical HT goal difference
   2. comeback_rate: % of times team came back from losing
   3. red_card_rate: historical red cards per game
   4. corner_efficiency: goals from corners / corners

C. DERIVED FEATURES:
   1. momentum: weighted recent results (more recent = more weight)
   2. consistency: std of recent results
   3. big_game_performance: results vs top 6 teams
   4. home_away_split: difference in home vs away performance

D. INTERACTION FEATURES:
   1. elo_x_form: ELO * recent form
   2. xg_x_conversion: xG * shot conversion rate
""")

print("\n5. QUICK TEST - SHOT CONVERSION RATE:")
print("-" * 70)

# Calculate shot conversion rate per team per match
match_stats = shots.groupby(['match_id', 'h_a']).agg({
    'is_goal': 'sum',
    'X': 'count'  # total shots
}).reset_index()
match_stats.columns = ['match_id', 'h_a', 'goals', 'shots']
match_stats['conversion'] = match_stats['goals'] / match_stats['shots']

print(f"Average conversion rate: {match_stats['conversion'].mean()*100:.1f}%")
print(f"Home conversion: {match_stats[match_stats['h_a']=='h']['conversion'].mean()*100:.1f}%")
print(f"Away conversion: {match_stats[match_stats['h_a']=='a']['conversion'].mean()*100:.1f}%")
