import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("VERIFICATION: CHECK FOR HARDCODED VALUES AND CORRECTNESS")
print("=" * 70)

# 1. Check referee_stats calculation
print("\n1. REFEREE STATS VERIFICATION:")
print("-" * 70)

df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

# Calculate referee stats from data
referee_stats = {}
for _, row in df.iterrows():
    ref = row['Referee']
    if ref not in referee_stats:
        referee_stats[ref] = {'home_wins': 0, 'total': 0}
    referee_stats[ref]['total'] += 1
    if row['FTR'] == 'H':
        referee_stats[ref]['home_wins'] += 1

# Check a few referees
print("Sample referee stats (calculated from data):")
for ref in list(referee_stats.keys())[:5]:
    stats = referee_stats[ref]
    rate = stats['home_wins'] / stats['total'] if stats['total'] > 0 else 0
    print(f"  {ref}: {stats['home_wins']}/{stats['total']} = {rate:.3f}")

# Overall home win rate
overall_home_rate = (df['FTR'] == 'H').mean()
print(f"\nOverall home win rate: {overall_home_rate:.4f}")
print(f"Hardcoded 0.446 vs actual: {abs(0.446 - overall_home_rate):.4f} difference")

# 2. Check xG calculation
print("\n2. XG CALCULATION VERIFICATION:")
print("-" * 70)

shots = pd.read_csv("understat_data/epl_shots_all.csv")
xg_df = pd.read_csv("understat_data/our_match_xg_v2.csv")

print(f"Total shots: {len(shots)}")
print(f"Total matches with xG: {len(xg_df)}")

# Check if our xG is calculated from raw data
print("\nSample xG values (our calculation vs Understat):")
sample_matches = xg_df.head(5)
for _, row in sample_matches.iterrows():
    match_id = row['match_id']
    match_shots = shots[shots['match_id'] == match_id]
    
    # Calculate xG from raw shots
    h_shots = match_shots[match_shots['h_a'] == 'h']
    a_shots = match_shots[match_shots['h_a'] == 'a']
    
    # Our xG should be from our model, not Understat's xG column
    print(f"  Match {match_id}:")
    print(f"    Home shots: {len(h_shots)}, Away shots: {len(a_shots)}")
    print(f"    Our h_xG: {row['h_xG_v2']:.3f}, Our a_xG: {row['a_xG_v2']:.3f}")

# 3. Check xG model
print("\n3. XG MODEL VERIFICATION:")
print("-" * 70)

# Load our xG model
import pickle
try:
    with open("understat_data/xg_model_v2.pkl", "rb") as f:
        xg_model_data = pickle.load(f)
    print(f"xG model loaded successfully")
    print(f"Model type: {type(xg_model_data.get('model', 'N/A'))}")
    print(f"Features used: {xg_model_data.get('features', 'N/A')}")
except Exception as e:
    print(f"Error loading xG model: {e}")

# 4. Check defaults calculation
print("\n4. DEFAULTS VERIFICATION:")
print("-" * 70)

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
train_seasons = SEASONS[:8]
train_df = df[df["season"].isin(train_seasons)]

print(f"Training data: {len(train_df)} matches from {train_seasons[0]} to {train_seasons[-1]}")

# Calculate defaults from training data
total = len(train_df)
defaults = {
    'home_win_rate': (train_df['FTR'] == 'H').sum() / total,
    'away_win_rate': (train_df['FTR'] == 'A').sum() / total,
    'draw_rate': (train_df['FTR'] == 'D').sum() / total,
    'avg_home_goals': train_df['FTHG'].mean(),
    'avg_away_goals': train_df['FTAG'].mean(),
    'avg_fouls': (train_df['HF'].mean() + train_df['AF'].mean()) / 2,
}

print("\nDefaults calculated from training data:")
for k, v in defaults.items():
    print(f"  {k}: {v:.4f}")

# 5. Check for data leakage
print("\n5. DATA LEAKAGE CHECK:")
print("-" * 70)

test_seasons = SEASONS[8:]
test_df = df[df["season"].isin(test_seasons)]
print(f"Test data: {len(test_df)} matches from {test_seasons[0]} to {test_seasons[-1]}")

# Check if referee stats use future data
print("\nReferee stats should only use historical data for prediction")
print("Current implementation uses ALL data to calculate referee home rate")
print("This is a potential issue - should use rolling/historical calculation")

# 6. Check feature calculation correctness
print("\n6. FEATURE CALCULATION CHECK:")
print("-" * 70)

# Check fouls_diff calculation
print("\nFouls data sample:")
print(f"  Home fouls avg: {df['HF'].mean():.2f}")
print(f"  Away fouls avg: {df['AF'].mean():.2f}")
print(f"  Combined avg: {(df['HF'].mean() + df['AF'].mean()) / 2:.2f}")

# Check if fouls_diff makes sense
# Higher fouls = more aggressive = could be good or bad
print("\nCorrelation of fouls with results:")
df['home_win'] = (df['FTR'] == 'H').astype(int)
print(f"  HF vs home_win: {df['HF'].corr(df['home_win']):.4f}")
print(f"  AF vs home_win: {df['AF'].corr(df['home_win']):.4f}")

# 7. Summary of potential issues
print("\n" + "=" * 70)
print("POTENTIAL ISSUES FOUND:")
print("=" * 70)

issues = []

# Issue 1: Referee stats use future data
issues.append("1. REFEREE STATS: Uses ALL data including future matches to calculate home rate")
issues.append("   FIX: Should use rolling/historical calculation like other features")

# Issue 2: Check if 0.446 is hardcoded
if abs(0.446 - overall_home_rate) > 0.01:
    issues.append(f"2. HARDCODED 0.446: Actual home rate is {overall_home_rate:.4f}")

for issue in issues:
    print(issue)

print("\n" + "=" * 70)
print("RECOMMENDATION: Fix referee_bias to use historical data only")
print("=" * 70)
