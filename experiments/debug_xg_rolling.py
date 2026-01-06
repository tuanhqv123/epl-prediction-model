import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from datetime import timedelta

# Load data
df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

xg_df = pd.read_csv("understat_data/our_match_xg.csv")
xg_df['date'] = pd.to_datetime(xg_df['date'])

print("DEBUG XG ROLLING CALCULATION")
print("-" * 60)

# Build lookup
team_mapping = {
    'Manchester United': 'Man United',
    'Manchester City': 'Man City', 
    'Newcastle United': 'Newcastle',
    'Wolverhampton Wanderers': 'Wolves',
    "Nottingham Forest": "Nott'm Forest",
    'West Bromwich Albion': 'West Brom',
    'Queens Park Rangers': 'QPR',
}
reverse_mapping = {v: k for k, v in team_mapping.items()}

xg_lookup = {}
for _, row in xg_df.iterrows():
    h_team = row['h_team']
    a_team = row['a_team']
    date = row['date'].date()
    data = {'h_xG': row['h_xG'], 'a_xG': row['a_xG']}
    for d in [date, date + timedelta(days=1), date - timedelta(days=1)]:
        xg_lookup[(h_team, a_team, d)] = data

def get_xg(home, away, date):
    key = (home, away, date)
    if key in xg_lookup:
        return xg_lookup[key]
    h_mapped = reverse_mapping.get(home, home)
    a_mapped = reverse_mapping.get(away, away)
    key = (h_mapped, a_mapped, date)
    if key in xg_lookup:
        return xg_lookup[key]
    return None

# Simulate rolling xG calculation
team_xg_history = {}
results = []

for idx, m in df.iterrows():
    home, away = m["HomeTeam"], m["AwayTeam"]
    match_date = m["Date"]
    
    # Get rolling xG BEFORE this match
    h_xg_hist = team_xg_history.get(home, [])
    a_xg_hist = team_xg_history.get(away, [])
    h_xg_rolling = np.mean(h_xg_hist[-5:]) if h_xg_hist else 1.35
    a_xg_rolling = np.mean(a_xg_hist[-5:]) if a_xg_hist else 1.35
    
    results.append({
        'date': match_date,
        'home': home,
        'away': away,
        'h_xg_rolling': h_xg_rolling,
        'a_xg_rolling': a_xg_rolling,
        'h_matches': len(h_xg_hist),
        'a_matches': len(a_xg_hist),
        'FTR': m['FTR']
    })
    
    # Update xG history AFTER match
    match_xg = get_xg(home, away, match_date.date())
    if match_xg:
        team_xg_history.setdefault(home, []).append(match_xg['h_xG'])
        team_xg_history.setdefault(away, []).append(match_xg['a_xG'])

results_df = pd.DataFrame(results)

print("1. SAMPLE ROLLING XG")
print("-" * 60)
print(results_df.head(20).to_string())

print("\n2. CHECK ROLLING XG DISTRIBUTION")
print("-" * 60)
print(f"h_xg_rolling: mean={results_df['h_xg_rolling'].mean():.3f}, std={results_df['h_xg_rolling'].std():.3f}")
print(f"a_xg_rolling: mean={results_df['a_xg_rolling'].mean():.3f}, std={results_df['a_xg_rolling'].std():.3f}")

# Check correlation with result
results_df['xg_diff'] = results_df['h_xg_rolling'] - results_df['a_xg_rolling']
results_df['home_win'] = (results_df['FTR'] == 'H').astype(int)
results_df['away_win'] = (results_df['FTR'] == 'A').astype(int)

print(f"\n3. XG DIFF VS RESULT")
print("-" * 60)
for result in ['H', 'D', 'A']:
    mask = results_df['FTR'] == result
    print(f"{result}: avg xg_diff = {results_df.loc[mask, 'xg_diff'].mean():.3f}")

print(f"\nCorrelation xg_diff vs home_win: {results_df['xg_diff'].corr(results_df['home_win']):.4f}")

# Check if early matches have default values
print(f"\n4. EARLY MATCHES (first 100)")
print("-" * 60)
early = results_df.head(100)
default_count = ((early['h_xg_rolling'] == 1.35) | (early['a_xg_rolling'] == 1.35)).sum()
print(f"Matches with default xG: {default_count}/100")

# Check late matches
print(f"\n5. LATE MATCHES (last 100)")
print("-" * 60)
late = results_df.tail(100)
print(late[['date', 'home', 'away', 'h_xg_rolling', 'a_xg_rolling', 'h_matches', 'a_matches']].head(10).to_string())

# The issue might be that xG rolling doesn't add much info beyond what ELO already captures
print(f"\n6. COMPARE XG ROLLING VS EXPECTED_DIFF (SOT-based)")
print("-" * 60)
# expected_diff is calculated from SOT * conversion_rate
# Let's see if xG rolling is correlated with it

# Load the feature data to compare
from epl_production_data_driven import load_non_league_data, load_lineup_data

# Quick check: is xG rolling just redundant with other features?
print("xG rolling might be redundant with existing features like:")
print("- expected_diff (SOT-based proxy xG)")
print("- att_def_matchup (attack/defense ELO)")
print("- elo_expected")
print("\nThis could explain why adding xG doesn't improve the model.")
