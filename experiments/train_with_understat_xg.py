import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from epl_production_data_driven import (
    load_non_league_data, load_lineup_data,
    get_avg_change_rate, get_winrate, get_clean_sheet_rate
)

# Load main data
df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

# Load Understat xG data
xg_df = pd.read_csv("understat_data/matches_with_xg.csv")
xg_df['date'] = pd.to_datetime(xg_df['date'])

print("INTEGRATE UNDERSTAT XG INTO MODEL")
print("-" * 60)

# Check date range overlap
print(f"Main data: {df['Date'].min()} to {df['Date'].max()}")
print(f"Understat: {xg_df['date'].min()} to {xg_df['date'].max()}")

# Team name mapping (Understat uses different names)
team_mapping = {
    'Manchester United': 'Man United',
    'Manchester City': 'Man City',
    'Newcastle United': 'Newcastle',
    'Wolverhampton Wanderers': 'Wolves',
    'West Ham': 'West Ham',
    'Tottenham': 'Tottenham',
    'Brighton': 'Brighton',
    'Nottingham Forest': "Nott'm Forest",
    'Sheffield United': 'Sheffield United',
    'Leeds': 'Leeds',
    'West Bromwich Albion': 'West Brom',
    'Queens Park Rangers': 'QPR',
}

# Reverse mapping for lookup
reverse_mapping = {v: k for k, v in team_mapping.items()}

# Build xG lookup with multiple date formats
xg_lookup = {}
for _, row in xg_df.iterrows():
    h_team = row['h_team']
    a_team = row['a_team']
    date = row['date'].date()
    data = {
        'h_xG': row['h_xG'],
        'a_xG': row['a_xG'],
        'forecast_h': row['forecast_h'],
        'forecast_d': row['forecast_d'],
        'forecast_a': row['forecast_a']
    }
    xg_lookup[(h_team, a_team, date)] = data
    # Also add with +/- 1 day for timezone issues
    from datetime import timedelta
    xg_lookup[(h_team, a_team, date + timedelta(days=1))] = data
    xg_lookup[(h_team, a_team, date - timedelta(days=1))] = data

# Function to lookup xG
def get_xg(home, away, date):
    # Try direct lookup
    key = (home, away, date)
    if key in xg_lookup:
        return xg_lookup[key]
    
    # Try with mapping
    h_mapped = reverse_mapping.get(home, home)
    a_mapped = reverse_mapping.get(away, away)
    key = (h_mapped, a_mapped, date)
    if key in xg_lookup:
        return xg_lookup[key]
    
    return None

# Check how many matches we can find xG for
found = 0
not_found = []
for _, row in df.iterrows():
    xg = get_xg(row['HomeTeam'], row['AwayTeam'], row['Date'].date())
    if xg:
        found += 1
    else:
        not_found.append((row['HomeTeam'], row['AwayTeam'], row['Date'].date()))

print(f"\nxG data found: {found}/{len(df)} matches ({found/len(df)*100:.1f}%)")
if not_found[:5]:
    print(f"Sample not found: {not_found[:5]}")

# Check what teams are in Understat but not matched
understat_teams = set(xg_df['h_team'].unique()) | set(xg_df['a_team'].unique())
main_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
print(f"\nUnderstat teams: {sorted(understat_teams)[:10]}...")
print(f"Main data teams: {sorted(main_teams)[:10]}...")
