import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta

# Load data
df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

xg_df = pd.read_csv("understat_data/our_match_xg.csv")
xg_df['date'] = pd.to_datetime(xg_df['date'])

print("DEBUG XG INTEGRATION")
print("-" * 60)

# Check date alignment
print("1. CHECK DATE ALIGNMENT")
print("-" * 60)

# Sample from main data
main_sample = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'season']].head(10)
print("Main data sample:")
print(main_sample.to_string())

# Sample from xG data
xg_sample = xg_df[['date', 'h_team', 'a_team', 'h_goals', 'a_goals', 'h_xG', 'a_xG', 'season']].head(10)
print("\nxG data sample:")
print(xg_sample.to_string())

# Check if seasons align
print("\n2. CHECK SEASON ALIGNMENT")
print("-" * 60)
print(f"Main data seasons: {sorted(df['season'].unique())}")
print(f"xG data seasons: {sorted(xg_df['season'].unique())}")

# The issue: main data uses "2015-2016" format, xG uses 2014, 2015, etc.
# xG season 2014 = 2014/15 season = main data 2015-2016? NO!
# xG season 2015 = 2015/16 season = main data 2015-2016? YES!

print("\n3. CHECK SEASON MAPPING")
print("-" * 60)
# Main data 2015-2016 starts Aug 2015
main_2015 = df[df['season'] == '2015-2016']
print(f"Main 2015-2016: {main_2015['Date'].min()} to {main_2015['Date'].max()}")

# xG data season 2015 
xg_2015 = xg_df[xg_df['season'] == 2015]
print(f"xG season 2015: {xg_2015['date'].min()} to {xg_2015['date'].max()}")

# xG data season 2014
xg_2014 = xg_df[xg_df['season'] == 2014]
print(f"xG season 2014: {xg_2014['date'].min()} to {xg_2014['date'].max()}")

# So xG season 2015 = main 2015-2016 (both start Aug 2015)
# And xG season 2014 = main 2014-2015 (but we don't have main 2014-2015!)

print("\n4. CHECK TEAM NAME MATCHING")
print("-" * 60)
main_teams = set(df['HomeTeam'].unique())
xg_teams = set(xg_df['h_team'].unique())

print(f"Main teams: {sorted(main_teams)}")
print(f"\nxG teams: {sorted(xg_teams)}")

# Find mismatches
only_main = main_teams - xg_teams
only_xg = xg_teams - main_teams
print(f"\nOnly in main: {only_main}")
print(f"Only in xG: {only_xg}")

print("\n5. TRY DIRECT MATCH")
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
    data = {'h_xG': row['h_xG'], 'a_xG': row['a_xG'], 'h_goals': row['h_goals'], 'a_goals': row['a_goals']}
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

# Check match rate and verify goals match
found = 0
goals_match = 0
goals_mismatch = []

for _, row in df.iterrows():
    xg = get_xg(row['HomeTeam'], row['AwayTeam'], row['Date'].date())
    if xg:
        found += 1
        if xg['h_goals'] == row['FTHG'] and xg['a_goals'] == row['FTAG']:
            goals_match += 1
        else:
            goals_mismatch.append({
                'date': row['Date'],
                'home': row['HomeTeam'],
                'away': row['AwayTeam'],
                'main_goals': f"{row['FTHG']}-{row['FTAG']}",
                'xg_goals': f"{xg['h_goals']}-{xg['a_goals']}"
            })

print(f"Found: {found}/{len(df)}")
print(f"Goals match: {goals_match}/{found}")
print(f"Goals mismatch: {len(goals_mismatch)}")

if goals_mismatch:
    print("\nSample mismatches:")
    for m in goals_mismatch[:5]:
        print(f"  {m['date']}: {m['home']} vs {m['away']}: main={m['main_goals']}, xg={m['xg_goals']}")
