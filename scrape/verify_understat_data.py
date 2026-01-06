import pandas as pd
import numpy as np

print("VERIFY UNDERSTAT SHOT DATA")
print("=" * 60)

# Load data
df = pd.read_csv("understat_data/epl_shots_all.csv")
df_matches = pd.read_csv("understat_data/epl_matches_all.csv")

print(f"\n1. BASIC INFO")
print("-" * 60)
print(f"Total shots: {len(df):,}")
print(f"Total matches: {len(df_matches):,}")
print(f"Columns: {list(df.columns)}")

print(f"\n2. SHOTS PER SEASON")
print("-" * 60)
season_stats = df.groupby('season').agg({
    'id': 'count',
    'match_id': 'nunique'
}).rename(columns={'id': 'shots', 'match_id': 'matches'})
season_stats['shots_per_match'] = (season_stats['shots'] / season_stats['matches']).round(1)

for season, row in season_stats.iterrows():
    status = "OK" if row['matches'] == 380 else f"MISSING {380 - row['matches']}"
    print(f"  {season}/{season+1}: {row['shots']:,} shots, {row['matches']} matches ({status}), {row['shots_per_match']} shots/match")

print(f"\n3. DATA QUALITY CHECK")
print("-" * 60)

# Check for missing values
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print("Missing values:")
    for col, count in missing_cols.items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
else:
    print("No missing values in key columns: OK")

# Check X, Y ranges
print(f"\nX range: {df['X'].min():.3f} - {df['X'].max():.3f} (expected: 0-1)")
print(f"Y range: {df['Y'].min():.3f} - {df['Y'].max():.3f} (expected: 0-1)")

# Check result values
print(f"\n4. RESULT DISTRIBUTION")
print("-" * 60)
result_counts = df['result'].value_counts()
total = len(df)
for result, count in result_counts.items():
    pct = count / total * 100
    print(f"  {result}: {count:,} ({pct:.1f}%)")

goals = (df['result'] == 'Goal').sum()
print(f"\nGoal rate: {goals/total*100:.2f}% ({goals:,} goals)")

print(f"\n5. SHOT TYPE DISTRIBUTION")
print("-" * 60)
for shot_type, count in df['shotType'].value_counts().items():
    print(f"  {shot_type}: {count:,} ({count/total*100:.1f}%)")

print(f"\n6. SITUATION DISTRIBUTION")
print("-" * 60)
for situation, count in df['situation'].value_counts().items():
    print(f"  {situation}: {count:,} ({count/total*100:.1f}%)")

print(f"\n7. VERIFY GOALS PER SEASON")
print("-" * 60)
goals_per_season = df[df['result'] == 'Goal'].groupby('season').size()
for season, goals in goals_per_season.items():
    # EPL typically has 900-1100 goals per season
    status = "OK" if 800 <= goals <= 1200 else "CHECK"
    print(f"  {season}/{season+1}: {goals} goals ({status})")

print(f"\n8. SAMPLE DATA")
print("-" * 60)
print(df[['date', 'h_team', 'a_team', 'player', 'X', 'Y', 'result', 'shotType', 'situation']].head(5).to_string())

print(f"\n9. XG COMPARISON (Understat vs our calculation)")
print("-" * 60)
# Convert xG to float
df['xG'] = df['xG'].astype(float)
df['X'] = df['X'].astype(float)
df['Y'] = df['Y'].astype(float)

# Simple xG calculation based on distance
df['distance'] = np.sqrt((1 - df['X'])**2 + (0.5 - df['Y'])**2)
df['is_goal'] = (df['result'] == 'Goal').astype(int)

# Check correlation
print(f"Understat xG mean: {df['xG'].mean():.4f}")
print(f"Actual goal rate: {df['is_goal'].mean():.4f}")
print(f"Correlation (xG vs is_goal): {df['xG'].corr(df['is_goal']):.4f}")

# Check by xG bins
print("\nCalibration check (Understat xG):")
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
df['xG_bin'] = pd.cut(df['xG'], bins=bins)
calibration = df.groupby('xG_bin').agg({
    'xG': 'mean',
    'is_goal': 'mean',
    'id': 'count'
}).rename(columns={'id': 'count'})
print(calibration.to_string())

print("VERIFICATION COMPLETE")

