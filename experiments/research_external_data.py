"""
RESEARCH: What other data could improve the model?

CURRENT DATA:
1. Match results (goals, shots, fouls, corners, cards)
2. xG from shot positions
3. Non-league matches (FA Cup, League Cup, Europa, Champions League)
4. Lineup data

POTENTIAL NEW DATA SOURCES (that we can calculate from existing data):

1. LEAGUE POSITION / POINTS
   - Current points
   - Position in table
   - Points per game
   - Goal difference

2. STREAK PATTERNS
   - Consecutive wins/losses/draws
   - Unbeaten streak
   - Winless streak

3. SCORING PATTERNS
   - First goal scorer rate
   - Late goals (75+ min)
   - Clean sheet streak

4. HOME/AWAY SPECIFIC
   - Home form vs Away form
   - Home goals vs Away goals

5. OPPONENT STRENGTH ADJUSTED
   - Performance vs top 6
   - Performance vs bottom 6

Let's test these!
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

df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
CLASSES = ["A", "D", "H"]

HOME_ADV, K_FACTOR, ELO_DIVISOR, C = 82, 28, 640, 0.15

train_df = df[df["season"].isin(train_seasons)]
defaults = {
    'home_win_rate': (train_df['FTR'] == 'H').sum() / len(train_df),
    'draw_rate': (train_df['FTR'] == 'D').sum() / len(train_df),
    'avg_home_goals': train_df['FTHG'].mean(),
    'avg_away_goals': train_df['FTAG'].mean(),
}

def create_features_with_league_position(df, defaults):
    elo = {}
    team_stats = {}
    league_table = {}  # {season: {team: {'points': x, 'gd': y, 'played': z}}}
    features = []
    
    for _, m in df.iterrows():
        season, match_date = m["season"], m["Date"]
        home, away = m["HomeTeam"], m["AwayTeam"]
        
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)
        elo_diff = h_elo - a_elo + HOME_ADV
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV) / ELO_DIVISOR))
        
        h_hist = team_stats.get(home, {"results": [], "goals": [], "ga": [], "home_results": [], "away_results": []})
        a_hist = team_stats.get(away, {"results": [], "goals": [], "ga": [], "home_results": [], "away_results": []})
        
        # League position features
        if season not in league_table:
            league_table[season] = {}
        
        h_table = league_table[season].get(home, {'points': 0, 'gd': 0, 'played': 0})
        a_table = league_table[season].get(away, {'points': 0, 'gd': 0, 'played': 0})
        
        h_ppg = h_table['points'] / h_table['played'] if h_table['played'] > 0 else 1.3
        a_ppg = a_table['points'] / a_table['played'] if a_table['played'] > 0 else 1.3
        
        # Streak features
        def get_streak(results, streak_type='win'):
            if not results: return 0
            streak = 0
            for r in reversed(results):
                if streak_type == 'win' and r == 'W':
                    streak += 1
                elif streak_type == 'unbeaten' and r in ['W', 'D']:
                    streak += 1
                elif streak_type == 'winless' and r in ['L', 'D']:
                    streak += 1
                else:
                    break
            return streak
        
        h_win_streak = get_streak(h_hist["results"], 'win')
        a_win_streak = get_streak(a_hist["results"], 'win')
        h_unbeaten = get_streak(h_hist["results"], 'unbeaten')
        a_unbeaten = get_streak(a_hist["results"], 'unbeaten')
        
        # Home/Away specific form
        h_home_wins = sum(1 for r in h_hist["home_results"][-5:] if r == 'W') / max(len(h_hist["home_results"][-5:]), 1)
        a_away_wins = sum(1 for r in a_hist["away_results"][-5:] if r == 'W') / max(len(a_hist["away_results"][-5:]), 1)
        
        feat = {
            "season": season, "FTR": m["FTR"],
            "elo_diff": elo_diff,
            "ppg_diff": h_ppg - a_ppg,
            "gd_diff": h_table['gd'] - a_table['gd'],
            "win_streak_diff": h_win_streak - a_win_streak,
            "unbeaten_diff": h_unbeaten - a_unbeaten,
            "home_form": h_home_wins,
            "away_form": a_away_wins,
            "home_away_diff": h_home_wins - a_away_wins,
        }
        features.append(feat)
        
        # Update
        actual = 1 if m["FTR"] == "H" else (0 if m["FTR"] == "A" else 0.5)
        elo[home] = h_elo + K_FACTOR * (actual - exp_h)
        elo[away] = a_elo + K_FACTOR * ((1 - actual) - (1 - exp_h))
        
        # Update league table
        h_goals, a_goals = m["FTHG"], m["FTAG"]
        if home not in league_table[season]:
            league_table[season][home] = {'points': 0, 'gd': 0, 'played': 0}
        if away not in league_table[season]:
            league_table[season][away] = {'points': 0, 'gd': 0, 'played': 0}
        
        league_table[season][home]['played'] += 1
        league_table[season][away]['played'] += 1
        league_table[season][home]['gd'] += h_goals - a_goals
        league_table[season][away]['gd'] += a_goals - h_goals
        
        if m["FTR"] == "H":
            league_table[season][home]['points'] += 3
        elif m["FTR"] == "A":
            league_table[season][away]['points'] += 3
        else:
            league_table[season][home]['points'] += 1
            league_table[season][away]['points'] += 1
        
        # Update team stats
        for team, is_home in [(home, True), (away, False)]:
            if team not in team_stats:
                team_stats[team] = {"results": [], "goals": [], "ga": [], "home_results": [], "away_results": []}
            result = 'W' if (is_home and m['FTR'] == 'H') or (not is_home and m['FTR'] == 'A') else ('D' if m['FTR'] == 'D' else 'L')
            team_stats[team]["results"].append(result)
            if is_home:
                team_stats[team]["home_results"].append(result)
            else:
                team_stats[team]["away_results"].append(result)
    
    return pd.DataFrame(features)

print("Testing league position and streak features...")
feat_df = create_features_with_league_position(df, defaults)

train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

# Test different feature combinations
feature_sets = [
    (["elo_diff"], "Baseline"),
    (["elo_diff", "ppg_diff"], "+ ppg_diff"),
    (["elo_diff", "gd_diff"], "+ gd_diff"),
    (["elo_diff", "win_streak_diff"], "+ win_streak"),
    (["elo_diff", "unbeaten_diff"], "+ unbeaten"),
    (["elo_diff", "home_away_diff"], "+ home_away_form"),
    (["elo_diff", "ppg_diff", "win_streak_diff"], "+ ppg + streak"),
    (["elo_diff", "ppg_diff", "gd_diff", "win_streak_diff", "home_away_diff"], "All new"),
]

print("\n" + "="*60)
for cols, name in feature_sets:
    X_train = feat_df[train_mask][cols].fillna(0).values
    y_train = feat_df[train_mask]["FTR"].values
    X_test = feat_df[test_mask][cols].fillna(0).values
    y_test = feat_df[test_mask]["FTR"].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(X_train_s, y_train)
    
    proba = model.predict_proba(X_test_s)
    proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
    
    loss = log_loss(y_test, proba_ord, labels=CLASSES)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    
    print(f"{name}: Loss={loss:.4f}, Acc={acc*100:.1f}%")
