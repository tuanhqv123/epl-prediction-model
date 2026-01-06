"""
RESEARCH MORE FEATURES TO IMPROVE MODEL
========================================
Current: Loss=0.9310, Acc=58.2%
Target: Push higher

Data sources we have:
1. Match stats (goals, shots, fouls, corners, cards)
2. xG from shots (X, Y, shotType, situation, lastAction)
3. Non-league matches
4. Lineup data

Ideas to explore:
1. Cards (yellow/red) - discipline patterns
2. Corners - attacking pressure
3. Shot accuracy (SOT/Shots ratio)
4. Goal efficiency (Goals/SOT)
5. Momentum (consecutive wins/losses)
6. Season position/points
7. Time since last goal conceded
8. Big match performance (vs top 6)
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

print("AVAILABLE COLUMNS:")
print(df.columns.tolist())

# Columns we haven't used yet:
# HC/AC - Corners
# HY/AY - Yellow cards  
# HR/AR - Red cards
# HTHG/HTAG - Half-time goals

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
train_df = df[df["season"].isin(train_seasons)]

# Calculate defaults
total = len(train_df)
defaults = {
    'home_win_rate': (train_df['FTR'] == 'H').sum() / total,
    'draw_rate': (train_df['FTR'] == 'D').sum() / total,
    'avg_home_goals': train_df['FTHG'].mean(),
    'avg_away_goals': train_df['FTAG'].mean(),
    'avg_corners': (train_df['HC'].mean() + train_df['AC'].mean()) / 2,
    'avg_yellows': (train_df['HY'].mean() + train_df['AY'].mean()) / 2,
    'avg_shots': (train_df['HS'].mean() + train_df['AS'].mean()) / 2,
    'avg_sot': (train_df['HST'].mean() + train_df['AST'].mean()) / 2,
}

print(f"\nDEFAULTS FROM TRAINING DATA:")
for k, v in defaults.items():
    print(f"  {k}: {v:.4f}")

# Test new features
print("\n" + "="*60)
print("TESTING NEW FEATURES")
print("="*60)

HOME_ADV, K_FACTOR, ELO_DIVISOR, C = 82, 28, 640, 0.15
CLASSES = ["A", "D", "H"]

def test_features(feature_list, name):
    elo = {}
    team_stats = {}
    features = []
    
    for _, m in df.iterrows():
        home, away = m["HomeTeam"], m["AwayTeam"]
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)
        elo_diff = h_elo - a_elo + HOME_ADV
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV) / ELO_DIVISOR))
        
        h_hist = team_stats.get(home, {"corners": [], "yellows": [], "reds": [], "shots": [], "sot": [], "goals": []})
        a_hist = team_stats.get(away, {"corners": [], "yellows": [], "reds": [], "shots": [], "sot": [], "goals": []})
        
        feat = {"season": m["season"], "FTR": m["FTR"], "elo_diff": elo_diff}
        
        if "corners_diff" in feature_list:
            h_corners = np.mean(h_hist["corners"][-5:]) if h_hist["corners"] else defaults['avg_corners']
            a_corners = np.mean(a_hist["corners"][-5:]) if a_hist["corners"] else defaults['avg_corners']
            feat["corners_diff"] = h_corners - a_corners
        
        if "yellows_diff" in feature_list:
            h_yellows = np.mean(h_hist["yellows"][-5:]) if h_hist["yellows"] else defaults['avg_yellows']
            a_yellows = np.mean(a_hist["yellows"][-5:]) if a_hist["yellows"] else defaults['avg_yellows']
            feat["yellows_diff"] = h_yellows - a_yellows
        
        if "discipline_ratio" in feature_list:
            h_yellows = np.mean(h_hist["yellows"][-5:]) if h_hist["yellows"] else defaults['avg_yellows']
            a_yellows = np.mean(a_hist["yellows"][-5:]) if a_hist["yellows"] else defaults['avg_yellows']
            feat["discipline_ratio"] = (h_yellows + 0.1) / (a_yellows + 0.1)
        
        if "shot_accuracy_diff" in feature_list:
            h_shots = h_hist["shots"][-5:] if h_hist["shots"] else [defaults['avg_shots']]
            h_sot = h_hist["sot"][-5:] if h_hist["sot"] else [defaults['avg_sot']]
            a_shots = a_hist["shots"][-5:] if a_hist["shots"] else [defaults['avg_shots']]
            a_sot = a_hist["sot"][-5:] if a_hist["sot"] else [defaults['avg_sot']]
            h_acc = sum(h_sot) / (sum(h_shots) + 0.1)
            a_acc = sum(a_sot) / (sum(a_shots) + 0.1)
            feat["shot_accuracy_diff"] = h_acc - a_acc
        
        if "goal_efficiency_diff" in feature_list:
            h_goals = h_hist["goals"][-5:] if h_hist["goals"] else [defaults['avg_home_goals']]
            h_sot = h_hist["sot"][-5:] if h_hist["sot"] else [defaults['avg_sot']]
            a_goals = a_hist["goals"][-5:] if a_hist["goals"] else [defaults['avg_away_goals']]
            a_sot = a_hist["sot"][-5:] if a_hist["sot"] else [defaults['avg_sot']]
            h_eff = sum(h_goals) / (sum(h_sot) + 0.1)
            a_eff = sum(a_goals) / (sum(a_sot) + 0.1)
            feat["goal_efficiency_diff"] = h_eff - a_eff
        
        if "momentum" in feature_list:
            h_recent = h_hist.get("results", [])[-3:]
            a_recent = a_hist.get("results", [])[-3:]
            h_momentum = sum(1 for r in h_recent if r == 'W') - sum(1 for r in h_recent if r == 'L')
            a_momentum = sum(1 for r in a_recent if r == 'W') - sum(1 for r in a_recent if r == 'L')
            feat["momentum"] = h_momentum - a_momentum
        
        features.append(feat)
        
        # Update
        actual = 1 if m["FTR"] == "H" else (0 if m["FTR"] == "A" else 0.5)
        elo[home] = h_elo + K_FACTOR * (actual - exp_h)
        elo[away] = a_elo + K_FACTOR * ((1 - actual) - (1 - exp_h))
        
        for team, is_home in [(home, True), (away, False)]:
            if team not in team_stats:
                team_stats[team] = {"corners": [], "yellows": [], "reds": [], "shots": [], "sot": [], "goals": [], "results": []}
            team_stats[team]["corners"].append(m["HC"] if is_home else m["AC"])
            team_stats[team]["yellows"].append(m["HY"] if is_home else m["AY"])
            team_stats[team]["reds"].append(m["HR"] if is_home else m["AR"])
            team_stats[team]["shots"].append(m["HS"] if is_home else m["AS"])
            team_stats[team]["sot"].append(m["HST"] if is_home else m["AST"])
            team_stats[team]["goals"].append(m["FTHG"] if is_home else m["FTAG"])
            result = 'W' if (is_home and m['FTR'] == 'H') or (not is_home and m['FTR'] == 'A') else 'L'
            team_stats[team]["results"].append(result)
    
    feat_df = pd.DataFrame(features)
    cols = ["elo_diff"] + [f for f in feature_list if f in feat_df.columns]
    
    train_mask = df["season"].isin(train_seasons).values
    test_mask = df["season"].isin(test_seasons).values
    
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
    return loss

# Baseline
baseline = test_features([], "Baseline (elo only)")

# Test each new feature
test_features(["corners_diff"], "+ corners_diff")
test_features(["yellows_diff"], "+ yellows_diff")
test_features(["discipline_ratio"], "+ discipline_ratio")
test_features(["shot_accuracy_diff"], "+ shot_accuracy_diff")
test_features(["goal_efficiency_diff"], "+ goal_efficiency_diff")
test_features(["momentum"], "+ momentum")

# Combinations
test_features(["corners_diff", "shot_accuracy_diff"], "+ corners + shot_acc")
test_features(["momentum", "goal_efficiency_diff"], "+ momentum + goal_eff")
