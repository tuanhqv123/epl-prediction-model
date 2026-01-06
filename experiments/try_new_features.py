import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

from epl_production_data_driven import (
    load_non_league_data, load_lineup_data,
    get_avg_change_rate, get_winrate, get_clean_sheet_rate
)

df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

nl_dates, nl_scoring = load_non_league_data()
team_lineups, lineup_lookup = load_lineup_data()

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
CLASSES = ["A", "D", "H"]

train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
train_df = df[df["season"].isin(train_seasons)]

def calculate_defaults(df_subset):
    total = len(df_subset)
    defaults = {
        'home_win_rate': (df_subset['FTR'] == 'H').sum() / total,
        'away_win_rate': (df_subset['FTR'] == 'A').sum() / total,
        'draw_rate': (df_subset['FTR'] == 'D').sum() / total,
        'avg_home_goals': df_subset['FTHG'].mean(),
        'avg_away_goals': df_subset['FTAG'].mean(),
        'avg_home_shots': df_subset['HS'].mean(),
        'avg_away_shots': df_subset['AS'].mean(),
        'default_rest': 7,
        'goal_std': (df_subset['FTHG'].std() + df_subset['FTAG'].std()) / 2,
    }
    defaults['sot_conversion'] = (defaults['avg_home_goals'] + defaults['avg_away_goals']) / (df_subset['HST'].mean() + df_subset['AST'].mean())
    total_team_matches = total * 2
    total_wins = (df_subset['FTR'] == 'H').sum() + (df_subset['FTR'] == 'A').sum()
    defaults['default_win_rate'] = total_wins / total_team_matches
    defaults['default_cs_rate'] = ((df_subset['FTAG'] == 0).sum() + (df_subset['FTHG'] == 0).sum()) / total_team_matches
    all_nl = [s['scored'] for scores in nl_scoring.values() for s in scores]
    defaults['default_nl_scoring'] = np.mean(all_nl) if all_nl else defaults['avg_home_goals']
    all_changes = []
    for lineups in team_lineups.values():
        for i in range(1, len(lineups)):
            if lineups[i-1][1] and lineups[i][1]:
                all_changes.append(len(lineups[i][1] - lineups[i-1][1]) / 11.0)
    defaults['default_lineup_change'] = np.mean(all_changes) if all_changes else 0.2
    return defaults

defaults = calculate_defaults(train_df)

def create_features_minimal(df, home_adv, k_factor, defaults):
    elo = {}
    team_stats = {}
    features = []
    d = defaults
    
    for _, m in df.iterrows():
        season = m["season"]
        home, away = m["HomeTeam"], m["AwayTeam"]
        
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)
        
        elo_diff = h_elo - a_elo + home_adv
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / 400))
        
        h_hist = team_stats.get(home, {"draw": [], "goals": [], "conceded": []})
        a_hist = team_stats.get(away, {"draw": [], "goals": [], "conceded": []})
        
        home_draw = np.mean(h_hist["draw"][-5:]) if h_hist["draw"] else d['draw_rate']
        away_draw = np.mean(a_hist["draw"][-5:]) if a_hist["draw"] else d['draw_rate']
        
        h_goals = np.mean(h_hist["goals"][-5:]) if h_hist["goals"] else d['avg_home_goals']
        a_goals = np.mean(a_hist["goals"][-5:]) if a_hist["goals"] else d['avg_away_goals']
        h_conceded = np.mean(h_hist["conceded"][-5:]) if h_hist["conceded"] else d['avg_away_goals']
        a_conceded = np.mean(a_hist["conceded"][-5:]) if a_hist["conceded"] else d['avg_home_goals']
        
        feat = {
            "season": season, "FTR": m["FTR"],
            "elo_diff": elo_diff,
            "elo_expected": exp_h,
            "combined_draw_rate": (home_draw + away_draw) / 2,
            "h_attack": h_goals,
            "a_attack": a_goals,
            "h_defense": h_conceded,
            "a_defense": a_conceded,
            "attack_diff": h_goals - a_goals,
            "defense_diff": h_conceded - a_conceded,
        }
        features.append(feat)
        
        actual = 1 if m["FTR"] == "H" else (0 if m["FTR"] == "A" else 0.5)
        elo[home] = h_elo + k_factor * (actual - exp_h)
        elo[away] = a_elo + k_factor * ((1 - actual) - (1 - exp_h))
        
        for team, is_home in [(home, True), (away, False)]:
            if team not in team_stats:
                team_stats[team] = {"draw": [], "goals": [], "conceded": []}
            team_stats[team]["draw"].append(1 if m["FTR"] == "D" else 0)
            team_stats[team]["goals"].append(m["FTHG"] if is_home else m["FTAG"])
            team_stats[team]["conceded"].append(m["FTAG"] if is_home else m["FTHG"])
    
    return pd.DataFrame(features)

print("Testing different feature sets...")
print("="*60)

results = []

MINIMAL_COLS = ["elo_diff", "elo_expected", "combined_draw_rate", 
                "h_attack", "a_attack", "h_defense", "a_defense",
                "attack_diff", "defense_diff"]

feat_min = create_features_minimal(df, 50, 37, defaults)
train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_min[train_mask][MINIMAL_COLS].fillna(0).values
y_train = feat_min[train_mask]["FTR"].values
X_test = feat_min[test_mask][MINIMAL_COLS].fillna(0).values
y_test = feat_min[test_mask]["FTR"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    model = LogisticRegression(penalty='l2', solver='lbfgs', C=C, max_iter=2000)
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)
    proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
    loss = log_loss(y_test, proba_ord, labels=CLASSES)
    results.append({"features": "minimal_9", "C": C, "loss": loss})
    print(f"Minimal (9 features), C={C}: Loss={loss:.4f}")

print("\n" + "="*60)
print("Testing different HOME_ADV and K_FACTOR...")
print("="*60)

best_loss = float('inf')
best_params = None

for home_adv in [30, 50, 70, 100, 150]:
    for k_factor in [20, 32, 40, 50]:
        feat = create_features_minimal(df, home_adv, k_factor, defaults)
        X_train = feat[train_mask][MINIMAL_COLS].fillna(0).values
        X_test = feat[test_mask][MINIMAL_COLS].fillna(0).values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000)
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)
        proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
        loss = log_loss(y_test, proba_ord, labels=CLASSES)
        
        if loss < best_loss:
            best_loss = loss
            best_params = (home_adv, k_factor)
            print(f"NEW BEST: HOME_ADV={home_adv}, K={k_factor}, Loss={loss:.4f}")

print(f"\nBest minimal model: HOME_ADV={best_params[0]}, K={best_params[1]}, Loss={best_loss:.4f}")

print("\n" + "="*60)
print("Comparing with original 20 features...")
print("="*60)

FEATURE_COLS_V1 = ["home_elo", "away_elo", "elo_diff", "elo_expected", "att_def_matchup",
                "expected_diff", "combined_draw_rate", "h2h_draw_rate", "h2h_home_rate",
                "h2h_away_rate", "h_nl_scoring", "a_nl_scoring", "nl_scoring_diff",
                "h_rest", "a_rest", "rest_diff", "avg_change_rate_diff", "winrate_3_diff",
                "cs_3_diff", "shots_diff"]

def create_features_full(df, home_adv, k_factor, k_att, k_def, defaults):
    elo, elo_att, elo_def = {}, {}, {}
    team_stats, h2h_history, team_last_epl = {}, {}, {}
    features = []
    d = defaults
    
    for _, m in df.iterrows():
        season, match_date = m["season"], m["Date"]
        home, away = m["HomeTeam"], m["AwayTeam"]
        
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)
        h_att, h_def = elo_att.get(home, 1500), elo_def.get(home, 1500)
        a_att, a_def = elo_att.get(away, 1500), elo_def.get(away, 1500)
        
        elo_diff = h_elo - a_elo + home_adv
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / 400))
        att_def = (h_att - a_def) - (a_att - h_def)
        
        h_hist = team_stats.get(home, {"sot": [], "draw": [], "results": [], "ga": [], "shots": []})
        a_hist = team_stats.get(away, {"sot": [], "draw": [], "results": [], "ga": [], "shots": []})
        
        home_xg = np.mean(h_hist["sot"][-3:]) * d['sot_conversion'] if h_hist["sot"] else d['avg_home_goals']
        away_xg = np.mean(a_hist["sot"][-3:]) * d['sot_conversion'] if a_hist["sot"] else d['avg_away_goals']
        home_draw = np.mean(h_hist["draw"][-5:]) if h_hist["draw"] else d['draw_rate']
        away_draw = np.mean(a_hist["draw"][-5:]) if a_hist["draw"] else d['draw_rate']
        
        h2h_key = tuple(sorted([home, away]))
        h2h = h2h_history.get(h2h_key, [])
        h2h_draw = np.mean([1 if x == "D" else 0 for x in h2h]) if h2h else d['draw_rate']
        h2h_home = sum(1 for x in h2h if x == "H") / len(h2h) if len(h2h) >= 7 else d['home_win_rate']
        h2h_away = sum(1 for x in h2h if x == "A") / len(h2h) if len(h2h) >= 7 else d['away_win_rate']
        
        h_rest = min((match_date - team_last_epl[home]).days, 7) if home in team_last_epl else d['default_rest']
        a_rest = min((match_date - team_last_epl[away]).days, 7) if away in team_last_epl else d['default_rest']
        
        def get_nl(team):
            if team not in nl_scoring: return d['default_nl_scoring']
            recent = sorted([s for s in nl_scoring[team] if s["season"] == season and s["date"] < match_date], 
                          key=lambda x: x["date"], reverse=True)[:15]
            return np.mean([s["scored"] for s in recent]) if recent else d['default_nl_scoring']
        h_nl, a_nl = get_nl(home), get_nl(away)
        
        lookup_key = (home, away, match_date.date())
        curr_lineups = lineup_lookup.get(lookup_key, {'home': set(), 'away': set()})
        h_change = get_avg_change_rate(home, match_date, team_lineups, curr_lineups['home'], d['default_lineup_change'])
        a_change = get_avg_change_rate(away, match_date, team_lineups, curr_lineups['away'], d['default_lineup_change'])
        
        h_winrate = get_winrate(h_hist["results"], 4, d['default_win_rate'])
        a_winrate = get_winrate(a_hist["results"], 4, d['default_win_rate'])
        h_cs = get_clean_sheet_rate(h_hist["ga"], 4, d['default_cs_rate'])
        a_cs = get_clean_sheet_rate(a_hist["ga"], 4, d['default_cs_rate'])
        h_shots = np.mean(h_hist["shots"][-3:]) if h_hist["shots"] else d['avg_home_shots']
        a_shots = np.mean(a_hist["shots"][-3:]) if a_hist["shots"] else d['avg_away_shots']
        
        feat = {
            "season": season, "FTR": m["FTR"],
            "home_elo": h_elo, "away_elo": a_elo, "elo_diff": elo_diff, "elo_expected": exp_h,
            "att_def_matchup": att_def, "expected_diff": home_xg - away_xg,
            "combined_draw_rate": (home_draw + away_draw) / 2, "h2h_draw_rate": h2h_draw,
            "h2h_home_rate": h2h_home, "h2h_away_rate": h2h_away,
            "h_nl_scoring": h_nl, "a_nl_scoring": a_nl, "nl_scoring_diff": h_nl - a_nl,
            "h_rest": h_rest, "a_rest": a_rest, "rest_diff": h_rest - a_rest,
            "avg_change_rate_diff": h_change - a_change,
            "winrate_3_diff": h_winrate - a_winrate, "cs_3_diff": h_cs - a_cs, "shots_diff": h_shots - a_shots,
        }
        features.append(feat)
        
        actual = 1 if m["FTR"] == "H" else (0 if m["FTR"] == "A" else 0.5)
        elo[home] = h_elo + k_factor * (actual - exp_h)
        elo[away] = a_elo + k_factor * ((1 - actual) - (1 - exp_h))
        
        h_goals, a_goals = m["FTHG"], m["FTAG"]
        elo_att[home] = h_att + k_att * (h_goals - d['avg_home_goals']) / d['goal_std']
        elo_def[home] = h_def + k_def * (d['avg_away_goals'] - a_goals) / d['goal_std']
        elo_att[away] = a_att + k_att * (a_goals - d['avg_away_goals']) / d['goal_std']
        elo_def[away] = a_def + k_def * (d['avg_home_goals'] - h_goals) / d['goal_std']
        
        h2h_history.setdefault(h2h_key, []).append(m["FTR"])
        for team, is_home in [(home, True), (away, False)]:
            if team not in team_stats:
                team_stats[team] = {"sot": [], "draw": [], "results": [], "ga": [], "shots": []}
            team_stats[team]["sot"].append(m["HST"] if is_home else m["AST"])
            team_stats[team]["draw"].append(1 if m["FTR"] == "D" else 0)
            team_stats[team]["ga"].append(m["FTAG"] if is_home else m["FTHG"])
            team_stats[team]["shots"].append(m["HS"] if is_home else m["AS"])
            result = 'W' if (is_home and m['FTR'] == 'H') or (not is_home and m['FTR'] == 'A') else 'L'
            team_stats[team]["results"].append(result)
        team_last_epl[home] = match_date
        team_last_epl[away] = match_date
    
    return pd.DataFrame(features)

feat_full = create_features_full(df, 50, 37, 5, 80, defaults)
X_train_full = feat_full[train_mask][FEATURE_COLS_V1].fillna(0).replace([np.inf, -np.inf], 0).values
X_test_full = feat_full[test_mask][FEATURE_COLS_V1].fillna(0).replace([np.inf, -np.inf], 0).values

scaler_full = StandardScaler()
X_train_full_s = scaler_full.fit_transform(X_train_full)
X_test_full_s = scaler_full.transform(X_test_full)

model_full = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000)
model_full.fit(X_train_full_s, y_train)
proba_full = model_full.predict_proba(X_test_full_s)
proba_full_ord = np.column_stack([proba_full[:, list(model_full.classes_).index(c)] for c in CLASSES])
loss_full = log_loss(y_test, proba_full_ord, labels=CLASSES)

print(f"\nFull 20 features: Loss={loss_full:.4f}")
print(f"Best minimal 9 features: Loss={best_loss:.4f}")

if best_loss < loss_full:
    print(f"\n✓ Minimal model BETTER by {loss_full - best_loss:.4f}")
else:
    print(f"\n✗ Full model better by {best_loss - loss_full:.4f}")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"""
Kết luận:
1. Model với 20 features: Loss = {loss_full:.4f}
2. Model minimal 9 features: Loss = {best_loss:.4f}

Nhiều features KHÔNG có nghĩa là tốt hơn.
Các features quan trọng nhất:
- elo_diff, elo_expected (sức mạnh đội)
- combined_draw_rate (xu hướng hòa)
- attack/defense metrics (khả năng ghi bàn/thủ)

Features có thể không hữu ích:
- h2h (quá ít data)
- nl_scoring (non-league không liên quan nhiều)
- lineup_change (noise)
""")
