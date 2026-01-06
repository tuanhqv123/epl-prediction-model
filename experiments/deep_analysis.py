import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
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

print("="*70)
print("DEEP ANALYSIS: Tìm vấn đề và cải thiện")
print("="*70)

print("\n" + "="*70)
print("PHẦN 1: PHÂN TÍCH VẤN ĐỀ DRAW")
print("="*70)

train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
train_df = df[df["season"].isin(train_seasons)]
test_df = df[df["season"].isin(test_seasons)]

print("\n1.1 Draw rate theo season:")
for s in SEASONS:
    s_df = df[df["season"] == s]
    draw_rate = (s_df["FTR"] == "D").mean()
    print(f"   {s}: {draw_rate*100:.1f}%")

print("\n1.2 Draw có pattern gì không?")
test_draws = test_df[test_df["FTR"] == "D"]
print(f"   Total draws in test: {len(test_draws)}")

goal_diff_draws = (test_draws["FTHG"] - test_draws["FTAG"]).abs().mean()
print(f"   Avg goal diff in draws: {goal_diff_draws:.2f} (should be 0)")

print("\n1.3 Khi nào Draw xảy ra?")
for s in test_seasons:
    s_df = test_df[test_df["season"] == s]
    draws = s_df[s_df["FTR"] == "D"]
    print(f"   {s}:")
    print(f"     Draws: {len(draws)}/{len(s_df)} ({len(draws)/len(s_df)*100:.1f}%)")
    if len(draws) > 0:
        avg_home_goals = draws["FTHG"].mean()
        avg_away_goals = draws["FTAG"].mean()
        print(f"     Avg score in draws: {avg_home_goals:.1f}-{avg_away_goals:.1f}")

print("\n" + "="*70)
print("PHẦN 2: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

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

FEATURE_COLS_V1 = ["home_elo", "away_elo", "elo_diff", "elo_expected", "att_def_matchup",
                "expected_diff", "combined_draw_rate", "h2h_draw_rate", "h2h_home_rate",
                "h2h_away_rate", "h_nl_scoring", "a_nl_scoring", "nl_scoring_diff",
                "h_rest", "a_rest", "rest_diff", "avg_change_rate_diff", "winrate_3_diff",
                "cs_3_diff", "shots_diff"]

def create_features_v2(df, home_adv, k_factor, k_att, k_def, defaults, include_table=True):
    elo, elo_att, elo_def = {}, {}, {}
    team_stats, h2h_history, team_last_epl = {}, {}, {}
    season_table = {}
    features = []
    d = defaults
    
    for _, m in df.iterrows():
        season, match_date = m["season"], m["Date"]
        home, away = m["HomeTeam"], m["AwayTeam"]
        
        if season not in season_table:
            season_table[season] = {}
        
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)
        h_att, h_def = elo_att.get(home, 1500), elo_def.get(home, 1500)
        a_att, a_def = elo_att.get(away, 1500), elo_def.get(away, 1500)
        
        elo_diff = h_elo - a_elo + home_adv
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / 400))
        att_def = (h_att - a_def) - (a_att - h_def)
        
        h_hist = team_stats.get(home, {"sot": [], "draw": [], "results": [], "ga": [], "shots": [], "goals": []})
        a_hist = team_stats.get(away, {"sot": [], "draw": [], "results": [], "ga": [], "shots": [], "goals": []})
        
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
        
        h_table = season_table[season].get(home, {"points": 0, "played": 0, "gd": 0})
        a_table = season_table[season].get(away, {"points": 0, "played": 0, "gd": 0})
        
        h_ppg = h_table["points"] / h_table["played"] if h_table["played"] > 0 else 1.0
        a_ppg = a_table["points"] / a_table["played"] if a_table["played"] > 0 else 1.0
        
        h_goals_avg = np.mean(h_hist["goals"][-5:]) if h_hist["goals"] else d['avg_home_goals']
        a_goals_avg = np.mean(a_hist["goals"][-5:]) if a_hist["goals"] else d['avg_away_goals']
        h_conceded_avg = np.mean(h_hist["ga"][-5:]) if h_hist["ga"] else d['avg_away_goals']
        a_conceded_avg = np.mean(a_hist["ga"][-5:]) if a_hist["ga"] else d['avg_home_goals']
        
        elo_closeness = 1 - abs(exp_h - 0.5) * 2
        strength_similarity = 1 / (1 + abs(h_elo - a_elo) / 100)
        
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
            "h_ppg": h_ppg, "a_ppg": a_ppg, "ppg_diff": h_ppg - a_ppg,
            "h_goals_avg": h_goals_avg, "a_goals_avg": a_goals_avg,
            "h_conceded_avg": h_conceded_avg, "a_conceded_avg": a_conceded_avg,
            "elo_closeness": elo_closeness,
            "strength_similarity": strength_similarity,
            "both_draw_prone": home_draw * away_draw,
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
        
        if m["FTR"] == "H":
            h_pts, a_pts = 3, 0
        elif m["FTR"] == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1
        
        if home not in season_table[season]:
            season_table[season][home] = {"points": 0, "played": 0, "gd": 0}
        if away not in season_table[season]:
            season_table[season][away] = {"points": 0, "played": 0, "gd": 0}
        
        season_table[season][home]["points"] += h_pts
        season_table[season][home]["played"] += 1
        season_table[season][home]["gd"] += h_goals - a_goals
        season_table[season][away]["points"] += a_pts
        season_table[season][away]["played"] += 1
        season_table[season][away]["gd"] += a_goals - h_goals
        
        for team, is_home in [(home, True), (away, False)]:
            if team not in team_stats:
                team_stats[team] = {"sot": [], "draw": [], "results": [], "ga": [], "shots": [], "goals": []}
            team_stats[team]["sot"].append(m["HST"] if is_home else m["AST"])
            team_stats[team]["draw"].append(1 if m["FTR"] == "D" else 0)
            team_stats[team]["ga"].append(m["FTAG"] if is_home else m["FTHG"])
            team_stats[team]["shots"].append(m["HS"] if is_home else m["AS"])
            team_stats[team]["goals"].append(m["FTHG"] if is_home else m["FTAG"])
            result = 'W' if (is_home and m['FTR'] == 'H') or (not is_home and m['FTR'] == 'A') else 'L'
            team_stats[team]["results"].append(result)
        team_last_epl[home] = match_date
        team_last_epl[away] = match_date
    
    return pd.DataFrame(features)

defaults = calculate_defaults(train_df)

print("\n2.1 Testing với features gốc:")
HOME_ADV, K_FACTOR, K_ATT, K_DEF = 50, 37, 5, 80

feat_df_v1 = create_features_v2(df, HOME_ADV, K_FACTOR, K_ATT, K_DEF, defaults)
train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_df_v1[train_mask][FEATURE_COLS_V1].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = feat_df_v1[train_mask]["FTR"].values
X_test = feat_df_v1[test_mask][FEATURE_COLS_V1].fillna(0).replace([np.inf, -np.inf], 0).values
y_test = feat_df_v1[test_mask]["FTR"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000)
model.fit(X_train_s, y_train)

proba = model.predict_proba(X_test_s)
proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
loss_v1 = log_loss(y_test, proba_ord, labels=CLASSES)
print(f"   V1 (20 features): Loss={loss_v1:.4f}")

print("\n2.2 Feature coefficients (cho Draw class):")
draw_idx = list(model.classes_).index('D')
coefs = model.coef_[draw_idx]
sorted_idx = np.argsort(np.abs(coefs))[::-1]
print("   Top features for Draw prediction:")
for i in sorted_idx[:10]:
    print(f"     {FEATURE_COLS_V1[i]}: {coefs[i]:.4f}")

print("\n" + "="*70)
print("PHẦN 3: THỬ FEATURES MỚI")
print("="*70)

FEATURE_COLS_V2 = FEATURE_COLS_V1 + [
    "h_ppg", "a_ppg", "ppg_diff",
    "h_goals_avg", "a_goals_avg", 
    "h_conceded_avg", "a_conceded_avg",
    "elo_closeness", "strength_similarity", "both_draw_prone"
]

X_train_v2 = feat_df_v1[train_mask][FEATURE_COLS_V2].fillna(0).replace([np.inf, -np.inf], 0).values
X_test_v2 = feat_df_v1[test_mask][FEATURE_COLS_V2].fillna(0).replace([np.inf, -np.inf], 0).values

scaler_v2 = StandardScaler()
X_train_v2_s = scaler_v2.fit_transform(X_train_v2)
X_test_v2_s = scaler_v2.transform(X_test_v2)

model_v2 = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000)
model_v2.fit(X_train_v2_s, y_train)

proba_v2 = model_v2.predict_proba(X_test_v2_s)
proba_v2_ord = np.column_stack([proba_v2[:, list(model_v2.classes_).index(c)] for c in CLASSES])
loss_v2 = log_loss(y_test, proba_v2_ord, labels=CLASSES)
print(f"\n3.1 V2 (30 features với table position): Loss={loss_v2:.4f}")

if loss_v2 < loss_v1:
    print(f"   ✓ Cải thiện: {loss_v1 - loss_v2:.4f}")
else:
    print(f"   ✗ Không cải thiện")

print("\n" + "="*70)
print("PHẦN 4: THỬ OUTPUT KHÁC - GOAL DIFFERENCE")
print("="*70)

y_train_gd = (train_df["FTHG"] - train_df["FTAG"]).values
y_test_gd = (test_df["FTHG"] - test_df["FTAG"]).values

from sklearn.linear_model import Ridge
model_gd = Ridge(alpha=1.0)
model_gd.fit(X_train_s, y_train_gd)
pred_gd = model_gd.predict(X_test_s)

print("4.1 Predict goal difference:")
print(f"   MAE: {np.abs(pred_gd - y_test_gd).mean():.3f}")
print(f"   Correlation: {np.corrcoef(pred_gd, y_test_gd)[0,1]:.3f}")

def gd_to_result(gd):
    if gd > 0.3:
        return "H"
    elif gd < -0.3:
        return "A"
    else:
        return "D"

pred_result = [gd_to_result(g) for g in pred_gd]
acc_gd = accuracy_score(y_test, pred_result)
print(f"   Accuracy from GD: {acc_gd*100:.1f}%")

print("\n" + "="*70)
print("PHẦN 5: THỬ PREDICT GOALS RIÊNG")
print("="*70)

y_train_hg = train_df["FTHG"].values
y_train_ag = train_df["FTAG"].values
y_test_hg = test_df["FTHG"].values
y_test_ag = test_df["FTAG"].values

from sklearn.linear_model import PoissonRegressor
model_hg = PoissonRegressor(alpha=0.1)
model_ag = PoissonRegressor(alpha=0.1)

model_hg.fit(X_train_s, y_train_hg)
model_ag.fit(X_train_s, y_train_ag)

pred_hg = model_hg.predict(X_test_s)
pred_ag = model_ag.predict(X_test_s)

print("5.1 Predict home goals:")
print(f"   MAE: {np.abs(pred_hg - y_test_hg).mean():.3f}")
print(f"   Actual mean: {y_test_hg.mean():.2f}, Pred mean: {pred_hg.mean():.2f}")

print("\n5.2 Predict away goals:")
print(f"   MAE: {np.abs(pred_ag - y_test_ag).mean():.3f}")
print(f"   Actual mean: {y_test_ag.mean():.2f}, Pred mean: {pred_ag.mean():.2f}")

from scipy.stats import poisson
def calc_match_probs(home_exp, away_exp, max_goals=6):
    probs = {"H": 0, "D": 0, "A": 0}
    for hg in range(max_goals):
        for ag in range(max_goals):
            p = poisson.pmf(hg, home_exp) * poisson.pmf(ag, away_exp)
            if hg > ag:
                probs["H"] += p
            elif hg < ag:
                probs["A"] += p
            else:
                probs["D"] += p
    total = sum(probs.values())
    return {k: v/total for k, v in probs.items()}

proba_poisson = []
for i in range(len(X_test_s)):
    p = calc_match_probs(pred_hg[i], pred_ag[i])
    proba_poisson.append([p["A"], p["D"], p["H"]])
proba_poisson = np.array(proba_poisson)

loss_poisson = log_loss(y_test, proba_poisson, labels=CLASSES)
print(f"\n5.3 Poisson model loss: {loss_poisson:.4f}")

print("\n" + "="*70)
print("PHẦN 6: PHÂN TÍCH DRAW CASES")
print("="*70)

test_feat = feat_df_v1[test_mask].copy()
test_feat["pred_A"] = proba_ord[:, 0]
test_feat["pred_D"] = proba_ord[:, 1]
test_feat["pred_H"] = proba_ord[:, 2]
test_feat["actual"] = y_test

draws = test_feat[test_feat["actual"] == "D"]
non_draws = test_feat[test_feat["actual"] != "D"]

print("6.1 So sánh features giữa Draw và Non-Draw:")
for col in ["elo_diff", "combined_draw_rate", "h2h_draw_rate", "elo_closeness", "strength_similarity"]:
    if col in test_feat.columns:
        draw_mean = draws[col].mean()
        non_draw_mean = non_draws[col].mean()
        print(f"   {col}: Draw={draw_mean:.3f}, Non-Draw={non_draw_mean:.3f}")

print("\n6.2 Predicted Draw probability:")
print(f"   Actual Draws - mean pred_D: {draws['pred_D'].mean():.3f}")
print(f"   Non-Draws - mean pred_D: {non_draws['pred_D'].mean():.3f}")

print("\n" + "="*70)
print("PHẦN 7: THỬ CLASS WEIGHTS")
print("="*70)

model_balanced = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000, class_weight='balanced')
model_balanced.fit(X_train_s, y_train)

proba_bal = model_balanced.predict_proba(X_test_s)
proba_bal_ord = np.column_stack([proba_bal[:, list(model_balanced.classes_).index(c)] for c in CLASSES])
loss_balanced = log_loss(y_test, proba_bal_ord, labels=CLASSES)
pred_balanced = model_balanced.predict(X_test_s)

print(f"7.1 Balanced class weights:")
print(f"   Loss: {loss_balanced:.4f}")
print(f"   Accuracy: {accuracy_score(y_test, pred_balanced)*100:.1f}%")
print(f"   Draw predictions: {(pred_balanced == 'D').sum()}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Model comparisons:
  V1 (20 features, L2 C=0.1):     Loss={loss_v1:.4f}
  V2 (30 features + table):      Loss={loss_v2:.4f}
  Poisson (goals prediction):    Loss={loss_poisson:.4f}
  Balanced class weights:        Loss={loss_balanced:.4f}

Best: {'V2' if loss_v2 < loss_v1 else 'V1'} with Loss={min(loss_v1, loss_v2):.4f}

Key findings:
1. Draw prediction is hard - model rarely predicts Draw
2. Adding table position features {'helps' if loss_v2 < loss_v1 else 'does not help'}
3. Poisson model {'better' if loss_poisson < loss_v1 else 'worse'} than classification
4. Balanced weights {'improve' if loss_balanced < loss_v1 else 'hurt'} performance
""")

results = pd.DataFrame([
    {"model": "V1_L2_C0.1", "loss": loss_v1, "features": 20},
    {"model": "V2_with_table", "loss": loss_v2, "features": 30},
    {"model": "Poisson", "loss": loss_poisson, "features": 20},
    {"model": "Balanced_weights", "loss": loss_balanced, "features": 20},
])
results = results.sort_values("loss")
results.to_csv("experiments/deep_analysis_results.csv", index=False)
print("\nResults saved to experiments/deep_analysis_results.csv")
