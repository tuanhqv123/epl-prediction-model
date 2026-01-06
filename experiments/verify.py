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

df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

nl_dates, nl_scoring = load_non_league_data()
team_lineups, lineup_lookup = load_lineup_data()

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
CLASSES = ["A", "D", "H"]

FEATURE_COLS = ["home_elo", "away_elo", "elo_diff", "elo_expected", "att_def_matchup",
                "expected_diff", "combined_draw_rate", "h2h_draw_rate", "h2h_home_rate",
                "h2h_away_rate", "h_nl_scoring", "a_nl_scoring", "nl_scoring_diff",
                "h_rest", "a_rest", "rest_diff", "avg_change_rate_diff", "winrate_3_diff",
                "cs_3_diff", "shots_diff"]

HOME_ADV, K_FACTOR, K_ATT, K_DEF = 50, 37, 5, 80

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

def create_features(df, home_adv, k_factor, k_att, k_def, defaults):
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
        att_def = (h_att - a_def + home_adv/2) - (a_att - h_def - home_adv/2)
        
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
        
        features.append({
            "season": season, "FTR": m["FTR"],
            "home_elo": h_elo, "away_elo": a_elo, "elo_diff": elo_diff, "elo_expected": exp_h,
            "att_def_matchup": att_def, "expected_diff": home_xg - away_xg,
            "combined_draw_rate": (home_draw + away_draw) / 2, "h2h_draw_rate": h2h_draw,
            "h2h_home_rate": h2h_home, "h2h_away_rate": h2h_away,
            "h_nl_scoring": h_nl, "a_nl_scoring": a_nl, "nl_scoring_diff": h_nl - a_nl,
            "h_rest": h_rest, "a_rest": a_rest, "rest_diff": h_rest - a_rest,
            "avg_change_rate_diff": h_change - a_change,
            "winrate_3_diff": h_winrate - a_winrate, "cs_3_diff": h_cs - a_cs, "shots_diff": h_shots - a_shots,
        })
        
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

print("="*60)
print("VERIFICATION: Kiểm tra chi tiết pipeline")
print("="*60)

print("\n1. DATA SPLIT")
train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
print(f"   Train: {train_seasons}")
print(f"   Test: {test_seasons}")

train_df = df[df["season"].isin(train_seasons)]
test_df = df[df["season"].isin(test_seasons)]
print(f"   Train matches: {len(train_df)}")
print(f"   Test matches: {len(test_df)}")

print("\n2. DATA LEAKAGE CHECK")
train_dates = train_df["Date"].max()
test_dates = test_df["Date"].min()
print(f"   Last train date: {train_dates}")
print(f"   First test date: {test_dates}")
print(f"   Gap: {(test_dates - train_dates).days} days")
if test_dates > train_dates:
    print("   ✓ No temporal leakage")
else:
    print("   ✗ TEMPORAL LEAKAGE DETECTED!")

print("\n3. CLASS DISTRIBUTION")
print("   Train:")
for c in CLASSES:
    pct = (train_df['FTR'] == c).mean() * 100
    print(f"     {c}: {pct:.1f}%")
print("   Test:")
for c in CLASSES:
    pct = (test_df['FTR'] == c).mean() * 100
    print(f"     {c}: {pct:.1f}%")

print("\n4. DEFAULTS (calculated from TRAIN only)")
defaults = calculate_defaults(train_df)
for k, v in defaults.items():
    if isinstance(v, float):
        print(f"   {k}: {v:.4f}")
    else:
        print(f"   {k}: {v}")

print("\n5. FEATURE CREATION")
feat_df = create_features(df, HOME_ADV, K_FACTOR, K_ATT, K_DEF, defaults)
train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_df[train_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = feat_df[train_mask]["FTR"].values
X_test = feat_df[test_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_test = feat_df[test_mask]["FTR"].values

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   NaN in X_train: {np.isnan(X_train).sum()}")
print(f"   NaN in X_test: {np.isnan(X_test).sum()}")
print(f"   Inf in X_train: {np.isinf(X_train).sum()}")
print(f"   Inf in X_test: {np.isinf(X_test).sum()}")

print("\n6. FEATURE STATISTICS (Train)")
for i, col in enumerate(FEATURE_COLS):
    print(f"   {col}: mean={X_train[:, i].mean():.4f}, std={X_train[:, i].std():.4f}")

print("\n7. SCALING")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"   Scaler fitted on train only: ✓")
print(f"   X_train_s mean: {X_train_s.mean():.6f} (should be ~0)")
print(f"   X_train_s std: {X_train_s.std():.6f} (should be ~1)")

print("\n8. MODEL TRAINING")
model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000)
model.fit(X_train_s, y_train)
print(f"   Model classes: {model.classes_}")
print(f"   Coefficients shape: {model.coef_.shape}")

print("\n9. PREDICTIONS")
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)
print(f"   Predictions shape: {y_pred.shape}")
print(f"   Probabilities shape: {y_proba.shape}")
print(f"   Probability sums (should be 1.0): {y_proba.sum(axis=1)[:5]}")

print("\n10. LOG LOSS CALCULATION")
proba_ord = np.column_stack([y_proba[:, list(model.classes_).index(c)] for c in CLASSES])
print(f"    Ordered probabilities shape: {proba_ord.shape}")
print(f"    Sample proba (first 3):")
for i in range(3):
    print(f"      {y_test[i]}: A={proba_ord[i,0]:.3f}, D={proba_ord[i,1]:.3f}, H={proba_ord[i,2]:.3f}")

test_loss = log_loss(y_test, proba_ord, labels=CLASSES)
print(f"\n    Test Log Loss: {test_loss:.6f}")

random_loss = np.log(3)
print(f"    Random baseline (log(3)): {random_loss:.6f}")
print(f"    Improvement: {(random_loss - test_loss) / random_loss * 100:.2f}%")

print("\n11. MANUAL LOG LOSS VERIFICATION")
manual_loss = 0
for i in range(len(y_test)):
    true_class = y_test[i]
    true_idx = CLASSES.index(true_class)
    prob = proba_ord[i, true_idx]
    prob = np.clip(prob, 1e-15, 1 - 1e-15)
    manual_loss -= np.log(prob)
manual_loss /= len(y_test)
print(f"    Manual calculation: {manual_loss:.6f}")
print(f"    sklearn calculation: {test_loss:.6f}")
print(f"    Match: {'✓' if abs(manual_loss - test_loss) < 0.0001 else '✗'}")

print("\n12. ACCURACY")
acc = accuracy_score(y_test, y_pred)
print(f"    Accuracy: {acc*100:.2f}%")
print(f"    Correct: {(y_pred == y_test).sum()} / {len(y_test)}")

print("\n13. CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
print(f"    Predicted →")
print(f"    Actual ↓   A    D    H")
for i, c in enumerate(CLASSES):
    print(f"         {c}   {cm[i, 0]:3d}  {cm[i, 1]:3d}  {cm[i, 2]:3d}")

print("\n14. CALIBRATION")
print("    Class | Predicted | Actual | Diff")
for i, c in enumerate(CLASSES):
    pred_pct = proba_ord[:, i].mean() * 100
    actual_pct = (y_test == c).mean() * 100
    diff = abs(pred_pct - actual_pct)
    status = '✓' if diff < 2 else '⚠'
    print(f"      {c}   |   {pred_pct:.1f}%   | {actual_pct:.1f}%  | {diff:.1f}% {status}")

print("\n15. PER-SEASON BREAKDOWN")
for season in test_seasons:
    mask = feat_df[test_mask]["season"].values == season
    if mask.sum() > 0:
        season_loss = log_loss(y_test[mask], proba_ord[mask], labels=CLASSES)
        season_acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"    {season}: Loss={season_loss:.4f}, Acc={season_acc*100:.1f}%, n={mask.sum()}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Model: LogisticRegression L2, C=0.1")
print(f"Features: {len(FEATURE_COLS)}")
print(f"Train: {len(X_train)} matches ({train_seasons[0]} to {train_seasons[-1]})")
print(f"Test: {len(X_test)} matches ({test_seasons[0]} to {test_seasons[-1]})")
print(f"Test Log Loss: {test_loss:.4f}")
print(f"Test Accuracy: {acc*100:.1f}%")
print(f"Data Leakage: None detected")
