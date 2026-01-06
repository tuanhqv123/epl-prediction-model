import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
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

train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
train_df = df[df["season"].isin(train_seasons)]
defaults = calculate_defaults(train_df)

FEATURE_COLS = ["home_elo", "away_elo", "elo_diff", "elo_expected", "att_def_matchup",
                "expected_diff", "combined_draw_rate", "h2h_draw_rate", "h2h_home_rate",
                "h2h_away_rate", "h_nl_scoring", "a_nl_scoring", "nl_scoring_diff",
                "h_rest", "a_rest", "rest_diff", "avg_change_rate_diff", "winrate_3_diff",
                "cs_3_diff", "shots_diff"]

def create_features(df, home_adv, k_factor, k_att, k_def, defaults, elo_divisor=400):
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
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / elo_divisor))
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

print("DRAW PREDICTION RESEARCH - PART 2")
print("Thử thêm các phương pháp khác\n")

feat_df = create_features(df, 82, 28, 5, 90, defaults, 640)

train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_df[train_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = feat_df[train_mask]["FTR"].values
X_test = feat_df[test_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_test = feat_df[test_mask]["FTR"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

draw_actual = (y_test == 'D').sum()

def eval_model(model, name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
        loss = log_loss(y_test, proba_ord, labels=CLASSES)
    else:
        loss = None
    
    acc = accuracy_score(y_test, y_pred)
    draw_pred = (y_pred == 'D').sum()
    draw_correct = ((y_pred == 'D') & (y_test == 'D')).sum()
    
    loss_str = f"{loss:.4f}" if loss else "N/A"
    print(f"{name:45s} | Loss={loss_str} | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred")
    
    return loss, acc, draw_pred, draw_correct

print("METHOD 10: Ensemble với different class weights")
print("-" * 100)

from sklearn.ensemble import RandomForestClassifier

estimators = [
    ('lr1', LogisticRegression(C=0.15, max_iter=2000)),
    ('lr2', LogisticRegression(C=0.1, max_iter=2000)),
    ('lr3', LogisticRegression(C=0.2, max_iter=2000)),
]
ensemble = VotingClassifier(estimators=estimators, voting='soft')
eval_model(ensemble, "VotingClassifier (soft) C=0.1,0.15,0.2", X_train_s, y_train, X_test_s, y_test)

print("\nMETHOD 11: Calibrated Classifier")
print("-" * 100)

base = LogisticRegression(C=0.15, max_iter=2000)
calibrated = CalibratedClassifierCV(base, method='isotonic', cv=5)
eval_model(calibrated, "CalibratedClassifierCV isotonic", X_train_s, y_train, X_test_s, y_test)

calibrated2 = CalibratedClassifierCV(base, method='sigmoid', cv=5)
eval_model(calibrated2, "CalibratedClassifierCV sigmoid", X_train_s, y_train, X_test_s, y_test)

print("\nMETHOD 12: Binary Draw classifier + combine")
print("-" * 100)

y_train_draw = (y_train == 'D').astype(int)
y_test_draw = (y_test == 'D').astype(int)

for w in [2, 3, 5, 10]:
    draw_clf = LogisticRegression(C=0.15, class_weight={0: 1, 1: w}, max_iter=2000)
    draw_clf.fit(X_train_s, y_train_draw)
    draw_proba = draw_clf.predict_proba(X_test_s)[:, 1]
    
    main_clf = LogisticRegression(C=0.15, max_iter=2000)
    main_clf.fit(X_train_s, y_train)
    main_proba = main_clf.predict_proba(X_test_s)
    
    combined_proba = main_proba.copy()
    draw_idx = list(main_clf.classes_).index('D')
    combined_proba[:, draw_idx] = draw_proba
    combined_proba = combined_proba / combined_proba.sum(axis=1, keepdims=True)
    
    y_pred = np.array([CLASSES[i] for i in combined_proba.argmax(axis=1)])
    
    loss = log_loss(y_test, combined_proba, labels=CLASSES)
    acc = accuracy_score(y_test, y_pred)
    draw_pred = (y_pred == 'D').sum()
    draw_correct = ((y_pred == 'D') & (y_test == 'D')).sum()
    print(f"{'Binary Draw clf weight=' + str(w) + ' + main':45s} | Loss={loss:.4f} | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred")

print("\nMETHOD 13: Adjust Draw probability post-hoc")
print("-" * 100)

model = LogisticRegression(C=0.15, max_iter=2000)
model.fit(X_train_s, y_train)
proba = model.predict_proba(X_test_s)
proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])

for boost in [1.2, 1.5, 2.0, 2.5, 3.0]:
    adjusted = proba_ord.copy()
    adjusted[:, 1] *= boost
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    
    y_pred = np.array([CLASSES[i] for i in adjusted.argmax(axis=1)])
    loss = log_loss(y_test, adjusted, labels=CLASSES)
    acc = accuracy_score(y_test, y_pred)
    draw_pred = (y_pred == 'D').sum()
    draw_correct = ((y_pred == 'D') & (y_test == 'D')).sum()
    print(f"{'Post-hoc Draw boost=' + str(boost):45s} | Loss={loss:.4f} | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred")

print("\nMETHOD 14: Focal Loss simulation (penalize confident wrong)")
print("-" * 100)

from sklearn.ensemble import GradientBoostingClassifier

for gamma in [0.5, 1.0, 2.0]:
    sample_weights = np.ones(len(y_train))
    
    temp_model = LogisticRegression(C=0.15, max_iter=2000)
    temp_model.fit(X_train_s, y_train)
    train_proba = temp_model.predict_proba(X_train_s)
    
    for i, (true_label, probs) in enumerate(zip(y_train, train_proba)):
        true_idx = list(temp_model.classes_).index(true_label)
        p_true = probs[true_idx]
        sample_weights[i] = (1 - p_true) ** gamma
        if true_label == 'D':
            sample_weights[i] *= 2
    
    model = LogisticRegression(C=0.15, max_iter=2000)
    model.fit(X_train_s, y_train, sample_weight=sample_weights)
    
    y_pred = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)
    proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
    
    loss = log_loss(y_test, proba_ord, labels=CLASSES)
    acc = accuracy_score(y_test, y_pred)
    draw_pred = (y_pred == 'D').sum()
    draw_correct = ((y_pred == 'D') & (y_test == 'D')).sum()
    print(f"{'Focal-like gamma=' + str(gamma) + ' + Draw*2':45s} | Loss={loss:.4f} | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred")

print("\nMETHOD 15: One-vs-Rest với Draw-focused classifier")
print("-" * 100)

from sklearn.multiclass import OneVsRestClassifier

for C_draw in [0.05, 0.1, 0.2]:
    ovr = OneVsRestClassifier(LogisticRegression(C=C_draw, max_iter=2000))
    eval_model(ovr, f"OneVsRest C={C_draw}", X_train_s, y_train, X_test_s, y_test)

print("\nMETHOD 16: Stacking với Draw specialist")
print("-" * 100)

from sklearn.ensemble import StackingClassifier

estimators = [
    ('lr_base', LogisticRegression(C=0.15, max_iter=2000)),
    ('lr_draw', LogisticRegression(C=0.1, max_iter=2000)),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), cv=5)
eval_model(stacking, "StackingClassifier", X_train_s, y_train, X_test_s, y_test)

print("\nSUMMARY")
print("-" * 100)
print("""
Kết luận từ tất cả experiments:

1. BEST cho Loss thấp: LogReg baseline (Loss=0.9362, Draw recall=1.1%)
2. BEST cho Draw recall cao: class_weight D=5 (Loss=1.19, Draw recall=93.1%)
3. BALANCED: class_weight D=2 (Loss=0.9757, Draw recall=40%, Acc=50.9%)

Trade-off không thể tránh:
- Muốn predict Draw tốt hơn -> Loss tăng, Accuracy giảm
- Muốn Loss thấp -> Gần như không predict Draw

Lý do Draw khó predict:
- Draw không có pattern rõ ràng trong features
- Draw xảy ra khi cả 2 đội "cân bằng" nhưng features không capture được điều này
- Cần thêm data như betting odds, xG, player injuries để cải thiện
""")
