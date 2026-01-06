import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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

print("DRAW PREDICTION RESEARCH")
print("Mục tiêu: Cải thiện khả năng predict Draw\n")

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
print(f"Test set: {len(y_test)} matches, {draw_actual} draws ({draw_actual/len(y_test)*100:.1f}%)\n")

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
    draw_precision = draw_correct / draw_pred if draw_pred > 0 else 0
    draw_recall = draw_correct / draw_actual
    
    loss_str = f"{loss:.4f}" if loss else "N/A"
    print(f"{name:40s} | Loss={loss_str} | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred, recall={draw_recall*100:.1f}%")
    
    return {'name': name, 'loss': loss, 'acc': acc, 'draw_pred': draw_pred, 'draw_correct': draw_correct}

print("METHOD 1: Different class weights")
print("-" * 90)
for w in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    model = LogisticRegression(C=0.15, class_weight={'A': 1, 'D': w, 'H': 1}, max_iter=2000)
    eval_model(model, f"LogReg class_weight D={w}", X_train_s, y_train, X_test_s, y_test)

print("\nMETHOD 2: SMOTE oversampling for Draw class")
print("-" * 90)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
    model = LogisticRegression(C=0.15, max_iter=2000)
    eval_model(model, "LogReg + SMOTE", X_train_smote, y_train_smote, X_test_s, y_test)
    
    adasyn = ADASYN(random_state=42)
    X_train_ada, y_train_ada = adasyn.fit_resample(X_train_s, y_train)
    model = LogisticRegression(C=0.15, max_iter=2000)
    eval_model(model, "LogReg + ADASYN", X_train_ada, y_train_ada, X_test_s, y_test)
    
    smote_tomek = SMOTETomek(random_state=42)
    X_train_st, y_train_st = smote_tomek.fit_resample(X_train_s, y_train)
    model = LogisticRegression(C=0.15, max_iter=2000)
    eval_model(model, "LogReg + SMOTETomek", X_train_st, y_train_st, X_test_s, y_test)
except ImportError:
    print("imblearn not installed, skipping SMOTE methods")

print("\nMETHOD 3: Different ML algorithms")
print("-" * 90)

models = [
    ("LogisticRegression baseline", LogisticRegression(C=0.15, max_iter=2000)),
    ("RandomForest balanced", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)),
    ("RandomForest balanced_subsample", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced_subsample', random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ("MLP (100,50)", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)),
    ("KNN k=15", KNeighborsClassifier(n_neighbors=15)),
    ("GaussianNB", GaussianNB()),
]

for name, model in models:
    eval_model(model, name, X_train_s, y_train, X_test_s, y_test)

print("\nMETHOD 4: Two-stage prediction (Win/NotWin then Draw/Away)")
print("-" * 90)

y_train_stage1 = np.where(y_train == 'H', 'H', 'NotH')
y_test_stage1 = np.where(y_test == 'H', 'H', 'NotH')

model1 = LogisticRegression(C=0.15, max_iter=2000)
model1.fit(X_train_s, y_train_stage1)
pred1 = model1.predict(X_test_s)
proba1 = model1.predict_proba(X_test_s)

not_h_train = y_train != 'H'
not_h_test = pred1 == 'NotH'

if not_h_train.sum() > 0 and not_h_test.sum() > 0:
    model2 = LogisticRegression(C=0.15, max_iter=2000, class_weight={'A': 1, 'D': 2})
    model2.fit(X_train_s[not_h_train], y_train[not_h_train])
    
    y_pred_final = pred1.copy()
    if not_h_test.sum() > 0:
        pred2 = model2.predict(X_test_s[not_h_test])
        y_pred_final[not_h_test] = pred2
    
    acc = accuracy_score(y_test, y_pred_final)
    draw_pred = (y_pred_final == 'D').sum()
    draw_correct = ((y_pred_final == 'D') & (y_test == 'D')).sum()
    print(f"{'Two-stage (H vs NotH, then D vs A)':40s} | Loss=N/A    | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred, recall={draw_correct/draw_actual*100:.1f}%")

print("\nMETHOD 5: Threshold tuning on Draw probability")
print("-" * 90)

model = LogisticRegression(C=0.15, max_iter=2000)
model.fit(X_train_s, y_train)
proba = model.predict_proba(X_test_s)
draw_idx = list(model.classes_).index('D')
draw_proba = proba[:, draw_idx]

for threshold in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
    y_pred = model.predict(X_test_s).copy()
    y_pred[draw_proba > threshold] = 'D'
    
    acc = accuracy_score(y_test, y_pred)
    draw_pred = (y_pred == 'D').sum()
    draw_correct = ((y_pred == 'D') & (y_test == 'D')).sum()
    print(f"{'Threshold=' + str(threshold):40s} | Loss=N/A    | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred, recall={draw_correct/draw_actual*100:.1f}%")

print("\nMETHOD 6: Ordinal Regression (treat as ordered: A < D < H)")
print("-" * 90)
try:
    from mord import LogisticAT
    
    y_train_ord = np.where(y_train == 'A', 0, np.where(y_train == 'D', 1, 2))
    y_test_ord = np.where(y_test == 'A', 0, np.where(y_test == 'D', 1, 2))
    
    model = LogisticAT(alpha=1.0)
    model.fit(X_train_s, y_train_ord)
    y_pred_ord = model.predict(X_test_s)
    y_pred = np.where(y_pred_ord == 0, 'A', np.where(y_pred_ord == 1, 'D', 'H'))
    
    acc = accuracy_score(y_test, y_pred)
    draw_pred = (y_pred == 'D').sum()
    draw_correct = ((y_pred == 'D') & (y_test == 'D')).sum()
    print(f"{'Ordinal LogisticAT':40s} | Loss=N/A    | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred, recall={draw_correct/draw_actual*100:.1f}%")
except ImportError:
    print("mord not installed, skipping ordinal regression")

print("\nMETHOD 7: XGBoost with scale_pos_weight")
print("-" * 90)
try:
    import xgboost as xgb
    
    y_train_enc = np.where(y_train == 'A', 0, np.where(y_train == 'D', 1, 2))
    y_test_enc = np.where(y_test == 'A', 0, np.where(y_test == 'D', 1, 2))
    
    for scale in [1, 2, 3]:
        sample_weights = np.ones(len(y_train_enc))
        sample_weights[y_train_enc == 1] = scale
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train_s, y_train_enc, sample_weight=sample_weights)
        y_pred_enc = model.predict(X_test_s)
        y_pred = np.where(y_pred_enc == 0, 'A', np.where(y_pred_enc == 1, 'D', 'H'))
        
        proba = model.predict_proba(X_test_s)
        loss = log_loss(y_test_enc, proba)
        acc = accuracy_score(y_test, y_pred)
        draw_pred = (y_pred == 'D').sum()
        draw_correct = ((y_pred == 'D') & (y_test == 'D')).sum()
        print(f"{'XGBoost draw_weight=' + str(scale):40s} | Loss={loss:.4f} | Acc={acc*100:.1f}% | Draw: {draw_correct}/{draw_pred} pred, recall={draw_correct/draw_actual*100:.1f}%")
except ImportError:
    print("xgboost not installed")

print("\nMETHOD 8: LightGBM with class_weight")
print("-" * 90)
try:
    import lightgbm as lgb
    
    for w in [1, 2, 3]:
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight={'A': 1, 'D': w, 'H': 1}, random_state=42, verbose=-1)
        eval_model(model, f"LightGBM draw_weight={w}", X_train_s, y_train, X_test_s, y_test)
except ImportError:
    print("lightgbm not installed")

print("\nMETHOD 9: CatBoost")
print("-" * 90)
try:
    from catboost import CatBoostClassifier
    
    for w in [1, 2, 3]:
        model = CatBoostClassifier(iterations=100, depth=5, class_weights={'A': 1, 'D': w, 'H': 1}, random_state=42, verbose=0)
        eval_model(model, f"CatBoost draw_weight={w}", X_train_s, y_train, X_test_s, y_test)
except ImportError:
    print("catboost not installed")

print("\nSUMMARY")
print("-" * 90)
print("""
Kết luận:
- Draw là class khó predict nhất vì không có pattern rõ ràng
- Tăng class_weight cho Draw giúp predict nhiều Draw hơn nhưng làm tăng Loss
- SMOTE/ADASYN không cải thiện đáng kể
- Threshold tuning có thể tăng Draw recall nhưng giảm accuracy
- Trade-off: Muốn predict Draw tốt hơn phải chấp nhận Loss cao hơn
""")
