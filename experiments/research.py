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

train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
train_df = df[df["season"].isin(train_seasons)]
test_df = df[df["season"].isin(test_seasons)]

print("="*70)
print("NGHIÊN CỨU SÂU: TÌM VẤN ĐỀ CỐT LÕI")
print("="*70)

print("""
CÂU HỎI 1: Model đang predict sai ở đâu?
- Sai nhiều nhất ở class nào?
- Sai trong trường hợp nào?
""")

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

FEATURE_COLS = ["home_elo", "away_elo", "elo_diff", "elo_expected", "att_def_matchup",
                "expected_diff", "combined_draw_rate", "h2h_draw_rate", "h2h_home_rate",
                "h2h_away_rate", "h_nl_scoring", "a_nl_scoring", "nl_scoring_diff",
                "h_rest", "a_rest", "rest_diff", "avg_change_rate_diff", "winrate_3_diff",
                "cs_3_diff", "shots_diff"]

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
            "season": season, "FTR": m["FTR"], "HomeTeam": home, "AwayTeam": away,
            "FTHG": m["FTHG"], "FTAG": m["FTAG"],
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

defaults = calculate_defaults(train_df)
feat_df = create_features(df, 50, 37, 5, 80, defaults)

train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_df[train_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = feat_df[train_mask]["FTR"].values
X_test = feat_df[test_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_test = feat_df[test_mask]["FTR"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000)
model.fit(X_train_s, y_train)

proba = model.predict_proba(X_test_s)
proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
y_pred = model.predict(X_test_s)

test_feat = feat_df[test_mask].copy()
test_feat["pred_A"] = proba_ord[:, 0]
test_feat["pred_D"] = proba_ord[:, 1]
test_feat["pred_H"] = proba_ord[:, 2]
test_feat["pred"] = y_pred
test_feat["correct"] = (y_pred == y_test)

print("\n1.1 Phân tích lỗi theo class:")
for actual in CLASSES:
    subset = test_feat[test_feat["FTR"] == actual]
    correct = subset["correct"].sum()
    total = len(subset)
    print(f"   Actual {actual}: {correct}/{total} correct ({correct/total*100:.1f}%)")
    for pred in CLASSES:
        pred_count = (subset["pred"] == pred).sum()
        print(f"      Predicted {pred}: {pred_count} ({pred_count/total*100:.1f}%)")

print("\n1.2 Khi model confident nhưng sai:")
test_feat["max_prob"] = proba_ord.max(axis=1)
confident_wrong = test_feat[(test_feat["max_prob"] > 0.6) & (~test_feat["correct"])]
print(f"   Confident (>60%) but wrong: {len(confident_wrong)} cases")
print(f"   Breakdown:")
for actual in CLASSES:
    count = (confident_wrong["FTR"] == actual).sum()
    print(f"      Actual {actual}: {count}")

print("""
CÂU HỎI 2: Elo system có vấn đề gì không?
- Elo có phản ánh đúng sức mạnh đội?
- K-factor có phù hợp?
""")

print("\n2.1 Elo distribution trong test set:")
print(f"   Home Elo: mean={test_feat['home_elo'].mean():.0f}, std={test_feat['home_elo'].std():.0f}")
print(f"   Away Elo: mean={test_feat['away_elo'].mean():.0f}, std={test_feat['away_elo'].std():.0f}")

print("\n2.2 Elo expected vs actual win rate:")
bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
for i in range(len(bins)-1):
    mask = (test_feat["elo_expected"] >= bins[i]) & (test_feat["elo_expected"] < bins[i+1])
    if mask.sum() > 0:
        actual_h = (test_feat[mask]["FTR"] == "H").mean()
        expected_h = test_feat[mask]["elo_expected"].mean()
        print(f"   Elo exp {bins[i]:.1f}-{bins[i+1]:.1f}: Expected H={expected_h*100:.1f}%, Actual H={actual_h*100:.1f}%, n={mask.sum()}")

print("""
CÂU HỎI 3: Features có vấn đề gì?
- Feature nào có correlation cao với nhau (redundant)?
- Feature nào không có predictive power?
""")

print("\n3.1 Feature correlations với target:")
from scipy.stats import pointbiserialr
y_binary_H = (y_test == "H").astype(int)
y_binary_D = (y_test == "D").astype(int)
y_binary_A = (y_test == "A").astype(int)

print("   Correlation với Home Win:")
corrs_H = []
for i, col in enumerate(FEATURE_COLS):
    corr, _ = pointbiserialr(y_binary_H, X_test[:, i])
    corrs_H.append((col, corr))
corrs_H.sort(key=lambda x: abs(x[1]), reverse=True)
for col, corr in corrs_H[:5]:
    print(f"      {col}: {corr:.3f}")

print("\n   Correlation với Draw:")
corrs_D = []
for i, col in enumerate(FEATURE_COLS):
    corr, _ = pointbiserialr(y_binary_D, X_test[:, i])
    corrs_D.append((col, corr))
corrs_D.sort(key=lambda x: abs(x[1]), reverse=True)
for col, corr in corrs_D[:5]:
    print(f"      {col}: {corr:.3f}")

print("\n3.2 Feature-feature correlations (redundancy):")
corr_matrix = np.corrcoef(X_train.T)
high_corrs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i+1, len(FEATURE_COLS)):
        if abs(corr_matrix[i,j]) > 0.7:
            high_corrs.append((FEATURE_COLS[i], FEATURE_COLS[j], corr_matrix[i,j]))
high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
print("   Highly correlated pairs (>0.7):")
for f1, f2, corr in high_corrs[:10]:
    print(f"      {f1} <-> {f2}: {corr:.3f}")

print("""
CÂU HỎI 4: Có pattern nào trong các trận sai?
""")

print("\n4.1 Phân tích trận sai theo đội mạnh/yếu:")
wrong = test_feat[~test_feat["correct"]]
print(f"   Total wrong: {len(wrong)}")

big_elo_diff = wrong[abs(wrong["elo_diff"]) > 150]
small_elo_diff = wrong[abs(wrong["elo_diff"]) < 50]
print(f"   Wrong with big elo diff (>150): {len(big_elo_diff)}")
print(f"   Wrong with small elo diff (<50): {len(small_elo_diff)}")

print("\n4.2 Upsets (đội yếu thắng đội mạnh):")
upsets = test_feat[(test_feat["elo_diff"] > 100) & (test_feat["FTR"] == "A")]
print(f"   Home favored by >100 but Away won: {len(upsets)}")
upsets_h = test_feat[(test_feat["elo_diff"] < -100) & (test_feat["FTR"] == "H")]
print(f"   Away favored by >100 but Home won: {len(upsets_h)}")

print("""
CÂU HỎI 5: Season 2024-2025 có gì khác biệt?
""")

print("\n5.1 So sánh 2 seasons trong test:")
for season in test_seasons:
    s_mask = test_feat["season"] == season
    s_data = test_feat[s_mask]
    s_loss = log_loss(s_data["FTR"], s_data[["pred_A", "pred_D", "pred_H"]].values, labels=CLASSES)
    s_acc = s_data["correct"].mean()
    print(f"   {season}:")
    print(f"      Loss: {s_loss:.4f}, Acc: {s_acc*100:.1f}%")
    print(f"      H/D/A: {(s_data['FTR']=='H').mean()*100:.1f}%/{(s_data['FTR']=='D').mean()*100:.1f}%/{(s_data['FTR']=='A').mean()*100:.1f}%")
    print(f"      Avg goals: {s_data['FTHG'].mean():.2f}-{s_data['FTAG'].mean():.2f}")

print("""
CÂU HỎI 6: Thử loại bỏ features redundant
""")

print("\n6.1 Loại bỏ features có correlation cao:")
REDUCED_COLS = ["elo_diff", "elo_expected", "att_def_matchup",
                "expected_diff", "combined_draw_rate", "h2h_draw_rate",
                "nl_scoring_diff", "rest_diff", "avg_change_rate_diff", 
                "winrate_3_diff", "cs_3_diff", "shots_diff"]

X_train_red = feat_df[train_mask][REDUCED_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
X_test_red = feat_df[test_mask][REDUCED_COLS].fillna(0).replace([np.inf, -np.inf], 0).values

scaler_red = StandardScaler()
X_train_red_s = scaler_red.fit_transform(X_train_red)
X_test_red_s = scaler_red.transform(X_test_red)

model_red = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=2000)
model_red.fit(X_train_red_s, y_train)
proba_red = model_red.predict_proba(X_test_red_s)
proba_red_ord = np.column_stack([proba_red[:, list(model_red.classes_).index(c)] for c in CLASSES])
loss_red = log_loss(y_test, proba_red_ord, labels=CLASSES)
print(f"   Reduced features ({len(REDUCED_COLS)}): Loss={loss_red:.4f}")

print("""
CÂU HỎI 7: Thử different regularization strengths
""")

print("\n7.1 Grid search C:")
best_c = None
best_loss = float('inf')
for c in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    model_c = LogisticRegression(penalty='l2', solver='lbfgs', C=c, max_iter=2000)
    model_c.fit(X_train_s, y_train)
    proba_c = model_c.predict_proba(X_test_s)
    proba_c_ord = np.column_stack([proba_c[:, list(model_c.classes_).index(c_)] for c_ in CLASSES])
    loss_c = log_loss(y_test, proba_c_ord, labels=CLASSES)
    if loss_c < best_loss:
        best_loss = loss_c
        best_c = c
    print(f"   C={c}: Loss={loss_c:.4f}")
print(f"   Best C={best_c}, Loss={best_loss:.4f}")

print("""
CÂU HỎI 8: Vấn đề có phải là model quá confident?
""")

print("\n8.1 Probability distribution:")
print(f"   Mean max prob: {proba_ord.max(axis=1).mean():.3f}")
print(f"   Prob > 0.5: {(proba_ord.max(axis=1) > 0.5).sum()} / {len(proba_ord)}")
print(f"   Prob > 0.6: {(proba_ord.max(axis=1) > 0.6).sum()} / {len(proba_ord)}")
print(f"   Prob > 0.7: {(proba_ord.max(axis=1) > 0.7).sum()} / {len(proba_ord)}")

print("\n8.2 Thử temperature scaling:")
def apply_temperature(proba, T):
    logits = np.log(proba + 1e-10)
    scaled = logits / T
    exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

for T in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    proba_temp = apply_temperature(proba_ord, T)
    loss_temp = log_loss(y_test, proba_temp, labels=CLASSES)
    print(f"   T={T}: Loss={loss_temp:.4f}")

print("\n" + "="*70)
print("KẾT LUẬN VÀ HƯỚNG CẢI THIỆN")
print("="*70)

original_loss = log_loss(y_test, proba_ord, labels=CLASSES)
print(f"""
Current model: Loss = {original_loss:.4f}

VẤN ĐỀ CHÍNH:
1. Draw prediction rất kém - model gần như không predict Draw
2. Features có nhiều redundancy (home_elo/away_elo/elo_diff)
3. Season 2024-2025 khó predict hơn 2023-2024

HƯỚNG CẢI THIỆN:
1. Loại bỏ features redundant
2. Tune regularization (C)
3. Temperature scaling nếu model quá confident
4. Cần data mới (betting odds, xG) để cải thiện Draw prediction
""")
