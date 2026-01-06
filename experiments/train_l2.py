import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import pickle
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
H2H_MIN, NL_WINDOW, REST_CAP, FORM_WINDOW = 7, 15, 7, 4
SOT_WINDOW, DRAW_WINDOW, SHOTS_WINDOW = 3, 5, 3

FEATURE_COLS = ["home_elo", "away_elo", "elo_diff", "elo_expected", "att_def_matchup",
                "expected_diff", "combined_draw_rate", "h2h_draw_rate", "h2h_home_rate",
                "h2h_away_rate", "h_nl_scoring", "a_nl_scoring", "nl_scoring_diff",
                "h_rest", "a_rest", "rest_diff", "avg_change_rate_diff", "winrate_3_diff",
                "cs_3_diff", "shots_diff"]

search_space = [
    Integer(50, 400, name='home_adv'),
    Integer(15, 50, name='k_factor'),
    Integer(5, 25, name='k_att'),
    Integer(20, 80, name='k_def'),
    Real(0.05, 0.5, prior='log-uniform', name='C'),
]

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
        
        home_xg = np.mean(h_hist["sot"][-SOT_WINDOW:]) * d['sot_conversion'] if h_hist["sot"] else d['avg_home_goals']
        away_xg = np.mean(a_hist["sot"][-SOT_WINDOW:]) * d['sot_conversion'] if a_hist["sot"] else d['avg_away_goals']
        home_draw = np.mean(h_hist["draw"][-DRAW_WINDOW:]) if h_hist["draw"] else d['draw_rate']
        away_draw = np.mean(a_hist["draw"][-DRAW_WINDOW:]) if a_hist["draw"] else d['draw_rate']
        
        h2h_key = tuple(sorted([home, away]))
        h2h = h2h_history.get(h2h_key, [])
        h2h_draw = np.mean([1 if x == "D" else 0 for x in h2h]) if h2h else d['draw_rate']
        h2h_home = sum(1 for x in h2h if x == "H") / len(h2h) if len(h2h) >= H2H_MIN else d['home_win_rate']
        h2h_away = sum(1 for x in h2h if x == "A") / len(h2h) if len(h2h) >= H2H_MIN else d['away_win_rate']
        
        h_rest = min((match_date - team_last_epl[home]).days, REST_CAP) if home in team_last_epl else d['default_rest']
        a_rest = min((match_date - team_last_epl[away]).days, REST_CAP) if away in team_last_epl else d['default_rest']
        
        def get_nl(team):
            if team not in nl_scoring: return d['default_nl_scoring']
            recent = sorted([s for s in nl_scoring[team] if s["season"] == season and s["date"] < match_date], 
                          key=lambda x: x["date"], reverse=True)[:NL_WINDOW]
            return np.mean([s["scored"] for s in recent]) if recent else d['default_nl_scoring']
        h_nl, a_nl = get_nl(home), get_nl(away)
        
        lookup_key = (home, away, match_date.date())
        curr_lineups = lineup_lookup.get(lookup_key, {'home': set(), 'away': set()})
        h_change = get_avg_change_rate(home, match_date, team_lineups, curr_lineups['home'], d['default_lineup_change'])
        a_change = get_avg_change_rate(away, match_date, team_lineups, curr_lineups['away'], d['default_lineup_change'])
        
        h_winrate = get_winrate(h_hist["results"], FORM_WINDOW, d['default_win_rate'])
        a_winrate = get_winrate(a_hist["results"], FORM_WINDOW, d['default_win_rate'])
        h_cs = get_clean_sheet_rate(h_hist["ga"], FORM_WINDOW, d['default_cs_rate'])
        a_cs = get_clean_sheet_rate(a_hist["ga"], FORM_WINDOW, d['default_cs_rate'])
        h_shots = np.mean(h_hist["shots"][-SHOTS_WINDOW:]) if h_hist["shots"] else d['avg_home_shots']
        a_shots = np.mean(a_hist["shots"][-SHOTS_WINDOW:]) if a_hist["shots"] else d['avg_away_shots']
        
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

def walk_forward_cv(home_adv, k_factor, k_att, k_def, C):
    folds = [
        (SEASONS[:3], [SEASONS[3]]), (SEASONS[:4], [SEASONS[4]]), (SEASONS[:5], [SEASONS[5]]),
        (SEASONS[:6], [SEASONS[6]]), (SEASONS[:7], [SEASONS[7]]),
    ]
    losses = []
    for train_s, val_s in folds:
        train_df = df[df["season"].isin(train_s)]
        defaults = calculate_defaults(train_df)
        all_df = df[df["season"].isin(train_s + val_s)]
        feat_df = create_features(all_df, home_adv, k_factor, k_att, k_def, defaults)
        
        train_mask = all_df["season"].isin(train_s).values
        val_mask = all_df["season"].isin(val_s).values
        
        X_train = feat_df[train_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
        y_train = feat_df[train_mask]["FTR"].values
        X_val = feat_df[val_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
        y_val = feat_df[val_mask]["FTR"].values
        
        scaler = StandardScaler()
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=C, max_iter=2000)
        model.fit(scaler.fit_transform(X_train), y_train)
        
        proba = model.predict_proba(scaler.transform(X_val))
        proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
        losses.append(log_loss(y_val, proba_ord, labels=CLASSES))
    return np.mean(losses)

print(f"Data: {len(df)} matches, {len(FEATURE_COLS)} features")
print("Bayesian Optimization with L2 (150 iterations)...")

best_cv = [float('inf')]
best_p = [None]
count = [0]

@use_named_args(search_space)
def objective(home_adv, k_factor, k_att, k_def, C):
    count[0] += 1
    try:
        loss = walk_forward_cv(home_adv, k_factor, k_att, k_def, C)
        if loss < best_cv[0]:
            best_cv[0] = loss
            best_p[0] = {'home_adv': home_adv, 'k_factor': k_factor, 'k_att': k_att, 'k_def': k_def, 'C': C}
            print(f"[{count[0]:3d}] CV={loss:.6f} | HOME_ADV={home_adv}, K={k_factor}, K_ATT={k_att}, K_DEF={k_def}, C={C:.4f}")
        elif count[0] % 25 == 0:
            print(f"[{count[0]:3d}] Best CV: {best_cv[0]:.6f}")
        return loss
    except:
        return 1.5

result = gp_minimize(objective, search_space, n_calls=150, n_random_starts=30, random_state=42, verbose=False)

HOME_ADV, K_FACTOR, K_ATT, K_DEF, C = int(result.x[0]), int(result.x[1]), int(result.x[2]), int(result.x[3]), float(result.x[4])
print(f"\nBest: HOME_ADV={HOME_ADV}, K={K_FACTOR}, K_ATT={K_ATT}, K_DEF={K_DEF}, C={C:.4f}, CV={result.fun:.6f}")

print("\nTraining final model...")
train_seasons, test_seasons = SEASONS[:8], SEASONS[8:]
train_df = df[df["season"].isin(train_seasons)]
defaults = calculate_defaults(train_df)
feat_df = create_features(df, HOME_ADV, K_FACTOR, K_ATT, K_DEF, defaults)

train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_df[train_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = feat_df[train_mask]["FTR"].values
X_test = feat_df[test_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_test = feat_df[test_mask]["FTR"].values

scaler = StandardScaler()
model = LogisticRegression(penalty='l2', solver='lbfgs', C=C, max_iter=2000)
model.fit(scaler.fit_transform(X_train), y_train)

proba = model.predict_proba(scaler.transform(X_test))
proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
test_loss = log_loss(y_test, proba_ord, labels=CLASSES)
acc = accuracy_score(y_test, model.predict(scaler.transform(X_test)))
random_loss = np.log(3)

print(f"\nResults (L2 Regularization):")
print(f"  CV Loss:   {result.fun:.4f}")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Accuracy:  {acc*100:.1f}%")
print(f"  vs Random: {(random_loss - test_loss) / random_loss * 100:.2f}% better")

print("\nCalibration:")
for i, c in enumerate(CLASSES):
    actual = np.mean(y_test == c)
    pred = proba_ord[:, i].mean()
    print(f"  {c}: Pred={pred*100:.1f}%, Actual={actual*100:.1f}%")

with open("epl_model_l2.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "hyperparameters": {"HOME_ADV": HOME_ADV, "K_FACTOR": K_FACTOR, "K_ATT": K_ATT, "K_DEF": K_DEF, "C": C}, "defaults": defaults, "feature_cols": FEATURE_COLS, "classes": CLASSES, "cv_loss": result.fun, "test_loss": test_loss, "regularization": "L2"}, f)
print("\nSaved: epl_model_l2.pkl")
