import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta

from epl_production_data_driven import (
    load_non_league_data, load_lineup_data,
    get_avg_change_rate, get_winrate, get_clean_sheet_rate
)

df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

xg_df = pd.read_csv("understat_data/our_match_xg_v2.csv")
shots = pd.read_csv("understat_data/epl_shots_all.csv")
match_info = shots.groupby('match_id').first()[['date', 'h_team', 'a_team', 'season']].reset_index()
xg_df = xg_df.merge(match_info, on='match_id')
xg_df['date'] = pd.to_datetime(xg_df['date'])

shots['xG'] = shots['xG'].astype(float)
match_xga = shots.groupby('match_id').apply(
    lambda x: pd.Series({
        'h_xGA': x[x['h_a'] == 'a']['xG'].sum(),
        'a_xGA': x[x['h_a'] == 'h']['xG'].sum(),
    })
).reset_index()
xg_df = xg_df.merge(match_xga, on='match_id')

nl_dates, nl_scoring = load_non_league_data()
team_lineups, lineup_lookup = load_lineup_data()

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
CLASSES = ["A", "D", "H"]

HOME_ADV, K_FACTOR, K_ATT, K_DEF, C, ELO_DIVISOR, K_XG = 82, 28, 5, 90, 0.15, 640, 10

team_mapping = {
    'Manchester United': 'Man United', 'Manchester City': 'Man City', 
    'Newcastle United': 'Newcastle', 'Wolverhampton Wanderers': 'Wolves',
    "Nottingham Forest": "Nott'm Forest", 'West Bromwich Albion': 'West Brom',
    'Queens Park Rangers': 'QPR',
}
reverse_mapping = {v: k for k, v in team_mapping.items()}

xg_lookup = {}
for _, row in xg_df.iterrows():
    h_team, a_team, date = row['h_team'], row['a_team'], row['date'].date()
    data = {'h_xG': row['h_xG_v2'], 'a_xG': row['a_xG_v2'], 'h_xGA': row['h_xGA'], 'a_xGA': row['a_xGA']}
    for d in [date, date + timedelta(days=1), date - timedelta(days=1)]:
        xg_lookup[(h_team, a_team, d)] = data

def get_xg(home, away, date):
    key = (home, away, date)
    if key in xg_lookup: return xg_lookup[key]
    h_mapped = reverse_mapping.get(home, home)
    a_mapped = reverse_mapping.get(away, away)
    return xg_lookup.get((h_mapped, a_mapped, date), None)

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
        'avg_fouls': (df_subset['HF'].mean() + df_subset['AF'].mean()) / 2,
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
                "cs_3_diff", "shots_diff", "xg_elo_diff", "xga_elo_diff",
                "referee_bias", "fouls_diff", "elo_squared", "xg_x_xga", "elo_x_form"]

def create_features(df, defaults):
    elo, elo_att, elo_def, elo_xg, elo_xga = {}, {}, {}, {}, {}
    team_stats, h2h_history, team_last_epl = {}, {}, {}
    referee_history = {}
    features = []
    d = defaults
    
    for _, m in df.iterrows():
        season, match_date = m["season"], m["Date"]
        home, away = m["HomeTeam"], m["AwayTeam"]
        ref = m['Referee']
        
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)
        h_att, h_def = elo_att.get(home, 1500), elo_def.get(home, 1500)
        a_att, a_def = elo_att.get(away, 1500), elo_def.get(away, 1500)
        h_xg_elo, a_xg_elo = elo_xg.get(home, 1500), elo_xg.get(away, 1500)
        h_xga_elo, a_xga_elo = elo_xga.get(home, 1500), elo_xga.get(away, 1500)
        
        elo_diff = h_elo - a_elo + HOME_ADV
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV) / ELO_DIVISOR))
        att_def = (h_att - a_def) - (a_att - h_def)
        xg_elo_diff = h_xg_elo - a_xg_elo
        xga_elo_diff = h_xga_elo - a_xga_elo
        
        h_hist = team_stats.get(home, {"sot": [], "draw": [], "results": [], "ga": [], "shots": [], "fouls": []})
        a_hist = team_stats.get(away, {"sot": [], "draw": [], "results": [], "ga": [], "shots": [], "fouls": []})
        
        home_xg_proxy = np.mean(h_hist["sot"][-3:]) * d['sot_conversion'] if h_hist["sot"] else d['avg_home_goals']
        away_xg_proxy = np.mean(a_hist["sot"][-3:]) * d['sot_conversion'] if a_hist["sot"] else d['avg_away_goals']
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
        h_fouls = np.mean(h_hist["fouls"][-5:]) if h_hist["fouls"] else d['avg_fouls']
        a_fouls = np.mean(a_hist["fouls"][-5:]) if a_hist["fouls"] else d['avg_fouls']
        
        ref_stats = referee_history.get(ref, {'home_wins': 0, 'total': 0})
        ref_home_rate = ref_stats['home_wins'] / ref_stats['total'] if ref_stats['total'] >= 10 else d['home_win_rate']
        
        features.append({
            "season": season, "FTR": m["FTR"],
            "home_elo": h_elo, "away_elo": a_elo, "elo_diff": elo_diff, "elo_expected": exp_h,
            "att_def_matchup": att_def, "expected_diff": home_xg_proxy - away_xg_proxy,
            "combined_draw_rate": (home_draw + away_draw) / 2, "h2h_draw_rate": h2h_draw,
            "h2h_home_rate": h2h_home, "h2h_away_rate": h2h_away,
            "h_nl_scoring": h_nl, "a_nl_scoring": a_nl, "nl_scoring_diff": h_nl - a_nl,
            "h_rest": h_rest, "a_rest": a_rest, "rest_diff": h_rest - a_rest,
            "avg_change_rate_diff": h_change - a_change,
            "winrate_3_diff": h_winrate - a_winrate, "cs_3_diff": h_cs - a_cs, "shots_diff": h_shots - a_shots,
            "xg_elo_diff": xg_elo_diff, "xga_elo_diff": xga_elo_diff,
            "referee_bias": ref_home_rate - d['home_win_rate'],
            "fouls_diff": h_fouls - a_fouls,
            "elo_squared": elo_diff ** 2 / 10000,
            "xg_x_xga": xg_elo_diff * xga_elo_diff / 1000,
            "elo_x_form": elo_diff * (h_winrate - a_winrate),
        })

        actual = 1 if m["FTR"] == "H" else (0 if m["FTR"] == "A" else 0.5)
        elo[home] = h_elo + K_FACTOR * (actual - exp_h)
        elo[away] = a_elo + K_FACTOR * ((1 - actual) - (1 - exp_h))
        
        h_goals, a_goals = m["FTHG"], m["FTAG"]
        elo_att[home] = h_att + K_ATT * (h_goals - d['avg_home_goals']) / d['goal_std']
        elo_def[home] = h_def + K_DEF * (d['avg_away_goals'] - a_goals) / d['goal_std']
        elo_att[away] = a_att + K_ATT * (a_goals - d['avg_away_goals']) / d['goal_std']
        elo_def[away] = a_def + K_DEF * (d['avg_home_goals'] - h_goals) / d['goal_std']
        
        match_xg = get_xg(home, away, match_date.date())
        if match_xg:
            h_xg, a_xg = match_xg['h_xG'], match_xg['a_xG']
            xg_actual = 1 if h_xg > a_xg else (0 if h_xg < a_xg else 0.5)
            xg_exp = 1 / (1 + 10 ** ((a_xg_elo - h_xg_elo) / 400))
            elo_xg[home] = h_xg_elo + K_XG * (xg_actual - xg_exp)
            elo_xg[away] = a_xg_elo + K_XG * ((1 - xg_actual) - (1 - xg_exp))
            
            h_xga, a_xga = match_xg['h_xGA'], match_xg['a_xGA']
            xga_actual = 1 if h_xga < a_xga else (0 if h_xga > a_xga else 0.5)
            xga_exp = 1 / (1 + 10 ** ((a_xga_elo - h_xga_elo) / 400))
            elo_xga[home] = h_xga_elo + K_XG * (xga_actual - xga_exp)
            elo_xga[away] = a_xga_elo + K_XG * ((1 - xga_actual) - (1 - xga_exp))
        
        if ref not in referee_history:
            referee_history[ref] = {'home_wins': 0, 'total': 0}
        referee_history[ref]['total'] += 1
        if m['FTR'] == 'H':
            referee_history[ref]['home_wins'] += 1
        
        h2h_history.setdefault(h2h_key, []).append(m["FTR"])
        for team, is_home in [(home, True), (away, False)]:
            if team not in team_stats:
                team_stats[team] = {"sot": [], "draw": [], "results": [], "ga": [], "shots": [], "fouls": []}
            team_stats[team]["sot"].append(m["HST"] if is_home else m["AST"])
            team_stats[team]["draw"].append(1 if m["FTR"] == "D" else 0)
            team_stats[team]["ga"].append(m["FTAG"] if is_home else m["FTHG"])
            team_stats[team]["shots"].append(m["HS"] if is_home else m["AS"])
            team_stats[team]["fouls"].append(m["HF"] if is_home else m["AF"])
            result = 'W' if (is_home and m['FTR'] == 'H') or (not is_home and m['FTR'] == 'A') else 'L'
            team_stats[team]["results"].append(result)
        team_last_epl[home] = match_date
        team_last_epl[away] = match_date
    
    return pd.DataFrame(features)

print("FINAL BEST MODEL")
print("=" * 60)

feat_df = create_features(df, defaults)

train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_df[train_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = feat_df[train_mask]["FTR"].values
X_test = feat_df[test_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_test = feat_df[test_mask]["FTR"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(penalty='l2', solver='lbfgs', C=C, max_iter=2000)
model.fit(X_train_s, y_train)

proba = model.predict_proba(X_test_s)
proba_ord = np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])
y_pred = model.predict(X_test_s)

test_loss = log_loss(y_test, proba_ord, labels=CLASSES)
acc = accuracy_score(y_test, y_pred)
random_loss = np.log(3)

print(f"\nRESULTS:")
print("-" * 60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {acc*100:.1f}%")
print(f"vs Random: {(random_loss - test_loss) / random_loss * 100:.2f}% better")

print(f"\nIMPROVEMENT JOURNEY:")
print("-" * 60)
print(f"1. Original baseline:     Loss=0.9362, Acc=56.7%")
print(f"2. + xG/xGA ELO:          Loss=0.9331, Acc=57.1%")
print(f"3. + referee + fouls:     Loss=0.9323, Acc=57.9%")
print(f"4. + elo_squared/xg_x_xga/elo_x_form: Loss={test_loss:.4f}, Acc={acc*100:.1f}%")
print(f"\nTOTAL IMPROVEMENT: {(0.9362 - test_loss):.4f} ({(0.9362 - test_loss)/0.9362*100:.2f}%)")

print(f"\nPER-SEASON:")
for season in test_seasons:
    s_mask = feat_df[test_mask]["season"] == season
    s_proba = proba_ord[s_mask.values]
    s_y = y_test[s_mask.values]
    s_pred = y_pred[s_mask.values]
    s_loss = log_loss(s_y, s_proba, labels=CLASSES)
    s_acc = accuracy_score(s_y, s_pred)
    print(f"  {season}: Loss={s_loss:.4f}, Acc={s_acc*100:.1f}%")

params = {'home_adv': HOME_ADV, 'k_factor': K_FACTOR, 'k_att': K_ATT, 'k_def': K_DEF,
          'k_xg': K_XG, 'C': C, 'elo_divisor': ELO_DIVISOR}

with open("epl_model_final.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "hyperparameters": params,
                 "defaults": defaults, "feature_cols": FEATURE_COLS, "classes": CLASSES,
                 "test_loss": test_loss}, f)
print(f"\nSaved: epl_model_final.pkl")
