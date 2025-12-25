#!/usr/bin/env python3
"""
EPL PRODUCTION MODEL - PROBABILITY PREDICTION (v12)
====================================================
Dự đoán xác suất kết quả trận đấu EPL

BÀI TOÁN:
  - Input:  20 features (Elo, form, H2H, rest days, etc.)
  - Output: P(Away), P(Draw), P(Home) - 3 xác suất cộng = 100%

METRIC ĐÁNH GIÁ:
  - Log Loss (Cross-Entropy): Đo chất lượng xác suất
  - Công thức: Loss = -log(P của kết quả thực tế)
  - Thấp hơn = Tốt hơn

PERFORMANCE:
  Log Loss:         0.9363 (14.78% better than random 1.0986)
  Brier Score:      0.1845
  Calibration:      Excellent (all < 2% error)
    - Away:  Pred=32.5%, Actual=33.6%, Diff=1.1%
    - Draw:  Pred=23.1%, Actual=23.0%, Diff=0.0%
    - Home:  Pred=44.5%, Actual=43.4%, Diff=1.0%

PARAMETERS:
  K-factor:         32
  Home advantage:   240 Elo points
  Season decay:     1.0 (no decay)
  Regularization:   C=0.15
  H2H minimum:      7 matches
  NL window:        15 matches
  Rest cap:         7 days
  Lineup window:    5 matches
  Form window:      4 matches
  K_ATT:            10 (Attack Elo update rate)
  K_DEF:            30 (Defense Elo update rate)
  D_BOOST:          0.004 (post-hoc draw probability adjustment)
  H_REDUCE:         0.018 (correct home overconfidence)

FEATURES (20):
  1-4:   Elo features (home_elo, away_elo, elo_diff, elo_expected)
  5:     Attack/Defense matchup (att_def_matchup)
  6-9:   xG and draw features
  10-12: Non-league scoring
  13-15: Rest days
  16:    Lineup change rate diff
  17-18: Form features (winrate, clean sheet)
  19:    Shots diff
  20:    H2H features

Usage:
    python epl_production_final.py
"""

import json
import pickle
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
K_FACTOR = 32
HOME_ADV = 240
DECAY = 1.0
BEST_C = 0.15
H2H_MIN = 7
NL_WINDOW = 15
REST_CAP = 7
LINEUP_WINDOW = 5
FORM_WINDOW = 4
K_ATT = 10
K_DEF = 30
D_BOOST = 0.004  # Post-hoc draw probability adjustment
H_REDUCE = 0.018  # Correct home overconfidence

CLASSES = ["A", "D", "H"]
SEASONS = [
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

FEATURE_COLS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_expected",
    "att_def_matchup",  # Attack/Defense Elo matchup ★ NEW (Exp 28-29)
    "expected_diff",
    "combined_draw_rate",
    "h2h_draw_rate",
    "h2h_home_rate",
    "h2h_away_rate",
    "h_nl_scoring",
    "a_nl_scoring",
    "nl_scoring_diff",
    "h_rest",
    "a_rest",
    "rest_diff",
    "avg_change_rate_diff",  # Lineup change rate difference
    "winrate_3_diff",  # Win rate diff in last 4 matches
    "cs_3_diff",  # Clean sheet rate diff in last 4 matches
    "shots_diff",  # Total shots difference (last 3 matches)
]

# Team name mapping: Non-League → EPL
TEAM_MAPPING = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Burnley FC": "Burnley",
    "Cardiff City": "Cardiff",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool FC": "Liverpool",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Middlesbrough FC": "Middlesbrough",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Southampton FC": "Southampton",
    "Stoke City": "Stoke",
    "Sunderland AFC": "Sunderland",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "Watford FC": "Watford",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


# =============================================================================
# LOAD NON-LEAGUE DATA
# =============================================================================
def load_non_league_data():
    """Load and process non-league data"""
    try:
        non_league = pd.read_csv("epl_non_league_matches.csv")
        non_league["date"] = pd.to_datetime(non_league["date"], format="%d.%m.%Y")
        non_league = non_league.drop_duplicates(subset=["team", "date", "opponent"])

        # Parse goals from result_score
        def parse_goals(row):
            try:
                score = row["result_score"]
                if pd.isna(score) or ":" not in str(score):
                    return None
                parts = str(score).split(":")
                team_goals = int(parts[0])
                if row["home_away"] == "A":
                    team_goals = int(parts[1])
                return team_goals
            except:
                return None

        non_league["goals_scored"] = non_league.apply(parse_goals, axis=1)
        non_league["team_epl"] = non_league["team"].map(TEAM_MAPPING)

        # Build lookups
        nl_dates = {}  # For rest days
        nl_scoring = {}  # For scoring rates

        for _, row in non_league.iterrows():
            team_epl = row["team_epl"]
            if pd.isna(team_epl):
                continue

            # Dates for rest
            if team_epl not in nl_dates:
                nl_dates[team_epl] = []
            nl_dates[team_epl].append((row["date"], row["season"]))

            # Scoring (only if valid)
            if pd.notna(row["goals_scored"]):
                if team_epl not in nl_scoring:
                    nl_scoring[team_epl] = []
                nl_scoring[team_epl].append(
                    {
                        "date": row["date"],
                        "season": row["season"],
                        "scored": row["goals_scored"],
                    }
                )

        return nl_dates, nl_scoring
    except FileNotFoundError:
        print("    Warning: epl_non_league_matches.csv not found")
        return {}, {}


# =============================================================================
# LOAD LINEUP DATA
# =============================================================================
def parse_lineup(lineup_str):
    """Parse lineup string into set of player names."""
    if pd.isna(lineup_str) or lineup_str == '':
        return set()
    return set(p.strip() for p in lineup_str.split(';') if p.strip())


def load_lineup_data():
    """Load and process lineup data for change rate calculation."""
    try:
        lineup_df = pd.read_csv('epl_dynamic_data/all_matches.csv')
        lineup_df['HomeTeam'] = lineup_df['Home'].map(TEAM_MAPPING)
        lineup_df['AwayTeam'] = lineup_df['Away'].map(TEAM_MAPPING)
        lineup_df['Date'] = pd.to_datetime(lineup_df['Date']).dt.tz_localize(None)
        lineup_df = lineup_df.sort_values('Date').reset_index(drop=True)
        
        # Build lineup history per team: team -> [(date, lineup_set), ...]
        team_lineups = defaultdict(list)
        for _, row in lineup_df.iterrows():
            date = row['Date']
            home_lineup = parse_lineup(row['HomeP'])
            away_lineup = parse_lineup(row['AwayP'])
            if home_lineup:
                team_lineups[row['HomeTeam']].append((date, home_lineup))
            if away_lineup:
                team_lineups[row['AwayTeam']].append((date, away_lineup))
        
        # Build lookup for current match lineups
        lineup_lookup = {}
        for _, row in lineup_df.iterrows():
            key = (row['HomeTeam'], row['AwayTeam'], row['Date'].date())
            lineup_lookup[key] = {
                'home': parse_lineup(row['HomeP']),
                'away': parse_lineup(row['AwayP']),
            }
        
        return team_lineups, lineup_lookup
    except FileNotFoundError:
        print("    Warning: epl_dynamic_data/all_matches.csv not found")
        return {}, {}


def get_avg_change_rate(team, match_date, team_lineups, current_lineup):
    """
    Calculate average lineup change rate for a team over last LINEUP_WINDOW matches.
    
    Returns: float (0-1), where 0 = no changes, 1 = all players changed
    """
    history = team_lineups.get(team, [])
    
    # Get lineups BEFORE this match
    recent = [(d, l) for d, l in history if d < match_date][-LINEUP_WINDOW:]
    
    if not recent:
        return 3.0 / 11.0  # Default ~27%
    
    # Calculate change rates between consecutive matches
    change_rates = []
    for i in range(1, len(recent)):
        prev_lineup = recent[i-1][1]
        curr_lineup = recent[i][1]
        if prev_lineup and curr_lineup:
            changes = len(curr_lineup - prev_lineup)
            change_rates.append(changes / 11.0)
    
    # Also include change from last historical to current
    if current_lineup and recent:
        last_lineup = recent[-1][1]
        if last_lineup:
            changes = len(current_lineup - last_lineup)
            change_rates.append(changes / 11.0)
    
    return np.mean(change_rates) if change_rates else 3.0 / 11.0


# =============================================================================
# FORM FEATURES
# =============================================================================
def get_winrate(results, window):
    """Calculate win rate in last N matches."""
    if not results:
        return 0.33
    recent = results[-window:]
    return sum(1 for r in recent if r == 'W') / len(recent)


def get_clean_sheet_rate(ga_list, window):
    """Calculate clean sheet rate in last N matches."""
    if not ga_list:
        return 0.33
    recent = ga_list[-window:]
    return sum(1 for g in recent if g == 0) / len(recent)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_features(df, nl_dates, nl_scoring, team_lineups, lineup_lookup):
    """Create optimized 20-feature set with no data leakage"""
    elo, team_stats, h2h_history = {}, {}, {}
    elo_att, elo_def = {}, {}  # Attack/Defense Elo ratings
    team_last_epl = {}  # Track last EPL match for rest calculation
    current_season = None
    features = []

    for _, match in df.iterrows():
        season = match["season"]
        match_date = match["Date"]

        if season != current_season:
            if current_season:
                for t in elo:
                    elo[t] = 1500 + DECAY * (elo[t] - 1500)
            current_season = season
            team_stats = {}
            team_last_epl = {}

        home, away = match["HomeTeam"], match["AwayTeam"]
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV) / 400))
        
        # Attack/Defense Elo
        h_att = elo_att.get(home, 1500)
        h_def = elo_def.get(home, 1500)
        a_att = elo_att.get(away, 1500)
        a_def = elo_def.get(away, 1500)
        
        # Attack vs Defense matchup
        h_att_vs_a_def = h_att - a_def + HOME_ADV/2
        a_att_vs_h_def = a_att - h_def - HOME_ADV/2
        att_def_matchup = h_att_vs_a_def - a_att_vs_h_def

        h_hist = team_stats.get(home, {"sot": [], "draw": [], "results": [], "ga": [], "shots": []})
        a_hist = team_stats.get(away, {"sot": [], "draw": [], "results": [], "ga": [], "shots": []})
        h2h_key = tuple(sorted([home, away]))
        h2h = h2h_history.get(h2h_key, [])

        # Calculate features from PREVIOUS matches only
        home_xg = np.mean(h_hist["sot"][-3:]) * 0.35 if h_hist["sot"] else 1.3
        away_xg = np.mean(a_hist["sot"][-3:]) * 0.35 if a_hist["sot"] else 1.3
        home_draw = np.mean(h_hist["draw"][-5:]) if h_hist["draw"] else 0.27
        away_draw = np.mean(a_hist["draw"][-5:]) if a_hist["draw"] else 0.27
        h2h_draw = np.mean([1 if m == "D" else 0 for m in h2h]) if h2h else 0.27

        if len(h2h) >= H2H_MIN:
            h2h_home = sum(1 for x in h2h if x == "H") / len(h2h)
            h2h_away = sum(1 for x in h2h if x == "A") / len(h2h)
        else:
            h2h_home, h2h_away = 0.45, 0.30

        # Rest days calculation
        def get_rest(team, match_date, season):
            # EPL rest
            rest_epl = team_last_epl.get(team)
            rest_epl = (match_date - rest_epl).days if rest_epl else 14

            # Non-league rest
            rest_nl = 14
            if team in nl_dates:
                for nl_date, nl_season in nl_dates[team]:
                    if nl_season == season and nl_date < match_date:
                        days = (match_date - nl_date).days
                        if 0 < days < rest_nl:
                            rest_nl = days

            return min(rest_epl, rest_nl, REST_CAP)

        h_rest = get_rest(home, match_date, season)
        a_rest = get_rest(away, match_date, season)

        # Non-league scoring rates
        def get_nl_scoring(team, match_date, season):
            if team not in nl_scoring:
                return 1.5
            recent = [
                s
                for s in nl_scoring[team]
                if s["season"] == season and s["date"] < match_date
            ]
            recent = sorted(recent, key=lambda x: x["date"], reverse=True)[:NL_WINDOW]
            return np.mean([s["scored"] for s in recent]) if recent else 1.5

        h_nl = get_nl_scoring(home, match_date, season)
        a_nl = get_nl_scoring(away, match_date, season)

        # Lineup change rate calculation
        lookup_key = (home, away, match_date.date())
        current_lineups = lineup_lookup.get(lookup_key, {'home': set(), 'away': set()})
        h_change_rate = get_avg_change_rate(home, match_date, team_lineups, current_lineups['home'])
        a_change_rate = get_avg_change_rate(away, match_date, team_lineups, current_lineups['away'])
        avg_change_rate_diff = h_change_rate - a_change_rate

        feat = {
            "season": season,
            "Date": match["Date"],
            "HomeTeam": home,
            "AwayTeam": away,
            "FTR": match["FTR"],
            "FTHG": match["FTHG"],
            "FTAG": match["FTAG"],
            # Elo features (4)
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": h_elo - a_elo + HOME_ADV,
            "elo_expected": exp_h,
            # Attack/Defense matchup (1) ★ NEW
            "att_def_matchup": att_def_matchup,
            # xG and draw features (5)
            "expected_diff": home_xg - away_xg,
            "combined_draw_rate": (home_draw + away_draw) / 2,
            "h2h_draw_rate": h2h_draw,
            "h2h_home_rate": h2h_home,
            "h2h_away_rate": h2h_away,
            # NL scoring features (3)
            "h_nl_scoring": h_nl,
            "a_nl_scoring": a_nl,
            "nl_scoring_diff": h_nl - a_nl,
            # Rest features (3)
            "h_rest": h_rest,
            "a_rest": a_rest,
            "rest_diff": h_rest - a_rest,
            # Lineup feature (1)
            "avg_change_rate_diff": avg_change_rate_diff,
            # Form features (2)
            "winrate_3_diff": get_winrate(h_hist["results"], FORM_WINDOW) - get_winrate(a_hist["results"], FORM_WINDOW),
            "cs_3_diff": get_clean_sheet_rate(h_hist["ga"], FORM_WINDOW) - get_clean_sheet_rate(a_hist["ga"], FORM_WINDOW),
            # Shots feature (1) - NEW
            "shots_diff": (np.mean(h_hist["shots"][-3:]) if h_hist["shots"] else 12) - 
                         (np.mean(a_hist["shots"][-3:]) if a_hist["shots"] else 12),
        }
        features.append(feat)

        # Update AFTER feature extraction (no leakage)
        actual = 1 if match["FTR"] == "H" else (0 if match["FTR"] == "A" else 0.5)
        elo[home] = h_elo + K_FACTOR * (actual - exp_h)
        elo[away] = a_elo + K_FACTOR * ((1 - actual) - (1 - exp_h))

        # Update Attack/Defense Elo based on goals
        h_goals, a_goals = match["FTHG"], match["FTAG"]
        elo_att[home] = h_att + K_ATT * (h_goals - 1.5) / 3
        elo_att[away] = a_att + K_ATT * (a_goals - 1.2) / 3
        elo_def[home] = h_def + K_DEF * (1.2 - a_goals) / 3
        elo_def[away] = a_def + K_DEF * (1.5 - h_goals) / 3

        if h2h_key not in h2h_history:
            h2h_history[h2h_key] = []
        h2h_history[h2h_key].append(match["FTR"])

        for team, is_home in [(home, True), (away, False)]:
            if team not in team_stats:
                team_stats[team] = {"sot": [], "draw": [], "results": [], "ga": [], "shots": []}
            team_stats[team]["sot"].append(match["HST"] if is_home else match["AST"])
            team_stats[team]["draw"].append(1 if match["FTR"] == "D" else 0)
            team_stats[team]["ga"].append(match["FTAG"] if is_home else match["FTHG"])
            team_stats[team]["shots"].append(match["HS"] if is_home else match["AS"])
            # Result from team's perspective
            if is_home:
                result = 'W' if match['FTR'] == 'H' else ('D' if match['FTR'] == 'D' else 'L')
            else:
                result = 'W' if match['FTR'] == 'A' else ('D' if match['FTR'] == 'D' else 'L')
            team_stats[team]["results"].append(result)

        # Update last EPL match
        team_last_epl[home] = match_date
        team_last_epl[away] = match_date

    return pd.DataFrame(features)


def get_ordered_proba(model, X):
    proba = model.predict_proba(X)
    return np.column_stack([proba[:, list(model.classes_).index(c)] for c in CLASSES])


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EPL PROBABILITY PREDICTION MODEL")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading data...")
    df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values(["season", "Date"]).reset_index(drop=True)
    print(f"    EPL matches: {len(df)}")

    nl_dates, nl_scoring = load_non_league_data()
    print(f"    Non-league teams: {len(nl_dates)}")

    team_lineups, lineup_lookup = load_lineup_data()
    print(f"    Lineup data teams: {len(team_lineups)}")

    # 2. Show parameters
    print(f"\n[2] Parameters:")
    print(f"    K={K_FACTOR}, HA={HOME_ADV}, C={BEST_C}")
    print(f"    H2H_MIN={H2H_MIN}, NL_WINDOW={NL_WINDOW}, REST_CAP={REST_CAP}")
    print(f"    LINEUP_WINDOW={LINEUP_WINDOW}, FORM_WINDOW={FORM_WINDOW}")
    print(f"    K_ATT={K_ATT}, K_DEF={K_DEF}")
    print(f"    D_BOOST={D_BOOST}, H_REDUCE={H_REDUCE}")

    # 3. Feature engineering
    print(f"\n[3] Engineering {len(FEATURE_COLS)} features...")
    features_df = create_features(df, nl_dates, nl_scoring, team_lineups, lineup_lookup)

    # 4. Temporal split
    print("\n[4] Temporal split...")
    train_seasons = SEASONS[:6]
    valid_seasons = SEASONS[6:8]
    test_seasons = SEASONS[8:]

    train_df = features_df[features_df["season"].isin(train_seasons)]
    valid_df = features_df[features_df["season"].isin(valid_seasons)]
    test_df = features_df[features_df["season"].isin(test_seasons)]

    print(f"    Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

    def prepare(df):
        X = (
            df[FEATURE_COLS]
            .fillna(df[FEATURE_COLS].median())
            .replace([np.inf, -np.inf], 0)
        )
        return X.values, df["FTR"].values

    X_train, y_train = prepare(train_df)
    X_valid, y_valid = prepare(valid_df)
    X_test, y_test = prepare(test_df)

    # 5. Train model
    print("\n[5] Training model...")
    X_train_full = np.vstack([X_train, X_valid])
    y_train_full = np.concatenate([y_train, y_valid])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=BEST_C)
    model.fit(X_train_s, y_train_full)
    print(f"    ✓ Model trained")

    # 6. Evaluate
    print("\n[6] Evaluation...")
    y_proba_raw = get_ordered_proba(model, X_test_s)
    
    # Apply probability adjustments
    y_proba = y_proba_raw.copy()
    y_proba[:, 1] += D_BOOST
    y_proba[:, 2] -= H_REDUCE
    y_proba = np.clip(y_proba, 0.001, 0.999)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    # Calculate metrics
    ll = log_loss(y_test, y_proba, labels=CLASSES)
    random_ll = log_loss(y_test, np.ones((len(y_test), 3)) / 3, labels=CLASSES)
    brier = (
        sum(
            brier_score_loss((y_test == c).astype(int), y_proba[:, i])
            for i, c in enumerate(CLASSES)
        )
        / 3
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"""
  📊 Log Loss:        {ll:.4f}
  📊 Random baseline: {random_ll:.4f}
  📊 Improvement:     {(random_ll - ll) / random_ll * 100:.2f}% better than random
  📊 Brier Score:     {brier:.4f}
""")

    print("CALIBRATION:")
    for i, cls in enumerate(CLASSES):
        actual = np.mean(y_test == cls)
        pred = y_proba[:, i].mean()
        diff = abs(pred - actual)
        print(
            f"  {cls}: Pred={pred*100:.1f}%, Actual={actual*100:.1f}%, Diff={diff*100:.1f}% {'✓' if diff < 0.02 else '⚠'}"
        )

    print("\nPER-SEASON LOG LOSS:")
    for season in test_seasons:
        mask = test_df["season"].values == season
        if mask.sum() > 0:
            ll_s = log_loss(y_test[mask], y_proba[mask], labels=CLASSES)
            print(f"  {season}: {ll_s:.4f}")

    # 7. Save model
    print("\n[7] Saving model...")
    with open("epl_probability_model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_cols": FEATURE_COLS,
                "params": {
                    "K_FACTOR": K_FACTOR,
                    "HOME_ADV": HOME_ADV,
                    "DECAY": DECAY,
                    "C": BEST_C,
                    "H2H_MIN": H2H_MIN,
                    "NL_WINDOW": NL_WINDOW,
                    "REST_CAP": REST_CAP,
                    "LINEUP_WINDOW": LINEUP_WINDOW,
                    "FORM_WINDOW": FORM_WINDOW,
                    "K_ATT": K_ATT,
                    "K_DEF": K_DEF,
                    "D_BOOST": D_BOOST,
                    "H_REDUCE": H_REDUCE,
                },
                "classes": CLASSES,
                "team_mapping": TEAM_MAPPING,
            },
            f,
        )

    with open("epl_probability_model_config.json", "w") as f:
        json.dump(
            {
                "model_type": "logistic_regression_v12",
                "C": BEST_C,
                "n_features": len(FEATURE_COLS),
                "features": FEATURE_COLS,
                "elo_params": {
                    "k_factor": K_FACTOR,
                    "home_adv": HOME_ADV,
                    "decay": DECAY,
                    "k_att": K_ATT,
                    "k_def": K_DEF,
                },
                "d_boost": D_BOOST,
                "h_reduce": H_REDUCE,
                "h2h_min": H2H_MIN,
                "nl_window": NL_WINDOW,
                "rest_cap": REST_CAP,
                "lineup_window": LINEUP_WINDOW,
                "form_window": FORM_WINDOW,
                "metrics": {
                    "log_loss": float(ll),
                    "brier_score": float(brier),
                    "improvement_vs_random": float((random_ll - ll) / random_ll * 100),
                },
                "train_seasons": train_seasons + valid_seasons,
                "test_seasons": test_seasons,
            },
            f,
            indent=2,
        )
    print("    ✓ Saved to epl_probability_model.pkl")

    # 8. Calibration curves
    print("\n[8] Generating calibration curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, cls in enumerate(CLASSES):
        y_true_binary = (y_test == cls).astype(int)
        prob_true, prob_pred = calibration_curve(
            y_true_binary, y_proba[:, i], n_bins=10
        )
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(prob_pred, prob_true, "o-", label=f"{cls}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration - {cls}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("epl_calibration_curves.png", dpi=150)
    print("    ✓ Saved to epl_calibration_curves.png")

    print("\n" + "=" * 70)
    print("✓ Production model ready")
    print("=" * 70)
