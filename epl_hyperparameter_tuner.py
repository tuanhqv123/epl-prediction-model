import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("EPL HYPERPARAMETER TUNER")
print("Bayesian Optimization + Walk-Forward CV")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

SEASONS = [
    "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025",
]
CLASSES = ["A", "D", "H"]
print(f"Loaded {len(df)} matches")

# =============================================================================
# SEARCH SPACE - Chỉ định nghĩa RANGE, không hardcode giá trị
# =============================================================================
search_space = [
    Integer(50, 400, name='home_adv'),
    Integer(10, 50, name='k_factor'),
    Integer(5, 30, name='k_att'),
    Integer(20, 80, name='k_def'),
    Real(0.01, 2.0, prior='log-uniform', name='C'),
]

print("\nSearch Space:")
for s in search_space:
    print(f"  {s.name}: {s.low} - {s.high}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calculate_defaults(df_subset):
    defaults = {}
    total = len(df_subset)
    defaults['home_win_rate'] = (df_subset['FTR'] == 'H').sum() / total
    defaults['draw_rate'] = (df_subset['FTR'] == 'D').sum() / total
    defaults['avg_home_goals'] = df_subset['FTHG'].mean()
    defaults['avg_away_goals'] = df_subset['FTAG'].mean()
    defaults['goal_std'] = (df_subset['FTHG'].std() + df_subset['FTAG'].std()) / 2
    return defaults

def create_features(df, home_adv, k_factor, k_att, k_def, defaults):
    elo, elo_att, elo_def = {}, {}, {}
    features = []
    
    avg_home_goals = defaults['avg_home_goals']
    avg_away_goals = defaults['avg_away_goals']
    goal_std = defaults['goal_std']
    
    for _, match in df.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        h_elo = elo.get(home, 1500)
        a_elo = elo.get(away, 1500)
        h_att, h_def = elo_att.get(home, 1500), elo_def.get(home, 1500)
        a_att, a_def = elo_att.get(away, 1500), elo_def.get(away, 1500)
        
        elo_diff = h_elo - a_elo + home_adv
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / 400))
        att_def = (h_att - a_def + home_adv/2) - (a_att - h_def - home_adv/2)
        
        features.append({
            'season': match['season'],
            'elo_diff': elo_diff, 'elo_expected': exp_h,
            'att_def_matchup': att_def,
            'home_elo': h_elo, 'away_elo': a_elo,
            'FTR': match['FTR']
        })
        
        actual = 1 if match["FTR"] == "H" else (0 if match["FTR"] == "A" else 0.5)
        elo[home] = h_elo + k_factor * (actual - exp_h)
        elo[away] = a_elo + k_factor * ((1 - actual) - (1 - exp_h))
        
        h_goals, a_goals = match["FTHG"], match["FTAG"]
        elo_att[home] = h_att + k_att * (h_goals - avg_home_goals) / goal_std
        elo_def[home] = h_def + k_def * (avg_away_goals - a_goals) / goal_std
        elo_att[away] = a_att + k_att * (a_goals - avg_away_goals) / goal_std
        elo_def[away] = a_def + k_def * (avg_home_goals - h_goals) / goal_std
    
    return pd.DataFrame(features)

FEATURE_COLS = ['elo_diff', 'elo_expected', 'att_def_matchup', 'home_elo', 'away_elo']

def walk_forward_cv(home_adv, k_factor, k_att, k_def, C):
    folds = [
        (["2015-2016", "2016-2017", "2017-2018"], ["2018-2019"]),
        (["2015-2016", "2016-2017", "2017-2018", "2018-2019"], ["2019-2020"]),
        (["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020"], ["2020-2021"]),
        (["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021"], ["2021-2022"]),
        (["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022"], ["2022-2023"]),
    ]
    
    fold_losses = []
    for train_seasons, val_seasons in folds:
        train_df = df[df["season"].isin(train_seasons)]
        defaults = calculate_defaults(train_df)
        
        all_df = df[df["season"].isin(train_seasons + val_seasons)]
        features_df = create_features(all_df, home_adv, k_factor, k_att, k_def, defaults)
        
        train_mask = all_df["season"].isin(train_seasons).values
        val_mask = all_df["season"].isin(val_seasons).values
        
        X_train = features_df[train_mask][FEATURE_COLS].fillna(0).values
        y_train = features_df[train_mask]["FTR"].values
        X_val = features_df[val_mask][FEATURE_COLS].fillna(0).values
        y_val = features_df[val_mask]["FTR"].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        model = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=2000)
        model.fit(X_train_s, y_train)
        
        y_proba = model.predict_proba(X_val_s)
        proba_ordered = np.column_stack([y_proba[:, list(model.classes_).index(c)] for c in CLASSES])
        fold_losses.append(log_loss(y_val, proba_ordered, labels=CLASSES))
    
    return np.mean(fold_losses)

# =============================================================================
# BAYESIAN OPTIMIZATION
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING BAYESIAN OPTIMIZATION (100 iterations)")
print("=" * 70)

iteration_count = [0]
best_loss = [float('inf')]

@use_named_args(search_space)
def objective(home_adv, k_factor, k_att, k_def, C):
    iteration_count[0] += 1
    try:
        loss = walk_forward_cv(home_adv, k_factor, k_att, k_def, C)
        if loss < best_loss[0]:
            best_loss[0] = loss
            print(f"[{iteration_count[0]:3d}] NEW BEST: {loss:.6f} | HOME_ADV={home_adv}, K={k_factor}, K_ATT={k_att}, K_DEF={k_def}, C={C:.4f}")
        elif iteration_count[0] % 20 == 0:
            print(f"[{iteration_count[0]:3d}] Current: {loss:.6f}")
        return loss
    except:
        return 1.5

result = gp_minimize(objective, search_space, n_calls=100, n_random_starts=20, random_state=42, verbose=False)

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("OPTIMAL HYPERPARAMETERS (found by Bayesian Optimization)")
print("=" * 70)

optimal_params = {
    'HOME_ADV': int(result.x[0]),
    'K_FACTOR': int(result.x[1]),
    'K_ATT': int(result.x[2]),
    'K_DEF': int(result.x[3]),
    'C': float(result.x[4])
}

for k, v in optimal_params.items():
    print(f"  {k}: {v}")
print(f"\nCV Loss: {result.fun:.6f}")

# Save
output = {
    'optimal_hyperparameters': optimal_params,
    'cv_loss': float(result.fun),
    'method': 'Bayesian Optimization + Walk-Forward CV',
    'n_iterations': 100,
    'search_space': {
        'HOME_ADV': [50, 400],
        'K_FACTOR': [10, 50],
        'K_ATT': [5, 30],
        'K_DEF': [20, 80],
        'C': [0.01, 2.0]
    }
}

with open('epl_optimal_hyperparameters.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to: epl_optimal_hyperparameters.json")
print("\nĐể sử dụng trong production:")
print("  from epl_hyperparameter_tuner import optimal_params")
print("  hoặc load từ epl_optimal_hyperparameters.json")
