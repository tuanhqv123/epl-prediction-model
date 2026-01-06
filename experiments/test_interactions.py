"""
Test interaction features and non-linear transformations
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import log_loss, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load current best model features
with open("epl_model_final.pkl", "rb") as f:
    saved = pickle.load(f)

print("Current best model:")
print(f"  Loss: {saved['test_loss']:.4f}")
print(f"  Features: {len(saved['feature_cols'])}")

# Load data and recreate features
df = pd.read_csv("data_dowload_source/data_processed/all_seasons.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values(["season", "Date"]).reset_index(drop=True)

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
train_seasons = SEASONS[:8]
test_seasons = SEASONS[8:]
CLASSES = ["A", "D", "H"]
C = 0.15

# Run final_best_model to get features
exec(open("experiments/final_best_model.py").read().split("print(\"FINAL BEST MODEL\")")[0])

from experiments.final_best_model import create_features, defaults, FEATURE_COLS

feat_df = create_features(df, defaults)

train_mask = df["season"].isin(train_seasons).values
test_mask = df["season"].isin(test_seasons).values

X_train = feat_df[train_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = feat_df[train_mask]["FTR"].values
X_test = feat_df[test_mask][FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_test = feat_df[test_mask]["FTR"].values

print("\n" + "="*60)
print("TESTING POLYNOMIAL FEATURES")
print("="*60)

for degree in [2]:
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_poly)
    X_test_s = scaler.transform(X_test_poly)
    
    for c in [0.01, 0.05, 0.1, 0.15, 0.2]:
        model = LogisticRegression(penalty='l2', C=c, max_iter=3000)
        model.fit(X_train_s, y_train)
        
        proba = model.predict_proba(X_test_s)
        proba_ord = np.column_stack([proba[:, list(model.classes_).index(c_)] for c_ in CLASSES])
        
        loss = log_loss(y_test, proba_ord, labels=CLASSES)
        acc = accuracy_score(y_test, model.predict(X_test_s))
        
        print(f"Poly degree={degree}, C={c}: Loss={loss:.4f}, Acc={acc*100:.1f}%, Features={X_train_poly.shape[1]}")
