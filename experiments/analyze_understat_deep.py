import pandas as pd
import numpy as np
import ast

print("DEEP ANALYSIS OF UNDERSTAT DATA")
print("-" * 60)

# Load data
shots = pd.read_csv('understat_data/epl_shots_all.csv')
matches = pd.read_csv('understat_data/epl_matches_all.csv')

print(f"Shots: {len(shots):,}")
print(f"Matches: {len(matches):,}")

# Parse match data - it has xG and forecast!
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}

matches['goals_parsed'] = matches['goals'].apply(safe_eval)
matches['xG_parsed'] = matches['xG'].apply(safe_eval)
matches['forecast_parsed'] = matches['forecast'].apply(safe_eval)
matches['h_parsed'] = matches['h'].apply(safe_eval)
matches['a_parsed'] = matches['a'].apply(safe_eval)

matches['h_goals'] = matches['goals_parsed'].apply(lambda x: int(x.get('h', 0)))
matches['a_goals'] = matches['goals_parsed'].apply(lambda x: int(x.get('a', 0)))
matches['h_xG'] = matches['xG_parsed'].apply(lambda x: float(x.get('h', 0)))
matches['a_xG'] = matches['xG_parsed'].apply(lambda x: float(x.get('a', 0)))
matches['forecast_h'] = matches['forecast_parsed'].apply(lambda x: float(x.get('w', 0)))
matches['forecast_d'] = matches['forecast_parsed'].apply(lambda x: float(x.get('d', 0)))
matches['forecast_a'] = matches['forecast_parsed'].apply(lambda x: float(x.get('l', 0)))
matches['h_team'] = matches['h_parsed'].apply(lambda x: x.get('title', ''))
matches['a_team'] = matches['a_parsed'].apply(lambda x: x.get('title', ''))
matches['date'] = pd.to_datetime(matches['datetime'])

print(f"\n1. UNDERSTAT XG (from match data)")
print("-" * 60)
print(f"Home xG mean: {matches['h_xG'].mean():.3f}")
print(f"Away xG mean: {matches['a_xG'].mean():.3f}")
print(f"Home Goals mean: {matches['h_goals'].mean():.3f}")
print(f"Away Goals mean: {matches['a_goals'].mean():.3f}")

# Check xG vs Goals correlation
print(f"\nCorrelation xG vs Goals:")
print(f"  Home: {matches['h_xG'].corr(matches['h_goals']):.4f}")
print(f"  Away: {matches['a_xG'].corr(matches['a_goals']):.4f}")

print(f"\n2. UNDERSTAT FORECAST (their prediction!)")
print("-" * 60)
print(f"Avg forecast Home win: {matches['forecast_h'].mean()*100:.1f}%")
print(f"Avg forecast Draw: {matches['forecast_d'].mean()*100:.1f}%")
print(f"Avg forecast Away win: {matches['forecast_a'].mean()*100:.1f}%")

# Calculate actual results
matches['result'] = np.where(matches['h_goals'] > matches['a_goals'], 'H',
                    np.where(matches['h_goals'] < matches['a_goals'], 'A', 'D'))

print(f"\nActual results:")
print(f"  Home win: {(matches['result'] == 'H').mean()*100:.1f}%")
print(f"  Draw: {(matches['result'] == 'D').mean()*100:.1f}%")
print(f"  Away win: {(matches['result'] == 'A').mean()*100:.1f}%")

print(f"\n3. EVALUATE UNDERSTAT FORECAST")
print("-" * 60)
from sklearn.metrics import log_loss, accuracy_score

# Prepare for log_loss
y_true = matches['result'].values
proba = matches[['forecast_a', 'forecast_d', 'forecast_h']].values

# Check for valid probabilities
valid_mask = (proba.sum(axis=1) > 0.99) & (proba.sum(axis=1) < 1.01)
print(f"Valid forecasts: {valid_mask.sum()}/{len(matches)}")

y_true_valid = y_true[valid_mask]
proba_valid = proba[valid_mask]

loss = log_loss(y_true_valid, proba_valid, labels=['A', 'D', 'H'])
print(f"Understat Forecast Log Loss: {loss:.4f}")

# Accuracy
pred = np.array(['A', 'D', 'H'])[proba_valid.argmax(axis=1)]
acc = accuracy_score(y_true_valid, pred)
print(f"Understat Forecast Accuracy: {acc*100:.1f}%")

print(f"\n4. COMPARE WITH OUR MODEL")
print("-" * 60)
print(f"Our model: Loss=0.9362, Acc=56.7%")
print(f"Understat: Loss={loss:.4f}, Acc={acc*100:.1f}%")

print(f"\n5. SAMPLE DATA")
print("-" * 60)
sample = matches[['date', 'h_team', 'a_team', 'h_goals', 'a_goals', 'h_xG', 'a_xG', 
                  'forecast_h', 'forecast_d', 'forecast_a', 'result']].head(10)
print(sample.to_string())

# Save clean match data
clean_matches = matches[['id', 'date', 'h_team', 'a_team', 'h_goals', 'a_goals', 
                         'h_xG', 'a_xG', 'forecast_h', 'forecast_d', 'forecast_a', 'result']]
clean_matches.to_csv('understat_data/matches_clean.csv', index=False)
print(f"\nSaved clean data to understat_data/matches_clean.csv")
