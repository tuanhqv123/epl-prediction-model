# EPL Prediction Model

English Premier League match outcome prediction using machine learning.

## 🎯 Model Performance

### Current Model (Final - Correct)
```
Test Loss:       0.9323
Accuracy:        57.9%
vs Random:       15.13% better
```

### Improvement Journey (All Verified - No Data Leakage)
| Step | Feature Added | Loss | Improvement |
|------|---------------|------|-------------|
| 1 | Baseline | 0.9362 | - |
| 2 | + xG ELO | 0.9333 | +0.0029 |
| 3 | + xGA ELO | 0.9331 | +0.0002 |
| 4 | + referee_bias | 0.9326 | +0.0005 |
| 5 | + fouls_diff | 0.9323 | +0.0003 |
| **Total** | All features | **0.9323** | **+0.0039 (0.41%)** |

### Per-Season Performance
```
2023-2024: Loss=0.8992, Acc=61.1%
2024-2025: Loss=0.9655, Acc=54.7%
```

---

## 📁 Project Structure

```
epl-prediction-model/
├── data_dowload_source/       # Downloaded and processed EPL data
├── understat_data/            # Shot data from Understat (106,924 shots)
├── epl_dynamic_data/          # Scraped match data
├── scrape/                    # Web scraping scripts
├── experiments/               # All experiments and analysis
│   ├── final_model_correct.py # Final model (use this!)
│   ├── xg_with_lastaction.py  # xG model from raw data
│   └── test_*.py              # Feature testing scripts
├── epl_production_data_driven.py  # Production code
├── epl_model_final.pkl        # Trained model
└── requirements.txt           # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
./venv/bin/python experiments/final_model_correct.py
```

---

## 📊 Features Used

### Base Features (20)
- ELO ratings (home_elo, away_elo, elo_diff, elo_expected)
- Attack/Defense matchup (att_def_matchup)
- Form stats (winrate_3_diff, cs_3_diff, shots_diff)
- H2H history (h2h_draw_rate, h2h_home_rate, h2h_away_rate)
- Rest days (h_rest, a_rest, rest_diff)
- Non-league scoring (h_nl_scoring, a_nl_scoring, nl_scoring_diff)
- Lineup changes (avg_change_rate_diff)
- Draw tendency (combined_draw_rate)

### New Features (4)
- **xg_elo_diff**: xG ELO (chance creation quality) - calculated from raw shot data
- **xga_elo_diff**: xGA ELO (defensive solidity) - calculated from raw shot data
- **referee_bias**: Referee's historical home win rate (rolling, no data leakage)
- **fouls_diff**: Team discipline (rolling average)

---

## 🔧 Model Configuration

### Hyperparameters (Optimized via Grid Search)
```python
HOME_ADV = 82       # Home advantage in ELO
K_FACTOR = 28       # ELO update factor
K_ATT = 5           # Attack rating update
K_DEF = 90          # Defense rating update
K_XG = 10           # xG ELO update factor
C = 0.15            # Logistic regression regularization
ELO_DIVISOR = 640   # ELO probability divisor
```

### Defaults (Calculated from Training Data)
All defaults are calculated from training data (2015-2023), no hardcoded values:
- home_win_rate, away_win_rate, draw_rate
- avg_home_goals, avg_away_goals
- avg_fouls, sot_conversion
- default_win_rate, default_cs_rate

---

## 📈 xG Model

Built from raw shot data (106,924 shots from Understat):
- **Features**: X, Y position, distance, angle, shotType, situation, lastAction
- **Model AUC**: 0.78
- **Data**: 4,180 matches (2014-2025)

**Important**: We use our own xG model, NOT Understat's xG values (which are model outputs).

---

## ✅ Verification

All features verified for:
1. **No data leakage** - Only historical data used for predictions
2. **No hardcoded values** - All defaults calculated from training data
3. **Correct calculation** - Each feature tested individually

---

## 📝 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- See `requirements.txt` for full list

---

**Date:** 2026-01-05
**Version:** Final (Correct)
**Status:** Production-Ready ✅
