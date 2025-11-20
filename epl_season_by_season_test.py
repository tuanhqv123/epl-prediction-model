#!/usr/bin/env python3
"""
EPL PREDICTION MODEL - SEASON BY SEASON TESTING
===============================================

Proper season-by-season evaluation like the old code:
- Train on seasons 2015-2016 to 2021-2022
- Test on each season individually (2022-2023, 2023-2024, 2024-2025)
- Shows performance for each test season separately
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def calculate_3match_form(df):
    """Calculate 3-match form (proven optimal)"""
    # Sort by season and date
    df = df.sort_values(['season', 'Date']).reset_index(drop=True)

    # Initialize form features
    df['home_form_3'] = 0.0
    df['away_form_3'] = 0.0
    df['form_diff_3'] = 0.0

    # Process each season separately
    for season in sorted(df['season'].unique()):
        season_mask = df['season'] == season
        season_indices = df[season_mask].index.tolist()

        # Initialize team tracking
        team_form = {}
        all_teams = set(df.loc[season_mask, 'HomeTeam'].unique()) | \
                   set(df.loc[season_mask, 'AwayTeam'].unique())

        for team in all_teams:
            team_form[team] = []

        # Process matches chronologically
        for idx in season_indices:
            row = df.loc[idx]
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            # Calculate current form
            home_form_avg = np.mean(team_form[home_team][-3:]) if team_form[home_team] else 0
            away_form_avg = np.mean(team_form[away_team][-3:]) if team_form[away_team] else 0

            df.loc[idx, 'home_form_3'] = home_form_avg
            df.loc[idx, 'away_form_3'] = away_form_avg
            df.loc[idx, 'form_diff_3'] = home_form_avg - away_form_avg

            # Update form with match result
            try:
                home_goals = int(row['FTHG']) if pd.notna(row['FTHG']) else 0
                away_goals = int(row['FTAG']) if pd.notna(row['FTAG']) else 0

                if home_goals > away_goals:
                    team_form[home_team].append(3)
                    team_form[away_team].append(0)
                elif home_goals < away_goals:
                    team_form[home_team].append(0)
                    team_form[away_team].append(3)
                else:
                    team_form[home_team].append(1)
                    team_form[away_team].append(1)

                # Keep only last 3 results
                team_form[home_team] = team_form[home_team][-3:]
                team_form[away_team] = team_form[away_team][-3:]

            except (ValueError, KeyError):
                continue

    return df

def create_features(df):
    """Create all features for the model"""
    # Shot accuracy
    df['home_shot_accuracy'] = np.where(df['HS'] > 0, df['HST'] / df['HS'], 0)
    df['away_shot_accuracy'] = np.where(df['AS'] > 0, df['AST'] / df['AS'], 0)

    # Rest days (use existing calculations)
    df['load_diff'] = df['load_diff_7days_fixed']
    df['home_rest_days'] = df['home_rest_days_fixed']
    df['away_rest_days'] = df['away_rest_days_fixed']

    # Optimal feature set (proven through testing)
    feature_cols = [
        # Core match statistics (12)
        'HST', 'AST', 'HS', 'AS', 'HY', 'AY', 'HR', 'AR', 'HC', 'AC', 'HF', 'AF',

        # Engineered features (2)
        'home_shot_accuracy', 'away_shot_accuracy',

        # Form features (3)
        'home_form_3', 'away_form_3', 'form_diff_3',

        # Competition features (3)
        'home_rest_days', 'away_rest_days', 'load_diff'
    ]

    return feature_cols

def test_model_by_season():
    """Test model season by season like the old code"""
    print("🏆 EPL PREDICTION MODEL - SEASON BY SEASON TESTING")
    print("="*60)
    print("Training: 2015-2016 to 2021-2022")
    print("Testing: Each season individually (2022-2023, 2023-2024, 2024-2025)")
    print("="*60)

    # Load data
    df = pd.read_csv('epl_enhanced_fixed.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'FTR', 'HomeTeam', 'AwayTeam'])

    print(f"📊 Dataset: {len(df)} matches across {len(df['season'].unique())} seasons")

    # Add features
    df = calculate_3match_form(df)
    create_features(df)
    feature_cols = [
        # Core match statistics (12)
        'HST', 'AST', 'HS', 'AS', 'HY', 'AY', 'HR', 'AR', 'HC', 'AC', 'HF', 'AF',

        # Engineered features (2)
        'home_shot_accuracy', 'away_shot_accuracy',

        # Form features (3)
        'home_form_3', 'away_form_3', 'form_diff_3',

        # Competition features (3)
        'home_rest_days', 'away_rest_days', 'load_diff'
    ]

    # Prepare features and target
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['FTR']

    # Define training and test seasons
    train_seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022']
    test_seasons = ['2022-2023', '2023-2024', '2024-2025']

    # Training data
    train_mask = df['season'].isin(train_seasons)
    X_train = X[train_mask]
    y_train = y[train_mask]

    print(f"\n🔄 TRAINING DATA:")
    print(f"   Training seasons: {train_seasons[0]} to {train_seasons[-1]}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Training target distribution: {y_train.value_counts().to_dict()}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model on training data
    print(f"\n🚀 TRAINING MODEL...")
    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print(f"✅ Model training completed!")

    # Test each season individually
    print(f"\n📊 SEASON BY SEASON RESULTS:")
    print("-"*60)

    season_results = []
    overall_predictions = []
    overall_actual = []

    for season in test_seasons:
        print(f"\n🔍 TESTING SEASON: {season}")
        print("-"*40)

        # Get test data for this season
        season_mask = df['season'] == season
        X_test = X[season_mask]
        y_test = y[season_mask]

        if len(X_test) == 0:
            print(f"   ❌ No data for season {season}")
            continue

        print(f"   Test samples: {len(X_test):,}")
        print(f"   Test target distribution: {y_test.value_counts().to_dict()}")

        # Scale test data
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        draw_f1 = f1_score(y_test, y_pred, labels=['D'], average='macro')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=['H', 'D', 'A'])

        season_result = {
            'season': season,
            'samples': len(X_test),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'draw_f1': draw_f1,
            'confusion_matrix': cm.tolist(),
            'target_dist': y_test.value_counts().to_dict()
        }
        season_results.append(season_result)

        # Collect for overall evaluation
        overall_predictions.extend(y_pred)
        overall_actual.extend(y_test)

        print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"   🎯 Draw F1: {draw_f1:.4f} ({draw_f1:.1%})")
        print(f"   🎯 Macro F1: {macro_f1:.4f} ({macro_f1:.1%})")

        print(f"   🔢 Confusion Matrix:")
        print(f"      {'':<8} {'Pred H':<8} {'Pred D':<8} {'Pred A':<8}")
        for i, actual in enumerate(['H', 'D', 'A']):
            print(f"      {actual:<8} {cm[i,0]:<8} {cm[i,1]:<8} {cm[i,2]:<8}")

    # Overall results
    print(f"\n📈 OVERALL RESULTS (All Test Seasons Combined):")
    print("="*60)

    overall_accuracy = accuracy_score(overall_actual, overall_predictions)
    overall_macro_f1 = f1_score(overall_actual, overall_predictions, average='macro')
    overall_draw_f1 = f1_score(overall_actual, overall_predictions, labels=['D'], average='macro')
    overall_cm = confusion_matrix(overall_actual, overall_predictions, labels=['H', 'D', 'A'])

    print(f"🎯 Overall Performance:")
    print(f"   Total Test Samples: {len(overall_actual):,}")
    print(f"   Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.1%})")
    print(f"   Overall Draw F1: {overall_draw_f1:.4f} ({overall_draw_f1:.1%})")
    print(f"   Overall Macro F1: {overall_macro_f1:.4f} ({overall_macro_f1:.1%})")

    print(f"\n🔢 Overall Confusion Matrix:")
    print(f"   {'':<8} {'Pred H':<8} {'Pred D':<8} {'Pred A':<8}")
    for i, actual in enumerate(['H', 'D', 'A']):
        print(f"   {actual:<8} {overall_cm[i,0]:<8} {overall_cm[i,1]:<8} {overall_cm[i,2]:<8}")

    # Feature importance
    print(f"\n🔍 Top 10 Feature Importance:")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, (_, row) in enumerate(importances.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} {row['importance']:.4f}")

    # Literature comparison
    literature_avg = 0.560
    improvement_vs_literature = ((overall_accuracy - literature_avg) / literature_avg) * 100

    print(f"\n📊 ACADEMIC COMPARISON:")
    print(f"   Literature Average: 56.0%")
    print(f"   Our Model: {overall_accuracy:.1%}")
    print(f"   Improvement: +{improvement_vs_literature:.1f}%")
    print(f"   Status: {'✅ BEATS LITERATURE' if improvement_vs_literature > 0 else '❌ Below literature'}")

    return {
        'season_results': season_results,
        'overall_results': {
            'accuracy': overall_accuracy,
            'macro_f1': overall_macro_f1,
            'draw_f1': overall_draw_f1,
            'confusion_matrix': overall_cm.tolist(),
            'total_samples': len(overall_actual)
        },
        'feature_importance': importances.to_dict('records'),
        'literature_comparison': improvement_vs_literature
    }

def main():
    """Main execution"""
    try:
        results = test_model_by_season()

        print(f"\n" + "="*80)
        print(f"🏆 SEASON BY SEASON TESTING COMPLETE!")
        print(f"="*80)

        print(f"\n✅ SUMMARY:")
        print(f"   Testing Method: Season by season (proper temporal validation)")
        print(f"   Test Seasons: {[r['season'] for r in results['season_results']]}")
        print(f"   Overall Accuracy: {results['overall_results']['accuracy']:.1%}")
        print(f"   Overall Draw F1: {results['overall_results']['draw_f1']:.1%}")
        print(f"   Literature Edge: +{results['literature_comparison']:.1f}%")

        return results

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()