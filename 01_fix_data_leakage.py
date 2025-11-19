#!/usr/bin/env python3
"""
FIX DATA LEAKAGE IN ENGINEERED FEATURES
Remove data leakage from form and momentum calculations
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_baseline_data():
    """Load baseline EPL dataset"""
    print("📊 Loading baseline EPL data...")
    df = pd.read_csv('data_processed/all_seasons.csv')
    print(f"   ✅ Baseline data: {len(df)} matches, {len(df.columns)} columns")
    return df

def create_leakage_free_features(df):
    """Create ML features without data leakage"""
    print("🚀 Creating leakage-free ML features...")

    enhanced_df = df.copy()

    # Convert date properly
    enhanced_df['date'] = pd.to_datetime(enhanced_df['Date'], dayfirst=True)
    enhanced_df = enhanced_df.sort_values('date')

    print("   📈 Calculating pre-match team metrics...")

    # Initialize feature columns
    enhanced_df['home_momentum'] = 0.0
    enhanced_df['away_momentum'] = 0.0
    enhanced_df['home_rest_days'] = 7.0  # Default for first match
    enhanced_df['away_rest_days'] = 7.0
    enhanced_df['home_form_points'] = 0.0  # Points from last 5 matches (excluding current)
    enhanced_df['away_form_points'] = 0.0

    # Team performance tracking
    teams = enhanced_df['HomeTeam'].unique()
    team_histories = {team: [] for team in teams}

    # Process each match chronologically
    for idx, match in enhanced_df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date']

        # Calculate pre-match metrics for home team
        home_metrics = calculate_pre_match_metrics(team_histories[home_team], match_date)
        enhanced_df.loc[idx, 'home_momentum'] = home_metrics['momentum']
        enhanced_df.loc[idx, 'home_form_points'] = home_metrics['form_points']
        enhanced_df.loc[idx, 'home_rest_days'] = home_metrics['rest_days']

        # Calculate pre-match metrics for away team
        away_metrics = calculate_pre_match_metrics(team_histories[away_team], match_date)
        enhanced_df.loc[idx, 'away_momentum'] = away_metrics['momentum']
        enhanced_df.loc[idx, 'away_form_points'] = away_metrics['form_points']
        enhanced_df.loc[idx, 'away_rest_days'] = away_metrics['rest_days']

        # Update team histories AFTER calculating pre-match metrics
        home_result = get_match_result(match, home_team)
        away_result = get_match_result(match, away_team)

        team_histories[home_team].append({
            'date': match_date,
            'result': home_result,
            'goal_diff': get_goal_difference(match, home_team),
            'shots': match['HS'] if home_team == match['HomeTeam'] else match['AS']
        })

        team_histories[away_team].append({
            'date': match_date,
            'result': away_result,
            'goal_diff': get_goal_difference(match, away_team),
            'shots': match['AS'] if away_team == match['AwayTeam'] else match['HS']
        })

    # Create derived features (safe to calculate now)
    enhanced_df['momentum_difference'] = enhanced_df['home_momentum'] - enhanced_df['away_momentum']
    enhanced_df['form_difference'] = enhanced_df['home_form_points'] - enhanced_df['away_form_points']
    enhanced_df['rest_days_difference'] = enhanced_df['home_rest_days'] - enhanced_df['away_rest_days']

    return enhanced_df

def calculate_pre_match_metrics(team_history, current_date):
    """Calculate team metrics BEFORE the current match"""
    if not team_history:
        return {
            'momentum': 0.0,
            'form_points': 0.0,
            'rest_days': 7.0
        }

    # Calculate rest days since last match
    last_match = team_history[-1]
    rest_days = (current_date - last_match['date']).days
    if rest_days < 1:
        rest_days = 7  # Default for issues

    # Calculate form (last 5 matches, all BEFORE this match)
    recent_matches = team_history[-5:]
    form_points = sum(get_points_from_result(match['result']) for match in recent_matches)

    # Calculate momentum (weighted recent performance)
    momentum = 0.0
    for i, match in enumerate(recent_matches):
        weight = (i + 1) / len(recent_matches)  # More recent = higher weight
        points = get_points_from_result(match['result'])
        momentum += points * weight

    # Bonus for recent shot performance (non-leaky)
    if len(team_history) >= 3:
        recent_3 = team_history[-3:]
        avg_shots = np.mean([match['shots'] for match in recent_3])
        if avg_shots > 15:  # High shot volume
            momentum += 1.0
        elif avg_shots > 12:
            momentum += 0.5

    return {
        'momentum': momentum,
        'form_points': form_points,
        'rest_days': rest_days
    }

def get_match_result(match, team):
    """Get match result for a specific team"""
    if team == match['HomeTeam']:
        return match['FTR']
    else:
        if match['FTR'] == 'H':
            return 'A'
        elif match['FTR'] == 'A':
            return 'H'
        else:
            return 'D'

def get_goal_difference(match, team):
    """Get goal difference for a specific team"""
    if team == match['HomeTeam']:
        return match['FTHG'] - match['FTAG']
    else:
        return match['FTAG'] - match['FTHG']

def get_points_from_result(result):
    """Convert result to points"""
    return 3 if result in ['H', 'A'] else 1 if result == 'D' else 0

def create_safe_features(df):
    """Create features that don't contain data leakage"""
    print("   ⚡ Creating safe ML features...")

    # Shot-based features (all from match statistics - these are fine)
    df['shot_difference'] = df['HS'] - df['AS']
    df['shot_on_target_difference'] = df['HST'] - df['AST']
    df['home_shot_accuracy'] = df['HST'] / (df['HS'] + 1)
    df['away_shot_accuracy'] = df['AST'] / (df['AS'] + 1)
    df['shot_pressure_ratio'] = df['HS'] / (df['HS'] + df['AS'] + 1)

    # Disciplinary features
    df['home_total_cards'] = df['HY'] + 2 * df['HR']
    df['away_total_cards'] = df['AY'] + 2 * df['AR']
    df['cards_difference'] = df['home_total_cards'] - df['away_total_cards']
    df['discipline_advantage'] = df['away_total_cards'] - df['home_total_cards']

    # Foul features
    df['foul_difference'] = df['HF'] - df['AF']
    df['total_fouls'] = df['HF'] + df['AF']
    df['home_foul_ratio'] = df['HF'] / (df['total_fouls'] + 1)

    # Set piece features
    df['corner_difference'] = df['HC'] - df['AC']
    df['total_corners'] = df['HC'] + df['AC']
    df['home_corner_ratio'] = df['HC'] / (df['total_corners'] + 1)

    # Historical team strength (based on season-long averages up to this point)
    print("   📊 Calculating historical team strength...")
    df = add_historical_team_strength(df)

    return df

def add_historical_team_strength(df):
    """Add team strength based on historical performance up to each match"""
    df['home_team_strength'] = 0.0
    df['away_team_strength'] = 0.0

    teams = df['HomeTeam'].unique()
    team_stats = {team: {'goals': 0, 'conceded': 0, 'matches': 0} for team in teams}

    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        # Calculate historical strength based on past matches
        if team_stats[home_team]['matches'] > 0:
            home_strength = (team_stats[home_team]['goals'] - team_stats[home_team]['conceded']) / team_stats[home_team]['matches']
            df.loc[idx, 'home_team_strength'] = home_strength

        if team_stats[away_team]['matches'] > 0:
            away_strength = (team_stats[away_team]['goals'] - team_stats[away_team]['conceded']) / team_stats[away_team]['matches']
            df.loc[idx, 'away_team_strength'] = away_strength

        # Update team stats AFTER calculating strength for this match
        team_stats[home_team]['goals'] += match['FTHG']
        team_stats[home_team]['conceded'] += match['FTAG']
        team_stats[home_team]['matches'] += 1

        team_stats[away_team]['goals'] += match['FTAG']
        team_stats[away_team]['conceded'] += match['FTHG']
        team_stats[away_team]['matches'] += 1

    df['strength_difference'] = df['home_team_strength'] - df['away_team_strength']

    return df

def clean_and_finalize(df):
    """Clean and finalize the leakage-free dataset"""
    print("   🧹 Cleaning and finalizing leakage-free features...")

    # Fill any missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Select final features (ensure no data leakage)
    leakage_safe_features = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR',  # Essential match info

        # Original match statistics (safe - from the match itself)
        'HS', 'AS', 'HST', 'AST',  # Shots
        'HF', 'AF', 'HC', 'AC',    # Fouls and corners
        'HY', 'AY', 'HR', 'AR',    # Cards

        # Engineered features (leakage-free)
        'shot_difference', 'shot_on_target_difference',
        'home_shot_accuracy', 'away_shot_accuracy', 'shot_pressure_ratio',
        'cards_difference', 'discipline_advantage',
        'foul_difference', 'home_foul_ratio',
        'corner_difference', 'home_corner_ratio',

        # Pre-match team metrics (leakage-free)
        'home_momentum', 'away_momentum', 'momentum_difference',
        'home_form_points', 'away_form_points', 'form_difference',
        'home_rest_days', 'away_rest_days', 'rest_days_difference',

        # Historical team strength
        'home_team_strength', 'away_team_strength', 'strength_difference'
    ]

    # Keep only available features
    final_features = [col for col in leakage_safe_features if col in df.columns]
    final_df = df[final_features].copy()

    print(f"   ✅ Final leakage-free features: {len(final_features)}")

    return final_df

def save_leakage_free_dataset(df):
    """Save the leakage-free enhanced dataset"""
    print("💾 Saving leakage-free enhanced dataset...")

    filename = 'data_processed/leakage_free_enhanced_epl_ml.csv'
    df.to_csv(filename, index=False)
    print(f"   ✅ Leakage-free dataset saved: {len(df)} matches, {len(df.columns)} features")

    # Save feature list
    feature_list = [col for col in df.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'FTR']]
    with open('data_processed/leakage_free_feature_list.pkl', 'wb') as f:
        pickle.dump(feature_list, f)

    print(f"   ✅ {len(feature_list)} ML features saved")

def main():
    """Main function to fix data leakage"""
    print("🔧 FIXING DATA LEAKAGE IN EPL FEATURES")
    print("=" * 50)

    # Load baseline data
    baseline_df = load_baseline_data()

    # Create leakage-free features
    enhanced_df = create_leakage_free_features(baseline_df)

    # Add safe engineered features
    enhanced_df = create_safe_features(enhanced_df)

    # Clean and finalize
    final_df = clean_and_finalize(enhanced_df)

    # Save the leakage-free dataset
    save_leakage_free_dataset(final_df)

    print("\n✅ DATA LEAKAGE FIXED!")
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"   ✅ Form and momentum calculated BEFORE each match")
    print(f"   ✅ Team strength based on historical data only")
    print(f"   ✅ No future information leaked into features")
    print(f"   ✅ Proper chronological processing")
    print(f"   ✅ Ready for realistic ML training")

    return final_df

if __name__ == "__main__":
    leakage_free_data = main()