#!/usr/bin/env python3
"""
FIXED FEATURE ENGINEERING FOR EPL PREDICTION
Create proper ML features with home_momentum and away_momentum instead of team-specific columns
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

def create_proper_ml_features(df):
    """Create proper ML features with correct momentum calculation"""
    print("🚀 Creating proper ML features...")

    enhanced_df = df.copy()

    # Convert date properly
    enhanced_df['date'] = pd.to_datetime(enhanced_df['Date'], dayfirst=True)
    enhanced_df = enhanced_df.sort_values('date')

    print("   📈 Calculating team metrics...")

    # Calculate form and momentum for each team properly
    teams = enhanced_df['HomeTeam'].unique()

    # Create dictionaries to store team metrics
    team_form = {}
    team_momentum = {}
    team_rest_days = {}

    for team in teams:
        # Get all matches for this team (both home and away)
        team_matches = enhanced_df[
            (enhanced_df['HomeTeam'] == team) | (enhanced_df['AwayTeam'] == team)
        ].copy().sort_values('date')

        form_scores = []
        momentum_scores = []
        rest_days_list = []

        previous_date = None

        for i, match in team_matches.iterrows():
            current_date = match['date']

            # Calculate rest days
            if previous_date is not None:
                days_diff = (current_date - previous_date).days
                if days_diff < 1:
                    days_diff = 7  # Default for consecutive matches
            else:
                days_diff = 7

            rest_days_list.append(days_diff)

            # Calculate result and form
            if match['HomeTeam'] == team:
                result = match['FTR']
                goal_diff = match['FTHG'] - match['FTAG']
                shots = match['HS']
                shots_on_target = match['HST']
            else:
                result = match['FTR']
                goal_diff = match['FTAG'] - match['FTHG']
                shots = match['AS']
                shots_on_target = match['AST']

            # Calculate form points
            if (result == 'H' and match['HomeTeam'] == team) or \
               (result == 'A' and match['AwayTeam'] == team):
                form_points = 3  # Win
            elif result == 'D':
                form_points = 1  # Draw
            else:
                form_points = 0  # Loss

            form_scores.append(form_points)

            # Calculate momentum (last 5 matches performance)
            if len(form_scores) >= 5:
                recent_form = form_scores[-5:]
            else:
                recent_form = form_scores

            # Enhanced momentum calculation
            momentum = sum(recent_form)

            # Bonus for recent performance quality
            if len(form_scores) >= 3:
                recent_3 = form_scores[-3:]
                momentum += sum(recent_3) * 0.5

            momentum_scores.append(momentum)

            previous_date = current_date

        # Store team metrics
        team_form[team] = form_scores
        team_momentum[team] = momentum_scores
        team_rest_days[team] = rest_days_list

    # Map team metrics back to dataframe
    print("   🔗 Mapping team metrics to matches...")

    home_momentum = []
    away_momentum = []
    home_form = []
    away_form = []
    home_rest = []
    away_rest = []

    for idx, match in enhanced_df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        # Get match index in team's history
        home_match_idx = get_match_index(enhanced_df, idx, home_team)
        away_match_idx = get_match_index(enhanced_df, idx, away_team)

        # Get momentum
        if home_match_idx < len(team_momentum[home_team]):
            home_momentum.append(team_momentum[home_team][home_match_idx])
            home_form.append(team_form[home_team][home_match_idx])
            home_rest.append(team_rest_days[home_team][home_match_idx])
        else:
            home_momentum.append(0)
            home_form.append(0)
            home_rest.append(7)

        if away_match_idx < len(team_momentum[away_team]):
            away_momentum.append(team_momentum[away_team][away_match_idx])
            away_form.append(team_form[away_team][away_match_idx])
            away_rest.append(team_rest_days[away_team][away_match_idx])
        else:
            away_momentum.append(0)
            away_form.append(0)
            away_rest.append(7)

    # Add proper features to dataframe
    enhanced_df['home_momentum'] = home_momentum
    enhanced_df['away_momentum'] = away_momentum
    enhanced_df['home_form'] = home_form
    enhanced_df['away_form'] = away_form
    enhanced_df['home_rest_days'] = home_rest
    enhanced_df['away_rest_days'] = away_rest

    # Create derived features
    enhanced_df['momentum_difference'] = enhanced_df['home_momentum'] - enhanced_df['away_momentum']
    enhanced_df['form_difference'] = enhanced_df['home_form'] - enhanced_df['away_form']
    enhanced_df['rest_days_difference'] = enhanced_df['home_rest_days'] - enhanced_df['away_rest_days']

    return enhanced_df

def get_match_index(df, match_idx, team):
    """Get the match index in team's chronological history"""
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    match_indices = team_matches.index.tolist()

    if match_idx in match_indices:
        return match_indices.index(match_idx)
    else:
        return len(team_matches)

def create_shot_features(df):
    """Create shot-based features"""
    print("   ⚡ Creating shot features...")

    # Shot difference features
    df['shot_difference'] = df['HS'] - df['AS']
    df['shot_on_target_difference'] = df['HST'] - df['AST']

    # Shot accuracy
    df['home_shot_accuracy'] = df['HST'] / (df['HS'] + 1)
    df['away_shot_accuracy'] = df['AST'] / (df['AS'] + 1)
    df['shot_accuracy_difference'] = df['home_shot_accuracy'] - df['away_shot_accuracy']

    # Shot pressure
    df['home_shot_pressure'] = df['HS'] / (df['HS'] + df['AS'] + 1)
    df['away_shot_pressure'] = df['AS'] / (df['HS'] + df['AS'] + 1)

def create_disciplinary_features(df):
    """Create disciplinary features"""
    print("   🟨 Creating disciplinary features...")

    # Total cards
    df['home_total_cards'] = df['HY'] + 2*df['HR']
    df['away_total_cards'] = df['AY'] + 2*df['AR']
    df['total_cards'] = df['home_total_cards'] + df['away_total_cards']
    df['cards_difference'] = df['home_total_cards'] - df['away_total_cards']

    # Card ratios
    df['home_card_ratio'] = df['home_total_cards'] / (df['total_cards'] + 1)
    df['discipline_advantage'] = df['away_total_cards'] - df['home_total_cards']

def create_foul_features(df):
    """Create foul-related features"""
    print("   🟨 Creating foul features...")

    # Foul differences
    df['foul_difference'] = df['HF'] - df['AF']
    df['total_fouls'] = df['HF'] + df['AF']

    # Foul ratios
    df['home_foul_ratio'] = df['HF'] / (df['total_fouls'] + 1)
    df['foul_pressure'] = df['HF'] / (df['HF'] + df['AF'] + 1)

def create_set_piece_features(df):
    """Create set piece features"""
    print("   🚩 Creating set piece features...")

    # Corner features
    df['corner_difference'] = df['HC'] - df['AC']
    df['total_corners'] = df['HC'] + df['AC']
    df['home_corner_ratio'] = df['HC'] / (df['total_corners'] + 1)

    # Set piece effectiveness
    df['home_set_piece_threat'] = (df['HC'] + df['HF']) / 2
    df['away_set_piece_threat'] = (df['AC'] + df['AF']) / 2
    df['set_piece_advantage'] = df['home_set_piece_threat'] - df['away_set_piece_threat']

def clean_and_finalize_features(df):
    """Clean and prepare final feature set"""
    print("   🧹 Cleaning and finalizing features...")

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Remove temporary columns
    exclude_cols = ['date']
    df = df.drop(columns=[col for col in exclude_cols if col in df.columns])

    # Select final features for ML
    core_features = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR',  # Essential match info

        # Shot statistics (most important)
        'HS', 'AS', 'HST', 'AST', 'shot_difference',
        'shot_on_target_difference', 'home_shot_accuracy', 'away_shot_accuracy',

        # Team momentum and form (our key engineered features)
        'home_momentum', 'away_momentum', 'momentum_difference',
        'home_form', 'away_form', 'form_difference',

        # Rest and fatigue
        'home_rest_days', 'away_rest_days', 'rest_days_difference',

        # Disciplinary features
        'HY', 'AY', 'HR', 'AR', 'cards_difference', 'discipline_advantage',

        # Foul features
        'HF', 'AF', 'foul_difference', 'home_foul_ratio',

        # Set piece features
        'HC', 'AC', 'corner_difference', 'home_corner_ratio'
    ]

    # Keep only available core features
    final_features = [col for col in core_features if col in df.columns]
    final_df = df[final_features].copy()

    return final_df

def save_fixed_enhanced_dataset(df):
    """Save the properly enhanced dataset"""
    print("💾 Saving fixed enhanced dataset...")

    filename = 'data_processed/fixed_enhanced_epl_ml.csv'
    df.to_csv(filename, index=False)
    print(f"   ✅ Fixed enhanced dataset saved: {len(df)} matches, {len(df.columns)} features")

    # Save feature list
    feature_list = [col for col in df.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'FTR']]
    with open('data_processed/fixed_feature_list.pkl', 'wb') as f:
        pickle.dump(feature_list, f)

    print(f"   ✅ Feature list saved: {len(feature_list)} ML features")

def main():
    """Main function to fix feature engineering"""
    print("🔧 FIXING EPL FEATURE ENGINEERING")
    print("=" * 50)

    # Load baseline data
    baseline_df = load_baseline_data()

    # Create proper ML features
    enhanced_df = create_proper_ml_features(baseline_df)

    # Add specific feature groups
    create_shot_features(enhanced_df)
    create_disciplinary_features(enhanced_df)
    create_foul_features(enhanced_df)
    create_set_piece_features(enhanced_df)

    # Clean and finalize
    final_df = clean_and_finalize_features(enhanced_df)

    # Save the fixed dataset
    save_fixed_enhanced_dataset(final_df)

    print("\n✅ FEATURE ENGINEERING FIXED!")
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"   ✅ Proper home_momentum and away_momentum columns")
    print(f"   ✅ Realistic form and rest days calculations")
    print(f"   ✅ Clean feature structure (no 40+ team columns)")
    print(f"   ✅ Shot, disciplinary, foul, and set piece features")
    print(f"   ✅ Ready for proper ML training")

    return final_df

if __name__ == "__main__":
    fixed_data = main()