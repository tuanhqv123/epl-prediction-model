#!/usr/bin/env python3
"""
FIX CRITICAL DATA ISSUES
Fix unreasonable rest days and shot consistency issues
"""

import pandas as pd
import numpy as np

def fix_data_issues():
    """Fix critical data issues identified in validation"""
    print("🔧 FIXING CRITICAL DATA ISSUES")
    print("=" * 40)

    # Load the data
    df = pd.read_csv('data_processed/leakage_free_enhanced_epl_ml.csv')
    print(f"📊 Loaded {len(df)} matches")

    # Issue 1: Fix unreasonable rest days
    print("\n🛠️  Fixing unreasonable rest days...")

    # Cap rest days at reasonable values (max 30 days)
    max_rest_days = 30
    min_rest_days = 2

    # Original ranges
    print(f"   Original home_rest_days: [{df['home_rest_days'].min():.1f}, {df['home_rest_days'].max():.1f}]")
    print(f"   Original away_rest_days: [{df['away_rest_days'].min():.1f}, {df['away_rest_days'].max():.1f}]")

    # Fix rest days
    df['home_rest_days'] = df['home_rest_days'].clip(min_rest_days, max_rest_days)
    df['away_rest_days'] = df['away_rest_days'].clip(min_rest_days, max_rest_days)
    df['rest_days_difference'] = df['home_rest_days'] - df['away_rest_days']

    # New ranges
    print(f"   Fixed home_rest_days: [{df['home_rest_days'].min():.1f}, {df['home_rest_days'].max():.1f}]")
    print(f"   Fixed away_rest_days: [{df['away_rest_days'].min():.1f}, {df['away_rest_days'].max():.1f}]")

    # Issue 2: Fix shot consistency
    print("\n🛠️  Fixing shot consistency issues...")

    # Ensure shots_on_target <= total_shots
    shot_issues = 0
    for idx, match in df.iterrows():
        if match['HST'] > match['HS']:
            df.loc[idx, 'HS'] = match['HST']  # Fix by setting total shots = shots on target
            shot_issues += 1
        if match['AST'] > match['AS']:
            df.loc[idx, 'AS'] = match['AST']
            shot_issues += 1

    print(f"   Fixed {shot_issues} shot consistency issues")

    # Recalculate derived features that might be affected
    print("\n🔄 Recalculating derived features...")

    df['shot_difference'] = df['HS'] - df['AS']
    df['shot_on_target_difference'] = df['HST'] - df['AST']
    df['home_shot_accuracy'] = df['HST'] / (df['HS'] + 1)
    df['away_shot_accuracy'] = df['AST'] / (df['AS'] + 1)
    df['total_fouls'] = df['HF'] + df['AF']
    df['home_foul_ratio'] = df['HF'] / (df['total_fouls'] + 1)
    df['total_corners'] = df['HC'] + df['AC']
    df['home_corner_ratio'] = df['HC'] / (df['total_corners'] + 1)

    # Save fixed data
    output_file = 'data_processed/clean_enhanced_epl_ml.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Fixed data saved to: {output_file}")
    print(f"   📊 {len(df)} matches, {len(df.columns)} features")

    # Summary of changes
    print(f"\n📋 SUMMARY OF FIXES:")
    print(f"   ✅ Rest days capped to {min_rest_days}-{max_rest_days} days")
    print(f"   ✅ Shot consistency fixed ({shot_issues} matches)")
    print(f"   ✅ Derived features recalculated")

    return df

if __name__ == "__main__":
    fixed_df = fix_data_issues()
    print("\n🎯 Data is now clean and ready for optimal model training!")