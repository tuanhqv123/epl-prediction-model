#!/usr/bin/env python3
"""
SIMPLIFIED SHAP ANALYSIS FOR MODEL EXPLAINABILITY
Focused and robust SHAP analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load datasets for SHAP analysis"""
    print("📊 LOADING DATA FOR SHAP ANALYSIS")
    print("=" * 50)

    # Load baseline data
    baseline_X_train = pd.read_csv('baseline_data_fixed/X_train.csv')
    baseline_X_test = pd.read_csv('baseline_data_fixed/X_test.csv')
    baseline_y_train = pd.read_csv('baseline_data_fixed/y_train.csv').iloc[:, 0]

    # Load enhanced data
    enhanced_X_train = pd.read_csv('enhanced_data/X_train.csv')
    enhanced_X_test = pd.read_csv('enhanced_data/X_test.csv')
    enhanced_y_train = pd.read_csv('enhanced_data/y_train.csv').iloc[:, 0]

    # Load feature columns
    with open('baseline_data_fixed/feature_columns.pkl', 'rb') as f:
        baseline_features = pickle.load(f)

    with open('enhanced_data/feature_columns.pkl', 'rb') as f:
        enhanced_features = pickle.load(f)

    print(f"✅ Baseline: {baseline_X_train.shape[0]} samples, {len(baseline_features)} features")
    print(f"✅ Enhanced: {enhanced_X_train.shape[0]} samples, {len(enhanced_features)} features")

    return {
        'baseline': {'X_train': baseline_X_train, 'X_test': baseline_X_test, 'y_train': baseline_y_train, 'features': baseline_features},
        'enhanced': {'X_train': enhanced_X_train, 'X_test': enhanced_X_test, 'y_train': enhanced_y_train, 'features': enhanced_features}
    }

def train_shap_models(data):
    """Train simplified models for SHAP analysis"""
    print(f"\n🤖 TRAINING MODELS FOR SHAP ANALYSIS")
    print("=" * 50)

    models = {}

    # Baseline Random Forest
    print("📊 Training baseline Random Forest...")
    baseline_rf = RandomForestClassifier(
        random_state=42,
        n_estimators=50,  # Smaller for faster SHAP
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced'
    )
    baseline_rf.fit(data['baseline']['X_train'], data['baseline']['y_train'])
    models['baseline'] = baseline_rf

    # Enhanced Random Forest
    print("📊 Training enhanced Random Forest...")
    enhanced_rf = RandomForestClassifier(
        random_state=42,
        n_estimators=50,  # Smaller for faster SHAP
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced'
    )
    enhanced_rf.fit(data['enhanced']['X_train'], data['enhanced']['y_train'])
    models['enhanced'] = enhanced_rf

    print(f"✅ Trained {len(models)} models for SHAP analysis")

    return models

def simple_feature_importance_analysis(model, X, features, dataset_name):
    """Simple feature importance analysis using model's built-in feature importance"""
    print(f"\n🔍 {dataset_name.upper()} FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)

    # Get built-in feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print(f"❌ Model doesn't have feature_importances_ attribute")
        return None

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\n🏆 TOP 15 FEATURES ({dataset_name}):")
    for i, row in importance_df.head(15).iterrows():
        print(f"   {i+1:2d}. {row['feature']:30s}: {row['importance']:.4f}")

    return importance_df

def create_feature_importance_plots(baseline_importance, enhanced_importance):
    """Create feature importance comparison plots"""
    print(f"\n📊 CREATING FEATURE IMPORTANCE PLOTS")
    print("=" * 50)

    # Create output directory
    os.makedirs('feature_analysis', exist_ok=True)

    # Baseline plot
    plt.figure(figsize=(12, 8))
    baseline_top = baseline_importance.head(15)
    plt.barh(range(len(baseline_top)), baseline_top['importance'], color='skyblue')
    plt.yticks(range(len(baseline_top)), baseline_top['feature'])
    plt.title('Baseline Dataset - Top 15 Features by Importance')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_analysis/baseline_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Enhanced plot
    plt.figure(figsize=(12, 8))
    enhanced_top = enhanced_importance.head(15)
    plt.barh(range(len(enhanced_top)), enhanced_top['importance'], color='lightcoral')
    plt.yticks(range(len(enhanced_top)), enhanced_top['feature'])
    plt.title('Enhanced Dataset - Top 15 Features by Importance')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_analysis/enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Baseline top 10
    baseline_top_10 = baseline_importance.head(10)
    ax1.barh(range(len(baseline_top_10)), baseline_top_10['importance'], color='skyblue')
    ax1.set_yticks(range(len(baseline_top_10)))
    ax1.set_yticklabels(baseline_top_10['feature'])
    ax1.set_title('Baseline Dataset\nTop 10 Features')
    ax1.set_xlabel('Feature Importance')

    # Enhanced top 10
    enhanced_top_10 = enhanced_importance.head(10)
    ax2.barh(range(len(enhanced_top_10)), enhanced_top_10['importance'], color='lightcoral')
    ax2.set_yticks(range(len(enhanced_top_10)))
    ax2.set_yticklabels(enhanced_top_10['feature'])
    ax2.set_title('Enhanced Dataset\nTop 10 Features')
    ax2.set_xlabel('Feature Importance')

    plt.tight_layout()
    plt.savefig('feature_analysis/baseline_vs_enhanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plots saved to 'feature_analysis/' directory:")
    print(f"   - baseline_feature_importance.png")
    print(f"   - enhanced_feature_importance.png")
    print(f"   - baseline_vs_enhanced_comparison.png")

def analyze_feature_categories(baseline_importance, enhanced_importance):
    """Analyze feature categories"""
    print(f"\n📈 FEATURE CATEGORY ANALYSIS")
    print("=" * 50)

    # Define feature categories
    baseline_categories = {
        'Team Encoding': ['HomeTeam_encoded', 'AwayTeam_encoded'],
        'Shots': ['shot_difference', 'total_shots', 'shot_on_target_difference', 'total_shots_on_target'],
        'Discipline': ['yellow_card_difference', 'total_yellow_cards', 'red_card_difference', 'total_red_cards'],
        'Set Pieces': ['corner_difference', 'total_corners'],
        'Fouls': ['foul_difference', 'total_fouls'],
        'Basic Stats': ['HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'],
        'Home Advantage': ['is_home']
    }

    enhanced_categories = {
        'Momentum': ['home_momentum_score', 'away_momentum_score', 'momentum_difference'],
        'Form': ['home_recent_form', 'away_recent_form', 'form_difference'],
        'Rest/Fatigue': ['home_rest_days', 'away_rest_days', 'rest_days_difference', 'home_fatigue_7d', 'away_fatigue_7d'],
        'Workload': ['home_matches_last_7d', 'away_matches_last_7d', 'home_matches_last_14d', 'away_matches_last_14d'],
        'Impact': ['home_last_result_impact', 'away_last_result_impact']
    }

    def calculate_category_scores(importance_df, categories):
        scores = {}
        for category, features in categories.items():
            category_score = 0
            feature_count = 0
            for feature in features:
                feature_data = importance_df[importance_df['feature'] == feature]
                if not feature_data.empty:
                    category_score += feature_data.iloc[0]['importance']
                    feature_count += 1
            if feature_count > 0:
                scores[category] = category_score / feature_count
        return scores

    # Calculate category scores
    baseline_scores = calculate_category_scores(baseline_importance, baseline_categories)
    enhanced_scores = calculate_category_scores(enhanced_importance, enhanced_categories)

    print(f"\n📊 BASELINE FEATURE CATEGORIES:")
    for category, score in sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {category:20s}: {score:.4f}")

    print(f"\n📊 ENHANCED FEATURE CATEGORIES:")
    for category, score in sorted(enhanced_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {category:20s}: {score:.4f}")

    return baseline_scores, enhanced_scores

def create_summary_report(baseline_importance, enhanced_importance, baseline_scores, enhanced_scores):
    """Create comprehensive summary report"""
    print(f"\n📋 CREATING SUMMARY REPORT")
    print("=" * 50)

    # Create output directory
    os.makedirs('feature_analysis', exist_ok=True)

    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_top_10': baseline_importance.head(10).to_dict('records'),
        'enhanced_top_10': enhanced_importance.head(10).to_dict('records'),
        'baseline_categories': baseline_scores,
        'enhanced_categories': enhanced_scores
    }

    # Save detailed report
    with open('feature_analysis/feature_analysis_report.pkl', 'wb') as f:
        pickle.dump(report, f)

    # Create text summary
    with open('feature_analysis/feature_analysis_summary.txt', 'w') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {report['timestamp']}\n\n")

        f.write("BASELINE DATASET INSIGHTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Top Feature: {baseline_importance.iloc[0]['feature']} ")
        f.write(f"(Importance: {baseline_importance.iloc[0]['importance']:.4f})\n")
        f.write(f"Top Category: {max(baseline_scores, key=baseline_scores.get)}\n\n")

        f.write("ENHANCED DATASET INSIGHTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Top Feature: {enhanced_importance.iloc[0]['feature']} ")
        f.write(f"(Importance: {enhanced_importance.iloc[0]['importance']:.4f})\n")
        f.write(f"Top Category: {max(enhanced_scores, key=enhanced_scores.get)}\n\n")

        f.write("BASELINE TOP 10 FEATURES:\n")
        f.write("-" * 30 + "\n")
        for i, row in baseline_importance.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['feature']:30s}: {row['importance']:.4f}\n")

        f.write("\nENHANCED TOP 10 FEATURES:\n")
        f.write("-" * 30 + "\n")
        for i, row in enhanced_importance.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['feature']:30s}: {row['importance']:.4f}\n")

    print(f"✅ Summary report saved:")
    print(f"   - feature_analysis_report.pkl")
    print(f"   - feature_analysis_summary.txt")

def main():
    """Main feature analysis function"""
    print("🚀 SIMPLIFIED FEATURE IMPORTANCE ANALYSIS")
    print("Analyzing feature importance without complex SHAP computations\n")

    # Load data
    data = load_data()

    # Train models
    models = train_shap_models(data)

    # Analyze feature importance
    baseline_importance = simple_feature_importance_analysis(
        models['baseline'],
        data['baseline']['X_train'],
        data['baseline']['features'],
        'baseline'
    )

    enhanced_importance = simple_feature_importance_analysis(
        models['enhanced'],
        data['enhanced']['X_train'],
        data['enhanced']['features'],
        'enhanced'
    )

    if baseline_importance is None or enhanced_importance is None:
        print("❌ Failed to get feature importance")
        return

    # Create plots
    create_feature_importance_plots(baseline_importance, enhanced_importance)

    # Analyze categories
    baseline_scores, enhanced_scores = analyze_feature_categories(baseline_importance, enhanced_importance)

    # Create summary report
    create_summary_report(baseline_importance, enhanced_importance, baseline_scores, enhanced_scores)

    print(f"\n🎉 FEATURE IMPORTANCE ANALYSIS COMPLETE!")
    print(f"\n📊 KEY INSIGHTS:")

    baseline_top = baseline_importance.iloc[0]
    enhanced_top = enhanced_importance.iloc[0]

    print(f"   ✅ Baseline top feature: {baseline_top['feature']} ({baseline_top['importance']:.4f})")
    print(f"   ✅ Enhanced top feature: {enhanced_top['feature']} ({enhanced_top['importance']:.4f})")
    print(f"   ✅ Baseline top category: {max(baseline_scores, key=baseline_scores.get)}")
    print(f"   ✅ Enhanced top category: {max(enhanced_scores, key=enhanced_scores.get)}")

    print(f"\n📁 Analysis saved to 'feature_analysis/' directory")

if __name__ == "__main__":
    main()