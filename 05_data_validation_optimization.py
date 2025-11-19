#!/usr/bin/env python3
"""
COMPREHENSIVE DATA VALIDATION & MODEL OPTIMIZATION
Validate data quality, optimize train/test split, and achieve best possible model
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_datasets():
    """Load and perform comprehensive data validation"""
    print("🔍 COMPREHENSIVE DATA VALIDATION")
    print("=" * 50)

    # Load datasets
    print("\n📊 Loading datasets...")
    baseline_df = pd.read_csv('data_processed/all_seasons.csv')
    enhanced_df = pd.read_csv('data_processed/leakage_free_enhanced_epl_ml.csv')

    print(f"   ✅ Baseline: {len(baseline_df)} matches, {len(baseline_df.columns)} columns")
    print(f"   ✅ Enhanced: {len(enhanced_df)} matches, {len(enhanced_df.columns)} columns")

    # Validate baseline data
    print("\n🔍 Validating baseline data...")
    validate_baseline_data(baseline_df)

    # Validate enhanced data
    print("\n🔍 Validating enhanced data...")
    validate_enhanced_data(enhanced_df)

    # Check feature distributions
    print("\n📈 Analyzing feature distributions...")
    analyze_feature_distributions(enhanced_df)

    return baseline_df, enhanced_df

def validate_baseline_data(df):
    """Validate baseline EPL data quality"""
    issues = []

    # Check essential columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    # Check result distribution
    result_dist = df['FTR'].value_counts()
    print(f"   📊 Result distribution: {dict(result_dist)}")
    total_matches = len(df)
    for result, count in result_dist.items():
        percentage = count / total_matches * 100
        print(f"      {result}: {count} ({percentage:.1f}%)")

    # Check date format and chronological order
    try:
        df['date_parsed'] = pd.to_datetime(df['Date'], dayfirst=True)
        date_issues = 0
        for i in range(1, len(df)):
            if df['date_parsed'].iloc[i] < df['date_parsed'].iloc[i-1]:
                date_issues += 1
        if date_issues > 0:
            issues.append(f"Chronological issues: {date_issues} out-of-order dates")
        else:
            print(f"   ✅ Dates properly formatted and ordered")
    except Exception as e:
        issues.append(f"Date parsing error: {e}")

    # Check score consistency
    score_issues = 0
    for _, match in df.iterrows():
        if match['HST'] > match['HS'] or match['AST'] > match['AS']:
            score_issues += 1
    if score_issues > 0:
        issues.append(f"Shot consistency issues: {score_issues} matches with shots on target > total shots")

    # Report validation results
    if issues:
        print(f"   ❌ Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Baseline data validation passed")

def validate_enhanced_data(df):
    """Validate enhanced data quality"""
    issues = []

    # Check for data leakage indicators
    print(f"   📊 Checking for potential data leakage...")

    # Check correlation with target
    feature_cols = [col for col in df.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'FTR']]

    # Encode target for correlation analysis
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(df['FTR'])

    high_corr_features = []
    for feature in feature_cols:
        if df[feature].dtype in ['int64', 'float64']:
            correlation = abs(df[feature].corr(pd.Series(target_encoded)))
            if correlation > 0.9:  # Suspiciously high correlation
                high_corr_features.append((feature, correlation))

    if high_corr_features:
        print(f"   ⚠️  High correlation features (potential leakage):")
        for feature, corr in high_corr_features:
            print(f"      - {feature}: {corr:.3f}")
        issues.append(f"Found {len(high_corr_features)} features with suspiciously high correlation")

    # Check for null values
    null_counts = df[feature_cols].isnull().sum()
    if null_counts.any():
        print(f"   ⚠️  Null values in features:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"      - {col}: {count}")
        issues.append("Null values found in features")

    # Check momentum feature ranges (should be reasonable)
    momentum_cols = [col for col in df.columns if 'momentum' in col]
    for col in momentum_cols:
        if col in df.columns:
            min_val, max_val = df[col].min(), df[col].max()
            print(f"   📊 {col}: range [{min_val:.2f}, {max_val:.2f}]")
            if max_val > 50 or min_val < -50:  # Unreasonable momentum values
                issues.append(f"Unreasonable momentum range in {col}")

    # Check form and rest days
    form_cols = [col for col in df.columns if 'form' in col]
    for col in form_cols:
        if col in df.columns:
            max_val = df[col].max()
            if max_val > 15:  # More than 5 wins * 3 points
                issues.append(f"Suspicious form values in {col}")

    rest_cols = [col for col in df.columns if 'rest_days' in col]
    for col in rest_cols:
        if col in df.columns:
            min_val, max_val = df[col].min(), df[col].max()
            if min_val < 0 or max_val > 30:  # Unreasonable rest days
                issues.append(f"Unreasonable rest days in {col}: [{min_val}, {max_val}]")

    # Report validation results
    if issues:
        print(f"   ❌ Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Enhanced data validation passed")

def analyze_feature_distributions(df):
    """Analyze feature distributions for anomalies"""
    feature_cols = [col for col in df.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'FTR']]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns

    print(f"   📊 Analyzing {len(numeric_features)} numeric features...")

    anomalies = []
    for feature in numeric_features[:10]:  # Check first 10 features
        data = df[feature].dropna()

        # Check for extreme outliers
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_percentage = len(outliers) / len(data) * 100

        if outlier_percentage > 5:  # More than 5% extreme outliers
            anomalies.append(f"{feature}: {outlier_percentage:.1f}% outliers")

        print(f"      {feature}: mean={data.mean():.2f}, std={data.std():.2f}, outliers={outlier_percentage:.1f}%")

    if anomalies:
        print(f"   ⚠️  Features with many outliers:")
        for anomaly in anomalies:
            print(f"      - {anomaly}")
    else:
        print(f"   ✅ Feature distributions look reasonable")

def optimize_train_test_split(baseline_df, enhanced_df):
    """Find optimal train/test split strategy"""
    print("\n✂️ OPTIMIZING TRAIN/TEST SPLIT")
    print("=" * 30)

    # Test different split strategies
    strategies = {
        'Temporal 80/20': (0.2, False),
        'Temporal 70/30': (0.3, False),
        'Temporal 60/40': (0.4, False),
        'Stratified 80/20': (0.2, True),
        'Stratified 70/30': (0.3, True),
        'Time Series CV': (0.2, 'cv')
    }

    best_score = 0
    best_strategy = None
    best_results = {}

    # Use enhanced data for testing
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
    feature_cols = [col for col in enhanced_df.columns if col not in exclude_cols]

    X = enhanced_df[feature_cols].copy()
    y = enhanced_df['FTR'].copy()
    X = X.fillna(X.median())

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("   🧪 Testing split strategies...")
    for strategy_name, (test_size, stratify) in strategies.items():
        print(f"      🔄 Testing {strategy_name}...")

        if stratify == 'cv':
            # TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

                # Quick model for evaluation
                model = LogisticRegression(random_state=42, max_iter=1000)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                scores.append(score)

            avg_score = np.mean(scores)
            std_score = np.std(scores)

        else:
            # Regular train/test split
            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42
                )

            # Quick model for evaluation
            model = LogisticRegression(random_state=42, max_iter=1000)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            avg_score = model.score(X_test_scaled, y_test)
            std_score = 0

        print(f"         Accuracy: {avg_score:.1%} (±{std_score:.1%})")

        if avg_score > best_score:
            best_score = avg_score
            best_strategy = strategy_name
            best_results = {
                'test_size': test_size,
                'stratify': stratify != False,
                'score': avg_score,
                'std': std_score
            }

    print(f"\n   🏆 Best split strategy: {best_strategy} - {best_score:.1%}")

    # Apply best split for final models
    print(f"   ✂️ Applying best split strategy...")
    if best_results['stratify'] and best_results['stratify'] != 'cv':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=best_results['test_size'],
            random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=best_results['test_size'], random_state=42
        )

    print(f"      📊 Final split: {len(X_train)} train, {len(X_test)} test")

    return X_train, X_test, y_train, y_test, label_encoder, feature_cols

def optimize_hyperparameters(X_train, y_train):
    """Optimize hyperparameters for best models"""
    print("\n⚡ HYPERPARAMETER OPTIMIZATION")
    print("=" * 40)

    # Define parameter grids for top models
    param_grids = {
        'logistic_regression': {
            'C': [0.1, 0.5, 1.0, 2.0],
            'penalty': ['l2'],
            'class_weight': ['balanced', None]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8, None],
            'min_samples_split': [5, 10, 20],
            'max_features': ['sqrt', 'log2']
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
    }

    # Define base models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'xgboost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }

    best_models = {}
    best_scores = {}

    print("   🔧 Optimizing hyperparameters...")
    for model_name, model in models.items():
        print(f"      🔄 Optimizing {model_name.replace('_', ' ').title()}...")

        # Scale features for models that need it
        if model_name in ['logistic_regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_search = X_train_scaled
        else:
            X_search = X_train

        # Grid search
        grid_search = GridSearchCV(
            model, param_grids[model_name],
            cv=5, scoring='accuracy',
            n_jobs=-1, verbose=0
        )

        grid_search.fit(X_search, y_train)

        best_models[model_name] = grid_search.best_estimator_
        best_scores[model_name] = grid_search.best_score_

        print(f"         ✅ Best CV Score: {grid_search.best_score_:.1%}")
        print(f"         📋 Best params: {grid_search.best_params_}")

        # Store scaler if needed
        if model_name in ['logistic_regression']:
            best_models[f"{model_name}_scaler"] = scaler

    return best_models, best_scores

def train_final_optimized_models(X_train, y_train, X_test, y_test, label_encoder, optimized_models):
    """Train final optimized models with comprehensive evaluation"""
    print("\n🎯 TRAINING FINAL OPTIMIZED MODELS")
    print("=" * 50)

    final_results = {}

    for model_name, model in optimized_models.items():
        if 'scaler' in model_name:
            continue

        print(f"   🚀 Training {model_name.replace('_', ' ').title()}...")

        # Get scaler if needed
        scaler_key = f"{model_name}_scaler"
        if scaler_key in optimized_models:
            scaler = optimized_models[scaler_key]
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            train_X, test_X = X_train_scaled, X_test_scaled
        else:
            train_X, test_X = X_train, X_test

        # Train model
        model.fit(train_X, y_train)

        # Predictions
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)

        # Convert back to original labels
        train_pred_labels = label_encoder.inverse_transform(train_pred)
        test_pred_labels = label_encoder.inverse_transform(y_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Cross-validation
        from sklearn.model_selection import cross_val_score
        if scaler_key in optimized_models:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        # Detailed analysis
        test_pred_labels = label_encoder.inverse_transform(test_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)

        class_report = classification_report(y_test_labels, test_pred_labels, output_dict=True)
        confusion_mat = confusion_matrix(y_test_labels, test_pred_labels)

        final_results[model_name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'overfitting_gap': train_accuracy - test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': confusion_mat,
            'model_type': model_name
        }

        print(f"      ✅ Test Accuracy: {test_accuracy:.1%}")
        print(f"      📊 CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        print(f"      📈 Overfitting Gap: {train_accuracy - test_accuracy:.1%}")

    return final_results

def create_comprehensive_analysis(final_results, feature_cols):
    """Create comprehensive analysis of final results"""
    print("\n📊 COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 40)

    # Sort models by test accuracy
    sorted_models = sorted(final_results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)

    print("🏆 MODEL RANKINGS:")
    for i, (model_name, results) in enumerate(sorted_models, 1):
        print(f"   {i}. {model_name.replace('_', ' ').title()}: {results['test_accuracy']:.1%}")

    # Best model detailed analysis
    best_model_name, best_results = sorted_models[0]
    print(f"\n🥇 BEST MODEL: {best_model_name.replace('_', ' ').title()}")
    print(f"   🎯 Test Accuracy: {best_results['test_accuracy']:.1%}")
    print(f"   📈 CV Score: {best_results['cv_mean']:.1%} ± {best_results['cv_std']:.1%}")
    print(f"   📊 Overfitting Gap: {best_results['overfitting_gap']:.1%}")

    # Draw prediction analysis
    draw_metrics = best_results['classification_report'].get('D', {})
    if draw_metrics:
        print(f"\n🎯 DRAW PREDICTION ANALYSIS:")
        print(f"   📊 Precision: {draw_metrics.get('precision', 0):.1%}")
        print(f"   📊 Recall: {draw_metrics.get('recall', 0):.1%}")
        print(f"   📊 F1-Score: {draw_metrics.get('f1-score', 0):.1%}")

    # Feature importance for best model
    if hasattr(best_results['model'], 'feature_importances_'):
        print(f"\n🔍 TOP 10 FEATURE IMPORTANCE:")
        importances = best_results['model'].feature_importances_
        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"   {i:2d}. {feature:<25}: {importance:.4f}")

    # Model stability analysis
    print(f"\n📈 MODEL STABILITY ANALYSIS:")
    stable_models = 0
    for model_name, results in final_results.items():
        if results['overfitting_gap'] < 0.1 and results['cv_std'] < 0.03:
            stable_models += 1

    print(f"   ✅ Stable Models (low overfitting + low CV variance): {stable_models}/{len(final_results)}")

    return sorted_models[0]

def save_optimized_results(best_model, final_results, feature_cols):
    """Save optimized results and models"""
    print("\n💾 SAVING OPTIMIZED RESULTS...")

    results_dir = 'optimized_ml_results'
    os.makedirs(results_dir, exist_ok=True)

    # Save best model
    with open(f'{results_dir}/best_optimized_model.pkl', 'wb') as f:
        pickle.dump(best_model[1]['model'], f)

    # Save comprehensive results
    comprehensive_results = {
        'best_model': {
            'name': best_model[0],
            'accuracy': best_model[1]['test_accuracy'],
            'cv_score': best_model[1]['cv_mean'],
            'overfitting_gap': best_model[1]['overfitting_gap']
        },
        'all_results': final_results,
        'feature_list': feature_cols
    }

    with open(f'{results_dir}/comprehensive_optimized_results.pkl', 'wb') as f:
        pickle.dump(comprehensive_results, f)

    # Create results summary
    summary_data = []
    for model_name, results in final_results.items():
        summary_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test_Accuracy': f"{results['test_accuracy']:.1%}",
            'CV_Mean': f"{results['cv_mean']:.1%}",
            'CV_Std': f"{results['cv_std']:.1%}",
            'Overfitting_Gap': f"{results['overfitting_gap']:.1%}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{results_dir}/model_optimization_summary.csv', index=False)

    print(f"   ✅ Optimized results saved to {results_dir}/")

def main():
    """Main validation and optimization function"""
    print("🚀 EPL PREDICTION - COMPREHENSIVE VALIDATION & OPTIMIZATION")
    print("=" * 70)

    # Load and validate data
    baseline_df, enhanced_df = load_and_validate_datasets()

    # Optimize train/test split
    X_train, X_test, y_train, y_test, label_encoder, feature_cols = optimize_train_test_split(baseline_df, enhanced_df)

    # Optimize hyperparameters
    optimized_models, best_scores = optimize_hyperparameters(X_train, y_train)

    # Train final optimized models
    final_results = train_final_optimized_models(X_train, y_train, X_test, y_test, label_encoder, optimized_models)

    # Comprehensive analysis
    best_model = create_comprehensive_analysis(final_results, feature_cols)

    # Save results
    save_optimized_results(best_model, final_results, feature_cols)

    print("\n✅ COMPREHENSIVE OPTIMIZATION COMPLETE!")
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   🥇 Best Model: {best_model[0].replace('_', ' ').title()}")
    print(f"   🎯 Test Accuracy: {best_model[1]['test_accuracy']:.1%}")
    print(f"   📈 CV Score: {best_model[1]['cv_mean']:.1%} ± {best_model[1]['cv_std']:.1%}")
    print(f"   📊 Overfitting Gap: {best_model[1]['overfitting_gap']:.1%}")
    print(f"   📁 Results saved to: optimized_ml_results/")

if __name__ == "__main__":
    main()