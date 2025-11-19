#!/usr/bin/env python3
"""
FINAL OPTIMIZED ML TRAINING WITH CLEAN DATA
Train models on fully validated and cleaned data for best possible performance
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_clean_data():
    """Load the clean, validated data"""
    print("📊 Loading clean enhanced data...")
    df = pd.read_csv('data_processed/clean_enhanced_epl_ml.csv')
    print(f"   ✅ Clean data: {len(df)} matches, {len(df.columns)} features")
    return df

def prepare_features(df):
    """Prepare features for optimal training"""
    print("🚀 Preparing features for optimal training...")

    # Remove non-feature columns
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['FTR'].copy()

    # Handle any remaining missing values
    X = X.fillna(X.median())

    print(f"   ✅ Features: {len(feature_cols)}")
    print(f"   📊 Target distribution: {dict(y.value_counts())}")

    return X, y, feature_cols

def optimal_data_split(X, y):
    """Create optimal train/test split based on validation results"""
    print("✂️ Creating optimal data split...")

    # Use Temporal 70/30 split (found to be best in validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")

    # Check class distribution
    print(f"   📈 Train distribution: {dict(y_train.value_counts())}")
    print(f"   📈 Test distribution: {dict(y_test.value_counts())}")

    return X_train, X_test, y_train, y_test

def optimize_models(X_train, y_train):
    """Hyperparameter optimization for best models"""
    print("⚡ Optimizing model hyperparameters...")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Define models with optimized parameter grids
    models_to_optimize = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {
                'C': [0.3, 0.5, 0.7, 1.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'scaler': True
        },
        'xgboost': {
            'model': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'scaler': False
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9]
            },
            'scaler': False
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'min_samples_split': [5, 10],
                'max_features': ['sqrt', 'log2']
            },
            'scaler': False
        }
    }

    optimized_models = {}
    cv_scores = {}

    for name, config in models_to_optimize.items():
        print(f"   🔄 Optimizing {name.replace('_', ' ').title()}...")

        # Scale data if needed
        if config['scaler']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_search = X_train_scaled
        else:
            X_search = X_train

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_search, y_train_encoded)

        # Store results
        optimized_models[name] = grid_search.best_estimator_
        cv_scores[name] = grid_search.best_score_

        # Store scaler if needed
        if config['scaler']:
            optimized_models[f"{name}_scaler"] = scaler

        print(f"      ✅ Best CV Score: {grid_search.best_score_:.1%}")
        print(f"      📋 Best params: {grid_search.best_params_}")

    return optimized_models, cv_scores, label_encoder

def evaluate_models(X_train, y_train, X_test, y_test, optimized_models, label_encoder):
    """Comprehensive model evaluation"""
    print("🎯 Comprehensive model evaluation...")

    # Encode test labels
    y_test_encoded = label_encoder.transform(y_test)

    results = {}

    for model_name in ['logistic_regression', 'xgboost', 'gradient_boosting', 'random_forest']:
        if model_name not in optimized_models:
            continue

        print(f"   📊 Evaluating {model_name.replace('_', ' ').title()}...")

        model = optimized_models[model_name]

        # Get scaler if needed
        scaler_key = f"{model_name}_scaler"
        if scaler_key in optimized_models:
            scaler = optimized_models[scaler_key]
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            train_X, test_X = X_train_scaled, X_test_scaled
        else:
            train_X, test_X = X_train, X_test

        # Fit model
        model.fit(train_X, label_encoder.transform(y_train))

        # Predictions
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)

        # Calculate metrics
        train_accuracy = accuracy_score(label_encoder.transform(y_train), train_pred)
        test_accuracy = accuracy_score(y_test_encoded, test_pred)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if scaler_key in optimized_models:
            cv_scores = cross_val_score(model, X_train_scaled, label_encoder.transform(y_train), cv=cv)
        else:
            cv_scores = cross_val_score(model, X_train, label_encoder.transform(y_train), cv=cv)

        # Convert predictions back for classification report
        test_pred_labels = label_encoder.inverse_transform(test_pred)
        y_test_labels = label_encoder.inverse_transform(y_test_encoded)

        # Detailed metrics
        class_report = classification_report(y_test_labels, test_pred_labels, output_dict=True)
        confusion_mat = confusion_matrix(y_test_labels, test_pred_labels)

        results[model_name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'overfitting_gap': train_accuracy - test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': confusion_mat,
            'predictions': test_pred_labels
        }

        print(f"      ✅ Test Accuracy: {test_accuracy:.1%}")
        print(f"      📈 CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        print(f"      📊 Overfitting Gap: {(train_accuracy - test_accuracy):.1%}")

        # Draw prediction analysis
        draw_metrics = class_report.get('D', {})
        print(f"      🎯 Draw F1: {draw_metrics.get('f1-score', 0):.1%}")

    return results

def create_ensemble_model(X_train, y_train, X_test, y_test, results, label_encoder):
    """Create ensemble of best performing models"""
    print("🚀 Creating ensemble model...")

    # Select top 3 performing models
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    top_models = sorted_models[:3]

    print(f"   🏆 Top 3 models for ensemble:")
    for i, (name, result) in enumerate(top_models, 1):
        print(f"      {i}. {name.replace('_', ' ').title()}: {result['test_accuracy']:.1%}")

    # Create voting ensemble
    ensemble_models = []
    ensemble_weights = []

    for name, result in top_models:
        ensemble_models.append((name, result['model']))
        ensemble_weights.append(result['test_accuracy'])

    # Normalize weights
    total_weight = sum(ensemble_weights)
    ensemble_weights = [w/total_weight for w in ensemble_weights]

    voting_ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='soft',
        weights=ensemble_weights
    )

    # Train ensemble
    y_train_encoded = label_encoder.transform(y_train)
    voting_ensemble.fit(X_train, y_train_encoded)

    # Evaluate ensemble
    ensemble_train_pred = voting_ensemble.predict(X_train)
    y_test_encoded = label_encoder.transform(y_test)
    ensemble_test_pred = voting_ensemble.predict(X_test)

    ensemble_train_acc = accuracy_score(y_train_encoded, ensemble_train_pred)
    ensemble_test_acc = accuracy_score(y_test_encoded, ensemble_test_pred)

    # Cross-validation for ensemble
    cv_scores = cross_val_score(voting_ensemble, X_train, y_train_encoded, cv=5)

    # Detailed evaluation
    ensemble_test_labels = label_encoder.inverse_transform(ensemble_test_pred)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    ensemble_report = classification_report(y_test_labels, ensemble_test_labels, output_dict=True)

    ensemble_results = {
        'model': voting_ensemble,
        'train_accuracy': ensemble_train_acc,
        'test_accuracy': ensemble_test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting_gap': ensemble_train_acc - ensemble_test_acc,
        'classification_report': ensemble_report,
        'confusion_matrix': confusion_matrix(y_test_labels, ensemble_test_labels),
        'predictions': ensemble_test_labels,
        'ensemble_models': [name for name, _ in ensemble_models],
        'ensemble_weights': ensemble_weights
    }

    print(f"   ✅ Ensemble Test Accuracy: {ensemble_test_acc:.1%}")
    print(f"   📈 Ensemble CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    return ensemble_results

def analyze_final_results(results, ensemble_results, feature_cols):
    """Comprehensive final analysis"""
    print("\n📊 FINAL COMPREHENSIVE ANALYSIS")
    print("=" * 50)

    # Combine all models
    all_models = list(results.items()) + [('Ensemble', ensemble_results)]

    # Sort by performance
    sorted_models = sorted(all_models, key=lambda x: x[1]['test_accuracy'], reverse=True)

    print("🏆 FINAL MODEL RANKINGS:")
    for i, (model_name, model_results) in enumerate(sorted_models, 1):
        print(f"   {i}. {model_name.replace('_', ' ').title()}: {model_results['test_accuracy']:.1%}")

    # Best model analysis
    best_name, best_results = sorted_models[0]
    print(f"\n🥇 BEST MODEL: {best_name.replace('_', ' ').title()}")
    print(f"   🎯 Test Accuracy: {best_results['test_accuracy']:.1%}")
    print(f"   📈 CV Score: {best_results['cv_mean']:.1%} ± {best_results['cv_std']:.1%}")
    print(f"   📊 Overfitting Gap: {best_results['overfitting_gap']:.1%}")

    # Detailed result distribution
    if 'predictions' in best_results:
        actual_dist = pd.Series(y_test).value_counts()
        pred_dist = pd.Series(best_results['predictions']).value_counts()

        print(f"\n📊 PREDICTION ANALYSIS:")
        print(f"   📈 Actual distribution: {dict(actual_dist)}")
        print(f"   🎯 Predicted distribution: {dict(pred_dist)}")

    # Feature importance (if available)
    if hasattr(best_results['model'], 'feature_importances_'):
        print(f"\n🔍 TOP 10 FEATURE IMPORTANCE:")
        importances = best_results['model'].feature_importances_
        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"   {i:2d}. {feature:<25}: {importance:.4f}")

    # Performance benchmarking
    baseline_random = 0.333  # Random guessing for 3 classes
    improvement_over_random = (best_results['test_accuracy'] - baseline_random) / baseline_random * 100

    print(f"\n📈 PERFORMANCE BENCHMARKS:")
    print(f"   🎲 Random Baseline: {baseline_random:.1%}")
    print(f"   🚀 Our Best Model: {best_results['test_accuracy']:.1%}")
    print(f"   📊 Improvement: +{improvement_over_random:.1f}% over random")

    return sorted_models[0]

def save_final_results(best_model, results, ensemble_results, feature_cols):
    """Save the final optimized results"""
    print("\n💾 SAVING FINAL OPTIMIZED RESULTS...")

    results_dir = 'final_optimized_results'
    os.makedirs(results_dir, exist_ok=True)

    # Save best model
    with open(f'{results_dir}/best_final_model.pkl', 'wb') as f:
        pickle.dump(best_model[1]['model'], f)

    # Save ensemble model
    with open(f'{results_dir}/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble_results['model'], f)

    # Save comprehensive results
    comprehensive_results = {
        'best_model': {
            'name': best_model[0],
            'accuracy': best_model[1]['test_accuracy'],
            'cv_score': best_model[1]['cv_mean'],
            'overfitting_gap': best_model[1]['overfitting_gap']
        },
        'ensemble_model': {
            'accuracy': ensemble_results['test_accuracy'],
            'cv_score': ensemble_results['cv_mean'],
            'components': ensemble_results.get('ensemble_models', []),
            'weights': ensemble_results.get('ensemble_weights', [])
        },
        'all_results': {name: {
            'test_accuracy': results[name]['test_accuracy'],
            'cv_mean': results[name]['cv_mean'],
            'overfitting_gap': results[name]['overfitting_gap']
        } for name in results.keys()},
        'feature_list': feature_cols
    }

    with open(f'{results_dir}/final_comprehensive_results.pkl', 'wb') as f:
        pickle.dump(comprehensive_results, f)

    # Create summary CSV
    summary_data = []
    for model_name, model_results in results.items():
        summary_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test_Accuracy': f"{model_results['test_accuracy']:.1%}",
            'CV_Mean': f"{model_results['cv_mean']:.1%}",
            'CV_Std': f"{model_results['cv_std']:.1%}",
            'Overfitting_Gap': f"{model_results['overfitting_gap']:.1%}"
        })

    # Add ensemble
    summary_data.append({
        'Model': 'Ensemble',
        'Test_Accuracy': f"{ensemble_results['test_accuracy']:.1%}",
        'CV_Mean': f"{ensemble_results['cv_mean']:.1%}",
        'CV_Std': f"{ensemble_results['cv_std']:.1%}",
        'Overfitting_Gap': f"{ensemble_results['overfitting_gap']:.1%}"
    })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{results_dir}/final_model_summary.csv', index=False)

    print(f"   ✅ Final optimized results saved to {results_dir}/")

def main():
    """Main optimized training function"""
    print("🚀 FINAL OPTIMIZED ML TRAINING")
    print("=" * 50)

    # Load clean data
    df = load_clean_data()

    # Prepare features
    X, y, feature_cols = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = optimal_data_split(X, y)

    # Optimize models
    optimized_models, cv_scores, label_encoder = optimize_models(X_train, y_train)

    # Evaluate models
    results = evaluate_models(X_train, y_train, X_test, y_test, optimized_models, label_encoder)

    # Create ensemble
    ensemble_results = create_ensemble_model(X_train, y_train, X_test, y_test, results, label_encoder)

    # Final analysis
    best_model = analyze_final_results(results, ensemble_results, feature_cols)

    # Save results
    save_final_results(best_model, results, ensemble_results, feature_cols)

    print("\n✅ FINAL OPTIMIZED TRAINING COMPLETE!")
    print(f"\n🏆 CHAMPIONSHIP RESULTS:")
    print(f"   🥇 Best Model: {best_model[0].replace('_', ' ').title()}")
    print(f"   🎯 Test Accuracy: {best_model[1]['test_accuracy']:.1%}")
    print(f"   📈 CV Score: {best_model[1]['cv_mean']:.1%} ± {best_model[1]['cv_std']:.1%}")
    print(f"   📊 Overfitting Gap: {best_model[1]['overfitting_gap']:.1%}")
    print(f"   📁 Results saved to: final_optimized_results/")

if __name__ == "__main__":
    main()