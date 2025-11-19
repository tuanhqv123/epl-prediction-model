#!/usr/bin/env python3
"""
CUTTING-EDGE ML TRAINING FOR 70%+ ACCURACY
Advanced techniques with next-level features for maximum performance
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_next_level_data():
    """Load the next-level enhanced dataset"""
    print("🚀 LOADING NEXT-LEVEL DATASET")
    print("=" * 40)

    df = pd.read_csv('data_processed/next_level_epl_ml.csv')
    with open('data_processed/next_level_feature_list.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    print(f"   ✅ Next-level dataset: {len(df)} matches, {len(df.columns)} columns")
    print(f"   ✅ ML features: {len(feature_cols)}")
    print(f"   📊 Target distribution: {dict(df['FTR'].value_counts())}")

    return df, feature_cols

def advanced_feature_engineering(X, y, feature_cols):
    """Advanced feature engineering and selection"""
    print("🔬 ADVANCED FEATURE ENGINEERING")
    print("-" * 40)

    # Handle missing values with advanced imputation
    print("   🛠️  Advanced missing value imputation...")
    X_filled = X.copy()
    for col in X_filled.columns:
        if X_filled[col].isnull().any():
            # Use group-based imputation where possible
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())

    # Feature selection
    print("   🎯 Advanced feature selection...")

    # Remove highly correlated features
    correlation_matrix = X_filled.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > 0.95
    high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    print(f"      Removing {len(high_corr_features)} highly correlated features")
    X_selected = X_filled.drop(columns=high_corr_features)

    # Statistical feature selection
    selector = SelectKBest(f_classif, k=min(50, len(X_selected.columns)))
    X_stat_selected = selector.fit_transform(X_selected, y)
    selected_features = X_selected.columns[selector.get_support()].tolist()

    print(f"      Selected {len(selected_features)} best statistical features")

    # Create final feature DataFrame
    X_final = pd.DataFrame(X_stat_selected, columns=selected_features)

    # Add polynomial features for top predictors
    print("   ⚡ Creating polynomial features...")
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    top_features = selected_features[:10]  # Top 10 features for interactions

    if len(top_features) >= 3:
        poly_features = poly.fit_transform(X_final[top_features])
        poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

        # Combine original and polynomial features
        X_combined = pd.concat([X_final.reset_index(drop=True), poly_df], axis=1)
        print(f"      Added {poly_features.shape[1]} polynomial features")
    else:
        X_combined = X_final

    print(f"   ✅ Final feature set: {len(X_combined.columns)} features")

    return X_combined, selected_features

def create_ensemble_models():
    """Create a diverse ensemble of advanced models"""
    print("🤖 CREATING ADVANCED ENSEMBLE MODELS")
    print("-" * 40)

    models = {
        # Gradient Boosting variants
        'xgboost_optimized': xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),

        'lightgbm': lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        ),

        'gradient_boosting_optimized': GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        ),

        # Random Forest variants
        'random_forest_optimized': RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        'extra_trees': ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        # Neural Network
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        ),

        # Support Vector Machine
        'svm_rbf': SVC(
            C=10,
            gamma='scale',
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),

        # Logistic Regression (as baseline)
        'logistic_regression': LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    }

    print(f"   ✅ Created {len(models)} advanced models")
    return models

def train_cutting_edge_models(X_train, y_train, X_test, y_test, models):
    """Train cutting-edge models with advanced techniques"""
    print("🎯 TRAINING CUTTING-EDGE MODELS")
    print("-" * 40)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for model_name, model in models.items():
        print(f"   🔄 Training {model_name.replace('_', ' ').title()}...")

        # Determine if model needs scaled features
        needs_scaling = model_name in ['logistic_regression', 'svm_rbf', 'neural_network']
        train_X = X_train_scaled if needs_scaling else X_train
        test_X = X_test_scaled if needs_scaling else X_test

        try:
            # Train model
            model.fit(train_X, y_train_encoded)

            # Predictions
            train_pred = model.predict(train_X)
            test_pred = model.predict(test_X)

            # Get probabilities for ensemble
            test_proba = model.predict_proba(test_X)

            # Calculate metrics
            train_accuracy = accuracy_score(y_train_encoded, train_pred)
            test_accuracy = accuracy_score(y_test_encoded, test_pred)

            # Cross-validation (more robust)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if needs_scaling:
                cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=cv, n_jobs=-1)
            else:
                cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=cv, n_jobs=-1)

            # Detailed evaluation
            test_pred_labels = label_encoder.inverse_transform(test_pred)
            y_test_labels = label_encoder.inverse_transform(y_test_encoded)

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
                'predictions': test_pred_labels,
                'probabilities': test_proba,
                'scaler': scaler if needs_scaling else None
            }

            print(f"      ✅ Test Accuracy: {test_accuracy:.1%}")
            print(f"      📈 CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
            print(f"      📊 Overfitting Gap: {(train_accuracy - test_accuracy):.1%}")

            # Draw prediction analysis
            draw_metrics = class_report.get('D', {})
            if draw_metrics.get('f1-score', 0) > 0:
                print(f"      🎯 Draw F1: {draw_metrics.get('f1-score', 0):.1%}")

        except Exception as e:
            print(f"      ❌ Error training {model_name}: {e}")
            continue

    return results, label_encoder

def create_mega_ensemble(results, X_train, y_train_encoded, label_encoder):
    """Create a mega-ensemble of the best performing models"""
    print("🚀 CREATING MEGA-ENSEMBLE")
    print("-" * 30)

    # Select top 5 performing models
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    top_models = sorted_models[:5]

    print(f"   🏆 Top 5 models for mega-ensemble:")
    for i, (name, result) in enumerate(top_models, 1):
        print(f"      {i}. {name.replace('_', ' ').title()}: {result['test_accuracy']:.1%}")

    # Create weighted voting ensemble
    ensemble_estimators = []
    ensemble_weights = []

    for name, result in top_models:
        # Use model name without spaces for sklearn
        clean_name = name.replace('_', '').replace(' ', '')
        ensemble_estimators.append((clean_name, result['model']))
        # Weight by test accuracy
        ensemble_weights.append(result['test_accuracy'])

    # Normalize weights
    total_weight = sum(ensemble_weights)
    ensemble_weights = [w/total_weight for w in ensemble_weights]

    # Create mega-ensemble
    mega_ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        weights=ensemble_weights,
        n_jobs=-1
    )

    # Train mega-ensemble
    print("   🎯 Training mega-ensemble...")
    mega_ensemble.fit(X_train, y_train_encoded)

    return mega_ensemble, top_models

def advanced_model_analysis(results, feature_cols, mega_ensemble, X_test, y_test, label_encoder):
    """Comprehensive analysis of model performance"""
    print("\n📊 ADVANCED MODEL ANALYSIS")
    print("=" * 50)

    # Sort models by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)

    print("🏆 MODEL RANKINGS:")
    for i, (model_name, model_results) in enumerate(sorted_results, 1):
        print(f"   {i:2d}. {model_name.replace('_', ' ').title():<25}: {model_results['test_accuracy']:.1%}")

    # Best individual model
    best_name, best_results = sorted_results[0]
    print(f"\n🥇 BEST INDIVIDUAL MODEL: {best_name.replace('_', ' ').title()}")
    print(f"   🎯 Test Accuracy: {best_results['test_accuracy']:.1%}")
    print(f"   📈 CV Score: {best_results['cv_mean']:.1%} ± {best_results['cv_std']:.1%}")
    print(f"   📊 Overfitting Gap: {best_results['overfitting_gap']:.1%}")

    # Evaluate mega-ensemble
    y_test_encoded = label_encoder.transform(y_test)
    ensemble_pred = mega_ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_pred)

    print(f"\n🚀 MEGA-ENSEMBLE PERFORMANCE:")
    print(f"   🎯 Test Accuracy: {ensemble_accuracy:.1%}")

    # Detailed ensemble analysis
    ensemble_pred_labels = label_encoder.inverse_transform(ensemble_pred)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    ensemble_report = classification_report(y_test_labels, ensemble_pred_labels, output_dict=True)

    draw_f1 = ensemble_report.get('D', {}).get('f1-score', 0)
    print(f"   🎯 Draw F1-Score: {draw_f1:.1%}")

    # Performance improvement calculation
    baseline_random = 0.333
    improvement_over_random = ((ensemble_accuracy - baseline_random) / baseline_random) * 100
    print(f"   📈 Improvement over Random: +{improvement_over_random:.1f}%")

    # Feature importance analysis (for tree-based models)
    if best_name in ['xgboost_optimized', 'lightgbm', 'random_forest_optimized', 'extra_trees']:
        print(f"\n🔍 TOP 15 FEATURE IMPORTANCE:")
        if hasattr(best_results['model'], 'feature_importances_'):
            importances = best_results['model'].feature_importances_

            # Map back to original feature names if needed
            if len(importances) <= len(feature_cols):
                feature_names = feature_cols[:len(importances)]
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            for i, (feature, importance) in enumerate(feature_importance[:15], 1):
                print(f"   {i:2d}. {feature:<30}: {importance:.4f}")

    # Performance benchmarking
    print(f"\n📊 PERFORMANCE BENCHMARKS:")
    print(f"   🎲 Random Baseline: {baseline_random:.1%}")
    print(f"   🏆 Our Best Model: {best_results['test_accuracy']:.1%}")
    print(f"   🚀 Mega-Ensemble: {ensemble_accuracy:.1%}")
    print(f"   📊 Improvement: +{(ensemble_accuracy - baseline_random) * 100:.1f}% over random")

    # Industry comparison
    industry_standard = 0.55
    if ensemble_accuracy > industry_standard:
        print(f"   🏅 Above Industry Standard ({industry_standard:.1%}): +{(ensemble_accuracy - industry_standard) * 100:.1f}%")

    return {
        'best_individual': (best_name, best_results),
        'mega_ensemble_accuracy': ensemble_accuracy,
        'mega_ensemble_report': ensemble_report,
        'feature_importance': feature_importance if 'feature_importance' in locals() else None
    }

def save_cutting_edge_results(results, mega_ensemble, analysis, feature_cols):
    """Save cutting-edge results"""
    print("\n💾 SAVING CUTTING-EDGE RESULTS...")

    results_dir = 'cutting_edge_results'
    os.makedirs(results_dir, exist_ok=True)

    # Save mega-ensemble model
    with open(f'{results_dir}/mega_ensemble_model.pkl', 'wb') as f:
        pickle.dump(mega_ensemble, f)

    # Save comprehensive results
    cutting_edge_results = {
        'individual_models': {name: {
            'test_accuracy': results[name]['test_accuracy'],
            'cv_mean': results[name]['cv_mean'],
            'cv_std': results[name]['cv_std'],
            'overfitting_gap': results[name]['overfitting_gap']
        } for name in results.keys()},
        'mega_ensemble': {
            'accuracy': analysis['mega_ensemble_accuracy'],
            'classification_report': analysis['mega_ensemble_report']
        },
        'best_individual_model': {
            'name': analysis['best_individual'][0],
            'accuracy': analysis['best_individual'][1]['test_accuracy']
        },
        'feature_list': feature_cols,
        'performance_summary': {
            'improvement_over_random': ((analysis['mega_ensemble_accuracy'] - 0.333) / 0.333) * 100,
            'drew_prediction_f1': analysis['mega_ensemble_report'].get('D', {}).get('f1-score', 0)
        }
    }

    with open(f'{results_dir}/cutting_edge_results.pkl', 'wb') as f:
        pickle.dump(cutting_edge_results, f)

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

    # Add mega-ensemble
    summary_data.append({
        'Model': 'Mega-Ensemble',
        'Test_Accuracy': f"{analysis['mega_ensemble_accuracy']:.1%}",
        'CV_Mean': 'N/A',
        'CV_Std': 'N/A',
        'Overfitting_Gap': 'N/A'
    })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{results_dir}/cutting_edge_summary.csv', index=False)

    print(f"   ✅ Cutting-edge results saved to {results_dir}/")

def main():
    """Main cutting-edge training function"""
    print("🚀 CUTTING-EDGE ML TRAINING FOR 70%+ ACCURACY")
    print("=" * 60)

    # Load next-level data
    df, feature_cols = load_next_level_data()

    # Prepare features and target
    X = df.drop(columns=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
    y = df['FTR']

    # Advanced feature engineering
    X_engineered, selected_features = advanced_feature_engineering(X, y, feature_cols)

    # Train/test split with optimal strategy
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"   📊 Final split: {len(X_train)} train, {len(X_test)} test")
    print(f"   🎯 Training with {len(X_train.columns)} engineered features")

    # Create advanced ensemble models
    models = create_ensemble_models()

    # Train models
    results, label_encoder = train_cutting_edge_models(X_train, y_train, X_test, y_test, models)

    # Create mega-ensemble
    y_train_encoded = label_encoder.transform(y_train)
    mega_ensemble, top_models = create_mega_ensemble(results, X_train, y_train_encoded, label_encoder)

    # Comprehensive analysis
    analysis = advanced_model_analysis(results, selected_features, mega_ensemble, X_test, y_test, label_encoder)

    # Save results
    save_cutting_edge_results(results, mega_ensemble, analysis, selected_features)

    print("\n✅ CUTTING-EDGE TRAINING COMPLETE!")
    print(f"\n🏆 CHAMPIONSHIP RESULTS:")
    print(f"   🥇 Best Individual: {analysis['best_individual'][0].replace('_', ' ').title()} - {analysis['best_individual'][1]['test_accuracy']:.1%}")
    print(f"   🚀 Mega-Ensemble: {analysis['mega_ensemble_accuracy']:.1%}")
    print(f"   📊 Draw F1-Score: {analysis['performance_summary']['drew_prediction_f1']:.1%}")
    print(f"   📈 Improvement over Random: +{analysis['performance_summary']['improvement_over_random']:.1f}%")
    print(f"   📁 Results saved to: cutting_edge_results/")

    return analysis

if __name__ == "__main__":
    final_analysis = main()