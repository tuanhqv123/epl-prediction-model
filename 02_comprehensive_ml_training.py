#!/usr/bin/env python3
"""
COMPREHENSIVE ML TRAINING - BASELINE vs ENHANCED
Compare baseline data with enhanced engineered features for EPL prediction
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load both baseline and enhanced datasets"""
    print("📊 Loading datasets...")

    # Load baseline data
    baseline_df = pd.read_csv('data_processed/all_seasons.csv')
    print(f"   ✅ Baseline: {len(baseline_df)} matches")

    # Load enhanced data
    try:
        enhanced_df = pd.read_csv('data_processed/enhanced_epl_ml.csv')
        print(f"   ✅ Enhanced: {len(enhanced_df)} matches")
    except FileNotFoundError:
        print("   ❌ Enhanced data not found, using baseline only")
        enhanced_df = baseline_df.copy()

    return baseline_df, enhanced_df

def prepare_baseline_data(df):
    """Prepare baseline dataset for ML"""
    print("   📋 Preparing baseline dataset...")

    # Remove data leakage features (goals and halftime info)
    leakage_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR']
    df_clean = df.drop(columns=leakage_features)

    # Select basic features for baseline
    basic_features = [
        'HS', 'AS', 'HST', 'AST',  # Shot statistics
        'HF', 'AF', 'HC', 'AC',    # Fouls and corners
        'HY', 'AY', 'HR', 'AR'     # Cards
    ]

    # Make sure all features exist
    available_features = [f for f in basic_features if f in df_clean.columns]
    X = df_clean[available_features].copy()
    y = df_clean['FTR'].copy()

    # Handle missing values
    X = X.fillna(X.median())

    print(f"      ✅ Baseline features: {len(available_features)}")

    return X, y, available_features

def prepare_enhanced_data(df):
    """Prepare enhanced dataset for ML"""
    print("   🚀 Preparing enhanced dataset...")

    # Remove basic match info and target
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Referee', 'season']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Remove any data leakage features
    leakage_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR']
    feature_cols = [col for col in feature_cols if col not in leakage_features]

    X = df[feature_cols].copy()
    y = df['FTR'].copy()

    # Handle missing values
    X = X.fillna(X.median())

    print(f"      ✅ Enhanced features: {len(feature_cols)}")

    return X, y, feature_cols

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data with temporal considerations"""
    # For proper temporal split, we'll sort by date if available
    if 'Date' in X.columns:
        X_sorted = X.sort_values('Date')
        y_sorted = y.loc[X_sorted.index]
        split_point = int(len(X_sorted) * (1 - test_size))
        X_train = X_sorted.iloc[:split_point]
        X_test = X_sorted.iloc[split_point:]
        y_train = y_sorted.iloc[:split_point]
        y_test = y_sorted.iloc[split_point:]
    else:
        # Standard random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    print(f"      📊 Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features for ML models"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train, y_train, X_test, y_test, dataset_name):
    """Train multiple models and evaluate performance"""
    print(f"   🤖 Training models on {dataset_name} data...")

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Encode target labels (A=0, D=1, H=2)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        print(f"      🔄 Training {name.replace('_', ' ').title()}...")

        # Train model
        if name in ['logistic_regression', 'knn']:
            model.fit(X_train_scaled, y_train_encoded)
            train_pred_encoded = model.predict(X_train_scaled)
            test_pred_encoded = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train_encoded)
            train_pred_encoded = model.predict(X_train)
            test_pred_encoded = model.predict(X_test)

        # Convert predictions back to original labels
        train_pred = label_encoder.inverse_transform(train_pred_encoded)
        test_pred = label_encoder.inverse_transform(test_pred_encoded)

        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # Cross-validation
        if name in ['logistic_regression', 'knn']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=5)
        else:
            cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5)

        # Store results
        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_predictions': test_pred,
            'classification_report': classification_report(y_test, test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'scaler': scaler if name in ['logistic_regression', 'knn'] else None
        }

        print(f"         ✅ Test Accuracy: {test_acc:.1%}")

    return results

def compare_results(baseline_results, enhanced_results):
    """Compare baseline vs enhanced results"""
    print("\n📊 BASELINE vs ENHANCED COMPARISON")
    print("=" * 60)

    comparison = []

    for model_name in baseline_results.keys():
        baseline_acc = baseline_results[model_name]['test_accuracy']
        enhanced_acc = enhanced_results[model_name]['test_accuracy']
        improvement = enhanced_acc - baseline_acc

        comparison.append({
            'Model': model_name.replace('_', ' ').title(),
            'Baseline': f"{baseline_acc:.1%}",
            'Enhanced': f"{enhanced_acc:.1%}",
            'Improvement': f"{improvement:+.1%}"
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))

    return comparison_df

def analyze_best_model(baseline_results, enhanced_results, baseline_features, enhanced_features):
    """Analyze the best performing model"""
    print("\n🏆 BEST MODEL ANALYSIS")
    print("=" * 40)

    # Find best model
    best_accuracy = 0
    best_model = None
    best_dataset = None
    best_name = None

    for dataset_name, results in [('Baseline', baseline_results), ('Enhanced', enhanced_results)]:
        for model_name, model_results in results.items():
            if model_results['test_accuracy'] > best_accuracy:
                best_accuracy = model_results['test_accuracy']
                best_model = model_results
                best_dataset = dataset_name
                best_name = model_name

    print(f"🥇 Best Model: {best_name.replace('_', ' ').title()}")
    print(f"📊 Dataset: {best_dataset}")
    print(f"🎯 Accuracy: {best_accuracy:.1%}")
    print(f"📈 CV Score: {best_model['cv_mean']:.1%} ± {best_model['cv_std']:.1%}")

    # Feature importance (if available)
    if hasattr(best_model['model'], 'feature_importances_'):
        print("\n🔍 TOP FEATURES IMPORTANCE:")
        if best_dataset == 'Baseline':
            importances = best_model['model'].feature_importances_
            feature_names = baseline_features
        else:
            importances = best_model['model'].feature_importances_
            feature_names = enhanced_features

        # Sort by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"   {i:2d}. {feature:<25}: {importance:.4f}")

    return best_model, best_dataset, best_name

def save_comprehensive_results(baseline_results, enhanced_results, comparison_df, best_analysis):
    """Save comprehensive training results"""
    print("\n💾 SAVING COMPREHENSIVE RESULTS...")

    results_dir = 'comprehensive_ml_results'
    os.makedirs(results_dir, exist_ok=True)

    # Save all results
    comprehensive_results = {
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'comparison': comparison_df.to_dict(),
        'best_analysis': {
            'model_name': best_analysis[2],
            'dataset': best_analysis[1],
            'accuracy': best_analysis[0]['test_accuracy'],
            'cv_score': best_analysis[0]['cv_mean']
        }
    }

    with open(f'{results_dir}/comprehensive_training_results.pkl', 'wb') as f:
        pickle.dump(comprehensive_results, f)

    # Save comparison CSV
    comparison_df.to_csv(f'{results_dir}/model_comparison.csv', index=False)

    # Save best model
    best_model = best_analysis[0]['model']
    with open(f'{results_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Save best model scaler if exists
    if best_analysis[0]['scaler'] is not None:
        with open(f'{results_dir}/best_model_scaler.pkl', 'wb') as f:
            pickle.dump(best_analysis[0]['scaler'], f)

    print(f"   ✅ Results saved to {results_dir}/")

def create_visualizations(baseline_results, enhanced_results):
    """Create performance comparison visualizations"""
    print("\n📈 CREATING VISUALIZATIONS...")

    # Extract accuracy data
    models = list(baseline_results.keys())
    baseline_accs = [baseline_results[m]['test_accuracy'] for m in models]
    enhanced_accs = [enhanced_results[m]['test_accuracy'] for m in models]

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(models))
    width = 0.35

    plt.bar(x_pos - width/2, baseline_accs, width, label='Baseline', alpha=0.8)
    plt.bar(x_pos + width/2, enhanced_accs, width, label='Enhanced', alpha=0.8)

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Baseline vs Enhanced Accuracy Comparison')
    plt.xticks(x_pos, [m.replace('_', '\n').title() for m in models], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improvement plot
    plt.subplot(2, 2, 2)
    improvements = [enh - base for enh, base in zip(enhanced_accs, baseline_accs)]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    plt.bar(range(len(models)), improvements, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Models')
    plt.ylabel('Accuracy Improvement')
    plt.title('Feature Engineering Impact')
    plt.xticks(range(len(models)), [m.replace('_', '\n').title() for m in models], rotation=45)
    plt.grid(True, alpha=0.3)

    # CV Scores comparison
    plt.subplot(2, 2, 3)
    baseline_cv = [baseline_results[m]['cv_mean'] for m in models]
    enhanced_cv = [enhanced_results[m]['cv_mean'] for m in models]

    plt.bar(x_pos - width/2, baseline_cv, width, label='Baseline', alpha=0.8)
    plt.bar(x_pos + width/2, enhanced_cv, width, label='Enhanced', alpha=0.8)

    plt.xlabel('Models')
    plt.ylabel('CV Accuracy')
    plt.title('Cross-Validation Comparison')
    plt.xticks(x_pos, [m.replace('_', '\n').title() for m in models], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Summary stats
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.9, f"Baseline Best: {max(baseline_accs):.1%}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.1, 0.8, f"Enhanced Best: {max(enhanced_accs):.1%}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.1, 0.7, f"Avg Improvement: {np.mean(improvements):+.1%}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.1, 0.6, f"Max Improvement: {max(improvements):+.1%}", transform=plt.gca().transAxes, fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('comprehensive_ml_results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Visualizations saved")

def main():
    """Main comprehensive training function"""
    print("🚀 COMPREHENSIVE ML TRAINING - BASELINE vs ENHANCED")
    print("=" * 60)

    # Load datasets
    baseline_df, enhanced_df = load_datasets()

    # Prepare datasets
    print("\n📋 PREPARING DATASETS")
    print("-" * 30)
    X_base, y_base, baseline_features = prepare_baseline_data(baseline_df)
    X_enh, y_enh, enhanced_features = prepare_enhanced_data(enhanced_df)

    # Split data
    print("\n✂️ SPLITTING DATA")
    print("-" * 20)
    X_base_train, X_base_test, y_base_train, y_base_test = split_data(X_base, y_base)
    X_enh_train, X_enh_test, y_enh_train, y_enh_test = split_data(X_enh, y_enh)

    # Train models
    print("\n🤖 TRAINING MODELS")
    print("-" * 20)
    baseline_results = train_models(X_base_train, y_base_train, X_base_test, y_base_test, "Baseline")
    enhanced_results = train_models(X_enh_train, y_enh_train, X_enh_test, y_enh_test, "Enhanced")

    # Compare results
    comparison_df = compare_results(baseline_results, enhanced_results)

    # Analyze best model
    best_analysis = analyze_best_model(baseline_results, enhanced_results, baseline_features, enhanced_features)

    # Save results
    save_comprehensive_results(baseline_results, enhanced_results, comparison_df, best_analysis)

    # Create visualizations
    create_visualizations(baseline_results, enhanced_results)

    print("\n✅ COMPREHENSIVE TRAINING COMPLETE!")
    print(f"\n🏆 BEST RESULTS:")
    print(f"   🥇 Model: {best_analysis[2].replace('_', ' ').title()}")
    print(f"   📊 Dataset: {best_analysis[1]}")
    print(f"   🎯 Accuracy: {best_analysis[0]['test_accuracy']:.1%}")
    print(f"   📁 Results saved to: comprehensive_ml_results/")

if __name__ == "__main__":
    main()