#!/usr/bin/env python3
"""
FINAL CORRECT ML TRAINING - LEAKAGE-FREE FEATURES
Train models on properly engineered features without data leakage
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
    """Load baseline and leakage-free enhanced datasets"""
    print("📊 Loading datasets...")

    # Load baseline data
    baseline_df = pd.read_csv('data_processed/all_seasons.csv')
    print(f"   ✅ Baseline: {len(baseline_df)} matches")

    # Load leakage-free enhanced data
    try:
        enhanced_df = pd.read_csv('data_processed/leakage_free_enhanced_epl_ml.csv')
        print(f"   ✅ Leakage-Free Enhanced: {len(enhanced_df)} matches")
    except FileNotFoundError:
        print("   ❌ Leakage-free enhanced data not found")
        return baseline_df, None

    return baseline_df, enhanced_df

def prepare_baseline_data(df):
    """Prepare baseline dataset for ML"""
    print("   📋 Preparing baseline dataset...")

    # Remove data leakage features
    leakage_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'Referee', 'season']
    df_clean = df.drop(columns=[col for col in leakage_features if col in df.columns])

    # Select basic features
    basic_features = [
        'HS', 'AS', 'HST', 'AST',  # Shot statistics
        'HF', 'AF', 'HC', 'AC',    # Fouls and corners
        'HY', 'AY', 'HR', 'AR'     # Cards
    ]

    X = df_clean[basic_features].copy()
    y = df_clean['FTR'].copy()

    # Handle missing values
    X = X.fillna(X.median())

    print(f"      ✅ Baseline features: {len(basic_features)}")
    return X, y, basic_features

def prepare_enhanced_data(df):
    """Prepare enhanced dataset for ML"""
    print("   🚀 Preparing leakage-free enhanced dataset...")

    # Remove basic match info and target
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['FTR'].copy()

    # Handle missing values
    X = X.fillna(X.median())

    print(f"      ✅ Enhanced features: {len(feature_cols)}")
    return X, y, feature_cols

def split_data_temporal(X, y, test_size=0.2):
    """Split data temporally to prevent data leakage"""
    print("      ✂️ Temporal split (chronological)...")

    # Convert to arrays if needed
    if hasattr(X, 'index'):
        indices = X.index.tolist()
    else:
        indices = list(range(len(X)))

    # Create temporal split (80% train, 20% test)
    split_point = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    if hasattr(X, 'iloc'):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    if hasattr(y, 'iloc'):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]

    print(f"         📊 Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def train_models_realistic(X_train, y_train, X_test, y_test, dataset_name):
    """Train models with realistic evaluation"""
    print(f"   🤖 Training models on {dataset_name}...")

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define models with conservative parameters to prevent overfitting
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, max_iter=1000,
            class_weight='balanced', C=0.5  # Regularization
        ),
        'random_forest': RandomForestClassifier(
            random_state=42, n_estimators=100,
            max_depth=6, min_samples_split=10,  # Conservative
            class_weight='balanced', max_features='sqrt'
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=100,
            max_depth=4, learning_rate=0.05,  # Conservative
            subsample=0.8
        ),
        'xgboost': xgb.XGBClassifier(
            random_state=42, n_estimators=100,
            max_depth=4, learning_rate=0.05,  # Conservative
            use_label_encoder=False, eval_metric='mlogloss',
            subsample=0.8, colsample_bytree=0.8
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=42, max_depth=5,  # Very conservative
            min_samples_split=20, class_weight='balanced'
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=15, weights='uniform'  # More neighbors
        )
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

        # Cross-validation (more realistic)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if name in ['logistic_regression', 'knn']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=cv)
        else:
            cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=cv)

        # Detailed classification report
        class_report = classification_report(y_test, test_pred, output_dict=True)

        # Store results
        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'overfitting_gap': train_acc - test_acc,
            'test_predictions': test_pred,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'scaler': scaler if name in ['logistic_regression', 'knn'] else None,
            'label_encoder': label_encoder
        }

        print(f"         ✅ Test Accuracy: {test_acc:.1%} (CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%})")
        print(f"         📊 Overfitting Gap: {(train_acc - test_acc):.1%}")

    return results

def scale_features(X_train, X_test):
    """Scale features for ML models"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def analyze_results(baseline_results, enhanced_results):
    """Analyze and compare results"""
    print("\n📊 REALISTIC RESULTS ANALYSIS")
    print("=" * 60)

    comparison_data = []

    for model_name in baseline_results.keys():
        baseline_acc = baseline_results[model_name]['test_accuracy']
        enhanced_acc = enhanced_results[model_name]['test_accuracy'] if enhanced_results else 0
        improvement = enhanced_acc - baseline_acc if enhanced_results else 0

        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Baseline': f"{baseline_acc:.1%}",
            'Enhanced': f"{enhanced_acc:.1%}" if enhanced_results else "N/A",
            'Improvement': f"{improvement:+.1%}" if enhanced_results else "N/A",
            'Baseline_CV': f"{baseline_results[model_name]['cv_mean']:.1%}",
            'Enhanced_CV': f"{enhanced_results[model_name]['cv_mean']:.1%}" if enhanced_results else "N/A"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    return comparison_df

def identify_best_model(results, dataset_name, feature_list=None):
    """Identify the best performing model"""
    print(f"\n🏆 BEST MODEL ANALYSIS - {dataset_name.upper()}")
    print("-" * 50)

    best_score = 0
    best_model = None
    best_name = None

    for name, model_results in results.items():
        # Balance test accuracy with CV stability
        combined_score = model_results['test_accuracy'] - (model_results['cv_std'] * 0.5)

        if combined_score > best_score:
            best_score = combined_score
            best_model = model_results
            best_name = name

    print(f"🥇 Best Model: {best_name.replace('_', ' ').title()}")
    print(f"🎯 Test Accuracy: {best_model['test_accuracy']:.1%}")
    print(f"📈 CV Score: {best_model['cv_mean']:.1%} ± {best_model['cv_std']:.1%}")
    print(f"📊 Overfitting Gap: {best_model['overfitting_gap']:.1%}")

    # Draw prediction analysis
    draw_report = best_model['classification_report'].get('D', {})
    if draw_report:
        print(f"\n🎯 Draw Prediction Performance:")
        print(f"   📊 Precision: {draw_report.get('precision', 0):.1%}")
        print(f"   📊 Recall: {draw_report.get('recall', 0):.1%}")
        print(f"   📊 F1-Score: {draw_report.get('f1-score', 0):.1%}")

    # Feature importance (if available)
    if hasattr(best_model['model'], 'feature_importances_') and feature_list:
        print(f"\n🔍 TOP 10 FEATURES IMPORTANCE:")
        importances = best_model['model'].feature_importances_
        feature_importance = list(zip(feature_list, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"   {i:2d}. {feature:<25}: {importance:.4f}")

    return best_model, best_name

def create_realistic_visualizations(baseline_results, enhanced_results, comparison_df):
    """Create realistic performance visualizations"""
    print("\n📈 CREATING REALISTIC VISUALIZATIONS...")

    plt.figure(figsize=(15, 10))

    # 1. Realistic Accuracy Comparison
    plt.subplot(2, 3, 1)
    models = list(baseline_results.keys())
    baseline_accs = [baseline_results[m]['test_accuracy'] for m in models]
    enhanced_accs = [enhanced_results[m]['test_accuracy'] for m in models] if enhanced_results else baseline_accs

    x_pos = np.arange(len(models))
    width = 0.35

    plt.bar(x_pos - width/2, baseline_accs, width, label='Baseline', alpha=0.8, color='lightblue')
    if enhanced_results:
        plt.bar(x_pos + width/2, enhanced_accs, width, label='Enhanced', alpha=0.8, color='lightgreen')

    # Add realistic accuracy reference line
    plt.axhline(y=0.55, color='red', linestyle='--', alpha=0.5, label='Good Baseline (55%)')
    plt.axhline(y=0.60, color='orange', linestyle='--', alpha=0.5, label='Excellent (60%)')

    plt.xlabel('Models')
    plt.ylabel('Test Accuracy')
    plt.title('Realistic Model Accuracy Comparison')
    plt.xticks(x_pos, [m.replace('_', ' ').title() for m in models], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Overfitting Analysis
    plt.subplot(2, 3, 2)
    overfitting_gaps = [baseline_results[m]['overfitting_gap'] for m in models]
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' for gap in overfitting_gaps]

    plt.bar(range(len(models)), overfitting_gaps, color=colors, alpha=0.7)
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High Overfitting')
    plt.xlabel('Models')
    plt.ylabel('Overfitting Gap')
    plt.title('Model Overfitting Analysis')
    plt.xticks(range(len(models)), [m.replace('_', ' ').title() for m in models], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Model Stability
    plt.subplot(2, 3, 3)
    cv_means = [baseline_results[m]['cv_mean'] for m in models]
    cv_stds = [baseline_results[m]['cv_std'] for m in models]

    plt.errorbar(range(len(models)), cv_means, yerr=cv_stds,
                fmt='o', capsize=5, capthick=2, markersize=8)
    plt.xlabel('Models')
    plt.ylabel('CV Accuracy')
    plt.title('Model Stability (CV Scores)')
    plt.xticks(range(len(models)), [m.replace('_', ' ').title() for m in models], rotation=45)
    plt.grid(True, alpha=0.3)

    # 4. Performance Distribution
    plt.subplot(2, 3, 4)
    all_accuracies = baseline_accs + (enhanced_accs if enhanced_results else [])
    plt.hist(all_accuracies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(all_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(all_accuracies):.1%}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Model Performance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Feature Engineering Impact
    plt.subplot(2, 3, 5)
    if enhanced_results:
        improvements = [enhanced_results[m]['test_accuracy'] - baseline_results[m]['test_accuracy']
                       for m in models]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        plt.bar(range(len(models)), improvements, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='2% improvement')
        plt.xlabel('Models')
        plt.ylabel('Accuracy Improvement')
        plt.title('Feature Engineering Impact')
        plt.xticks(range(len(models)), [m.replace('_', ' ').title() for m in models], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 6. Summary
    plt.subplot(2, 3, 6)
    plt.axis('off')

    best_baseline = max(baseline_accs)
    best_enhanced = max(enhanced_accs) if enhanced_results else best_baseline
    avg_improvement = np.mean([enhanced_results[m]['test_accuracy'] - baseline_results[m]['test_accuracy']
                              for m in models]) if enhanced_results else 0
    stable_models = sum(1 for std in cv_stds if std < 0.02)

    summary_text = f"""
    📊 REALISTIC PERFORMANCE SUMMARY

    Best Accuracy: {max(best_baseline, best_enhanced):.1%}
    Average Improvement: {avg_improvement:+.1%}
    Stable Models: {stable_models}/{len(models)}

    📈 Accuracy Range: {min(all_accuracies):.1%} - {max(all_accuracies):.1%}
    📊 Mean Performance: {np.mean(all_accuracies):.1%}
    🎯 Good Models (>55%): {sum(1 for acc in all_accuracies if acc > 0.55)}/{len(all_accuracies)}
    """

    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('final_ml_results/realistic_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Realistic visualizations saved")

def save_final_results(baseline_results, enhanced_results, comparison_df, best_analysis):
    """Save final realistic results"""
    print("\n💾 SAVING FINAL REALISTIC RESULTS...")

    results_dir = 'final_ml_results'
    os.makedirs(results_dir, exist_ok=True)

    # Save comprehensive results
    final_results = {
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'comparison': comparison_df.to_dict(),
        'best_model': {
            'name': best_analysis[1],
            'accuracy': best_analysis[0]['test_accuracy'],
            'cv_score': best_analysis[0]['cv_mean'],
            'overfitting_gap': best_analysis[0]['overfitting_gap'],
            'classification_report': best_analysis[0]['classification_report']
        }
    }

    with open(f'{results_dir}/final_realistic_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)

    # Save comparison CSV
    comparison_df.to_csv(f'{results_dir}/realistic_model_comparison.csv', index=False)

    # Save best model
    best_model = best_analysis[0]['model']
    with open(f'{results_dir}/best_realistic_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print(f"   ✅ Realistic results saved to {results_dir}/")

def main():
    """Main final training function"""
    print("🚀 FINAL REALISTIC ML TRAINING")
    print("=" * 60)

    # Load datasets
    baseline_df, enhanced_df = load_datasets()

    # Prepare datasets
    print("\n📋 PREPARING DATASETS")
    print("-" * 30)
    X_base, y_base, baseline_features = prepare_baseline_data(baseline_df)
    X_enh, y_enh, enhanced_features = prepare_enhanced_data(enhanced_df) if enhanced_df is not None else (None, None, None)

    # Split data temporally
    print("\n✂️ TEMPORAL DATA SPLITTING")
    print("-" * 30)
    X_base_train, X_base_test, y_base_train, y_base_test = split_data_temporal(X_base, y_base)
    if enhanced_df is not None:
        X_enh_train, X_enh_test, y_enh_train, y_enh_test = split_data_temporal(X_enh, y_enh)

    # Train models
    print("\n🤖 TRAINING MODELS REALISTICALLY")
    print("-" * 30)
    baseline_results = train_models_realistic(X_base_train, y_base_train, X_base_test, y_base_test, "Baseline")
    enhanced_results = train_models_realistic(X_enh_train, y_enh_train, X_enh_test, y_enh_test, "Enhanced") if enhanced_df is not None else None

    # Analyze results
    comparison_df = analyze_results(baseline_results, enhanced_results)

    # Identify best models
    best_baseline = identify_best_model(baseline_results, "Baseline", baseline_features)
    best_enhanced = identify_best_model(enhanced_results, "Enhanced", enhanced_features) if enhanced_results else None

    # Create visualizations
    create_realistic_visualizations(baseline_results, enhanced_results, comparison_df)

    # Save results
    save_final_results(baseline_results, enhanced_results, comparison_df, best_baseline)

    print("\n✅ FINAL REALISTIC ML TRAINING COMPLETE!")
    print(f"\n🏆 REALISTIC RESULTS:")
    print(f"   🥇 Best Model: {best_baseline[1].replace('_', ' ').title()} - {best_baseline[0]['test_accuracy']:.1%}")
    if enhanced_results:
        print(f"   🚀 Enhanced Best: {best_enhanced[1].replace('_', ' ').title()} - {best_enhanced[0]['test_accuracy']:.1%}")
        improvement = best_enhanced[0]['test_accuracy'] - best_baseline[0]['test_accuracy']
        print(f"   📈 Feature Engineering Improvement: {improvement:+.1%}")
    print(f"   📁 Results saved to: final_ml_results/")

if __name__ == "__main__":
    main()