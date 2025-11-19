#!/usr/bin/env python3
"""
FIXED COMPREHENSIVE ML TRAINING PIPELINE
Trains multiple algorithms on leakage-free baseline and enhanced datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load both leakage-free baseline and enhanced datasets"""
    print("📊 LOADING DATASETS FOR COMPREHENSIVE TRAINING")
    print("=" * 60)

    datasets = {}

    # Load leakage-free baseline data
    print("📋 Loading leakage-free baseline dataset...")
    datasets['baseline'] = {
        'X_train': pd.read_csv('baseline_data_fixed/X_train.csv'),
        'X_test': pd.read_csv('baseline_data_fixed/X_test.csv'),
        'y_train': pd.read_csv('baseline_data_fixed/y_train.csv').iloc[:, 0],
        'y_test': pd.read_csv('baseline_data_fixed/y_test.csv').iloc[:, 0]
    }
    print(f"   Baseline - Train: {datasets['baseline']['X_train'].shape}, Test: {datasets['baseline']['X_test'].shape}")

    # Load enhanced data
    print("📋 Loading enhanced dataset...")
    datasets['enhanced'] = {
        'X_train': pd.read_csv('enhanced_data/X_train.csv'),
        'X_test': pd.read_csv('enhanced_data/X_test.csv'),
        'y_train': pd.read_csv('enhanced_data/y_train.csv').iloc[:, 0],
        'y_test': pd.read_csv('enhanced_data/y_test.csv').iloc[:, 0]
    }
    print(f"   Enhanced - Train: {datasets['enhanced']['X_train'].shape}, Test: {datasets['enhanced']['X_test'].shape}")

    # Load feature columns and summaries
    with open('baseline_data_fixed/data_summary.pkl', 'rb') as f:
        datasets['baseline']['summary'] = pickle.load(f)

    with open('enhanced_data/data_summary.pkl', 'rb') as f:
        datasets['enhanced']['summary'] = pickle.load(f)

    return datasets

def initialize_models():
    """Initialize all models including XGBoost with class weighting"""
    print(f"\n🤖 INITIALIZING MODELS")
    print("=" * 50)

    models = {}

    # Calculate class weights for imbalanced dataset (draws are harder to predict)
    def get_class_weights(y):
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    print("📊 Calculating class weights for imbalanced data...")

    # Logistic Regression with class weights
    models['logistic_regression'] = {
        'model': LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced'
        ),
        'name': 'Logistic Regression',
        'params': {'balanced_class_weights': True}
    }

    # Decision Tree with depth limit to prevent overfitting
    models['decision_tree'] = {
        'model': DecisionTreeClassifier(
            random_state=42,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'
        ),
        'name': 'Decision Tree',
        'params': {'max_depth': 8, 'balanced_class_weights': True}
    }

    # Random Forest with proper regularization
    models['random_forest'] = {
        'model': RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'
        ),
        'name': 'Random Forest',
        'params': {'n_estimators': 100, 'max_depth': 10, 'balanced_class_weights': True}
    }

    # KNN with optimized parameters
    models['knn'] = {
        'model': KNeighborsClassifier(
            n_neighbors=15,
            weights='distance'
        ),
        'name': 'K-Nearest Neighbors',
        'params': {'n_neighbors': 15}
    }

    # XGBoost with class weighting
    models['xgboost'] = {
        'model': xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'name': 'XGBoost',
        'params': {'n_estimators': 100, 'max_depth': 6}
    }

    print(f"✅ Initialized {len(models)} models:")
    for model_key, model_info in models.items():
        print(f"   - {model_info['name']}")

    return models

def train_and_evaluate_models(models, datasets):
    """Train and evaluate all models on both datasets"""
    print(f"\n🚀 TRAINING AND EVALUATION")
    print("=" * 50)

    results = {}

    for dataset_type, data in datasets.items():
        print(f"\n📊 TRAINING ON {dataset_type.upper()} DATASET")
        print("-" * 40)

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        dataset_results = {}

        # Calculate class weights for XGBoost manually
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)

        for model_key, model_info in models.items():
            print(f"\n🤖 Training {model_info['name']}...")

            model = model_info['model']

            # Special handling for XGBoost class weights
            if model_key == 'xgboost':
                # Create sample weights for XGBoost
                sample_weights = np.array([class_weights[i] for i in y_train])
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)

            # Training predictions
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # Test predictions
            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Detailed metrics
            precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)

            # Classification report
            class_report = classification_report(y_test, y_test_pred,
                                                target_names=['A (Away)', 'D (Draw)', 'H (Home)'],
                                                output_dict=True, zero_division=0)

            # Overfitting analysis
            overfitting_gap = train_accuracy - test_accuracy

            # Store results
            dataset_results[model_key] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'confusion_matrix': cm,
                'classification_report': class_report,
                'overfitting_gap': overfitting_gap,
                'predictions': y_test_pred,
                'feature_importance': None
            }

            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                dataset_results[model_key]['feature_importance'] = feature_importance

            print(f"   Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
            print(f"   F1 Score: {f1:.4f} | CV: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print(f"   Overfitting: {overfitting_gap:.4f}")

        results[dataset_type] = dataset_results

    return results

def analyze_results(results, datasets):
    """Comprehensive analysis of results"""
    print(f"\n📈 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 60)

    analysis = {}

    for dataset_type, dataset_results in results.items():
        print(f"\n📊 {dataset_type.upper()} DATASET ANALYSIS")
        print("-" * 40)

        # Create comparison table
        print(f"{'Model':<20} {'Train':<8} {'Test':<8} {'F1':<8} {'CV':<8} {'Overfit':<10} {'Status'}")
        print("-" * 80)

        best_model = None
        best_test_acc = 0

        for model_key, model_results in dataset_results.items():
            train_acc = model_results['train_accuracy']
            test_acc = model_results['test_accuracy']
            f1 = model_results['f1_score']
            cv = model_results['cv_mean']
            overfit = model_results['overfitting_gap']

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = model_key

            # Status determination
            if overfit < 0.05:
                status = "✅ GOOD"
            elif overfit < 0.15:
                status = "⚠️ OK"
            else:
                status = "❌ BAD"

            model_name = model_results['model'].__class__.__name__
            print(f"{model_name:<20} {train_acc:<8.4f} {test_acc:<8.4f} {f1:<8.4f} {cv:<8.4f} {overfit:<10.4f} {status}")

        print(f"\n🏆 Best Model: {best_model} (Test Acc: {best_test_acc:.4f})")

        # Detailed class performance for best model
        best_results = dataset_results[best_model]
        class_report = best_results['classification_report']
        cm = best_results['confusion_matrix']

        print(f"\n📋 Best Model Class Performance:")
        for class_name in ['A (Away)', 'D (Draw)', 'H (Home)']:
            if class_name in class_report:
                class_metrics = class_report[class_name]
                print(f"   {class_name:12s}: Precision={class_metrics['precision']:.3f}, "
                      f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")

        # Store analysis
        analysis[dataset_type] = {
            'best_model': best_model,
            'best_test_accuracy': best_test_acc,
            'model_count': len(dataset_results),
            'best_f1_score': best_results['f1_score'],
            'best_cv_score': best_results['cv_mean'],
            'best_overfitting_gap': best_results['overfitting_gap']
        }

    # Dataset comparison
    print(f"\n🔍 BASELINE vs ENHANCED COMPARISON")
    print("-" * 40)

    baseline_best = analysis['baseline']['best_test_accuracy']
    enhanced_best = analysis['enhanced']['best_test_accuracy']
    improvement = enhanced_best - baseline_best
    improvement_pct = (improvement / baseline_best) * 100 if baseline_best > 0 else 0

    print(f"Baseline Best Accuracy:  {baseline_best:.4f}")
    print(f"Enhanced Best Accuracy:  {enhanced_best:.4f}")
    print(f"Improvement:            {improvement:+.4f} ({improvement_pct:+.1f}%)")

    if improvement > 0.01:  # 1% improvement threshold
        print(f"✅ Enhanced features significantly improved performance!")
    elif improvement > 0:
        print(f"✅ Enhanced features slightly improved performance")
    else:
        print(f"⚠️ Enhanced features did not improve performance")

    # Random baseline comparison
    random_baseline = 1/3  # For 3 classes
    baseline_vs_random = baseline_best - random_baseline
    enhanced_vs_random = enhanced_best - random_baseline

    print(f"\n🎯 Performance vs Random Baseline (33.3%):")
    print(f"   Baseline: {baseline_vs_random:+.4f} ({(baseline_vs_random/random_baseline)*100:+.1f}%)")
    print(f"   Enhanced: {enhanced_vs_random:+.4f} ({(enhanced_vs_random/random_baseline)*100:+.1f}%)")

    return analysis

def save_results(results, analysis):
    """Save all results for later analysis"""
    print(f"\n💾 SAVING COMPREHENSIVE RESULTS")
    print("=" * 50)

    # Create results directory
    os.makedirs('comprehensive_results_fixed', exist_ok=True)

    # Save detailed results
    with open('comprehensive_results_fixed/training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save analysis
    with open('comprehensive_results_fixed/analysis_summary.pkl', 'wb') as f:
        pickle.dump(analysis, f)

    # Create summary report
    summary_report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_results': {},
        'enhanced_results': {},
        'comparison': analysis,
        'data_leakage_fixed': True
    }

    for model_key, model_results in results['baseline'].items():
        summary_report['baseline_results'][model_key] = {
            'test_accuracy': model_results['test_accuracy'],
            'f1_score': model_results['f1_score'],
            'cv_mean': model_results['cv_mean'],
            'overfitting_gap': model_results['overfitting_gap']
        }

    for model_key, model_results in results['enhanced'].items():
        summary_report['enhanced_results'][model_key] = {
            'test_accuracy': model_results['test_accuracy'],
            'f1_score': model_results['f1_score'],
            'cv_mean': model_results['cv_mean'],
            'overfitting_gap': model_results['overfitting_gap']
        }

    with open('comprehensive_results_fixed/summary_report.pkl', 'wb') as f:
        pickle.dump(summary_report, f)

    print(f"✅ Results saved to 'comprehensive_results_fixed/' directory:")
    print(f"   - training_results.pkl")
    print(f"   - analysis_summary.pkl")
    print(f"   - summary_report.pkl")

def main():
    """Main comprehensive training function"""
    print("🚀 FIXED COMPREHENSIVE ML TRAINING PIPELINE")
    print("Training multiple algorithms on leakage-free baseline and enhanced datasets\n")

    # Load datasets
    datasets = load_datasets()

    # Initialize models
    models = initialize_models()

    # Train and evaluate
    results = train_and_evaluate_models(models, datasets)

    # Analyze results
    analysis = analyze_results(results, datasets)

    # Save results
    save_results(results, analysis)

    print(f"\n🎉 FIXED COMPREHENSIVE TRAINING COMPLETE!")
    print(f"\n📊 KEY FINDINGS:")
    print(f"   ✅ Trained {len(models)} models on 2 datasets")
    print(f"   ✅ Data leakage fixed (no goal-based features)")
    print(f"   ✅ Best baseline model: {analysis['baseline']['best_model']} ({analysis['baseline']['best_test_accuracy']:.4f})")
    print(f"   ✅ Best enhanced model: {analysis['enhanced']['best_model']} ({analysis['enhanced']['best_test_accuracy']:.4f})")

    # Calculate improvement
    improvement = analysis['enhanced']['best_test_accuracy'] - analysis['baseline']['best_test_accuracy']
    print(f"   ✅ Feature engineering impact: {improvement:+.4f} ({(improvement/analysis['baseline']['best_test_accuracy']*100):+.1f}%)")

    # Compare to random baseline
    random_baseline = 1/3
    baseline_improvement = analysis['baseline']['best_test_accuracy'] - random_baseline
    enhanced_improvement = analysis['enhanced']['best_test_accuracy'] - random_baseline

    print(f"   ✅ Baseline vs random: +{baseline_improvement:.4f} ({(baseline_improvement/random_baseline)*100:.1f}%)")
    print(f"   ✅ Enhanced vs random: +{enhanced_improvement:.4f} ({(enhanced_improvement/random_baseline)*100:.1f}%)")

    print(f"\n🚀 READY FOR SHAP ANALYSIS & ABLATION STUDIES!")

if __name__ == "__main__":
    main()