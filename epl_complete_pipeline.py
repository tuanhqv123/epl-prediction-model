#!/usr/bin/env python3
"""
EPL PREDICTION MODEL - COMPLETE END-TO-END PRODUCTION PIPELINE
================================================================

Real-world standards implementation with comprehensive logging and validation:
1. Data Processing & Cleaning
2. Feature Engineering
3. Data Quality Validation
4. Temporal Train/Test Split
5. Model Training with Cross-Validation
6. Comprehensive Evaluation
7. Production Logging & Metrics

🎯 Goal: Production-ready model with full traceability
"""

import pandas as pd
import numpy as np
import logging
import json
import datetime
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('epl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EPLPipelineLogger:
    """Comprehensive pipeline logger for real-world production tracking"""

    def __init__(self):
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_log = {}
        self.pipeline_log = []

    def log_phase(self, phase_name, status, details=None):
        """Log each pipeline phase with details"""
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'phase': phase_name,
            'status': status,
            'details': details
        }
        self.pipeline_log.append(log_entry)

        logger.info(f"PHASE {phase_name}: {status}")
        if details:
            for key, value in details.items():
                logger.info(f"  - {key}: {value}")

    def log_metrics(self, phase, metrics):
        """Log performance metrics"""
        self.metrics_log[phase] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics
        }

        logger.info(f"METRICS {phase}:")
        for key, value in metrics.items():
            logger.info(f"  - {key}: {value}")

    def save_pipeline_report(self, filename='pipeline_report.json'):
        """Save complete pipeline report"""
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        report = {
            'run_id': self.run_id,
            'pipeline_log': convert_types(self.pipeline_log),
            'metrics_log': convert_types(self.metrics_log),
            'completion_time': datetime.datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Pipeline report saved to {filename}")

class EPLDataProcessor:
    """Data processing with comprehensive validation and logging"""

    def __init__(self, logger):
        self.logger = logger
        self.data_quality_report = {}

    def load_and_validate_data(self, csv_path='epl_enhanced_fixed.csv'):
        """Load data with comprehensive quality checks"""
        self.logger.log_phase("DATA_LOADING", "STARTED", {'file_path': csv_path})

        # Load data
        df = pd.read_csv(csv_path)
        original_size = len(df)

        # Basic data validation
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range_start': df['Date'].min(),
            'date_range_end': df['Date'].max(),
            'seasons': sorted(df['season'].unique()),
            'null_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }

        # Check target distribution
        target_dist = df['FTR'].value_counts().to_dict()
        validation_results['target_distribution'] = target_dist

        self.logger.log_phase("DATA_LOADING", "COMPLETED", validation_results)
        self.logger.log_metrics("DATA_QUALITY", validation_results)

        return df

    def process_data(self, df):
        """Process and clean data with logging"""
        self.logger.log_phase("DATA_PROCESSING", "STARTED")

        # Convert dates (try both formats)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Remove only rows with essential null values
        original_size = len(df)
        df = df.dropna(subset=['Date', 'FTR', 'HomeTeam', 'AwayTeam'])
        cleaned_size = len(df)

        processing_stats = {
            'rows_before_cleaning': original_size,
            'rows_after_cleaning': cleaned_size,
            'rows_removed': original_size - cleaned_size,
            'removal_rate': (original_size - cleaned_size) / original_size * 100 if original_size > 0 else 0
        }

        self.logger.log_phase("DATA_PROCESSING", "COMPLETED", processing_stats)

        return df

class EPLFeatureEngineer:
    """Feature engineering with comprehensive tracking"""

    def __init__(self, logger):
        self.logger = logger
        self.feature_definitions = {}

    def calculate_3match_form(self, df):
        """Calculate 3-match form (proven optimal)"""
        self.logger.log_phase("FORM_CALCULATION", "STARTED", {'form_window': 3})

        # Sort by season and date
        df = df.sort_values(['season', 'Date']).reset_index(drop=True)

        # Initialize form features
        df['home_form_3'] = 0.0
        df['away_form_3'] = 0.0
        df['form_diff_3'] = 0.0

        # Process each season separately
        for season in sorted(df['season'].unique()):
            season_mask = df['season'] == season
            season_indices = df[season_mask].index.tolist()

            # Initialize team tracking
            team_form = {}
            all_teams = set(df.loc[season_mask, 'HomeTeam'].unique()) | \
                       set(df.loc[season_mask, 'AwayTeam'].unique())

            for team in all_teams:
                team_form[team] = []

            # Process matches chronologically
            for idx in season_indices:
                row = df.loc[idx]
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']

                # Calculate current form
                home_form_avg = np.mean(team_form[home_team][-3:]) if team_form[home_team] else 0
                away_form_avg = np.mean(team_form[away_team][-3:]) if team_form[away_team] else 0

                df.loc[idx, 'home_form_3'] = home_form_avg
                df.loc[idx, 'away_form_3'] = away_form_avg
                df.loc[idx, 'form_diff_3'] = home_form_avg - away_form_avg

                # Update form with match result
                try:
                    home_goals = int(row['FTHG']) if pd.notna(row['FTHG']) else 0
                    away_goals = int(row['FTAG']) if pd.notna(row['FTAG']) else 0

                    if home_goals > away_goals:
                        team_form[home_team].append(3)
                        team_form[away_team].append(0)
                    elif home_goals < away_goals:
                        team_form[home_team].append(0)
                        team_form[away_team].append(3)
                    else:
                        team_form[home_team].append(1)
                        team_form[away_team].append(1)

                    # Keep only last 3 results
                    team_form[home_team] = team_form[home_team][-3:]
                    team_form[away_team] = team_form[away_team][-3:]

                except (ValueError, KeyError):
                    continue

        form_stats = {
            'home_form_range': [df['home_form_3'].min(), df['home_form_3'].max()],
            'away_form_range': [df['away_form_3'].min(), df['away_form_3'].max()],
            'form_diff_range': [df['form_diff_3'].min(), df['form_diff_3'].max()]
        }

        self.logger.log_phase("FORM_CALCULATION", "COMPLETED", form_stats)
        return df

    def create_engineered_features(self, df):
        """Create engineered features with logging"""
        self.logger.log_phase("FEATURE_ENGINEERING", "STARTED")

        # Shot accuracy
        df['home_shot_accuracy'] = np.where(df['HS'] > 0, df['HST'] / df['HS'], 0)
        df['away_shot_accuracy'] = np.where(df['AS'] > 0, df['AST'] / df['AS'], 0)

        # Rest days (already calculated in enhanced data)
        if 'home_rest_days_fixed' not in df.columns:
            # Simplified rest days calculation
            df = df.sort_values(['season', 'Date'])
            df['home_rest_days'] = 3.0  # Default
            df['away_rest_days'] = 3.0
            df['load_diff'] = df['home_rest_days'] - df['away_rest_days']
        else:
            # Use existing calculations
            df['load_diff'] = df['load_diff_7days_fixed']
            df['home_rest_days'] = df['home_rest_days_fixed']
            df['away_rest_days'] = df['away_rest_days_fixed']

        feature_stats = {
            'shot_accuracy_home': [df['home_shot_accuracy'].mean(), df['home_shot_accuracy'].std()],
            'shot_accuracy_away': [df['away_shot_accuracy'].mean(), df['away_shot_accuracy'].std()],
            'rest_days_home': [df['home_rest_days'].mean(), df['home_rest_days'].std()],
            'rest_days_away': [df['away_rest_days'].mean(), df['away_rest_days'].std()]
        }

        self.logger.log_phase("FEATURE_ENGINEERING", "COMPLETED", feature_stats)
        return df

    def select_optimal_features(self, df):
        """Select optimal feature set (proven through testing)"""
        self.logger.log_phase("FEATURE_SELECTION", "STARTED")

        # Optimal 18 features (proven best)
        feature_cols = [
            # Core match statistics (12)
            'HST', 'AST', 'HS', 'AS', 'HY', 'AY', 'HR', 'AR', 'HC', 'AC', 'HF', 'AF',

            # Engineered features (2)
            'home_shot_accuracy', 'away_shot_accuracy',

            # Form features (3)
            'home_form_3', 'away_form_3', 'form_diff_3',

            # Competition features (3)
            'home_rest_days', 'away_rest_days', 'load_diff'
        ]

        # Verify all features exist
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        selection_stats = {
            'total_features_selected': len(feature_cols),
            'core_stats': 12,
            'engineered': 2,
            'form_features': 3,
            'competition': 3
        }

        self.logger.log_phase("FEATURE_SELECTION", "COMPLETED", selection_stats)

        return feature_cols

class EPLModelTrainer:
    """Model training with comprehensive validation and logging"""

    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def create_temporal_split(self, df, feature_cols):
        """Create temporal train/test split with logging"""
        self.logger.log_phase("TRAIN_TEST_SPLIT", "STARTED")

        # Prepare features and target
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['FTR']

        # Temporal split (real-world standard)
        test_seasons = ['2023-2024', '2024-2025']
        train_df = df[~df['season'].isin(test_seasons)]
        test_df = df[df['season'].isin(test_seasons)]

        X_train = X.loc[train_df.index]
        X_test = X.loc[test_df.index]
        y_train = y.loc[train_df.index]
        y_test = y.loc[test_df.index]

        split_stats = {
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'training_seasons': f"{train_df['season'].min()}-{train_df['season'].max()}",
            'testing_seasons': f"{test_df['season'].min()}-{test_df['season'].max()}",
            'feature_count': len(feature_cols),
            'train_target_dist': y_train.value_counts().to_dict(),
            'test_target_dist': y_test.value_counts().to_dict()
        }

        self.logger.log_phase("TRAIN_TEST_SPLIT", "COMPLETED", split_stats)

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Scale features with logging"""
        self.logger.log_phase("FEATURE_SCALING", "STARTED")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        scaling_stats = {
            'scaler_type': 'StandardScaler',
            'training_mean': X_train.mean().mean(),
            'training_std': X_train.std().mean(),
            'scaled_feature_range': 'Approximately N(0,1)'
        }

        self.logger.log_phase("FEATURE_SCALING", "COMPLETED", scaling_stats)

        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train):
        """Train model with cross-validation and logging"""
        self.logger.log_phase("MODEL_TRAINING", "STARTED")

        # Initialize model (proven optimal configuration)
        self.model = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # Cross-validation for robust evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='accuracy')

        # Train on full training data
        self.model.fit(X_train, y_train)

        training_stats = {
            'model_type': 'ExtraTreesClassifier',
            'n_estimators': 100,
            'max_depth': 6,
            'class_weight': 'balanced',
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
            'cv_scores_list': cv_scores.tolist()
        }

        self.logger.log_phase("MODEL_TRAINING", "COMPLETED", training_stats)
        self.logger.log_metrics("CROSS_VALIDATION", {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'scores': cv_scores.tolist()
        })

        return cv_scores

    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation with logging"""
        self.logger.log_phase("MODEL_EVALUATION", "STARTED")

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        draw_f1 = f1_score(y_test, y_pred, labels=['D'], average='macro')

        # Detailed classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=['H', 'D', 'A'])

        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = float(self.model.feature_importances_[i])

        evaluation_stats = {
            'test_accuracy': accuracy,
            'macro_f1_score': macro_f1,
            'draw_f1_score': draw_f1,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance
        }

        self.logger.log_phase("MODEL_EVALUATION", "COMPLETED", {
            'accuracy': f"{accuracy:.4f}",
            'draw_f1': f"{draw_f1:.4f}",
            'macro_f1': f"{macro_f1:.4f}"
        })

        self.logger.log_metrics("TEST_PERFORMANCE", {
            'accuracy': accuracy,
            'draw_f1': draw_f1,
            'macro_f1': macro_f1
        })

        return evaluation_stats

def main():
    """Complete end-to-end pipeline execution"""
    logger = EPLPipelineLogger()

    logger.log_phase("PIPELINE_START", "INITIATED", {
        'purpose': 'EPL Prediction Model - Complete Pipeline',
        'standards': 'Real-world production standards with full logging'
    })

    try:
        # Phase 1: Data Processing
        processor = EPLDataProcessor(logger)
        df = processor.load_and_validate_data()
        df = processor.process_data(df)

        # Phase 2: Feature Engineering
        engineer = EPLFeatureEngineer(logger)
        df = engineer.calculate_3match_form(df)
        df = engineer.create_engineered_features(df)
        feature_cols = engineer.select_optimal_features(df)

        # Phase 3: Model Training
        trainer = EPLModelTrainer(logger)
        trainer.feature_names = feature_cols

        X_train, X_test, y_train, y_test = trainer.create_temporal_split(df, feature_cols)
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)

        cv_scores = trainer.train_model(X_train_scaled, y_train)
        evaluation_results = trainer.evaluate_model(X_test_scaled, y_test)

        # Phase 4: Final Report
        final_summary = {
            'model_performance': {
                'test_accuracy': evaluation_results['test_accuracy'],
                'draw_f1': evaluation_results['draw_f1_score'],
                'macro_f1': evaluation_results['macro_f1_score'],
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            },
            'dataset_info': {
                'total_matches': len(df),
                'training_matches': len(X_train),
                'testing_matches': len(X_test),
                'feature_count': len(feature_cols),
                'seasons_covered': f"{df['season'].min()}-{df['season'].max()}"
            },
            'model_specification': {
                'algorithm': 'ExtraTreesClassifier',
                'n_estimators': 100,
                'max_depth': 6,
                'feature_count': len(feature_cols)
            }
        }

        logger.log_phase("PIPELINE_COMPLETE", "SUCCESS", final_summary)
        logger.log_metrics("FINAL_RESULTS", final_summary['model_performance'])

        # Save comprehensive report
        logger.save_pipeline_report('epl_pipeline_final_report.json')

        # Save final model specification
        model_spec = {
            'feature_names': feature_cols,
            'model_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'performance_metrics': final_summary['model_performance'],
            'training_date': datetime.datetime.now().isoformat()
        }

        with open('epl_model_specification.json', 'w') as f:
            json.dump(model_spec, f, indent=2)

        print("\n" + "="*80)
        print("🏆 EPL PREDICTION PIPELINE - COMPLETE SUCCESS!")
        print("="*80)
        print(f"\n✅ FINAL PERFORMANCE:")
        print(f"   Test Accuracy: {final_summary['model_performance']['test_accuracy']:.4f} ({final_summary['model_performance']['test_accuracy']:.1%})")
        print(f"   Draw F1 Score: {final_summary['model_performance']['draw_f1']:.4f} ({final_summary['model_performance']['draw_f1']:.1%})")
        print(f"   CV Accuracy: {final_summary['model_performance']['cv_accuracy']:.4f} ± {final_summary['model_performance']['cv_std']:.4f}")

        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total Matches: {final_summary['dataset_info']['total_matches']:,}")
        print(f"   Training: {final_summary['dataset_info']['training_matches']:,}")
        print(f"   Testing: {final_summary['dataset_info']['testing_matches']:,}")
        print(f"   Features: {final_summary['dataset_info']['feature_count']}")
        print(f"   Seasons: {final_summary['dataset_info']['seasons_covered']}")

        print(f"\n📋 FILES GENERATED:")
        print(f"   • epl_pipeline.log - Complete execution log")
        print(f"   • epl_pipeline_final_report.json - Comprehensive pipeline report")
        print(f"   • epl_model_specification.json - Model specification")

        return trainer, evaluation_results, df

    except Exception as e:
        logger.log_phase("PIPELINE_ERROR", "FAILED", {'error': str(e)})
        logger.save_pipeline_report('epl_pipeline_error_report.json')
        raise

if __name__ == "__main__":
    model, results, data = main()