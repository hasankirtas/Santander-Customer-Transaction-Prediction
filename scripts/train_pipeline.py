"""
Main training pipeline script.
This script orchestrates the entire training process.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_selector import FeatureSelector
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("train_pipeline")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Starting Santander Customer Transaction Prediction Training Pipeline")
    logger.info("="*60)
    
    # Step 1: Load data
    logger.info("\n[Step 1/6] Loading training data...")
    data_loader = DataLoader()
    train_data = data_loader.load_train_data(use_processed=False)
    
    # Step 2: Feature selection
    logger.info("\n[Step 2/6] Selecting features...")
    feature_selector = FeatureSelector()
    rf_features = feature_selector.get_rf_features()
    xgb_features = feature_selector.get_xgb_features()
    
    # Extract features and target
    X, y = data_loader.get_features_and_target(train_data)
    X_rf = X[rf_features].values
    X_xgb = X[xgb_features].values
    y_values = y.values
    
    # Step 3: Resample data
    logger.info("\n[Step 3/6] Resampling data to handle class imbalance...")
    preprocessor = DataPreprocessor()
    
    # Combine features for resampling (we'll split after)
    X_combined = pd.DataFrame(X, columns=X.columns)
    X_resampled, y_resampled = preprocessor.resample_data(
        X_combined,
        pd.Series(y_values),
        id_column=train_data['ID_code']
    )
    
    # Split resampled data back to RF and XGB features
    X_rf_final = X_resampled[rf_features].values
    X_xgb_final = X_resampled[xgb_features].values
    y_final = y_resampled.values
    
    # Step 4: Cross-validation
    logger.info("\n[Step 4/6] Performing cross-validation...")
    trainer = ModelTrainer()
    cv_results = trainer.cross_validate_models(
        X_rf_final,
        X_xgb_final,
        y_final
    )
    
    # Step 5: Train final models
    logger.info("\n[Step 5/6] Training final models on full dataset...")
    class_weights = trainer.calculate_class_weights(pd.Series(y_final))
    
    # Train Random Forest
    rf_model = trainer.train_random_forest(
        X_rf_final,
        y_final,
        class_weight=class_weights['class_weight']
    )
    
    # Train XGBoost
    xgb_model = trainer.train_xgboost(
        X_xgb_final,
        y_final,
        scale_pos_weight=class_weights['scale_pos_weight']
    )
    
    # Train Voting Classifier
    voting_model = trainer.train_voting_classifier(
        X_rf_final,
        X_xgb_final,
        y_final,
        rf_model=rf_model,
        xgb_model=xgb_model
    )
    
    # Step 6: Evaluate and save
    logger.info("\n[Step 6/6] Evaluating models and saving...")
    evaluator = ModelEvaluator()
    
    # Evaluate on resampled data (for demonstration)
    y_pred_rf = rf_model.predict(X_rf_final)
    y_proba_rf = rf_model.predict_proba(X_rf_final)[:, 1]
    evaluator.print_evaluation_summary(
        y_final,
        y_pred_rf,
        y_proba_rf,
        model_name="Random Forest"
    )
    
    y_pred_xgb = xgb_model.predict(X_xgb_final)
    y_proba_xgb = xgb_model.predict_proba(X_xgb_final)[:, 1]
    evaluator.print_evaluation_summary(
        y_final,
        y_pred_xgb,
        y_proba_xgb,
        model_name="XGBoost"
    )
    
    # Save models
    trainer.save_models()
    
    logger.info("\n" + "="*60)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

