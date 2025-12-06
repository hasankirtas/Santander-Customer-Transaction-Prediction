"""
Prediction pipeline script.
This script loads trained models and generates predictions on test data.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import joblib
from src.data.data_loader import DataLoader
from src.features.feature_selector import FeatureSelector
from src.config import VOTING_MODEL_PATH, SUBMISSION_DIR
from src.utils.logger import setup_logger

logger = setup_logger("predict_pipeline")


def main():
    """Main prediction pipeline."""
    logger.info("="*60)
    logger.info("Starting Prediction Pipeline")
    logger.info("="*60)
    
    # Step 1: Load test data
    logger.info("\n[Step 1/4] Loading test data...")
    data_loader = DataLoader()
    test_data = data_loader.load_test_data()
    sample_submission = data_loader.load_sample_submission()
    
    # Step 2: Select features
    logger.info("\n[Step 2/4] Selecting features...")
    feature_selector = FeatureSelector()
    rf_features = feature_selector.get_rf_features()
    xgb_features = feature_selector.get_xgb_features()
    
    # Extract features
    X_test = test_data.drop(columns=['ID_code'])
    X_rf_test = X_test[rf_features].values
    X_xgb_test = X_test[xgb_features].values
    
    # Step 3: Load model
    logger.info("\n[Step 3/4] Loading trained model...")
    if not VOTING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {VOTING_MODEL_PATH}. "
            "Please run train_pipeline.py first."
        )
    
    voting_model = joblib.load(VOTING_MODEL_PATH)
    logger.info("Model loaded successfully")
    
    # Step 4: Generate predictions
    logger.info("\n[Step 4/4] Generating predictions...")
    X_combined = np.hstack((X_rf_test, X_xgb_test))
    proba_predictions = voting_model.predict_proba(X_combined)[:, 1]
    
    # Prepare submission
    submission = sample_submission.copy()
    submission['target'] = proba_predictions
    
    # Save submission
    submission_path = SUBMISSION_DIR / "voting_classifier_sample_submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"Submission saved to {submission_path}")
    logger.info(f"Prediction statistics:")
    logger.info(f"  Min probability: {proba_predictions.min():.4f}")
    logger.info(f"  Max probability: {proba_predictions.max():.4f}")
    logger.info(f"  Mean probability: {proba_predictions.mean():.4f}")
    logger.info(f"  Std probability: {proba_predictions.std():.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("Prediction pipeline completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

