"""
Model training utilities.
"""

import numpy as np
import pandas as pd
import time
import joblib
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from src.config import (
    RF_HYPERPARAMETERS,
    XGB_HYPERPARAMETERS,
    CV_CONFIG,
    VOTING_MODEL_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Class for training machine learning models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.logger = logger
        self.random_state = random_state
        self.rf_model = None
        self.xgb_model = None
        self.voting_model = None
    
    def calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced data.
        
        Args:
            y: Target Series
        
        Returns:
            Dictionary with class weights
        """
        class_0_count = np.sum(y == 0)
        class_1_count = np.sum(y == 1)
        scale_pos_weight = class_0_count / class_1_count
        
        return {
            'scale_pos_weight': scale_pos_weight,
            'class_weight': {0: 1, 1: scale_pos_weight}
        }
    
    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None,
        class_weight: Optional[Dict[int, float]] = None
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X: Feature array
            y: Target array
            hyperparameters: Optional hyperparameters dict
            class_weight: Optional class weights
        
        Returns:
            Trained RandomForestClassifier
        """
        self.logger.info("Training Random Forest model...")
        
        params = (hyperparameters or RF_HYPERPARAMETERS).copy()
        if class_weight:
            params['class_weight'] = class_weight
        
        start_time = time.time()
        self.rf_model = RandomForestClassifier(**params)
        self.rf_model.fit(X, y)
        training_time = time.time() - start_time
        
        self.logger.info(f"Random Forest training completed in {training_time:.2f} seconds")
        return self.rf_model
    
    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None,
        scale_pos_weight: Optional[float] = None
    ) -> XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X: Feature array
            y: Target array
            hyperparameters: Optional hyperparameters dict
            scale_pos_weight: Optional scale_pos_weight parameter
        
        Returns:
            Trained XGBClassifier
        """
        self.logger.info("Training XGBoost model...")
        
        params = (hyperparameters or XGB_HYPERPARAMETERS).copy()
        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight
        
        start_time = time.time()
        self.xgb_model = XGBClassifier(**params)
        self.xgb_model.fit(X, y)
        training_time = time.time() - start_time
        
        self.logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
        return self.xgb_model
    
    def train_voting_classifier(
        self,
        X_rf: np.ndarray,
        X_xgb: np.ndarray,
        y: np.ndarray,
        rf_model: Optional[RandomForestClassifier] = None,
        xgb_model: Optional[XGBClassifier] = None
    ) -> VotingClassifier:
        """
        Train Voting Classifier combining RF and XGBoost.
        
        Args:
            X_rf: Features for Random Forest
            X_xgb: Features for XGBoost
            y: Target array
            rf_model: Optional pre-trained RF model
            xgb_model: Optional pre-trained XGBoost model
        
        Returns:
            Trained VotingClassifier
        """
        self.logger.info("Training Voting Classifier...")
        
        # Use existing models or train new ones
        if rf_model is None:
            rf_model = self.rf_model
        if xgb_model is None:
            xgb_model = self.xgb_model
        
        if rf_model is None or xgb_model is None:
            raise ValueError("RF and XGBoost models must be trained first")
        
        # Combine features
        X_combined = np.hstack((X_rf, X_xgb))
        
        # Create voting classifier
        self.voting_model = VotingClassifier(
            estimators=[('rf', rf_model), ('xgb', xgb_model)],
            voting='soft',
            n_jobs=-1
        )
        
        start_time = time.time()
        self.voting_model.fit(X_combined, y)
        training_time = time.time() - start_time
        
        self.logger.info(f"Voting Classifier training completed in {training_time:.2f} seconds")
        return self.voting_model
    
    def cross_validate_models(
        self,
        X_rf: np.ndarray,
        X_xgb: np.ndarray,
        y: np.ndarray,
        rf_hyperparameters: Optional[Dict[str, Any]] = None,
        xgb_hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for both models.
        
        Args:
            X_rf: Features for Random Forest
            X_xgb: Features for XGBoost
            y: Target array
            rf_hyperparameters: Optional RF hyperparameters
            xgb_hyperparameters: Optional XGBoost hyperparameters
        
        Returns:
            Dictionary with CV results
        """
        self.logger.info("Starting cross-validation...")
        
        skf = StratifiedKFold(
            n_splits=CV_CONFIG['n_splits'],
            shuffle=CV_CONFIG['shuffle'],
            random_state=self.random_state
        )
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(pd.Series(y))
        
        rf_roc_aucs = []
        xgb_roc_aucs = []
        rf_times = []
        xgb_times = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_rf, y), 1):
            self.logger.info(f"Processing fold {fold}/{CV_CONFIG['n_splits']}...")
            
            X_rf_train, X_rf_test = X_rf[train_idx], X_rf[test_idx]
            X_xgb_train, X_xgb_test = X_xgb[train_idx], X_xgb[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and evaluate RF
            rf_params = (rf_hyperparameters or RF_HYPERPARAMETERS).copy()
            rf_params['class_weight'] = class_weights['class_weight']
            
            start_time = time.time()
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_rf_train, y_train)
            rf_times.append(time.time() - start_time)
            
            from sklearn.metrics import roc_auc_score
            rf_proba = rf_model.predict_proba(X_rf_test)[:, 1]
            rf_roc_aucs.append(roc_auc_score(y_test, rf_proba))
            
            # Train and evaluate XGBoost
            xgb_params = (xgb_hyperparameters or XGB_HYPERPARAMETERS).copy()
            xgb_params['scale_pos_weight'] = class_weights['scale_pos_weight']
            
            start_time = time.time()
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(X_xgb_train, y_train)
            xgb_times.append(time.time() - start_time)
            
            xgb_proba = xgb_model.predict_proba(X_xgb_test)[:, 1]
            xgb_roc_aucs.append(roc_auc_score(y_test, xgb_proba))
        
        results = {
            'rf_mean_roc_auc': np.mean(rf_roc_aucs),
            'rf_std_roc_auc': np.std(rf_roc_aucs),
            'xgb_mean_roc_auc': np.mean(xgb_roc_aucs),
            'xgb_std_roc_auc': np.std(xgb_roc_aucs),
            'rf_mean_time': np.mean(rf_times),
            'xgb_mean_time': np.mean(xgb_times),
            'rf_scores': rf_roc_aucs,
            'xgb_scores': xgb_roc_aucs
        }
        
        self.logger.info(f"CV Results - RF: {results['rf_mean_roc_auc']:.4f} ± {results['rf_std_roc_auc']:.4f}")
        self.logger.info(f"CV Results - XGBoost: {results['xgb_mean_roc_auc']:.4f} ± {results['xgb_std_roc_auc']:.4f}")
        
        return results
    
    def save_models(
        self,
        voting_model_path: Optional[Path] = None,
        rf_model_path: Optional[Path] = None,
        xgb_model_path: Optional[Path] = None
    ) -> None:
        """
        Save trained models to disk.
        
        Args:
            voting_model_path: Path to save voting model
            rf_model_path: Path to save RF model
            xgb_model_path: Path to save XGBoost model
        """
        if self.voting_model:
            path = voting_model_path or VOTING_MODEL_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.voting_model, path)
            self.logger.info(f"Voting model saved to {path}")
        
        if self.rf_model:
            path = rf_model_path or RF_MODEL_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.rf_model, path)
            self.logger.info(f"RF model saved to {path}")
        
        if self.xgb_model:
            path = xgb_model_path or XGB_MODEL_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.xgb_model, path)
            self.logger.info(f"XGBoost model saved to {path}")

