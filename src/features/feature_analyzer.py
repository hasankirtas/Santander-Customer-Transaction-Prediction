"""
Feature analysis utilities for EDA.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
from typing import Dict, List, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureAnalyzer:
    """Class for feature analysis and importance calculation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureAnalyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.logger = logger
        self.random_state = random_state
    
    def analyze_lasso_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        C: float = 0.01,
        top_n: int = 40
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform Lasso (L1) regression feature selection.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            C: Regularization strength
            top_n: Number of top features to return per class
        
        Returns:
            Dictionary with 'class_0' and 'class_1' DataFrames
        """
        self.logger.info("Performing Lasso feature selection...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Standardize for Lasso
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit Lasso
        lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            max_iter=2000,
            random_state=self.random_state
        )
        lasso.fit(X_train_scaled, y_train)
        
        # Get coefficients
        lasso_coefficients = pd.DataFrame({
            "Feature": X.columns,
            "Lasso_Coefficient": lasso.coef_[0]
        }).sort_values(by="Lasso_Coefficient", key=abs, ascending=False)
        
        # Separate by class
        lasso_zeros = lasso_coefficients[
            lasso_coefficients["Lasso_Coefficient"] < 0
        ].sort_values(by="Lasso_Coefficient", ascending=True).head(top_n)
        
        lasso_ones = lasso_coefficients[
            lasso_coefficients["Lasso_Coefficient"] > 0
        ].sort_values(by="Lasso_Coefficient", ascending=False).head(top_n)
        
        self.logger.info(f"Top {top_n} features for class 0: {len(lasso_zeros)}")
        self.logger.info(f"Top {top_n} features for class 1: {len(lasso_ones)}")
        
        return {
            'class_0': lasso_zeros,
            'class_1': lasso_ones,
            'overall': lasso_coefficients.head(top_n * 2)
        }
    
    def analyze_rf_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        top_n: int = 60
    ) -> pd.Series:
        """
        Calculate Random Forest feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_estimators: Number of trees
            top_n: Number of top features to return
        
        Returns:
            Series with feature importances
        """
        self.logger.info("Calculating Random Forest feature importance...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Fit Random Forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = pd.Series(
            rf.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        self.logger.info(f"Top {top_n} RF features identified")
        return feature_importance.head(top_n)
    
    def analyze_xgb_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        top_n: int = 60
    ) -> pd.Series:
        """
        Calculate XGBoost feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_estimators: Number of trees
            top_n: Number of top features to return
        
        Returns:
            Series with feature importances
        """
        self.logger.info("Calculating XGBoost feature importance...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Fit XGBoost
        xgb = XGBClassifier(
            n_estimators=n_estimators,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = pd.Series(
            xgb.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        self.logger.info(f"Top {top_n} XGBoost features identified")
        return feature_importance.head(top_n)
    
    def analyze_shap_values(
        self,
        model: XGBClassifier,
        X: pd.DataFrame,
        sample_size: int = 1000
    ) -> shap.Explanation:
        """
        Calculate SHAP values for model explainability.
        
        Args:
            model: Trained XGBoost model
            X: Feature DataFrame
            sample_size: Number of samples to use for SHAP calculation
        
        Returns:
            SHAP Explanation object
        """
        self.logger.info("Calculating SHAP values...")
        
        # Sample data if too large
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = X
        
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        
        self.logger.info("SHAP values calculated")
        return shap_values

