"""
Feature selection utilities.
"""

from typing import List
from src.config import (
    OVERALL_IMPORTANT_FEATURES,
    CLASS_0_IMPORTANT_FEATURES,
    CLASS_1_IMPORTANT_FEATURES
)


class FeatureSelector:
    """Class for feature selection operations."""
    
    @staticmethod
    def get_overall_features() -> List[str]:
        """
        Get overall important features.
        
        Returns:
            List of feature names
        """
        return OVERALL_IMPORTANT_FEATURES.copy()
    
    @staticmethod
    def get_class_0_features() -> List[str]:
        """
        Get features important for class 0.
        
        Returns:
            List of feature names
        """
        return CLASS_0_IMPORTANT_FEATURES.copy()
    
    @staticmethod
    def get_class_1_features() -> List[str]:
        """
        Get features important for class 1.
        
        Returns:
            List of feature names
        """
        return CLASS_1_IMPORTANT_FEATURES.copy()
    
    @staticmethod
    def get_rf_features() -> List[str]:
        """
        Get features for Random Forest model (class 0 focused).
        
        Returns:
            List of feature names
        """
        return CLASS_0_IMPORTANT_FEATURES.copy()
    
    @staticmethod
    def get_xgb_features() -> List[str]:
        """
        Get features for XGBoost model (class 1 focused).
        
        Returns:
            List of feature names
        """
        return CLASS_1_IMPORTANT_FEATURES.copy()

