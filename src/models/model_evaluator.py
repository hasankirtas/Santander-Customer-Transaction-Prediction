"""
Model evaluation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Class for model evaluation."""
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.logger = logger
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Confusion matrix array
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred)
    
    def print_evaluation_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> None:
        """
        Print comprehensive evaluation summary.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Evaluation Summary for {model_name}")
        self.logger.info(f"{'='*50}")
        
        metrics = self.evaluate_model(y_true, y_pred, y_proba)
        
        for metric, value in metrics.items():
            self.logger.info(f"{metric.upper()}: {value:.4f}")
        
        cm = self.get_confusion_matrix(y_true, y_pred)
        self.logger.info(f"\nConfusion Matrix:\n{cm}")
        
        report = self.get_classification_report(y_true, y_pred)
        self.logger.info(f"\nClassification Report:\n{report}")

