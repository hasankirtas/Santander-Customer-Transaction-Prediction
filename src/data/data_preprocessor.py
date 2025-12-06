"""
Data preprocessing utilities including resampling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from pathlib import Path
from src.config import (
    RESAMPLING_CONFIG,
    UNDERSAMPLED_TRAIN_DATA_PATH
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """Class for data preprocessing operations."""
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.logger = logger
        self.tomek = None
        self.rus = None
    
    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = "IQR"
    ) -> dict:
        """
        Detect outliers using specified method.
        
        Args:
            data: DataFrame with numeric columns
            method: Method to use ('IQR' or 'Z-score')
        
        Returns:
            Dictionary mapping column names to outlier indices
        """
        outliers = {}
        
        if method == "IQR":
            for column in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = data[
                    (data[column] < lower_bound) | (data[column] > upper_bound)
                ].index.tolist()
                outliers[column] = outlier_indices
        
        elif method == "Z-score":
            for column in data.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outlier_indices = data[z_scores > 3].index.tolist()
                outliers[column] = outlier_indices
        
        # Count total unique outliers
        all_outlier_indices = set()
        for indices in outliers.values():
            all_outlier_indices.update(indices)
        
        self.logger.info(f"Total unique outlier observations: {len(all_outlier_indices)}")
        return outliers
    
    def calculate_skewness_kurtosis(
        self,
        data: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Calculate average skewness and kurtosis for numeric columns.
        
        Args:
            data: DataFrame with numeric columns
        
        Returns:
            Tuple of (average_skewness, average_kurtosis)
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        skewness_values = [data[col].skew() for col in numeric_columns]
        kurtosis_values = [data[col].kurtosis() for col in numeric_columns]
        
        avg_skewness = np.mean(skewness_values)
        avg_kurtosis = np.mean(kurtosis_values)
        
        self.logger.info(f"Average Skewness: {avg_skewness:.4f}")
        self.logger.info(f"Average Kurtosis: {avg_kurtosis:.4f}")
        
        return avg_skewness, avg_kurtosis
    
    def resample_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        id_column: Optional[pd.Series] = None,
        save_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling to balance classes.
        Uses TomekLinks followed by RandomUnderSampler.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            id_column: Optional ID column to preserve
            save_path: Optional path to save resampled data
        
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        self.logger.info("Starting resampling process...")
        self.logger.info(f"Original class distribution:\n{y.value_counts()}")
        
        # Step 1: Apply TomekLinks
        if RESAMPLING_CONFIG['tomek_links']:
            self.logger.info("Applying TomekLinks...")
            self.tomek = TomekLinks()
            X_resampled, y_resampled = self.tomek.fit_resample(X, y)
            self.logger.info(f"After TomekLinks: {pd.Series(y_resampled).value_counts().to_dict()}")
        else:
            X_resampled, y_resampled = X.values, y.values
        
        # Step 2: Apply RandomUnderSampler
        majority_class_count = pd.Series(y_resampled).value_counts()[0]
        minority_class_count = pd.Series(y_resampled).value_counts()[1]
        target_majority_count = int(
            majority_class_count * RESAMPLING_CONFIG['random_undersample_ratio']
        )
        
        self.logger.info("Applying RandomUnderSampler...")
        self.rus = RandomUnderSampler(
            sampling_strategy={0: target_majority_count, 1: minority_class_count},
            random_state=RESAMPLING_CONFIG['random_state']
        )
        X_resampled, y_resampled = self.rus.fit_resample(X_resampled, y_resampled)
        
        # Convert back to DataFrame/Series
        X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled_series = pd.Series(y_resampled, name='target')
        
        self.logger.info(f"Final class distribution:\n{y_resampled_series.value_counts()}")
        
        # Save if path provided
        if save_path:
            save_path = save_path or UNDERSAMPLED_TRAIN_DATA_PATH
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            resampled_data = X_resampled_df.copy()
            if id_column is not None:
                resampled_data.insert(0, 'ID_code', id_column.iloc[:len(resampled_data)].values)
            resampled_data.insert(len(resampled_data.columns), 'target', y_resampled_series.values)
            
            resampled_data.to_csv(save_path, index=False)
            self.logger.info(f"Resampled data saved to {save_path}")
        
        return X_resampled_df, y_resampled_series

