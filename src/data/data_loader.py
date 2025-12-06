"""
Data loading utilities.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    SAMPLE_SUBMISSION_PATH,
    UNDERSAMPLED_TRAIN_DATA_PATH
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Class for loading datasets."""
    
    def __init__(self):
        """Initialize DataLoader."""
        self.logger = logger
    
    def load_train_data(
        self,
        path: Optional[Path] = None,
        use_processed: bool = False
    ) -> pd.DataFrame:
        """
        Load training data.
        
        Args:
            path: Optional custom path to train data
            use_processed: If True, load undersampled processed data
        
        Returns:
            Training DataFrame
        """
        if use_processed:
            data_path = path or UNDERSAMPLED_TRAIN_DATA_PATH
            self.logger.info(f"Loading processed training data from {data_path}")
        else:
            data_path = path or TRAIN_DATA_PATH
            self.logger.info(f"Loading raw training data from {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        data = pd.read_csv(data_path).copy()
        self.logger.info(f"Loaded training data: {data.shape}")
        return data
    
    def load_test_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load test data.
        
        Args:
            path: Optional custom path to test data
        
        Returns:
            Test DataFrame
        """
        data_path = path or TEST_DATA_PATH
        self.logger.info(f"Loading test data from {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Test data not found at {data_path}")
        
        data = pd.read_csv(data_path).copy()
        self.logger.info(f"Loaded test data: {data.shape}")
        return data
    
    def load_sample_submission(
        self,
        path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load sample submission file.
        
        Args:
            path: Optional custom path to sample submission
        
        Returns:
            Sample submission DataFrame
        """
        data_path = path or SAMPLE_SUBMISSION_PATH
        self.logger.info(f"Loading sample submission from {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Sample submission not found at {data_path}")
        
        data = pd.read_csv(data_path).copy()
        self.logger.info(f"Loaded sample submission: {data.shape}")
        return data
    
    def get_features_and_target(
        self,
        data: pd.DataFrame,
        features: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target from training data.
        
        Args:
            data: Training DataFrame
            features: Optional list of features to select
        
        Returns:
            Tuple of (X, y)
        """
        if features is None:
            # Use all features except ID_code and target
            X = data.drop(columns=['ID_code', 'target'])
        else:
            X = data[features]
        
        y = data['target']
        
        self.logger.info(f"Extracted features: {X.shape}, target: {y.shape}")
        return X, y

