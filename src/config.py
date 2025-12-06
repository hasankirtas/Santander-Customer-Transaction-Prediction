"""
Configuration file for the Santander Customer Transaction Prediction project.
Contains all feature lists, model parameters, and paths.
"""

from pathlib import Path
from typing import List, Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                 REPORTS_DIR, SUBMISSION_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"
UNDERSAMPLED_TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "undersampled_train_data.csv"

# Model paths
VOTING_MODEL_PATH = MODELS_DIR / "voting_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"

# Feature lists (from EDA and feature selection)
OVERALL_IMPORTANT_FEATURES: List[str] = [
    'var_81', 'var_12', 'var_6', 'var_26', 'var_53', 'var_110', 'var_146',
    'var_174', 'var_109', 'var_22', 'var_166', 'var_99', 'var_80', 'var_21', 'var_76',
    'var_133', 'var_2', 'var_198', 'var_165', 'var_190', 'var_179', 'var_0',
    'var_78', 'var_148', 'var_40', 'var_44', 'var_34', 'var_170', 'var_94',
    'var_92', 'var_164', 'var_115', 'var_33', 'var_67', 'var_121', 'var_184', 'var_177',
    'var_149', 'var_108', 'var_18', 'var_154', 'var_169', 'var_192', 'var_173', 'var_191',
    'var_127', 'var_75', 'var_118', 'var_122', 'var_91', 'var_107', 'var_123',
    'var_56', 'var_155', 'var_147', 'var_86', 'var_95', 'var_172', 'var_162',
    "var_36", "var_188", "var_87", "var_197", "var_93", "var_31", "var_89",
    "var_35", "var_48", "var_199", "var_32", "var_90", "var_71", 
    "var_157", "var_130", "var_135"
]

CLASS_0_IMPORTANT_FEATURES: List[str] = [
    "var_81", "var_146", "var_12", "var_76", "var_174", "var_34", "var_21", "var_165",
    "var_109", "var_44", "var_166", "var_198", "var_192", "var_148", "var_33", "var_80",
    "var_169", "var_115", "var_92", "var_149", "var_154", "var_121", "var_107", "var_127",
    "var_122", "var_172", "var_177", "var_36", "var_108", "var_75", "var_188", "var_123",
    "var_87", "var_197", "var_86", "var_93", "var_31"
]

CLASS_1_IMPORTANT_FEATURES: List[str] = [
    "var_6", "var_53", "var_26", "var_110", "var_99", "var_190", "var_133", "var_22",
    "var_179", "var_2", "var_94", "var_40", "var_78", "var_173", "var_184", "var_170",
    "var_0", "var_1", "var_191", "var_67", "var_118", "var_147", "var_18", "var_164",
    "var_89", "var_35", "var_48", "var_95", "var_199", "var_155", "var_32", "var_5",
    "var_91", "var_90", "var_71", "var_157", "var_162", "var_130", "var_135", "var_52"
]

# Model hyperparameters (from Optuna tuning)
RF_HYPERPARAMETERS: Dict[str, Any] = {
    'n_estimators': 308,
    'max_depth': 18,
    'min_samples_split': 8,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42
}

XGB_HYPERPARAMETERS: Dict[str, Any] = {
    'n_estimators': 342,
    'learning_rate': 0.08587039035397225,
    'max_depth': 3,
    'colsample_bytree': 0.8295825085075643,
    'subsample': 0.8089687433060441,
    'gamma': 0.2160641676443201,
    'reg_alpha': 0.43595756958749793,
    'reg_lambda': 1.105284812022964,
    'n_jobs': -1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# Resampling parameters
RESAMPLING_CONFIG: Dict[str, Any] = {
    'tomek_links': True,
    'random_undersample_ratio': 0.7,  # Keep 70% of majority class
    'random_state': 42
}

# Cross-validation parameters
CV_CONFIG: Dict[str, Any] = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Training parameters
TRAINING_CONFIG: Dict[str, Any] = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

