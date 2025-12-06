"""
Visualization utilities for EDA and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from src.config import REPORTS_DIR


def plot_target_distribution(y: pd.Series, save_path: Optional[Path] = None) -> None:
    """
    Plot the distribution of the target variable.
    
    Args:
        y: Target variable series
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette="viridis")
    plt.title("Target Variable Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    
    if save_path:
        if not save_path.is_absolute():
            save_path = REPORTS_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_feature_distributions(
    df_class_0: pd.DataFrame,
    df_class_1: pd.DataFrame,
    features: List[str],
    label_0: str = "0",
    label_1: str = "1",
    save_path: Optional[Path] = None,
    n_cols: int = 10
) -> None:
    """
    Plot feature distributions for two classes side by side.
    
    Args:
        df_class_0: DataFrame for class 0
        df_class_1: DataFrame for class 1
        features: List of feature names to plot
        label_0: Label for class 0
        label_1: Label for class 1
        save_path: Optional path to save the plot
        n_cols: Number of columns in the subplot grid
    """
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.distplot(df_class_0[feature], hist=False, label=label_0, ax=axes[i])
            sns.distplot(df_class_1[feature], hist=False, label=label_1, ax=axes[i])
            axes[i].set_xlabel(feature, fontsize=9)
            axes[i].tick_params(axis='x', which='major', labelsize=6, pad=-6)
            axes[i].tick_params(axis='y', which='major', labelsize=6)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        if not save_path.is_absolute():
            save_path = REPORTS_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        if not save_path.is_absolute():
            save_path = REPORTS_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

