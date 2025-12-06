# Project: Santander Customer Transaction Prediction

### Overview

This project is based on the Kaggle competition **"Santander Customer Transaction Prediction"**, where the main objective is to predict whether a customer will make a specific transaction or not. 

The dataset consists of **200 anonymized and purely numerical independent features**, and the target variable `target` is binary (0 or 1). The goal is to develop a robust classification model that can accurately identify customers likely to make a transaction using these independent variables.

This project holds a special place in my journey as a data scientist. It has been the most demanding and intellectually challenging project I've worked on so far. It pushed me to revisit core statistical theories, and even led me to start reading *An Introduction to Statistical Learning*, a classic in machine learning literature. The insights gained from this book shaped many of the modeling strategies I implemented throughout the project.

---

## 🏗️ Project Structure

```
Santander Project/
│
├── src/                          # Main source code
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Centralized configuration
│   │
│   ├── data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Data loading utilities
│   │   └── data_preprocessor.py  # Preprocessing and resampling
│   │
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_analyzer.py   # Feature analysis (Lasso, RF, XGB, SHAP)
│   │   └── feature_selector.py   # Feature selection utilities
│   │
│   ├── models/                   # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── model_trainer.py      # Model training utilities
│   │   └── model_evaluator.py   # Model evaluation metrics
│   │
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── logger.py             # Logging configuration
│       └── visualization.py      # Plotting utilities
│
├── scripts/                      # Executable pipelines
│   ├── __init__.py
│   ├── train_pipeline.py        # Main training pipeline
│   └── predict_pipeline.py      # Prediction pipeline
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data files (not in git)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/                # Processed data
│       └── undersampled_train_data.csv
│
├── models/                       # Saved models (not in git)
│   ├── voting_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
│
├── reports/                      # Reports and visualizations
│   └── *.png
│
├── submission/                    # Submission files
│   └── voting_classifier_sample_submission.csv
│
├── requirements.txt              # Python dependencies (core)
├── requirements-dev.txt          # Development dependencies (optional)
├── environment.yml              # Conda environment
├── setup.py                     # Package setup
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

### Module Descriptions

#### `src/config.py`
Centralized configuration file containing:
- Project paths
- Feature lists (overall, class 0, class 1)
- Model hyperparameters
- Resampling parameters
- Cross-validation settings
- Logging configuration

#### `src/data/data_loader.py`
- `DataLoader`: Loads training, test, and submission data
- Methods: `load_train_data()`, `load_test_data()`, `get_features_and_target()`

#### `src/data/data_preprocessor.py`
- `DataPreprocessor`: Handles data preprocessing
- Methods: `detect_outliers()`, `calculate_skewness_kurtosis()`, `resample_data()`

#### `src/features/feature_analyzer.py`
- `FeatureAnalyzer`: Analyzes feature importance
- Methods: `analyze_lasso_features()`, `analyze_rf_features()`, `analyze_xgb_features()`, `analyze_shap_values()`

#### `src/features/feature_selector.py`
- `FeatureSelector`: Provides feature lists
- Methods: `get_overall_features()`, `get_class_0_features()`, `get_class_1_features()`, `get_rf_features()`, `get_xgb_features()`

#### `src/models/model_trainer.py`
- `ModelTrainer`: Trains machine learning models
- Methods: `train_random_forest()`, `train_xgboost()`, `train_voting_classifier()`, `cross_validate_models()`, `save_models()`

#### `src/models/model_evaluator.py`
- `ModelEvaluator`: Evaluates model performance
- Methods: `evaluate_model()`, `get_confusion_matrix()`, `get_classification_report()`, `print_evaluation_summary()`

#### `src/utils/logger.py`
- `setup_logger()`: Configures logging with file and console handlers

#### `src/utils/visualization.py`
- Plotting utilities: `plot_target_distribution()`, `plot_feature_distributions()`, `plot_confusion_matrix()`

### Pipeline Scripts

#### `scripts/train_pipeline.py`
Main training pipeline that:
1. Loads training data
2. Selects features
3. Resamples data to handle class imbalance
4. Performs cross-validation
5. Trains final models (RF, XGBoost, Voting Classifier)
6. Evaluates and saves models

#### `scripts/predict_pipeline.py`
Prediction pipeline that:
1. Loads test data
2. Selects features
3. Loads trained model
4. Generates predictions
5. Saves submission file

---

## 🎯 Objectives

1. Perform extensive exploratory data analysis (EDA).
2. Identify key predictive features using domain-agnostic feature selection techniques.
3. Address class imbalance through smart under-sampling strategies.
4. Train and combine models that focus on distinct parts of the class distribution.
5. Evaluate and integrate models using Stratified K-Fold and Soft Voting methods.
6. Deploy the final model and assess its performance on Kaggle's test set.

---

## 📋 Steps Followed

### **1. Exploratory Data Analysis**

- Conducted a thorough inspection of the dataset structure and the target distribution.
- Noticed a **class imbalance** where approximately 10% of samples belonged to class 1 and 90% to class 0 (`target:1 ≈ 20,000`, `target:0 ≈ 180,000`).
- Investigated **outliers** but decided not to take action due to their minimal presence and because tree-based models are inherently robust to them.

### **2. Statistical Feature Analysis**

- Analyzed the **skewness and kurtosis** of each feature. Although many variables showed deviations from normality, transformations were not applied since models employed were non-parametric and non-linear.
- Explored **correlations** among features and discovered very weak or no correlations, which allowed for greater freedom during feature selection.

### **3. Class-wise Feature Distribution Comparison**

- Visualized the distribution of each independent variable separately for class 0 and class 1.
- Identified features with **notably different distributional behaviors across classes**, which I flagged as potential predictors. This class-wise comparative analysis was a cornerstone in my feature selection process.

### **4. Feature Selection Techniques**

To identify the most impactful features, I employed multiple techniques and cross-validated their outputs:

- **Lasso Regression (L1 Regularization)**
- **XGBoost Feature Importance**
- **Random Forest Feature Importance**
- **SHAP Values (SHapley Additive exPlanations)**

I compared the top features obtained from each method with the features flagged during distribution analysis to refine a set of **truly discriminative features for each class**.

### **5. Handling Class Imbalance**

To counter the skewed class distribution without introducing synthetic data, I:

- Applied **TomekLinks()** and **RandomUnderSampler()** to reduce the majority class, while preserving minority class originality and **minimizing information loss**.
- This allowed better learning for the minority class during model training.

### **6. Targeted Modeling by Class**

Recognizing the limitations of a single model in handling imbalanced data:

- Trained an **XGBoost model focused on the minority class** (`target: 1`).
- Trained a **Random Forest model specialized on the majority class** (`target: 0`).
- Used **class-weight adjustments** to further tune the sensitivity of each model toward its respective class.
- Employed **Stratified K-Fold Cross-Validation** to ensure robust and class-balanced evaluation during model development.

### **7. Ensemble Modeling and Final Prediction**

- Combined the XGBoost and Random Forest models using a **soft-voting VotingClassifier**, which allowed the ensemble to leverage the strengths of each base model.
- Generated predictions on the test set and submitted them to Kaggle.
- Achieved a **ROC-AUC score of approximately 0.85500**, indicating solid performance and reliable separation of the classes.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hasankirtas/Santander-Customer-Transaction-Prediction.git
   cd Santander-Customer-Transaction-Prediction
   ```

2. **Install dependencies:**

   **For basic use (scripts only):**
   ```bash
   pip install -r requirements.txt
   ```

   **For development (including optional dependencies):**
   ```bash
   pip install -r requirements-dev.txt
   ```
   
   > **Note**: The original Jupyter notebooks have been removed as the project now uses pipeline scripts. All functionality is available through `scripts/train_pipeline.py` and `scripts/predict_pipeline.py`.

   **Using conda:**
   ```bash
   conda env create -f environment.yml
   conda activate sandanter_customer_transaction_prediction
   ```

3. **Prepare data:**
   - Download the competition data from [Kaggle](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)
   - Place `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/raw/` directory

### Usage

#### Training the Model

Run the training pipeline:

```bash
python scripts/train_pipeline.py
```

This will:
1. Load and preprocess the training data
2. Apply resampling to handle class imbalance
3. Perform cross-validation
4. Train final models (Random Forest, XGBoost, and Voting Classifier)
5. Evaluate models and save them to `models/`

#### Making Predictions

After training, generate predictions on test data:

```bash
python scripts/predict_pipeline.py
```

This will:
1. Load the trained voting classifier
2. Load and preprocess test data
3. Generate predictions
4. Save submission file to `submission/`

---

## 🔧 Configuration

All configuration parameters are centralized in `src/config.py`, including:

- Feature lists (overall, class 0, class 1)
- Model hyperparameters
- Resampling parameters
- Cross-validation settings
- File paths

You can modify these settings directly in the config file without changing the main code.

---

## 📊 Model Architecture

The final model is a **Soft Voting Classifier** that combines:

1. **Random Forest Classifier** (focused on class 0)
   - 37 features selected for class 0
   - Hyperparameters optimized via Optuna
   - Class weights adjusted for imbalanced data

2. **XGBoost Classifier** (focused on class 1)
   - 40 features selected for class 1
   - Hyperparameters optimized via Optuna
   - Scale pos weight adjusted for imbalanced data

The ensemble uses soft voting to combine probability predictions from both models.

---

## 📈 Results

- **ROC-AUC Score**: ~0.85500
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Final Model**: Soft Voting Classifier (RF + XGBoost)

---

## 🛠️ Dependencies

### Core Dependencies (Required)

These dependencies are **required** for the main code (scripts in `scripts/`):

| Package | Version | Usage |
|---------|---------|-------|
| `pandas` | 2.2.2 | Data manipulation and analysis |
| `numpy` | 1.26.4 | Numerical computations |
| `scikit-learn` | 1.6.1 | Machine learning models and utilities |
| `xgboost` | 2.1.4 | Gradient boosting classifier |
| `joblib` | 1.4.2 | Model serialization |
| `matplotlib` | 3.9.2 | Plotting and visualization |
| `seaborn` | 0.13.2 | Statistical data visualization |
| `imbalanced-learn` | 0.12.3 | Resampling techniques (TomekLinks, RandomUnderSampler) |
| `shap` | >=0.42.0 | Model explainability and feature importance |

### Optional Dependencies

These dependencies are **optional** and only needed for specific use cases:

| Package | Version | Usage |
|---------|---------|-------|
| `statsmodels` | >=0.14.0 | Statistical analysis (optional) |
| `scipy` | >=1.11.0 | Statistical functions (included in conda env) |
| `optuna` | >=3.0.0 | Hyperparameter optimization (optional) |

### Dependency Usage by Module

- **`src/data/`**: pandas, numpy, imbalanced-learn
- **`src/features/`**: pandas, numpy, scikit-learn, xgboost, shap
- **`src/models/`**: scikit-learn, xgboost, numpy, joblib
- **`src/utils/`**: matplotlib, seaborn, pandas, numpy, logging (built-in)
- **`scripts/`**: All core dependencies

### Version Compatibility

- Python: 3.8+ (tested with 3.12)
- All dependencies are pinned to specific versions for reproducibility
- Optional dependencies use `>=` to allow newer compatible versions

---

## 🎓 Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Reusability**: Functions and classes can be easily reused
3. **Configurability**: All settings centralized in `config.py`
4. **Maintainability**: Clear structure and documentation
5. **Extensibility**: Easy to add new features or models

### Benefits of This Structure

1. **Separation of Concerns**: Data, features, models, and utilities are separated
2. **Easy Testing**: Each module can be tested independently
3. **Version Control**: Clear structure makes git management easier
4. **Collaboration**: Team members can work on different modules
5. **Scalability**: Easy to add new models, features, or preprocessing steps

---

## 📝 Personal Insight & Additional Experiments

This project wasn't just technical — it was personal. I had set it aside multiple times, but persistent curiosity and a sense of unfinished business kept bringing me back. Along the way, I:

- Revisited foundational statistical theory through *An Introduction to Statistical Learning*, which inspired several modeling experiments.
- Tested advanced models like **Quadratic Discriminant Analysis** and **Naive Bayes**, influenced by the book's *Classification* chapter. Although they didn't outperform tree-based models, they offered valuable learning opportunities.
- Created **polynomial interaction features** in an attempt to capture possible non-linear relationships between variables.

Through this project and similar Kaggle challenges, I better understood **why the industry often favors models like XGBoost, Random Forest, LightGBM, and Deep Learning** over classical techniques like Linear or Logistic Regression — especially in complex, high-dimensional, and noisy datasets.

---

## 🎓 Key Learnings

This project reinforced the value of:
- **Perseverance**: Coming back to challenging problems
- **Curiosity**: Exploring different approaches and techniques
- **Foundational Knowledge**: Understanding theory behind methods
- **Modular Design**: Building reusable, maintainable code
- **Systematic Approach**: Following a structured pipeline from EDA to deployment

---

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome!

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgements

- Kaggle for providing the dataset and platform
- Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani for *An Introduction to Statistical Learning*, which enriched my theoretical understanding

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🔄 Version History

- **v1.0.0** (Current): Modular refactoring
  - Modular code structure
  - Pipeline scripts
  - Centralized configuration
  - Comprehensive documentation

---

## 💭 My Thoughts and Insights

This project reinforced the value of perseverance, curiosity, and foundational knowledge in machine learning. It taught me that data science isn't just about models — it's about understanding the data, asking the right questions, and never hesitating to go back to theory when things don't work out as expected.

The refactoring of this project into a modular structure represents another learning milestone — practicing to write maintainable code that can be easily understood, modified, and extended.

---

## 📚 References

- [Kaggle Competition](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
