# Project: Santander Customer Transaction Prediction

### Overview
This project is based on the Kaggle competition **"Santander Customer Transaction Prediction"**, where the main objective is to predict whether a customer will make a specific transaction or not. 

The dataset consists of **200 anonymized and purely numerical independent features**, and the target variable `target` is binary (0 or 1). The goal is to develop a robust classification model that can accurately identify customers likely to make a transaction using these independent variables.

This project holds a special place in my journey as a data scientist. It has been the most demanding and intellectually challenging project I've worked on so far. It pushed me to revisit core statistical theories, and even led me to start reading *An Introduction to Statistical Learning*, a classic in machine learning literature. The insights gained from this book shaped many of the modeling strategies I implemented throughout the project.

---

### Objectives
1. Perform extensive exploratory data analysis (EDA).
2. Identify key predictive features using domain-agnostic feature selection techniques.
3. Address class imbalance through smart under-sampling strategies.
4. Train and combine models that focus on distinct parts of the class distribution.
5. Evaluate and integrate models using Stratified K-Fold and Soft Voting methods.
6. Deploy the final model and assess its performance on Kaggle’s test set.

---

### Steps Followed

#### **1. Exploratory Data Analysis**
- Conducted a thorough inspection of the dataset structure and the target distribution.
- Noticed a **class imbalance** where approximately 10% of samples belonged to class 1 and 90% to class 0 (`target:1 ≈ 20,000`, `target:0 ≈ 180,000`).
- Investigated **outliers** but decided not to take action due to their minimal presence and because tree-based models are inherently robust to them.

#### **2. Statistical Feature Analysis**
- Analyzed the **skewness and kurtosis** of each feature. Although many variables showed deviations from normality, transformations were not applied since models employed were non-parametric and non-linear.
- Explored **correlations** among features and discovered very weak or no correlations, which allowed for greater freedom during feature selection.

#### **3. Class-wise Feature Distribution Comparison**
- Visualized the distribution of each independent variable separately for class 0 and class 1.
- Identified features with **notably different distributional behaviors across classes**, which I flagged as potential predictors. This class-wise comparative analysis was a cornerstone in my feature selection process.

#### **4. Feature Selection Techniques**
To identify the most impactful features, I employed multiple techniques and cross-validated their outputs:
- **Lasso Regression (L1 Regularization)**
- **XGBoost Feature Importance**
- **Random Forest Feature Importance**
- **SHAP Values (SHapley Additive exPlanations)**

I compared the top features obtained from each method with the features flagged during distribution analysis to refine a set of **truly discriminative features for each class**.

#### **5. Handling Class Imbalance**
To counter the skewed class distribution without introducing synthetic data, I:
- Applied **TomekLinks()** and **RandomUnderSampler()** to reduce the majority class, while preserving minority class originality and **minimizing information loss**.
- This allowed better learning for the minority class during model training.

#### **6. Targeted Modeling by Class**
Recognizing the limitations of a single model in handling imbalanced data:
- Trained an **XGBoost model focused on the minority class** (`target: 1`).
- Trained a **Random Forest model specialized on the majority class** (`target: 0`).
- Used **class-weight adjustments** to further tune the sensitivity of each model toward its respective class.
- Employed **Stratified K-Fold Cross-Validation** to ensure robust and class-balanced evaluation during model development.

#### **7. Ensemble Modeling and Final Prediction**
- Combined the XGBoost and Random Forest models using a **soft-voting VotingClassifier**, which allowed the ensemble to leverage the strengths of each base model.
- Generated predictions on the test set and submitted them to Kaggle.
- Achieved a **ROC-AUC score of approximately 0.84**, indicating solid performance and reliable separation of the classes.

---

### Personal Insight & Additional Experiments
This project wasn’t just technical — it was personal. I had set it aside multiple times, but persistent curiosity and a sense of unfinished business kept bringing me back. Along the way, I:

- Revisited foundational statistical theory through *An Introduction to Statistical Learning*, which inspired several modeling experiments.
- Tested advanced models like **Quadratic Discriminant Analysis** and **Naive Bayes**, influenced by the book’s *Classification* chapter. Although they didn’t outperform tree-based models, they offered valuable learning opportunities.
- Created **polynomial interaction features** in an attempt to capture possible non-linear relationships between variables.

Through this project and similar Kaggle challenges, I better understood **why the industry often favors models like XGBoost, Random Forest, LightGBM, and Deep Learning** over classical techniques like Linear or Logistic Regression — especially in complex, high-dimensional, and noisy datasets.

---

### Tools and Libraries Used
- **Python**
- **Libraries**:
  - pandas, numpy (Data Handling)
  - matplotlib, seaborn (Data Visualization)
  - scikit-learn (Modeling, Resampling)
  - xgboost (Gradient Boosting)
  - shap (Model Explainability)
  - imbalanced-learn (Sampling Techniques)

---

### Conclusion
This project was a milestone in my data science learning path. It challenged me to merge practical skills with theoretical foundations, and helped me grow both analytically and intellectually. The final model successfully balanced performance and interpretability, and I’m confident that the methods applied here can scale to real-world, high-impact classification problems.

---

### Acknowledgements
- Kaggle for providing the dataset and platform.
- Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani for An Introduction to Statistical Learning, which enriched my theoretical understanding.

---

###  My Thoughts and Insights
This project reinforced the value of perseverance, curiosity, and foundational knowledge in machine learning. It taught me that data science isn't just about models — it's about understanding the data, asking the right questions, and never hesitating to go back to theory when things don't work out as expected.

---