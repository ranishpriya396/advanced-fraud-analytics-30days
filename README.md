# 30-Day Advanced Machine Learning Lifecycle: Credit Card Fraud Analytics

An end-to-end, production-grade machine learning framework designed to detect fraudulent e-commerce credit card transactions under extreme conditions: heavy class imbalance (<1%), high dimensionality, missing fields, and severe outlier contamination. 

This repository documents a rigorous, 30-day daily practice routine (2 hours/day) tracking the complete machine learning lifecycle—from raw, messy data preprocessing to automated pipeline deployment and a final regression pivot.

## 🛠️ Tech Stack & Core Libraries
* **Data Core:** Python, NumPy, Pandas
* **Visualization:** Seaborn, Matplotlib
* **Classical & Linear Baseline Models:** Scikit-Learn (Logistic Regression, SVC, ComplementNB)
* **Tree Ensemble Kings:** XGBoost, LightGBM, CatBoost
* **Advanced Frameworks:** Optuna (Bayesian Optimization & TPE Sampler)

## 📊 Dataset Constraints & Challenges
* **Target Imbalance:** <1% Fraudulent transactions (requires custom scoring thresholds and stratified evaluation).
* **Missing Data:** Deep structural missingness across user and device columns.
* **Skewness:** Highly right-skewed continuous variables (Transaction Amounts).
* **Dimensionality:** Over 100 features requiring statistical and dynamic pruning.

---

## 🗺️ 30-Day Execution Roadmap

### 📦 Week 1: Advanced Preprocessing & Data Integrity
* [ ] **Day 01**: Outlier Detection (Interquartile Range [IQR] & Trim Boundaries)
* [ ] **Day 02**: Outlier Isolation (Contamination tuning via Isolation Forests)
* [ ] **Day 03**: Feature Transformation (Diagnosing skewness with Skew metrics)
* [ ] **Day 04**: Power Transforms (Applying Logarithmic & Box-Cox conversions)
* [ ] **Day 05**: Data Imputation (Baseline Median & Mode fill operations)
* [ ] **Day 06**: Advanced Imputation (Multi-variable filling via KNNImputer)
* [ ] **Day 07**: Weekly Review (Consolidating transformation code structures)

### 🔒 Week 2: Categorical Encoding & Leakage Defence
* [ ] **Day 08**: Nominal Encoding (One-Hot & Frequency mapping for low cardinality)
* [ ] **Day 09**: High-Cardinality Encoding (Target Encoding & M-Estimate methods)
* [ ] **Day 10**: Imbalanced Feature Selection (Mutual Information & Kendall's Tau)
* [ ] **Day 11**: Statistical Filtering (ANOVA F-Tests & Chi-Square selection)
* [ ] **Day 12**: Leakage Architecture (Mapping fit/transform boundaries)
* [ ] **Day 13**: Scikit-Learn Pipelines (Bundling Preprocessing & Feature Selection)
* [ ] **Day 14**: Weekly Review (Testing Pipeline integrity without target leakage)

### 📈 Week 3: Robust Validation & Linear/Probabilistic Baselines
* [ ] **Day 15**: Cross-Validation Setup (RepeatedStratifiedKFold configuration)
* [ ] **Day 16**: Imbalanced Evaluation (Coding PR-AUC & F-Beta [Beta=2] scoring)
* [ ] **Day 17**: Naive Bayes Implementation (ComplementNB for highly uneven classes)
* [ ] **Day 18**: Logistic Regression (Baseline training with L1 / Lasso penalties)
* [ ] **Day 19**: Ridge & ElasticNet (L2 Regularization & mixed-penalty tuning)
* [ ] **Day 20**: Support Vector Machines (Linear & RBF Kernels with Class Weights)
* [ ] **Day 21**: Weekly Review (Comparing Precision-Recall curves across baselines)

### 👑 Week 4: Tree Ensembles & State-of-the-Art Optimization
* [ ] **Day 22**: Bagging Foundations (Random Forest & Extra Trees architecture)
* [ ] **Day 23**: Boosting Intro (XGBoost Classifier + tuning scale_pos_weight)
* [ ] **Day 24**: Fast Histogram Trees (LightGBM setup with custom leaf growth)
* [ ] **Day 25**: Native Categories (CatBoost Classifier with raw text indexing)
* [ ] **Day 26**: Optuna Framework (Writing Objectives & setting up TPE samplers)
* [ ] **Day 27**: Modern Tuning (Bayesian Optimization of Ensemble Hyperparameters)
* [ ] **Day 28**: Weekly Review (Visualizing Optuna contour & slice parameter plots)

### 🎯 Final Stage: The Regression Pivot & Production Deployment
* [ ] **Day 29**: Regression Transition (Isolating fraud targets to predict financial loss)
* [ ] **Day 30**: Final Stack (Evaluating Regressors with MAE & RMSE metrics)

---

## 🚫 Critical Project Rules
1. **Zero Data Leakage:** All preprocessing steps (Imputation, Scaling, Target Encoding) are wrapped entirely inside Scikit-Learn pipelines. Transformers are strictly *fitted* only on training splits and applied via *transform* to testing splits.
2. **No Pure Accuracy Metric:** Because the dataset is severely skewed, predictive performance is judged exclusively by **PR-AUC (Precision-Recall Area Under Curve)** and **$F_2$-Score** to heavily penalize missing actual fraud cases.
3. **No Blind Copying:** Every line of training code is written manually during daily 2-hour implementation sessions to build absolute mastery over library syntax and parameters.
