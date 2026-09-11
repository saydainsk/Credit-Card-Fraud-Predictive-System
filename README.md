# Credit Card Fraud Predictive System

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0)](https://seaborn.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189FDD)](https://xgboost.readthedocs.io/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE%20%7C%20Sampling-8A2BE2)](https://imbalanced-learn.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Fraud%20Detection-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

An end-to-end machine learning capstone for detecting fraudulent credit-card transactions in a severely imbalanced dataset. The project combines exploratory analysis, feature engineering, sampling experiments, classifier comparison, holdout evaluation, and a deployment-oriented cost-benefit study.

## Project Overview

Credit-card fraud causes substantial losses for financial institutions and their customers. Because fraudulent transactions represent only a small fraction of total activity, a model can achieve very high accuracy while failing to identify the cases that matter.

This project compares three classification algorithms across four class-balancing strategies:

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**
- Raw imbalanced data
- NearMiss under-sampling
- Random over-sampling
- Synthetic Minority Over-sampling Technique (SMOTE)

The notebook prioritizes **fraud recall** during model selection because false negatives represent missed fraudulent transactions. It also demonstrates why recall must be evaluated together with precision, false-positive volume, and operational cost before deployment.

## Business Objective

Build a machine learning system that can flag potentially fraudulent transactions and quantify whether deploying it would reduce the bank's total fraud-management cost.

The analysis addresses these questions:

- Which engineered transaction features contain useful fraud signals?
- How severely does class imbalance affect model behavior?
- Which sampling strategy produces the best validation recall?
- Do the strongest validation candidates generalize to a separate holdout file?
- Is the model's reduction in missed fraud worth the cost of investigating false alarms?

## Dataset

The project uses the [Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

| File | Transactions | Raw columns | Role |
|---|---:|---:|---|
| `fraudTrain.csv` | 1,296,675 | 23 | Model development and internal validation |
| `fraudTest.csv` | 555,719 | 23 | Separate holdout evaluation |

The training file contains **7,506 fraudulent transactions**, approximately **0.58%** of its records. This extreme imbalance is the central modeling challenge.

### Raw data attributes

The source data includes:

- Transaction date and time
- Card number and transaction identifier
- Merchant name and category
- Transaction amount
- Cardholder name, gender, street, city, state, ZIP code, and job
- Cardholder date of birth
- Cardholder latitude and longitude
- Merchant latitude and longitude
- Fraud label: `is_fraud`

Personally identifying and high-cardinality raw attributes are removed after the required modeling features are derived.

## Feature Engineering

The notebook produces seven predictors from the original 23 columns:

| Feature | Description | Transformation |
|---|---|---|
| `amt` | Transaction amount | Natural-log transformation |
| `gender` | Cardholder gender | Binary encoding: F = 1, M = 0 |
| `city_pop` | Population of the cardholder's city | Natural-log transformation |
| `Age` | Cardholder age at transaction time | Derived from date of birth and transaction date |
| `Day_of_Week` | Transaction weekday | Encoded Monday = 1 through Sunday = 7 |
| `Month` | Transaction month | Extracted from transaction timestamp |
| `Dist` | Customer-to-merchant distance | Derived with the Haversine formula |

The target variable is:

```text
is_fraud = 0  -> legitimate transaction
is_fraud = 1  -> fraudulent transaction
```

## Exploratory Data Analysis

The notebook investigates:

- Fraud and non-fraud class frequencies
- Missing values and duplicate observations
- Transaction-amount and city-population distributions
- Transaction amount versus city population
- Customer gender distribution
- Transaction amount by fraud class
- Feature correlations

### Main observations

- Fraud represents less than 1% of the training data.
- `amt` and `city_pop` are right-skewed and are log-transformed.
- Transaction amount and city population show no clear direct relationship.
- Transaction amount has a modest correlation of approximately **0.12** with `is_fraud`.
- No single feature is sufficient to distinguish fraudulent transactions reliably.

## Modeling Workflow

### 1. Development split

The labeled training data is divided into an **80/20 stratified split** using `random_state=42`.

| Partition | Total rows | Fraud cases |
|---|---:|---:|
| Internal training | 1,037,340 | 6,005 |
| Internal validation | 259,335 | 1,501 |

Stratification preserves the rare fraud proportion in both partitions.

### 2. Feature scaling

`StandardScaler` is used to standardize the seven model inputs before classification.

### 3. Class-imbalance strategies

Each model family is evaluated with four data treatments:

| Strategy | Description |
|---|---|
| Imbalanced | Uses the original class distribution |
| NearMiss under-sampling | Reduces the majority class to match the minority class |
| Random over-sampling | Duplicates minority observations |
| SMOTE | Generates synthetic minority-class observations |

### 4. Candidate models

#### Logistic Regression

Used as an interpretable linear baseline with recall-focused tuning.

#### Random Forest

Used to learn nonlinear interactions among transaction, demographic, temporal, and geographic features.

#### XGBoost

Used as a gradient-boosted tree model. The notebook manually searches the number of estimators and records **27 trees** for the imbalanced and over-sampled experiments.

### 5. Evaluation approach

The notebook uses:

- Recall
- Precision
- F1-score
- ROC-AUC
- Average precision
- Confusion matrices
- Cross-validation and grid search

Recall is emphasized during candidate selection, but operational viability requires precision and false-positive cost to be considered as well.

## Internal Validation Results

The following values are recorded in the notebook's model summary:

| Model and sampling strategy | Train recall | Validation recall | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression - Imbalanced | 0.000 | 0.000 | 0.84 |
| Logistic Regression - Under-sampling | 0.744 | 0.750 | 0.75 |
| Logistic Regression - Over-sampling | 0.774 | 0.781 | 0.84 |
| Logistic Regression - SMOTE | 0.776 | 0.786 | 0.84 |
| Random Forest - Imbalanced | 0.904 | 0.353 | 0.89 |
| Random Forest - Under-sampling | 0.826 | 0.821 | 0.90 |
| Random Forest - Over-sampling | 0.896 | 0.837 | 0.96 |
| **Random Forest - SMOTE** | **0.895** | **0.916** | **0.95** |
| XGBoost - Imbalanced | 0.364 | 0.316 | 0.97 |
| XGBoost - Under-sampling | 0.906 | 0.852 | 0.92 |
| XGBoost - Over-sampling | 0.977 | 0.834 | 0.97 |
| **XGBoost - SMOTE** | **0.963** | **0.960** | **0.96** |

### Validation interpretation

- Models trained on the raw imbalanced data produced poor fraud recall.
- Sampling substantially improved minority-class detection.
- Random Forest with SMOTE reached **91.6% validation recall** with an ROC-AUC of **0.95**.
- XGBoost with SMOTE recorded the highest validation recall at **96.0%**, with an ROC-AUC of **0.96**.
- The difference between validation metrics and separate holdout behavior demonstrates the importance of a consistent end-to-end preprocessing pipeline.

## Separate Holdout Evaluation

Four trained candidates were applied to `fraudTest.csv`:

| Model | Fraud recall | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---|
| Random Forest - SMOTE | **0.9986** | `[[58116, 495458], [3, 2142]]` |
| Random Forest - Over-sampling | 0.1068 | `[[449555, 104019], [1916, 229]]` |
| XGBoost - SMOTE | 0.0462 | `[[518438, 35136], [2046, 99]]` |
| XGBoost - Over-sampling | 0.0410 | `[[519446, 34128], [2057, 88]]` |

### Critical interpretation of the Random Forest-SMOTE result

The Random Forest-SMOTE model missed only 3 of 2,145 fraudulent transactions, producing **99.86% recall**. However, the same confusion matrix also shows that it flagged **495,458 legitimate transactions**.

Metrics calculated directly from the recorded confusion matrix are:

| Metric | Value |
|---|---:|
| Recall / sensitivity | 99.86% |
| Precision | 0.43% |
| Specificity | 10.50% |
| False-positive rate | 89.50% |
| Accuracy | 10.84% |
| Transactions flagged | 497,600 of 555,719 (89.54%) |

Therefore, the holdout result should be treated as a **diagnostic result, not production evidence**. The classifier catches nearly every fraud case by flagging almost every transaction, which would create an unsustainable review burden.

The notebook also applies the final models to holdout features that were not transformed with the fitted training scaler. Correcting this train-serving skew is necessary before comparing the final candidates or making a deployment decision.

## Cost-Benefit Analysis

The notebook records the following business assumptions and outputs:

| Quantity | Notebook value |
|---|---:|
| Average transactions per month | 77,183 |
| Average fraudulent transactions per month | 402 |
| Average amount per fraudulent transaction | $121 |
| Estimated monthly fraud cost before deployment | $48,642 |
| Reported monthly support cost after deployment | $459 |
| Reported monthly savings | $48,183 |

### Cost-analysis caveat

The reported $459 support cost counts detected **actual fraud cases**, rather than every transaction predicted as fraud. Operational review cost should include true positives **and false positives**.

Using the recorded Random Forest-SMOTE confusion matrix and the notebook's seven-month holdout assumption:

- 497,600 transactions were flagged.
- This is approximately 71,086 flagged transactions per month.
- At $1.50 per investigation, the implied review cost is approximately **$106,629 per month**, before adding losses from missed fraud.

For this reason, the reported $48,183 monthly savings is an **unvalidated scenario estimate** and should not be used as a deployment claim until preprocessing and cost accounting are corrected.

## Key Takeaways

- Severe class imbalance makes accuracy an unreliable standalone metric.
- Resampling greatly improves recall compared with training directly on the raw distribution.
- Recall must be balanced against precision, specificity, alert volume, and investigation capacity.
- Strong internal validation performance does not guarantee performance on a separate dataset.
- Feature engineering converts raw transaction data into a compact set of behavioral, temporal, demographic, and geographic predictors.
- A cost-benefit analysis is valuable only when it includes the cost of every model-generated alert.
- A single reproducible preprocessing and inference pipeline is essential for trustworthy holdout evaluation.

## Repository Structure

```text
Credit-Card-Fraud-Predictive-System/
├── Credit Card Fraud Detection System Python Code.ipynb
├── Credit Card Fraud Detection System Capstone Presentation.pptx
└── README.md
```

The Kaggle CSV files are not committed to the repository and must be downloaded separately.

## Technologies Used

| Category | Tools |
|---|---|
| Language and environment | Python, Jupyter Notebook |
| Data processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Feature preprocessing | Log transformation, StandardScaler |
| Feature engineering | Datetime extraction, binary encoding, Haversine distance |
| Classification | Logistic Regression, Random Forest, XGBoost |
| Imbalance handling | NearMiss, RandomOverSampler, SMOTE |
| Model evaluation | scikit-learn metrics, cross-validation, grid search |

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/saydainsk/Credit-Card-Fraud-Predictive-System.git
   cd Credit-Card-Fraud-Predictive-System
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install jupyter numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```

4. Download the data from the [Kaggle Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

5. Place these files in the repository root beside the notebook:

   ```text
   fraudTrain.csv
   fraudTest.csv
   ```

6. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

7. Open `Credit Card Fraud Detection System Python Code.ipynb` and run the cells in order.

> The full notebook trains multiple models on more than one million records. Runtime and memory usage can be substantial.

## Recommended Improvements

- Create a single `Pipeline` or `imblearn.pipeline.Pipeline` for feature transformation, scaling, resampling, and inference.
- Fit the scaler on training data only; use `transform` on validation and final holdout data.
- Place SMOTE inside each cross-validation fold to prevent synthetic-data leakage.
- Correct the Haversine calculation by converting latitude values to radians before applying cosine.
- Optimize a cost-sensitive metric instead of recall alone.
- Tune the classification threshold using precision-recall curves and the bank's investigation capacity.
- Compare class weights with under-sampling, over-sampling, and SMOTE.
- Report PR-AUC because it is more informative than ROC-AUC for rare-event detection.
- Evaluate false positives per 1,000 transactions and fraud dollars captured.
- Rebuild the cost-benefit analysis using all predicted positives, including false alarms.
- Add temporal validation to simulate future transaction behavior and concept drift.
- Save the trained preprocessing pipeline and model with Joblib.
- Add monitoring for feature drift, prediction drift, precision, recall, alert volume, and investigation outcomes.

## Responsible Use

This project is an academic prototype. It should not be used to automatically block transactions or make adverse decisions about customers without further validation, calibrated thresholds, security review, fairness analysis, and human oversight.

## Author

**Saydain Sheikh**  
Credit Card Fraud Detection System - Capstone Project

- [GitHub](https://github.com/saydainsk)
- [LinkedIn](https://www.linkedin.com/in/saydain-sheikh/)


